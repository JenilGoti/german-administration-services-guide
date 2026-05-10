from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from brain.agents.german_admin.graph import GermanAdminGuideAgent


st.set_page_config(page_title="Administrative Assistant", page_icon="A", layout="wide")


def new_conversation_id() -> str:
    return f"chat-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"


def init_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = new_conversation_id()
    if "agent_key" not in st.session_state:
        st.session_state.agent_key = None
    if "agent" not in st.session_state or st.session_state.agent_key != st.session_state.conversation_id:
        st.session_state.agent = GermanAdminGuideAgent(
            user_id="streamlit",
            conv_id=st.session_state.conversation_id,
        )
        st.session_state.agent_key = st.session_state.conversation_id


def run_turn(prompt: str) -> None:
    prompt = (prompt or "").strip()
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    status_slot = st.empty()

    def show_agent_status(agent_name: str) -> None:
        status_slot.info(f"{agent_name} is working...")

    st.session_state.agent.progress_callback = show_agent_status
    try:
        response = st.session_state.agent.chat(prompt)
    except Exception as exc:
        response = (
            "I could not complete this request because the backend raised an error.\n\n"
            f"`{type(exc).__name__}: {exc}`"
        )
    finally:
        st.session_state.agent.progress_callback = None
        status_slot.empty()

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})


def reset_chat() -> None:
    old_agent = st.session_state.get("agent")
    if old_agent:
        try:
            old_agent.close()
        except Exception:
            pass
    st.session_state.messages = []
    st.session_state.pending_prompt = None
    st.session_state.agent = GermanAdminGuideAgent(
        user_id="streamlit",
        conv_id=st.session_state.conversation_id,
    )
    st.session_state.agent_key = st.session_state.conversation_id


def latest_state() -> dict[str, Any]:
    agent = st.session_state.get("agent")
    if not agent:
        return {}
    try:
        return dict(agent.get_last_state() or {})
    except Exception:
        return {}


DOCUMENT_KEYS = {
    "documents",
    "required_documents",
    "required documents",
    "required_document",
    "required document",
    "unterlagen",
}


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[\-*\d\.\)\s]+", "", text)
    return text.strip(" :-")


def unique_items(items: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = set()
    unique = []
    for item in items:
        name = clean_text(item.get("name", ""))
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append({
            "name": name,
            "description": clean_text(item.get("description", "")),
        })
    return unique


def collect_document_candidates(value: Any, in_document_section: bool = False) -> list[dict[str, str]]:
    docs: list[dict[str, str]] = []
    if isinstance(value, dict):
        if in_document_section:
            name = value.get("name") or value.get("title") or value.get("label") or value.get("document")
            if name:
                return [{
                    "name": clean_text(name),
                    "description": clean_text(value.get("description", "")),
                }]
        for key, item in value.items():
            lowered = str(key).lower()
            docs.extend(collect_document_candidates(item, lowered in DOCUMENT_KEYS))
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                if in_document_section:
                    docs.append({"name": clean_text(item), "description": ""})
            elif isinstance(item, dict):
                docs.extend(collect_document_candidates(item, in_document_section))
    return docs


def parse_documents_from_response(response: str) -> list[dict[str, str]]:
    docs: list[dict[str, str]] = []
    in_section = False
    stop_headings = (
        "steps",
        "procedure",
        "links",
        "forms",
        "authority",
        "office",
        "follow",
        "questions",
        "note",
    )

    for raw_line in (response or "").splitlines():
        line = raw_line.strip()
        lowered = line.lower().strip("#: ")
        if not line:
            continue
        if any(title in lowered for title in ["required documents", "documents", "unterlagen"]):
            in_section = True
            continue
        if in_section and (line.startswith("#") or lowered.endswith(":")):
            if any(lowered.startswith(stop) for stop in stop_headings):
                break
        if not in_section:
            continue
        if re.match(r"^(\-|\*|\d+[\.\)])\s+", line):
            name = clean_text(line.split(":", 1)[0])
            desc = clean_text(line.split(":", 1)[1]) if ":" in line else ""
            docs.append({"name": name, "description": desc})

    return docs


def extract_documents(state: dict[str, Any]) -> list[dict[str, str]]:
    findings = state.get("findings") or {}
    docs = collect_document_candidates(findings.get("service_details", []))
    if not docs:
        docs = parse_documents_from_response(state.get("response", ""))
    return unique_items(docs)[:10]


def extract_links(state: dict[str, Any]) -> list[dict[str, str]]:
    links: list[dict[str, str]] = []
    findings = state.get("findings") or {}

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            url = value.get("url") or value.get("link")
            title = value.get("title") or value.get("name") or value.get("label") or url
            if url and str(url).startswith(("http://", "https://")):
                links.append({"title": clean_text(title), "url": str(url)})
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(findings.get("service_details", []))
    walk((findings.get("web_findings") or {}).get("results", []))

    seen = set()
    unique = []
    for link in links:
        if link["url"] in seen:
            continue
        seen.add(link["url"])
        unique.append(link)
    return unique[:6]


def state_summary(state: dict[str, Any]) -> tuple[dict[str, Any], str]:
    intake = state.get("intake") or {}
    route = state.get("route") or intake.get("route") or "not started"
    return intake, route


def safe_json(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def followup_button(label: str, prompt: str, key: str) -> None:
    if st.button(label, key=key, use_container_width=True):
        st.session_state.pending_prompt = prompt
        st.rerun()


def render_case_panel(state: dict[str, Any]) -> None:
    new_conv_id = st.text_input("Conversation ID", value=st.session_state.conversation_id)
    if new_conv_id != st.session_state.conversation_id:
        st.session_state.conversation_id = new_conv_id.strip() or new_conversation_id()
        reset_chat()
        st.rerun()

    if st.button("New chat", use_container_width=True):
        st.session_state.conversation_id = new_conversation_id()
        reset_chat()
        st.rerun()

    st.divider()

    intake, route = state_summary(state)
    st.subheader("Current Case")
    st.markdown(f'<span class="assistant-chip">Route: {clean_text(route)}</span>', unsafe_allow_html=True)

    problem_type = clean_text(intake.get("problem_type", ""))
    # if problem_type:
    #     st.write("**Problem**")
    #     st.write(problem_type)

    known_facts = intake.get("known_facts") or []
    if known_facts:
        st.write("**Known facts**")
        for fact in known_facts[:6]:
            st.write(f"- {clean_text(fact)}")

    # missing_info = intake.get("missing_information") or []
    # if missing_info:
    #     st.write("**Missing information**")
    #     for index, item in enumerate(missing_info[:6]):
    #         item_text = clean_text(item)
    #         followup_button(
    #             item_text,
    #             f"Here is the missing information for this case: {item_text}",
    #             f"missing-{index}-{item_text}",
    #         )

    docs = extract_documents(state)
    if docs:
        st.divider()
        st.subheader("Documents")
        for index, doc in enumerate(docs):
            label = doc["name"]
            prompt = (
                f"Help me understand {label} for this case. "
                "What is it, where do I get it, and when is it needed?"
            )
            followup_button(label, prompt, f"doc-{index}-{label}")
            if doc.get("description"):
                st.caption(doc["description"])

    links = extract_links(state)
    if links:
        st.divider()
        st.subheader("Useful Links")
        for link in links:
            st.markdown(f"- [{link['title']}]({link['url']})")

    with st.expander("Debug state"):
        st.json(safe_json(state))


def render_chat() -> None:
    st.title("Administrative Assistant")
    st.caption("Describe your situation once. Then use follow-ups for documents, offices, forms, deadlines, or next steps.")

    if not st.session_state.messages:
        st.info("Try: I came from India to Aalen for work. What should I do?")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.session_state.pending_prompt:
        queued_prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None
        run_turn(queued_prompt)
        st.rerun()

    prompt = st.chat_input("Ask about your situation, documents, office, form, deadline, or a follow-up.")
    if prompt:
        st.session_state.pending_prompt = prompt
        st.rerun()


init_session()

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.25rem; max-width: 1500px; }
    [data-testid="stHorizontalBlock"] {
        align-items: flex-start;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="column"]:first-child > div {
        max-height: calc(100vh - 2.5rem);
        overflow-y: auto;
        padding-right: 0.35rem;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="column"]:last-child > div {
        position: sticky;
        top: 1rem;
        max-height: calc(100vh - 2rem);
        overflow-y: auto;
        padding: 0 0.35rem 1rem 0.35rem;
        border-left: 1px solid #eaecf0;
    }
    .assistant-muted { color: #667085; font-size: 0.9rem; }
    .assistant-chip {
        display: inline-block;
        border: 1px solid #d0d5dd;
        border-radius: 999px;
        padding: 0.15rem 0.6rem;
        margin: 0.15rem 0.2rem 0.15rem 0;
        font-size: 0.82rem;
        color: #344054;
        background: #f9fafb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([0.64, 0.36], gap="large")
with left_col:
    render_chat()
with right_col:
    st.title("Case Panel")
    st.caption("The panel updates after each assistant response.")
    render_case_panel(latest_state())
