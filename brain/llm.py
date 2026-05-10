import json
from typing import Any, Dict, List, Optional

from config import LLM_PROVIDER, get_chat_model
from brain.prompts import (
    BASE_LLM_PROMPT,
    FORMATTED_RESPONSE_PROMPT_BLOCK,
    TOOLS_PROMPT_BLOCK,
    json_text,
)
from utilities import safe_json_parse


def get_prompt(system_message: str, query: str, formate=None, tools=None, memory=None):
    prompt = BASE_LLM_PROMPT.format(
        system_message=system_message,
        query=query,
    )
    if memory:
        prompt = memory.conversation.prompt() + prompt
    if tools:
        prompt += TOOLS_PROMPT_BLOCK.format(
            tools=json_text([f"Tool: {tool}, args: {json.dumps(tools[tool]['args'])}" for tool in tools]),
        )
        return prompt
    if formate:
        prompt += FORMATTED_RESPONSE_PROMPT_BLOCK.format(formate=formate)

    return prompt


class Llm:
    BASE_MODEL = get_chat_model("default", provider="ollama")
    BRAIN_LLM = BASE_MODEL
    WEB_SEARCH_LLM = BASE_MODEL

    def __init__(
        self,
        system_message: str,
        tools=None,
        memory=None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        provider: Optional[str] = None,
        role: str = "default",
    ):
        self.provider = (provider or LLM_PROVIDER).lower()
        self.role = role
        self.model = model or get_chat_model(role=role, provider=self.provider)
        self.llm = build_chat_model(
            provider=self.provider,
            model=self.model,
            temperature=temperature,
        )
        self.system_message = system_message
        self.tools = tools
        self.memory = memory

    @staticmethod
    def get_llm(model: Optional[str] = None, provider: Optional[str] = None, temperature: float = 0.2):
        selected_provider = provider or LLM_PROVIDER
        return build_chat_model(
            provider=selected_provider,
            model=model or get_chat_model(provider=selected_provider),
            temperature=temperature,
        )

    @staticmethod
    def bind_tools(llm, tools):
        if not hasattr(llm, "bind_tools"):
            raise TypeError("Selected LLM does not support LangChain tool binding.")
        return llm.bind_tools(tools)

    @property
    def runnable(self):
        return self.llm

    @property
    def tool_runnable(self):
        if not self.tools:
            return self.llm
        return self.bind_tools(self.llm, self._langchain_tools())

    def _langchain_tools(self):
        try:
            from langchain_core.tools import StructuredTool
        except ImportError as exc:
            raise ImportError("Tool binding requires langchain-core tools.") from exc

        langchain_tools = []
        for name, definition in (self.tools or {}).items():
            tool_fn = definition["tool"]
            description = definition.get("description") or f"Run {name}."
            langchain_tools.append(
                StructuredTool.from_function(
                    func=tool_fn,
                    name=name,
                    description=description,
                )
            )
        return langchain_tools

    def invoke_messages(self, messages: List[Dict[str, str]]) -> str:
        response = self.llm.invoke(messages)
        return normalize_llm_response(response)

    def invoke_tool_messages(self, messages: List[Dict[str, str]]):
        response = self.tool_runnable.invoke(messages)
        return response

    def _invoke_text(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return normalize_llm_response(response)

    def invoke_with_tools(self, query: str):
        if self.tools and hasattr(self.llm, "bind_tools"):
            response = self.invoke_tool_messages([
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": str(query)},
            ])
            tool_calls = getattr(response, "tool_calls", None) or []
            if tool_calls:
                results = []
                for tool_call in tool_calls:
                    name = tool_call.get("name")
                    args = tool_call.get("args") or {}
                    results.append(self.tools[name]["tool"](**args))
                return results[0] if len(results) == 1 else results

        selected_tool = safe_json_parse(
            self._invoke_text(get_prompt(self.system_message, query, tools=self.tools, memory=self.memory))
        )
        if not isinstance(selected_tool, dict) or selected_tool.get("tool") == "end":
            return None
        return self.tools[selected_tool["tool"]]["tool"](**selected_tool["args"])

    def invoke_with_formated_response(self, query: str, formate: str):
        return safe_json_parse(
            self._invoke_text(get_prompt(self.system_message, query, formate, memory=self.memory))
        )

    def invoke(self, query: str):
        return self._invoke_text(get_prompt(self.system_message, query, memory=self.memory))


def build_chat_model(provider: str, model: Optional[str], temperature: float = 0.2):
    selected_provider = (provider or LLM_PROVIDER).lower()
    selected_model = model or get_chat_model(provider=selected_provider)

    if selected_provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
            return ChatOllama(model=selected_model, temperature=temperature)
        except ImportError:
            from langchain_community.chat_models import ChatOllama
            return ChatOllama(model=selected_model, temperature=temperature)

    if selected_provider == "groq":
        try:
            from langchain_groq import ChatGroq
        except ImportError as exc:
            raise ImportError(
                "Groq provider requires langchain-groq. Install it with `pip install langchain-groq` "
                "and set GROQ_API_KEY."
            ) from exc

        return ChatGroq(model=selected_model, temperature=temperature)

    raise ValueError(f"Unsupported LLM provider: {selected_provider}")


def normalize_llm_response(response: Any) -> str:
    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or item))
            else:
                parts.append(str(item))
        return "\n".join(parts)

    return str(content)


LLM_V1 = Llm
