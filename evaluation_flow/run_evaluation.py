import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

os.environ.setdefault("MallocStackLogging", "0")
os.environ.setdefault("MallocStackLoggingNoCompact", "0")

from brain.agents.german_admin.graph import GermanAdminGuideAgent
from brain.llm import Llm
from config import QUALITY_LLM_PROVIDER


JUDGE_SYSTEM = """
You are an independent evaluator for a German administrative guidance assistant.
Your task is to judge whether the assistant answer is satisfactory for the user's question.
Be strict but fair.
Use the provided answer and limited internal state only as evidence.
Do not reward invented details.
Do not require legal-advice precision, but the answer must be useful, grounded, clear, and safe.
Return only valid JSON.
"""

JUDGE_SCHEMA = {
    "satisfactory": True,
    "rank": "excellent|good|partial|poor|fail",
    "score": 1,
    "relevance": 1,
    "grounding": 1,
    "completeness": 1,
    "clarity": 1,
    "safety": 1,
    "reason": "short explanation",
    "missing_or_wrong": ["specific missing or wrong points"],
    "improvement": "short improvement suggestion",
}

JUDGE_PROMPT = """
Evaluate this assistant answer.

User question:
{question}

Assistant answer:
{answer}

Limited internal state:
{state_json}

Scoring rules:
- score is 1 to 5.
- 5 = excellent: directly answers, well grounded, complete, clear, safe.
- 4 = good: useful answer with only minor gaps.
- 3 = partial: some useful guidance but important gaps or uncertainty.
- 2 = poor: weak, vague, likely wrong, or not grounded enough.
- 1 = fail: unrelated, unsafe, refuses incorrectly, or mostly unusable.
- satisfactory should be true only for score 4 or 5.
- rank must match the score: 5 excellent, 4 good, 3 partial, 2 poor, 1 fail.
- grounding means the answer is supported by retrieved service details, ServiceQA, official web findings, or clearly stated uncertainty.
"""


def read_questions(path: Path, limit: int = 0) -> List[str]:
    questions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        question = line.strip()
        if not question or question.startswith("#"):
            continue
        questions.append(question)
        if limit > 0 and len(questions) >= limit:
            break
    return questions


def compact_state(state: Dict[str, Any]) -> Dict[str, Any]:
    findings = state.get("findings", {}) if isinstance(state, dict) else {}
    service_details = findings.get("service_details", []) if isinstance(findings, dict) else []
    search_results = findings.get("search_results", {}) if isinstance(findings, dict) else {}

    return {
        "route": state.get("route"),
        "target_language": state.get("target_language"),
        "intake": state.get("intake", {}),
        "knowledge_query": state.get("knowledge_query", ""),
        "knowledge_queries": state.get("knowledge_queries", []),
        "german_search_terms": state.get("german_search_terms", []),
        "search_result_counts": {
            "SubSituation": len(search_results.get("SubSituation", []) or []),
            "Service": len(search_results.get("Service", []) or []),
            "ServiceQA": len(search_results.get("ServiceQA", []) or []),
            "service_details": len(service_details or []),
        },
        "top_services": [
            {
                "id": (item.get("service") or {}).get("id"),
                "name": (item.get("service") or {}).get("name"),
                "url": (item.get("service") or {}).get("url"),
                "retrieval_reasons": item.get("retrieval_reasons", []),
                "requirements_count": len(item.get("requirements", []) or []),
                "documents_count": len(item.get("documents", []) or []),
                "service_qa_count": len(item.get("service_qa", []) or []),
            }
            for item in (service_details or [])[:3]
            if isinstance(item, dict)
        ],
        "web_query": (findings.get("web_findings") or {}).get("query") if isinstance(findings, dict) else "",
        "tool_error": findings.get("tool_error") if isinstance(findings, dict) else None,
    }


def judge_answer(
    question: str,
    answer: str,
    state: Dict[str, Any],
    provider: str,
    role: str,
) -> Dict[str, Any]:
    judge = Llm(
        system_message=JUDGE_SYSTEM,
        provider=provider,
        role=role,
        temperature=0.0,
    )
    result = judge.invoke_with_formated_response(
        query=JUDGE_PROMPT.format(
            question=question,
            answer=answer,
            state_json=json.dumps(compact_state(state), ensure_ascii=False, indent=2),
        ),
        formate=JUDGE_SCHEMA,
    )

    if not isinstance(result, dict):
        raise ValueError(f"Judge returned invalid result: {result!r}")

    score = int(result.get("score", 1) or 1)
    score = max(1, min(5, score))
    expected_rank = {
        5: "excellent",
        4: "good",
        3: "partial",
        2: "poor",
        1: "fail",
    }[score]

    return {
        "satisfactory": bool(result.get("satisfactory", score >= 4)),
        "rank": result.get("rank") or expected_rank,
        "score": score,
        "relevance": bounded_score(result.get("relevance")),
        "grounding": bounded_score(result.get("grounding")),
        "completeness": bounded_score(result.get("completeness")),
        "clarity": bounded_score(result.get("clarity")),
        "safety": bounded_score(result.get("safety")),
        "reason": str(result.get("reason", "")).strip(),
        "missing_or_wrong": result.get("missing_or_wrong", []),
        "improvement": str(result.get("improvement", "")).strip(),
    }


def bounded_score(value: Any) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError):
        score = 1
    return max(1, min(5, score))


def run_one(
    index: int,
    question: str,
    run_id: str,
    shared_agent: Optional[GermanAdminGuideAgent],
    judge_provider: str,
    judge_role: str,
) -> Dict[str, Any]:
    started_at = datetime.utcnow().isoformat()
    start = time.time()
    agent = shared_agent or GermanAdminGuideAgent(
        user_id="evaluation",
        conv_id=f"{run_id}-{index:03d}",
    )

    answer = ""
    state: Dict[str, Any] = {}
    status = "ok"
    error = ""

    try:
        answer = agent.chat(question)
        state = agent.get_last_state()
    except Exception as exc:
        status = "agent_error"
        error = str(exc)
        judge = {
            "satisfactory": False,
            "rank": "fail",
            "score": 1,
            "relevance": 1,
            "grounding": 1,
            "completeness": 1,
            "clarity": 1,
            "safety": 1,
            "reason": "The assistant failed before producing a valid answer.",
            "missing_or_wrong": [str(exc)],
            "improvement": "Fix the agent runtime error and rerun this question.",
        }
    else:
        try:
            judge = judge_answer(
                question=question,
                answer=answer,
                state=state,
                provider=judge_provider,
                role=judge_role,
            )
        except Exception as exc:
            status = "judge_error"
            error = str(exc)
            judge = {
                "satisfactory": False,
                "rank": "fail",
                "score": 1,
                "relevance": 1,
                "grounding": 1,
                "completeness": 1,
                "clarity": 1,
                "safety": 1,
                "reason": "The judge failed to return a valid structured evaluation.",
                "missing_or_wrong": [str(exc)],
                "improvement": "Improve judge structured-output parsing or rerun with another judge model.",
            }
    finally:
        if shared_agent is None:
            try:
                agent.close()
            except Exception:
                pass

    elapsed_seconds = round(time.time() - start, 2)
    return {
        "index": index,
        "question": question,
        "answer": answer,
        "judge": judge,
        "status": status,
        "error": error,
        "started_at": started_at,
        "elapsed_seconds": elapsed_seconds,
        "state": compact_state(state),
    }

def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "index",
        "status",
        "score",
        "rank",
        "satisfactory",
        "relevance",
        "grounding",
        "completeness",
        "clarity",
        "safety",
        "elapsed_seconds",
        "question",
        "reason",
        "improvement",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            judge = row["judge"]
            writer.writerow({
                "index": row["index"],
                "status": row["status"],
                "score": judge["score"],
                "rank": judge["rank"],
                "satisfactory": judge["satisfactory"],
                "relevance": judge["relevance"],
                "grounding": judge["grounding"],
                "completeness": judge["completeness"],
                "clarity": judge["clarity"],
                "safety": judge["safety"],
                "elapsed_seconds": row["elapsed_seconds"],
                "question": row["question"],
                "reason": judge["reason"],
                "improvement": judge["improvement"],
                "error": row["error"],
            })


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "total": 0,
            "satisfactory": 0,
            "unsatisfactory": 0,
            "average_score": 0,
            "rank_counts": {},
            "status_counts": {},
        }

    rank_counts: Dict[str, int] = {}
    status_counts: Dict[str, int] = {}
    for row in rows:
        rank = row["judge"].get("rank", "unknown")
        status = row.get("status", "unknown")
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
        status_counts[status] = status_counts.get(status, 0) + 1

    satisfactory = sum(1 for row in rows if row["judge"].get("satisfactory"))
    total_score = sum(int(row["judge"].get("score", 1)) for row in rows)
    return {
        "total": len(rows),
        "satisfactory": satisfactory,
        "unsatisfactory": len(rows) - satisfactory,
        "satisfactory_rate": round(satisfactory / len(rows), 3),
        "average_score": round(total_score / len(rows), 2),
        "rank_counts": rank_counts,
        "status_counts": status_counts,
        "average_elapsed_seconds": round(
            sum(float(row.get("elapsed_seconds", 0)) for row in rows) / len(rows),
            2,
        ),
    }


def parse_args() -> argparse.Namespace:
    default_questions = Path(__file__).resolve().parent / "questions.txt"
    default_out = Path(__file__).resolve().parent / "results"

    parser = argparse.ArgumentParser(
        description="Run German Administrative Assistant evaluation questions and judge each answer."
    )
    parser.add_argument("--questions", type=Path, default=default_questions)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=default_out)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--judge-provider", default=QUALITY_LLM_PROVIDER)
    parser.add_argument("--judge-role", default="judge")
    parser.add_argument("--shared-conversation", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = read_questions(args.questions, limit=args.limit)
    if not questions:
        raise SystemExit(f"No questions found in {args.questions}")

    run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out_dir = args.out_dir or (args.results_root / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    shared_agent = None
    if args.shared_conversation:
        shared_agent = GermanAdminGuideAgent(
            user_id="evaluation",
            conv_id=f"{run_id}-shared",
        )

    rows = []
    try:
        print(
            f"Evaluation started: questions={len(questions)}, out_dir={out_dir}, "
            f"judge_provider={args.judge_provider}, shared_conversation={args.shared_conversation}"
        )
        for index, question in enumerate(questions, start=1):
            row = run_one(
                index=index,
                question=question,
                run_id=run_id,
                shared_agent=shared_agent,
                judge_provider=args.judge_provider,
                judge_role=args.judge_role,
            )
            rows.append(row)
            if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(questions)):
                print(
                    f"[{index}/{len(questions)}] score={row['judge']['score']} "
                    f"rank={row['judge']['rank']} satisfactory={row['judge']['satisfactory']} "
                    f"status={row['status']} seconds={row['elapsed_seconds']}"
                )

            write_jsonl(out_dir / "results.jsonl", rows)
            write_csv(out_dir / "results.csv", rows)
            (out_dir / "summary.json").write_text(
                json.dumps(summarize(rows), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    finally:
        if shared_agent is not None:
            shared_agent.close()

    summary = summarize(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()
