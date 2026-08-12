import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from brain.llm import LLM_V1
from brain.prompts import (
    SUB_SITUATION_SUMMARY_SYSTEM,
    build_sub_situation_summary_prompt,
)
from config import KNOWLEDGE_DB
from graph_db import DatabaseManager


SUMMARY_SCHEMA = {"summary": "string"}


def load_sub_situations(
    db_manager: DatabaseManager,
    db_name: str,
    overwrite: bool = False,
    limit: int = 0,
) -> List[Dict[str, Any]]:
    query = """
    MATCH (sub:SubSituation)
    WHERE coalesce(sub.description, "") <> ""
      AND ($overwrite OR coalesce(sub.summary, "") = "")
    RETURN sub.id AS id,
           coalesce(sub.name, "") AS name,
           coalesce(sub.description, "") AS description,
           coalesce(sub.summary, "") AS summary
    ORDER BY sub.id
    """

    if limit > 0:
        query += "\nLIMIT $limit"

    params = {"overwrite": overwrite, "limit": limit}
    with db_manager.get_session(db_name) as session:
        return [dict(record) for record in session.run(query, params)]


def save_summary(
    db_manager: DatabaseManager,
    db_name: str,
    sub_situation_id: str,
    summary: str,
) -> None:
    with db_manager.get_session(db_name) as session:
        session.run(
            """
            MATCH (sub:SubSituation {id: $id})
            SET sub.summary = $summary
            """,
            {"id": sub_situation_id, "summary": summary},
        )


def generate_summary(llm: LLM_V1, name: str, description: str) -> str:
    response = llm.invoke_with_formated_response(
        query=build_sub_situation_summary_prompt(name=name, description=description),
        formate=SUMMARY_SCHEMA,
    )

    if not isinstance(response, dict):
        raise ValueError(f"LLM returned non-object response: {response!r}")

    summary = str(response.get("summary", "")).strip()
    if not summary:
        raise ValueError("LLM returned an empty summary.")

    return summary


def backfill_summaries(
    db_manager: DatabaseManager,
    db_name: str = KNOWLEDGE_DB,
    overwrite: bool = False,
    limit: int = 0,
    delay_seconds: float = 0.0,
) -> Dict[str, Any]:
    rows = load_sub_situations(
        db_manager=db_manager,
        db_name=db_name,
        overwrite=overwrite,
        limit=limit,
    )
    llm = LLM_V1(system_message=SUB_SITUATION_SUMMARY_SYSTEM)

    result = {
        "selected": len(rows),
        "updated": 0,
        "failed": 0,
        "errors": [],
    }

    for index, row in enumerate(rows, start=1):
        sub_id = str(row["id"])
        print(f"[{index}/{len(rows)}] Summarizing sub-situation {sub_id}: {row['name']}")

        try:
            summary = generate_summary(
                llm=llm,
                name=row["name"],
                description=row["description"],
            )
            save_summary(
                db_manager=db_manager,
                db_name=db_name,
                sub_situation_id=sub_id,
                summary=summary,
            )
            result["updated"] += 1
        except Exception as exc:
            result["failed"] += 1
            result["errors"].append(
                {
                    "id": sub_id,
                    "name": row["name"],
                    "error": str(exc),
                }
            )
            print(f"FAILED {sub_id}: {exc}")

        if delay_seconds > 0:
            time.sleep(delay_seconds)

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and store LLM summaries for SubSituation nodes."
    )
    parser.add_argument(
        "--db-name",
        default=KNOWLEDGE_DB,
        help=f"Neo4j database name. Default: {KNOWLEDGE_DB}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate summaries even when a summary already exists.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process the first N matching nodes. 0 means no limit.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.0,
        help="Optional delay between LLM calls.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_manager = DatabaseManager()
    try:
        result = backfill_summaries(
            db_manager=db_manager,
            db_name=args.db_name,
            overwrite=args.overwrite,
            limit=args.limit,
            delay_seconds=args.delay_seconds,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        db_manager.close()


if __name__ == "__main__":
    main()
