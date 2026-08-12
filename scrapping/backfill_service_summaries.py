import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from brain.llm import LLM_V1
from brain.prompts import SERVICE_SUMMARY_SYSTEM, build_service_summary_prompt
from config import KNOWLEDGE_DB, LLM_PROVIDER
from graph_db import DatabaseManager


SUMMARY_SCHEMA = {"summary": "string"}


def load_services(
    db_manager: DatabaseManager,
    db_name: str,
    overwrite: bool = False,
    limit: int = 0,
) -> List[Dict[str, Any]]:
    query = """
    MATCH (svc:Service)
    WHERE $overwrite OR coalesce(svc.summary, "") = ""

    CALL (svc) {
      OPTIONAL MATCH (svc)-[:HAS_SECTION]->(sec:ServiceSection)
      RETURN collect(DISTINCT {
        title: sec.title,
        type: sec.type,
        text: sec.text,
        order: sec.order
      }) AS sections
    }
    CALL (svc) {
      OPTIONAL MATCH (svc)-[:REQUIRES]->(req:Requirement)
      OPTIONAL MATCH (req)-[:REQUIRES_DOCUMENT]->(doc:Document)
      RETURN collect(DISTINCT {
        text: req.text,
        type: req.type,
        mandatory: req.mandatory,
        document_name: doc.name,
        document_description: doc.description
      }) AS requirements
    }
    CALL (svc) {
      OPTIONAL MATCH (svc)-[:ISSUES]->(doc:Document)
      RETURN collect(DISTINCT {
        name: doc.name,
        description: doc.description
      }) AS issued_documents
    }
    CALL (svc) {
      OPTIONAL MATCH (svc)-[:HANDLED_BY]->(auth:Authority)
      RETURN collect(DISTINCT {
        name: auth.name,
        type: auth.type,
        location_context: auth.locationContext
      }) AS authorities
    }
    CALL (svc) {
      OPTIONAL MATCH (svc)-[:HAS_FORM]->(form:Form)
      RETURN collect(DISTINCT {
        name: form.name,
        type: form.type
      }) AS forms
    }
    CALL (svc) {
      OPTIONAL MATCH (svc)-[:HAS_STEP]->(step:ProcessStep)
      RETURN collect(DISTINCT {
        title: step.title,
        description: step.description,
        order: step.order,
        channel: step.channel
      }) AS process_steps
    }
    CALL (svc) {
      OPTIONAL MATCH (svc)-[:HAS_LEGAL_BASIS]->(law:LegalBasis)
      RETURN collect(DISTINCT {
        title: law.title,
        law_code: law.lawCode,
        paragraph: law.paragraph,
        text: law.text
      }) AS legal_basis
    }
    CALL (svc) {
      OPTIONAL MATCH (goal:Goal)-[:ACHIEVED_BY]->(svc)
      RETURN collect(DISTINCT {
        name: goal.name,
        description: goal.description
      }) AS goals
    }
    CALL (svc) {
      OPTIONAL MATCH (svc)-[:HAS_PROBLEM]->(problem:DependencyProblem)
      RETURN collect(DISTINCT {
        type: problem.type,
        description: problem.description,
        severity: problem.severity
      }) AS dependency_problems
    }

    WITH svc,
         sections,
         requirements,
         issued_documents,
         authorities,
         forms,
         process_steps,
         legal_basis,
         goals,
         dependency_problems
    WHERE coalesce(svc.description, "") <> ""
       OR size(sections) > 0
       OR size(requirements) > 0
       OR size(issued_documents) > 0
       OR size(authorities) > 0
       OR size(forms) > 0
       OR size(process_steps) > 0
       OR size(legal_basis) > 0
       OR size(goals) > 0
       OR size(dependency_problems) > 0

    RETURN svc.id AS id,
           coalesce(svc.name, "") AS name,
           coalesce(svc.description, "") AS description,
           coalesce(svc.summary, "") AS summary,
           sections,
           requirements,
           issued_documents,
           authorities,
           forms,
           process_steps,
           legal_basis,
           goals,
           dependency_problems
    ORDER BY svc.id
    """

    if limit > 0:
        query += "\nLIMIT $limit"

    with db_manager.get_session(db_name) as session:
        return [
            clean_record(dict(record))
            for record in session.run(query, {"overwrite": overwrite, "limit": limit})
        ]


def clean_record(row: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = dict(row)
    for key in [
        "sections",
        "requirements",
        "issued_documents",
        "authorities",
        "forms",
        "process_steps",
        "legal_basis",
        "goals",
        "dependency_problems",
    ]:
        cleaned[key] = [
            item
            for item in (cleaned.get(key) or [])
            if isinstance(item, dict) and any(value not in (None, "") for value in item.values())
        ]
    return cleaned


def save_summary(
    db_manager: DatabaseManager,
    db_name: str,
    service_id: str,
    summary: str,
) -> None:
    with db_manager.get_session(db_name) as session:
        session.run(
            """
            MATCH (svc:Service {id: $id})
            SET svc.summary = $summary
            """,
            {"id": service_id, "summary": summary},
        )


def generate_summary(
    row: Dict[str, Any],
    provider: str,
    role: str,
) -> str:
    llm = LLM_V1(
        system_message=SERVICE_SUMMARY_SYSTEM,
        provider=provider,
        role=role,
        temperature=0.1,
    )
    response = llm.invoke_with_formated_response(
        query=build_service_summary_prompt(row),
        formate=SUMMARY_SCHEMA,
    )

    if not isinstance(response, dict):
        raise ValueError(f"LLM returned non-object response: {response!r}")

    summary = str(response.get("summary", "")).strip()
    if not summary:
        raise ValueError("LLM returned an empty summary.")

    return summary


def process_service(
    db_manager: DatabaseManager,
    db_name: str,
    row: Dict[str, Any],
    provider: str,
    role: str,
    delay_seconds: float,
) -> Dict[str, Any]:
    if delay_seconds > 0:
        time.sleep(delay_seconds)

    summary = generate_summary(row=row, provider=provider, role=role)
    save_summary(
        db_manager=db_manager,
        db_name=db_name,
        service_id=str(row["id"]),
        summary=summary,
    )
    return {"id": str(row["id"]), "name": row["name"]}


def backfill_summaries(
    db_manager: DatabaseManager,
    db_name: str = KNOWLEDGE_DB,
    overwrite: bool = False,
    limit: int = 0,
    delay_seconds: float = 0.0,
    workers: int = 4,
    provider: str = LLM_PROVIDER,
    role: str = "reasoning",
) -> Dict[str, Any]:
    rows = load_services(
        db_manager=db_manager,
        db_name=db_name,
        overwrite=overwrite,
        limit=limit,
    )
    result = {
        "selected": len(rows),
        "updated": 0,
        "failed": 0,
        "errors": [],
    }

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                process_service,
                db_manager,
                db_name,
                row,
                provider,
                role,
                delay_seconds,
            ): row
            for row in rows
        }

        for index, future in enumerate(as_completed(futures), start=1):
            row = futures[future]
            service_id = str(row["id"])
            try:
                future.result()
                result["updated"] += 1
                print(f"[{index}/{len(rows)}] Summarized service {service_id}: {row['name']}")
            except Exception as exc:
                result["failed"] += 1
                result["errors"].append(
                    {
                        "id": service_id,
                        "name": row["name"],
                        "error": str(exc),
                    }
                )
                print(f"FAILED {service_id}: {exc}")

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and store detailed LLM summaries for Service nodes."
    )
    parser.add_argument("--db-name", default=KNOWLEDGE_DB)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--delay-seconds", type=float, default=0.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of services to summarize in parallel. Default: 4.",
    )
    parser.add_argument(
        "--provider",
        default=LLM_PROVIDER,
        help=f"LLM provider to use. Default: {LLM_PROVIDER}",
    )
    parser.add_argument(
        "--role",
        default="reasoning",
        help="Model role to use. Default: reasoning.",
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
            workers=args.workers,
            provider=args.provider,
            role=args.role,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        db_manager.close()


if __name__ == "__main__":
    main()
