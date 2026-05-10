import json
import time
from pathlib import Path
from typing import Any, Dict, List

from db_schema.services import ServiceBWGraphWriter
from graph_db import DatabaseManager
from scrapping.scrape_extract_save_pipeline_sub_situation import SingleSubSituationFlushAgent
from config import KNOWLEDGE_DB

SCRAPPING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRAPPING_DIR.parent
SERVICE_BW_OUTPUT_PATH = SCRAPPING_DIR / "service_bw_output.json"
LEGACY_SERVICE_BW_OUTPUT_PATH = PROJECT_ROOT / "service_bw_output.json"


def load_json(path: str | Path = SERVICE_BW_OUTPUT_PATH) -> Dict[str, Any]:
    candidates = [Path(path), SERVICE_BW_OUTPUT_PATH, LEGACY_SERVICE_BW_OUTPUT_PATH]
    for candidate in candidates:
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as file:
                return json.load(file)
    raise FileNotFoundError(f"Could not find service-bw output JSON. Expected {SERVICE_BW_OUTPUT_PATH}")


def sub_situation_sort_key(sub_situation: Dict[str, Any]):
    sub_id = str(sub_situation.get("id", ""))
    return int(sub_id) if sub_id.isdigit() else sub_id


def extract_unique_sub_situations(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    sub_situations_by_id = {}

    for situation in data.get("situations", []):
        for sub in situation.get("sub_situations", []):
            sub_id = str(sub.get("id", ""))
            if not sub_id:
                continue

            sub_situations_by_id[sub_id] = {
                "id": sub_id,
                "name": sub.get("name", ""),
                "url": sub.get("url") or f"https://www.service-bw.de/zufi/lebenslagen/{sub_id}",
            }

    return sorted(sub_situations_by_id.values(), key=sub_situation_sort_key)


def filter_sub_situations(
    sub_situations: List[Dict[str, Any]],
    start_from: str = "",
    limit: int = 0,
) -> List[Dict[str, Any]]:
    if start_from:
        sub_situations = [
            sub
            for sub in sub_situations
            if sub_situation_sort_key(sub) >= sub_situation_sort_key({"id": start_from})
        ]

    if limit and limit > 0:
        sub_situations = sub_situations[:limit]

    return sub_situations


def flush_sub_situations(
    sub_situations: List[Dict[str, Any]],
    graph_writer,
    delay_seconds: float = 0.5,
) -> Dict[str, Any]:
    agent = SingleSubSituationFlushAgent(graph_writer=graph_writer)

    summary = {
        "total": len(sub_situations),
        "success": 0,
        "failed": 0,
        "errors": [],
    }

    for index, sub in enumerate(sub_situations, start=1):
        sub_id = sub["id"]
        url = sub["url"]

        print(f"[{index}/{len(sub_situations)}] Flushing sub-situation {sub_id}: {sub.get('name', '')}")

        try:
            result = agent.run(url)
            summary["success"] += 1
            print(result.get("response", "OK"))
        except Exception as exc:
            summary["failed"] += 1
            summary["errors"].append(
                {
                    "id": sub_id,
                    "url": url,
                    "error": str(exc),
                }
            )
            print(f"FAILED {sub_id}: {exc}")

        if delay_seconds > 0:
            time.sleep(delay_seconds)

    return summary


def main():
    data = load_json()
    sub_situations = extract_unique_sub_situations(data)
    sub_situations = filter_sub_situations(
        sub_situations,
        start_from="",
        limit=0,
    )

    print(f"Sub-situations selected: {len(sub_situations)}")

    db_manager = DatabaseManager()
    writer = ServiceBWGraphWriter(db_manager, db_name=KNOWLEDGE_DB, enable_embeddings=True)

    summary = flush_sub_situations(
        sub_situations,
        graph_writer=writer,
        delay_seconds=0.5,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
