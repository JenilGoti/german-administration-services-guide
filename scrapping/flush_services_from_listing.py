import json
import time
from typing import Any, Dict, List

from graph_db import DatabaseManager
from db_schema.services import ServiceBWGraphWriter
from scrapping.scrape_extract_save_pipeline_service import SingleServiceFlushAgent


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def service_sort_key(service: Dict[str, Any]):
    service_id = str(service.get("id", ""))
    return int(service_id) if service_id.isdigit() else service_id


def extract_unique_services(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    services_by_id = {}

    for situation in data.get("situations", []):
        for sub in situation.get("sub_situations", []):
            for service in sub.get("services", []):
                service_id = str(service.get("id", ""))
                if not service_id:
                    continue

                services_by_id[service_id] = {
                    "id": service_id,
                    "name": service.get("name", ""),
                    "url": service.get("url") or f"https://www.service-bw.de/zufi/leistungen/{service_id}",
                }

    return sorted(services_by_id.values(), key=service_sort_key)


def filter_services(
    services: List[Dict[str, Any]],
    start_from: str = "",
    limit: int = 0,
) -> List[Dict[str, Any]]:
    if start_from:
        services = [
            service
            for service in services
            if service_sort_key(service) >= service_sort_key({"id": start_from})
        ]

    if limit and limit > 0:
        services = services[:limit]

    return services


def print_service_list(services: List[Dict[str, Any]]) -> None:
    for index, service in enumerate(services, start=1):
        print(f"{index:04d}. {service['id']} | {service.get('name', '')} | {service['url']}")


def flush_services(
    services: List[Dict[str, Any]],
    graph_writer,
    delay_seconds: float = 0.5,
) -> Dict[str, Any]:
    agent = SingleServiceFlushAgent(graph_writer=graph_writer)

    summary = {
        "total": len(services),
        "success": 0,
        "failed": 0,
        "errors": [],
    }

    for index, service in enumerate(services, start=1):
        service_id = service["id"]
        url = service["url"]

        print(f"[{index}/{len(services)}] Flushing service {service_id}: {service.get('name', '')}")

        try:
            result = agent.run(url)
            summary["success"] += 1
            print(result.get("response", "OK"))
        except Exception as exc:
            summary["failed"] += 1
            summary["errors"].append(
                {
                    "id": service_id,
                    "url": url,
                    "error": str(exc),
                }
            )
            print(f"FAILED {service_id}: {exc}")

        if delay_seconds > 0:
            time.sleep(delay_seconds)

    return summary


def main():

    data = load_json("scrapping/service_bw_output.json")
    # data = load_json("scrapping/failed_to_scrape.json")
    # services = data['errors']
    services = extract_unique_services(data)
    services = filter_services(
        services,
        start_from="",
        limit=0
    )

    print(f"Services selected: {len(services)}")


    # Uncomment and adjust these lines in your project.
    db_manager = DatabaseManager()
    writer = ServiceBWGraphWriter(db_manager, db_name="dev-graph", enable_embeddings=True)
    writer.ensure_constraints()
    summary = flush_services(services, graph_writer=writer, delay_seconds=0.5)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
