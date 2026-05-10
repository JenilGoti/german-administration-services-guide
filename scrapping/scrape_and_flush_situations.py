import json
from typing import Any, Dict

from db_schema.services import ServiceBWGraphWriter
from graph_db import DatabaseManager
from scrapping.all_links_scrapping import ServiceBWCrawler
from config import KNOWLEDGE_DB


def save_json(data: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def main() -> None:
    crawler = ServiceBWCrawler(delay_seconds=0.1)
    data = crawler.crawl()

    save_json(data, "scrapping/service_bw_output.json")

    db_manager = DatabaseManager()
    writer = ServiceBWGraphWriter(
        db_manager,
        db_name=KNOWLEDGE_DB,
        enable_embeddings=True,
    )
    writer.ensure_constraints()

    result = writer.insert_situation_tree(data.get("situations", []))

    print(
        json.dumps(
            {
                "scraped": data.get("summary", {}),
                "saved": result,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
