import json
from pathlib import Path

from graph_db import DatabaseManager
from db_schema.services import ServiceBWGraphWriter
from scrapping.all_links_scrapping import main as all_links_scrapping
from config import KNOWLEDGE_DB


SCRAPPING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRAPPING_DIR.parent
SERVICE_BW_OUTPUT_PATH = SCRAPPING_DIR / "service_bw_output.json"
LEGACY_SERVICE_BW_OUTPUT_PATH = PROJECT_ROOT / "service_bw_output.json"


def load_service_bw_output(path: str | Path | None = None):
    candidates = [
        Path(path) if path else None,
        SERVICE_BW_OUTPUT_PATH,
        LEGACY_SERVICE_BW_OUTPUT_PATH,
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            with candidate.open("r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(
        f"Could not find service-bw output JSON. Expected {SERVICE_BW_OUTPUT_PATH}"
    )


def main():
    output_path = all_links_scrapping()
    data = load_service_bw_output(output_path)

    payload = {
        "situations": data.get("situations", []),
        "derive_dependencies": False
    }

    db_manager = DatabaseManager()
    writer = ServiceBWGraphWriter(db_manager, db_name=KNOWLEDGE_DB)

    writer.ensure_constraints()

    result = writer.insert_situation_tree(payload["situations"])
    print(result)

if __name__ == "__main__":
    main()
