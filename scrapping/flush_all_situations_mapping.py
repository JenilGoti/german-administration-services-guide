import json

from graph_db import DatabaseManager
from db_schema.services import ServiceBWGraphWriter

def main():
    with open("scrapping/service_bw_output.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    payload = {
            "situations": data.get("situations", []),
        "derive_dependencies": False
    }

    db_manager = DatabaseManager()
    writer = ServiceBWGraphWriter(db_manager, db_name="dev-graph")

    writer.ensure_constraints()

    result = writer.insert_situation_tree(payload["situations"])
    print(result)

if __name__ == "__main__":
    main()