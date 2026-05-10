import json
from typing import Any, Dict, TypedDict

from langgraph.graph import END, START, StateGraph

from scrapping.agent_page_scraper_tool import AgentPageScraperTool
from brain.llm import LLM_V1
from brain.prompts import SERVICE_EXTRACT_SYSTEM, build_service_extract_prompt


class SingleServiceFlushState(TypedDict, total=False):
    url: str
    scraped: Dict[str, Any]
    extracted: Dict[str, Any]
    db_result: Dict[str, Any]
    response: str


class SingleServiceFlushAgent:
    """
    Pipeline for one Service page only:
    1. Scrape one Service-BW service URL.
    2. Send page text to LLM.
    3. Get service-only structured JSON.
    4. Save full service-related schema data to Neo4j.

    The Service.id is forced to the numeric Service-BW id from the URL/API.
    Example: https://www.service-bw.de/zufi/leistungen/599 => Service.id = "599"
    """

    output_schema = {
        "services": [
            {
                "id": "string",
                "name": "string",
                "url": "string",
                "description": "string",
                "source": "service-bw",
                "regionalisierbar": None,
                "scrapedAt": "string",
            }
        ],
        "service_sections": [
            {
                "id": "string (service_id + '_' + order)",
                "service_id": "string",
                "type": "string",
                "title": "string",
                "text": "string",
                "html": "string",
                "order": 0,
            }
        ],
        "requirements": [
            {
                "id": "string (service_id + '_' + order)",
                "service_id": "string",
                "text": "string",
                "type": "document|condition|action|legal|location|payment|identity|appointment",
                "mandatory": True,
                "confidence": 0.0,
                "sourceText": "string",
                "document_id": None,
                "document_name": "string",
                "document_normalized_name": "string",
                "document_description": "string",
                "document_language": "de",
            }
        ],
        "document_issuers": [
            {
                "service_id": "string",
                "document_id": "string (service_id + '_' + order)",
            }
        ],
        "authorities": [
            {
                "id": "string (service_id + '_' + order)",
                "service_id": "string",
                "name": "string",
                "type": "string",
                "address": "string",
                "phone": "string",
                "email": "string",
                "locationContext": "string",
            }
        ],
        "forms": [
            {
                "id": "string (service_id + '_' + order)",
                "service_id": "string",
                "name": "string",
                "url": "string",
                "type": "string",
            }
        ],
        "process_steps": [
            {
                "id": "string (service_id + '_' + order)",
                "service_id": "string",
                "title": "string",
                "description": "string",
                "order": 0,
                "channel": "string",
            }
        ],
        "legal_basis": [
            {
                "id": "string (service_id + '_' + order)",
                "service_id": "string",
                "title": "string",
                "lawCode": "string",
                "paragraph": "string",
                "url": "string",
                "text": "string",
            }
        ],
        "goals": [
            {
                "id": "string (service_id + '_' + order)",
                "name": "string",
                "description": "string",
                "service_id": "string",
            }
        ],
        "dependency_problems": [
            {
                "id": "string (service_id + '_' + order)",
                "type": "string",
                "description": "string",
                "severity": "string",
                "detectedAt": "string",
                "service_id": "string",
                "document_id": "string",
            }
        ],
        "derive_dependencies": True,
    }

    def __init__(self, graph_writer):
        self.scraper = AgentPageScraperTool()
        self.graph_writer = graph_writer
        self.llm = LLM_V1(system_message=SERVICE_EXTRACT_SYSTEM)
        self.app = self.build_graph()

    # -----------------------------
    # NODE 1: SCRAPE ONE SERVICE
    # -----------------------------
    def scrape_node(self, state: SingleServiceFlushState):
        scraped = self.scraper.scrape(state["url"])

        if scraped.get("sourceType") != "service":
            raise ValueError("This pipeline only accepts Service-BW service URLs.")

        return {
            **state,
            "scraped": scraped,
        }

    # -----------------------------
    # NODE 2: LLM EXTRACT FULL SERVICE DATA
    # -----------------------------
    def extract_node(self, state: SingleServiceFlushState):
        prompt = self._build_service_prompt(state["scraped"])

        extracted = self.llm.invoke_with_formated_response(
            query=prompt,
            formate=self.output_schema,
        )

        return {
            **state,
            "extracted": self._normalize_service_payload(
                extracted,
                state["scraped"],
            ),
        }

    # -----------------------------
    # NODE 3: SAVE FULL SERVICE DATA
    # -----------------------------
    def save_node(self, state: SingleServiceFlushState):
        payload = state["extracted"]
        # print(json.dumps(payload, ensure_ascii=False, indent=2))
        db_result = self.graph_writer.bulk_import(payload)

        return {
            **state,
            "db_result": db_result,
            "response": json.dumps(
                {
                    "url": state["url"],
                    "saved": db_result,
                },
                ensure_ascii=False,
                indent=2,
            ),
        }

    # -----------------------------
    # GRAPH
    # -----------------------------
    def build_graph(self):
        builder = StateGraph(SingleServiceFlushState)

        builder.add_node("scrape", self.scrape_node)
        builder.add_node("extract", self.extract_node)
        builder.add_node("save", self.save_node)

        builder.add_edge(START, "scrape")
        builder.add_edge("scrape", "extract")
        builder.add_edge("extract", "save")
        builder.add_edge("save", END)

        return builder.compile()

    def run(self, url: str):
        return self.app.invoke({"url": url})

    # -----------------------------
    # PROMPT
    # -----------------------------
    def _build_service_prompt(self, scraped: Dict[str, Any]) -> str:
        return build_service_extract_prompt(scraped)

    # -----------------------------
    # FULL SERVICE PAYLOAD, WITH SERVICE ID NORMALIZATION
    # -----------------------------
    def _normalize_service_payload(
        self,
        payload: Dict[str, Any],
        scraped: Dict[str, Any],
    ) -> Dict[str, Any]:
        service_id = str(scraped.get("sourceId", ""))
        service_url = scraped.get("url", "")
        service_name = scraped.get("title", "")

        services = payload.get("services", [])
        if not services:
            services = [{}]

        normalized_services = []
        for service in services:
            normalized_services.append(
                {
                    **service,
                    "id": service_id,
                    "name": service.get("name") or service_name,
                    "url": service.get("url") or service_url,
                    "source": service.get("source") or "service-bw",
                }
            )

        def force_service_id(items):
            fixed = []
            for item in items or []:
                fixed.append({**item, "service_id": service_id})
            return fixed

        document_issuers = []
        for item in payload.get("document_issuers", []) or []:
            document_id = item.get("document_id")
            if document_id:
                document_issuers.append(
                    {
                        **item,
                        "service_id": service_id,
                    }
                )

        return {
            "situations": [],
            "services": normalized_services,
            "service_sections": force_service_id(payload.get("service_sections", [])),
            "requirements": force_service_id(payload.get("requirements", [])),
            "document_issuers": document_issuers,
            "authorities": force_service_id(payload.get("authorities", [])),
            "forms": force_service_id(payload.get("forms", [])),
            "process_steps": force_service_id(payload.get("process_steps", [])),
            "legal_basis": force_service_id(payload.get("legal_basis", [])),
            "goals": force_service_id(payload.get("goals", [])),
            "dependency_problems": force_service_id(payload.get("dependency_problems", [])),
            "derive_dependencies": payload.get("derive_dependencies", True),
        }

if __name__ == "__main__":
    def main():
        scraper = AgentPageScraperTool()
        graph_writer = ServiceBWGraphWriter(db_name="dev-graph")
        pipeline = Pipeline(graph_writer)
        pipeline.run("https://www.service-bw.de/zufi/leistungen/1")
    main()
