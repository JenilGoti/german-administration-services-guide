import json
from typing import Any, Dict, TypedDict

from langgraph.graph import END, START, StateGraph

from brain.llm import LLM_V1
from scrapping.agent_page_scraper_tool import AgentPageScraperTool


class SingleSubSituationFlushState(TypedDict, total=False):
    url: str
    scraped: Dict[str, Any]
    extracted: Dict[str, Any]
    db_result: Dict[str, Any]
    response: str


class SingleSubSituationFlushAgent:
    """
    Pipeline for one Service-BW sub-situation page:
    1. Scrape one /zufi/lebenslagen/{id} page through the Service-BW API.
    2. Ask the LLM for a clean description and concise summary.
    3. Store the enriched SubSituation node in Neo4j.
    4. Let the graph writer create embedding chunks for the saved text.
    """

    output_schema = {
        "sub_situations": [
            {
                "id": "string",
                "name": "string",
                "url": "string",
                "description": "string",
                "summary": "string",
                "source": "service-bw",
                "rawText": "string",
                "textHash": "string",
                "scrapedAt": "string",
            }
        ]
    }

    def __init__(self, graph_writer):
        self.scraper = AgentPageScraperTool()
        self.graph_writer = graph_writer
        self.llm = LLM_V1(
            system_message="""
            You are an expert German public-administration content cleaner.
            Extract and summarize one Service-BW life-situation page.
            Return only valid JSON.
            Do not include markdown.
            Do not invent facts that are not supported by the page text.
            """
        )
        self.app = self.build_graph()

    def scrape_node(self, state: SingleSubSituationFlushState):
        scraped = self.scraper.scrape(state["url"])

        if scraped.get("sourceType") not in {"situation", "sub_situation"}:
            raise ValueError("This pipeline only accepts Service-BW /lebenslagen/ URLs.")

        return {
            **state,
            "scraped": scraped,
        }

    def extract_node(self, state: SingleSubSituationFlushState):
        prompt = self._build_sub_situation_prompt(state["scraped"])
        # extracted = self.llm.invoke_with_formated_response(
        #     query=prompt,
        #     formate=self.output_schema,
        # )
        extracted = {}

        return {
            **state,
            "extracted": self._normalize_sub_situation_payload(
                extracted,
                state["scraped"],
            ),
        }

    def save_node(self, state: SingleSubSituationFlushState):
        payload = state["extracted"]
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

    def build_graph(self):
        builder = StateGraph(SingleSubSituationFlushState)

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

    def _build_sub_situation_prompt(self, scraped: Dict[str, Any]) -> str:
        context = {
            "url": scraped.get("url", ""),
            "sourceType": scraped.get("sourceType", ""),
            "sourceId": scraped.get("sourceId", ""),
            "title": scraped.get("title", ""),
            "rawText": scraped.get("rawText", ""),
            "description": (scraped.get("situation") or {}).get("description", ""),
            "services": scraped.get("services") or [],
            "subSituations": scraped.get("subSituations") or [],
        }

        return f"""
Scraped Service-BW sub-situation page:
{json.dumps(context, ensure_ascii=False, indent=2)}

Extract structured data for ONE sub-situation page.

Rules:
- Output only one sub_situations item.
- SubSituation.id MUST be "{scraped.get("sourceId", "")}".
- Use the page title as name if present.
- description should be cleaned German page content, preserving useful administrative meaning.
- summary should be a concise 2-4 sentence German summary for retrieval and user matching.
- rawText MUST use the provided raw page text, not a rewritten version.
- Do not output services as separate graph records.
- Do not output situations.
- If a field is unknown, use an empty string.
"""

    def _normalize_sub_situation_payload(
        self,
        payload: Dict[str, Any],
        scraped: Dict[str, Any],
    ) -> Dict[str, Any]:
        sub_id = str(scraped.get("sourceId", ""))
        sub_url = scraped.get("url", "")
        sub_name = scraped.get("title", "")
        raw_text = scraped.get("rawText", "")
        situation = scraped.get("situation") or {}

        sub_situations = payload.get("sub_situations", [])
        sub = sub_situations[0] if sub_situations else {}

        return {
            "sub_situations": [
                {
                    **sub,
                    "id": sub_id,
                    "name": sub.get("name") or sub_name,
                    "url": sub.get("url") or sub_url,
                    "description": sub.get("description") or situation.get("description", ""),
                    "summary": sub.get("summary", ""),
                    "source": sub.get("source") or "service-bw",
                    "rawText": raw_text,
                    "textHash": scraped.get("textHash", ""),
                    "scrapedAt": scraped.get("scrapedAt", ""),
                }
            ],
            "derive_dependencies": False,
        }
