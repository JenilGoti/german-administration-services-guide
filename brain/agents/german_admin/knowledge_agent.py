from typing import Any, Dict
import json

from langgraph.prebuilt import ToolNode

from brain.agents.german_admin.helpers import compact_terms, fallback_knowledge_query
from brain.agents.german_admin.schemas import KnowledgeInput, KnowledgeOutput
from brain.logger import logger
from brain.tool_registry import ToolRegistry
from utilities import safe_json_parse


class KnowledgeAgent:
    InputSchema = KnowledgeInput
    OutputSchema = KnowledgeOutput

    def __init__(self):
        registry = ToolRegistry()
        self.tool_node = ToolNode(
            registry.get_langchain_tools([
                "search_problem_knowledge",
                "web_search",
                "scrape",
            ])
        )

    def run(self, data: KnowledgeInput) -> KnowledgeOutput:
        query = data.get("knowledge_query") or fallback_knowledge_query(data["query"], {})
        search_terms = self._knowledge_search_terms(data)
        extra_terms = data.get("supervisor", {}).get("extra_search_terms", [])
        if extra_terms:
            search_terms = compact_terms([*search_terms, *extra_terms])

        graph_search = self._search_knowledge(search_terms)
        web_findings = self._web_enrichment(data, search_terms, graph_search)

        return {
            "findings": {
                "query": query,
                "knowledge_query": query,
                "search_terms": search_terms,
                "web_search_terms": data.get("web_search_terms", []),
                "search_strategy": "Compact German situation queries -> SubSituation vector search + Service vector fallback + ServiceQA vector search -> service details -> web fallback when needed.",
                "search_results": {
                    "SubSituation": self._trim_items(graph_search.get("SubSituation", []), limit=3),
                    "Service": self._trim_items(graph_search.get("Service", []), limit=3),
                    "ServiceQA": self._trim_items(graph_search.get("ServiceQA", []), limit=4),
                },
                "service_details": self._trim_items(graph_search.get("services", []), limit=3),
                "web_findings": self._trim_web_findings(web_findings),
                "tool_error": graph_search.get("error"),
            }
        }

    def _knowledge_search_terms(self, data: KnowledgeInput) -> list[str]:
        terms = [
            *(data.get("knowledge_queries") or []),
            data.get("knowledge_query", ""),
            data.get("german_query", ""),
            *(data.get("german_search_terms") or [])[:3],
        ]
        return compact_terms(terms, limit=5)

    def _search_knowledge(self, search_terms: list[str]) -> Dict[str, Any]:
        tool_call = {
            "tool": "search_problem_knowledge",
            "args": {"queries": search_terms, "top_k": 3},
        }
        self._log_tool("[TOOL_CALL]", tool_call)

        try:
            graph_search = self._run_tool("search_problem_knowledge", tool_call["args"])
            if not isinstance(graph_search, dict):
                raise TypeError(f"Invalid KB tool result: {type(graph_search).__name__}")
            self._log_tool("[TOOL_RESULT]", {
                "status": "success",
                "sub_situation_count": len(graph_search.get("SubSituation", [])),
                "service_vector_count": len(graph_search.get("Service", [])),
                "service_qa_count": len(graph_search.get("ServiceQA", [])),
                "service_count": len(graph_search.get("services", [])),
            })
            return graph_search
        except Exception as exc:
            error = {
                **tool_call,
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            self._log_tool("[TOOL_ERROR]", error)
            return {"SubSituation": [], "Service": [], "ServiceQA": [], "services": [], "error": error}

    def _web_enrichment(self, data: KnowledgeInput, search_terms: list[str], graph_search: Dict[str, Any]):
        service_count = len(graph_search.get("services", []))
        location_known = self._has_location_context(data, search_terms)
        if service_count >= 2 and graph_search.get("SubSituation") and not location_known:
            return {}

        web_terms = data.get("web_search_terms") or search_terms
        query = web_terms[0] if web_terms else data.get("german_query", data["query"])
        if "service-bw" not in query.lower():
            query = f"{query} service-bw"

        tool_call = {
            "tool": "web_search",
            "args": {"query": query, "max_results": 5},
        }
        self._log_tool("[TOOL_CALL]", tool_call)

        try:
            results = self._run_tool("web_search", tool_call["args"])
            results = results if isinstance(results, list) else []
            selected_results = results[:3]
            scraped_pages = []
            for result in selected_results[:2]:
                url = result.get("url")
                if not url:
                    continue
                self._log_tool("[TOOL_CALL]", {"tool": "scrape", "args": {"url": url}})
                scraped = self._run_tool("scrape", {"url": url})
                scraped_pages.append(scraped if isinstance(scraped, dict) else {"url": url, "status": "error"})

            self._log_tool("[TOOL_RESULT]", {
                **tool_call,
                "status": "success",
                "result_count": len(results),
                "scraped_count": len(scraped_pages),
            })
            return {"query": query, "results": selected_results, "scraped_pages": scraped_pages}
        except Exception as exc:
            error = {
                **tool_call,
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            self._log_tool("[TOOL_ERROR]", error)
            return {"query": query, "error": error}

    def _trim_web_findings(self, web_findings):
        if not isinstance(web_findings, dict):
            return web_findings

        return {
            "query": web_findings.get("query", ""),
            "error": web_findings.get("error"),
            "results": self._trim_items(web_findings.get("results", []), limit=3),
            "scraped_pages": self._trim_items(web_findings.get("scraped_pages", []), limit=2),
        }

    def _trim_items(self, items, limit: int):
        if not isinstance(items, list):
            return items
        return [self._trim_value(item) for item in items[:limit]]

    def _trim_value(self, value, max_chars: int = 1200):
        if isinstance(value, str):
            if len(value) <= max_chars:
                return value
            return value[:max_chars].rstrip() + "..."

        if isinstance(value, list):
            return [self._trim_value(item, max_chars=max_chars) for item in value[:8]]

        if isinstance(value, dict):
            trimmed = {}
            for key, item in value.items():
                if key in {"raw", "html", "markdown", "content", "text", "page_content"}:
                    trimmed[key] = self._trim_value(item, max_chars=max_chars)
                else:
                    trimmed[key] = self._trim_value(item, max_chars=max_chars)
            return trimmed

        return value

    def _run_tool(self, tool_name: str, args: Dict[str, Any]):
        from langchain_core.messages import AIMessage

        result = self.tool_node.invoke({
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": tool_name, "args": args, "id": f"{tool_name}_call"}],
                )
            ]
        })
        tool_messages = result.get("messages", [])
        if not tool_messages:
            return None

        content = getattr(tool_messages[-1], "content", tool_messages[-1])
        if isinstance(content, str):
            parsed = safe_json_parse(content)
            return parsed if parsed is not None else content
        return content

    def _has_location_context(self, data: KnowledgeInput, search_terms: list[str]) -> bool:
        text = " ".join([
            data.get("query", ""),
            data.get("knowledge_query", ""),
            " ".join(search_terms or []),
            " ".join(data.get("web_search_terms", [])),
        ]).lower()
        return any(marker in text for marker in [
            " in ", "city", "stadt", "gemeinde", "landkreis", "kreis",
            "germany", "deutschland",
        ])

    def _log_tool(self, label: str, payload: Dict[str, Any]):
        logger.debug("%s %s", label, json.dumps(payload, ensure_ascii=False))
