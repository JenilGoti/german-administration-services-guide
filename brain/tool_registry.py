from typing import Any, Dict, List

from client.web_client import MCPWebClient
from langchain_core.tools import StructuredTool


class ToolRegistry:
    def __init__(self):
        self.web_client = MCPWebClient()
        self.tools = {
            "web_search": {
                "args": {"query": "str", "max_results": "int"},
                "description": "Search the web for official public-service pages.",
                "tool": self.web_search,
            },
            "scrape": {
                "args": {"url": "str"},
                "description": "Read one web page and return extracted content.",
                "tool": self.scrape,
            },
            "search_problem_knowledge": {
                "args": {"queries": "list[str]", "top_k": "int"},
                "description": "Search the German public-administration knowledge graph across situations, services, and precomputed service Q&A facts.",
                "tool": self.search_problem_knowledge,
            },
            "service_details": {
                "args": {"service_id": "str"},
                "description": "Fetch details for one service from the knowledge graph.",
                "tool": self.service_details,
            },
        }

    def web_search(self, query: str, max_results: int = 5):
        return self._call("web_search", {"query": query, "max_results": max_results})

    def scrape(self, url: str):
        return self._call("scrape", {"url": url})

    def search_problem_knowledge(self, queries: List[str], top_k: int = 3):
        return self._call("search_problem_knowledge", {"queries": queries, "top_k": top_k})

    def service_details(self, service_id: str):
        return self._call("service_details", {"service_id": service_id})

    def _call(self, tool_name: str, args: Dict[str, Any]):
        return self.web_client.call(tool_name, args)

    def get_tools(self, tool_names: List[str]):
        return {name: self.tools[name] for name in tool_names}

    def get_langchain_tools(self, tool_names: List[str]):
        selected_tools = self.get_tools(tool_names)
        langchain_tools = []
        for name, definition in selected_tools.items():
            langchain_tools.append(
                StructuredTool.from_function(
                    func=definition["tool"],
                    name=name,
                    description=definition.get("description") or f"Run {name}.",
                )
            )
        return langchain_tools
