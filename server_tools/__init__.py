import logging
from typing import Any, Dict, List

from mcp.server.fastmcp import FastMCP

from brain.logger import mcp_logger
from server_tools.tools.graph_tools import GermanAdminGraphTools
from server_tools.tools.web_scraper import WebScraper
from server_tools.tools.web_search import WebSearch


logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

mcp = FastMCP("AgentTools")

graph_tools = GermanAdminGraphTools()
web_search_tool = WebSearch()
web_scraper_tool = WebScraper()


@mcp.tool(description="Search the web for relevant information based on a query")
def web_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    mcp_logger.info("SERVER tool=web_search query=%s max_results=%s", query, max_results)
    return web_search_tool.search(query=query, max_results=max_results)


@mcp.tool(description="Scrape and extract readable page content from a given URL")
def scrape(url: str) -> Dict[str, Any]:
    mcp_logger.info("SERVER tool=scrape url=%s", url)
    return web_scraper_tool.scrape(url)


@mcp.tool(description="Search German administration knowledge base using user situation queries, service vectors, and precomputed ServiceQA facts")
def search_problem_knowledge(queries: List[str], top_k: int = 3) -> Dict[str, Any]:
    mcp_logger.info("SERVER tool=search_problem_knowledge queries=%s top_k=%s", queries, top_k)
    result = graph_tools.search_problem_knowledge(queries=queries, top_k=top_k)
    mcp_logger.info(
        "SERVER tool=search_problem_knowledge result_counts sub_situations=%s services=%s service_qa=%s service_details=%s",
        len(result.get("SubSituation", [])),
        len(result.get("Service", [])),
        len(result.get("ServiceQA", [])),
        len(result.get("services", [])),
    )
    return result


@mcp.tool(description="Get detailed information for one German administration service by service id")
def service_details(service_id: str) -> Dict[str, Any]:
    mcp_logger.info("SERVER tool=service_details service_id=%s", service_id)
    return graph_tools.get_service_details(service_id)


def run(transport: str = "stdio"):
    mcp.run(transport=transport)
