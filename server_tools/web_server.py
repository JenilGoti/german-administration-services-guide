from mcp.server.fastmcp import FastMCP
from tools.web_search import WebSearch
from tools.web_scraper import WebScraper

mcp = FastMCP("WebTools")

ws = WebSearch()
sc = WebScraper()

@mcp.tool(description="Search the web for relevant information based on a query")
def web_search(query: str):
    return ws.search(query)

@mcp.tool(description="Scrape and extract full content from a given URL")
def scrape(url: str):
    return sc.scrape(url)

if __name__ == "__main__":
    mcp.run(transport="stdio")