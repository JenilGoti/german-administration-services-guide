from client.web_client import MCPWebClient

class ToolRegistry:
    def __init__(self):
        self.web_client = MCPWebClient()
        self.tools = {
            "web_search": {"args": {"query": "str"}, "tool": lambda query: self.web_client.call("web_search", {"query": query})},
            "scrape": {"args": {"url": "str"}, "tool": lambda url: self.web_client.call("scrape", {"url": url})}
            }

    def get_tools(self, tool_names: List[str]):
        return {name: self.tools[name] for name in tool_names}

    