import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPWebClient:

    def __init__(self):
        self.server = StdioServerParameters(
            command="python",
            args=["server_tools/web_server.py"]
        )

    async def _call(self, tool_name: str, payload: dict):

        async with stdio_client(self.server) as (r, w):

            async with ClientSession(r, w) as session:

                await session.initialize()

                return await session.call_tool(tool_name, payload)

    def call(self, tool_name: str, payload: dict):

        loop = None
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self._call(tool_name, payload))

    # def search(self, query: str):
    #     return self.call("web_search", {"query": query})

    # def scrape(self, url: str):
    #     return self.call("scrape", {"url": url})

