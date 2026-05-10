import asyncio
import json
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from brain.logger import logger, mcp_logger
from utilities import safe_json_parse


class MCPWebClient:

    def __init__(self):
        env = os.environ.copy()
        env.setdefault("PYTHONWARNINGS", "ignore::UserWarning")
        env.setdefault("LOG_LEVEL", os.getenv("LOG_LEVEL", "INFO"))
        env.setdefault("RUST_LOG", "error")
        env.setdefault("MallocStackLogging", "0")
        env.setdefault("MallocStackLoggingNoCompact", "0")
        self.server = StdioServerParameters(
            command=sys.executable,
            args=["-W", "ignore::UserWarning", "-m", "server_tools"],
            env=env,
        )

    async def _call(self, tool_name: str, payload: dict):

        async with stdio_client(self.server) as (r, w):

            async with ClientSession(r, w) as session:

                await session.initialize()

                result = await session.call_tool(tool_name, payload)
                return self._unwrap_result(tool_name, result)

    def call(self, tool_name: str, payload: dict):

        loop = None
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        mcp_logger.info("CALL tool=%s input=%s", tool_name, self._json(payload))
        result = loop.run_until_complete(self._call(tool_name, payload))
        mcp_logger.info("RESULT tool=%s output=%s", tool_name, self._json(result))
        logger.debug("mcp.call.end tool=%s summary=%s", tool_name, self._summary(result))
        return result

    def _unwrap_result(self, tool_name: str, result):
        if getattr(result, "isError", False):
            raise RuntimeError(f"MCP tool {tool_name} failed: {result}")

        content = getattr(result, "content", result)
        if not isinstance(content, list):
            return content

        values = []
        for item in content:
            text = getattr(item, "text", None)
            if text is not None:
                parsed = safe_json_parse(text)
                values.append(parsed if parsed is not None else text)
                continue

            data = getattr(item, "data", None)
            if data is not None:
                values.append(data)
                continue

            values.append(item)

        if len(values) == 1:
            return values[0]
        return values

    def _summary(self, result):
        if isinstance(result, dict):
            return {
                key: len(value) if isinstance(value, list) else type(value).__name__
                for key, value in result.items()
            }
        if isinstance(result, list):
            return f"list[{len(result)}]"
        try:
            return json.dumps(result)[:300]
        except TypeError:
            return type(result).__name__

    def _json(self, value):
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except TypeError:
            text = str(value)

        max_chars = int(os.getenv("MCP_LOG_MAX_CHARS", "12000"))
        if len(text) > max_chars:
            return text[:max_chars] + f"... <truncated {len(text) - max_chars} chars>"
        return text

    # def search(self, query: str):
    #     return self.call("web_search", {"query": query})

    # def scrape(self, url: str):
    #     return self.call("scrape", {"url": url})
