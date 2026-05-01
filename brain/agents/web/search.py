from client.web_client import MCPWebClient
from typing import TypedDict, List, Dict, Any, NotRequired, Annotated
from langgraph.graph import StateGraph, END
from utilities import safe_json_parse
from brain.llm import LLM_V1
from brain.tool_registry import ToolRegistry
from langgraph.types import Send
import json
import operator

class WebSearchState(TypedDict, total=False):
    query: str
    results: List[Dict]
    ranked: List[Dict]
    findings: Annotated[list[Dict], operator.add]
    final_summary: NotRequired[str]
    error: str

class WebSearchOutput(TypedDict):
    findings: List[Dict]
    final_summary: str

class WebAgent:
    def __init__(self):
        self.web_search = MCPWebClient()
        self.web_scrape = ToolRegistry().tools["scrape"]["tool"]
        self.llm_with_tools = LLM_V1("You are a web search agent. Task: Search the web for the query.",tools=ToolRegistry().get_tools(["web_search"]))
        self.llm = LLM_V1("You are a web ranking system. Task: Rank results by relevance to the query.")
        self.summarizer = LLM_V1("You are a summarizer. Task: Summarize the content to figure out the answer to the query.")

        self.app = self.build_graph()

    def search_node(self, state: WebSearchState):
        results = self.llm_with_tools.invoke_with_tools(state["query"])
        return {"results": results}

    def rank_node(self, state: WebSearchState):
        raw = self.llm.invoke_with_formated_response(
            state["results"],
            formate={"url": "string", "score": "float"}
        )

        return {
            "ranked": raw
        }
    
    def fanout(self, state: WebSearchState):
        urls = [url.get("url") for url in state.get("ranked", [])]
        return [
            Send("scrape_and_summarize", {"url": url, "query": state["query"]})
            for url in urls
        ]
    
    def scrape_and_summarize_node(self, state: WebSearchState):
        url = state["url"]
        result = self.web_scrape(url)
        content = result.content if hasattr(result, "content") else result

        data = [c.text if hasattr(c, "text") else str(c) for c in content]
        summary = self.summarizer.invoke(f"""
        Query: {state['query']}
        Content: {json.dumps(data)}
        """)

        return {"findings": [{"url": url, "summary": summary, "content": content}]}
    
    def combine_node(self, state: WebSearchState):
        findings = state.get("findings", [])

        summaries = [
            f"URL: {f['url']}\nSummary: {f['summary']}"
            for f in findings
        ]

        final_summary = self.summarizer.invoke(f"""
        Combine the following summaries into a final answer:

        Query: {state['query']}

        Summaries:
        {json.dumps(summaries)}
        """)

        return {
            "findings": findings,   # keep raw findings
            "final_summary": final_summary
        }

    def build_graph(self):

        builder = StateGraph(WebSearchState, output_schema=WebSearchOutput)

        builder.add_node("search", self.search_node)
        builder.add_node("rank", self.rank_node)
        builder.add_node("scrape_and_summarize", self.scrape_and_summarize_node)
        builder.add_node("combine", self.combine_node)

        builder.set_entry_point("search")
        builder.add_edge("search","rank")
        builder.add_conditional_edges(
            "rank",
            self.fanout,
            ["scrape_and_summarize"]
        )
        builder.add_edge("scrape_and_summarize", "combine")
        builder.add_edge("combine", END)

        return builder.compile()