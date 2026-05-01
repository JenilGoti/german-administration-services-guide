from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, START, END
from brain.llm import Llm
import json
from brain.agents.web.search import WebAgent
from brain.agents.web.scrapper import ScraperAgent
from utilities import safe_json_parse


class SupervisorState(TypedDict):
    query: str
    plan: Dict
    search_results: List[Dict]
    scraped_data: List[Dict]
    step: str
    iteration: int
    final_output: Dict


class SupervisorAgent:

    def __init__(self):
        self.llm = Llm.get_llm()
        self.app = self.build_graph()
        self.web_agent = WebAgent()
        self.scraper_agent = ScraperAgent()

    def plan_prompt(self, state: SupervisorState):

        return f"""
You are a Supervisor Agent.

You DO NOT execute tools.

You ONLY decide which agent should run next.

Agents available:
1. SEARCH_AGENT
2. SCRAPER_AGENT
3. FINISH

Rules:
- If no search results → SEARCH_AGENT
- If enough scraped data → FINISH

Return ONLY JSON:
{{
  "next": "SEARCH_AGENT | SCRAPER_AGENT | FINISH",
  "reason": "..."
}}

STATE:
Query: {state["query"]}
Iteration: {state.get("iteration", 0)}
Search Results: {state.get("search_results", [])}
Scraped Data: {state.get("scraped_data", [])}
"""

    def plan_node(self, state: SupervisorState):

        response = self.llm.invoke(self.plan_prompt(state))
        print(response)
        try:
            
            plan = safe_json_parse(response)
        except:
            plan = {"next": "FINISH", "reason": "error parsing"}
        print(plan)
        return {
            **state,
            "plan": plan
        }

    def route(self, state: SupervisorState):
        print(state)
        decision = state.get("plan", {}).get("next", "FINISH")
        print(state)
        if state["iteration"] >= 5:
            return "finish"

        if decision == "SEARCH_AGENT":
            return "search"

        if decision == "SCRAPER_AGENT":
            return "scrape"

        return "finish"
    def web_search_node(self, state):
        return {
            **state,
            "search_results": self.web_agent.invoke(state["query"]),
            "iteration": state["iteration"] + 1
        }

    def scraper_node(self, state):
        result = self.scraper_agent.invoke(state["search_results"][0]["url"])
        return {
            **state,
            "scraped_data": result,
            "iteration": state["iteration"] + 1
        }


    def final_node(self, state: SupervisorState):

        combined = {
            "query": state["query"],
            "search_results": state.get("search_results", []),
            "scraped_data": state.get("scraped_data", {})
        }

        result = self.llm.invoke(f"""
Create final structured JSON:

{json.dumps(combined)}
""")

        return {
            **state,
            "final_output": result
        }

    def build_graph(self):

        builder = StateGraph(SupervisorState)
        
        builder.add_node("plan", self.plan_node)
        builder.add_node("search", self.web_search_node)
        builder.add_node("scrape", self.scraper_node)
        builder.add_node("finish", self.final_node)

        builder.set_entry_point("plan")

        builder.add_conditional_edges(
            "plan",
            self.route,
            {
                "search": "search",
                "scrape": "scrape",
                "finish": "finish"
            }
        )
        builder.add_edge(START, "plan")
        builder.add_edge("search", "plan")
        builder.add_edge("scrape", "plan")
        builder.add_edge("finish", END)

        return builder.compile()