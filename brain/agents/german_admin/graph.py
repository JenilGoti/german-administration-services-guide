from langgraph.graph import END, START, StateGraph

from brain.agents.german_admin.clarification_agent import ClarificationAgent
from brain.agents.german_admin.direct_response_agent import DirectResponseAgent
from brain.agents.german_admin.final_response_agent import FinalResponseAgent
from brain.agents.german_admin.followup_agent import FollowupAgent
from brain.agents.german_admin.helpers import clean_assistant_response
from brain.agents.german_admin.intake_agent import IntakeAgent
from brain.agents.german_admin.knowledge_agent import KnowledgeAgent
from brain.agents.german_admin.memory_recall_agent import MemoryRecallAgent
from brain.agents.german_admin.planner_agent import PlannerAgent
from brain.agents.german_admin.retrieval_agent import RetrievalAgent
from brain.agents.german_admin.revision_agent import RevisionAgent
from brain.agents.german_admin.schemas import GermanAdminState
from brain.agents.german_admin.solution_agent import SolutionAgent
from brain.agents.german_admin.supervisor_agent import SupervisorAgent
from brain.checkpoint import CheckpointerFactory
from brain.memory.factory import create_conversation_memory


class GermanAdminGuideAgent:
    def __init__(self, user_id: str, conv_id: str, progress_callback=None):
        self.user_id = user_id
        self.conv_id = conv_id
        self.thread_id = f"german-admin-v2:{self.user_id}:{self.conv_id}"
        self.progress_callback = progress_callback
        self.current_agent = ""

        self.intake_agent = IntakeAgent()
        self.clarification_agent = ClarificationAgent()
        self.direct_response_agent = DirectResponseAgent()
        self.followup_agent = FollowupAgent()
        self.memory_recall_agent = MemoryRecallAgent()
        self.retrieval_agent = RetrievalAgent()
        self.planner_agent = PlannerAgent()
        self.knowledge_agent = KnowledgeAgent()
        self.solution_agent = SolutionAgent()
        self.supervisor_agent = SupervisorAgent()
        self.revision_agent = RevisionAgent()
        self.final_response_agent = FinalResponseAgent()
        self.memory = create_conversation_memory(user_id=user_id, conv_id=conv_id)

        self.checkpointer_factory = CheckpointerFactory()
        self.checkpointer = self.checkpointer_factory.create()
        self.last_state = {}
        self.app = self.build_graph()

    def _progress(self, agent_name: str):
        self.current_agent = agent_name
        if self.progress_callback:
            self.progress_callback(agent_name)

    def intake_node(self, state: GermanAdminState) -> GermanAdminState:
        self._progress("Intake agent")
        output = self.intake_agent.run({
            "query": state["query"],
            "conversation_memory": self.memory.prompt(state["query"]),
        })
        return {**state, **output}

    def intake_route(self, state: GermanAdminState) -> str:
        if state.get("route") == "admin":
            return "retrieval"
        if state.get("route") == "clarify":
            return "clarification"
        if state.get("route") == "memory_recall":
            return "memory_recall"
        if state.get("route") == "followup":
            return "followup"
        return "direct_response"

    def clarification_node(self, state: GermanAdminState) -> GermanAdminState:
        self._progress("Clarification agent")
        output = self.clarification_agent.run({
            "query": state["query"],
            "intake": state.get("intake", {}),
            "target_language": state.get("target_language", "English"),
        })
        return {**state, **output}

    def memory_recall_node(self, state: GermanAdminState) -> GermanAdminState:
        self._progress("Memory recall agent")
        output = self.memory_recall_agent.run({
            "query": state["query"],
            "conversation_memory": self.memory.transcript(),
        })
        return {**state, **output}

    def direct_response_node(self, state: GermanAdminState) -> GermanAdminState:
        self._progress("Direct response agent")
        output = self.direct_response_agent.run({
            "query": state["query"],
            "route": state.get("route", "out_of_scope"),
            "intake": state.get("intake", {}),
        })
        return {**state, **output}

    def followup_node(self, state: GermanAdminState) -> GermanAdminState:
        self._progress("Follow-up agent")
        output = self.followup_agent.run({
            "query": state["query"],
            "conversation_memory": self.memory.transcript(),
        })
        return {**state, **output}

    def retrieval_node(self, state: GermanAdminState) -> GermanAdminState:
        self._progress("Retrieval agent")
        output = self.retrieval_agent.run({
            "query": state["query"],
            "intake": state.get("intake", {}),
            "target_language": state.get("target_language", "English"),
        })
        return {**state, **output}

    def planner_node(self, state: GermanAdminState) -> GermanAdminState:
        self._progress("Planner agent")
        output = self.planner_agent.run({
            "knowledge_query": state.get("knowledge_query", ""),
            "intake": state.get("intake", {}),
        })
        return {**state, **output}

    def knowledge_node(self, state: GermanAdminState) -> GermanAdminState:
        self._progress("Knowledge agent")
        output = self.knowledge_agent.run({
            "query": state["query"],
            "knowledge_query": state.get("knowledge_query", ""),
            "knowledge_queries": state.get("knowledge_queries", []),
            "german_query": state.get("german_query", ""),
            "german_search_terms": state.get("german_search_terms", []),
            "web_search_terms": state.get("web_search_terms", []),
            "supervisor": state.get("supervisor", {}),
        })
        return {**state, **output}

    def solution_node(self, state: GermanAdminState) -> GermanAdminState:
        self._progress("Solution agent")
        output = self.solution_agent.run({
            "query": state["query"],
            "intake": state.get("intake", {}),
            "findings": state.get("findings", {}),
        })
        return {**state, **output}

    def supervisor_node(self, state: GermanAdminState) -> GermanAdminState:
        self._progress("Supervisor agent")
        output = self.supervisor_agent.run({
            "query": state["query"],
            "plan": state.get("plan", {}),
            "findings": state.get("findings", {}),
            "draft_answer": state.get("draft_answer", ""),
        })
        return {
            **state,
            **output,
            "supervisor_rounds": state.get("supervisor_rounds", 0) + 1,
        }

    def supervisor_route(self, state: GermanAdminState) -> str:
        supervisor = state.get("supervisor", {})
        if supervisor.get("needs_more_search") and state.get("supervisor_rounds", 0) < 2:
            return "knowledge"
        return "revision"

    def revision_node(self, state: GermanAdminState) -> GermanAdminState:
        self._progress("Revision agent")
        output = self.revision_agent.run({
            "query": state["query"],
            "findings": state.get("findings", {}),
            "draft_answer": state.get("draft_answer", ""),
            "supervisor": state.get("supervisor", {}),
        })
        return {**state, **output}

    def final_response_node(self, state: GermanAdminState) -> GermanAdminState:
        self._progress("Final response agent")
        if state.get("route") != "admin" and state.get("direct_response"):
            output = {"response": state["direct_response"]}
        else:
            output = self.final_response_agent.run({
                "draft_answer": state.get("draft_answer", ""),
                "target_language": state.get("target_language", "English"),
            })

        output["response"] = clean_assistant_response(output.get("response", ""))
        self.memory.save_turn(state.get("query", ""), output["response"])
        return {**state, **output}

    def build_graph(self):
        builder = StateGraph(GermanAdminState)
        builder.add_node("intake", self.intake_node)  # Understand user problem and language.
        builder.add_node("clarification", self.clarification_node)  # Ask for missing information.
        builder.add_node("direct_response", self.direct_response_node)  # Reply without retrieval.
        builder.add_node("followup", self.followup_node)  # ReAct answer for contextual follow-ups.
        builder.add_node("memory_recall", self.memory_recall_node)  # Answer from saved conversation.
        builder.add_node("retrieval", self.retrieval_node)  # Build German KB/web search queries.
        builder.add_node("planner", self.planner_node)  # Decide how to use retrieved evidence.
        builder.add_node("knowledge", self.knowledge_node)  # Call MCP tools and collect findings.
        builder.add_node("solution", self.solution_node)  # Draft the administrative answer.
        builder.add_node("supervisor", self.supervisor_node)  # Check grounding and completeness.
        builder.add_node("revision", self.revision_node)  # Apply supervisor corrections if needed.
        builder.add_node("final_response", self.final_response_node)  # Return answer in user's language.

        builder.add_edge(START, "intake")
        builder.add_conditional_edges(
            "intake",
            self.intake_route,
            {
                "retrieval": "retrieval",
                "clarification": "clarification",
                "memory_recall": "memory_recall",
                "followup": "followup",
                "direct_response": "direct_response",
            },
        )
        builder.add_edge("clarification", "final_response")
        builder.add_edge("direct_response", "final_response")
        builder.add_edge("followup", "final_response")
        builder.add_edge("memory_recall", "final_response")
        builder.add_edge("retrieval", "planner")
        builder.add_edge("planner", "knowledge")
        builder.add_edge("knowledge", "solution")
        builder.add_edge("solution", "supervisor")
        builder.add_conditional_edges(
            "supervisor",
            self.supervisor_route,
            {
                "knowledge": "knowledge",
                "revision": "revision",
            },
        )
        builder.add_edge("revision", "final_response")
        builder.add_edge("final_response", END)

        return builder.compile(checkpointer=self.checkpointer)

    def chat(self, query: str) -> str:
        result = self.app.invoke({
            "query": query,
            "supervisor_rounds": 0,
        }, config={
            "configurable": {"thread_id": self.thread_id},
        })
        self.last_state = result
        return result["response"]

    def get_last_state(self) -> GermanAdminState:
        return self.last_state

    def close(self):
        self.checkpointer_factory.close()
