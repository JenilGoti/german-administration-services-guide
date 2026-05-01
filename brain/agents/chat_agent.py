from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from brain.llm import LLM_V1
from brain.memory.short_term_graph_memory import Neo4jMemory
from brain.memory.conversation import Message


# -----------------------------
# STATE
# -----------------------------
class ChatState(TypedDict, total=False):
    query: str
    response: str


# -----------------------------
# CHAT AGENT
# -----------------------------
class ChatAgent:

    def __init__(self, user_id: str, conv_id: str):
        self.llm = LLM_V1("You are a helpful assistant.", memory=Neo4jMemory.get(user_id=user_id, conv_id=conv_id))
        self.app = self.build_graph()

    
    def chat_node(self, state: ChatState):

        response = self.llm.invoke(state["query"])

        return {
            **state,
            "response": response
        }

    def build_graph(self):

        builder = StateGraph(ChatState)
        builder.add_node("chat", self.chat_node)

        builder.add_edge(START, "chat")
        builder.add_edge("chat", END)

        return builder.compile()


    def chat(self, query: str):

        result = self.app.invoke({
            "query": query
        })

        return result["response"]