from langchain_core.messages import HumanMessage, SystemMessage

from brain.agents.german_admin.schemas import FollowupInput, FollowupOutput
from brain.llm import Llm, normalize_llm_response
from brain.prompts import GERMAN_ADMIN_FOLLOWUP_SYSTEM
from brain.tool_registry import ToolRegistry


class FollowupAgent:
    InputSchema = FollowupInput
    OutputSchema = FollowupOutput

    def __init__(self):
        self.llm = Llm(
            GERMAN_ADMIN_FOLLOWUP_SYSTEM,
            role="reasoning",
        ).runnable
        self.tools = ToolRegistry().get_langchain_tools(["web_search", "scrape"])
        self.agent = self._build_agent()

    def run(self, data: FollowupInput) -> FollowupOutput:
        user_message = f"""
{data.get("conversation_memory", "")}

Current user message:
{data["query"]}
"""
        result = self.agent.invoke({
            "messages": [
                SystemMessage(content=GERMAN_ADMIN_FOLLOWUP_SYSTEM),
                HumanMessage(content=user_message),
            ]
        })
        messages = result.get("messages", []) if isinstance(result, dict) else []
        response = normalize_llm_response(messages[-1]) if messages else normalize_llm_response(result)
        return {"direct_response": response}

    def _build_agent(self):
        try:
            from langgraph.prebuilt import create_react_agent
        except ImportError as exc:
            raise ImportError("FollowupAgent requires langgraph.prebuilt.create_react_agent.") from exc

        return create_react_agent(self.llm, self.tools)
