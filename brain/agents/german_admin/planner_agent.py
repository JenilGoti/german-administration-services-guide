from brain.agents.german_admin.helpers import fallback_knowledge_query, parse_json_response
from brain.agents.german_admin.schemas import PlannerInput, PlannerOutput
from brain.llm import Llm
from brain.prompts import (
    GERMAN_ADMIN_PLANNER_PROMPT,
    GERMAN_ADMIN_STRUCTURED_SYSTEM,
    json_text,
)


class PlannerAgent:
    InputSchema = PlannerInput
    OutputSchema = PlannerOutput

    def __init__(self):
        self.llm = Llm(GERMAN_ADMIN_STRUCTURED_SYSTEM, role="structured")

    def run(self, data: PlannerInput) -> PlannerOutput:
        knowledge_query = data.get("knowledge_query") or fallback_knowledge_query("", data.get("intake", {}))
        prompt = GERMAN_ADMIN_PLANNER_PROMPT.format(
            knowledge_query=knowledge_query,
            intake_json=json_text(data.get("intake", {})),
        )
        plan = parse_json_response(self.llm.invoke(prompt), {
            "strategy": "Search SubSituation nodes and expand matching services with details.",
            "needs_service_details": True,
            "needs_clarification": False,
            "clarifying_questions": [],
        })
        return {"plan": plan}
