from brain.agents.german_admin.helpers import parse_json_response
from brain.agents.german_admin.schemas import SupervisorInput, SupervisorOutput
from brain.llm import Llm
from config import QUALITY_LLM_PROVIDER
from brain.prompts import (
    GERMAN_ADMIN_SUPERVISOR_PROMPT,
    GERMAN_ADMIN_SUPERVISOR_SYSTEM,
    json_text,
)


class SupervisorAgent:
    InputSchema = SupervisorInput
    OutputSchema = SupervisorOutput

    def __init__(self):
        self.llm = Llm(
            GERMAN_ADMIN_SUPERVISOR_SYSTEM,
            role="supervisor",
            provider=QUALITY_LLM_PROVIDER,
        )

    def run(self, data: SupervisorInput) -> SupervisorOutput:
        prompt = GERMAN_ADMIN_SUPERVISOR_PROMPT.format(
            query=data["query"],
            plan_json=json_text(data.get("plan", {})),
            findings_json=json_text(data.get("findings", {})),
            draft_answer=data.get("draft_answer", ""),
        )
        review = parse_json_response(self.llm.invoke(prompt), {
            "approved": True,
            "reason": "Fallback approval because supervisor output was not valid JSON.",
            "needs_more_search": False,
            "extra_search_terms": [],
            "required_changes": [],
        })
        return {"supervisor": review}
