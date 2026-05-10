from brain.agents.german_admin.schemas import ClarificationInput, ClarificationOutput
from brain.llm import Llm
from brain.prompts import (
    GERMAN_ADMIN_CLARIFICATION_PROMPT,
    GERMAN_ADMIN_CLARIFICATION_SYSTEM,
    json_text,
)


class ClarificationAgent:
    InputSchema = ClarificationInput
    OutputSchema = ClarificationOutput

    def __init__(self):
        self.llm = Llm(GERMAN_ADMIN_CLARIFICATION_SYSTEM, role="structured")

    def run(self, data: ClarificationInput) -> ClarificationOutput:
        prompt = GERMAN_ADMIN_CLARIFICATION_PROMPT.format(
            query=data["query"],
            intake_json=json_text(data.get("intake", {})),
        )
        response = self.llm.invoke(prompt).strip()
        return {"direct_response": response}
