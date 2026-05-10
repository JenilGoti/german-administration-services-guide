from brain.agents.german_admin.schemas import DirectResponseInput, DirectResponseOutput
from brain.llm import Llm
from brain.prompts import (
    GERMAN_ADMIN_DIRECT_RESPONSE_PROMPT,
    GERMAN_ADMIN_DIRECT_RESPONSE_SYSTEM,
    json_text,
)


class DirectResponseAgent:
    InputSchema = DirectResponseInput
    OutputSchema = DirectResponseOutput

    def __init__(self):
        self.llm = Llm(GERMAN_ADMIN_DIRECT_RESPONSE_SYSTEM, role="structured")

    def run(self, data: DirectResponseInput) -> DirectResponseOutput:
        prompt = GERMAN_ADMIN_DIRECT_RESPONSE_PROMPT.format(
            route=data.get("route", "out_of_scope"),
            query=data["query"],
            intake_json=json_text(data.get("intake", {})),
        )
        return {"direct_response": self.llm.invoke(prompt).strip()}
