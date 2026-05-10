from brain.agents.german_admin.schemas import SolutionInput, SolutionOutput
from brain.llm import Llm
from brain.prompts import GERMAN_ADMIN_REASONING_SYSTEM, GERMAN_ADMIN_SOLUTION_PROMPT, json_text


class SolutionAgent:
    InputSchema = SolutionInput
    OutputSchema = SolutionOutput

    def __init__(self):
        self.llm = Llm(GERMAN_ADMIN_REASONING_SYSTEM, role="reasoning")

    def run(self, data: SolutionInput) -> SolutionOutput:
        prompt = GERMAN_ADMIN_SOLUTION_PROMPT.format(
            query=data["query"],
            intake_json=json_text(data.get("intake", {})),
            findings_json=json_text(data.get("findings", {})),
        )
        return {"draft_answer": self.llm.invoke(prompt)}
