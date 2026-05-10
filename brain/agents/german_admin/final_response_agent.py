from brain.agents.german_admin.helpers import clean_assistant_response, is_german_language
from brain.agents.german_admin.schemas import FinalResponseInput, FinalResponseOutput
from brain.llm import Llm
from brain.prompts import GERMAN_ADMIN_LANGUAGE_GUARD_PROMPT, GERMAN_ADMIN_RETRIEVAL_SYSTEM


class FinalResponseAgent:
    InputSchema = FinalResponseInput
    OutputSchema = FinalResponseOutput

    def __init__(self):
        self.llm = Llm(GERMAN_ADMIN_RETRIEVAL_SYSTEM, role="translation")

    def run(self, data: FinalResponseInput) -> FinalResponseOutput:
        response = data.get("draft_answer", "")
        target_language = data.get("target_language") or "English"
        if is_german_language(target_language):
            return {"response": clean_assistant_response(response)}

        prompt = GERMAN_ADMIN_LANGUAGE_GUARD_PROMPT.format(
            target_language=target_language,
            draft_answer=response,
        )
        return {"response": clean_assistant_response(self.llm.invoke(prompt))}
