from brain.agents.german_admin.schemas import RevisionInput, RevisionOutput
from brain.llm import Llm
from config import QUALITY_LLM_PROVIDER
from brain.prompts import GERMAN_ADMIN_REASONING_SYSTEM, GERMAN_ADMIN_REVISION_PROMPT, json_text


class RevisionAgent:
    InputSchema = RevisionInput
    OutputSchema = RevisionOutput

    def __init__(self):
        self.llm = Llm(
            GERMAN_ADMIN_REASONING_SYSTEM,
            role="reasoning",
            provider=QUALITY_LLM_PROVIDER,
        )

    def run(self, data: RevisionInput) -> RevisionOutput:
        supervisor = data.get("supervisor", {})
        changes = supervisor.get("required_changes", [])
        if supervisor.get("approved") and not changes:
            return {"draft_answer": data.get("draft_answer", "")}

        prompt = GERMAN_ADMIN_REVISION_PROMPT.format(
            query=data["query"],
            findings_json=json_text(data.get("findings", {})),
            draft_answer=data.get("draft_answer", ""),
            supervisor_json=json_text(supervisor),
        )
        return {"draft_answer": self.llm.invoke(prompt)}
