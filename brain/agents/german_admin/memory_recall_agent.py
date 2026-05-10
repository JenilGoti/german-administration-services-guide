from brain.agents.german_admin.schemas import MemoryRecallInput, MemoryRecallOutput
from brain.llm import Llm
from brain.prompts import (
    GERMAN_ADMIN_MEMORY_RECALL_PROMPT,
    GERMAN_ADMIN_MEMORY_RECALL_SYSTEM,
)


class MemoryRecallAgent:
    InputSchema = MemoryRecallInput
    OutputSchema = MemoryRecallOutput

    def __init__(self):
        self.llm = Llm(GERMAN_ADMIN_MEMORY_RECALL_SYSTEM, role="structured")

    def run(self, data: MemoryRecallInput) -> MemoryRecallOutput:
        prompt = GERMAN_ADMIN_MEMORY_RECALL_PROMPT.format(
            query=data["query"],
            conversation_memory=data.get("conversation_memory", ""),
        )
        return {"direct_response": self.llm.invoke(prompt).strip()}
