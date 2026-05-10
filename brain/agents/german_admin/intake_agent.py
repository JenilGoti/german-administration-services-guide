from brain.agents.german_admin.helpers import parse_json_response, target_language
from brain.agents.german_admin.schemas import IntakeInput, IntakeOutput
from brain.llm import Llm
from brain.prompts import GERMAN_ADMIN_INTAKE_PROMPT, GERMAN_ADMIN_STRUCTURED_SYSTEM
from brain.logger import logger

class IntakeAgent:
    InputSchema = IntakeInput
    OutputSchema = IntakeOutput

    def __init__(self):
        self.llm = Llm(GERMAN_ADMIN_STRUCTURED_SYSTEM, role="structured")

    def run(self, data: IntakeInput) -> IntakeOutput:
        query = data["query"]
        prompt = GERMAN_ADMIN_INTAKE_PROMPT.format(
            conversation_memory=data.get("conversation_memory", ""),
            query=query,
        )
        print("Prompt:", prompt)
        fallback = {
            "problem_type": "unknown",
            "detected_language": "unknown",
            "route": "clarify",
            "user_goal": query,
            "known_facts": [],
            "missing_information": ["unclear situation"],
            "urgency": "medium",
            "search_terms": [],
        }
        intake = parse_json_response(self.llm.invoke(prompt), fallback)
        route = self._normalize_route(intake.get("route"))
        if route == "admin" and self._needs_clarification(intake):
            route = "clarify"
        logger.info(f"Intake agent output: {intake}")
        return {
            "intake": intake,
            "target_language": target_language(query, intake.get("detected_language")),
            "route": route,
        }

    def _normalize_route(self, route: str) -> str:
        route = (route or "").lower().strip()
        if route in {"admin", "clarify", "small_talk", "memory_recall", "followup", "out_of_scope"}:
            return route
        return "clarify"

    def _needs_clarification(self, intake: dict) -> bool:
        missing = intake.get("missing_information", [])
        search_terms = intake.get("search_terms", [])
        if not intake.get("user_goal"):
            return True
        if not search_terms:
            return True
        if isinstance(missing, list) and len(missing) >= 3 and not intake.get("known_facts"):
            return True
        return False
