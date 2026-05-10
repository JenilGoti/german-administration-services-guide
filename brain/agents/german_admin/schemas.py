from typing import Any, Dict, List, TypedDict


class IntakeInput(TypedDict):
    query: str
    conversation_memory: str


class IntakeOutput(TypedDict):
    intake: Dict[str, Any]
    target_language: str
    route: str


class DirectResponseInput(TypedDict):
    query: str
    route: str
    intake: Dict[str, Any]


class DirectResponseOutput(TypedDict):
    direct_response: str


class RetrievalInput(TypedDict):
    query: str
    intake: Dict[str, Any]
    target_language: str


class ClarificationInput(TypedDict):
    query: str
    intake: Dict[str, Any]
    target_language: str


class ClarificationOutput(TypedDict):
    direct_response: str


class MemoryRecallInput(TypedDict):
    query: str
    conversation_memory: str


class MemoryRecallOutput(TypedDict):
    direct_response: str


class FollowupInput(TypedDict):
    query: str
    conversation_memory: str


class FollowupOutput(TypedDict):
    direct_response: str


class RetrievalOutput(TypedDict):
    knowledge_query: str
    knowledge_queries: List[str]
    german_query: str
    german_search_terms: List[str]
    web_search_terms: List[str]


class PlannerInput(TypedDict):
    knowledge_query: str
    intake: Dict[str, Any]


class PlannerOutput(TypedDict):
    plan: Dict[str, Any]


class KnowledgeInput(TypedDict):
    query: str
    knowledge_query: str
    knowledge_queries: List[str]
    german_query: str
    german_search_terms: List[str]
    web_search_terms: List[str]
    supervisor: Dict[str, Any]


class KnowledgeOutput(TypedDict):
    findings: Dict[str, Any]


class SolutionInput(TypedDict):
    query: str
    intake: Dict[str, Any]
    findings: Dict[str, Any]


class SolutionOutput(TypedDict):
    draft_answer: str


class SupervisorInput(TypedDict):
    query: str
    plan: Dict[str, Any]
    findings: Dict[str, Any]
    draft_answer: str


class SupervisorOutput(TypedDict):
    supervisor: Dict[str, Any]


class RevisionInput(TypedDict):
    query: str
    findings: Dict[str, Any]
    draft_answer: str
    supervisor: Dict[str, Any]


class RevisionOutput(TypedDict):
    draft_answer: str


class FinalResponseInput(TypedDict):
    draft_answer: str
    target_language: str


class FinalResponseOutput(TypedDict):
    response: str


class GermanAdminState(TypedDict, total=False):
    query: str
    response: str
    route: str
    direct_response: str
    target_language: str
    intake: Dict[str, Any]
    clarification: Dict[str, Any]
    knowledge_query: str
    knowledge_queries: List[str]
    german_query: str
    german_search_terms: List[str]
    web_search_terms: List[str]
    plan: Dict[str, Any]
    findings: Dict[str, Any]
    draft_answer: str
    supervisor: Dict[str, Any]
    supervisor_rounds: int
