from brain.agents.german_admin.helpers import (
    as_list,
    clean_retrieval_term,
    compact_terms,
    fallback_knowledge_queries,
    fallback_knowledge_query,
    fallback_search_query,
    fallback_search_terms,
    is_german_language,
    parse_json_response,
)
from brain.agents.german_admin.schemas import RetrievalInput, RetrievalOutput
from brain.llm import Llm
from brain.prompts import (
    GERMAN_ADMIN_RETRIEVAL_PROMPT,
    GERMAN_ADMIN_RETRIEVAL_SYSTEM,
    json_text,
)


class RetrievalAgent:
    InputSchema = RetrievalInput
    OutputSchema = RetrievalOutput

    def __init__(self):
        self.llm = Llm(GERMAN_ADMIN_RETRIEVAL_SYSTEM, role="translation")

    def run(self, data: RetrievalInput) -> RetrievalOutput:
        query = data["query"]
        intake = data.get("intake", {})

        if is_german_language(data.get("target_language", "")):
            fallback_terms = fallback_search_terms(query, intake)
            german_query = fallback_search_query(query, intake)
            knowledge_queries = self._knowledge_queries({}, query, intake)
            return {
                "knowledge_query": knowledge_queries[0],
                "knowledge_queries": knowledge_queries,
                "german_query": german_query,
                "german_search_terms": compact_terms([german_query, *knowledge_queries, *fallback_terms]),
                "web_search_terms": compact_terms(fallback_terms),
            }

        prompt = GERMAN_ADMIN_RETRIEVAL_PROMPT.format(
            query=query,
            intake_json=json_text(intake),
        )
        retrieval = parse_json_response(self.llm.invoke(prompt), {
            "knowledge_query": fallback_knowledge_query(query, intake),
            "knowledge_queries": fallback_knowledge_queries(query, intake),
            "german_query": fallback_search_query(query, intake),
            "search_terms": fallback_search_terms(query, intake),
            "web_search_terms": [],
        })

        knowledge_queries = self._knowledge_queries(retrieval, query, intake)
        knowledge_query = knowledge_queries[0]
        german_query = retrieval.get("german_query") or fallback_search_query(query, intake)
        search_terms = retrieval.get("search_terms") or [german_query]
        web_search_terms = retrieval.get("web_search_terms") or search_terms
        fallback_terms = fallback_search_terms(query, intake)

        return {
            "knowledge_query": knowledge_query,
            "knowledge_queries": knowledge_queries,
            "german_query": clean_retrieval_term(german_query) or knowledge_query,
            "german_search_terms": compact_terms([german_query, *knowledge_queries, *search_terms, *fallback_terms]),
            "web_search_terms": compact_terms([*web_search_terms, *fallback_terms]),
        }

    def _knowledge_queries(self, retrieval: dict, query: str, intake: dict) -> list[str]:
        candidates = [
            *as_list(retrieval.get("knowledge_queries")),
            retrieval.get("knowledge_query", ""),
            *fallback_knowledge_queries(query, intake),
        ]
        cleaned = [clean_retrieval_term(term) for term in candidates]
        compacted = compact_terms(cleaned, limit=5)
        return compacted or [clean_retrieval_term(query)]
