from typing import Any, Dict, Iterable, List

from utilities import safe_json_parse


def as_list(value) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def compact_terms(terms: Iterable[Any], limit: int = 8) -> List[str]:
    compacted = []
    for term in terms or []:
        if not isinstance(term, str):
            continue
        cleaned = " ".join(term.replace("\n", " ").split()).strip()
        if cleaned and cleaned not in compacted:
            compacted.append(cleaned)
    return compacted[:limit]


def parse_json_response(raw: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    parsed = safe_json_parse(raw)
    if isinstance(parsed, dict):
        return parsed
    return {**fallback, "raw_output": raw}


def is_german_language(language: str) -> bool:
    return (language or "").lower() in {"german", "deutsch", "de"}


def target_language(query: str, detected_language: str = None) -> str:
    text = (query or "").lower()
    english_markers = {
        "hi", "hello", "recently", "moved", "need", "guidance", "which",
        "what", "documents", "process", "office", "contact", "from", "to",
    }
    german_markers = {
        "hallo", "ich", "bin", "brauche", "welche", "unterlagen", "verfahren",
        "zuständig", "anmeldung", "wohnsitz", "gezogen",
    }

    english_score = sum(1 for marker in english_markers if marker in text)
    german_score = sum(1 for marker in german_markers if marker in text)

    if english_score >= german_score:
        return "English"
    if german_score > english_score:
        return "German"

    if is_german_language(detected_language or ""):
        return "German"
    return "English"


def clean_retrieval_term(term: str) -> str:
    if not isinstance(term, str):
        return ""

    cleaned = " ".join(term.replace("\n", " ").split()).strip()
    cleaned = cleaned.removeprefix("Administrative Situation des Nutzers:").strip()
    if not cleaned:
        return ""

    words = cleaned.split()
    if len(words) <= 12 and len(cleaned) <= 120:
        return cleaned.rstrip(".")

    first_sentence = cleaned.split(".")[0].strip()
    if first_sentence and len(first_sentence.split()) <= 12:
        return first_sentence

    return " ".join(words[:12]).rstrip(".,;:")


def fallback_knowledge_query(query: str, intake: Dict[str, Any]) -> str:
    known_facts = intake.get("known_facts", []) if isinstance(intake, dict) else []
    user_goal = intake.get("user_goal", "") if isinstance(intake, dict) else ""
    search_terms = intake.get("search_terms", []) if isinstance(intake, dict) else []

    for value in [*as_list(search_terms), intake.get("problem_type", ""), user_goal, *as_list(known_facts)]:
        cleaned = clean_retrieval_term(value)
        if cleaned:
            return cleaned

    return clean_retrieval_term(query) or query


def fallback_knowledge_queries(query: str, intake: Dict[str, Any]) -> List[str]:
    candidates = [
        *(as_list(intake.get("search_terms")) if isinstance(intake, dict) else []),
        intake.get("problem_type", "") if isinstance(intake, dict) else "",
        intake.get("user_goal", "") if isinstance(intake, dict) else "",
        *(as_list(intake.get("known_facts")) if isinstance(intake, dict) else []),
        fallback_knowledge_query(query, intake),
    ]
    return compact_terms((clean_retrieval_term(term) for term in candidates), limit=5)


def fallback_search_query(query: str, intake: Dict[str, Any] = None) -> str:
    return fallback_knowledge_query(query, intake or {}).split(".")[0]


def fallback_search_terms(query: str, intake: Dict[str, Any]) -> List[str]:
    terms = []
    for value in [
        intake.get("problem_type", ""),
        intake.get("user_goal", ""),
        *as_list(intake.get("search_terms", [])),
        *as_list(intake.get("known_facts", [])),
        query,
    ]:
        if isinstance(value, str) and value.strip():
            terms.append(value.strip())
    return compact_terms(terms)
