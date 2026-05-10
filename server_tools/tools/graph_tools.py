from typing import Any, Dict, List, Optional

from config import KNOWLEDGE_DB
from graph_db import DatabaseManager, GraphMemoryIngestor


SEARCHABLE_NODE_LABELS = [
    "Situation",
    "SubSituation",
    "Service",
    "ServiceSection",
    "Requirement",
    "Document",
    "Authority",
    "Form",
    "ProcessStep",
    "LegalBasis",
    "Goal",
    "DependencyProblem",
]

DEFAULT_ADMIN_SEARCH_LABELS = ["SubSituation"]


class GermanAdminGraphTools:
    def __init__(self, db_name: str = KNOWLEDGE_DB):
        self.db_name = db_name
        self.db = DatabaseManager()
        self.ingestor = GraphMemoryIngestor(db_name)

    def _session(self):
        return self.db.get_session(self.db_name)

    def vector_search(self, query: str, node_name: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if node_name not in SEARCHABLE_NODE_LABELS:
            raise ValueError(f"Unsupported node label: {node_name}")

        return self.ingestor.vector_cypher_search(
            query=query,
            node_name=node_name,
            top_k=top_k
        )

    def search_admin_knowledge(
        self,
        query: str,
        node_names: Optional[List[str]] = None,
        top_k: int = 3
    ) -> Dict[str, List[Dict[str, Any]]]:
        labels = node_names or DEFAULT_ADMIN_SEARCH_LABELS

        results = {}
        for label in labels:
            try:
                results[label] = self.vector_search(query, label, top_k=top_k)
            except Exception as exc:
                results[label] = [{"error": str(exc)}]

        return results

    def find_services_for_sub_situations(
        self,
        sub_situation_ids: List[str],
        limit: int = 8
    ) -> List[Dict[str, Any]]:
        if not sub_situation_ids:
            return []

        with self._session() as session:
            result = session.run("""
            UNWIND $sub_situation_ids AS sub_situation_id
            MATCH (sub:SubSituation {id: sub_situation_id})
            OPTIONAL MATCH (sub)-[:HAS_SERVICE]->(sub_service:Service)
            OPTIONAL MATCH (:Situation)-[:HAS_SUB_SITUATION]->(sub)-[:HAS_SERVICE]->(situation_service:Service)

            WITH collect(DISTINCT sub_service) + collect(DISTINCT situation_service) AS services
            UNWIND services AS svc
            WITH DISTINCT svc
            WHERE svc IS NOT NULL
            RETURN svc
            LIMIT $limit
            """, {
                "sub_situation_ids": sub_situation_ids,
                "limit": limit
            })

            services = []
            for record in result:
                service = dict(record["svc"]) if record["svc"] else {}
                if service.get("id"):
                    services.append(self.get_service_details(service["id"]))
            return services

    def search_sub_situations_with_services(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        sub_situations = self.vector_search(query, "SubSituation", top_k=top_k)
        sub_situation_ids = [
            item.get("id")
            for item in sub_situations
            if item.get("id")
        ]

        return {
            "SubSituation": sub_situations,
            "services": self.find_services_for_sub_situations(sub_situation_ids),
        }

    def search_problem_knowledge(self, queries: List[str], top_k: int = 3) -> Dict[str, Any]:
        sub_situations = []
        services = []
        service_details_by_id = {}
        seen_sub_situations = set()
        seen_services = set()
        query_tokens = self._tokens(" ".join(queries))

        for query in queries:
            if not query or not query.strip():
                continue

            for sub_situation in self.vector_search(query, "SubSituation", top_k=top_k):
                sub_situation_id = sub_situation.get("id")
                if sub_situation_id and sub_situation_id not in seen_sub_situations:
                    seen_sub_situations.add(sub_situation_id)
                    sub_situations.append(sub_situation)

            for service in self.vector_search(query, "Service", top_k=top_k):
                service_id = service.get("id")
                if service_id and service_id not in seen_services:
                    seen_services.add(service_id)
                    services.append(service)

        sub_situation_ids = [item["id"] for item in sub_situations if item.get("id")]
        for detail in self.find_services_for_sub_situations(sub_situation_ids):
            service_id = detail.get("service", {}).get("id")
            if service_id:
                self._add_ranked_service_detail(
                    service_details_by_id,
                    service_id,
                    detail,
                    query_tokens,
                    reason="sub_situation_relationship",
                    base_score=2.0
                )

        for service in services:
            service_id = service.get("id")
            if not service_id:
                continue
            detail = self.get_service_details(service_id)
            detail_id = detail.get("service", {}).get("id")
            if detail_id:
                self._add_ranked_service_detail(
                    service_details_by_id,
                    detail_id,
                    detail,
                    query_tokens,
                    reason="service_vector_match",
                    base_score=10.0 + float(service.get("score", 0) or 0)
                )

        service_details = sorted(
            service_details_by_id.values(),
            key=lambda item: item.get("retrieval_score", 0),
            reverse=True
        )[:5]

        return {
            "SubSituation": sub_situations,
            "Service": services,
            "services": service_details,
        }

    def _add_ranked_service_detail(
        self,
        service_details_by_id: Dict[str, Dict[str, Any]],
        service_id: str,
        detail: Dict[str, Any],
        query_tokens: set,
        reason: str,
        base_score: float
    ):
        text = self._service_text(detail)
        score = base_score + self._keyword_score(query_tokens, text)

        if service_id in service_details_by_id:
            existing = service_details_by_id[service_id]
            existing["retrieval_score"] = max(existing.get("retrieval_score", 0), score)
            existing.setdefault("retrieval_reasons", [])
            if reason not in existing["retrieval_reasons"]:
                existing["retrieval_reasons"].append(reason)
            return

        detail["retrieval_score"] = score
        detail["retrieval_reasons"] = [reason]
        service_details_by_id[service_id] = detail

    def _service_text(self, detail: Dict[str, Any]) -> str:
        service = detail.get("service", {})
        sections = detail.get("sections", [])
        requirements = detail.get("requirements", [])
        documents = detail.get("documents", [])
        steps = detail.get("steps", [])

        parts = [
            service.get("name", ""),
            service.get("description", ""),
            service.get("url", ""),
        ]
        parts.extend(section.get("title", "") + " " + section.get("text", "") for section in sections)
        parts.extend(requirement.get("text", "") for requirement in requirements)
        parts.extend(document.get("name", "") + " " + document.get("description", "") for document in documents)
        parts.extend(step.get("title", "") + " " + step.get("description", "") for step in steps)
        return " ".join(parts)

    def _tokens(self, text: str) -> set:
        stopwords = {
            "der", "die", "das", "und", "oder", "in", "im", "am", "an", "bei",
            "nach", "zu", "zur", "zum", "mit", "ein", "eine", "einer", "einen",
            "aalen", "service", "bw"
        }
        return {
            token.strip(".,;:!?()[]{}\"'").lower()
            for token in text.split()
            if len(token.strip(".,;:!?()[]{}\"'")) > 3
            and token.strip(".,;:!?()[]{}\"'").lower() not in stopwords
        }

    def _keyword_score(self, query_tokens: set, text: str) -> float:
        text_tokens = self._tokens(text)
        if not query_tokens or not text_tokens:
            return 0.0
        return float(len(query_tokens.intersection(text_tokens)))

    def get_service_details(self, service_id: str) -> Dict[str, Any]:
        with self._session() as session:
            result = session.run("""
            MATCH (svc:Service {id: $service_id})

            OPTIONAL MATCH (svc)-[:HAS_SECTION]->(section:ServiceSection)
            OPTIONAL MATCH (svc)-[:REQUIRES]->(requirement:Requirement)
            OPTIONAL MATCH (requirement)-[:REQUIRES_DOCUMENT]->(document:Document)
            OPTIONAL MATCH (svc)-[:HANDLED_BY]->(authority:Authority)
            OPTIONAL MATCH (svc)-[:HAS_FORM]->(form:Form)
            OPTIONAL MATCH (svc)-[:HAS_STEP]->(step:ProcessStep)
            OPTIONAL MATCH (svc)-[:HAS_LEGAL_BASIS]->(legal:LegalBasis)
            OPTIONAL MATCH (goal:Goal)-[:ACHIEVED_BY]->(svc)
            OPTIONAL MATCH (svc)-[:DEPENDS_ON]->(dependency:Service)
            OPTIONAL MATCH (svc)-[:HAS_PROBLEM]->(problem:DependencyProblem)

            RETURN
                svc,
                collect(DISTINCT section) AS sections,
                collect(DISTINCT requirement) AS requirements,
                collect(DISTINCT document) AS documents,
                collect(DISTINCT authority) AS authorities,
                collect(DISTINCT form) AS forms,
                collect(DISTINCT step) AS steps,
                collect(DISTINCT legal) AS legal_basis,
                collect(DISTINCT goal) AS goals,
                collect(DISTINCT dependency) AS dependencies,
                collect(DISTINCT problem) AS problems
            """, {"service_id": service_id})

            record = result.single()
            if not record:
                return {"service_id": service_id, "found": False}

            return {
                "found": True,
                "service": dict(record["svc"]),
                "sections": [dict(item) for item in record["sections"] if item],
                "requirements": [dict(item) for item in record["requirements"] if item],
                "documents": [dict(item) for item in record["documents"] if item],
                "authorities": [dict(item) for item in record["authorities"] if item],
                "forms": [dict(item) for item in record["forms"] if item],
                "steps": [dict(item) for item in record["steps"] if item],
                "legal_basis": [dict(item) for item in record["legal_basis"] if item],
                "goals": [dict(item) for item in record["goals"] if item],
                "dependencies": [dict(item) for item in record["dependencies"] if item],
                "problems": [dict(item) for item in record["problems"] if item],
            }

    def get_related_services(self, node_name: str, node_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        if node_name not in SEARCHABLE_NODE_LABELS:
            raise ValueError(f"Unsupported node label: {node_name}")

        if node_name == "Service":
            details = self.get_service_details(node_id)
            return [details] if details.get("found") else []

        with self._session() as session:
            result = session.run(f"""
            MATCH (n:{node_name} {{id: $node_id}})
            MATCH path = (n)-[*1..3]-(svc:Service)
            RETURN DISTINCT svc
            LIMIT $limit
            """, {
                "node_id": node_id,
                "limit": limit
            })

            services = []
            for record in result:
                service = dict(record["svc"]) if record["svc"] else {}
                if service.get("id"):
                    services.append(self.get_service_details(service["id"]))
            return services
