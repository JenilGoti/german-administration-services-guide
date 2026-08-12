from typing import Dict, Any, List
import hashlib
from datetime import datetime
from graph_db import GraphMemoryIngestor
from config import KNOWLEDGE_DB


class ServiceBWGraphWriter:
    def __init__(self, db_manager, db_name=KNOWLEDGE_DB, enable_embeddings=True, verbose=False):
        self.db = db_manager
        self.db_name = db_name
        self.enable_embeddings = enable_embeddings
        self.verbose = verbose
        self.ingestor = GraphMemoryIngestor(db_name) if enable_embeddings else None

    def _session(self):
        return self.db.get_session(self.db_name)

    def _log(self, message: str):
        if self.verbose:
            print(message)

    def _safe_list(self, value):
        return value if isinstance(value, list) else []

    def _make_id(self, *parts):
        text = "::".join(str(p) for p in parts if p is not None)
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _embed_node(self, node_name: str, node_id: str, text: str):
        if not self.enable_embeddings or not self.ingestor:
            return 0
        if not node_id or not text or not text.strip():
            return 0

        result = self.ingestor.ingest(
            node_name=node_name,
            node_id=str(node_id),
            text=text.strip()
        )
        return result.get("chunks", 0)

    def _embed_unique_rows(self, node_name: str, rows: List[Dict[str, Any]], text_builder):
        seen = set()
        chunks = 0

        for row in rows:
            node_id = str(row.get("id", ""))
            if not node_id or node_id in seen:
                continue

            seen.add(node_id)
            text = text_builder(row)
            chunks += self._embed_node(node_name, node_id, text)

        return chunks

    # ------------------------
    # CONSTRAINTS
    # ------------------------
    def ensure_constraints(self):
        queries = [
            "CREATE CONSTRAINT situation_id IF NOT EXISTS FOR (n:Situation) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT sub_situation_id IF NOT EXISTS FOR (n:SubSituation) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT service_id IF NOT EXISTS FOR (n:Service) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT service_section_id IF NOT EXISTS FOR (n:ServiceSection) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT requirement_id IF NOT EXISTS FOR (n:Requirement) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (n:Document) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT authority_id IF NOT EXISTS FOR (n:Authority) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT form_id IF NOT EXISTS FOR (n:Form) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT process_step_id IF NOT EXISTS FOR (n:ProcessStep) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT legal_basis_id IF NOT EXISTS FOR (n:LegalBasis) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT goal_id IF NOT EXISTS FOR (n:Goal) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT dependency_problem_id IF NOT EXISTS FOR (n:DependencyProblem) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT service_qa_id IF NOT EXISTS FOR (n:ServiceQA) REQUIRE n.id IS UNIQUE",
        ]

        with self._session() as session:
            for query in queries:
                session.run(query)

    # ------------------------
    # SITUATION TREE
    # ------------------------
    def insert_situation_tree(self, situations: List[Dict[str, Any]]):
        rows = []

        for situation in self._safe_list(situations):
            situation_id = str(situation.get("id", ""))
            if not situation_id:
                continue

            for sub in self._safe_list(situation.get("sub_situations")):
                sub_id = str(sub.get("id", ""))
                if not sub_id:
                    continue

                services = self._safe_list(sub.get("services"))

                if not services:
                    rows.append({
                        "situation_id": situation_id,
                        "situation_name": situation.get("name", ""),
                        "situation_url": situation.get("url", ""),
                        "sub_id": sub_id,
                        "sub_name": sub.get("name", ""),
                        "sub_url": sub.get("url", ""),
                        "sub_description": sub.get("description", ""),
                        "is_self": sub.get("isSelf", sub.get("is_self", False)),
                        "service_id": None,
                        "service_name": None,
                        "service_url": None,
                        "service_description": None,
                    })

                for service in services:
                    service_id = str(service.get("id", ""))
                    if not service_id:
                        continue

                    rows.append({
                        "situation_id": situation_id,
                        "situation_name": situation.get("name", ""),
                        "situation_url": situation.get("url", ""),
                        "sub_id": sub_id,
                        "sub_name": sub.get("name", ""),
                        "sub_url": sub.get("url", ""),
                        "sub_description": sub.get("description", ""),
                        "is_self": sub.get("isSelf", sub.get("is_self", False)),
                        "service_id": service_id,
                        "service_name": service.get("name", ""),
                        "service_url": service.get("url", ""),
                        "service_description": service.get("description", ""),
                    })

        if not rows:
            return {"situation_tree": 0}

        with self._session() as session:
            session.run("""
            UNWIND $rows AS row

            MERGE (sit:Situation {id: row.situation_id})
            SET sit.name = row.situation_name,
                sit.url = row.situation_url

            MERGE (sub:SubSituation {id: row.sub_id})
            SET sub.name = row.sub_name,
                sub.url = row.sub_url,
                sub.description = CASE
                    WHEN row.sub_description <> "" THEN row.sub_description
                    ELSE sub.description
                END,
                sub.isSelf = row.is_self

            MERGE (sit)-[:HAS_SUB_SITUATION]->(sub)

            WITH row, sit, sub
            WHERE row.service_id IS NOT NULL

            MERGE (svc:Service {id: row.service_id})
            SET svc.name = row.service_name,
                svc.url = row.service_url,
                svc.description = row.service_description,
                svc.source = "service-bw"

            MERGE (sit)-[:HAS_SERVICE]->(svc)
            MERGE (sub)-[:HAS_SERVICE]->(svc)
            """, {"rows": rows})

        situation_rows = {}
        sub_rows = {}
        service_rows = {}

        for row in rows:
            situation_rows[row["situation_id"]] = {
                "id": row["situation_id"],
                "name": row["situation_name"],
                "url": row["situation_url"],
            }

            sub_rows[row["sub_id"]] = {
                "id": row["sub_id"],
                "name": row["sub_name"],
                "description": row["sub_description"],
                "url": row["sub_url"],
            }

            if row.get("service_id"):
                service_rows[row["service_id"]] = {
                    "id": row["service_id"],
                    "name": row["service_name"],
                    "description": row["service_description"],
                    "url": row["service_url"],
                }

        situation_chunks = self._embed_unique_rows(
            "Situation",
            list(situation_rows.values()),
            lambda r: f"{r.get('name', '')}\n{r.get('url', '')}"
        )

        sub_situation_chunks = self._embed_unique_rows(
            "SubSituation",
            list(sub_rows.values()),
            lambda r: f"{r.get('name', '')}\n{r.get('description', '')}"
        )

        service_chunks = self._embed_unique_rows(
            "Service",
            list(service_rows.values()),
            lambda r: f"{r.get('name', '')}\n{r.get('description', '')}"
        )

        return {
            "situation_tree": len(rows),
            "situation_chunks": situation_chunks,
            "sub_situation_chunks": sub_situation_chunks,
            "service_chunks": service_chunks,
        }

    # ------------------------
    # SUB-SITUATION DETAILS
    # ------------------------
    def insert_sub_situation_details(self, sub_situations: List[Dict[str, Any]]):
        rows = [
            {
                "id": str(s.get("id")),
                "name": s.get("name", ""),
                "url": s.get("url", ""),
                "description": s.get("description", ""),
                "summary": s.get("summary", ""),
                "scrapedAt": s.get("scrapedAt", datetime.utcnow().isoformat()),
                "source": s.get("source", "service-bw"),
            }
            for s in self._safe_list(sub_situations)
            if s.get("id")
        ]

        if not rows:
            return {"sub_situations": 0, "sub_situation_chunks": 0}

        with self._session() as session:
            session.run("""
            UNWIND $rows AS row

            MERGE (sub:SubSituation {id: row.id})
            SET sub.name = row.name,
                sub.url = row.url,
                sub.description = row.description,
                sub.summary = row.summary,
                sub.scrapedAt = row.scrapedAt
            """, {"rows": rows})

        chunks = self._embed_unique_rows(
            "SubSituation",
            rows,
            lambda r: "\n".join(
                part
                for part in [
                    r.get("name", ""),
                    r.get("summary", ""),
                    r.get("description", ""),
                ]
                if part
            )
        )

        return {"sub_situations": len(rows), "sub_situation_chunks": chunks}

    # ------------------------
    # SERVICES
    # ------------------------
    def insert_services(self, services: List[Dict[str, Any]]):
        rows = [
            {
                "id": str(s.get("id")),
                "name": s.get("name", ""),
                "url": s.get("url", ""),
                "description": s.get("description", ""),
                "summary": s.get("summary", ""),
                "source": s.get("source", "service-bw"),
                "regionalisierbar": s.get("regionalisierbar"),
                "scrapedAt": s.get("scrapedAt", datetime.utcnow().isoformat()),
            }
            for s in self._safe_list(services)
            if s.get("id")
        ]

        if not rows:
            return {"services": 0, "service_chunks": 0}

        with self._session() as session:
            session.run("""
            UNWIND $rows AS row
            MERGE (svc:Service {id: row.id})
            SET svc.name = row.name,
                svc.url = row.url,
                svc.description = row.description,
                svc.summary = row.summary,
                svc.source = row.source,
                svc.regionalisierbar = row.regionalisierbar,
                svc.scrapedAt = row.scrapedAt
            """, {"rows": rows})

        chunks = self._embed_unique_rows(
            "Service",
            rows,
            lambda r: "\n".join(
                part
                for part in [
                    r.get("name", ""),
                    r.get("summary", ""),
                    r.get("description", ""),
                ]
                if part
            )
        )

        return {"services": len(rows), "service_chunks": chunks}

    # ------------------------
    # SERVICE SECTIONS
    # ------------------------
    def insert_service_sections(self, sections: List[Dict[str, Any]]):
        rows = [
            s for s in self._safe_list(sections)
            if s.get("id") and s.get("service_id")
        ]

        if not rows:
            return {"service_sections": 0, "service_section_chunks": 0}

        with self._session() as session:
            session.run("""
            UNWIND $rows AS row

            MERGE (sec:ServiceSection {id: row.id})
            SET sec.type = row.type,
                sec.title = row.title,
                sec.text = row.text,
                sec.html = row.html,
                sec.order = row.order

            MERGE (svc:Service {id: row.service_id})
            MERGE (svc)-[:HAS_SECTION]->(sec)
            """, {"rows": rows})

        chunks = self._embed_unique_rows(
            "ServiceSection",
            rows,
            lambda r: f"{r.get('title', '')}\n{r.get('type', '')}\n{r.get('text', '')}"
        )

        return {"service_sections": len(rows), "service_section_chunks": chunks}

    # ------------------------
    # REQUIREMENTS + DOCUMENTS
    # ------------------------
    def insert_requirements(self, requirements: List[Dict[str, Any]]):
        rows = [
            r for r in self._safe_list(requirements)
            if r.get("id") and r.get("service_id")
        ]

        if not rows:
            return {
                "requirements": 0,
                "requirement_chunks": 0,
                "document_chunks": 0,
            }

        with self._session() as session:
            session.run("""
            UNWIND $rows AS row

            MERGE (req:Requirement {id: row.id})
            SET req.text = row.text,
                req.type = row.type,
                req.mandatory = row.mandatory,
                req.confidence = row.confidence,
                req.sourceText = row.sourceText

            MERGE (svc:Service {id: row.service_id})
            MERGE (svc)-[:REQUIRES]->(req)

            WITH row, req
            WHERE row.document_normalized_name IS NOT NULL

            MERGE (doc:Document {normalizedName: row.document_normalized_name})
            SET doc.id = row.document_normalized_name,
                doc.name = row.document_name,
                doc.description = row.document_description,
                doc.language = row.document_language

            MERGE (req)-[:REQUIRES_DOCUMENT]->(doc)
            """, {"rows": rows})

        requirement_chunks = self._embed_unique_rows(
            "Requirement",
            rows,
            lambda r: f"{r.get('type', '')}\n{r.get('text', '')}"
        )

        document_rows = []
        for row in rows:
            if row.get("document_normalized_name"):
                document_rows.append({
                    "id": row.get("document_normalized_name", ""),
                    "name": row.get("document_name", ""),
                    "normalizedName": row.get("document_normalized_name", ""),
                    "description": row.get("document_description", ""),
                    "language": row.get("document_language", "de"),
                })

        document_chunks = self._embed_unique_rows(
            "Document",
            document_rows,
            lambda r: f"{r.get('name', '')}\n{r.get('normalizedName', '')}\n{r.get('description', '')}"
        )

        return {
            "requirements": len(rows),
            "requirement_chunks": requirement_chunks,
            "document_chunks": document_chunks,
        }

    # ------------------------
    # DOCUMENT ISSUERS
    # ------------------------
    def insert_document_issuers(self, issuers: List[Dict[str, Any]]):
        rows = [
            r for r in self._safe_list(issuers)
            if r.get("service_id") and r.get("document_id")
        ]

        if not rows:
            return {"document_issuers": 0}

        with self._session() as session:
            session.run("""
            UNWIND $rows AS row

            MERGE (svc:Service {id: row.service_id})
            MERGE (doc:Document {normalizedName: row.document_normalized_name})
            SET doc.id = row.document_normalized_name

            MERGE (svc)-[:ISSUES]->(doc)
            MERGE (doc)-[:OBTAINED_BY]->(svc)
            """, {"rows": rows})

        return {"document_issuers": len(rows)}

    # ------------------------
    # AUTHORITIES
    # ------------------------
    def insert_authorities(self, authorities: List[Dict[str, Any]]):
        rows = [
            a for a in self._safe_list(authorities)
            if a.get("id") and a.get("service_id")
        ]

        if not rows:
            return {"authorities": 0, "authority_chunks": 0}

        with self._session() as session:
            session.run("""
            UNWIND $rows AS row

            MERGE (auth:Authority {id: row.id})
            SET auth.name = row.name,
                auth.type = row.type,
                auth.address = row.address,
                auth.phone = row.phone,
                auth.email = row.email,
                auth.locationContext = row.locationContext

            MERGE (svc:Service {id: row.service_id})
            MERGE (svc)-[:HANDLED_BY]->(auth)
            """, {"rows": rows})

        chunks = self._embed_unique_rows(
            "Authority",
            rows,
            lambda r: f"{r.get('name', '')}\n{r.get('type', '')}\n{r.get('address', '')}\n{r.get('locationContext', '')}"
        )

        return {"authorities": len(rows), "authority_chunks": chunks}

    # ------------------------
    # FORMS
    # ------------------------
    def insert_forms(self, forms: List[Dict[str, Any]]):
        rows = [
            f for f in self._safe_list(forms)
            if f.get("id") and f.get("service_id")
        ]

        if not rows:
            return {"forms": 0, "form_chunks": 0}

        with self._session() as session:
            session.run("""
            UNWIND $rows AS row

            MERGE (form:Form {id: row.id})
            SET form.name = row.name,
                form.url = row.url,
                form.type = row.type

            MERGE (svc:Service {id: row.service_id})
            MERGE (svc)-[:HAS_FORM]->(form)
            """, {"rows": rows})

        chunks = self._embed_unique_rows(
            "Form",
            rows,
            lambda r: f"{r.get('name', '')}\n{r.get('type', '')}\n{r.get('url', '')}"
        )

        return {"forms": len(rows), "form_chunks": chunks}

    # ------------------------
    # PROCESS STEPS
    # ------------------------
    def insert_process_steps(self, steps: List[Dict[str, Any]]):
        rows = [
            s for s in self._safe_list(steps)
            if s.get("id") and s.get("service_id")
        ]

        if not rows:
            return {"process_steps": 0, "process_step_chunks": 0}

        with self._session() as session:
            session.run("""
            UNWIND $rows AS row

            MERGE (step:ProcessStep {id: row.id})
            SET step.title = row.title,
                step.description = row.description,
                step.order = row.order,
                step.channel = row.channel

            MERGE (svc:Service {id: row.service_id})
            MERGE (svc)-[:HAS_STEP]->(step)
            """, {"rows": rows})

        chunks = self._embed_unique_rows(
            "ProcessStep",
            rows,
            lambda r: f"{r.get('title', '')}\n{r.get('description', '')}\n{r.get('channel', '')}"
        )

        return {"process_steps": len(rows), "process_step_chunks": chunks}

    # ------------------------
    # LEGAL BASIS
    # ------------------------
    def insert_legal_basis(self, legal_basis: List[Dict[str, Any]]):
        rows = [
            l for l in self._safe_list(legal_basis)
            if l.get("id") and l.get("service_id")
        ]

        if not rows:
            return {"legal_basis": 0, "legal_basis_chunks": 0}

        with self._session() as session:
            session.run("""
            UNWIND $rows AS row

            MERGE (law:LegalBasis {id: row.id})
            SET law.title = row.title,
                law.lawCode = row.lawCode,
                law.paragraph = row.paragraph,
                law.url = row.url,
                law.text = row.text

            MERGE (svc:Service {id: row.service_id})
            MERGE (svc)-[:HAS_LEGAL_BASIS]->(law)
            """, {"rows": rows})

        chunks = self._embed_unique_rows(
            "LegalBasis",
            rows,
            lambda r: f"{r.get('title', '')}\n{r.get('lawCode', '')}\n{r.get('paragraph', '')}\n{r.get('text', '')}"
        )

        return {"legal_basis": len(rows), "legal_basis_chunks": chunks}

    # ------------------------
    # GOALS
    # ------------------------
    def insert_goals(self, goals: List[Dict[str, Any]]):
        rows = [
            g for g in self._safe_list(goals)
            if g.get("id") and g.get("service_id")
        ]

        if not rows:
            return {"goals": 0, "goal_chunks": 0}

        with self._session() as session:
            session.run("""
            UNWIND $rows AS row

            MERGE (goal:Goal {id: row.id})
            SET goal.name = row.name,
                goal.description = row.description

            MERGE (svc:Service {id: row.service_id})
            MERGE (goal)-[:ACHIEVED_BY]->(svc)
            """, {"rows": rows})

        chunks = self._embed_unique_rows(
            "Goal",
            rows,
            lambda r: f"{r.get('name', '')}\n{r.get('description', '')}"
        )

        return {"goals": len(rows), "goal_chunks": chunks}

    # ------------------------
    # DEPENDENCY PROBLEMS
    # ------------------------
    def insert_dependency_problems(self, problems: List[Dict[str, Any]]):
        rows = [
            p for p in self._safe_list(problems)
            if p.get("id")
        ]

        if not rows:
            return {"dependency_problems": 0, "dependency_problem_chunks": 0}

        with self._session() as session:
            session.run("""
            UNWIND $rows AS row

            MERGE (problem:DependencyProblem {id: row.id})
            SET problem.type = row.type,
                problem.description = row.description,
                problem.severity = row.severity,
                problem.detectedAt = row.detectedAt

            WITH row, problem
            OPTIONAL MATCH (svc:Service {id: row.service_id})
            FOREACH (_ IN CASE WHEN svc IS NULL THEN [] ELSE [1] END |
                MERGE (svc)-[:HAS_PROBLEM]->(problem)
                MERGE (problem)-[:INVOLVES_SERVICE]->(svc)
            )

            WITH row, problem
            OPTIONAL MATCH (doc:Document {normalizedName: row.document_normalized_name})
            FOREACH (_ IN CASE WHEN doc IS NULL THEN [] ELSE [1] END |
                MERGE (problem)-[:INVOLVES_DOCUMENT]->(doc)
            )
            """, {"rows": rows})

        chunks = self._embed_unique_rows(
            "DependencyProblem",
            rows,
            lambda r: f"{r.get('type', '')}\n{r.get('severity', '')}\n{r.get('description', '')}"
        )

        return {
            "dependency_problems": len(rows),
            "dependency_problem_chunks": chunks,
        }

    # ------------------------
    # SERVICE Q&A FACTS
    # ------------------------
    def insert_service_qa(self, service_qa: List[Dict[str, Any]]):
        rows = []
        for index, item in enumerate(self._safe_list(service_qa), start=1):
            service_id = str(item.get("service_id", ""))
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            if not service_id or not question or not answer:
                continue

            order = item.get("order") or index
            rows.append({
                "id": str(item.get("id") or f"{service_id}_qa_{order}"),
                "service_id": service_id,
                "category": item.get("category", "overview"),
                "question": question,
                "answer": answer,
                "sourceText": item.get("sourceText", ""),
                "confidence": item.get("confidence", 0.8),
                "order": order,
                "scrapedAt": item.get("scrapedAt", datetime.utcnow().isoformat()),
            })

        if not rows:
            return {"service_qa": 0, "service_qa_chunks": 0}

        with self._session() as session:
            session.run("""
            UNWIND $rows AS row

            MERGE (qa:ServiceQA {id: row.id})
            SET qa.service_id = row.service_id,
                qa.category = row.category,
                qa.question = row.question,
                qa.answer = row.answer,
                qa.sourceText = row.sourceText,
                qa.confidence = row.confidence,
                qa.order = row.order,
                qa.scrapedAt = row.scrapedAt

            MERGE (svc:Service {id: row.service_id})
            MERGE (svc)-[:HAS_QA]->(qa)
            """, {"rows": rows})

        chunks = self._embed_unique_rows(
            "ServiceQA",
            rows,
            lambda r: "\n".join(
                part
                for part in [
                    r.get("category", ""),
                    r.get("question", ""),
                    r.get("answer", ""),
                    r.get("sourceText", ""),
                ]
                if part
            )
        )

        return {"service_qa": len(rows), "service_qa_chunks": chunks}

    # ------------------------
    # DERIVE DEPENDS_ON
    # ------------------------
    def derive_service_dependencies(self):
        with self._session() as session:
            result = session.run("""
            MATCH (a:Service)-[:REQUIRES]->(:Requirement)-[:REQUIRES_DOCUMENT]->(doc:Document)
            MATCH (doc)-[:OBTAINED_BY]->(b:Service)
            WHERE a.id <> b.id
            MERGE (a)-[:DEPENDS_ON]->(b)
            RETURN count(*) AS count
            """)
            return {"dependencies": result.single()["count"]}

    # ------------------------
    # BULK IMPORT
    # ------------------------
    def bulk_import(self, payload: Dict[str, Any]):
        result = {}
        if payload.get("situations", []):
            result.update(self.insert_situation_tree(payload.get("situations", [])))
            self._log("inserted situations")
        if payload.get("sub_situations", []):
            result.update(self.insert_sub_situation_details(payload.get("sub_situations", [])))
            self._log("inserted sub situations")
        if payload.get("services", []):
            result.update(self.insert_services(payload.get("services", [])))
            self._log("inserted services")
        if payload.get("service_sections", []):
            result.update(self.insert_service_sections(payload.get("service_sections", [])))
            self._log("inserted service sections")
        if payload.get("requirements", []):
            result.update(self.insert_requirements(payload.get("requirements", [])))
            self._log("inserted requirements")
        if payload.get("document_issuers", []):
            result.update(self.insert_document_issuers(payload.get("document_issuers", [])))
            self._log("inserted document issuers")
        if payload.get("authorities", []):
            result.update(self.insert_authorities(payload.get("authorities", [])))
            self._log("inserted authorities")
        if payload.get("forms", []):
            result.update(self.insert_forms(payload.get("forms", [])))
            self._log("inserted forms")
        if payload.get("process_steps", []):
            result.update(self.insert_process_steps(payload.get("process_steps", [])))
            self._log("inserted process steps")
        if payload.get("legal_basis", []):
            result.update(self.insert_legal_basis(payload.get("legal_basis", [])))
            self._log("inserted legal basis")
        if payload.get("goals", []):
            result.update(self.insert_goals(payload.get("goals", [])))
            self._log("inserted goals")
        if payload.get("dependency_problems", []):
            result.update(self.insert_dependency_problems(payload.get("dependency_problems", [])))
            self._log("inserted dependency problems")
        if payload.get("service_qa", []):
            result.update(self.insert_service_qa(payload.get("service_qa", [])))
            self._log("inserted service qa")
        if payload.get("derive_dependencies", True):
            result.update(self.derive_service_dependencies())
            self._log("derived dependencies")

        return result
