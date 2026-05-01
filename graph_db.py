from neo4j import GraphDatabase
from config import GDB_URL, GDB_USER, GDB_PASSWORD
from neo4j_graphrag.embeddings import OllamaEmbeddings
from neo4j_graphrag.retrievers import VectorCypherRetriever
from brain.llm import LLM_V1
import uuid
import json


class DatabaseManager:
    def __init__(self):
        self.driver = GraphDatabase.driver(GDB_URL, auth=(GDB_USER, GDB_PASSWORD))
    
    def get_session(self, db_name):
        return self.driver.session(database=db_name)
    
    def close(self):
        self.driver.close()



class GraphMemoryIngestor:
    def __init__(self, db_name="short-term"):
        self.embedding_dim = 1024
        self.db = DatabaseManager()
        self.db_name = db_name
        self.embedder = OllamaEmbeddings(model="mxbai-embed-large")
        self.llm = LLM_V1(
            system_message="""
            You are an expert entity extraction system.
            Your job is to extract entities from text and classify them.
            """
        )      
        self.ensure_vector_index(dimensions=self.embedding_dim)

    def _session(self):
        return self.db.get_session(self.db_name)

    def chunk_text(self, text, max_chars=1000, overlap_chars=150):
        text = " ".join(text.split())

        if not text:
            return []

        chunks = []
        start = 0

        while start < len(text):
            end = min(start + max_chars, len(text))

            if end < len(text):
                split_at = text.rfind(" ", start, end)
                if split_at > start:
                    end = split_at

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end >= len(text):
                break

            start = max(0, end - overlap_chars)

        return chunks


    def embed(self, text):
        return self.embedder.embed_query(text)

    def extract_entities(self, text):
        return self.llm.invoke_with_formated_response(text, formate=json.dumps([
              {"name": "...", "type": "Person|Org|Concept|Other"}
            ]))

    def ingest(self, node_name, node_id, text):
        entities = self.extract_entities(text)
        chunks = self.chunk_text(text)

        with self._session() as session:

            for i, chunk in enumerate(chunks):

                chunk_id = str(uuid.uuid4())
                embedding = self.embed(chunk)
                session.run(f"""
                MATCH (n:{node_name} {{id: $node_id}})

                CREATE (c:Chunk {{
                    id: $chunk_id,
                    embedding: $embedding,
                    index: $index
                }})

                MERGE (n)-[:HAS_CHUNK]->(c)
                """, {
                    "node_id": node_id,
                    "chunk_id": chunk_id,
                    "embedding": embedding,
                    "index": str(i)
                })

                
                for entity in entities:

                    session.run("""
                    MERGE (e:Entity {name: $name})
                    SET e.type = $type

                    WITH e

                    MATCH (c:Chunk {id: $chunk_id})
                    MERGE (c)-[:MENTIONS]->(e)
                    """, {
                        "name": entity["name"],
                        "type": entity["type"],
                        "chunk_id": chunk_id
                    })

        return {
            "node_id": node_id,
            "chunks": len(chunks)
        }

    def ensure_vector_index(
        self,
        index_name="chunk_embedding_index",
        label="Chunk",
        property_name="embedding",
        dimensions=768,
        similarity="cosine",
    ):
        print("index_name",index_name)
        with self._session() as session:

            # 🔍 Check existing indexes
            result = session.run("SHOW INDEXES YIELD name, type, labelsOrTypes, properties, options")
            
            existing_index = None
            for record in result:
                print("record",record["name"])
                if record["name"] == index_name:
                    existing_index = record
                    break
                    print("existing_index",existing_index)

            # 🚫 If exists → validate
            if existing_index:
                opts = existing_index.get("options", {}).get("indexConfig", {})

                existing_dim = opts.get("vector.dimensions")
                existing_sim = opts.get("vector.similarity_function")

                if existing_dim != dimensions or existing_sim != similarity:
                    print(f"[WARN] Index {index_name} config mismatch. Recreating...")

                    session.run(f"DROP INDEX {index_name} IF EXISTS")

                    session.run(f"""
                    CREATE VECTOR INDEX {index_name}
                    FOR (n:{label})
                    ON (n.{property_name})
                    OPTIONS {{
                        indexConfig: {{
                            `vector.dimensions`: {dimensions},
                            `vector.similarity_function`: '{similarity}'
                        }}
                    }}
                    """)
                else:
                    print(f"[OK] Index {index_name} already valid")

            # 🆕 If not exists → create
            else:
                print(f"[CREATE] Creating index {index_name}")

                session.run(f"""
                CREATE VECTOR INDEX {index_name}
                FOR (n:{label})
                ON (n.{property_name})
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: {dimensions},
                        `vector.similarity_function`: '{similarity}'
                    }}
                }}
                """)

    
    def vector_cypher_search(self, query: str, node_name: str, top_k: int = 5):
        """
        Perform Vector + Graph search filtered by node type
        """
        def format_result(record):
            return {
                **dict(record.get("parent")),
                "metadata":{
                    **dict(record.get("parent")),
                    "score": record.get("score"),
                    "entities": record.get("entities"),
                    "related_entities": record.get("related_entities")               
                }
            }

        retriever = VectorCypherRetriever(
                driver=self.db.driver,
                index_name="chunk_embedding_index",
                embedder=self.embedder,
                neo4j_database=self.db_name,

                retrieval_query=f"""
                MATCH (node)<-[:HAS_CHUNK]-(parent:{node_name})

                OPTIONAL MATCH (node)-[:MENTIONS]->(e:Entity)
                OPTIONAL MATCH (e)-[:RELATED_TO]->(e2:Entity)

                RETURN
                    collect(DISTINCT e.name) AS entities,
                    collect(DISTINCT e2.name) AS related_entities,
                    parent,
                    score
                ORDER BY score DESC
                """,
                result_formatter=format_result   # 🔥 THIS FIXES EVERYTHING
            )

        return [dict(item)["metadata"] for item in dict(retriever.search(query_text=query, top_k=top_k))["items"]]