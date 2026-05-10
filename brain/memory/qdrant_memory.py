from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid

from config import (
    OLLAMA_EMBEDDING_MODEL,
    QDRANT_API_KEY,
    QDRANT_MEMORY_COLLECTION,
    QDRANT_MEMORY_ENABLED,
    QDRANT_MEMORY_SAVE_MAX_CHARS,
    QDRANT_MEMORY_TEXT_MAX_CHARS,
    QDRANT_MEMORY_TOP_K,
    QDRANT_URL,
    QDRANT_VECTOR_SIZE,
)


class QdrantConversationMemory:
    def __init__(
        self,
        user_id: str,
        conv_id: str,
        collection_name: str = QDRANT_MEMORY_COLLECTION,
    ):
        self.user_id = user_id
        self.conv_id = conv_id
        self.collection_name = collection_name
        self.enabled = QDRANT_MEMORY_ENABLED and bool(QDRANT_URL)
        self.client = None
        self.embedder = None

        if self.enabled:
            self.client = self._build_client()
            self.embedder = self._build_embedder()
            self._ensure_collection()

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.enabled or not query:
            return []

        vector = self.embedder.embed_query(query)
        query_filter = self._conversation_filter()

        try:
            points = self.client.query_points(
                collection_name=self.collection_name,
                query=vector,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            ).points
        except AttributeError:
            points = self.client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )

        memories = []
        for point in points:
            payload = getattr(point, "payload", {}) or {}
            memories.append({
                "text": payload.get("text", ""),
                "role": payload.get("role", ""),
                "created_at": payload.get("created_at", ""),
                "score": getattr(point, "score", None),
            })
        return memories

    def save_turn(self, user_message: str, assistant_message: str):
        if not self.enabled:
            return

        self.save_message("user", self._truncate(user_message, QDRANT_MEMORY_SAVE_MAX_CHARS))
        self.save_message("assistant", self._truncate(assistant_message, QDRANT_MEMORY_SAVE_MAX_CHARS))

    def save_message(self, role: str, text: str):
        if not self.enabled or not text:
            return

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                self._point(
                    role=role,
                    text=text,
                    vector=self.embedder.embed_query(text),
                )
            ],
        )

    def prompt(self, query: str, top_k: int = QDRANT_MEMORY_TOP_K) -> str:
        memories = self.search(query, top_k=top_k)
        if not memories:
            return ""

        lines = ["Relevant past conversation memory:"]
        for memory in memories:
            role = memory.get("role") or "memory"
            text = memory.get("text") or ""
            if text:
                lines.append(f"{role}: {self._truncate(text, QDRANT_MEMORY_TEXT_MAX_CHARS)}")
        return "\n".join(lines)

    def _build_client(self):
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise ImportError("Qdrant memory requires qdrant-client. Install it with `pip install qdrant-client`.") from exc

        return QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY or None,
        )

    def _build_embedder(self):
        from neo4j_graphrag.embeddings import OllamaEmbeddings

        return OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)

    def _ensure_collection(self):
        from qdrant_client.models import Distance, VectorParams

        exists = self.client.collection_exists(self.collection_name)
        if exists:
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=QDRANT_VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )

    def _conversation_filter(self):
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        return Filter(
            must=[
                FieldCondition(key="user_id", match=MatchValue(value=self.user_id)),
                FieldCondition(key="conv_id", match=MatchValue(value=self.conv_id)),
            ]
        )

    def _point(self, role: str, text: str, vector: List[float]):
        from qdrant_client.models import PointStruct

        return PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "user_id": self.user_id,
                "conv_id": self.conv_id,
                "role": role,
                "text": text,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    def _truncate(self, text: str, max_chars: int) -> str:
        if not text or len(text) <= max_chars:
            return text or ""
        return text[:max_chars].rstrip() + "..."
