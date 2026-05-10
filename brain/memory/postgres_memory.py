from datetime import datetime, timezone
from typing import List, Tuple

from config import (
    CONVERSATION_POSTGRES_URL,
    SQL_MEMORY_ENABLED,
    SQL_MEMORY_RECENT_LIMIT,
    SQL_MEMORY_SAVE_MAX_CHARS,
    SQL_MEMORY_TEXT_MAX_CHARS,
)


class PostgresConversationMemory:
    def __init__(self, user_id: str, conv_id: str, db_url: str = CONVERSATION_POSTGRES_URL):
        self.user_id = user_id
        self.conv_id = conv_id
        self.db_url = db_url
        self.enabled = SQL_MEMORY_ENABLED and bool(db_url)

        if self.enabled:
            self._ensure_schema()

    def prompt(self, query: str = "", limit: int = SQL_MEMORY_RECENT_LIMIT) -> str:
        if not self.enabled:
            return ""

        rows = self.recent(limit=limit)
        if not rows:
            return ""

        lines = ["Prior conversation messages:"]
        for role, text in rows:
            lines.append(f"{role}: {self._truncate(text, SQL_MEMORY_TEXT_MAX_CHARS)}")
        return "\n".join(lines) if len(lines) > 1 else ""

    def transcript(self, limit: int = SQL_MEMORY_RECENT_LIMIT) -> str:
        if not self.enabled:
            return ""

        rows = self.recent(limit=limit)
        if not rows:
            return ""

        lines = []
        for role, text in rows:
            lines.append(f"{role}: {self._truncate(text, SQL_MEMORY_TEXT_MAX_CHARS)}")
        return "\n".join(lines)

    def recent(self, limit: int = SQL_MEMORY_RECENT_LIMIT) -> List[Tuple[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content
                FROM conversation_messages
                WHERE user_id = %s AND conv_id = %s
                ORDER BY created_at DESC, id DESC
                LIMIT %s
                """,
                (self.user_id, self.conv_id, limit),
            ).fetchall()

        return [(row["role"], row["content"]) for row in reversed(rows)]

    def save_turn(self, user_message: str, assistant_message: str):
        if not self.enabled:
            return

        self.save_message("user", user_message)
        self.save_message("assistant", assistant_message)

    def save_message(self, role: str, content: str):
        if not self.enabled or not content:
            return

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversation_messages
                (user_id, conv_id, role, content, created_at)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    self.user_id,
                    self.conv_id,
                    role,
                    self._truncate(content, SQL_MEMORY_SAVE_MAX_CHARS),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def _ensure_schema(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id BIGSERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    conv_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversation_messages_lookup
                ON conversation_messages (user_id, conv_id, created_at)
                """
            )

    def _connect(self):
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise ImportError(
                "Postgres conversation memory requires psycopg. "
                "Install it with `pip install psycopg[binary,pool]`."
            ) from exc

        return psycopg.connect(self.db_url, row_factory=dict_row)

    def _truncate(self, text: str, max_chars: int) -> str:
        if not text or len(text) <= max_chars:
            return text or ""
        return text[:max_chars].rstrip() + "..."

    def _is_low_value_for_context(self, text: str) -> bool:
        normalized = " ".join((text or "").lower().split())
        low_value = {
            "hi",
            "hello",
            "hey",
            "thank you",
            "thanks",
            "thak you for your help",
            "you're welcome! is there anything else i can help you with?",
            "hi, i can help with german administrative questions. what do you need guidance with?",
        }
        return normalized in low_value

    def _compact_assistant_text(self, text: str) -> str:
        lines = []
        for raw_line in (text or "").splitlines():
            line = raw_line.strip(" -#*\t")
            if not line:
                continue
            lowered = line.lower()
            if any(marker in lowered for marker in [
                "likely",
                "procedure",
                "authority",
                "documents",
                "required",
                "steps",
                "follow",
                "question",
                "service",
            ]):
                lines.append(line)
            if len(lines) >= 6:
                break

        compacted = " | ".join(lines) if lines else text
        return self._truncate(compacted, SQL_MEMORY_TEXT_MAX_CHARS)
