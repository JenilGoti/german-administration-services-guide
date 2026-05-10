from config import CONVERSATION_MEMORY_BACKEND, CONVERSATION_POSTGRES_URL
from brain.memory.postgres_memory import PostgresConversationMemory
from brain.memory.sql_memory import SqlConversationMemory


def create_conversation_memory(user_id: str, conv_id: str):
    backend = (CONVERSATION_MEMORY_BACKEND or "auto").lower()

    if backend == "postgres":
        return PostgresConversationMemory(user_id=user_id, conv_id=conv_id)

    if backend == "sqlite":
        return SqlConversationMemory(user_id=user_id, conv_id=conv_id)

    if CONVERSATION_POSTGRES_URL:
        return PostgresConversationMemory(user_id=user_id, conv_id=conv_id)

    return SqlConversationMemory(user_id=user_id, conv_id=conv_id)
