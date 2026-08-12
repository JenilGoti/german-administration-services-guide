import os
from typing import Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


GDB_URL = os.getenv("GDB_URL")
GDB_USER = os.getenv("GDB_USER")
GDB_PASSWORD = os.getenv("GDB_PASSWORD")


SHORT_TERM_DB = os.getenv("SHORT_TERM_DB", "short-term")
KNOWLEDGE_DB = os.getenv("KNOWLEDGE_DB", "dev-graph")   

# Chat / reasoning LLM provider.
# Use LLM_PROVIDER=ollama or LLM_PROVIDER=groq.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()

# Strong model provider for quality-control agents.
# Keep normal agents on LLM_PROVIDER and use this for supervisor/revision.
QUALITY_LLM_PROVIDER = os.getenv("QUALITY_LLM_PROVIDER", "groq").lower()

# Ollama chat models.
OLLAMA_DEFAULT_MODEL = os.getenv("OLLAMA_DEFAULT_MODEL", "qwen2.5-coder:7b")
OLLAMA_TRANSLATION_MODEL = os.getenv("OLLAMA_TRANSLATION_MODEL", "aya:8b")
OLLAMA_REASONING_MODEL = os.getenv("OLLAMA_REASONING_MODEL", "qwen2.5:7b-instruct")
OLLAMA_STRUCTURED_MODEL = os.getenv("OLLAMA_STRUCTURED_MODEL", "llama3.1:latest")
OLLAMA_SUPERVISOR_MODEL = os.getenv("OLLAMA_SUPERVISOR_MODEL", "llama3.1:latest")

# Groq chat models. Requires GROQ_API_KEY in the environment.
GROQ_DEFAULT_MODEL = os.getenv("GROQ_DEFAULT_MODEL", "meta-llama/llama-prompt-guard-2-86m")
GROQ_TRANSLATION_MODEL = os.getenv("GROQ_TRANSLATION_MODEL", GROQ_DEFAULT_MODEL)
GROQ_REASONING_MODEL = os.getenv("GROQ_REASONING_MODEL", GROQ_DEFAULT_MODEL)
GROQ_STRUCTURED_MODEL = os.getenv("GROQ_STRUCTURED_MODEL", GROQ_DEFAULT_MODEL)
GROQ_SUPERVISOR_MODEL = os.getenv("GROQ_SUPERVISOR_MODEL", GROQ_DEFAULT_MODEL)
GROQ_JUDGE_MODEL = os.getenv("GROQ_JUDGE_MODEL", GROQ_DEFAULT_MODEL)

# Keep embeddings on Ollama. Existing KB vectors were created with this model,
# so changing it would make vector search inconsistent until the KB is re-embedded.
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large")

# Optional Qdrant semantic memory. This is for long-term searchable memory,
# not LangGraph checkpoint storage.
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_MEMORY_COLLECTION = os.getenv("QDRANT_MEMORY_COLLECTION", "conversation_memory")
QDRANT_VECTOR_SIZE = int(os.getenv("QDRANT_VECTOR_SIZE", "1024"))
QDRANT_MEMORY_ENABLED = os.getenv("QDRANT_MEMORY_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
QDRANT_MEMORY_TOP_K = int(os.getenv("QDRANT_MEMORY_TOP_K", "3"))
QDRANT_MEMORY_TEXT_MAX_CHARS = int(os.getenv("QDRANT_MEMORY_TEXT_MAX_CHARS", "900"))
QDRANT_MEMORY_SAVE_MAX_CHARS = int(os.getenv("QDRANT_MEMORY_SAVE_MAX_CHARS", "1600"))

# SQL conversation memory for the German admin agent.
SQL_MEMORY_PATH = os.getenv("SQL_MEMORY_PATH", "data/memory.sqlite3")
SQL_MEMORY_ENABLED = os.getenv("SQL_MEMORY_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
SQL_MEMORY_RECENT_LIMIT = int(os.getenv("SQL_MEMORY_RECENT_LIMIT", "12"))
SQL_MEMORY_TEXT_MAX_CHARS = int(os.getenv("SQL_MEMORY_TEXT_MAX_CHARS", "700"))
SQL_MEMORY_SAVE_MAX_CHARS = int(os.getenv("SQL_MEMORY_SAVE_MAX_CHARS", "4000"))

# LangGraph checkpoint storage. Use a PostgreSQL connection string, not the
# Supabase project API URL.
LANGGRAPH_POSTGRES_URL = os.getenv("LANGGRAPH_POSTGRES_URL", "")
LANGGRAPH_POSTGRES_SETUP = os.getenv("LANGGRAPH_POSTGRES_SETUP", "true").lower() in {"1", "true", "yes", "on"}

# Clean user/assistant conversation memory. Use "auto" to store in Postgres
# when a Postgres URL is configured, otherwise SQLite.
CONVERSATION_MEMORY_BACKEND = os.getenv("CONVERSATION_MEMORY_BACKEND", "auto").lower()
CONVERSATION_POSTGRES_URL = os.getenv("CONVERSATION_POSTGRES_URL", LANGGRAPH_POSTGRES_URL)


def get_chat_model(role: str = "default", provider: Optional[str] = None) -> str:
    selected_provider = (provider or LLM_PROVIDER).lower()
    normalized_role = (role or "default").lower()

    models_by_provider = {
        "ollama": {
            "default": OLLAMA_DEFAULT_MODEL,
            "translation": OLLAMA_TRANSLATION_MODEL,
            "reasoning": OLLAMA_REASONING_MODEL,
            "structured": OLLAMA_STRUCTURED_MODEL,
            "supervisor": OLLAMA_SUPERVISOR_MODEL,
            "judge": OLLAMA_STRUCTURED_MODEL,
        },
        "groq": {
            "default": GROQ_DEFAULT_MODEL,
            "translation": GROQ_TRANSLATION_MODEL,
            "reasoning": GROQ_REASONING_MODEL,
            "structured": GROQ_STRUCTURED_MODEL,
            "supervisor": GROQ_SUPERVISOR_MODEL,
            "judge": GROQ_JUDGE_MODEL,
        },
    }

    if selected_provider not in models_by_provider:
        raise ValueError(f"Unsupported LLM provider: {selected_provider}")
    return models_by_provider[selected_provider].get(
        normalized_role,
        models_by_provider[selected_provider]["default"],
    )
