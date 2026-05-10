from config import LANGGRAPH_POSTGRES_SETUP, LANGGRAPH_POSTGRES_URL


try:
    from langgraph.checkpoint.memory import InMemorySaver
except ImportError:
    from langgraph.checkpoint.memory import MemorySaver as InMemorySaver


class CheckpointerFactory:
    def __init__(self):
        self.context_manager = None

    def create(self):
        if not LANGGRAPH_POSTGRES_URL:
            return InMemorySaver()

        try:
            from langgraph.checkpoint.postgres import PostgresSaver
        except ImportError as exc:
            raise ImportError(
                "Postgres checkpoints require langgraph-checkpoint-postgres. "
                "Install it with `pip install langgraph-checkpoint-postgres psycopg[binary,pool]`."
            ) from exc

        self.context_manager = PostgresSaver.from_conn_string(LANGGRAPH_POSTGRES_URL)
        checkpointer = self.context_manager.__enter__()
        if LANGGRAPH_POSTGRES_SETUP:
            checkpointer.setup()
        return checkpointer

    def close(self):
        if self.context_manager:
            self.context_manager.__exit__(None, None, None)
            self.context_manager = None
