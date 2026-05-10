import logging
import os
import warnings
from logging.handlers import RotatingFileHandler
from pathlib import Path


LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "agent.log"
MCP_LOG_FILE = LOG_DIR / "mcp.log"


def _file_logger(name: str, path: Path, level: int) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        handler = RotatingFileHandler(
            path,
            maxBytes=2_000_000,
            backupCount=5,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s"
        ))
        logger.addHandler(handler)

    return logger


def setup_logging():
    warnings.filterwarnings(
        "ignore",
        message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
        category=UserWarning,
    )

    LOG_DIR.mkdir(exist_ok=True)

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logger = _file_logger("agent", LOG_FILE, level)
    mcp_logger = _file_logger("agent.mcp", MCP_LOG_FILE, logging.DEBUG)

    logging.getLogger("mcp").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("langchain_core").setLevel(logging.ERROR)
    logging.getLogger("ddgs").setLevel(logging.ERROR)
    logging.getLogger("duckduckgo_search").setLevel(logging.ERROR)
    logging.getLogger("primp").setLevel(logging.ERROR)

    return logger, mcp_logger


logger, mcp_logger = setup_logging()
