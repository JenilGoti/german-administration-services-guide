import os
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("MallocStackLogging", "0")
os.environ.setdefault("MallocStackLoggingNoCompact", "0")

from brain.logger import logger
from brain.agents.app import chat_agent


def print_response(response: str):
    try:
        from rich.console import Console
        from rich.markdown import Markdown
        from rich.panel import Panel
    except ImportError:
        print("\nAdministrative Assistant:\n")
        print(response)
        print()
        return

    console = Console()
    console.print()
    console.print(Panel(Markdown(response), title="Administrative Assistant", border_style="cyan"))
    console.print()


while True:
    query = input("You: ")
    logger.info("chat.user_query length=%s", len(query))
    response = chat_agent.chat(query)
    logger.info("chat.response length=%s", len(response))
    print_response(response)