# from brain.agents.web.search import WebAgent

# search_agent = WebAgent()

# search_graph = search_agent.app

from brain.agents.german_admin_agent import GermanAdminGuideAgent

chat_agent = GermanAdminGuideAgent(user_id="4", conv_id="1")

chat_graph = chat_agent.app


