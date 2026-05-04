# from brain.agents.app import chat_agent


# while True:
#     query = input("You: ")
#     response = chat_agent.chat(query)
#     print("Wraith: ", response)

# from graph_db import GraphMemoryIngestor

# memory = GraphMemoryIngestor(db_name="short-term")

# print(memory.vector_cypher_search("What is football?", "Message", top_k=10))
# from scrapping.flush_all_situations_mapping import main
# main()
# from scrapping.flush_services_from_listing import main
# main()
from scrapping.flush_sub_situations_from_listing import main
main()

