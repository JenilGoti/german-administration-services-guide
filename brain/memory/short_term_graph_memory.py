from graph_db import DatabaseManager, GraphMemoryIngestor
from config import SHORT_TERM_DB
from brain.memory.conversation import Conversation, Message, ConversationSummary
from brain.llm import LLM_V1

MAX_MESSAGES = 10

class Neo4jMemory:
    _instances = {}

    @classmethod
    def get(cls, user_id: str, conv_id: str):
        key = (user_id, conv_id)

        if key not in cls._instances:
            cls._instances[key] = Neo4jMemory(user_id, conv_id)

        return cls._instances[key]

    def __init__(self, user_id, conv_id):
        self.db_manager = DatabaseManager()
        self.ingestor = GraphMemoryIngestor(SHORT_TERM_DB)
        self.user_id = user_id
        self.conv_id = conv_id
        self.conversation = Conversation(conv_id, user_id)
        self.init_graph()
        self.load_conversation()
        self.llm = LLM_V1(
            "You are a memory compression system. Summarize the conversation as a third person clearly, preserving key facts, decisions, and context."
        )
        # print('conversation',self.conversation.prompt())

    def _session(self):
        return self.db_manager.get_session(SHORT_TERM_DB)

    def init_graph(self):
        with self._session() as session:
            session.run("""
            MERGE (u:User {id: $user_id})
            MERGE (c:Conversation {id: $conv_id})
            MERGE (u)-[:HAS_CONVERSATION]->(c)
            """, {
                "user_id": self.user_id,
                "conv_id": self.conv_id
            })
            
    def save_message(self, message):
        with self._session() as session:
            session.run("""
            MERGE (u:User {id: $user_id})
            MERGE (c:Conversation {id: $conv_id})
            MERGE (u)-[:HAS_CONVERSATION]->(c)

            CREATE (m:Message {
                id: $id,
                role: $role,
                content: $content,
                type: $type,
                timestamp: $timestamp
            })

            MERGE (c)-[:HAS_MESSAGE]->(m)
            """, {
                "user_id": self.user_id,
                "conv_id": self.conv_id,
                "id": message.id,
                "role": message.role,
                "content": message.content,
                "type": message.type,
                "timestamp": message.timestamp
            })
        self.ingestor.ingest(node_name="Message", node_id=message.id, text=message.content)
    
    def link_messages(self, prev_id, curr_id):
        with self._session() as session:
            session.run("""
            MATCH (a:Message {id: $prev})
            MATCH (b:Message {id: $curr})
            MERGE (a)-[:NEXT]->(b)
            """, {
                "prev": prev_id,
                "curr": curr_id
            })

    def _get_last_message_id(self):
        """Helper to find the most recent message ID in the graph."""
        with self._session() as session:
            result = session.run("""
            MATCH (c:Conversation {id: $conv_id})-[:HAS_MESSAGE]->(m:Message)
            WHERE NOT (m)-[:NEXT]->()
            RETURN m.id AS id
            ORDER BY m.timestamp DESC
            LIMIT 1
            """, {"conv_id": self.conv_id})
            record = result.single()
            return record["id"] if record else None

    def _get_last_saved_message(self):
        messages = self.conversation.get_only_messages()
        for message in reversed(messages):
            if message.metadata["memory_status"] == "saved":
                return message.id
        return None
    def save_conversation(self):
        
        messages = self.conversation.get_only_messages()
        summariesed_messages_ids = []
        
        prev_id = self._get_last_saved_message() or self._get_last_message_id()
        # print('prev_id',prev_id)
        for msg in messages:
            # print('msg',msg.prompt())
            if msg.metadata["memory_status"] == "new":
                self.save_message(msg)
                msg.metadata["memory_status"] = "saved"
                if prev_id:
                    self.link_messages(prev_id, msg.id)

            prev_id = msg.id
        if self.conversation.count_messages() > MAX_MESSAGES:
            massages_to_summarise = self.conversation.get_only_messages(to_index=MAX_MESSAGES)
            summary = ConversationSummary(conversation=self.conversation, messages=massages_to_summarise, llm=self.llm)
            summariesed_messages_ids=[msg.id for msg in massages_to_summarise]
            self.save_summary(summary)
            self.link_summary_to_messages(summary)
        if summariesed_messages_ids:
            print('conversation',self.conversation.prompt())
            print('summariesed_messages_ids',summariesed_messages_ids)
            for id in summariesed_messages_ids:
                self.conversation.delete_message(id)
            print('conversation',self.conversation.prompt())
        
    def link_summary_to_messages(self, summary):
        message_ids = [msg.id for msg in summary.messages]

        with self._session() as session:
            session.run("""
            MATCH (s:ConversationSummary {id: $summary_id})
            UNWIND $message_ids AS mid
            MATCH (m:Message {id: mid})
            MERGE (s)-[:SUMMARIZES]->(m)
            """, {
                "summary_id": summary.id,
                "message_ids": message_ids
            })
    def save_summary(self, summary):
        with self._session() as session:
            session.run("""
            MERGE (u:User {id: $user_id})
            MERGE (c:Conversation {id: $conv_id})
            MERGE (u)-[:HAS_CONVERSATION]->(c)

            CREATE (s:ConversationSummary {
                id: $id,
                conversation_id: $conversation_id,
                type: $type,
                content: $content,
                timestamp: $timestamp,
                start_timestamp: $start_timestamp,
                end_timestamp: $end_timestamp,
                message_count: $message_count
            })

            MERGE (c)-[:HAS_SUMMARY]->(s)
            """, {
                "user_id": self.user_id,
                "conv_id": self.conv_id,
                "id": summary.id,
                "conversation_id": summary.conversation_id,
                "type": summary.type,
                "content": summary.content,
                "timestamp": summary.timestamp,
                "start_timestamp": summary.start_timestamp,
                "end_timestamp": summary.end_timestamp,
                "message_count": summary.message_count
            })
        self.ingestor.ingest(node_name="ConversationSummary", node_id=summary.id, text=summary.content)
        
    def load_summaries(self):
        with self._session() as session:
            result = session.run("""
            MATCH (c:Conversation {id: $conv_id})-[:HAS_SUMMARY]->(s:ConversationSummary)
            RETURN s
            ORDER BY s.start_timestamp ASC
            """, {
                "conv_id": self.conv_id
            })

            summaries = []
            for record in result:
                s = record["s"]

                summaries.append({
                    "id": s["id"],
                    "content": s["content"],
                    "start_timestamp": s["start_timestamp"],
                    "end_timestamp": s["end_timestamp"],
                    "message_count": s["message_count"],
                    "type": "summary",
                    "timestamp": s["timestamp"]
                })

            return summaries
            

    def load_conversation(self, n: int = MAX_MESSAGES):
        summaries = self.load_summaries()

        with self._session() as session:
            result = session.run("""
            MATCH (c:Conversation {id: $conv_id})-[:HAS_MESSAGE]->(m:Message)
            WHERE NOT EXISTS {
                MATCH (:ConversationSummary)-[:SUMMARIZES]->(m)
            }
            RETURN m
            ORDER BY m.timestamp DESC
            LIMIT $n
            """, {
                "conv_id": self.conv_id,
                "n": n
            })

            messages = [m["m"] for m in result]

            self.conversation.load_summaries(summaries)
            self.conversation.load_messages(reversed(messages))