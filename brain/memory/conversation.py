import uuid
import time
import json


class Message:
    def __init__(self, role, content):
        self.id = str(uuid.uuid4())
        self.role = role
        self.content = content
        self.type = str(type(content).__name__)
        self.timestamp = time.time()
        self.metadata = {
            "memory_status": "new"
        }

    def to_dict(self):
        return {
            "id": self.id,
            "role": self.role,
            "content": json.dumps(self.content) if isinstance(self.content, dict) else self.content,
            "type": self.type,
            "timestamp": self.timestamp,
        }
    
    def prompt(self):
        return f"""[{self.role},{self.timestamp}]: {json.dumps(self.content) if isinstance(self.content, dict) else self.content}\n"""

class ConversationSummary:
    def __init__(self, conversation: Conversation, messages, llm=None):
        self.id = str(uuid.uuid4())
        self.conversation_id = conversation.id
        self.user_id = conversation.user_id

        self.messages = messages

        self.type = "conversation_summary"

        if messages:
            self.timestamp = messages[0].timestamp
            self.start_timestamp = messages[0].timestamp
            self.end_timestamp = messages[-1].timestamp
            self.message_count = len(messages)
        else:
            self.timestamp = None
            self.start_timestamp = None
            self.end_timestamp = None
            self.message_count = 0

        

        self.metadata = {
            "memory_type": "summary",
            "status": "generated"
        }

        if llm:
            self.content = self._generate_summary(llm)
        else:
            self.content = ""

    def _generate_summary(self, llm):
        text_block = "\n".join([
            f"{m.role}: {m.content}" for m in self.messages
        ])

        return llm.invoke(f"""
        Summarize the following conversation chunk briefly:

        {text_block}

        Requirements:
        - Keep important facts
        - don't miss any person name or id
        - Preserve user intent
        - Remove repetition
        - Be concise but meaningful
        """)

        self.conversation.add_summary(summary=self)

    def to_dict(self):
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "type": self.type,
            "content": self.content,
            "timestamp": self.timestamp,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "message_count": self.message_count,
        }
    def prompt(self):
        return f"""
        [SUMMARY | {self.start_timestamp} → {self.end_timestamp}] \n
        {self.content}
        """


class Conversation:
    def __init__(self, id, user_id):
        self.id = id
        self.user_id = user_id
        self.messages = []

    def add_message(self, message: Message):
        self.messages.append(message)
    
    def add_summary(self, summary: ConversationSummary):
        self.messages.append(summary)
    
    def update_message(self, message_id: str, text: str):
        for message in self.messages:
            if message.id == message_id:
                message.text = text
                break

    def get_last_n_messages(self, n=5):
        return self.messages[-n:]
    
    def load_summaries(self, summaries):
        for summary in summaries:
            s = ConversationSummary(conversation=self, messages=[], llm=None)
            s.id = summary["id"]
            s.content = summary["content"]
            s.timestamp = summary["timestamp"]
            s.start_timestamp = summary["start_timestamp"]
            s.end_timestamp = summary["end_timestamp"]
            s.message_count = summary["message_count"]
            self.messages.append(s)
    
    def load_messages(self, messages):
        for msg in messages:
            if type(msg["content"]) == str:
                try:
                    msg["content"] = json.loads(msg["content"])
                except:
                    pass
            m = Message(msg["role"], msg["content"])
            m.id = msg["id"]
            m.type = type(msg["content"]).__name__
            m.timestamp = msg["timestamp"]
            m.metadata["memory_status"] = "saved"
            self.add_message(m)
    
    
    
    def get_between_timestamps(self, start_time, end_time):
        messages = []
        for message in self.messages:
            if type(message) == Message:
                if message.timestamp >= start_time and message.timestamp <= end_time:
                    messages.append(message)
        return messages
    
    def get_only_messages(self,from_index=None,to_index=None):
        messages = []
        for message in self.messages:
            if type(message) == Message:
                messages.append(message)
        if from_index:
            return messages[from_index:]
        if to_index:
            return messages[:to_index]
        if from_index and to_index:
            return messages[from_index:to_index]
        return messages

    def count_messages(self):
        return len(self.get_only_messages())
    
    def delete_message(self, message_id: str):
        for i, message in enumerate(self.messages):
            if message.id == message_id:
                self.messages.pop(i)
                break
    
    def prompt(self):
        conversation  = ""
        for message in sorted(self.messages, key=lambda x: x.timestamp):
            if type(message) == Message:
                conversation += message.prompt()
            elif type(message) == ConversationSummary:
                conversation += message.prompt()
        return f"""
        following is our privious conversation:
        {conversation}
        """


class User:
    def __init__(self, id):
        self.id = id
        self.conversations = []

    def add_conversation(self, conversation: Conversation):
        self.conversations.append(conversation)

    def get_conversation(self, conversation_id: str):
        for conversation in self.conversations:
            if conversation.id == conversation_id:
                return conversation
        return None

    