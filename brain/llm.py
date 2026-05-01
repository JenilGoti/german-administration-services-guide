
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from utilities import safe_json_parse
from brain.memory.conversation import Message
import json


class Llm:
    BRAIN_LLM = "qwen2.5-coder:7b"
    WEB_SEARCH_LLM = "qwen2.5-coder:7b"

    @staticmethod
    def get_llm(model: str = BRAIN_LLM):
        return Ollama(
            model=model,
            temperature=0.2
        )

def get_prompt(system_message: str, query: str, formate = None,tools = None, memory = None):
    prompt = f"""
                    {system_message}
                    
                    Respond to the query: {query}
                    
                    """
    if memory:
        prompt = memory.conversation.prompt() + prompt
    if tools:
        prompt += f"""

                        Tools:
                        {json.dumps([f"Tool: {tool}, args: {json.dumps(tools[tool]['args'])}" for tool in tools])}
                        
                        IMPORTANT: 
                            - If you not found any tool to execute return {{tool: "end", args: {{}}}}
                            - If you found a tool to execute return {{tool: "tool_name", args: {{arg1: "value1", arg2: "value2",..}}}}
                                ** make sure to use right data type for right args in json format
                        """
        return prompt
    if formate:
        prompt += f"""

                        Reply only with this JSON formate: {formate}
                        """


    return prompt

class LLM_V1:
    BASE_MODEL = "qwen2.5-coder:7b"

    def __init__(self, system_message: str,tools = None, memory = None):
        self.llm = Ollama(model=self.BASE_MODEL, temperature=0.2)
        self.system_message = system_message
        self.tools = tools
        self.memory = memory
    def add_conversation_to_memory(self, query: str, response: str):
        if self.memory:
            self.memory.conversation.add_message(
                Message("user", query)
            )
            self.memory.conversation.add_message(
                Message("assistant", response)
            )
            self.memory.save_conversation()
        
    def invoke_with_tools(self, query: str):
        selected_tool = safe_json_parse(self.llm.invoke(get_prompt(self.system_message, query, tools=self.tools, memory=self.memory)))
        response = self.tools[selected_tool["tool"]]["tool"](**selected_tool["args"])
        self.add_conversation_to_memory(query, response)
        return response

    def invoke_with_formated_response(self, query: str, formate: str):
        response = safe_json_parse(self.llm.invoke(get_prompt(self.system_message, query, formate, memory=self.memory)))
        self.add_conversation_to_memory(query, response)
        return response
    
    def invoke(self, query: str):
        # if self.memory:
        #     print(get_prompt(self.system_message, query, memory=self.memory))
        response = self.llm.invoke(get_prompt(self.system_message, query, memory=self.memory))
        self.add_conversation_to_memory(query, response)
        return response
        
    
