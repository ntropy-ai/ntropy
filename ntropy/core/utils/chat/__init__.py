

from .format import ChatMessage, ChatHistory
from typing import Optional, Literal, List
from datetime import datetime

class ChatManager:
    def __init__(self):
        self.chat_history = ChatHistory(messages=[])

    def add_message(
            self, 
            role: Literal['user', 'assistant', 'system'], 
            content: Optional[str] = None, 
            images: List[str] = None, 
            tool_call: list = None, 
            tool_call_response: str = None,
            tools: List[dict] = None):
        if images:
            message = ChatMessage(
                role=role, 
                content=content, 
                images=images, 
                timestamp=datetime.now().isoformat(), 
                tool_call=tool_call, 
                tool_call_response=tool_call_response,
                tools=tools)
        else:
            message = ChatMessage(
                role=role, 
                content=content, 
                timestamp=datetime.now().isoformat(), 
                tool_call=tool_call, 
                tool_call_response=tool_call_response,
                tools=tools)
        self.chat_history.messages.append(message)

    def get_history(self) -> list:
        return [
            {
                'role': message.role,
                'content': message.content,
                'images': [image for image in message.images] if message.images else None,
                'tool_call': message.tool_call,
                'tool_call_response': message.tool_call_response,
                'tools': message.tools,
                'timestamp': message.timestamp
            }
            for message in self.chat_history.messages
        ]
