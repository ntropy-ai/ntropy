

from .format import ChatMessage, ChatHistory
from typing import Optional, Literal, List
from datetime import datetime

class ChatManager:
    def __init__(self):
        self.chat_history = ChatHistory(messages=[])

    def add_message(self, role: Literal['user', 'assistant', 'system'], content: str, images: List[str] = None):
        if images:
            message = ChatMessage(role=role, content=content, images=images, timestamp=datetime.now().isoformat())
        else:
            message = ChatMessage(role=role, content=content, timestamp=datetime.now().isoformat())
        self.chat_history.messages.append(message)

    def get_history(self) -> list:
        return [
            {
                'role': message.role,
                'content': message.content,
                'images': [image for image in message.images] if message.images else None,
                'timestamp': message.timestamp
            }
            for message in self.chat_history.messages
        ]
