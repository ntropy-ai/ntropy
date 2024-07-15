


from pydantic import BaseModel
from typing import Optional, List, Literal

class ChatMessage(BaseModel):
    role: Literal['user', 'assistant', 'system']
    content: str
    images: Optional[List[str]] = None
    timestamp: Optional[str] = None

class ChatHistory(BaseModel):
    messages: List[ChatMessage]