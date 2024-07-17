


from pydantic import BaseModel
from typing import Optional, List, Literal

class ChatMessage(BaseModel):
    role: Literal['user', 'assistant', 'system', 'function']
    content: Optional[str] = None
    images: Optional[List[str]] = None
    tools: Optional[List[dict]] = None
    tool_call: Optional[dict] = None
    tool_call_response: Optional[str] = None
    timestamp: Optional[str] = None

class ChatHistory(BaseModel):
    messages: List[ChatMessage]