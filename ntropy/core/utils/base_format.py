"""

Document Format
"""
from pydantic import BaseModel, Field
from typing import List, Union
from uuid import uuid4

class BaseDocument(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    metadata: dict = Field(default={})

class Document(BaseDocument):
    page_number: int
    page_content: Union[str, None] = None
    image: Union[str, None] = None

class TextChunk(BaseDocument):
    chunk: str
    chunk_number: int
    document_id: str


class Vector(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    document_id: str
    vector: List[float] = Field()
    size: int
    data_type: str
    content: str
    metadata: dict = Field(default={})


