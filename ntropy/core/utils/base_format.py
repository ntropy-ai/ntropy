"""

Document Format
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Union
from uuid import uuid4

class BaseDocument(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    metadata: dict = Field(default={})

class Document(BaseDocument):
    page_number: Union[int, None] = None
    content: Union[str, None] = None
    image: Union[str, None] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


class TextChunk(BaseDocument):
    chunk: str
    chunk_number: int
    document_id: str

class Vector(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    document_id: str
    score: Union[float, None] = None # only for results vector
    vector: List[float] = Field()
    size: int
    data_type: str
    content: Union[str, None] = None
    document_metadata: dict = Field(default={})
    output_metadata: dict = Field(default={})
    #metadata: dict = Field(default={})

    class Config:
        extra = "allow"


