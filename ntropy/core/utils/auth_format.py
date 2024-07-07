from pydantic import BaseModel, Field
from typing import List, Union, Optional
from uuid import uuid4

# AWS: https://aws.amazon.com/
class AWSauth(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    service_name: str = Field(default="AWS")
    name: Optional[str] = Field(default=None)
    access_key: str = Field(default=None)
    secret_access_key: str = Field(default=None)
    other_setting: Optional[dict] = Field(default=None)

# OpenAI: https://openai.com/
class OpenAIauth(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    service_name: str = Field(default="OpenAI")
    name: Optional[str] = Field(default=None)
    api_key: str = Field(default=None)
    other_setting: Optional[dict] = Field(default=None)

# Anthropic: https://www.anthropic.com/
class Anthropicauth(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    service_name: str = Field(default="Anthropic")
    name: Optional[str] = Field(default=None)
    api_key: str = Field(default=None)
    other_setting: Optional[dict] = Field(default=None)

# Mistral: https://www.mistralai.com/
class Mistralauth(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    service_name: str = Field(default="Mistral")
    name: Optional[str] = Field(default=None)
    api_key: str = Field(default=None)
    other_setting: Optional[dict] = Field(default=None)
