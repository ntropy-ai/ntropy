"""
providers for models, embeddings
"""

import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from pydantic import BaseModel, Field, ConfigDict
from pydantic.fields import PydanticUndefined
from typing import Union
import base64
import json
from datetime import datetime
import warnings

from ntropy.core.utils.base_format import Vector, Document, TextChunk
from ntropy.core.utils.settings import ModelsBaseSettings
from ntropy.core.utils.connections_manager import ConnectionManager

"""
pre defined models schema for aws requests

service used: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock.html
"""

class EmbeddingModels():
    # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-mm.html

    class AmazonTitanMultimodalEmbeddingsG1Input(BaseModel):
        model_name: str = "amazon.titan-embed-image-v1"
        model_settings: dict = Field(default_factory=lambda: {
            'embeddingConfig': {
                'outputEmbeddingLength': "Only the following values are accepted: 256, 512, 1024."
            }
        })
        class ModelInputSchema(BaseModel):
            inputText: Union[str, None] = None # Document, TextChunk -> string
            inputImage: Union[str, None] = None  # base64-encoded string
            embeddingConfig: Union[dict, None] = Field(default_factory=lambda: {
                "outputEmbeddingLength": Field(default=512, description="Only the following values are accepted: 256, 512, 1024.", ge=256, le=1024)
            })
        model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html
    class AmazonTitanEmbedTextV2Input(BaseModel):
        model_name: str = "amazon.titan-embed-text-v2:0"
        model_settings: dict = Field(default_factory=lambda: {
            "dimensions": "Only the following values are accepted: 1024 (default), 512, 256.",
            "normalize": "True or False"
        })
        class ModelInputSchema(BaseModel):
            inputText: Union[str, None] = None
            # additional model settings
            dimensions: Union[int, None] = Field(default=1024, description="Only the following values are accepted: 1024 (default), 512, 256.", ge=256, le=1024)
            normalize: Union[bool, None] = True
        model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())



class AWSConnection:
    def __init__(self, access_key: str, secret_access_key: str, other_setting: dict, **kwargs):
        self.aws_access_key_id = access_key
        self.aws_secret_access_key = secret_access_key
        # other settings
        self.region_name = other_setting.get("region_name", "us-east-1") # default to us-east-1 if not provided
        self.service_name = other_setting.get("service_name", "bedrock")
        self.client = None

    def init_connection(self):
        try:
            self.client = boto3.client(
                service_name=self.service_name,
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.region_name
            )
            print("AWS connection initialized successfully.")
            
        except (NoCredentialsError, PartialCredentialsError) as e:
            raise Exception(f"Error initializing AWS connection: {e}")

    def get_client(self):
        if self.client is None:
            self.init_connection()
        return self.client
    
def get_client():
    return ConnectionManager().get_connection("AWS").get_client()

def require_login(func):
    def wrapper(*args, **kwargs):
        if ConnectionManager().get_connection("AWS") is None:
            raise Exception("AWS connection not found. Please initialize the connection.")
        return func(*args, **kwargs)
    return wrapper



@require_login
def create_embeddings(model: str, document: Document | TextChunk | str, model_settings: dict) -> Vector:
    accept = "application/json"
    content_type = "application/json"

    embedding_model_setting = ModelsBaseSettings().providers_list_map["AWS"]["embeddings_model"]["models_map"].get(model).ModelInputSchema
    if model_settings is None:
        model_settings = dict()
        warnings.warn(f"Model settings for model {model} not provided. Using default settings.")
        model_settings_ = ModelsBaseSettings().providers_list_map["AWS"]["embeddings_model"]["models_map"].get(model)().model_settings    
    if embedding_model_setting is None:
        raise ValueError(f"Model {model} not found in settings. please check the model name.")
    output_metadata = {
            'model': model,
            'model_settings': model_settings,
            'timestamp': datetime.now()
        }
        
    text_input = document.page_content if isinstance(document, Document) or isinstance(document, str) else document.chunk
    image_input = document.image if isinstance(document, Document) else None

    body_fields = {key: value.default for key, value in embedding_model_setting.model_fields.items()}

    # Update body_fields with provided model settings
    for key, value in model_settings.items():
        if key in body_fields:
            body_fields[key] = value

    # Set inputText and inputImage fields
    body_fields["inputText"] = text_input
    output_metadata['chunk'] = document.chunk_number if hasattr(document, 'chunk_number') else None
    output_metadata['content'] = text_input

    if image_input:
        body_fields["inputImage"] = base64.b64encode(open(image_input, 'rb').read()).decode('utf8')
        output_metadata['image_path'] = image_input

    # Set model_name field
    body_fields["model_name"] = model

    # Remove any fields with PydanticUndefined value
    keys_to_delete = [key for key, value in body_fields.items() if value is PydanticUndefined]
    for key in keys_to_delete:
        del body_fields[key]
    try:
        embedding_model_setting.model_validate(body_fields) # validate with pydantic
    except Exception:
        raise ValueError(f"Error. please check if the settings are correct. use model_settings(model) to check the correct settings.")
    if "model_name" in body_fields:
        del body_fields["model_name"]
    client = get_client()

    response = client.invoke_model(
        body=json.dumps(body_fields), modelId=model, accept=accept, contentType=content_type
    )
    response_body = json.loads(response.get('body').read())
    response_embeddings = response_body['embedding']

    return Vector(
        document_id=document.id,
        vector=response_embeddings,
        size=len(response_embeddings),
        data_type = "text" if text_input else "image",
        content=text_input if text_input else image_input,
        metadata=output_metadata
    )