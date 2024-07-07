"""
providers for models, embeddings
"""

import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from pydantic import BaseModel, Field, ConfigDict
from typing import Union
import base64
import json
from datetime import datetime

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
            embeddingConfig: dict = Field(default_factory=lambda: {
                "outputEmbeddingLength": [256, 512, 1024]
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
            dimensions: int = Field(default=1024, description="Only the following values are accepted: 1024 (default), 512, 256.", ge=256, le=1024)
            normalize: Union[bool, None] = None
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
    
def list_models():
    embeddings_models =  ModelsBaseSettings().providers_list_map["AWS"]["embeddings_model"]["models_map"].keys()
    return {
        "embeddings_models": list(embeddings_models)
    }

def get_client():
    return ConnectionManager().get_connection("AWS").get_client()


def create_embeddings(model: str, document: Document | TextChunk | str, model_settings: dict):
    accept = "application/json"
    content_type = "application/json"
    output_metadata = {
                'model': model,
                'model_settings': model_settings,
                'timestamp': datetime.now()
            }
    embedding_model_setting = ModelsBaseSettings().providers_list_map["AWS"]["embeddings_model"]["models_map"].get(model).ModelInputSchema

    if embedding_model_setting is None:
        raise ValueError(f"Model {model} not found in settings. please check the model name.")
    
    text_input = document.page_content if isinstance(document, Document) or isinstance(document, str) else document.chunk
    image_input = document.image if isinstance(document, Document) else None
    body_fields = embedding_model_setting.model_fields
    for key, value in body_fields.items():
        if key == "inputText":
            body_fields["inputText"] = text_input
            output_metadata['chunk'] = document.chunk_number if hasattr(document, 'chunk_number') else None
            output_metadata['content'] = text_input

        elif key == "inputImage":
            body_fields["inputImage"] = base64.b64encode(open(image_input, 'rb').read()).decode('utf8') if image_input else None
            output_metadata['image_path'] = image_input
            
        elif key == "model_name":
            body_fields["model_name"] = model
        elif key in model_settings:
            body_fields[key] = model_settings[key]
        else:
            body_fields[key] = value
    try:
        embedding_model_setting.model_validate(body_fields) # validate with pydantic
    except Exception:
        raise ValueError(f"Error. please check if the settings are correct. use model_settings(model) to check the correct settings.")
    if "model_name" in body_fields:
        del body_fields["model_name"]
    client = get_client()

    print(body_fields)
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