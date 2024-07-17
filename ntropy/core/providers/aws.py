from pydantic import BaseModel, Field, ConfigDict
from pydantic.fields import PydanticUndefined
from typing import Union
import base64
import json
from datetime import datetime
import warnings
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from ntropy.core.utils.base_format import Vector, Document, TextChunk
from ntropy.core.utils.settings import ModelsBaseSettings
from ntropy.core.utils.connections_manager import ConnectionManager
import boto3


# AWSConnection class handles the connection to AWS services using boto3
class AWSConnection:
    def __init__(self, access_key: str, secret_access_key: str, other_setting: dict, **kwargs):
        """
        Initializes the AWSConnection with the provided credentials and settings.

        Args:
            access_key (str): AWS access key ID.
            secret_access_key (str): AWS secret access key.
            other_setting (dict): Additional settings for the connection.
            **kwargs: Additional keyword arguments.
        """
        self.other_setting = other_setting
        self.aws_access_key_id = access_key
        self.aws_secret_access_key = secret_access_key
        # Set the region name, defaulting to 'us-east-1' if not provided
        self.region_name = other_setting.get("region_name", "us-east-1")
        self.session = None

    def init_connection(self):
        """
        Initializes the AWS session using the provided credentials and settings.
        """
        try:
            self.session = boto3.Session(
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.region_name
            )
            print("AWS connection initialized successfully.")
        except (NoCredentialsError, PartialCredentialsError) as e:
            raise Exception(f"Error initializing AWS connection: {e}")

    def get_client(self):
        """
        Returns the AWS session client, initializing the connection if necessary.

        Returns:
            boto3.Session: The AWS session client.
        """
        if self.session is None:
            self.init_connection()
        return self.session
    
    def get_other_setting(self):
        """
        Returns the additional settings for the connection.

        Returns:
            dict: The additional settings.
        """
        return self.other_setting

# Utility class for AWS-related operations
class utils:
    @staticmethod
    def get_client():
        """
        Retrieves the AWS client from the connection manager.

        Returns:
            boto3.Session: The AWS session client.
        """
        return ConnectionManager().get_connection("AWS").get_client()

    @staticmethod
    def get_other_settings():
        """
        Retrieves the additional settings from the connection manager.

        Returns:
            dict: The additional settings.
        """
        return ConnectionManager().get_connection("AWS").get_other_setting()

    @staticmethod
    def require_login(func):
        """
        Decorator to ensure that the AWS connection is initialized before executing the function.

        Args:
            func (function): The function to be decorated.

        Returns:
            function: The decorated function.
        """
        def wrapper(*args, **kwargs):
            if ConnectionManager().get_connection("AWS") is None:
                raise Exception("AWS connection not found. Please initialize the connection.")
            return func(*args, **kwargs)
        return wrapper

    @staticmethod
    @require_login
    def upload_to_s3(file_name: str, bucket: str = None, object_name: str = None):
        """
        Uploads a file to an S3 bucket.

        Args:
            file_name (str): The name of the file to be uploaded.
            bucket (str, optional): The name of the S3 bucket. Defaults to the default S3 bucket in settings.
            object_name (str, optional): The name of the object in the S3 bucket. Defaults to the file name.

        Returns:
            str: The URL of the uploaded file, or None if an error occurred.
        """
        s3_client = utils.get_client().client("s3")
        bucket = ModelsBaseSettings().providers_list_map["AWS"]["settings"]["default_s3_bucket"]
        try:
            s3_client.upload_file(file_name, bucket, object_name or file_name)
            file_url = f"https://{bucket}.s3.amazonaws.com/{object_name or file_name}"
        except FileNotFoundError:
            print("The file was not found")
            return None
        except NoCredentialsError:
            print("Credentials not available")
            return None
        return file_url


"""
Predefined models schema for AWS requests

Service used: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock.html
"""

class AWSEmbeddingModels:
    
    # Amazon Titan Multimodal Embeddings G1 Input Model
    class AmazonTitanMultimodalEmbeddingsG1Input(BaseModel):
        model_name: str = "amazon.titan-embed-image-v1"
        model_settings: dict = Field(default_factory=lambda: {
            'embeddingConfig': {
                'outputEmbeddingLength': "Only the following values are accepted: 256, 384, 1024."
            }
        })
        class ModelInputSchema(BaseModel):
            inputText: Union[str, None] = None  # Document, TextChunk -> string
            inputImage: Union[str, None] = None  # base64-encoded string
            embeddingConfig: Union[dict, None] = Field(default_factory=lambda: {
                "outputEmbeddingLength": Field(default=1024, description="Only the following values are accepted: 256, 384, 1024.", enum=[256, 384, 1024])
            })
        model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    # Amazon Titan Embed Text V2 Input Model
    class AmazonTitanEmbedTextV2Input(BaseModel):
        model_name: str = "amazon.titan-embed-text-v2:0"
        model_settings: dict = Field(default_factory=lambda: {
            "dimensions": "Only the following values are accepted: 1024 (default), 512, 256.",
            "normalize": "True or False"
        })
        class ModelInputSchema(BaseModel):
            inputText: Union[str, None] = None
            # Additional model settings
            dimensions: Union[int, None] = Field(default=1024, description="Only the following values are accepted: 1024 (default), 512, 256.", ge=256, le=1024)
            normalize: Union[bool, None] = True
        model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())


@utils.require_login
def AWSEmbeddings(model: str, document: Document | TextChunk | str, model_settings: dict) -> Vector:
    """
    Generates embeddings for a given document or text chunk using the specified AWS model.

    Args:
        model (str): The name of the AWS model to use.
        document (Document | TextChunk | str): The document or text chunk to generate embeddings for.
        model_settings (dict): The settings for the model.

    Returns:
        Vector: The generated embeddings as a Vector object.
    """
    accept = "application/json"
    content_type = "application/json"

    # Retrieve the model input schema from the settings
    embedding_model_setting = ModelsBaseSettings().providers_list_map["AWS"]["embeddings_models"]["models_map"].get(model).ModelInputSchema
    if model_settings is None:
        model_settings = dict()
        warnings.warn(f"Model settings for model {model} not provided. Using default settings.")
        model_settings_ = ModelsBaseSettings().providers_list_map["AWS"]["embeddings_models"]["models_map"].get(model)().model_settings    
    if embedding_model_setting is None:
        raise ValueError(f"Model {model} not found in settings. Please check the model name.")
    
    # Prepare metadata for the output
    output_metadata = {
        'model': model,
        'model_settings': model_settings,
        'timestamp': datetime.now().isoformat()
    }
    
    # Extract text and image inputs from the document
    text_input = document.content if isinstance(document, Document) or isinstance(document, str) else document.chunk
    image_input = document.image if isinstance(document, Document) else None

    # Initialize body fields with default values from the model input schema
    body_fields = {key: value.default for key, value in embedding_model_setting.model_fields.items()}

    # Update body fields with provided model settings
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
    
    # Check if the keys of the input model_settings are actual keys of the model
    for key in model_settings.keys():
        if key not in body_fields:
            raise ValueError(f"Model setting [{key}] does not exist for model {model}.")
    
    # Remove any fields with PydanticUndefined value
    keys_to_delete = [key for key, value in body_fields.items() if value is PydanticUndefined]
    for key in keys_to_delete:
        del body_fields[key]
    
    # Validate the body fields with Pydantic
    try:
        embedding_model_setting.model_validate(body_fields)
    except Exception:
        raise ValueError(f"Error. Please check if the settings are correct. Use model_settings(model) to check the correct settings.")
    
    # Remove model_name from body fields before sending the request
    if "model_name" in body_fields:
        del body_fields["model_name"]
    
    # Get the AWS Bedrock runtime client
    client = utils.get_client().client("bedrock-runtime")

    # Invoke the model with the prepared body fields
    response = client.invoke_model(
        body=json.dumps(body_fields), modelId=model, accept=accept, contentType=content_type
    )
    response_body = json.loads(response.get('body').read())
    response_embeddings = response_body['embedding']

    # Return the generated embeddings as a Vector object
    return Vector(
        document_id=document.id,
        vector=response_embeddings,
        size=len(response_embeddings),
        data_type="text" if text_input else "image",
        content=text_input if text_input else image_input,
        metadata=output_metadata
    )

