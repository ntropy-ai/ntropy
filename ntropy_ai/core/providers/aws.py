from pydantic import BaseModel, Field, ConfigDict
from pydantic.fields import PydanticUndefined
from typing import Union, List
import base64
import json
from datetime import datetime
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from ntropy_ai.core.utils.base_format import Vector, Document, TextChunk
from ntropy_ai.core.utils.settings import ModelsBaseSettings
from ntropy_ai.core.utils.connections_manager import ConnectionManager
import boto3
import os
import random
import time
from ntropy_ai.core.utils import Loader, ensure_local_file
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from ntropy_ai.core.utils.chat import ChatManager, ChatHistory
from ntropy_ai.core.utils.settings import logger
import logging


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
        self.region_name = other_setting.get("region_name", "us-east-1") if other_setting else "us-east-1"
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
            logger.info("AWS connection initialized successfully.")
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
    
    @require_login
    def textract(document: str, retries: int = 0):

        if document.startswith("https://"):
            try:
                document_s3_bucket = document.split('//')[1].split('.s3.amazonaws.com/')[0]
                document_s3_path = document.split('.s3.amazonaws.com/')[1]
            except IndexError:
                raise ValueError("Invalid S3 URL format. Please provide a valid S3 URL.")
            
            if not document_s3_path or not document_s3_bucket:
                if retries > 1:
                    raise ValueError(f"Invalid S3 URL format. Tried to upload to S3 bucket {retries} times. Please provide a valid S3 URL.")
                # attempt to upload to a s3 bucket
                document = s3_utils.upload_to_s3(document)
                # recursive recall the function 
                return utils.textract(document, retries + 1)
        else:
            raise ValueError("Invalid S3 URL format. Please provide a valid S3 URL.")
            
        
        if os.path.exists(document):
            document = s3_utils.upload_to_s3(document)
            document_s3_bucket = document.split('//')[1].split('.s3.amazonaws.com/')[0]
            document_s3_path = document.split('//')[1].split('.s3.amazonaws.com/')[1]
            utils.textract(document, retries=0)

        else:
            if not document_s3_bucket or not document_s3_path:
                raise ValueError("Invalid document format. Please provide a valid document containing a valid S3 URL or a local path.")

        textract_client = utils.get_client().client("textract")
        response = textract_client.start_document_text_detection(
                   DocumentLocation={
                       'S3Object': {
                           'Bucket': document_s3_bucket, 
                           'Name': document_s3_path
                       }
                   },
                   ClientRequestToken=str(random.randint(1,1e10)))
        jobid = response['JobId']

        with Loader("Waiting for Textract Job to complete..."):
            while True:
                time.sleep(0.5)
                response = textract_client.get_document_text_detection(JobId=jobid)
                if response['JobStatus'] == 'SUCCEEDED':
                    break
        blocks = response['Blocks']
        text = ''
        for block in blocks:
            if block['BlockType'] == 'LINE':
                text += block['Text'] + '\n'

        return Document(
            content=text,
            metadata={
                's3_url': document
            }
        )
    
    def ensure_base64(image: str):
        if image.startswith("http"):
            image = ensure_local_file(image)
        img = open(image, 'rb').read()
        return base64.b64encode(img).decode('utf-8')
    
    def format_chat_to_bedrock_format(chat: ChatHistory):
        messages = []
        for message in chat:
            if message['role'] in ['user', 'assistant']:
                content = [{"type": "text", "text": message['content']}]
                if message['images']:
                    for image in message['images']:
                        content.append({
                                "type": "image",
                                "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": utils.ensure_base64(image)
                                }
                            })
                messages.append({"role": message['role'], "content": content})
        return messages
    

class s3_utils():

    def __init__(self, default_bucket: str = None):
        if not default_bucket:
            default_bucket = ModelsBaseSettings().providers_list_map["AWS"]["settings"]["default_s3_bucket"]
        self.default_bucket = default_bucket
        self.s3_client = utils.get_client().client("s3")

    @utils.require_login
    def upload_to_s3(self, file_name: str, bucket: str = None, object_name: str = None):
        """
        Uploads a file to an S3 bucket.

        Args:
            file_name (str): The name of the file to be uploaded.
            bucket (str, optional): The name of the S3 bucket. Defaults to the default S3 bucket in settings.
            object_name (str, optional): The name of the object in the S3 bucket. Defaults to the file name.

            Returns:
                str: The URL of the uploaded file, or None if an error occurred.
        """
        if bucket:
            self.default_bucket = bucket
        try:
            self.s3_client.upload_file(file_name, self.default_bucket, object_name or file_name)
            file_url = f"https://{self.default_bucket}.s3.amazonaws.com/{object_name or file_name}"
        except FileNotFoundError:
            logger.error("The file was not found")
            return None
        except NoCredentialsError:
            logger.error("Credentials not available")
            return None
        return file_url
    
    @utils.require_login
    def list_bucket_objects(self, bucket: str = None, folder: str = None):
        """
        Lists the objects in an S3 bucket.

        Args:
            bucket (str): The name of the S3 bucket.

        Returns:
            list: A list of objects in the S3 bucket.
        """
        if bucket:
            self.default_bucket = bucket
        if folder:
            response = self.s3_client.list_objects_v2(Bucket=self.default_bucket, Prefix=folder)
        else:
            response = self.s3_client.list_objects_v2(Bucket=self.default_bucket)
        return [item['Key'] for item in response.get('Contents', [])]
    
    @utils.require_login
    def download_from_s3(self, file_name: str, bucket: str = None, object_name: str = None):
        """
        Downloads a file from an S3 bucket.

        Args:
            file_name (str): The name of the file to be downloaded.
            bucket (str, optional): The name of the S3 bucket. Defaults to the default S3 bucket in settings.
            object_name (str, optional): The name of the object in the S3 bucket. Defaults to the file name.

        Returns:
            str: The path to the downloaded file, or None if an error occurred.
        """
        if bucket:
            self.default_bucket = bucket
        try:
            self.s3_client.download_file(self.default_bucket, object_name or file_name, file_name)
            file_path = os.path.abspath(file_name)
        except FileNotFoundError:
            logger.error("The file was not found in the bucket")
            return None
        except NoCredentialsError:
            logger.error("Credentials not available")
            return None
        return file_path



"""
Open Search Serverless

https://opensearch.org/docs/latest/clients/python-low-level/#connecting-to-amazon-opensearch-service
https://github.com/opensearch-project/opensearch-py/blob/main/samples/knn/knn_basics.py
"""
@utils.require_login
class OpenSearchServerless:
    """
    
    host: cluster endpoint, for example: my-test-domain.us-east-1.aoss.amazonaws.com
    """
    def __init__(
            self, 
            host: str, 
            default_index: str = None, 
            region: str = "us-east-1"
        ):
        self.embedding_func = None
        self.embedding_model_settings = None
        self.embedding_model_name = None
        self.default_index = default_index
        self.credentials = utils.get_client().get_credentials()
        self.host = host
        self.awsv4auth = AWSV4SignerAuth(
            self.credentials,
            region,
            'aoss' # service
        )
        self.opensearch_client = OpenSearch(
            hosts = [{'host': self.host, 'port': 443}],
            http_auth = self.awsv4auth,
            use_ssl = True,
            verify_certs = True,
            connection_class = RequestsHttpConnection,
            pool_maxsize = 20
        )  
    
    def set_embeddings_model(self, model: str, model_settings: dict = None):
        self.embedding_model_settings = model_settings
        self.embedding_model_name = model
        for provider in ModelsBaseSettings().providers_list_map:
            if "embeddings_models" in ModelsBaseSettings().providers_list_map[provider]:
                for model_name in ModelsBaseSettings().providers_list_map[provider]['embeddings_models']['models_map']:
                    if model_name == model:
                        self.embedding_func = ModelsBaseSettings().providers_list_map[provider]['functions']['embeddings']
                        # this needs to be fixed !
                        break
        if not self.embedding_func:
            raise Exception(f"model {model} not found !")

    
    def create_index(self, index_name: str, dimension: int):
        if self.opensearch_client.indices.exists(index=index_name):
            logger.warning(f"Index {index_name} already exists.")
            return 
        self.opensearch_client.indices.create(
            index_name,
            body={
                "settings": {"index.knn": True},
                "mappings": {
                    "properties": {
                        "values": {"type": "knn_vector", "dimension": dimension},

                    }
                },
            },
        )


    def add_vectors(self, vectors: List[float], index: str = None):
        """
        Adds vectors to the specified OpenSearch index using the bulk API.

        Args:
            vectors (List[Vector]): The list of vectors to add.
            index (str, optional): The name of the index. Defaults to the default index.
        """
        index = index or self.default_index
        if not index:
            raise ValueError("Index must be specified either as a parameter or as a default index.")

        actions = []
        for vector in vectors:
            action_metadata = {
                "index": {
                    "_index": index
                }
            }
            document = {
                "values": vector.vector,
                "document_id": vector.document_id,
                "content": vector.content,
                "data_type": vector.data_type,
                "document_metadata": json.dumps(vector.document_metadata) if isinstance(vector.document_metadata, dict) else vector.document_metadata,
                "output_metadata": json.dumps(vector.output_metadata) if isinstance(vector.output_metadata, dict) else vector.output_metadata,
                "metadata": json.dumps(vector.metadata) if isinstance(vector.metadata, dict) else vector.metadata
            }
            actions.append(action_metadata)
            actions.append(document)

        # Perform the bulk operation
        response = self.opensearch_client.bulk(body=actions)
        if response['errors']:
            raise Exception("Errors occurred during bulk indexing: {}".format(response['items']))
        
    def query(
            self, 
            query_vector: List[float] = None, 
            query_text: str = None,
            index: str = None, 
            top_k: int = 3
        ):
        index = index or self.default_index
        model = self.embedding_model_name
        model_settings = self.embedding_model_settings

        if not model or not model_settings:
            raise ValueError("model and model_settings are required !")
        if not index:
            raise ValueError("Index must be specified either as a parameter or as a default index.")
        

        if not query_vector:
            query_vector_func = None
            # if the user did not set a default embedding model but specified one in the parameters
            if not self.embedding_func:
                if not model_settings:
                    if not self.embedding_model_settings:
                        raise Exception("model settings is required to match the output format !")
                    model_settings = self.embedding_model_settings
                for provider in ModelsBaseSettings().providers_list_map:
                    if "embeddings_models" in ModelsBaseSettings().providers_list_map[provider]:
                        for model_name in ModelsBaseSettings().providers_list_map[provider]['embeddings_models']['models_map']:
                            if model_name == model:
                                query_vector_func = ModelsBaseSettings().providers_list_map[provider]['functions']['embeddings']
                                break
            else:
                logger.warning("using default embedding model")
                query_vector_func = self.embedding_func
            if not query_vector_func:
                raise Exception(f"model {model} not found !")

            if query_text:
                document = Document(content=query_text, page_number=-1, data_type="text")
            else:
                raise Exception("query_text or query_image is required !")
            query_vector = query_vector_func(model, document, model_settings)


        search_query = {
            "query": 
            {"knn": 
             {"values": {
                 "vector": query_vector.vector if isinstance(query_vector, Vector) else query_vector,
                   "k": top_k}}}
        }
        results = self.opensearch_client.search(index=index, body=search_query)
        
        final_out = []
        for hit in results['hits']['hits']:
            final_out.append(
                Vector(
                    score=hit['_score'],
                    size=len(hit['_source']['values']),
                    document_id=hit['_source']['document_id'],
                    vector=hit['_source']['values'],
                    content=hit['_source']['content'],
                    data_type=hit['_source']['data_type'],
                    document_metadata=json.loads(hit['_source']['document_metadata']),
                    output_metadata=json.loads(hit['_source']['output_metadata']),
                    metadata={**json.loads(hit['_source']['metadata']), "_id": hit['_id'], "_index": hit['_index']},
                )
            )
        return final_out


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
        logger.warning(f"Model settings for model {model} not provided. Using default settings.")
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
        image_input = ensure_local_file(image_input)
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


class AWSBedrockModelsSettings:

    class AnthropicClaude3HaikuInput(BaseModel):
        model_name: str = "anthropic.claude-3-haiku-20240307-v1:0"
        model_settings: dict = Field(default_factory=lambda: {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": int,
            "temperature": float,
            "messages": dict()
        })
    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())


class AWSBedrockModels():
    """
    """
    def __init__(
            self, 
            model_name: str, 
            system_prompt: str = None, 
            retriever: object = None, 
            agent_prompt: BaseModel = None,
            temperature: float = 0.5,
            max_tokens: int = 1024,
            ):
            if model_name not in ModelsBaseSettings().providers_list_map["AWS"]["models"]:
                raise ValueError(f"Model {model_name} not found in AWS settings.")
            self.model_name = model_name
            self.temperature = temperature
            self.max_tokens = max_tokens
            self.system_prompt = system_prompt # 
            self.retriever = retriever
            self.history = ChatManager() # initialize empty chat history
            # add system prompt
            if system_prompt:
                self.history.add_message(role='system', content=system_prompt)
            self.agent_prompt = agent_prompt
            self.aws_bedrock_client = utils.get_client().client("bedrock-runtime")

    @utils.require_login
    def chat(self, query: str, images: str = None):
        if self.retriever:
            context = []
            if query:
                context.extend(self.retriever(query_text=query))
            elif query and images:
                context.extend(self.retriever(query_image=images))
            if not self.agent_prompt:
                logger.error("agent_prompt is not defined.")
            prompt = self.agent_prompt(query=query, context=context)
            final_prompt = prompt.prompt
            # print('used docs: ', prompt.context_doc) access source if you want
            
            # add the prompt and the context to the chat history

            if prompt.images_list:
                self.history.add_message(role='user', content=final_prompt, images=prompt.images_list)
            else:
                self.history.add_message(role='user', content=final_prompt)


            # validate the model settings
            model_settings = ModelsBaseSettings().providers_list_map["AWS"]["models"][self.model_name]().model_settings
            model_settings['messages'] = utils.format_chat_to_bedrock_format(chat=self.history.get_history())
            model_settings['max_tokens'] = self.max_tokens
            model_settings['temperature'] = self.temperature
            ModelsBaseSettings().providers_list_map["AWS"]["models"][self.model_name]().model_validate(model_settings)

            model_requests = json.dumps(model_settings)
            try:
                # Invoke the model with the request.
                response = self.aws_bedrock_client.invoke_model(modelId=self.model_name, body=model_requests)
                model_response = json.loads(response["body"].read())
                response_text = model_response["content"][0]["text"]
                self.history.add_message(role='assistant', content=response_text)
                return response_text

            except Exception as e:
                logger.error(f"ERROR: Can't invoke '{self.model_name}'. Reason: {e}")

        
        else:
            self.history.add_message(role='user', content=query, images=images)
            model_settings = ModelsBaseSettings().providers_list_map["AWS"]["models"][self.model_name]().model_settings
            model_settings['messages'] = utils.format_chat_to_bedrock_format(chat=self.history.get_history())
            model_settings['max_tokens'] = self.max_tokens
            model_settings['temperature'] = self.temperature
            ModelsBaseSettings().providers_list_map["AWS"]["models"][self.model_name]().model_validate(model_settings)
            model_requests = json.dumps(model_settings)
            try:
                # Invoke the model with the request.
                response = self.aws_bedrock_client.invoke_model(modelId=self.model_name, body=model_requests)
                model_response = json.loads(response["body"].read())
                response_text = model_response["content"][0]["text"]
                self.history.add_message(role='assistant', content=response_text)
                return response_text

            except Exception as e:
                logger.error(f"ERROR: Can't invoke '{self.model_name}'. Reason: {e}")
