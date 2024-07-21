from ntropy_ai.core.utils.connections_manager import ConnectionManager
from ntropy_ai.core.utils.settings import logger
from ntropy_ai.core.utils.base_format import Vector, Document
from typing import List
from ntropy_ai.core.utils.settings import ModelsBaseSettings
from ntropy_ai.core import utils
from pinecone import Pinecone as PineconeLib
from pinecone import ServerlessSpec
import json
import logging

def get_client():
    return ConnectionManager().get_connection("Pinecone").get_client()

def require_login(func):
    def wrapper(*args, **kwargs):
        if ConnectionManager().get_connection("Pinecone") is None:
            raise Exception("Pinecone connection not found. Please initialize the connection.")
        return func(*args, **kwargs)
    return wrapper

class PineconeConnection:
    def __init__(self, api_key: str, other_setting: dict, **kwargs):
        self.api_key = api_key
        self.client = None
        self.other_setting = other_setting

    def init_connection(self):
        try:
            self.client = PineconeLib(api_key=self.api_key)
            logger.info("Pinecone connection initialized successfully.")
        except Exception as e:
            raise Exception(f"Error initializing Pinecone connection: {e}")
        
    def get_client(self):
        if self.client is None:
            self.init_connection()
        return self.client
    
    def get_other_setting(self):
        return self.other_setting


@require_login
class Pinecone:
    def __init__(self, index_name: str = None):
        self.client = get_client()
        self.other_settings = ConnectionManager().get_connection("Pinecone").get_other_setting()
        self.embedding_func = None
        self.embedding_model_settings = None
        self.embedding_model_name = None
        self.embedding_model_settings_top_k = None
        self.embedding_model_settings_include_values = None
        if not index_name:
            if not self.other_settings:
                logger.error("No index name specified for Pinecone, please provide an index name !")
            #if not self.other_settings:
            #    raise Exception("No index name specified for Pinecone, please provide an index name !")
            self.index_name = self.other_settings.get("index_name", None) if self.other_settings else None
            if self.index_name:
                logger.warning(f"No index name specified, using default index {self.index_name}")
        else:
            self.index_name = index_name
            try:
                self.pinecone_index = self.get_index(self.index_name)
            except:
                raise Exception(f"Index {self.index_name} not found, please create an index first !")
                
                
        
    def create_index(self, index_name: str, dimension: int, metric: str, spec: ServerlessSpec = ServerlessSpec(cloud='aws', region='us-east-1')):
        self.pinecone_index = self.client.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=spec
        )
        return self.pinecone_index
    
    def set_index(self, index_name: str):
        self.index_name = index_name
    
    def get_index(self, index_name: str):
        return self.client.Index(index_name)
    
    def sanitize_metadata(self, metadata):
        sanitized = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                sanitized[k] = v
            elif isinstance(v, list) and all(isinstance(i, str) for i in v):
                sanitized[k] = v
            else:
                sanitized[k] = str(v)
        return sanitized
    
    def add_vectors(self, vectors: List[Vector], namespace: str = None):
        if vectors[0].document_id or vectors[0].size:
            logger.warning("Only the fields 'id' and 'values' are supported by Pinecone. The remaining fields will be stored in 'metadata'.")
        for v in vectors:
            self.get_index(self.index_name).upsert(
                vectors=[
                    {
                        "id": v.id,
                        "values": v.vector, 
                        'metadata': self.sanitize_metadata({
                            "document_id": v.document_id,
                            'content': v.content,
                            "size": v.size,
                            'data_type': v.data_type,
                            "document_metadata": v.document_metadata,
                            "output_metadata": v.output_metadata
                        })
                    }
                ],
                namespace=namespace
            )


    # set embeddings model default
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
        

    def fetch_vectors(self, ids: List[str]):
        return self.get_index(self.index_name).fetch(ids=ids)
    
    def set_retriever_settings(self, top_k: int, include_values: bool):
        self.embedding_model_settings_top_k = top_k
        self.embedding_model_settings_include_values = include_values

    # this function returns a vector, it will be modified to return the results directly
    def query(self, 
              query_vector: List[float] = None, 
              model_settings: dict = None, 
              model: str = None, 
              query_text: str = None, 
              query_image: str = None, 
              top_k: int = 5, 
              include_values: bool = False, 
              namespace: str = None) -> list:
           
        # the model name is required
        query_dimension = self.client.describe_index(self.index_name)["dimension"]
        metric = self.client.describe_index(self.index_name)["metric"]
        if not model:
            if not self.embedding_model_name:
                raise Exception("model is required !")
            model = self.embedding_model_name
        if not query_vector:
            query_vector_func = None
            # if the user did not set a default embedding model but specified one in the parameters
            if not self.embedding_func:
                if not model_settings:
                    if not self.embedding_model_settings:
                        raise Exception("model settings is required to match the output format !")
                for provider in ModelsBaseSettings().providers_list_map:
                    if "embeddings_models" in ModelsBaseSettings().providers_list_map[provider]:
                        for model_name in ModelsBaseSettings().providers_list_map[provider]['embeddings_models']['models_map']:
                            if model_name == model:
                                query_vector_func = ModelsBaseSettings().providers_list_map[provider]['functions']['embeddings']
                                break
            else:
                logger.warning("using default embedding model")
                model_settings = self.embedding_model_settings
                query_vector_func = self.embedding_func
            if not query_vector_func:
                raise Exception(f"model {model} not found !")

            if query_text:
                document = Document(content=query_text, page_number=-1, data_type="text")
            elif query_image:
                if query_image.startswith('http'):
                    document = utils.save_img_to_temp_file(query_image, return_doc=True)
                else:
                    document = Document(image=query_image, page_number=-1, data_type="image")
            else:
                raise Exception("query_text or query_image is required !")
            
            query_vector = query_vector_func(model, document, model_settings)

        if query_vector.size != query_dimension:
            logger.warning(f"query_vector shape does not match the vector store dimension (which is {query_dimension}). use model_settings to set the correct dimension !")


        results =  self.get_index(self.index_name).query(
            vector=query_vector.vector if isinstance(query_vector, Vector) else query_vector,
            top_k=self.embedding_model_settings_top_k if self.embedding_model_settings_top_k else top_k,
            include_values=self.embedding_model_settings_include_values if self.embedding_model_settings_include_values else include_values,
            namespace=namespace
        )

        results_ids = {}
        for v in results['matches']:
            results_ids[v['id']] = v['score']

        original_vectors = self.fetch_vectors(list(results_ids.keys())) # use the fetch vectors by id to fetch back the vectors and its content inside metadata from pinecone
        # remap the vector to map the original universal format
        results_vectors = []
        for vector in original_vectors['vectors']:
            v = original_vectors['vectors'][vector]
            results_vectors.append(
                    Vector(
                        id=v['id'], 
                        score=results_ids[v['id']],
                        document_id=v['metadata']['document_id'],
                        vector=v['values'], 
                        content=v['metadata']['content'],
                        data_type=v['metadata']['data_type'],
                        size=v['metadata']['size'],
                        document_metadata = json.loads(v['metadata']['document_metadata'].replace("'", '"').replace("None", "null")),
                        output_metadata = json.loads(v['metadata']['output_metadata'].replace("'", '"').replace("None", "null"))
                    )
            )
        return results_vectors
