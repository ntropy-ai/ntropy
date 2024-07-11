
from ntropy.core.utils.connections_manager import ConnectionManager
import warnings
from pinecone import ServerlessSpec
from ntropy.core.utils.base_format import Vector, Document
from typing import List
from ntropy.core.utils.settings import ModelsBaseSettings

def get_client():
    return ConnectionManager().get_connection("Pinecone").get_client()

def require_login(func):
    def wrapper(*args, **kwargs):
        if ConnectionManager().get_connection("Pinecone") is None:
            raise Exception("Pinecone connection not found. Please initialize the connection.")
        return func(*args, **kwargs)
    return wrapper


@require_login
class Pinecone:
    def __init__(self, index_name: str = None):
        self.client = get_client()
        self.other_settings = ConnectionManager().get_connection("Pinecone").get_other_setting()
        self.embedding_func = None
        self.embedding_model_settings = None
        self.embedding_model_name = None
        if not index_name:
            if not self.other_settings:
                raise Exception("No index name specified for Pinecone, please provide an index name !")
            self.index_name = self.other_settings.get("index_name", None)
            warnings.warn(f"No index name specified, using default index {self.index_name}")
        else:
            self.index_name = index_name
        
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
    
    
    def add_vectors(self, vectors: List[Vector], namespace: str = None):
        self.get_index(self.index_name).upsert(
            vectors=[
                {"id": v.id, "values": v.vector,'document_id': v.document_id, 'data_type': v.data_type, 'content': v.content, 'metadata': v.metadata} for v in vectors
            ],
            namespace=namespace
        )


    # set embeddings model default
    def set_embeddings_model(self, model: str, model_settings: dict):
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
        


    # this function returns a vector, it will be modified to return the results directly
    def query(self, query_vector: List[float] = None, model_settings: dict = None, model: str = None, query_text: str = None, query_image: str = None, top_k: int = 5, include_values: bool = False, namespace: str = None):   
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
                    model_settings = self.embedding_model_settings
                for provider in ModelsBaseSettings().providers_list_map:
                    if "embeddings_models" in ModelsBaseSettings().providers_list_map[provider]:
                        for model_name in ModelsBaseSettings().providers_list_map[provider]['embeddings_models']['models_map']:
                            if model_name == model:
                                query_vector_func = ModelsBaseSettings().providers_list_map[provider]['functions']['embeddings']
                                break
            else:
                warnings.warn("using default embedding model")
                query_vector_func = self.embedding_func
            if not query_vector_func:
                raise Exception(f"model {model} not found !")

            if query_text:
                document = Document(page_content=query_text, page_number=-1)
            elif query_image:
                document = Document(image=query_image, page_number=-1)
            else:
                raise Exception("query_text or query_image is required !")
            query_vector = query_vector_func(model, document, model_settings)

        if query_vector.size != query_dimension:
            warnings.warn(f"query_vector shape does not match the vector store dimension (which is {query_dimension}). use model_settings to set the correct dimension !")


        results =  self.get_index(self.index_name).query(
            query_vector=query_vector.vector,
            top_k=top_k,
            include_values=include_values,
            namespace=namespace
        )

        return results

