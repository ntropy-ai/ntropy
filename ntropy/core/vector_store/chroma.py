import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from typing import List
from pydantic import BaseModel
from ntropy.core.utils.settings import ModelsBaseSettings
from ntropy.core.utils.base_format import Document as NtropyDocument

class Chroma():
    def __init__(self, client: chromadb.Client):
        self.client = client


#def save_to_chroma(collection_name: str, documents: NtropyDocument):
    

class ChromaEmbeddings(EmbeddingFunction):
    def __init__(self, embedding_function, provider_embeddings_function, model_settings: dict = None):
        self.user_model_settings = model_settings
        self.embedding_function = embedding_function
        self.provider_embeddings_function = None
        self.embeddings_model_function = None
        for provider in ModelsBaseSettings().providers_list_map.values():
            if provider_embeddings_function in provider["embeddings_model"]["models_map"]:
                self.embeddings_model_function = provider["embeddings_model"]["models_map"][provider_embeddings_function]
                self.provider_embeddings_function = provider["functions"]["embeddings"]
                break

        self.embeddings_model_settings = self.embeddings_model_function().model_settings
        self.embeddings_model_input_schema = self.embeddings_model_function().ModelInputSchema

    def __call__(self, documents: NtropyDocument) -> Embeddings:
        embeddings: Embeddings = []
        for doc in documents:
            # call the provider embeddings function with the model_settings and the documents
            embedding = self.provider_embeddings_function(
                model=self.embeddings_model_function().model_name,
                document=doc,
                model_settings=self.user_model_settings
            )
            embeddings.extend(embedding.vector)
        print(embeddings)
        return embeddings
