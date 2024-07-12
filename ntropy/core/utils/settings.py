
from ntropy.core.utils.auth_format import *
from ntropy.core.utils.connections_manager import ConnectionManager


class ModelsBaseSettings():
    def __init__(self):
        self.providers_list_map = {}

        try:
            from ntropy.core.providers import AWSConnection
            from ntropy.core.embeddings.aws import AWSEmbeddingModels
            from ntropy.core.embeddings.aws import AWSEmbeddings
            self.providers_list_map["AWS"] = {
                "auth": AWSAuth,
                "connect": AWSConnection,
                "functions": {
                    "embeddings": AWSEmbeddings
                },
                "embeddings_models": {
                    # input format map
                    "models_map": {
                        "amazon.titan-embed-image-v1": AWSEmbeddingModels.AmazonTitanMultimodalEmbeddingsG1Input,
                        "amazon.titan-embed-text-v2:0": AWSEmbeddingModels.AmazonTitanEmbedTextV2Input
                    }
                },
            }
        except ImportError:
            pass

        try:
            from ntropy.core.providers import OpenAIConnection
            from ntropy.core.embeddings.openai import OpenAIEmbeddingModels
            from ntropy.core.embeddings.openai import OpenAIEmbeddings
            self.providers_list_map["OpenAI"] = {
                "auth": OpenAIAuth,
                "connect": OpenAIConnection,
                "functions": {
                    "embeddings": OpenAIEmbeddings
                },
                "embeddings_models": {
                    "models_map": {
                        'openai.clip-vit-base-patch32': OpenAIEmbeddingModels.OpenAIclipVIT32
                    }
                }
            }
        except ImportError:
            pass


        try: 
            from ntropy.core.providers import PineconeConnection
            self.providers_list_map["Pinecone"] = {
                "auth": PineconeAuth,
                "connect": PineconeConnection,
            }
        except ImportError:
            pass
