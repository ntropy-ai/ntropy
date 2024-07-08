
from ntropy.core.utils.auth_format import *



class ModelsBaseSettings():
    def __init__(self):
        self.providers_list_map = {}

        try:
            from ntropy.core.providers.aws import AWSConnection
            from ntropy.core.providers.aws import AWSEmbeddingModels
            from ntropy.core.providers.aws import AWSEmbeddings
            self.providers_list_map["AWS"] = {
                "auth": AWSauth,
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
                }
            }
        except ImportError:
            pass

        try:
            from ntropy.core.providers.openai import OpenAIConnection
            from ntropy.core.providers.openai import OpenAIEmbeddingModels
            from ntropy.core.providers.openai import OpenAIEmbeddings
            self.providers_list_map["OpenAI"] = {
                "auth": OpenAIauth,
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
