
from ntropy.core.utils.auth_format import *



class ModelsBaseSettings():
    def __init__(self):
        from ntropy.core.providers.openai import OpenAIConnection
        from ntropy.core.providers.aws import AWSConnection
        from ntropy.core.providers.aws import EmbeddingModels as AWS_EmbeddingModels
        from ntropy.core.providers.aws import create_embeddings as AWS_create_embeddings
        from ntropy.core.providers.openai import EmbeddingModels as OpenAI_EmbeddingModels
        from ntropy.core.providers.openai import create_embeddings as OpenAI_create_embeddings

        self.providers_list_map = {
            "AWS": {
                "auth": AWSauth,
                "connect": AWSConnection,
                "functions": {
                    "embeddings": AWS_create_embeddings
                },
                "embeddings_model": {
                    # input format map
                    "models_map": {
                        "amazon.titan-embed-image-v1": AWS_EmbeddingModels.AmazonTitanMultimodalEmbeddingsG1Input,
                        "amazon.titan-embed-text-v2:0": AWS_EmbeddingModels.AmazonTitanEmbedTextV2Input
                    }
                }
            },
            "OpenAI": {
                "auth": OpenAIauth,
                "connect": OpenAIConnection,
                "functions": {
                    "embeddings": OpenAI_create_embeddings
                },
                "embeddings_model": {
                    "models_map": {
                        'openai.clip-vit-base-patch32': OpenAI_EmbeddingModels.OpenAIclipVIT32
                    }
                }
                }

            }
        



