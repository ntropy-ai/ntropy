
from ntropy.core.utils.auth_format import *
from ntropy.core.utils.connections_manager import ConnectionManager


class ModelsBaseSettings():
    def __init__(self):
        self.providers_list_map = {}

        try:
            from ntropy.core.providers import aws
            self.providers_list_map["AWS"] = {
                "auth": AWSAuth,
                "connect": aws.AWSConnection,
                "functions": {
                    "embeddings": aws.AWSEmbeddings,
                },
                "embeddings_models": {
                    # input format map
                    "models_map": {
                        "amazon.titan-embed-image-v1": aws.AWSEmbeddingModels.AmazonTitanMultimodalEmbeddingsG1Input,
                        "amazon.titan-embed-text-v2:0": aws.AWSEmbeddingModels.AmazonTitanEmbedTextV2Input
                    }
                },
                'settings': {
                    'default_s3_bucket': 'ntropy-test'
                }
            }
        except ImportError:
            pass

        try:
            from ntropy.core.providers.openai import OpenAIConnection, OpenAIEmbeddings, OpenaiModel, OpenAIEmbeddingModels
            self.providers_list_map["OpenAI"] = {
                "auth": OpenAIAuth,
                "connect": OpenAIConnection,
                "functions": {
                    "embeddings": OpenAIEmbeddings,
                    "chat": OpenaiModel.chat
                },
                "embeddings_models": {
                    "models_map": {
                        'openai.clip-vit-base-patch32': OpenAIEmbeddingModels.OpenAIclipVIT32
                    }
                },
                "models": {
                    "gpt-4o": OpenaiModel
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


        # models providers

        try:
            from ntropy.core.models import ollama
            self.providers_list_map['Ollama'] = {
                'functions': {
                    'generate': ollama.OllamaModel.generate,
                    'chat': ollama.OllamaModel.chat
                },
                'models': {
                    model: ollama.OllamaModelsInput for model in ollama.list_models()
                }
            }
        except ImportError:
            pass