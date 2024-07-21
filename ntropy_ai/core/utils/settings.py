from ntropy_ai.core.utils.auth_format import *
import logging


# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Create a logger instance
logger = logging.getLogger('ntropy_ai')



class ModelsBaseSettings():
    def __init__(self):
        self.providers_list_map = {}

        try:
            from ntropy_ai.core.providers import aws
            self.providers_list_map["AWS"] = {
                "auth": AWSAuth,
                "connect": aws.AWSConnection,
                "functions": {
                    "embeddings": aws.AWSEmbeddings,
                    "chat": aws.AWSBedrockModels.chat
                },

                "embeddings_models": {
                    # input format map because each models has different input format
                    "models_map": {
                        "amazon.titan-embed-image-v1": aws.AWSEmbeddingModels.AmazonTitanMultimodalEmbeddingsG1Input,
                        "amazon.titan-embed-text-v2:0": aws.AWSEmbeddingModels.AmazonTitanEmbedTextV2Input
                    }
                },
                "models": {
                    "anthropic.claude-3-haiku-20240307-v1:0": aws.AWSBedrockModelsSettings.AnthropicClaude3HaikuInput
                },
                'settings': {
                    'default_s3_bucket': 'ntropy-test-2'
                }
            }
        except ImportError:
            pass

        try:
            from ntropy_ai.core.providers.openai import OpenAIConnection, OpenAIEmbeddings, OpenaiModel, OpenAIEmbeddingModels
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
                    "gpt-4o": OpenaiModel,
                    "gpt-4o-mini": OpenaiModel,
                    "gpt-4-turbo": OpenaiModel,
                    "gpt-4": OpenaiModel
                }
            }
        except ImportError:
            pass


        try: 
            from ntropy_ai.core.providers.pinecone import PineconeConnection
            self.providers_list_map["Pinecone"] = {
                "auth": PineconeAuth,
                "connect": PineconeConnection,
            }
        except ImportError:
            pass

        # models providers
        try:
            from ntropy_ai.core.providers import ollama
            self.providers_list_map['Ollama'] = {
                'functions': {
                    'generate': ollama.OllamaModel.generate,
                    'chat': ollama.OllamaModel.chat,
                    'sgenerate': ollama.OllamaModel.sgenerate,
                    'schat': ollama.OllamaModel.schat,
                },
                'models': {
                    model: model for model in ollama.list_models()
                }
            }
        except Exception: # it can be ImportError or Httpx Ollama connection error (when the Ollama service is not started)
            pass

        try:
            from ntropy_ai.core.providers import anthropic
            self.providers_list_map['Anthropic'] = {
                "auth": AnthropicAuth,
                "connect": anthropic.AnthropicConnection,
                'functions': {
                    'chat': anthropic.AnthropicModel.chat
                },
                "models": {
                    "claude-3-5-sonnet-20240620": anthropic.AnthropicModel,
                    "claude-3-opus-20240229": anthropic.AnthropicModel,
                    "claude-3-sonnet-202402290": anthropic.AnthropicModel,
                    "claude-3-haiku-20240307": anthropic.AnthropicModel,
                }
            }
        except Exception:
            pass