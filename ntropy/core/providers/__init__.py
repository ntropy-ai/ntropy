
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import openai
from pinecone import Pinecone
from ntropy.core.utils.settings import ModelsBaseSettings

def list_models(by_provider: str = None, models_only: bool = False, embeddings_only: bool = False):
    out = {}
    if embeddings_only:
        for provider in ModelsBaseSettings().providers_list_map:
            if 'embeddings_models' in ModelsBaseSettings().providers_list_map[provider]:
                out[provider] = {"embeddings_models": []}
                for model in ModelsBaseSettings().providers_list_map[provider]['embeddings_models']['models_map']:
                    out[provider]["embeddings_models"].append(model)
    elif models_only:
        for provider in ModelsBaseSettings().providers_list_map:
            if 'models' in ModelsBaseSettings().providers_list_map[provider]:
                out[provider] = {"models": []}
                for model in ModelsBaseSettings().providers_list_map[provider]['models']:
                    out[provider]["models"].append(model)
    elif by_provider:
        if 'embeddings_models' in ModelsBaseSettings().providers_list_map[by_provider]:
            out[by_provider] = {"embeddings_models": []}
            for model in ModelsBaseSettings().providers_list_map[by_provider]['embeddings_models']['models_map']:
                out[by_provider]["embeddings_models"].append(model)
        if 'models' in ModelsBaseSettings().providers_list_map[by_provider]:
            out[by_provider] = {"models": []}
            for model in ModelsBaseSettings().providers_list_map[by_provider]['models']:
                out[by_provider]["models"].append(model)

    else:
        for provider in ModelsBaseSettings().providers_list_map:
            out[provider] = {}
            if 'embeddings_models' in ModelsBaseSettings().providers_list_map[provider]:
                out[provider]["embeddings_models"] = []
                for model in ModelsBaseSettings().providers_list_map[provider]['embeddings_models']['models_map']:
                    out[provider]["embeddings_models"].append(model)
            if 'models' in ModelsBaseSettings().providers_list_map[provider]:
                out[provider]["models"] = []
                for model in ModelsBaseSettings().providers_list_map[provider]['models']:
                    out[provider]["models"].append(model)

            if not out[provider]:
                del out[provider]
    return out

def get_model_settings(model: str):
    for provider in ModelsBaseSettings().providers_list_map:
        if 'embeddings_models' in ModelsBaseSettings().providers_list_map[provider] and model in ModelsBaseSettings().providers_list_map[provider]["embeddings_models"]["models_map"]:
            return ModelsBaseSettings().providers_list_map[provider]["embeddings_models"]["models_map"][model]().model_settings
        
    raise ValueError(f"Model {model} not found in settings.")



# -----------


class PineconeConnection:
    def __init__(self, api_key: str, other_setting: dict, **kwargs):
        self.api_key = api_key
        self.client = None
        self.other_setting = other_setting

    def init_connection(self):
        try:
            self.client = Pinecone(api_key=self.api_key)
            print("Pinecone connection initialized successfully.")
        except Exception as e:
            raise Exception(f"Error initializing Pinecone connection: {e}")
        
    def get_client(self):
        if self.client is None:
            self.init_connection()
        return self.client
    
    def get_other_setting(self):
        return self.other_setting
