


from ntropy.core.utils.settings import ModelsBaseSettings
from ntropy.core.utils.base_format import Document, TextChunk, Vector
import json


def create_embeddings(model: str, document: Document | TextChunk | str, model_settings: dict = None):
    for provider in ModelsBaseSettings().providers_list_map.values():
        if model in provider["embeddings_model"]["models_map"]:
            provider_embeddings_function = provider["functions"]["embeddings"]
    try:
        return provider_embeddings_function(model, document, model_settings)
    except UnboundLocalError: # provider_embeddings_function is not defined because the model is not found
        raise ValueError(f"Model {model} not found")


def list_models():
    out = {}
    for provider in ModelsBaseSettings().providers_list_map:
        out[provider] = {"embeddings_model": []}
        for model in ModelsBaseSettings().providers_list_map[provider]["embeddings_model"]["models_map"]:
            out[provider]["embeddings_model"].append(model)
    return out

def get_model_settings(model: str):
    for provider in ModelsBaseSettings().providers_list_map:
        if model in ModelsBaseSettings().providers_list_map[provider]["embeddings_model"]["models_map"]:
            return ModelsBaseSettings().providers_list_map[provider]["embeddings_model"]["models_map"][model].ModelInputSchema.model_fields