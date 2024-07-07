


from ntropy.core.utils.settings import ModelsBaseSettings
from ntropy.core.utils.base_format import Document, TextChunk, Vector
import json


def create_embeddings(model: str, document: Document | TextChunk | str, model_settings: dict = None):
    for provider in ModelsBaseSettings().providers_list_map.values():
        if model in provider["embeddings_model"]["models_map"]:
            provider_embeddings_function = provider["functions"]["embeddings"]

    return provider_embeddings_function(model, document, model_settings)


def list_models():
    out = {}
    for provider in ModelsBaseSettings().providers_list_map:
        out[provider] = {"embeddings_model": []}
        for model in ModelsBaseSettings().providers_list_map[provider]["embeddings_model"]["models_map"]:
            out[provider]["embeddings_model"].append(model)
    return out
