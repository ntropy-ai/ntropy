
from ntropy.core.utils.settings import ModelsBaseSettings


def list_models():
    out = {}
    for provider in ModelsBaseSettings().providers_list_map:
        out[provider] = {"embeddings_models": []}
        for model in ModelsBaseSettings().providers_list_map[provider]["embeddings_models"]["models_map"]:
            out[provider]["embeddings_models"].append(model)
    return out

def get_model_settings(model: str):
    for provider in ModelsBaseSettings().providers_list_map:
        if model in ModelsBaseSettings().providers_list_map[provider]["embeddings_models"]["models_map"]:
            return ModelsBaseSettings().providers_list_map[provider]["embeddings_models"]["models_map"][model].ModelInputSchema.model_fields