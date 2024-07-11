
from ntropy.core.utils.settings import ModelsBaseSettings


def list_models():
    out = {}
    for provider in ModelsBaseSettings().providers_list_map:
        out[provider] = {"embeddings_models": []}
        if 'embeddings_models' in ModelsBaseSettings().providers_list_map[provider]:
            for model in ModelsBaseSettings().providers_list_map[provider]['embeddings_models']['models_map']:
                out[provider]["embeddings_models"].append(model)
    return out

def get_model_settings(model: str):
    for provider in ModelsBaseSettings().providers_list_map:
        if 'embeddings_models' in ModelsBaseSettings().providers_list_map[provider] and model in ModelsBaseSettings().providers_list_map[provider]["embeddings_models"]["models_map"]:
            return ModelsBaseSettings().providers_list_map[provider]["embeddings_models"]["models_map"][model]().model_settings
        
    raise ValueError(f"Model {model} not found in settings.")