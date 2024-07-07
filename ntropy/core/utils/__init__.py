

from ntropy.core.utils.settings import ModelsBaseSettings

def get_model_settings(model_name: str):
    for provider in ModelsBaseSettings().providers_list_map.values():
        if model_name in provider["embeddings_model"]["models_map"]:
            return provider["embeddings_model"]["models_map"].get(model_name)().model_settings
    raise ValueError(f"Model {model_name} not found in the settings.")


