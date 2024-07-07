from pydantic import BaseModel, Field, ConfigDict
from typing import Union
import openai

import clip as OpenaiCLIP # pip install git+https://github.com/openai/CLIP.git
from ntropy.core.utils.settings import ModelsBaseSettings
from ntropy.core.utils.connections_manager import ConnectionManager
from ntropy.core.utils.base_format import Document, TextChunk, Vector
from datetime import datetime
import torch
from PIL import Image
import warnings



class EmbeddingModels():

    # https://github.com/openai/CLIP
    class OpenAIclipVIT32(BaseModel):
        model_name: str = "openai.clip-vit-base-patch32"
        model_settings: dict = Field(default_factory=lambda: {
            "device": "torch device: mps, cpu, cuda"
        })
        config: Union[dict, None] = Field(default_factory=lambda: {
            "variant": "clip",
            'model_name': 'ViT-B/32'
        })
        class ModelInputSchema(BaseModel):
            input_document: Union[Document, TextChunk, None] = None

        model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

class OpenAIConnection():
    def __init__(self, api_key: str, other_setting: dict, **kwargs):
        self.api_key = api_key
        self.client = None

    def init_connection(self):
        try: 
            self.client = openai.OpenAI(api_key=self.api_key)
            print("OpenAI connection initialized successfully.")
        except Exception as e:
            raise Exception(f"Error initializing OpenAI connection: {e}")
        

    def get_client(self):
        if self.client is None:
            self.init_connection()
        return self.client
    

class CLIPmodel():
    _model_cache = {}
    # ensure the model is loaded only once
    def __init__(self, model: str, model_settings: dict = None):
        self.model = ModelsBaseSettings().providers_list_map["OpenAI"]["embeddings_model"]["models_map"].get(model)().config['model_name']
        self.device = model_settings.get("device") if model_settings and "device" in model_settings else "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        if model not in self._model_cache:
            self.clip_model_pipe, self.clip_processor = OpenaiCLIP.load(self.model, device=self.device)
            self._model_cache[model] = {
                "clip_model_pipe": self.clip_model_pipe,
                "clip_processor": self.clip_processor
            }
        self.clip_model_pipe = self._model_cache[model]["clip_model_pipe"]
        self.clip_processor = self._model_cache[model]["clip_processor"]

    def create_embeddings_clip(self, body_fields: dict, model_settings: dict = None):
        input_document = body_fields.get('input_document')
        if input_document is None:
            raise ValueError("input_document is required for creating embeddings.")

        if isinstance(input_document, Document):
            text_input = input_document.page_content
            if text_input:
                warnings.warn("The input_document is a Document object. ClIP embeddings model has token limits. Please use TextChunk for embedding if you have long text.")
            image_input = input_document.image
        elif isinstance(input_document, TextChunk):
            text_input = input_document.chunk
            image_input = None
        else:
            raise ValueError("input_document must be of type Document or TextChunk.")
        
        if text_input and image_input:
            raise ValueError("input_document must contain either text or image content.")
        if text_input:
            text = OpenaiCLIP.tokenize([text_input]).to(self.device)
            with torch.no_grad():
                text_features = self.clip_model_pipe.encode_text(text)
            embeddings = text_features.cpu().tolist()[0]
        elif image_input:
            image = self.clip_processor(Image.open(image_input)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embeddings = self.clip_model_pipe.encode_image(image)
            embeddings = embeddings.cpu().tolist()[0]
        else:
            raise ValueError("input_document must contain either text or image content.")
        return embeddings



def list_models():
    embeddings_models =  ModelsBaseSettings().providers_list_map["OpenAI"]["embeddings_model"]["models_map"].keys()
    return {
        "embeddings_models": list(embeddings_models)
    }

def get_client():
    return ConnectionManager().get_connection("OpenAI").get_client()


def create_embeddings(model: str, document: Document | TextChunk | str, model_settings: dict = None):
    output_metadata = {
        'model': model,
        'model_settings': model_settings,
        'timestamp': datetime.now()
    }
    
    embedding_model_setting = ModelsBaseSettings().providers_list_map["OpenAI"]["embeddings_model"]["models_map"].get(model).ModelInputSchema
    if embedding_model_setting is None:
        raise ValueError(f"Model {model} not found in settings. Please check the model name.")

    body_fields = embedding_model_setting.model_fields
    body_fields['input_document'] = document

    try:
        embedding_model_setting.model_validate(body_fields) # validate with pydantic
    except Exception:
        raise ValueError(f"Error. please check if the settings are correct. use get_model_settings(model) to check the correct settings.")

    if ModelsBaseSettings().providers_list_map["OpenAI"]["embeddings_model"]["models_map"].get(model)().config['variant'] == 'clip':
        # cuz our function takes the document object directly
        embeddings =  CLIPmodel(model).create_embeddings_clip(body_fields, model_settings)

    content = document.page_content if isinstance(document, Document) else document.chunk if isinstance(document, TextChunk) else None

    return Vector(
        document_id=document.id,
        vector=embeddings,
        size=len(embeddings),
        data_type="text" if isinstance(document, TextChunk) else "image",
        content=content,
        metadata=output_metadata
    )