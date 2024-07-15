from pydantic import BaseModel, Field, ConfigDict
from typing import Union

import clip as OpenaiCLIP # pip install git+https://github.com/openai/CLIP.git
from ntropy.core.utils.settings import ModelsBaseSettings
from ntropy.core.utils.connections_manager import ConnectionManager
from ntropy.core.utils.base_format import Document, TextChunk, Vector
from ntropy.core.utils import save_img_to_temp_file
from datetime import datetime
import torch
from PIL import Image
import warnings
from ntropy.core import utils
from ntropy.core.utils.chat import ChatManager, ChatHistory
import openai as openaiClient


def get_client():
    """
    Retrieve the OpenAI client from the connection manager.

    Returns:
        openaiClient.OpenAI: The OpenAI client instance.
    """
    return ConnectionManager().get_connection("OpenAI").get_client()

def get_other_settings():
    """
    Retrieve other settings related to the OpenAI connection.

    Returns:
        dict: A dictionary containing other settings.
    """
    return ConnectionManager().get_connection("OpenAI").get_other_setting()

def require_login(func):
    """
    Decorator to ensure that the OpenAI connection is initialized before calling the function.

    Args:
        func (function): The function to be decorated.

    Returns:
        function: The wrapped function.
    """
    def wrapper(*args, **kwargs):
        if ConnectionManager().get_connection("OpenAI") is None:
            raise Exception("OpenAI connection not found. Please initialize the connection.")
        return func(*args, **kwargs)
    return wrapper

class OpenAIConnection():
    """
    Class to manage the connection to the OpenAI API.
    """
    def __init__(self, api_key: str, other_setting: dict, **kwargs):
        """
        Initialize the OpenAIConnection instance.

        Args:
            api_key (str): The API key for OpenAI.
            other_setting (dict): Additional settings for the connection.
        """
        self.api_key = api_key
        self.client = None
        self.other_setting = other_setting

    def init_connection(self):
        """
        Initialize the connection to the OpenAI API.
        """
        try: 
            self.client = openaiClient.OpenAI(api_key=self.api_key)
            print("OpenAI connection initialized successfully.")
        except Exception as e:
            raise Exception(f"Error initializing OpenAI connection: {e}")
        

    def get_client(self):
        """
        Retrieve the OpenAI client, initializing the connection if necessary.

        Returns:
            openaiClient.OpenAI: The OpenAI client instance.
        """
        if self.client is None:
            self.init_connection()
        return self.client
    
    def get_other_setting(self):
        """
        Retrieve other settings related to the OpenAI connection.

        Returns:
            dict: A dictionary containing other settings.
        """
        return self.other_setting


class utils:
    """
    Utility class for various helper functions.
    """
    def format_chat_to_openai_format(chat: ChatHistory):
        """
        Format the chat history to the OpenAI format.

        Args:
            chat (ChatHistory): The chat history.

        Returns:
            list: A list of formatted messages.
        """
        messages = []
        for message in chat:
            content = [{"type": "text", "text": message['content']}]
            if message['images']:
                for image in message['images']:
                    if image.startswith("http"):
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": image}
                        })
                    else:
                        raise ValueError("Image must be a URL for OpenAI")
            messages.append({"role": message['role'], "content": content})
        return messages
    


class OpenAIEmbeddingModels():
    """
    Class to manage OpenAI embedding models.
    """

    # https://github.com/openai/CLIP
    class OpenAIclipVIT32(BaseModel):
        """
        Model configuration for OpenAI CLIP ViT-B/32.
        """
        model_name: str = "openai.clip-vit-base-patch32"
        model_settings: dict = Field(default_factory=lambda: {
            "device": "torch device: mps, cpu, cuda"
        })
        config: Union[dict, None] = Field(default_factory=lambda: {
            "variant": "clip",
            'model_name': 'ViT-B/32'
        })
        class ModelInputSchema(BaseModel):
            """
            Schema for model input.
            """
            input_document: Union[Document, TextChunk, None] = None

        model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    
class CLIPmodel():
    """
    Class to manage the CLIP model and create embeddings.
    """
    _model_cache = {}
    # ensure the model is loaded only once
    def __init__(self, model: str, model_settings: dict = None):
        """
        Initialize the CLIPmodel instance.

        Args:
            model (str): The model name.
            model_settings (dict, optional): Additional settings for the model.
        """
        self.model = ModelsBaseSettings().providers_list_map["OpenAI"]["embeddings_models"]["models_map"].get(model)().config['model_name']
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
        """
        Create embeddings using the CLIP model.

        Args:
            body_fields (dict): The input fields for the model.
            model_settings (dict, optional): Additional settings for the model.

        Returns:
            list: The generated embeddings.
        """
        input_document = body_fields.get('input_document')
        if input_document is None:
            raise ValueError("input_document is required for creating embeddings.")

        if isinstance(input_document, Document):
            text_input = input_document.content
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
            if image_input.startswith("http"):
                image = self.clip_processor(Image.open(save_img_to_temp_file(image_input, return_doc=False))).unsqueeze(0).to(self.device)
            else:
                image = self.clip_processor(Image.open(image_input)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embeddings = self.clip_model_pipe.encode_image(image)
            embeddings = embeddings.cpu().tolist()[0]
        else:
            raise ValueError("input_document must contain either text or image content.")
        return embeddings


def OpenAIEmbeddings(model: str, document: Document | TextChunk | str, model_settings: dict = None):
    """
    Generate embeddings using the specified OpenAI model.

    Args:
        model (str): The model name.
        document (Document | TextChunk | str): The input document.
        model_settings (dict, optional): Additional settings for the model.

    Returns:
        Vector: The generated embeddings as a Vector object.
    """
    output_metadata = {
        'model': model,
        'model_settings': model_settings,
        'timestamp': datetime.now()
    }
    
    if model not in ModelsBaseSettings().providers_list_map["OpenAI"]["embeddings_models"]["models_map"]:
        raise ValueError(f"Model {model} not found in OpenAI settings.")
    embedding_model_setting = ModelsBaseSettings().providers_list_map["OpenAI"]["embeddings_models"]["models_map"].get(model).ModelInputSchema
    if embedding_model_setting is None:
        raise ValueError(f"Model {model} not found in settings. Please check the model name.")

    body_fields = embedding_model_setting.model_fields
    body_fields['input_document'] = document

    try:
        embedding_model_setting.model_validate(body_fields) # validate with pydantic
    except Exception:
        raise ValueError(f"Error. please check if the settings are correct. use get_model_settings(model) to check the correct settings.")

    if ModelsBaseSettings().providers_list_map["OpenAI"]["embeddings_models"]["models_map"].get(model)().config['variant'] == 'clip':
        # cuz our function takes the document object directly
        embeddings =  CLIPmodel(model).create_embeddings_clip(body_fields, model_settings)

    content = document.image if isinstance(document, Document) and document.image else document.content if isinstance(document, Document) else document.chunk if isinstance(document, TextChunk) else None

    return Vector(
        document_id=document.id,
        vector=embeddings,
        size=len(embeddings),
        data_type="text" if isinstance(document, TextChunk) else "image",
        content=content,
        metadata=output_metadata
    )


class OpenaiModel():
    """
    Class to manage interactions with the OpenAI model.
    """
    def __init__(self, model_name: str, system_prompt: str = None, retriever: object = None, agent_prompt: BaseModel = None):
        """
        Initialize the OpenaiModel instance.

        Args:
            model_name (str): The name of the model.
            system_prompt (str, optional): The system prompt.
            retriever (object, optional): The retriever object.
            agent_prompt (BaseModel, optional): The agent prompt. (if you use RAG for example, because it is different from the system prompt)
        """
        self.model_name = model_name
        self.system_prompt = system_prompt # 
        self.retriever = retriever
        self.history = ChatManager() # initialize empty chat history
        # add system prompt
        if system_prompt:
            self.history.add_message(role='system', content=system_prompt)
        self.agent_prompt = agent_prompt
        self.openai_client = get_client()

    # note that OpenAI requires image url, and it has a specific chat format too, that's why we have a format_chat_to_openai_format function
    @require_login
    def chat(self, query: str, image: str = None):
        """
        Generate a chat response from the OpenAI model.

        Args:
            query (str): The query text.
            image (str, optional): The image URL.

        Returns:
            str: The generated response.
        """
        if self.retriever:
            context = []
            if query:
                context.extend(self.retriever(query_text=query))
            elif query and image:
                context.extend(self.retriever(query_image=image))
            if not self.agent_prompt:
                warnings.warn("agent_prompt is not defined.")
            prompt = self.agent_prompt(query=query, context=context)
            final_prompt = prompt.prompt
            # print('used docs: ', prompt.context_doc) access source if you want
            
            # add the prompt and the context to the chat history
            if prompt.images_list:
                self.history.add_message(role='user', content=final_prompt, images=prompt.images_list)
            else:
                self.history.add_message(role='user', content=final_prompt)

            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=utils.format_chat_to_openai_format(self.history.get_history())
            )
        else:
            images = [utils.save_img_to_temp_file(image, return_doc=False) for image in image]
            self.history.add_message(role='user', content=query, images=images)
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=utils.format_chat_to_openai_format(self.history.get_history())
            )

        self.history.add_message(role='assistant', content=response.choices[0].message.content)
        return response.choices[0].message.content