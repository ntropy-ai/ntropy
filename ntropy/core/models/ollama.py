

import ollama
from pydantic import BaseModel, Field, ConfigDict
from ntropy.core import utils
from ntropy.core.utils.base_format import Document
from typing import List
from PIL import Image
import requests
import tempfile

class OllamaModelsInput():
    model_name: str
    model_settings: dict = None
    prompt: str
    context: List[Document]
    images: List[str] = None
    prompt: str = None



class OllamaModel():
    def __init__(self, model_name: str, prompt: BaseModel = None, retriever: object = None):
        self.model_name = model_name
        self.prompt = prompt
        self.retriever = retriever

    def generate(self, query: str = None, image: str = None, top_k: int = None):
        if self.retriever:
            context = []
            if query:
                context.extend(self.retriever(query_text=query, top_k=top_k))
            elif query and image:
                context.extend(self.retriever(query_image=image, top_k=top_k))
            prompt = self.prompt(query=query, context=context)
            final_prompt = prompt.prompt
            print(prompt.images_list)
            if prompt.images_list:
                response = ollama.generate(model=self.model_name, prompt=final_prompt, images=prompt.images_list)
            elif image:
                image_path = utils.save_img_to_temp_file(image, return_doc=False)
                response = ollama.generate(model=self.model_name, prompt=final_prompt, images=[image_path])
            else:
                response = ollama.generate(model=self.model_name, prompt=final_prompt)
            return response['response']
        
        else:
            if image:
                image_path = utils.save_img_to_temp_file(image, return_doc=False)
                response = ollama.generate(model=self.model_name, prompt=query, images=[image_path])
            else:
                response = ollama.generate(model=self.model_name, prompt=query)
            return response['response']