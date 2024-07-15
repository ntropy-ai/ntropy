

import ollama
from pydantic import BaseModel
from ntropy.core import utils
from ntropy.core.utils.base_format import Document
from typing import List
import warnings
from ntropy.core.utils.chat import ChatManager, ChatMessage


class OllamaModelsInput():
    model_name: str
    model_settings: dict = None
    prompt: str
    context: List[Document]
    images: List[str] = None
    prompt: str = None


def list_models():
    models = ollama.list()
    return [model['name'] for model in models['models'] if 'clip' in model['details']['families']]

class OllamaModel():
    def __init__(self, model_name: str, system_prompt: str = None, retriever: object = None, agent_prompt: BaseModel = None):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.retriever = retriever
        self.history = ChatManager() # initialize empty chat history
        # add system prompt
        if system_prompt:
            self.history.add_message(role='system', content=system_prompt)
        self.agent_prompt = agent_prompt



    # completion
    def generate(self, query: str = None, image: str = None, top_k: int = None):
        if self.system_prompt:
            warnings.warn("system prompt is only supported for chat methods")
        if self.retriever:
            context = []
            if query:
                context.extend(self.retriever(query_text=query, top_k=top_k))
            elif query and image:
                context.extend(self.retriever(query_image=image, top_k=top_k))
            if not self.agent_prompt:
                warnings.warn("agent_prompt is not defined.")
            prompt = self.agent_prompt(query=query, context=context)
            final_prompt = prompt.prompt
            if prompt.images_list:
                response = ollama.generate(model=self.model_name, prompt=final_prompt, images=prompt.images_list)
            elif image:
                image_path = utils.save_img_to_temp_file(image, return_doc=False)
                response = ollama.generate(model=self.model_name, prompt=final_prompt, images=[image_path])
            else:
                response = ollama.generate(model=self.model_name, prompt=final_prompt)

            self.history.add_message(role='user', content=final_prompt, images=prompt.images_list)
            self.history.add_message(role='assistant', content=response['response'])
            return response['response']
        
        else:
            if image:
                image_path = utils.save_img_to_temp_file(image, return_doc=False)
                self.history.add_message(role='user', content=query, images=[image_path])
                response = ollama.generate(model=self.model_name, prompt=query, images=[image_path])
            else:
                self.history.add_message(role='user', content=query)
                response = ollama.generate(model=self.model_name, prompt=query)
            self.history.add_message(role='assistant', content=response['response'])
            return response['response']
        
    
    # chat
    def chat(self, query: str, image: str = None, top_k: int = None):
        if self.retriever:
            context = []
            if query:
                context.extend(self.retriever(query_text=query, top_k=top_k))
            elif query and image:
                context.extend(self.retriever(query_image=image, top_k=top_k))
            if not self.agent_prompt:
                warnings.warn("agent_prompt is not defined.")
            prompt = self.agent_prompt(query=query, context=context)
            final_prompt = prompt.prompt
            
            # add the prompt and the context to the chat history
            if prompt.images_list:
                self.history.add_message(role='user', content=final_prompt, images=prompt.images_list)
            else:
                self.history.add_message(role='user', content=final_prompt)
            
            response = ollama.chat(model=self.model_name, messages=self.history.get_history())
        else:
            images = [utils.save_img_to_temp_file(image, return_doc=False) for image in image]
            self.history.add_message(role='user', content=query, images=images)
            response = ollama.chat(model=self.model_name, messages=self.history.get_history())

        self.history.add_message(role='assistant', content=response['message']['content'])
        return response['message']['content']
