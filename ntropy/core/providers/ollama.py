import ollama
from pydantic import BaseModel
from ntropy.core import utils
from ntropy.core.utils.base_format import Document
from typing import List
import warnings
from ntropy.core.utils.chat import ChatManager, ChatMessage
from ntropy.core.utils import save_img_to_temp_file

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

class utils:
    def ensure_local_image(image: str):
        """
        Ensure the image is local and not a URL because ollama does not support URLs
        """
        if image.startswith('http'):
            return save_img_to_temp_file(image, return_doc=False)
        else:
            return image
        
        

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
    def generate(self, query: str = None, image: str = None):
        if self.system_prompt:
            warnings.warn("system prompt is only supported for chat methods")
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
            if prompt.images_list:
                images_list = [utils.ensure_local_image(image) for image in prompt.images_list]
                response = ollama.generate(model=self.model_name, prompt=final_prompt, images=images_list)
            elif image:
                image_path = utils.ensure_local_image(image)
                response = ollama.generate(model=self.model_name, prompt=final_prompt, images=[image_path])
            else:
                response = ollama.generate(model=self.model_name, prompt=final_prompt)

            self.history.add_message(role='user', content=final_prompt, images=[utils.ensure_local_image(image) for image in prompt.images_list])
            self.history.add_message(role='assistant', content=response['response'])
            return response['response']
        
        else:
            if image:
                image_path = utils.ensure_local_image(image)
                self.history.add_message(role='user', content=query, images=[image_path])
                response = ollama.generate(model=self.model_name, prompt=query, images=[image_path])
            else:
                self.history.add_message(role='user', content=query)
                response = ollama.generate(model=self.model_name, prompt=query)
            self.history.add_message(role='assistant', content=response['response'])
            return response['response']
        
    
    # chat
    def chat(self, query: str, image: str = None):
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
                images_list = [utils.ensure_local_image(image) for image in prompt.images_list]
                self.history.add_message(role='user', content=final_prompt, images=images_list)
            else:
                self.history.add_message(role='user', content=final_prompt)
            
            response = ollama.chat(model=self.model_name, messages=self.history.get_history())
        else:
            images = [utils.ensure_local_image(image) for image in image]
            self.history.add_message(role='user', content=query, images=images)
            response = ollama.chat(model=self.model_name, messages=self.history.get_history())

        self.history.add_message(role='assistant', content=response['message']['content'])
        return response['message']['content']

    # streaming, i prefer to separate stream and non stream
    def schat(self, query: str, image: str = None):
        final_res = ''
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
                self.history.add_message(role='user', content=final_prompt, images=[utils.ensure_local_image(image) for image in prompt.images_list])
            else:
                self.history.add_message(role='user', content=final_prompt)
            
            stream = ollama.chat(model=self.model_name, messages=self.history.get_history(), stream=True)
            for chunk in stream:
                final_res += chunk['message']['content']
                yield chunk['message']['content'] 
        else:
            if image:
                images = [utils.ensure_local_image(image) for image in image]
                self.history.add_message(role='user', content=query, images=images)
            else:
                self.history.add_message(role='user', content=query)
            stream = ollama.chat(model=self.model_name, messages=self.history.get_history(), stream=True)
            for chunk in stream:
                final_res += chunk['message']['content']
                yield chunk['message']['content'] 

        self.history.add_message(role='assistant', content=final_res)
        
    def sgenerate(self, query: str, image: str = None):
        final_res = ''
        if self.system_prompt:
            warnings.warn("system prompt is only supported for chat methods")
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
            if prompt.images_list:
                response = ollama.generate(model=self.model_name, prompt=final_prompt, images=prompt.images_list)
            elif image:
                image_path = utils.ensure_local_image(image)
                stream = ollama.generate(model=self.model_name, prompt=final_prompt, images=[image_path], stream=True)
                for chunk in stream:
                    final_res += chunk['response']
                    yield chunk['response']
            else:
                stream = ollama.generate(model=self.model_name, prompt=final_prompt, stream=True)
                for chunk in stream:
                    final_res += chunk['response']
                    yield chunk['response']

            self.history.add_message(role='user', content=final_prompt, images=[utils.ensure_local_image(image) for image in prompt.images_list])
            self.history.add_message(role='assistant', content=final_res)
        
        else:
            if image:
                image_path = utils.ensure_local_image(image)
                self.history.add_message(role='user', content=query, images=[image_path])
                stream = ollama.generate(model=self.model_name, prompt=query, images=[image_path], stream=True)
                for chunk in stream:
                    final_res += chunk['response']
                    yield chunk['response']
            else:
                self.history.add_message(role='user', content=query)
                stream = ollama.generate(model=self.model_name, prompt=query, stream=True)
                for chunk in stream:
                    final_res += chunk['response']
                    yield chunk['response']
            self.history.add_message(role='assistant', content=final_res)