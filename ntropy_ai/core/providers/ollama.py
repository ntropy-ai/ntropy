import ollama
from pydantic import BaseModel
from ntropy_ai.core.utils import ensure_local_file, save_img_to_temp_file
from ntropy_ai.core.utils.settings import logger
from ntropy_ai.core.utils.chat import ChatManager
import logging


def list_models():
    """
    Lists all available models that belong to the 'clip' family.

    Returns:
        List[str]: A list of model names.
    """
    models = ollama.list()
    return [model['name'] for model in models['models'] if 'clip' in model['details']['families']]

class OllamaModel():
    """
    A class to represent an Ollama model.

    Attributes:
        model_name (str): The name of the model.
        system_prompt (str): The system prompt for the model.
        retriever (object): The retriever object for the model.
        agent_prompt (BaseModel): The agent prompt for the model.
        history (ChatManager): The chat history manager.
    """
    def __init__(self, model_name: str, system_prompt: str = None, retriever: object = None, agent_prompt: BaseModel = None):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.retriever = retriever
        self.history = ChatManager() # initialize empty chat history
        # add system prompt
        if system_prompt:
            self.history.add_message(role='system', content=system_prompt)
        self.agent_prompt = agent_prompt

    def generate(self, query: str = None, image: str = None):
        """
        Generates a response from the model based on the query and image.

        usage: 
        - generate(query="what is the meaning of life?", image="https://upload.wikimedia.org/wikipedia/commons/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg")
        - generate(query="what is the meaning of life?", image="path/to/image.jpg")
        -> return text

        Args:
            query (str): The query text.
            image (str): The image path or URL.

        Returns:
            str: The generated response.
        """
        if self.system_prompt:
            logger.warning("system prompt is only supported for chat methods")
        if self.retriever:
            context = []
            if query:
                context.extend(self.retriever(query_text=query))
            elif query and image:
                context.extend(self.retriever(query_image=image))
            if not self.agent_prompt:
                logger.warning("agent_prompt is not defined.")
            prompt = self.agent_prompt(query=query, context=context)
            final_prompt = prompt.prompt
            # print('used docs: ', prompt.context_doc) access source if you want
            if prompt.images_list:
                images_list = [ensure_local_file(image) for image in prompt.images_list]
                response = ollama.generate(model=self.model_name, prompt=final_prompt, images=images_list)
            elif image:
                image_path = ensure_local_file(image)
                response = ollama.generate(model=self.model_name, prompt=final_prompt, images=[image_path])
            else:
                response = ollama.generate(model=self.model_name, prompt=final_prompt)

            self.history.add_message(role='user', content=final_prompt, images=[ensure_local_file(image) for image in prompt.images_list])
            self.history.add_message(role='assistant', content=response['response'])
            return response['response']
        
        else:
            if image:
                image_path = ensure_local_file(image)
                self.history.add_message(role='user', content=query, images=[image_path])
                response = ollama.generate(model=self.model_name, prompt=query, images=[image_path])
            else:
                self.history.add_message(role='user', content=query)
                response = ollama.generate(model=self.model_name, prompt=query)
            self.history.add_message(role='assistant', content=response['response'])
            return response['response']
        
    
    def chat(self, query: str, image: str = None):
        """
        Engages in a chat with the model based on the query and image.

        Args:
            query (str): The query text.
            image (str): The image path or URL.

        usage: 
        - chat(query="what is the meaning of life?", image="https://upload.wikimedia.org/wikipedia/commons/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg")
        - chat(query="what is the meaning of life?", image="path/to/image.jpg")
        -> return text

        Returns:
            str: The chat response.
        """
        if self.retriever:
            context = []
            if query:
                context.extend(self.retriever(query_text=query))
            elif query and image:
                context.extend(self.retriever(query_image=image))
            if not self.agent_prompt:
                logger.warning("agent_prompt is not defined.")
            prompt = self.agent_prompt(query=query, context=context)
            final_prompt = prompt.prompt
            # print('used docs: ', prompt.context_doc) access source if you want
            
            # add the prompt and the context to the chat history
            if prompt.images_list:
                images_list = [ensure_local_file(image) for image in prompt.images_list]
                self.history.add_message(role='user', content=final_prompt, images=images_list)
            else:
                self.history.add_message(role='user', content=final_prompt)
            
            response = ollama.chat(model=self.model_name, messages=self.history.get_history())
        else:
            images = [ensure_local_file(image)] if image else []
            self.history.add_message(role='user', content=query, images=images)
            response = ollama.chat(model=self.model_name, messages=self.history.get_history())

        self.history.add_message(role='assistant', content=response['message']['content'])
        return response['message']['content']

    def schat(self, query: str, image: str = None):
        """
        Engages in a streaming chat with the model based on the query and image.

        Args:
            query (str): The query text.
            image (str): The image path or URL.

        usage: 
        - stream = schat(query="what is the meaning of life?", image="https://upload.wikimedia.org/wikipedia/commons/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg")
        - stream = schat(query="what is the meaning of life?", image="path/to/image.jpg")
        -> for chunk in stream:
            print(chunk)


        Yields:
            str: The streaming chat response.
        """
        final_res = ''
        if self.retriever:
            context = []
            if query:
                context.extend(self.retriever(query_text=query))
            elif query and image:
                context.extend(self.retriever(query_image=image))
            if not self.agent_prompt:
                logger.warning("agent_prompt is not defined.")
            prompt = self.agent_prompt(query=query, context=context)
            final_prompt = prompt.prompt
            # print('used docs: ', prompt.context_doc) access source if you want
            
            # add the prompt and the context to the chat history
            if prompt.images_list:
                self.history.add_message(role='user', content=final_prompt, images=[ensure_local_file(image) for image in prompt.images_list])
            else:
                self.history.add_message(role='user', content=final_prompt)
            
            stream = ollama.chat(model=self.model_name, messages=self.history.get_history(), stream=True)
            for chunk in stream:
                final_res += chunk['message']['content']
                yield chunk['message']['content'] 
        else:
            if image:
                images = [ensure_local_file(image) for image in image]
                self.history.add_message(role='user', content=query, images=images)
            else:
                self.history.add_message(role='user', content=query)
            stream = ollama.chat(model=self.model_name, messages=self.history.get_history(), stream=True)
            for chunk in stream:
                final_res += chunk['message']['content']
                yield chunk['message']['content'] 

        self.history.add_message(role='assistant', content=final_res)
        
    def sgenerate(self, query: str, image: str = None):
        """
        Generates a streaming response from the model based on the query and image.

        Args:
            query (str): The query text.
            image (str): The image path or URL.

        usage: 
        - stream = sgenerate(query="what is the meaning of life?", image="https://upload.wikimedia.org/wikipedia/commons/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg")
        - stream = sgenerate(query="what is the meaning of life?", image="path/to/image.jpg")
        -> for chunk in stream:
            print(chunk)

        Yields:
            str: The streaming generated response.
        """
        final_res = ''
        if self.system_prompt:
            logger.warning("system prompt is only supported for chat methods")
        if self.retriever:
            context = []
            if query:
                context.extend(self.retriever(query_text=query))
            elif query and image:
                context.extend(self.retriever(query_image=image))
            if not self.agent_prompt:
                logger.warning("agent_prompt is not defined.")
            prompt = self.agent_prompt(query=query, context=context)
            final_prompt = prompt.prompt
            # print('used docs: ', prompt.context_doc) access source if you want
            if prompt.images_list:
                images_list = [ensure_local_file(image) for image in prompt.images_list]
                stream = ollama.generate(model=self.model_name, prompt=final_prompt, images=images_list, stream=True)
                for chunk in stream:
                    final_res += chunk['response']
                    yield chunk['response']
            elif image:
                image_path = ensure_local_file(image)
                stream = ollama.generate(model=self.model_name, prompt=final_prompt, images=[image_path], stream=True)
                for chunk in stream:
                    final_res += chunk['response']
                    yield chunk['response']
            else:
                stream = ollama.generate(model=self.model_name, prompt=final_prompt, stream=True)
                for chunk in stream:
                    final_res += chunk['response']
                    yield chunk['response']

            self.history.add_message(role='user', content=final_prompt, images=[ensure_local_file(image) for image in prompt.images_list])
            self.history.add_message(role='assistant', content=final_res)
        
        else:
            if image:
                image_path = ensure_local_file(image)
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