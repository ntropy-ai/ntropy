import json
import base64
from ntropy_ai.core.utils.chat import ChatHistory, ChatManager
from ntropy_ai.core.utils import ensure_local_file
from ntropy_ai.core.utils.connections_manager import ConnectionManager
import os
import anthropic as AnthropicClient
from pydantic import BaseModel
import warnings
import logging
from ntropy_ai.core.utils.settings import logger

class AnthropicConnection():
    """
    Class to manage the connection to the Anthropic API.
    """
    def __init__(self, api_key: str, other_setting: dict, **kwargs):
        """
        Initialize the AnthropicConnection instance.

        Args:
            api_key (str): The API key for Anthropic.
            other_setting (dict): Additional settings for the connection.
        """
        self.api_key = api_key
        self.client = None
        self.other_setting = other_setting

    def init_connection(self):
        """
        Initialize the connection to the Anthropic API.
        """
        try: 
            self.client = AnthropicClient.Anthropic(api_key=self.api_key)
            logger.info("Anthropic connection initialized successfully.")
        except Exception as e:
            raise Exception(f"Error initializing Anthropic connection: {e}")
        

    def get_client(self):
        """
        Retrieve the Anthropic client, initializing the connection if necessary.

        Returns:
            anthropic.Anthropic: The Anthropic client instance.
        """
        if self.client is None:
            self.init_connection()
        return self.client
    
    def get_other_setting(self):
        """
        Retrieve other settings related to the Anthropic connection.

        Returns:
            dict: A dictionary containing other settings.
        """
        return self.other_setting



def get_client():
    """
    Retrieve the Anthropic client from the connection manager.

    Returns:
        anthropic.Anthropic: The Anthropic client instance.
    """
    return ConnectionManager().get_connection("Anthropic").get_client()


def require_login(func):
    """
    Decorator to ensure that the Anthropic connection is initialized before calling the function.

    Args:
        func (function): The function to be decorated.

    Returns:
        function: The wrapped function.
    """
    def wrapper(*args, **kwargs):
        if ConnectionManager().get_connection("Anthropic") is None:
            raise Exception("Anthropic connection not found. Please initialize the connection.")
        return func(*args, **kwargs)
    return wrapper


class utils:

    """
    format chat to anthropic format
    """
    def format_chat_to_anthropic_format(chat: ChatHistory):
        messages = []
        for message in chat:
            if message['role'] in ['user', 'assistant']:
                content = [{"type": "text", "text": message['content']}]
                if message['images']:
                    for image in message['images']:
                        if image.startswith("http"):
                            local_file_path = ensure_local_file(image)
                            content.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": f"image/{os.path.splitext(image)[1][1:]}", # need to make sure orioginal image is png
                                    "data": base64.b64encode(open(local_file_path, "rb").read()).decode('utf-8')
                                }
                            })
                        else:
                            try:
                                content.append({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": f"image/{os.path.splitext(image)[1][1:]}",
                                        "data": base64.b64encode(open(image, "rb").read()).decode('utf-8')
                                    }
                                })
                            except Exception as e:
                                raise Exception(f"Exception {e} while processing image, make sure file is a valid image")
                messages.append({"role": message['role'], "content": content})
        return messages
    



class AnthropicModel():
    """
    class to manage Anthropic models

    support:
    - claude-3-5-sonnet-20240620
    - claude-3-opus-20240229
    - claude-3-sonnet-20240229
    - claude-3-haiku-20240307

    no tool calling for now

    """

    def __init__(
        self, 
        model_name: str, 
        system_prompt: str = None, 
        retriever: object = None, 
        agent_prompt: BaseModel = None):
        """
        Initialize the AnthropicModel instance.

        Args:
            model_name (str): The name of the model.
            system_prompt (str): The system prompt for the model.
            retriever (object): The retriever object.
            agent_prompt (BaseModel): The agent prompt.
        """

        self.model_name = model_name
        self.system_prompt = system_prompt # 
        self.retriever = retriever
        self.history = ChatManager() # initialize empty chat history
        # add system prompt
        if system_prompt:
            self.history.add_message(role='system', content=system_prompt)
        self.agent_prompt = agent_prompt
        self.anthropic_client = get_client()

    @require_login
    def chat(self, query: str, images: list = []):
        """
        chat with the model
        """
        if self.retriever:
            context = []
            if query:
                context.extend(self.retriever(query_text=query))
            elif query and images:
                if len(images) > 1:
                    warnings.warn("Only one image is supported for now.")
                context.extend(self.retriever(query_image=images[0]))
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

            response = self.anthropic_client.messages.create(
                model=self.model_name,
                max_tokens=1024,
                messages=utils.format_chat_to_anthropic_format(self.history.get_history())
            )
        
  
        else:
            self.history.add_message(role='user', content=query, images=images)
            response = self.anthropic_client.messages.create(
                model=self.model_name,
                max_tokens=1024,
                messages=utils.format_chat_to_anthropic_format(self.history.get_history())
            )


        self.history.add_message(role='assistant', content=response.content[0].text)
        return response.content[0].text