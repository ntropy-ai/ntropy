
from ntropy.core.utils.base_format import Document
from ntropy.core.utils.chat import ChatMessage
from typing import List
import os
from PIL import Image
import tempfile
import requests

class RagPrompt():
    def __init__(self, query: str, context: List[Document]): # image is Document format
        self.images_list = []
        self.doc_list = []
        for doc in context:
            if doc.data_type == 'image':
                self.images_list.append(doc.content)
                if doc.content.startswith('http'):
                    with tempfile.NamedTemporaryFile(suffix=".jpg") as temp_file:
                        img = Image.open(requests.get(doc.content, stream=True).raw)
                        img.save(temp_file.name)
                        temp_file.flush()
                        self.images_list.append(temp_file.name)
                else:
                    if os.path.exists(doc.content):
                        self.images_list.append(doc.content)
                    else:
                        raise ValueError(f"Image {doc.content} not found")
                    
            else:
                self.doc_list.append(doc.content)

        self.context_doc = context
        if self.images_list:
            #self.prompt = ChatMessage(role='system', content=f"Using this data: {' '.join(self.doc_list)} and the images. Respond to this prompt: {query}", images=self.images_list)
            self.prompt = f"Using this data: {' '.join(self.doc_list)} and the images. Respond to this prompt: {query}"
        else:
            self.prompt = f"Using this data: {' '.join(self.doc_list)}. Respond to this prompt: {query}"

