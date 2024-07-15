
from ntropy.core.utils.base_format import Vector
from ntropy.core.utils import save_img_to_temp_file
from typing import List
import os
from PIL import Image
import tempfile
import requests

class RagPrompt():
    def __init__(self, query: str, context: List[Vector]):
        self.doc_list = []
        self.images_list = []
        for doc in context:
            if doc.data_type == 'image':
                if not doc.content.startswith('http'):
                    self.images_list.append(save_img_to_temp_file(doc.content, return_doc=False))
                else:
                    #if os.path.exists(doc.content):
                    self.images_list.append(doc.content)
                    #else:
                    #    raise ValueError(f"Image {doc.content} not found")
                    
            else:
                self.doc_list.append(doc.content)

        self.context_doc = context
        if self.images_list:
            #self.prompt = ChatMessage(role='system', content=f"Using this data: {' '.join(self.doc_list)} and the images. Respond to this prompt: {query}", images=self.images_list)
            self.prompt = f"Using this data: {' '.join(self.doc_list)} and the images. Respond to this prompt: {query}"
        else:
            self.prompt = f"Using this data: {' '.join(self.doc_list)}. Respond to this prompt: {query}"

