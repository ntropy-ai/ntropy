
from ntropy.core.utils.base_format import Vector
from ntropy.core.utils import save_img_to_temp_file
from typing import List
import os
from PIL import Image
import tempfile
import requests

class RagPrompt():
    """
    prompt:
    Using this data: DOCUMENT_TEXTS and the images. Respond to this prompt: USER_QUERY

    access the prompt and images to pass the model with RagPrompt.images_list and RagPrompt.prompt
    """
    def __init__(self, query: str, context: List[Vector]):
        self.doc_list = []
        self.images_list = []
        for doc in context:
            if doc.data_type == 'image':
                if not doc.content.startswith('http'):
                    self.images_list.append(save_img_to_temp_file(doc.content, return_doc=False))
                else:
                    self.images_list.append(doc.content)
            else:
                self.doc_list.append(doc.content)

        self.context_doc = context
        if self.images_list:
            self.prompt = f"Using this data: {' '.join(self.doc_list)} and the images. Respond to this prompt: {query}"
        else:
            self.prompt = f"Using this data: {' '.join(self.doc_list)}. Respond to this prompt: {query}"

