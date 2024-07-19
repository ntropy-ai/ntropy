from ntropy_ai.core.utils.base_format import Document
from typing import List, Union
import json

class JsonLoader:
    def __init__(self, file_path: str, text_content_path: str = None, image_path: str = None, exclude: bool = False, image_suffix: Union[str, List[str]] = None):
        """
        Initializes the JsonLoader with the given file path and optional paths for text content and images.

        Args:
            file_path (str): The path to the JSON file.
            text_content_path (str, optional): The path to the text content within the JSON structure. Defaults to None.
            image_path (str, optional): The path to the image content within the JSON structure. Defaults to None.
            exclude (bool, optional): Whether to exclude non-matching paths. Defaults to False.
            image_suffix (Union[str, List[str]], optional): The suffix or list of suffixes to filter image files. Defaults to common image extensions.
        """
        self.file_path = file_path
        self.text_content_path = text_content_path
        self.image_path = image_path
        self.exclude = exclude
        self.image_suffix = image_suffix or ['.png', '.jpeg', '.jpg', '.gif', '.bmp']

    def load(self) -> List[Document]:
        """
        Loads the JSON file and processes its data into a list of Document objects.

        Returns:
            List[Document]: A list of Document objects created from the JSON data.
        """
        with open(self.file_path, 'r') as f:
            data = json.load(f)
    
        documents = self._process_data(data)
        return documents

    def _process_data(self, data: Union[dict, list], path: str = '') -> List[Document]:
        """
        Recursively processes the JSON data to create Document objects.

        Args:
            data (Union[dict, list]): The JSON data to process.
            path (str, optional): The current path within the JSON structure. Defaults to ''.

        Returns:
            List[Document]: A list of Document objects created from the JSON data.
        """
        documents = []
        if isinstance(data, dict):
            # Process each key-value pair in the dictionary
            for key, value in data.items():
                new_path = f"{path}/{key}" if path else key
                documents.extend(self._process_data(value, new_path))
        elif isinstance(data, list):
            # Process each item in the list
            for index, item in enumerate(data):
                new_path = f"{path}/{index}"
                documents.extend(self._process_data(item, new_path))
        else:
            # Process individual data items
            if self._matches_path(path, self.image_path) and isinstance(data, str) and any(data.lower().endswith(suffix) for suffix in self.image_suffix):
                # Create a Document for image data
                doc = Document(content=None, image=data, metadata={'path': path, 'type': 'image'})
                documents.append(doc)
            elif self._matches_path(path, self.text_content_path):
                # Create a Document for text content
                doc = Document(content=str(data), metadata={'path': path, 'type': 'text'})
                documents.append(doc)
            else:
                if not self.exclude:
                    # Create a Document for other data if not excluded
                    if any(data.lower().endswith(suffix) for suffix in self.image_suffix):
                        doc = Document(content=None, image=data, metadata={'path': path, 'type': 'image'})
                    else:
                        doc = Document(content=str(data), metadata={'path': path, 'type': 'text'})
                    documents.append(doc)
        return documents

    def _matches_path(self, current_path: str, target_path: str) -> bool:
        """
        Checks if the current path matches the target path.

        Args:
            current_path (str): The current path within the JSON structure.
            target_path (str): The target path to match against.

        Returns:
            bool: True if the current path matches the target path, False otherwise.
        """
        if not target_path:
            return False
        return current_path.endswith(target_path)