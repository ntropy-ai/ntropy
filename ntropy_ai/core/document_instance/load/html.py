from bs4 import BeautifulSoup
from ntropy_ai.core.utils.base_format import Document
from typing import List

class HtmlFileLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        with open(self.file_path, 'r') as file:
            soup = BeautifulSoup(file, 'html.parser')
            text_content = soup.get_text()
            img_tags = soup.find_all('img')
            image_urls = [img_tag['src'] for img_tag in img_tags if img_tag and 'src' in img_tag.attrs]
            return [
                Document(content=text_content, metadata={'type': 'text'}),
                *[Document(image=image_url, metadata={'source': image_url, 'type': 'image'}) for image_url in image_urls]
            ]