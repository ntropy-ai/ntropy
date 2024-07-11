import pymupdf
from ntropy.core.utils.base_format import Document
from typing import List
import os


class PDFLoader:
    def __init__(self, file_path: str, img_path: str):
        self.file_path = file_path
        self.img_path = img_path
        self.pdf = pymupdf.open(file_path)

    

    def extract_text(self):
        documents: List[Document] = []
        for page_number in range(len(self.pdf)):
            page = self.pdf[page_number]
            text_content = page.get_text().encode('utf-8')
            documents.append(
                Document(
                    page_number=page_number,
                    content=text_content,
                    image=None,
                    metadata={"type": "text"}
                )
            )
        return documents

    def extract_images(self):
        documents: List[Document] = []
        for page_number in range(len(self.pdf)):
            page = self.pdf[page_number]

            images = page.get_images()
            for image_index, image in enumerate(images, start=1):
                xref = image[0]
                pixmap = pymupdf.Pixmap(self.pdf, xref)

                if pixmap.n - pixmap.alpha > 3:
                    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pixmap)
                if not os.path.exists(self.img_path):
                    os.makedirs(self.img_path)
                image_path = f"{self.img_path}/image_{page_number}_{image_index}.png"
                pixmap.save(image_path)

                documents.append(
                    Document(
                        page_number=page_number,
                        content=None,
                        image=image_path,
                        metadata={"type": "image"}
                    )
                )
        return documents
