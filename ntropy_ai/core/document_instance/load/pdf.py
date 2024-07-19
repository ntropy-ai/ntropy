import pymupdf
from ntropy_ai.core.utils.base_format import Document
from typing import List
import os
import tempfile

class PDFLoader:
    """
    PDFLoader is a utility class for extracting text and images from PDF files.

    Attributes:
        file_path (str): The path to the PDF file.
        output_img_path (str): The directory path where extracted images will be saved.
        pdf (pymupdf.Document): The opened PDF document.
    """
    def __init__(self, file_path: str, output_img_path: str = None):
        """
        Initializes the PDFLoader with the given file path and optional output image path.

        Args:
            file_path (str): The path to the PDF file.
            output_img_path (str, optional): The directory path where extracted images will be saved. Defaults to a temporary directory.
        """
        self.file_path = file_path
        
        if output_img_path is None:
            self.output_img_path = tempfile.mkdtemp()
        else:
            self.output_img_path = output_img_path
        
        self.pdf = pymupdf.open(file_path)

    def extract_text(self) -> List[Document]:
        """
        Extracts text content from each page of the PDF and returns it as a list of Document objects.

        Returns:
            List[Document]: A list of Document objects containing text content from each page of the PDF.
        """
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

    def extract_images(self) -> List[Document]:
        """
        Extracts images from each page of the PDF and returns them as a list of Document objects.
        The images are saved to the specified output directory.

        Returns:
            List[Document]: A list of Document objects containing image paths from each page of the PDF.
        """
        documents: List[Document] = []
        for page_number in range(len(self.pdf)):
            page = self.pdf[page_number]

            # Get all images on the current page
            images = page.get_images()
            for image_index, image in enumerate(images, start=1):
                xref = image[0]
                pixmap = pymupdf.Pixmap(self.pdf, xref)

                # Convert CMYK or grayscale images to RGB
                if pixmap.n - pixmap.alpha > 3:
                    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pixmap)
                
                # Create the output directory if it doesn't exist
                if not os.path.exists(self.output_img_path):
                    os.makedirs(self.output_img_path)
                
                # Save the image to the specified path
                image_path = f"{self.output_img_path}/image_{page_number}_{image_index}.png"
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
