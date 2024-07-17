from ntropy_ai.core.utils.base_format import Document
from typing import List
from ntropy_ai.core.utils.base_format import TextChunk, Document
import re


def BasicTextChunk(chunk_size: int, document: Document) -> List[TextChunk]:
    """
    The chunking method splits the document's text content into smaller chunks of a specified size without any context awareness.

    Parameters:
    - chunk_size (int): The size of each chunk in terms of the number of characters.
    - document (Document): The document to be chunked, which contains the text content to be split.
    
    Returns:
    - List[TextChunk]: A list of TextChunk objects, each representing a chunk of the original document's text.
    """
    chunk_list = []
    text = document.content
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        text_chunk = TextChunk(
            chunk=chunk, 
            chunk_number=i // chunk_size, 
            document_id=document.id,
            metadata={
                "type": "text",
                "page_number": document.page_number,
            }
        )
        chunk_list.append(text_chunk)
    return chunk_list


def RecursiveTextChunk(chunk_size: int, document: Document, start_index: int = 0, chunk_list: List[TextChunk] = None) -> List[TextChunk]:
    """
    Splits the document's text content into smaller chunks of a specified size using a recursive approach.

    Parameters:
    - chunk_size (int): The size of each chunk in terms of the number of characters.
    - document (Document): The document to be chunked, which contains the text content to be split.
    - start_index (int): The starting index for the current chunk.
    - chunk_list (List[TextChunk]): The list of TextChunk objects created so far.
    
    Returns:
    - List[TextChunk]: A list of TextChunk objects, each representing a chunk of the original document's text.
    """
    if chunk_list is None:
        chunk_list = []

    text = document.content
    if start_index >= len(text):
        return chunk_list

    chunk = text[start_index:start_index + chunk_size]
    text_chunk = TextChunk(
        chunk=chunk, 
        chunk_number=start_index // chunk_size, 
        document_id=document.id,
        metadata={
            "type": "text",
            "page_number": document.page_number,
        }
    )
    chunk_list.append(text_chunk)

    return RecursiveTextChunk(chunk_size, document, start_index + chunk_size, chunk_list)


def SentenceAwareChunk(chunk_size: int, document: Document) -> List[TextChunk]:
    """
    Splits the document's text content into smaller chunks of a specified size without breaking sentences.
    note: the chunk size wont be equal

    Parameters:
    - chunk_size (int): The size of each chunk in terms of the number of characters.
    - document (Document): The document to be chunked, which contains the text content to be split.
    
    Returns:
    - List[TextChunk]: A list of TextChunk objects, each representing a chunk of the original document's text.
    """
    chunk_list = []
    text = document.content
    sentences = re.split(r'(?<=[.!?]) +', text)
    current_chunk = ""
    chunk_number = 0

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
        else:
            text_chunk = TextChunk(
                chunk=current_chunk.strip(), 
                chunk_number=chunk_number, 
                document_id=document.id,
                metadata={
                    "type": "text",
                    "page_number": document.page_number,
                }
            )
            chunk_list.append(text_chunk)
            chunk_number += 1
            current_chunk = sentence + " "

    if current_chunk:
        text_chunk = TextChunk(
            chunk=current_chunk.strip(), 
            chunk_number=chunk_number, 
            document_id=document.id,
            metadata={
                "type": "text",
                "page_number": document.page_number,
            }
        )
        chunk_list.append(text_chunk)
    return chunk_list
