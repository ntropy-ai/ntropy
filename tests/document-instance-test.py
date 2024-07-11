import pytest
from ntropy.core.document_instance.load.pdf import PDFLoader
from ntropy.core.document_instance.process.chunk_text import BasicTextChunk, RecursiveTextChunk, SentenceAwareChunk
from ntropy.core.utils.base_format import Document, TextChunk
import os
import requests


@pytest.fixture
def sample_pdf(tmp_path):
    # Download a sample PDF file for testing
    pdf_url = "https://pdfobject.com/pdf/sample.pdf"
    response = requests.get(pdf_url)
    pdf_path = tmp_path / "sample.pdf"
    with open(pdf_path, "wb") as f:
        f.write(response.content)
    return pdf_path
@pytest.fixture
def sample_document():
    return Document(
        page_number=0,
        content="This is a sample document. It contains multiple sentences. For testing purposes.",
        image=None,
        metadata={"type": "text"}
    )

def test_pdf_loader_extract_text(sample_pdf):
    loader = PDFLoader(file_path=str(sample_pdf), img_path="/tmp")
    documents = loader.extract_text()
    assert isinstance(documents, list)
    assert all(isinstance(doc, Document) for doc in documents)

def test_pdf_loader_extract_images(sample_pdf):
    loader = PDFLoader(file_path=str(sample_pdf), img_path="/tmp")
    documents = loader.extract_images()
    assert isinstance(documents, list)
    assert all(isinstance(doc, Document) for doc in documents)
    for doc in documents:
        assert os.path.exists(doc.image)

def test_basic_text_chunk(sample_document):
    chunks = BasicTextChunk(chunk_size=10, document=sample_document)
    assert isinstance(chunks, list)
    assert all(isinstance(chunk, TextChunk) for chunk in chunks)

def test_recursive_text_chunk(sample_document):
    chunks = RecursiveTextChunk(chunk_size=10, document=sample_document)
    assert isinstance(chunks, list)
    assert all(isinstance(chunk, TextChunk) for chunk in chunks)

def test_sentence_aware_chunk(sample_document):
    chunks = SentenceAwareChunk(chunk_size=20, document=sample_document)
    assert isinstance(chunks, list)
    assert all(isinstance(chunk, TextChunk) for chunk in chunks)
