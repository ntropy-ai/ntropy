from unittest.mock import patch, MagicMock
from ntropy.core.embeddings.openai import get_client, OpenAIEmbeddings
from ntropy.core.providers import OpenAIConnection
from ntropy.core.utils.base_format import TextChunk, Document
from ntropy.core.embeddings import list_models
import pytest
import requests
import os

# Define the fixture
@pytest.fixture
def mock_models_base_settings():
    return {
        "embeddings_models": ["openai.clip-vit-base-patch32"]
    }

# Test OpenAIConnection initialization
def test_openai_connection_init():
    openai_conn = OpenAIConnection("api_key", {})
    assert openai_conn.api_key == "api_key"
    assert openai_conn.client is None

# Test OpenAIConnection init_connection
@patch('openai.OpenAI')
def test_openai_connection_init_connection(mock_openai_client):
    openai_conn = OpenAIConnection("api_key", {})
    openai_conn.init_connection()
    mock_openai_client.assert_called_once_with(api_key="api_key")

# Test list_models function
def test_list_models(mock_models_base_settings):
    models = list_models()
    assert "OpenAI" in models
    assert "embeddings_models" in models['OpenAI']
    assert "openai.clip-vit-base-patch32" in models['OpenAI']["embeddings_models"]

# Test get_client function
@patch('ntropy.core.utils.connections_manager.ConnectionManager.get_connection')
def test_get_client(mock_get_connection):
    mock_conn = MagicMock()
    mock_get_connection.return_value = mock_conn
    client = get_client()
    mock_get_connection.assert_called_once_with("OpenAI")
    mock_conn.get_client.assert_called_once()

# Test create_embeddings function
@patch('ntropy.core.embeddings.openai.get_client')
def test_create_embeddings(mock_get_client, mock_models_base_settings):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    chunk = TextChunk(id="doc1", chunk="This is a test document.", chunk_number=1, document_id="doc1")
    model_settings = {"device": "cpu"}
    vector = OpenAIEmbeddings("openai.clip-vit-base-patch32", chunk, model_settings)

    assert vector.document_id == "doc1"
    assert isinstance(vector.vector, list)
    assert vector.size == len(vector.vector)
    assert vector.data_type == "text"
    assert vector.content == "This is a test document."
    assert vector.metadata['model'] == "openai.clip-vit-base-patch32"
    assert vector.metadata['model_settings'] == model_settings


# Test create_embeddings function for image
@patch('ntropy.core.embeddings.openai.get_client')
def test_create_embeddings_image(mock_get_client, mock_models_base_settings):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    image_url = "https://i.natgeofe.com/n/4f5aaece-3300-41a4-b2a8-ed2708a0a27c/domestic-dog_thumb_square.jpg"
    response = requests.get(image_url)
    image_path = "/tmp/sample_image.jpg"
    with open(image_path, "wb") as f:
        f.write(response.content)
    document = Document(id="doc1", page_content=None, image=image_path, page_number=1)
    model_settings = {"device": "cpu"}
    vector = OpenAIEmbeddings("openai.clip-vit-base-patch32", document, model_settings)
    os.remove(image_path)
    assert vector.document_id == "doc1"
    assert isinstance(vector.vector, list)
    assert vector.size == len(vector.vector)
    assert vector.data_type == "image"
    assert vector.content is None
    assert vector.metadata['model'] == "openai.clip-vit-base-patch32"
    assert vector.metadata['model_settings'] == model_settings
