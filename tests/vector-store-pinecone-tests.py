import pytest
import json
from unittest.mock import patch, MagicMock
from ntropy.core.vector_store.pinecone import Pinecone, get_client
from ntropy.core.utils.base_format import Document, Vector
from ntropy.core.utils.settings import ModelsBaseSettings
from pinecone import ServerlessSpec

# Mock settings for ModelsBaseSettings
@pytest.fixture
def mock_models_base_settings():
    with patch('ntropy.core.utils.settings.ModelsBaseSettings') as MockSettings:
        mock_instance = MockSettings.return_value
        mock_instance.providers_list_map = {
            "Pinecone": {
                "functions": {
                    "embeddings": MagicMock()
                }
            }
        }
        yield mock_instance

# Mock connection manager
@pytest.fixture
def mock_connection_manager():
    with patch('ntropy.core.vector_store.pinecone.ConnectionManager') as MockConnectionManager:
        mock_instance = MockConnectionManager.return_value
        mock_connection = MagicMock()
        mock_instance.get_connection.return_value = mock_connection
        yield mock_connection

# Test Pinecone initialization
@patch('ntropy.core.vector_store.pinecone.get_client', return_value=MagicMock())
@patch('ntropy.core.vector_store.pinecone.ConnectionManager')
def test_pinecone_init(mock_connection_manager, mock_get_client):
    pinecone = Pinecone(index_name="test_index")
    assert pinecone.client == mock_get_client.return_value
    assert pinecone.index_name == "test_index"

# Test Pinecone create_index
@patch('ntropy.core.vector_store.pinecone.get_client', return_value=MagicMock())
@patch('ntropy.core.vector_store.pinecone.ConnectionManager')
def test_create_index(mock_connection_manager, mock_get_client):
    pinecone = Pinecone(index_name="test_index")
    pinecone.create_index("test_index", 512, "cosine", spec=ServerlessSpec(cloud='aws', region='us-east-1'))
    mock_get_client.return_value.create_index.assert_called_once_with(
        name="test_index",
        dimension=512,
        metric="cosine",
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

# Test Pinecone add_vectors
@patch('ntropy.core.vector_store.pinecone.get_client', return_value=MagicMock())
@patch('ntropy.core.vector_store.pinecone.ConnectionManager')
def test_add_vectors(mock_connection_manager, mock_get_client):
    pinecone = Pinecone(index_name="test_index")
    vectors = [Vector(id="vec1", vector=[0.1, 0.2, 0.3], metadata={}, document_id="doc1", size=3, data_type="text", content="test content")]
    pinecone.add_vectors(vectors)
    mock_get_client.return_value.Index().upsert.assert_called_once()

# Test Pinecone set_embeddings_model
@patch('ntropy.core.embeddings.openai.get_client')
@patch('ntropy.core.vector_store.pinecone.ConnectionManager')
def test_set_embeddings_model(mock_connection_manager, mock_models_base_settings):
    pinecone = Pinecone(index_name="test_index")
    pinecone.set_embeddings_model(model="openai.clip-vit-base-patch32")
    assert pinecone.embedding_model_name == "openai.clip-vit-base-patch32"

# Test Pinecone query
@patch('ntropy.core.vector_store.pinecone.get_client', return_value=MagicMock())
@patch('ntropy.core.vector_store.pinecone.ConnectionManager')
def test_query(mock_connection_manager, mock_get_client, mock_models_base_settings):
    mock_client = mock_get_client.return_value
    mock_client.describe_index.return_value = {"dimension": 512, "metric": "cosine"}
    mock_client.Index().query.return_value = {"matches": [{"id": "vec1", "score": 0.9}]}
    mock_client.Index().fetch.return_value = {"vectors": {"vec1": {"id": "vec1", "values": [0.1, 0.2, 0.3], "metadata": {"document_id": "doc1", "content": "test content", "data_type": "text", "size": 3}}}}
    
    pinecone = Pinecone(index_name="test_index")
    pinecone.set_embeddings_model(model="openai.clip-vit-base-patch32")
    results = pinecone.query(query_text="test query")
    assert len(results) == 1
    assert results[0].id == "vec1"
    assert results[0].score == 0.9
    assert results[0].document_id == "doc1"
    assert results[0].vector == [0.1, 0.2, 0.3]
    assert results[0].content == "test content"
    assert results[0].data_type == "text"
    assert results[0].size == 3