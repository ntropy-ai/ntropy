import pytest
import json
from unittest.mock import patch, MagicMock
from ntropy.core.providers.aws import AWSConnection, list_models, get_client, create_embeddings
from ntropy.core.utils.base_format import Document, TextChunk
from ntropy.core.utils.settings import ModelsBaseSettings

# Mock settings for ModelsBaseSettings
@pytest.fixture
def mock_models_base_settings():
    with patch('ntropy.core.utils.settings.ModelsBaseSettings') as MockSettings:
        mock_instance = MockSettings.return_value
        mock_instance.providers_list_map = {
            "AWS": {
                "embeddings_model": {
                    "models_map": {
                        "amazon.titan-embed-text-v2:0": MagicMock(),
                        "amazon.titan-embed-image-v1": MagicMock()
                    }
                }
            }
        }
        yield mock_instance

# Test AWSConnection initialization
def test_aws_connection_init():
    aws_conn = AWSConnection("access_key", "secret_key", {"region_name": "us-west-2"})
    assert aws_conn.aws_access_key_id == "access_key"
    assert aws_conn.aws_secret_access_key == "secret_key"
    assert aws_conn.region_name == "us-west-2"
    assert aws_conn.service_name == "bedrock"

# Test AWSConnection init_connection
@patch('boto3.client')
def test_aws_connection_init_connection(mock_boto_client):
    aws_conn = AWSConnection("access_key", "secret_key", {"region_name": "us-west-2"})
    aws_conn.init_connection()
    mock_boto_client.assert_called_once_with(
        service_name="bedrock",
        aws_access_key_id="access_key",
        aws_secret_access_key="secret_key",
        region_name="us-west-2"
    )

# Test list_models function
def test_list_models(mock_models_base_settings):
    models = list_models()
    assert "embeddings_models" in models
    assert "amazon.titan-embed-text-v2:0" in models["embeddings_models"]
    assert "amazon.titan-embed-image-v1" in models["embeddings_models"]

# Test get_client function
@patch('ntropy.core.utils.connections_manager.ConnectionManager.get_connection')
def test_get_client(mock_get_connection):
    mock_conn = MagicMock()
    mock_get_connection.return_value = mock_conn
    client = get_client()
    mock_get_connection.assert_called_once_with("AWS")
    mock_conn.get_client.assert_called_once()

# Test create_embeddings function
@patch('ntropy.core.providers.aws.get_client')
def test_create_embeddings(mock_get_client, mock_models_base_settings):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.invoke_model.return_value = {
        'body': MagicMock(read=MagicMock(return_value=json.dumps({'embedding': [0.1, 0.2, 0.3]})))
    }

    document = Document(id="doc1", page_content="This is a test document. This is a test document. This is a test document. This is a test document. This is a test document.", page_number=1)
    model_settings = {"dimensions": 512, "normalize": True}
    vector = create_embeddings("amazon.titan-embed-text-v2:0", document, model_settings)

    assert vector.document_id == "doc1"
    assert vector.vector == [0.1, 0.2, 0.3]
    assert vector.size == 3
    assert vector.data_type == "text"
    assert vector.content == "This is a test document. This is a test document. This is a test document. This is a test document. This is a test document."
    assert vector.metadata['model'] == "amazon.titan-embed-text-v2:0"
    assert vector.metadata['model_settings'] == model_settings

