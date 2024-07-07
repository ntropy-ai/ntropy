import os
import pytest
from ntropy.core.auth import BaseAuth


@pytest.fixture
def setup_db():
    # Setup: Create a temporary database and key file
    db_instance = BaseAuth()
    db_instance.create_db()
    key_file = os.path.join(db_instance.db_base_path, "private_key.pem")
    yield db_instance, key_file
    # Teardown: Remove the temporary database and key file
    os.remove(db_instance.db_location)
    os.remove(key_file)

def test_create_db(setup_db):
    db_instance, key_file = setup_db
    assert os.path.exists(db_instance.db_location)
    assert os.path.exists(key_file)

def test_connect_db(setup_db):
    db_instance, key_file = setup_db
    db_instance.connect(key_file=key_file)
    assert db_instance.db is not None

def test_add_provider(setup_db):
    db_instance, key_file = setup_db
    db_instance.connect(key_file=key_file)
    from ntropy.core.utils.auth_format import AWSauth
    provider = AWSauth(
        name="Test AWS",
        access_key="test_access_key",
        secret_access_key="test_secret_key"
    )
    db_instance.add_provider(provider)
    providers = db_instance.list_providers()
    assert "Test AWS" in providers

def test_list_providers(setup_db):
    db_instance, key_file = setup_db
    db_instance.connect(key_file=key_file)
    providers = db_instance.list_providers()
    assert isinstance(providers, str)  # Assuming list_providers returns a JSON string

def test_delete_provider(setup_db):
    db_instance, key_file = setup_db
    db_instance.connect(key_file=key_file)
    from ntropy.core.utils.auth_format import AWSauth
    provider = AWSauth(
        name="Test AWS",
        access_key="test_access_key",
        secret_access_key="test_secret_key"
    )
    db_instance.add_provider(provider)
    db_instance.delete_provider(provider)
    providers = db_instance.list_providers()
    assert "Test AWS" not in providers