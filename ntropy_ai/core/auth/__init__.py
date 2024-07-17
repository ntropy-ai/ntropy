import os
import sqlite3
import json
from tabulate import tabulate
from pydantic import BaseModel
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes
from ntropy_ai.core.utils.settings import ModelsBaseSettings
from ntropy_ai.core.utils.connections_manager import ConnectionManager

class BaseAuth():
    """
    BaseAuth class handles the authentication mechanism using RSA encryption.
    It manages the creation, connection, and operations on a SQLite database 
    that stores provider credentials in an encrypted format.
    """
    def __init__(self):
        """
        Initializes the BaseAuth class with default database paths and keys.
        """
        self.db_base_path = os.path.join(os.path.dirname(__file__), "..", "utils", "db") # Base path for the database
        self.db_location = os.path.join(self.db_base_path, "Login.db") # Full path to the database file
        self.db = None
        self.private_key = None
        self.public_key = None

    def encrypt(self, data: str) -> bytes:
        """
        Encrypts the given data using the public key.

        Args:
            data (str): The data to be encrypted.

        Returns:
            bytes: The encrypted data.
        """
        return self.public_key.encrypt(
            data.encode('utf-8'),
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
    
    def decrypt(self, data: bytes) -> str:
        """
        Decrypts the given data using the private key.

        Args:
            data (bytes): The data to be decrypted.

        Returns:
            str: The decrypted data.
        """
        return self.private_key.decrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        ).decode('utf-8')
    
    def create_db(self):
        """
        Creates a new SQLite database and generates RSA keys.
        Prompts the user if the database already exists.
        """
        if os.path.exists(self.db_location):
            response = input("Database already exists. Do you want to override it? (yes/no): ")
            if response.lower() != 'yes':
                print("Operation cancelled.")
                return
        else:
            if not os.path.exists(self.db_base_path):
                os.makedirs(self.db_base_path, exist_ok=True)
        print("Creating database...")
        if os.path.exists(self.db_location):
            os.remove(self.db_location)
        self.db = sqlite3.connect(self.db_location)
        
        # Generate RSA keys
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        self.public_key = self.private_key.public_key()
        
        # Store the public key in the database
        public_key_pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        private_key_pem = self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        )
        print("Save this private key securely. You will need it to decrypt your data:")
        print(private_key_pem.decode('utf-8'))
        with open(os.path.join(self.db_base_path, "private_key.pem"), 'wb') as file:
            file.write(private_key_pem)
        file.close()
        print("Private key saved to: ", os.path.join(self.db_base_path, "private_key.pem"))

        # Create tables in the database
        self.db.execute("CREATE TABLE IF NOT EXISTS auth (public_key BLOB)")
        self.db.execute("CREATE TABLE IF NOT EXISTS providers (service_name TEXT, name TEXT, api_key TEXT, api_secret TEXT, access_key TEXT, other_setting JSON)")
        self.db.commit()
        self.db.execute("INSERT INTO auth (public_key) VALUES (?)", (public_key_pem,))
        self.db.commit()
        self.db.close()

    def connect(self, private_key: str = None, db_path: str = None, key_file: str = None):
        """
        Connects to the SQLite database and loads the private key.

        Args:
            private_key (str): The private key as a string.
            db_path (str): The path to the database file.
            key_file (str): The path to the private key file.
        """
        if db_path is not None:
            self.db_location = db_path
        if not os.path.exists(self.db_location):
            raise Exception("Database file does not exist. Please create the database first using create_db().")
        if key_file:
            with open(key_file, 'r') as file:
                private_key = file.read()
            file.close()
            self.private_key = private_key.encode('utf-8')
            
        else:
            self.private_key = private_key.encode('utf-8')

        self.private_key = serialization.load_pem_private_key(
            private_key.encode('utf-8'),
            password=None,
        )
        self.db = sqlite3.connect(self.db_location)
        cursor = self.db.cursor()
        cursor.execute("SELECT public_key FROM auth")
        public_key_pem = cursor.fetchone()[0]
        self.public_key = serialization.load_pem_public_key(public_key_pem)

        # Automatically connect for all providers
        creds_list = self.get_creds()
        for cred in creds_list:
            provider_class = ModelsBaseSettings().providers_list_map.get(cred.get("service_name"))
            if provider_class:
                auth_creds = provider_class['auth'].model_validate(cred).dict()
                conn = provider_class['connect'](**auth_creds)
                conn.init_connection()

                ConnectionManager().add_connection(cred.get("service_name"), conn)

    def update_provider(self, provider: BaseModel, **kwargs):
        """
        Updates the provider information in the database.

        Args:
            provider (BaseModel): The provider model containing updated information.
        """
        if self.db is None:
            raise Exception("Database not connected, connect first using Database().connect(password) or create a db using Database().create_db()")
        
        provider_data = provider.model_dump()
        encrypted_api_key = self.encrypt(provider_data.get("api_key")) if provider_data.get("api_key") else None
        encrypted_access_key = self.encrypt(provider_data.get("access_key")) if provider_data.get("access_key") else None
        encrypted_secret_key = self.encrypt(provider_data.get("secret_access_key")) if provider_data.get("secret_access_key") else None
        other_setting_json = json.dumps(provider_data.get("other_setting")) if provider_data.get("other_setting") else None

        cursor = self.db.cursor()
        cursor.execute("""
            UPDATE providers 
            SET name = COALESCE(?, name), api_key = COALESCE(?, api_key), api_secret = COALESCE(?, api_secret), access_key = COALESCE(?, access_key), other_setting = COALESCE(?, other_setting)
            WHERE service_name = ?
        """, (provider_data.get("name"), encrypted_api_key, encrypted_secret_key, encrypted_access_key, other_setting_json, provider_data.get("service_name")))
        
        self.db.commit()
        

    def add_provider(self, provider: BaseModel, **kwargs):
        """
        Adds a new provider to the database.

        Args:
            provider (BaseModel): The provider model containing the information to be added.
        """
        if self.db is None:
            raise Exception("Database not connected, connect first using Database().connect(password) or create a db using Database().create_db()")
       
        cursor = self.db.cursor()
        cursor.execute("SELECT public_key FROM auth")
        public_key_pem = cursor.fetchone()
        if public_key_pem is None:
            raise Exception("Public key does not exist in the database. Please add the public key first.")
        
        provider_data = provider.model_dump()

        encrypted_api_key = self.encrypt(provider_data.get("api_key")) if provider_data.get("api_key") else None
        encrypted_access_key = self.encrypt(provider_data.get("access_key")) if provider_data.get("access_key") else None
        encrypted_secret_key = self.encrypt(provider_data.get("secret_access_key")) if provider_data.get("secret_access_key") else None
        other_setting_json = json.dumps(provider_data.get("other_setting")) if provider_data.get("other_setting") else None
        
        self.db.execute("INSERT INTO providers (service_name, name, api_key, api_secret, access_key, other_setting) VALUES (?, ?, ?, ?, ?, ?)", 
                        (provider_data.get("service_name"), provider_data.get("name"), encrypted_api_key, encrypted_secret_key, encrypted_access_key, other_setting_json))
        self.db.commit()

    def list_providers(self):
        """
        Lists all providers in the database with decrypted credentials.

        Returns:
            str: JSON string of the list of providers with decrypted credentials.
        """
        if self.db is None:
            raise Exception("Database not connected, connect first using Database().connect(password) or create a db using Database().create_db()")
        cursor = self.db.cursor()
        cursor.execute("SELECT * FROM providers")
        providers_data = cursor.fetchall()

        decrypted_providers = []
        for provider in providers_data:
            service_name, name, encrypted_api_key, encrypted_api_secret, encrypted_access_key, other_setting = provider
            decrypted_api_key = self.decrypt(encrypted_api_key) if encrypted_api_key else None
            decrypted_api_secret = self.decrypt(encrypted_api_secret) if encrypted_api_secret else None
            decrypted_access_key = self.decrypt(encrypted_access_key) if encrypted_access_key else None
            decrypted_providers.append([
                service_name,
                name,
                decrypted_api_key[:2] + "******" if decrypted_api_key else None,
                "******",  # Hide the secret completely
                decrypted_access_key[:2] + "******" + decrypted_access_key[-2:] if decrypted_access_key else None,
                other_setting
            ])

        print(tabulate(decrypted_providers, headers=["Service Name", "Name", "API Key", "Secret Key", "Access Key", "Other Setting"]))
        return decrypted_providers

    def delete_provider(self, provider: BaseModel):
        """
        Deletes a provider from the database.

        Args:
            provider (BaseModel): The provider model containing the information to be deleted.
        """
        if self.db is None:
            raise Exception("Database not connected, connect first using Database().connect(password) or create a db using Database().create_db()")
       
        provider_name = provider.service_name
        cursor = self.db.cursor()
        if cursor is None:
            raise Exception("Failed to create a database cursor. check if the database is connected")
        
        cursor.execute("SELECT COUNT(*) FROM providers WHERE service_name = ?", (provider_name,))
        provider_count = cursor.fetchone()[0]
        
        if provider_count == 0:
            print(f"No provider found with the service_name: {provider_name}")
            return
        
        cursor.execute("DELETE FROM providers WHERE service_name = ?", (provider_name,))
        self.db.commit()
        print(f"Provider '{provider_name}' deleted successfully.")

    def get_creds(self, provider: BaseModel = None):
        """
        Retrieves the credentials for a specific provider or all providers.

        Args:
            provider (BaseModel, optional): The provider model to retrieve credentials for. Defaults to None.

        Returns:
            dict or list: Decrypted credentials for the specified provider or a list of all providers' credentials.
        """
        if self.db is None:
            raise Exception("Database not connected, connect first using Database().connect(password) or create a db using Database().create_db()")
        cursor = self.db.cursor()
        if provider:
            cursor.execute("SELECT * FROM providers WHERE service_name = ?", (provider().service_name,))
            provider_data = cursor.fetchone()
            if provider_data:
                service_name, name, encrypted_api_key, encrypted_api_secret, encrypted_access_key, other_setting = provider_data
                decrypted_api_key = self.decrypt(encrypted_api_key) if encrypted_api_key else None
                decrypted_api_secret = self.decrypt(encrypted_api_secret) if encrypted_api_secret else None
                decrypted_access_key = self.decrypt(encrypted_access_key) if encrypted_access_key else None
                other_setting = json.loads(other_setting) if other_setting else None
                return {
                    "service_name": service_name,
                    "name": name,
                    "api_key": decrypted_api_key,
                    "secret_access_key": decrypted_api_secret,
                    "access_key": decrypted_access_key,
                    "other_setting": other_setting
                }
        else:
            cursor.execute("SELECT * FROM providers")
            providers_data = cursor.fetchall()
            decrypted_providers = []
            for provider_data in providers_data:
                service_name, name, encrypted_api_key, encrypted_api_secret, encrypted_access_key, other_setting = provider_data
                decrypted_api_key = self.decrypt(encrypted_api_key) if encrypted_api_key else None
                decrypted_api_secret = self.decrypt(encrypted_api_secret) if encrypted_api_secret else None
                decrypted_access_key = self.decrypt(encrypted_access_key) if encrypted_access_key else None
                other_setting = json.loads(other_setting) if other_setting else None
                decrypted_providers.append({
                    "service_name": service_name,
                    "name": name,
                    "api_key": decrypted_api_key,
                    "secret_access_key": decrypted_api_secret,
                    "access_key": decrypted_access_key,
                    "other_setting": other_setting
                })
            return decrypted_providers
