
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import openai


class OpenAIConnection():
    def __init__(self, api_key: str, other_setting: dict, **kwargs):
        self.api_key = api_key
        self.client = None

    def init_connection(self):
        try: 
            self.client = openai.OpenAI(api_key=self.api_key)
            print("OpenAI connection initialized successfully.")
        except Exception as e:
            raise Exception(f"Error initializing OpenAI connection: {e}")
        

    def get_client(self):
        if self.client is None:
            self.init_connection()
        return self.client
    
class AWSConnection:
    def __init__(self, access_key: str, secret_access_key: str, other_setting: dict, **kwargs):
        self.aws_access_key_id = access_key
        self.aws_secret_access_key = secret_access_key
        # other settings
        self.region_name = other_setting.get("region_name", "us-east-1") # default to us-east-1 if not provided
        self.service_name = other_setting.get("service_name", "bedrock")
        self.client = None

    def init_connection(self):
        try:
            self.client = boto3.client(
                service_name=self.service_name,
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.region_name
            )
            print("AWS connection initialized successfully.")
            
        except (NoCredentialsError, PartialCredentialsError) as e:
            raise Exception(f"Error initializing AWS connection: {e}")

    def get_client(self):
        if self.client is None:
            self.init_connection()
        return self.client