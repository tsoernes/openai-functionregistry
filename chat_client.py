import os

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI, AzureOpenAI

load_dotenv()


class Client:
    """Configuration for OpenAI model endpoints"""

    def __init__(self, is_mini: bool = True, api_version: str | None = None):
        if is_mini:
            self.azure_endpoint = os.environ["OAI-GPT4O-mini-18072024-ENDPOINT"]
            self.api_key = os.environ["OAI-GPT4O-mini-18072024-API-KEY"]
            self.model = "m-gpt-4o-mini-18072024"
        else:
            self.azure_endpoint = os.environ["OAI-GPT4O-06082024-ENDPOINT"]
            self.api_key = os.environ["OAI-GPT4O-06082024-API-KEY"]
            self.model = "m-gpto-06082024"

        if api_version is None:
            api_version = os.getenv("OAI-GPT4O-API-VERSION", "2024-08-01-preview")

        self.client = AzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            api_version=api_version,
        )
