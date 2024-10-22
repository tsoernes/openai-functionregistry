import os

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI, AzureOpenAI
import datetime
import tiktoken
from dataclasses import dataclass

load_dotenv()

@dataclass
class LLMCost:
    n_input_tokens: int = 0
    input_cost: float = 0
    n_output_tokens: int = 0
    output_cost: float = 0
    currency: str = "NOK"

    @property
    def total(self) -> float:
        return self.input_cost + self.output_cost

    def __str__(self) -> str:
        return f"{self.n_input_tokens:,} input tokens cost {self.input_cost:.4f}, {self.n_output_tokens:,} output tokens cost {self.output_cost:.4f}; total {self.total:.4f} {self.currency}"

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
            self.api_version = os.getenv("OAI-GPT4O-API-VERSION", "2024-08-01-preview")

        self.is_mini = is_mini
        self.client = AzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
        )

    def calculate_cost(self, input_tokens: str | int = 0, output_tokens: str | int = 0) -> LLMCost:
        """
        Returns a LLM cost object.

        In NOK.

        https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/
        """
        if self.is_mini:
            # GPT-4o-mini Global Deployment
            cost_per_1m_inp_nok = 1.57738
            cost_per_1m_out_nok = 6.3095
        elif datetime.datetime.strptime(self.api_version[:10], "%Y-%m-%d") >= datetime.datetime(year=2024, month=8, day=6):
            # gpt-4o-2024-08-06 Global Deployment
            cost_per_1m_inp_nok = 26.2896
            cost_per_1m_out_nok = 105.158001
        else:
            # GPT-4o Global Deployment
            cost_per_1m_inp_nok = 52.5791
            cost_per_1m_out_nok = 157.7371

        encoder = tiktoken.encoding_for_model("gpt-4o")  # same encoding with 4o-mini as with 4o
        n_input_tokens = input_tokens if isinstance(input_tokens, int) else len(encoder.encode(input_tokens))
        n_output_tokens = output_tokens if isinstance(output_tokens, int) else len(encoder.encode(output_tokens))

        mil = 1_000_000
        input_cost = n_input_tokens * cost_per_1m_inp_nok / mil
        output_cost = n_output_tokens * cost_per_1m_out_nok / mil
        return LLMCost(n_input_tokens=n_input_tokens, input_cost=input_cost, n_output_tokens=n_output_tokens, output_cost=output_cost, currency="NOK")
