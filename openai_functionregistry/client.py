import datetime
import os
import re
from dataclasses import dataclass
from typing import Iterable

import tiktoken
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI, AzureOpenAI

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

    def __add__(self, other):
        if isinstance(other, LLMCost):
            return LLMCost(
                n_input_tokens=self.n_input_tokens + other.n_input_tokens,
                input_cost=self.input_cost + other.input_cost,
                n_output_tokens=self.n_output_tokens + other.n_output_tokens,
                output_cost=self.output_cost + other.output_cost,
                currency=self.currency,
            )
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, LLMCost):
            return LLMCost(
                n_input_tokens=self.n_input_tokens - other.n_input_tokens,
                input_cost=self.input_cost - other.input_cost,
                n_output_tokens=self.n_output_tokens - other.n_output_tokens,
                output_cost=self.output_cost - other.output_cost,
                currency=self.currency,
            )
        return NotImplemented

    def __mul__(self, factor):
        if isinstance(factor, (int, float)):
            return LLMCost(
                n_input_tokens=int(self.n_input_tokens * factor),
                input_cost=self.input_cost * factor,
                n_output_tokens=int(self.n_output_tokens * factor),
                output_cost=self.output_cost * factor,
                currency=self.currency,
            )
        return NotImplemented

    def __truediv__(self, factor):
        if isinstance(factor, (int, float)):
            return LLMCost(
                n_input_tokens=int(self.n_input_tokens / factor),
                input_cost=self.input_cost / factor,
                n_output_tokens=int(self.n_output_tokens / factor),
                output_cost=self.output_cost / factor,
                currency=self.currency,
            )
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, LLMCost):
            return self.total == other.total
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, LLMCost):
            return self.total < other.total
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, LLMCost):
            return self.total <= other.total
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, LLMCost):
            return self.total > other.total
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, LLMCost):
            return self.total >= other.total
        return NotImplemented


class Client:
    """Configuration for OpenAI model endpoints"""

    def __init__(self, azure_endpoint: str,
                 api_key: str,
                 model: str,
                 api_version: str,
                 async_: bool = False,
                 tokens_per_minute_limit: int = 450_000,
                 requests_per_minute_limit: int = 4_500):
        self.tokens_per_minute_limit = tokens_per_minute_limit
        self.requests_per_minute_limit = requests_per_minute_limit
        self.azure_endpoint = azure_endpoint
        self.api_key = api_key
        self.model = model
        self.api_version = api_version
        self.async_ = async_

        if async_:
            self.client = AsyncAzureOpenAI(
                azure_endpoint=self.azure_endpoint,
                api_key=self.api_key,
                api_version=self.api_version,
                # azure_deployment=self.model,
            )
        else:
            self.client = AzureOpenAI(
                azure_endpoint=self.azure_endpoint,
                api_key=self.api_key,
                api_version=self.api_version,
                # azure_deployment=self.model,
            )
        self.encoder = tiktoken.encoding_for_model(
            "gpt-4o"
        )  # same encoding with 4o-mini as with 4o

    def calculate_cost(
        self,
        input_tokens: str | Iterable[str] | int = 0,
        output_tokens: str | Iterable[str] | int = 0,
    ) -> LLMCost:
        """
        tokens: a text string, a list of text strings, or the number of tokens (int)
        Returns a LLM cost object.

        In NOK.

        https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/
        """
        return calculate_cost(
            is_mini="mini" in self.model.lower(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


def calculate_cost(
    is_mini: bool,
    input_tokens: str | Iterable[str] | int = 0,
    output_tokens: str | Iterable[str] | int = 0,
) -> LLMCost:
    """
    tokens: a text string, a list of text strings, or the number of tokens (int)
    Returns a LLM cost object.

    In NOK.

    https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/
    """
    if is_mini:
        # GPT-4o-mini Regional API
        cost_per_1m_inp_nok = 1.735108
        cost_per_1m_out_nok = 6.94043
    else:
        # gpt-4o-2024-08-06 Global Deployment
        cost_per_1m_inp_nok = 26.2896
        cost_per_1m_out_nok = 105.158001

    encoder = tiktoken.encoding_for_model(
        "gpt-4o"
    )  # same encoding with 4o-mini as with 4o
    if isinstance(input_tokens, int):
        n_input_tokens = input_tokens
    elif isinstance(input_tokens, str):
        n_input_tokens = len(encoder.encode(input_tokens))
    elif isinstance(input_tokens, Iterable):
        n_input_tokens = sum(map(len, encoder.encode_batch(list(input_tokens))))
    else:
        raise TypeError(input_tokens)
    if isinstance(output_tokens, int):
        n_output_tokens = output_tokens
    elif isinstance(output_tokens, str):
        n_output_tokens = len(encoder.encode(output_tokens))
    elif isinstance(output_tokens, list):
        n_output_tokens = sum(map(len, encoder.encode_batch(output_tokens)))
    else:
        raise TypeError(output_tokens)

    mil = 1_000_000
    input_cost = n_input_tokens * cost_per_1m_inp_nok / mil
    output_cost = n_output_tokens * cost_per_1m_out_nok / mil

    return LLMCost(
        n_input_tokens=n_input_tokens,
        input_cost=input_cost,
        n_output_tokens=n_output_tokens,
        output_cost=output_cost,
        currency="NOK",
    )
