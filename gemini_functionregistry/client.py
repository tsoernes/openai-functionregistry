import datetime
import os
from dataclasses import dataclass
from typing import Iterable

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMCost:
    n_input_tokens: int = 0
    input_cost: float = 0
    n_output_tokens: int = 0
    output_cost: float = 0
    currency: str = "USD"

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
    """Configuration for Google Gemini model endpoints"""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-1.5-flash",
        tokens_per_minute_limit: int = 1_000_000,
        requests_per_minute_limit: int = 1_500,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") 
        if not self.api_key:
            raise ValueError("API key required. Set GEMINI_API_KEY environment variable or pass api_key parameter.")
            
        self.model = model
        self.tokens_per_minute_limit = tokens_per_minute_limit
        self.requests_per_minute_limit = requests_per_minute_limit

        # Configure the API key
        genai.configure(api_key=self.api_key)
        
        # Create the generative model
        self.client = genai.GenerativeModel(model_name=model)

    def calculate_cost(
        self,
        input_tokens: str | Iterable[str] | int = 0,
        output_tokens: str | Iterable[str] | int = 0,
    ) -> LLMCost:
        """
        tokens: a text string, a list of text strings, or the number of tokens (int)
        Returns a LLM cost object.

        In USD based on Google AI pricing.
        """
        return calculate_cost(
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


def calculate_cost(
    model: str,
    input_tokens: str | Iterable[str] | int = 0,
    output_tokens: str | Iterable[str] | int = 0,
) -> LLMCost:
    """
    Calculate cost for Google Gemini models.
    
    Pricing as of 2024 from https://ai.google.dev/pricing
    """
    # Approximate token counting for strings (Gemini uses similar token counts to OpenAI)
    def count_tokens(text_input):
        if isinstance(text_input, int):
            return text_input
        elif isinstance(text_input, str):
            # Rough approximation: 4 characters per token
            return len(text_input) // 4
        elif isinstance(text_input, Iterable):
            return sum(count_tokens(t) for t in text_input)
        else:
            raise TypeError(f"Unsupported input type: {type(text_input)}")

    n_input_tokens = count_tokens(input_tokens)
    n_output_tokens = count_tokens(output_tokens)

    # Gemini pricing per 1M tokens (USD)
    if "flash" in model.lower():
        # Gemini 1.5 Flash
        cost_per_1m_inp_usd = 0.075
        cost_per_1m_out_usd = 0.30
    elif "pro" in model.lower():
        # Gemini 1.5 Pro
        cost_per_1m_inp_usd = 3.50
        cost_per_1m_out_usd = 10.50
    else:
        # Default to Flash pricing
        cost_per_1m_inp_usd = 0.075
        cost_per_1m_out_usd = 0.30

    mil = 1_000_000
    input_cost = n_input_tokens * cost_per_1m_inp_usd / mil
    output_cost = n_output_tokens * cost_per_1m_out_usd / mil

    return LLMCost(
        n_input_tokens=n_input_tokens,
        input_cost=input_cost,
        n_output_tokens=n_output_tokens,
        output_cost=output_cost,
        currency="USD",
    )