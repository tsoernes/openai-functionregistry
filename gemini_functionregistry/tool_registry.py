"""
Defines a `ParserRegistry` and a `FunctionRegistry` for Google Gemini models
"""

import asyncio
import inspect
import json
import logging
import random
import string
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Sequence, TypeVar

import google.generativeai as genai
from betterpathlib import Path
from pydantic import BaseModel

from gemini_functionregistry.client import Client

# Configure logging
logging.basicConfig(level=logging.INFO)


Model = TypeVar("Model", bound=BaseModel)


@dataclass
class FunctionCall:
    """Function call arguments and the result of the function call."""

    arguments: Model
    result: Any


class LLMError(Exception):
    """Base exception for LLM-related errors"""

    pass


class ModelFailedError(LLMError):
    """Raised when both mini and regular models fail"""

    pass


class MultipleToolCallsError(LLMError):
    """Raised when multiple tool calls are received but only one was expected"""

    pass


class NoToolCallsError(LLMError):
    """Raised when no tool calls are received"""

    pass


@dataclass
class RequestMetrics:
    """Tracks request and token counts within a time window"""

    timestamp: datetime
    token_count: int


class AsyncRateLimiter:
    """Rate limiter for async API calls that handles both request and token limits"""

    def __init__(
        self, requests_per_minute: int, tokens_per_minute: int, window_size: int = 60
    ):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.window_size = window_size  # in seconds
        self.request_history: deque[datetime] = deque()
        self.token_history: deque[RequestMetrics] = deque()

    async def acquire(self, token_count: int = 0):
        """Wait until rate limits allow another request"""
        while True:
            current_time = datetime.now()
            window_start = current_time - timedelta(seconds=self.window_size)

            # Clean up old history
            while self.request_history and self.request_history[0] < window_start:
                self.request_history.popleft()
            while self.token_history and self.token_history[0].timestamp < window_start:
                self.token_history.popleft()

            # Check both rate limits
            current_requests = len(self.request_history)
            current_tokens = sum(m.token_count for m in self.token_history)

            if (
                current_requests < self.requests_per_minute
                and current_tokens + token_count <= self.tokens_per_minute
            ):
                # Add new request to history
                self.request_history.append(current_time)
                if token_count > 0:
                    self.token_history.append(RequestMetrics(current_time, token_count))
                return

            # Wait before checking again
            await asyncio.sleep(0.1)


# Do not retry on these exceptions as it is pointless.
exclude_exceptions = (TypeError, ValueError)


def with_model_fallback(func: Callable) -> Callable:
    """Decorator to attempt flash model first, then fall back to pro model"""
    sig = inspect.signature(func)

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        bound_args = sig.bind(self, *args, **kwargs)
        bound_args.apply_defaults()
        bound_dict = dict(bound_args.arguments)
        bound_dict.pop("is_mini", None)

        try:
            return func(**bound_dict, is_mini=True)
        except exclude_exceptions as e:
            raise e
        except Exception as e:
            if not self.allow_fallback:
                raise
            try:
                return func(**bound_dict, is_mini=False)
            except Exception as e2:
                raise ModelFailedError(f"Both models failed. Flash: {e}, Pro: {e2}")

    return wrapper


def get_tool_call_id(response) -> str:
    """Extract tool call ID from a Gemini response"""
    if not response.candidates or not response.candidates[0].content.parts:
        raise NoToolCallsError(response)
    
    parts = response.candidates[0].content.parts
    function_calls = [part for part in parts if hasattr(part, 'function_call')]
    
    if not function_calls:
        raise NoToolCallsError(response)
    if len(function_calls) > 1:
        raise MultipleToolCallsError(response)
    
    # Generate a simple ID for the function call
    return f"call_{hash(str(function_calls[0]))}"


def get_tool_call_ids(response) -> list[str]:
    """Extract tool call IDs from a Gemini response"""
    if not response.candidates or not response.candidates[0].content.parts:
        raise NoToolCallsError(response)
    
    parts = response.candidates[0].content.parts
    function_calls = [part for part in parts if hasattr(part, 'function_call')]
    
    if not function_calls:
        raise NoToolCallsError(response)
    
    return [f"call_{hash(str(fc))}" for fc in function_calls]


def pydantic_to_gemini_function(model_class: type[BaseModel]) -> dict:
    """Convert a Pydantic model to Gemini function declaration format"""
    schema = model_class.model_json_schema()
    
    return {
        "name": model_class.__name__,
        "description": schema.get("description", model_class.__doc__ or ""),
        "parameters": {
            "type": "object",
            "properties": schema.get("properties", {}),
            "required": schema.get("required", [])
        }
    }


class BaseRegistry:
    """Base registry for LLM function calls and parsing"""

    def __init__(
        self,
        mini_client: Client,
        regular_client: Client | None = None,
        mini_batch_client: Client | None = None,
        regular_batch_client: Client | None = None,
        allow_fallback: bool = True,
    ):
        self.allow_fallback = allow_fallback
        self.mini_client = mini_client
        self.regular_client = regular_client
        self.mini_batch_client = mini_batch_client
        self.regular_batch_client = regular_batch_client

    def _get_client(self, is_mini: bool, batch: bool = False) -> Client:
        if batch:
            return self.mini_batch_client if is_mini else self.regular_batch_client
        return self.mini_client if is_mini else self.regular_client

    def _retry_chat(
        self,
        messages: Sequence[dict],
        tools: list[dict],
        parse_fn: Callable,
        is_mini: bool,
        max_retries: int = 5,
        retry_temperature: float = 0.1,
        tool_choice: Any = None,
    ) -> tuple[Any, Model]:
        """Generic retry logic for chat completions"""
        client = self._get_client(is_mini)
        exceptions = []

        # Convert messages to Gemini format
        gemini_messages = self._convert_messages_to_gemini(messages)
        
        # Configure tools
        tools_config = [genai.protos.Tool(function_declarations=tools)] if tools else None

        for retry in range(max_retries):
            temperature = retry_temperature if retry > 0 else 0
            try:
                response = client.client.generate_content(
                    gemini_messages,
                    tools=tools_config,
                    generation_config=genai.GenerationConfig(temperature=temperature)
                )
                result = parse_fn(response)
                return response, result
            except exclude_exceptions as e:
                raise e
            except Exception as e:
                logging.warning(
                    f"Attempt {retry + 1} failed: {type(e).__name__}: {str(e)}"
                )
                logging.debug(
                    f"Attempt {retry + 1}/{max_retries} with temperature={temperature}"
                )
                logging.debug(f"Messages:\n{messages}")
                logging.debug(f"Tools:\n{tools}")

                exceptions.append(e)

        raise ExceptionGroup(f"Failed after {max_retries} retries", exceptions)

    def _convert_messages_to_gemini(self, messages: Sequence[dict]) -> list:
        """Convert OpenAI-style messages to Gemini format"""
        gemini_messages = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                # System messages become user messages with system instruction
                gemini_messages.append({"role": "user", "parts": [{"text": f"System: {content}"}]})
            elif role == "user":
                gemini_messages.append({"role": "user", "parts": [{"text": content}]})
            elif role == "assistant":
                gemini_messages.append({"role": "model", "parts": [{"text": content}]})
            elif role == "tool":
                # Tool responses become function responses in Gemini
                tool_call_id = msg.get("tool_call_id", "")
                gemini_messages.append({
                    "role": "function",
                    "parts": [{"function_response": {"name": tool_call_id, "response": {"result": content}}}]
                })
        
        return gemini_messages


class FunctionRegistry(BaseRegistry):
    """Registry for parameter-based function calls"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.paramdef_to_function: dict[Model, Callable] = {}
        self.name_to_paramdef: dict[str, Model] = {}

    def __str__(self) -> str:
        return str(self.paramdef_to_function)

    __repr__ = __str__

    def register(self, func: Callable, param_model: type[Model] | None = None) -> None:
        """
        Register a function with optional parameter specification.
        If param_model is not provided, the first parameter must be a Model.
        If param_model is provided, its fields must match the function parameters.
        """
        sig = inspect.signature(func)
        params = list(sig.parameters.values())

        if not params:
            raise ValueError("Function must have at least one parameter")

        if param_model is None:
            # Traditional case: first parameter must be Model
            if not issubclass(params[0].annotation, BaseModel):
                raise ValueError(
                    "Function must have BaseModel parameter specification or provide param_model"
                )
            param_type = params[0].annotation
        else:
            # Verify param_model fields match function parameters
            param_type = param_model
            model_fields = param_model.model_fields

            for param in params:
                if param.name not in model_fields:
                    if param.default is param.empty:
                        raise ValueError(
                            f"Parameter {param.name} not found in {param_model.__name__} "
                            "and has no default value"
                        )

        self.paramdef_to_function[param_type] = func
        self.name_to_paramdef[param_type.__name__] = param_type

    def get_tools(
        self, is_mini: bool, subset: str | list[str] | None = None
    ) -> list[dict]:
        """Get Gemini tools.
        `subset` optionally specifies a single function or a subset of functions.
        """
        if subset is None:
            subset = []
        elif isinstance(subset, str):
            subset = [subset]
        tools = []
        for param_def in self.paramdef_to_function:
            if subset and param_def.__name__ not in subset:
                continue

            tool = pydantic_to_gemini_function(param_def)
            tools.append(tool)
        if subset and len(tools) != len(subset):
            raise ValueError("Could not find all {tools=}. Found {subset=}")
        return tools

    @with_model_fallback
    def call_functions(
        self,
        messages: Sequence[dict],
        function_subset: str | list[str] | None = None,
        target_function: str | None = None,
        max_retries: int = 5,
        is_mini: bool = True,
        retry_temperature: float = 0.1,
    ) -> tuple[Any, list[FunctionCall]]:
        """Call multiple functions using the LLM"""
        if target_function:
            function_subset = [target_function]
        tools = self.get_tools(is_mini, function_subset)
        
        def parse_response(response) -> list[Model]:
            if not response.candidates or not response.candidates[0].content.parts:
                return []
            
            parts = response.candidates[0].content.parts
            function_calls = [part for part in parts if hasattr(part, 'function_call')]
            
            parsed_args = []
            for fc in function_calls:
                param_def = self.name_to_paramdef[fc.function_call.name]
                # Convert function call args to dict
                args_dict = {}
                for key, value in fc.function_call.args.items():
                    args_dict[key] = value
                parsed_args.append(param_def.model_validate(args_dict))
            return parsed_args

        response, parsed_args_list = self._retry_chat(
            messages=messages,
            tools=tools,
            parse_fn=parse_response,
            is_mini=is_mini,
            max_retries=max_retries,
            retry_temperature=retry_temperature,
        )

        results = []
        for parsed_args in parsed_args_list:
            function = self.paramdef_to_function[type(parsed_args)]
            result = function(**parsed_args.model_dump())
            results.append(FunctionCall(arguments=parsed_args, result=result))

        return response, results

    def call_function(
        self,
        messages: Sequence[dict],
        function_subset: str | list[str] | None = None,
        target_function: str | None = None,
        is_mini: bool = True,
    ) -> tuple[Any, FunctionCall]:
        """Call a single function using the LLM, raise exception if multiple tool calls are returned"""
        response, results = self.call_functions(
            messages=messages,
            function_subset=function_subset,
            target_function=target_function,
            is_mini=is_mini,
        )
        if not results:
            raise NoToolCallsError
        if len(results) > 1:
            raise MultipleToolCallsError(f"{response=}\n{results=}")
        return response, results[0]


class ParserRegistry(BaseRegistry):
    """Registry for parsing unstructured responses into structured data"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.response_models: dict[str, Model] = {}

    def __str__(self) -> str:
        return str(self.response_models)

    __repr__ = __str__

    def register(self, model: type[Model]) -> None:
        """Register a response model"""
        self.response_models[model.__name__] = model

    def get_tools(
        self, is_mini: bool, subset: str | list[str] | None = None
    ) -> list[dict]:
        """Get Gemini tools.
        `subset` optionally specifies a single function or a subset of functions.
        """
        if subset is None:
            subset = []
        elif isinstance(subset, str):
            subset = [subset]
        tools = []
        for model_name, model in self.response_models.items():
            if subset and model_name not in subset:
                continue

            tool = pydantic_to_gemini_function(model)
            tools.append(tool)
        if subset and len(tools) != len(subset):
            raise ValueError("Could not find all {tools=}. Found {subset=}")
        return tools

    @with_model_fallback
    def parse_responses(
        self,
        messages: Sequence[dict],
        model_subset: str | list[str] | None = None,
        target_model: str | None = None,
        is_mini: bool = True,
        max_retries: int = 5,
    ) -> tuple[Any, list[Model]]:
        """Parse multiple unstructured responses into structured data"""
        if target_model:
            model_subset = [target_model]
        tools = self.get_tools(is_mini, subset=model_subset)

        def parse_result(response) -> list[Model]:
            if not response.candidates or not response.candidates[0].content.parts:
                return []
            
            parts = response.candidates[0].content.parts
            function_calls = [part for part in parts if hasattr(part, 'function_call')]
            
            parsed_results = []
            for fc in function_calls:
                response_model = self.response_models[fc.function_call.name]
                # Convert function call args to dict
                args_dict = {}
                for key, value in fc.function_call.args.items():
                    args_dict[key] = value
                parsed_results.append(response_model.model_validate(args_dict))
            return parsed_results

        response, parsed_results = self._retry_chat(
            messages=messages,
            tools=tools,
            parse_fn=parse_result,
            is_mini=is_mini,
            max_retries=max_retries,
        )

        return response, parsed_results

    def parse_response(
        self,
        messages: Sequence[dict] | list[dict],
        model_subset: str | list[str] | None = None,
        target_model: str | None = None,
        is_mini: bool = True,
    ) -> tuple[Any, Model]:
        """Parse a single unstructured response into structured data, raise exception if multiple tool calls are returned"""
        response, results = self.parse_responses(
            messages=messages,  # type: ignore
            model_subset=model_subset,
            target_model=target_model,
            is_mini=is_mini,
        )
        if not results:
            raise NoToolCallsError
        if len(results) > 1:
            raise MultipleToolCallsError(f"{response=}\n{results=}")
        return response, results[0]