"""
Defines a `ParserRegistry` and a `FunctionRegistry` to make it convenient
"""

import inspect
import logging
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Sequence, Type, TypeVar

import openai
from openai import NOT_GIVEN
from openai.types.chat import (ChatCompletionMessageParam,
                               ChatCompletionToolParam)
from openai.types.chat.chat_completion import ChatCompletion
from pydantic import BaseModel

from openai_functionregistry.client import Client


@dataclass
class FunctionCall:
    "Function call arguments and the result of the function call."
    arguments: Type[BaseModel]
    result: Any


T = TypeVar("T", bound=BaseModel)


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


# Do not retry on these exceptions as it is pointless.
exclude_exceptions = (TypeError, openai.BadRequestError)


def with_model_fallback(func):
    """Decorator to attempt mini-model first, then fall back to regular model"""
    sig = inspect.signature(func)

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Bind the arguments to the function signature
        bound_args = sig.bind(self, *args, **kwargs)
        bound_args.apply_defaults()
        # Remove is_mini from bound arguments if present
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
                raise ModelFailedError(f"Both models failed. Mini: {e}, Regular: {e2}")

    return wrapper


def get_tool_call_id(response: ChatCompletion) -> str:
    """Extract tool call ID from a chat completion response"""
    tool_calls = response.choices[0].message.tool_calls
    if not tool_calls:
        raise NoToolCallsError(response)
    if len(tool_calls) > 2:
        raise MultipleToolCallsError(response)
    return tool_calls[0].id


def get_tool_call_ids(response: ChatCompletion) -> list[str]:
    """Extract tool call IDs from a chat completion response"""
    tool_calls = response.choices[0].message.tool_calls
    if not tool_calls:
        raise NoToolCallsError(response)
    return [t.id for t in tool_calls]


class BaseRegistry:
    """Base registry for LLM function calls and parsing"""

    def __init__(
        self, mini_client: Client, regular_client: Client, allow_fallback: bool = True
    ):
        self.allow_fallback = allow_fallback
        self.mini_client = mini_client
        self.regular_client = regular_client

    def _get_client(self, is_mini: bool) -> Client:
        return self.mini_client if is_mini else self.regular_client

    def _retry_chat(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        tools: list[ChatCompletionToolParam],
        parse_fn: Callable[[ChatCompletion], T],
        is_mini: bool,
        max_retries: int = 5,
        retry_temperature: float = 0.1,
        tool_choice: Any = NOT_GIVEN,
    ) -> tuple[ChatCompletion, T]:
        """Generic retry logic for chat completions"""
        client = self._get_client(is_mini)
        exceptions = []

        for retry in range(max_retries):
            temperature = retry_temperature if retry > 0 else 0
            try:
                response = client.client.chat.completions.create(
                    model=client.model,
                    messages=messages,
                    tools=tools,
                    temperature=temperature,
                    tool_choice=tool_choice,
                )
                result = parse_fn(response)
                return response, result
            except exclude_exceptions as e:
                raise e
            except Exception as e:
                logging.info(
                    f"Attempt {retry + 1} failed: {type(e).__name__}: {str(e)}"
                )
                logging.debug(
                    f"Attempt {retry + 1}/{max_retries} with temperature={temperature}"
                )
                logging.debug(f"Messages:\n{messages}")
                logging.debug(f"Tools:\n{tools}")

                exceptions.append(e)

        raise ExceptionGroup(f"Failed after {max_retries} retries", exceptions)


class FunctionRegistry(BaseRegistry):
    """Registry for parameter-based function calls"""

    def __init__(
        self, mini_client: Client, regular_client: Client, allow_fallback: bool = True
    ):
        super().__init__(mini_client, regular_client, allow_fallback)
        self.paramdef_to_function: dict[Type[BaseModel], Callable] = {}
        self.name_to_paramdef: dict[str, Type[BaseModel]] = {}

    def __str__(self) -> str:
        return str(self.paramdef_to_function)

    __repr__ = __str__

    def register(
        self, func: Callable, param_model: Type[BaseModel] | None = None
    ) -> None:
        """
        Register a function with optional parameter specification.
        If param_model is not provided, the first parameter must be a BaseModel.
        If param_model is provided, its fields must match the function parameters.
        """
        sig = inspect.signature(func)
        params = list(sig.parameters.values())

        if not params:
            raise ValueError("Function must have at least one parameter")

        if param_model is None:
            # Traditional case: first parameter must be BaseModel
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
                else:
                    # Could add type checking here if desired
                    pass

        self.paramdef_to_function[param_type] = func
        self.name_to_paramdef[param_type.__name__] = param_type

    def get_tools(
        self, is_mini: bool, subset: None | str | list[str] = None
    ) -> list[ChatCompletionToolParam]:
        """Get OpenAI tools with appropriate strictness.
        `target_functions` optionally specifies a single function or a subset of functions.
        """
        if subset is None:
            subset = []
        elif isinstance(subset, str):
            subset = [subset]
        tools = []
        for param_def in self.paramdef_to_function:
            if subset and param_def.__name__ not in subset:
                continue

            tool = openai.pydantic_function_tool(param_def)
            if is_mini:
                del tool["function"]["strict"]
            tools.append(tool)
        if subset and len(tools) != len(subset):
            raise ValueError("Could not find all {tools=}. Found {subset=}")
        return tools

    @with_model_fallback
    def call_functions(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        function_subset=None,
        target_function: None | str = None,
        max_retries: int = 5,
        is_mini: bool = True,
        retry_temperature=0.1,
    ) -> tuple[ChatCompletion, list[FunctionCall]]:
        """Call multiple functions using the LLM"""
        if target_function:
            function_subset = [target_function]
        tools = self.get_tools(is_mini, function_subset)
        tool_choice = (
            {"type": "function", "function": {"name": target_function}}
            if target_function
            else NOT_GIVEN
        )

        def parse_response(response: ChatCompletion) -> list[BaseModel]:
            tool_calls = response.choices[0].message.tool_calls
            if not tool_calls:
                return []
            parsed_args = []
            for tool_call in tool_calls:
                param_def = self.name_to_paramdef[tool_call.function.name]
                parsed_args.append(
                    param_def.model_validate_json(tool_call.function.arguments)
                )
            return parsed_args

        response, parsed_args_list = self._retry_chat(
            messages=messages,
            tools=tools,
            parse_fn=parse_response,
            is_mini=is_mini,
            tool_choice=tool_choice,
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
        messages: Sequence[ChatCompletionMessageParam],
        function_subset=None,
        target_function: str | None = None,
        is_mini: bool = True,
    ) -> tuple[ChatCompletion, FunctionCall]:
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

    def __init__(
        self, mini_client: Client, regular_client: Client, allow_fallback: bool = True
    ):
        super().__init__(mini_client, regular_client, allow_fallback)
        self.response_models: dict[str, Type[BaseModel]] = {}

    def __str__(self) -> str:
        return str(self.response_models)

    __repr__ = __str__

    def register(self, model: Type[BaseModel]) -> None:
        """Register a response model"""
        self.response_models[model.__name__] = model

    def get_tools(
        self, is_mini: bool, subset: None | str | list[str] = None
    ) -> list[ChatCompletionToolParam]:
        """Get OpenAI tools with appropriate strictness.
        `target_functions` optionally specifies a single function or a subset of functions.
        """
        if subset is None:
            subset = []
        elif isinstance(subset, str):
            subset = [subset]
        tools = []
        for model_name, model in self.response_models.items():
            if subset and model_name not in subset:
                continue

            tool = openai.pydantic_function_tool(model)
            if is_mini:
                del tool["function"]["strict"]
            tools.append(tool)
        if subset and len(tools) != len(subset):
            raise ValueError("Could not find all {tools=}. Found {subset=}")
        return tools

    @with_model_fallback
    def parse_responses(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        model_subset: None | str | list[str] = None,
        target_model: None | str = None,
        is_mini: bool = True,
        max_retries: int = 5,
    ) -> tuple[ChatCompletion, list[BaseModel]]:
        """Parse multiple unstructured responses into structured data"""
        if target_model:
            model_subset = [target_model]
        tools = self.get_tools(is_mini, subset=model_subset)
        tool_choice = (
            {"type": "function", "function": {"name": target_model}}
            if target_model
            else NOT_GIVEN
        )

        def parse_result(response: ChatCompletion) -> list[BaseModel]:
            tool_calls = response.choices[0].message.tool_calls
            if not tool_calls:
                return []
            parsed_results = []
            for tool_call in tool_calls:
                response_model = self.response_models[tool_call.function.name]
                parsed_results.append(
                    response_model.model_validate_json(tool_call.function.arguments)
                )
            return parsed_results

        response, parsed_results = self._retry_chat(
            messages=messages,
            tools=tools,
            parse_fn=parse_result,
            is_mini=is_mini,
            tool_choice=tool_choice,
            max_retries=max_retries,
        )

        return (response, parsed_results)

    def parse_response(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        model_subset: None | str | list[str] = None,
        target_model: str | None = None,
        is_mini: bool = True,
    ) -> tuple[ChatCompletion, BaseModel]:
        """Parse a single unstructured response into structured data, raise exception if multiple tool calls are returned"""
        response, results = self.parse_responses(
            messages=messages,
            model_subset=model_subset,
            target_model=target_model,
            is_mini=is_mini,
        )
        if not results:
            raise NoToolCallsError
        if len(results) > 1:
            raise MultipleToolCallsError(f"{response=}\n{results=}")
        return response, results[0]
