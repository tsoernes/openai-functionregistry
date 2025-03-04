"""
Defines a `ParserRegistry` and a `FunctionRegistry` to make it convenient
"""

from datetime import datetime, timedelta
from typing import Any, TypeVar, Sequence
from collections import deque
import asyncio
import json
import time
import random
import string
import inspect
import logging
from dataclasses import dataclass
from functools import wraps
from typing import Any, TypeVar
from collections.abc import Callable

from betterpathlib import Path
import openai
from openai import NOT_GIVEN
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from openai.types.chat.chat_completion import ChatCompletion
from pydantic import BaseModel

from openai_functionregistry.client import Client

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

    def __init__(self, requests_per_minute: int, tokens_per_minute: int, window_size: int = 60):
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

            if (current_requests < self.requests_per_minute and
                current_tokens + token_count <= self.tokens_per_minute):
                # Add new request to history
                self.request_history.append(current_time)
                if token_count > 0:
                    self.token_history.append(RequestMetrics(current_time, token_count))
                return

            # Wait before checking again
            await asyncio.sleep(0.1)

# Do not retry on these exceptions as it is pointless.
exclude_exceptions = (TypeError, openai.BadRequestError)


def with_model_fallback(func: Callable) -> Callable:
    """Decorator to attempt mini-model first, then fall back to regular model"""
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
        self,
        mini_client: Client,
        regular_client: Client,
        mini_async_client: Client,
        regular_async_client: Client,
        mini_batch_client: Client,
        regular_batch_client: Client,
        allow_fallback: bool = True,
    ):
        self.allow_fallback = allow_fallback
        self.mini_client = mini_client
        self.regular_client = regular_client
        self.mini_async_client = mini_async_client
        self.regular_async_client = regular_async_client
        self.mini_batch_client = mini_batch_client
        self.regular_batch_client = regular_batch_client

    def _get_client(self, is_mini: bool, async_: bool = False, batch: bool = False) -> Client:
        if async_:
            return self.mini_async_client if is_mini else self.regular_async_client
        elif batch:
            return self.mini_batch_client if is_mini else self.regular_batch_client
        return self.mini_client if is_mini else self.regular_client

    def _retry_chat(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        tools: list[ChatCompletionToolParam],
        parse_fn: Callable[[ChatCompletion], Model],
        is_mini: bool,
        max_retries: int = 5,
        retry_temperature: float = 0.1,
        tool_choice: Any = NOT_GIVEN,
    ) -> tuple[ChatCompletion, Model]:
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


class FunctionRegistry(BaseRegistry):
    """Registry for parameter-based function calls"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.paramdef_to_function: dict[Model, Callable] = {}
        self.name_to_paramdef: dict[str, Model] = {}

    def __str__(self) -> str:
        return str(self.paramdef_to_function)

    __repr__ = __str__

    def register(
        self, func: Callable, param_model: type[Model] | None = None
    ) -> None:
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
        function_subset: str | list[str] | None = None,
        target_function: str | None = None,
        max_retries: int = 5,
        is_mini: bool = True,
        retry_temperature: float = 0.1,
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

        def parse_response(response: ChatCompletion) -> list[Model]:
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
        function_subset: str | list[str] | None = None,
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
        model_subset: str | list[str] | None = None,
        target_model: str | None = None,
        is_mini: bool = True,
        max_retries: int = 5,
    ) -> tuple[ChatCompletion, list[Model]]:
        """Parse multiple unstructured responses into structured data"""
        if target_model:
            model_subset = [target_model]
        tools = self.get_tools(is_mini, subset=model_subset)
        tool_choice = (
            {"type": "function", "function": {"name": target_model}}
            if target_model
            else NOT_GIVEN
        )

        def parse_result(response: ChatCompletion) -> list[Model]:
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

        return response, parsed_results

    def parse_response(
        self,
        messages: Sequence[ChatCompletionMessageParam] | list[dict[str, str]],
        model_subset: str | list[str] | None = None,
        target_model: str | None = None,
        is_mini: bool = True,
    ) -> tuple[ChatCompletion, Model]:
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

    def parse_responses_batch(
        self,
        messages_list: Sequence[Sequence[ChatCompletionMessageParam]],
        model_subset: str | list[str] | None = None,
        target_model: str | None = None,
        is_mini: bool = True,
        max_retries: int = 5,
        custom_id: str | None = None,
        sleep_time: float = 10,
        temperature: float = 0,
    ) -> list[tuple[ChatCompletion, list[Model]]]:
        """Parse multiple lists of unstructured responses into structured data using the Azure OpenAI Batch API"""
        if custom_id is None:
            custom_id = f"{self._generate_random_string()}"
        tools = self.get_tools(is_mini, subset=model_subset)
        tool_choice = (
            {"type": "function", "function": {"name": target_model}}
            if target_model
            else NOT_GIVEN
        )
        batch_requests = {}
        for i, messages in enumerate(messages_list):
            custom_id_ = f"{custom_id}-{i}"
            batch_requests[custom_id_] = {
                "custom_id": custom_id_,
                "method": "POST",
                "url": "/chat/completions",
                "body": {
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": tool_choice,
                    "model": self.mini_batch_client.model if is_mini else self.regular_batch_client.model,
                    "temperature": temperature,
                },
            }

        batch_file_path = Path.tempdir() / f"{custom_id}.jsonl"
        with open(batch_file_path, "w") as f:
            for request in batch_requests.values():
                f.write(json.dumps(request) + "\n")

        client = self._get_client(is_mini=is_mini, batch=True).client

        batch_file = client.files.create(
            file=open(batch_file_path, "rb"), purpose="batch"
        )
        logging.info("Batch File ID: %s", batch_file.id)

        # Wait for file upload
        while True:
            batch_file = client.files.retrieve(batch_file.id)
            if batch_file.status != 'pending':
                break
            else:
                time.sleep(1)
        if batch_file.status != "processed":
            raise Exception(f"File upload failed {batch_file}")  # type:ignoe
        # Initialize batch jobs from uploaded file
        batch_job = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        logging.info("Batch Job ID: %s", batch_job.id)

        start_time = datetime.now()
        while True:
            batch_job = client.batches.retrieve(batch_job.id)
            if batch_job.status in ["completed", "failed"]:
                break
            time.sleep(sleep_time)

        duration = datetime.now() - start_time
        logging.info(f"Batch Job finished in {duration} with status: %s", batch_job.status)

        if batch_job.status == "completed":
            output_file = client.files.content(batch_job.output_file_id)  # type: ignore
            results = []

            for line in output_file.iter_lines():
                try:
                    response = json.loads(line)
                    tool_calls = response["response"]["body"]["choices"][0]["message"][
                        "tool_calls"
                    ]
                    parsed_results = []
                    for tool_call in tool_calls:
                        try:
                            response_model = self.response_models[
                                tool_call["function"]["name"]
                            ]
                            model_inst = response_model.model_validate_json(
                                tool_call["function"]["arguments"]
                            )
                            parsed_results.append(model_inst)
                        except Exception as e:
                            logging.error(f"Failed to parse model: {e}")
                            logging.error(f"{line=}")
                            logging.error(f"{response_model=}")
                            break
                    else:
                        results.append((response, parsed_results))
                except Exception as e:
                    logging.error(f"Failed to decode {line=}\n{e}")
            # Use regular non-batch API for failed requests
            failed_ids = list(batch_requests.keys() - set(r[0]["custom_id"] for r in results))
            if failed_ids:
                failed_ids.sort()
                failed_messages_list = [
                    batch_requests[id_]["body"]["messages"] for id_ in failed_ids
                ]
                more_results = self.parse_responses_batch_async(
                    messages_list=failed_messages_list,
                    model_subset=model_subset,
                    target_model=target_model,
                    is_mini=is_mini,
                    max_retries=max_retries - 1,
                    init_temperature=0.1,
                )
                more_results = list(zip(failed_ids, more_results))
                results.extend(more_results)
            # sort by custom_id in order to maintain original order
            results.sort(key=lambda t: t[0]["custom_id"])
            return results
        else:
            raise Exception(f"Batch job failed: {batch_job.status}\n{batch_job.errors}")  # type:ignore

    def _generate_random_string(self, length: int = 3) -> str:
        dt = datetime.now().strftime("%Y-%m-%dT%H%M")
        return (
            dt
            + "_"
            + "".join(random.choices(string.ascii_letters + string.digits, k=length))
        )

    def parse_responses_batch_async(
        self,
        messages_list: Sequence[Sequence[ChatCompletionMessageParam]],
        model_subset: str | list[str] | None = None,
        target_model: str | None = None,
        is_mini: bool = True,
        max_retries: int = 5,
        init_temperature: float = 0,
    ) -> list[tuple[ChatCompletion, list[Model]]]:
        """
        Parse multiple unstructured responses into structured data for a batch of
        messages asynchronously.
        """

        async def async_wrapper():
            results = await self._parse_responses_batch_async(
                messages_list=messages_list,
                model_subset=model_subset,
                target_model=target_model,
                is_mini=is_mini,
                max_retries=max_retries,
                init_temperature=init_temperature,
            )
            return results

        results = asyncio.run(async_wrapper())
        return results

    async def _parse_responses_batch_async(
        self,
        messages_list: Sequence[Sequence[ChatCompletionMessageParam]],
        model_subset: str | list[str] | None = None,
        target_model: str | None = None,
        is_mini: bool = True,
        max_retries: int = 2,
        init_temperature: float = 0,
    ) -> list[tuple[ChatCompletion, list[Model]]]:
        """Parse multiple unstructured responses into structured data for a batch of messages asynchronously."""
        if target_model:
            model_subset = [target_model]
        tools = self.get_tools(is_mini, subset=model_subset)
        tool_choice = (
            {"type": "function", "function": {"name": target_model}}
            if target_model
            else NOT_GIVEN
        )

        client = self._get_client(is_mini, async_=True)
        # Initialize rate limiter
        rate_limiter = AsyncRateLimiter(
            requests_per_minute=client.requests_per_minute_limit,
            tokens_per_minute=client.tokens_per_minute_limit
        )

        async def parse_result(response: ChatCompletion) -> list[Model]:
            tool_calls = response.choices[0].message.tool_calls
            if not tool_calls:
                return []
            parsed_results = []
            for tool_call in tool_calls:
                response_model = self.response_models[tool_call.function.name]
                model_inst = response_model.model_validate_json(tool_call.function.arguments)
                parsed_results.append(model_inst)
            return parsed_results

        async def process_single_request(
            messages: Sequence[ChatCompletionMessageParam],
        ) -> tuple[ChatCompletion, list[Model]]:
            exceptions = []

            client = self._get_client(is_mini=True, async_=True)

            # Calculate token count for this request
            token_count = len(client.encoder.encode_batch(
                [f"{m['role']} {m['content']}" for m in messages]
            ))

            for retry_n in range(max_retries):
                temperature = 0.1 if retry_n > 0 else init_temperature
                try:
                    await rate_limiter.acquire(token_count)
                    response = await client.client.chat.completions.create(
                        model=client.model,
                        messages=messages,
                        tools=tools,
                        temperature=temperature,
                        tool_choice=tool_choice,  # type: ignore
                    )
                    result = await parse_result(response)
                    return response, result
                except exclude_exceptions as e:
                    raise e
                except Exception as e:
                    logging.warning(
                        f"Attempt {retry_n + 1} failed with mini model: {type(e).__name__}: {str(e)}"
                    )
                    exceptions.append(e)

            if self.allow_fallback:
                client = self._get_client(is_mini=False, async_=True)
                for retry in range(max_retries):
                    temperature = 0.1 if retry > 0 else init_temperature
                    try:
                        await rate_limiter.acquire(token_count)
                        response = await client.client.chat.completions.create(
                            model=client.model,
                            messages=messages,
                            tools=tools,
                            temperature=temperature,
                            tool_choice=tool_choice,  # type: ignore
                        )
                        result = await parse_result(response)
                        return response, result
                    except exclude_exceptions as e:
                        raise e
                    except Exception as e:
                        logging.warning(
                            f"Attempt {retry + 1} failed with regular model: {type(e).__name__}: {str(e)}"
                        )
                        exceptions.append(e)

                raise ExceptionGroup(
                    f"Failed after {max_retries} retries with both models", exceptions
                )
            raise ExceptionGroup(
                f"Failed after {max_retries} retries with mini models", exceptions
            )

        async def rate_limited_process(messages):
            return await process_single_request(messages)

        tasks = [rate_limited_process(messages) for messages in messages_list]
        results_and_errs = await asyncio.gather(*tasks, return_exceptions=True)
        results = []

        for result_or_err in results_and_errs:
            if isinstance(result_or_err, Exception):
                logging.error(f"Error processing batch request: {result_or_err}")
            else:
                results.append(result_or_err)

        return results
