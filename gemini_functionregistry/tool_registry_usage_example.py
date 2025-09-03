"""
Shows example usage of the Gemini `FunctionRegistry` and `ParserRegistry`.
"""

import random
from enum import StrEnum

from pydantic import BaseModel, Field

from gemini_functionregistry.client import Client
from gemini_functionregistry.tool_registry import (FunctionRegistry,
                                                   ParserRegistry,
                                                   get_tool_call_id,
                                                   get_tool_call_ids)


class TemperatureUnit(StrEnum):
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"


class GetWeather(BaseModel):
    """Get the weather at the given location in the given temperature unit of measurement (doc1)"""

    location: str = Field(description="The location to get the weather for")
    unit: TemperatureUnit = Field(description="Temperature Unit")


def get_weather(location: str, unit: TemperatureUnit = TemperatureUnit.CELSIUS) -> str:
    """Get the weather at the given location in the given temperature unit of measurement (doc2)"""
    return f"The weather in {location} is {random.randint(20, 40)} degrees {unit} with a windy breeze."


def get_weather_alt(params: GetWeather) -> str:
    """Get the weather at the given location in the given temperature unit of measurement (doc3)"""
    location = params.location
    unit = params.unit
    return f"The weather in {location} is {random.randint(20, 40)} degrees {unit} with a windy breeze."


class WeatherResponse(BaseModel):
    """The temperature in a given unit at a specified location"""

    location: str
    temperature: float
    unit: str


def main():
    # Create shared clients
    # Note: You need to set GEMINI_API_KEY environment variable
    mini_client = Client(model="gemini-1.5-flash")
    regular_client = Client(model="gemini-1.5-pro")

    # Register function calls
    func_registry = FunctionRegistry(mini_client, regular_client)
    # Register with explicit param model
    func_registry.register(get_weather, GetWeather)
    # Or register a function that takes a parameter spec as the first argument and autodetect parameter model
    # func_registry.register(get_weather_alt)

    # Make function call
    messages = [
        {"role": "system", "content": "Extract the weather information."},
        {"role": "user", "content": "What's the weather like in London?"},
    ]
    response1, function_call1 = func_registry.call_function(
        messages, target_function="GetWeather"
    )

    # Parse the result
    parser_registry = ParserRegistry(mini_client, regular_client)
    parser_registry.register(WeatherResponse)

    # The previous messages are not required in order to parse the response.
    parse_messages = [
        {"role": "assistant", "content": "I'll get the weather information for you."},
        {
            "role": "tool",
            "content": function_call1.result,
            "tool_call_id": get_tool_call_id(response1),
        },
    ]

    response2, weather_response = parser_registry.parse_response(
        messages=parse_messages,
        target_model="WeatherResponse",
    )

    print(f"Parsed response: {weather_response}")

    ### TRY MULTIPLE TOOL CALLS. Get the weather for London and New York at the same time.

    # Make function call
    messages = [
        {"role": "system", "content": "Extract the weather information."},
        {"role": "user", "content": "What's the weather like in London and New York?"},
    ]

    # NOTE: When `func_registry.call_functions` is given a `target_function`
    # then it only returns one input even though it might be proper to return many.
    # Use `function_subset` to attempt restrict the model to a specific function instead

    response3, function_calls3 = func_registry.call_functions(messages)

    # The previous messages are not required in order to parse the response.
    parse_messages = [
        {
            "role": "system",
            "content": "Call the supplied functions in order to structure the given information:",
        },
        {"role": "assistant", "content": "I'll get the weather for both cities."},
    ]
    for function_call, tool_call_id in zip(
        function_calls3, get_tool_call_ids(response3)
    ):
        parse_messages.append(
            {
                "role": "tool",
                "content": function_call.result,
                "tool_call_id": tool_call_id,
            },
        )

    response4, weather_responses = parser_registry.parse_responses(
        messages=parse_messages,
        model_subset=["WeatherResponse"],
    )

    print(f"Parsed responses: {weather_responses}")


def demo_without_api_key():
    """Demo that doesn't require an actual API key"""
    print("Demo: Gemini Function Registry")
    print("=" * 40)
    
    # Show function registration
    try:
        # This will fail without API key, but shows the setup
        mini_client = Client(api_key="demo_key", model="gemini-1.5-flash")
    except Exception as e:
        print(f"Note: Would need valid API key to run. Error: {e}")
        
    print("\nFunction definitions:")
    print(f"GetWeather model: {GetWeather}")
    print(f"WeatherResponse model: {WeatherResponse}")
    
    print("\nExample messages:")
    messages = [
        {"role": "system", "content": "Extract the weather information."},
        {"role": "user", "content": "What's the weather like in London?"},
    ]
    print(f"Messages: {messages}")
    
    print("\nTo run with actual API:")
    print("1. Set GEMINI_API_KEY environment variable")
    print("2. Run: poetry run python gemini_functionregistry/tool_registry_usage_example.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error running main (likely missing API key): {e}")
        print("\nRunning demo instead...")
        demo_without_api_key()