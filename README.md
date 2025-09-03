# Gemini Function Registry

A function registry to interact with Google Gemini models function (tool) calling API.

This is a clone of the OpenAI Function Registry that works with Google Gemini models instead of OpenAI/Azure models, while maintaining the exact same API for easy migration.

## 🚀 Features

- **FunctionRegistry**: Register Python functions to be called by Gemini models
- **ParserRegistry**: Parse unstructured responses into structured Pydantic models  
- **Client**: Wrapper around Google Generative AI client with cost calculation
- **Rate limiting**: Built-in rate limiting for API requests
- **Model fallback**: Automatic fallback from Flash to Pro models on failure
- **Cost tracking**: Track usage costs with detailed token counting
- **Enum support**: Full support for Pydantic enums and complex types

## 💰 Cost Comparison

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|-------------------------|
| **Gemini 1.5 Flash** | $0.075 | $0.30 |
| **Gemini 1.5 Pro** | $3.50 | $10.50 |
| GPT-4o mini | $0.15 | $0.60 |
| GPT-4o | $2.50 | $10.00 |

Gemini Flash is **2x cheaper** than GPT-4o mini and **33x cheaper** than GPT-4o!

## 📦 Installation

```bash
poetry install
```

## 🔧 Setup

Get your Google AI API key from [Google AI Studio](https://makersuite.google.com/app/apikey).

Set your API key:

```bash
export GEMINI_API_KEY="your-api-key-here"
```

Or create a `.env` file:

```
GEMINI_API_KEY=your-api-key-here
```

## 🏃‍♂️ Quick Start

```python
from gemini_functionregistry import Client, FunctionRegistry, ParserRegistry
from pydantic import BaseModel, Field
from enum import StrEnum

# Setup clients
mini_client = Client(model="gemini-1.5-flash")
regular_client = Client(model="gemini-1.5-pro")

# Define function parameters
class TemperatureUnit(StrEnum):
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"

class GetWeather(BaseModel):
    location: str = Field(description="The location to get weather for")
    unit: TemperatureUnit = Field(description="Temperature unit")

# Define function
def get_weather(location: str, unit: TemperatureUnit = TemperatureUnit.CELSIUS) -> str:
    return f"The weather in {location} is 22°{unit} with clear skies."

# Register function
func_registry = FunctionRegistry(mini_client, regular_client)
func_registry.register(get_weather, GetWeather)

# Call function
messages = [
    {"role": "system", "content": "You are a weather assistant."},
    {"role": "user", "content": "What's the weather like in London?"},
]

response, function_call = func_registry.call_function(
    messages, 
    target_function="GetWeather"
)

print(f"Result: {function_call.result}")
# Output: "Result: The weather in London is 22°celsius with clear skies."
```

## 📚 Examples

See `gemini_functionregistry/tool_registry_usage_example.py` for complete examples including:

- Function calling with multiple parameters
- Response parsing into structured data
- Multiple function calls in one request
- Error handling and model fallback

## 🧪 Testing

Run tests without requiring an API key:

```bash
poetry run python test_gemini_registry.py
```

## 🔄 Migration from OpenAI Function Registry

The API is **100% compatible**! Just change your imports and client setup:

```python
# From:
from openai_functionregistry import Client, FunctionRegistry, ParserRegistry
client = Client(endpoint="...", api_key="...", model="gpt-4o-mini")

# To:
from gemini_functionregistry import Client, FunctionRegistry, ParserRegistry  
client = Client(api_key="...", model="gemini-1.5-flash")
```

See [COMPARISON.md](COMPARISON.md) for detailed migration guide.

## 🏗️ Architecture

```
gemini_functionregistry/
├── client.py              # Google AI client wrapper
├── tool_registry.py       # Function & parser registries  
├── utils.py               # Utility functions
└── tool_registry_usage_example.py  # Examples
```

## 🤝 Original Project

This is based on the [OpenAI Function Registry](https://github.com/tsoernes/openai-functionregistry) by Torstein Sørnes.

## 📄 License

Same as original project.
