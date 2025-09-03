# OpenAI vs Gemini Function Registry Comparison

This document shows the key differences between the original OpenAI Function Registry and the new Gemini Function Registry.

## Installation & Setup

### OpenAI Version
```python
from openai_functionregistry import Client, FunctionRegistry, ParserRegistry

# Azure OpenAI setup
client = Client(
    endpoint="https://your-endpoint.azure.com/",
    api_key="your-azure-key",
    model="gpt-4o-mini",
    api_version="2024-05-01-preview"
)
```

### Gemini Version
```python
from gemini_functionregistry import Client, FunctionRegistry, ParserRegistry

# Google AI setup
client = Client(
    api_key="your-google-ai-key",  # or set GEMINI_API_KEY env var
    model="gemini-1.5-flash"
)
```

## Function Registration

Both versions use identical API for function registration:

```python
from pydantic import BaseModel, Field
from enum import StrEnum

class TemperatureUnit(StrEnum):
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"

class GetWeather(BaseModel):
    location: str = Field(description="The location to get weather for")
    unit: TemperatureUnit = Field(description="Temperature unit")

def get_weather(location: str, unit: TemperatureUnit = TemperatureUnit.CELSIUS) -> str:
    return f"Weather in {location}: 20°{unit}"

# Same API for both versions
func_registry = FunctionRegistry(mini_client, regular_client)
func_registry.register(get_weather, GetWeather)
```

## Function Calling

Identical API:

```python
messages = [
    {"role": "system", "content": "Get weather information"},
    {"role": "user", "content": "What's the weather in London?"}
]

response, function_call = func_registry.call_function(
    messages, 
    target_function="GetWeather"
)
```

## Response Parsing

Identical API:

```python
class WeatherResponse(BaseModel):
    location: str
    temperature: float
    unit: str

parser_registry = ParserRegistry(mini_client, regular_client)
parser_registry.register(WeatherResponse)

response, parsed = parser_registry.parse_response(
    messages,
    target_model="WeatherResponse"
)
```

## Key Technical Differences

### Function Declaration Format

**OpenAI:**
```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get weather information",
    "parameters": {
      "type": "object",
      "properties": {...},
      "required": [...]
    }
  }
}
```

**Gemini:**
```python
protos.FunctionDeclaration(
    name="get_weather",
    description="Get weather information", 
    parameters=protos.Schema(
        type_=protos.Type.OBJECT,
        properties={...},
        required=[...]
    )
)
```

### Message Format

**OpenAI:** Direct dictionary format
```python
{"role": "user", "content": "Hello"}
```

**Gemini:** Protobuf format
```python
protos.Content(
    role="user",
    parts=[protos.Part(text="Hello")]
)
```

### Models & Pricing

**OpenAI:**
- Models: `gpt-4o`, `gpt-4o-mini`
- Pricing: ~$3-10 per 1M tokens
- Fallback: mini → regular

**Gemini:**
- Models: `gemini-1.5-flash`, `gemini-1.5-pro`
- Pricing: ~$0.075-3.5 per 1M tokens (much cheaper!)
- Fallback: flash → pro

### Rate Limits

**OpenAI:** 4,500 requests/min, 450K tokens/min

**Gemini:** 1,500 requests/min, 1M tokens/min

## Migration Guide

1. **Change imports:**
   ```python
   # From:
   from openai_functionregistry import Client, FunctionRegistry, ParserRegistry
   
   # To:
   from gemini_functionregistry import Client, FunctionRegistry, ParserRegistry
   ```

2. **Update client setup:**
   ```python
   # From:
   client = Client(endpoint="...", api_key="...", model="gpt-4o-mini", api_version="...")
   
   # To:
   client = Client(api_key="...", model="gemini-1.5-flash")
   ```

3. **Set environment variable:**
   ```bash
   # From:
   export AZURE_OPENAI_API_KEY="..."
   
   # To:
   export GEMINI_API_KEY="..."
   ```

4. **Update model names:**
   ```python
   # From:
   mini_client = Client(model="gpt-4o-mini")
   regular_client = Client(model="gpt-4o")
   
   # To:
   mini_client = Client(model="gemini-1.5-flash")
   regular_client = Client(model="gemini-1.5-pro")
   ```

5. **No other code changes needed!** The function registration, calling, and parsing APIs are identical.

## Benefits of Gemini Version

- **Cost:** ~10-100x cheaper than OpenAI
- **Speed:** Often faster response times
- **Simplicity:** Simpler setup (no Azure endpoints)
- **Features:** Same function calling capabilities
- **Compatibility:** Identical API to original

## Get Started

1. Get API key: https://makersuite.google.com/app/apikey
2. Install: `poetry install`
3. Set key: `export GEMINI_API_KEY="your-key"`
4. Run: `poetry run python gemini_functionregistry/tool_registry_usage_example.py`