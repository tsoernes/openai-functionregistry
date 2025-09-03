# Gemini Function Registry

A function registry to interact with Google Gemini models function (tool) calling API.

This is a clone of the OpenAI Function Registry that works with Google Gemini models instead of OpenAI/Azure models.

## Features

- **FunctionRegistry**: Register Python functions to be called by Gemini models
- **ParserRegistry**: Parse unstructured responses into structured Pydantic models  
- **Client**: Wrapper around Google Generative AI client with cost calculation
- **Rate limiting**: Built-in rate limiting for API requests
- **Model fallback**: Automatic fallback from Flash to Pro models on failure

## Installation

```bash
poetry install
```

## Setup

Set your Google AI API key:

```bash
export GEMINI_API_KEY="your-api-key-here"
```

Or create a `.env` file:

```
GEMINI_API_KEY=your-api-key-here
```

## Usage

See `gemini_functionregistry/tool_registry_usage_example.py` for complete examples.

## Migration from OpenAI Function Registry

This maintains the same public API as the original OpenAI Function Registry, so migration is straightforward:

1. Replace `openai_functionregistry` imports with `gemini_functionregistry`
2. Update client configuration to use Gemini API key instead of OpenAI/Azure
3. Replace model names with Gemini models (`gemini-1.5-flash`, `gemini-1.5-pro`)

## Original Project

This is based on the [OpenAI Function Registry](https://github.com/tsoernes/openai-functionregistry) by Torstein Sørnes.
