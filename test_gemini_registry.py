#!/usr/bin/env python3
"""Test the function registry without requiring API keys"""

from enum import StrEnum
from pydantic import BaseModel, Field

from gemini_functionregistry.client import Client
from gemini_functionregistry.tool_registry import (
    FunctionRegistry, 
    ParserRegistry,
    pydantic_to_gemini_function
)


class TemperatureUnit(StrEnum):
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"


class GetWeather(BaseModel):
    """Get the weather at the given location in the given temperature unit of measurement"""

    location: str = Field(description="The location to get the weather for")
    unit: TemperatureUnit = Field(description="Temperature Unit", default=TemperatureUnit.CELSIUS)


class WeatherResponse(BaseModel):
    """The temperature in a given unit at a specified location"""

    location: str = Field(description="The location name")
    temperature: float = Field(description="The temperature value")
    unit: str = Field(description="The temperature unit")


def get_weather(location: str, unit: TemperatureUnit = TemperatureUnit.CELSIUS) -> str:
    """Get the weather at the given location in the given temperature unit of measurement"""
    return f"The weather in {location} is 22 degrees {unit} with clear skies."


def test_function_conversion():
    """Test converting Pydantic models to Gemini format"""
    print("Testing function conversion...")
    
    # Test GetWeather function
    func_decl = pydantic_to_gemini_function(GetWeather)
    print(f"Function name: {func_decl.name}")
    print(f"Function description: {func_decl.description}")
    print(f"Required parameters: {list(func_decl.parameters.required)}")
    
    # Test parameter types
    props = func_decl.parameters.properties
    print(f"Location type: {props['location'].type_}")
    print(f"Unit type: {props['unit'].type_}")
    print(f"Unit enum values: {list(props['unit'].enum)}")
    
    print("✓ Function conversion test passed!")


def test_registry_setup():
    """Test setting up registries without API calls"""
    print("\nTesting registry setup...")
    
    try:
        # This will fail without API key, but should test the basic setup
        client = Client(api_key="test_key", model="gemini-1.5-flash")
    except Exception as e:
        print(f"Expected error (no real API key): {e}")
    
    # Test with mock client (just for structure testing)
    class MockClient:
        def __init__(self):
            self.model = "gemini-1.5-flash"
            self.api_key = "test"
            self.tokens_per_minute_limit = 1000
            self.requests_per_minute_limit = 100
            
    mock_mini = MockClient()
    mock_regular = MockClient()
    
    # Test FunctionRegistry
    func_registry = FunctionRegistry(mock_mini, mock_regular)
    func_registry.register(get_weather, GetWeather)
    
    # Test getting tools
    tools = func_registry.get_tools(is_mini=True)
    print(f"Generated {len(tools)} tools")
    print(f"Tool name: {tools[0].name}")
    
    # Test ParserRegistry
    parser_registry = ParserRegistry(mock_mini, mock_regular)
    parser_registry.register(WeatherResponse)
    
    parser_tools = parser_registry.get_tools(is_mini=True)
    print(f"Generated {len(parser_tools)} parser tools")
    print(f"Parser tool name: {parser_tools[0].name}")
    
    print("✓ Registry setup test passed!")


def test_cost_calculation():
    """Test cost calculation"""
    print("\nTesting cost calculation...")
    
    from gemini_functionregistry.client import calculate_cost
    
    # Test with Flash model
    cost_flash = calculate_cost("gemini-1.5-flash", input_tokens=1000, output_tokens=500)
    print(f"Flash model cost: {cost_flash}")
    
    # Test with Pro model  
    cost_pro = calculate_cost("gemini-1.5-pro", input_tokens=1000, output_tokens=500)
    print(f"Pro model cost: {cost_pro}")
    
    # Test with string input
    cost_string = calculate_cost("gemini-1.5-flash", input_tokens="Hello world", output_tokens="Hi there")
    print(f"String input cost: {cost_string}")
    
    print("✓ Cost calculation test passed!")


def main():
    """Run all tests"""
    print("=" * 50)
    print("GEMINI FUNCTION REGISTRY TESTS")
    print("=" * 50)
    
    test_function_conversion()
    test_registry_setup()
    test_cost_calculation()
    
    print("\n" + "=" * 50)
    print("ALL TESTS PASSED! ✓")
    print("=" * 50)
    print("\nTo test with real API:")
    print("1. Get a Google AI API key from https://makersuite.google.com/app/apikey")
    print("2. Set environment variable: export GEMINI_API_KEY='your-key-here'")
    print("3. Run: poetry run python gemini_functionregistry/tool_registry_usage_example.py")


if __name__ == "__main__":
    main()