#!/usr/bin/env python3
"""Test Google Gemini function calling format to understand the API"""

import os
import json
import google.generativeai as genai
from pydantic import BaseModel, Field

# Configure API key (don't actually run this without a real API key)
# genai.configure(api_key="your_api_key_here")

class WeatherParams(BaseModel):
    """Parameters for getting weather information"""
    location: str = Field(description="The location to get weather for")
    unit: str = Field(description="Temperature unit (celsius or fahrenheit)")

def get_weather(location: str, unit: str = "celsius") -> str:
    """Get weather information for a location"""
    return f"The weather in {location} is 20 degrees {unit}"

def explore_gemini_function_format():
    """Explore how to define functions for Gemini"""
    
    # Gemini function declaration format
    function_declaration = {
        "name": "get_weather",
        "description": "Get the weather at a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The location to get weather for"
                },
                "unit": {
                    "type": "string", 
                    "description": "Temperature unit",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["location"]
        }
    }
    
    print("Gemini function declaration format:")
    print(json.dumps(function_declaration, indent=2))
    
    # Compare with OpenAI format
    openai_function = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather at a given location", 
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get weather for"
                    },
                    "unit": {
                        "type": "string",
                        "description": "Temperature unit",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"]
            }
        }
    }
    
    print("\nOpenAI function format:")
    print(json.dumps(openai_function, indent=2))

if __name__ == "__main__":
    explore_gemini_function_format()