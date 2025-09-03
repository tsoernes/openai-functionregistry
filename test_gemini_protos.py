#!/usr/bin/env python3
"""Test Google Gemini API format more thoroughly"""

import os
import json
import google.generativeai as genai
from google.generativeai import protos

def test_gemini_format():
    """Test the actual Gemini API format"""
    
    # Example function declaration  
    get_weather_func = protos.FunctionDeclaration(
        name="get_weather",
        description="Get the current weather in a location",
        parameters=protos.Schema(
            type=protos.Type.OBJECT,
            properties={
                "location": protos.Schema(type=protos.Type.STRING, description="Location name"),
                "unit": protos.Schema(
                    type=protos.Type.STRING,
                    description="Temperature unit",
                    enum=["celsius", "fahrenheit"]
                )
            },
            required=["location"]
        )
    )
    
    # Create tool with function declaration
    tool = protos.Tool(function_declarations=[get_weather_func])
    
    print("Gemini Tool structure:")
    print(f"Tool: {tool}")
    print(f"Function: {get_weather_func}")
    
    # Test message format
    messages = [
        protos.Content(
            role="user",
            parts=[protos.Part(text="What's the weather in London?")]
        )
    ]
    
    print("\nGemini Messages structure:")
    for msg in messages:
        print(f"Message: {msg}")

if __name__ == "__main__":
    test_gemini_format()