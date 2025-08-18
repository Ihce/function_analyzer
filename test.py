#!/usr/bin/env python3
# test_function_calling.py - Simple test for function calling

import json
from openai import OpenAI

def test_basic_function_calling():
    """Test basic function calling with a simple weather function"""

    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather in a given city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "The city name"}
                    },
                    "required": ["city"]
                },
            },
        }
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-oss:20b",
            messages=[{"role": "user", "content": "What's the weather in Berlin right now?"}],
            tools=tools,
            tool_choice="auto"
        )

        message = response.choices[0].message
        print("Response:", message)

        if hasattr(message, 'tool_calls') and message.tool_calls:
            print("✅ Function calling works!")
            for tool_call in message.tool_calls:
                print(f"Tool: {tool_call.function.name}")
                print(f"Args: {tool_call.function.arguments}")
        else:
            print("❌ No function calls detected")
            print("Message content:", message.content)

    except Exception as e:
        print(f"❌ Error: {e}")

def test_binary_analysis_tools():
    """Test your specific binary analysis tools"""

    from src.function_analyzer.agent import BinaryAnalysisAgent

    # Create agent
    agent = BinaryAnalysisAgent(verbose=True)

    # Test with a simple message
    messages = [
        {"role": "system", "content": "You are a binary analyst. Use tools to analyze binaries."},
        {"role": "user", "content": "I want to analyze a binary file. What tools do you have available?"}
    ]

    try:
        resp = agent.client.chat.completions.create(
            model=agent.model_id,
            messages=messages,
            tools=agent.tools,
            tool_choice="auto"
        )

        print("✅ Binary analysis tools test successful!")
        print("Response:", resp.choices[0].message.content)

    except Exception as e:
        print(f"❌ Binary analysis tools test failed: {e}")

if __name__ == "__main__":
    print("Testing basic function calling...")
    test_basic_function_calling()
    print("\n" + "="*50 + "\n")
    print("Testing binary analysis tools...")
    test_binary_analysis_tools()
