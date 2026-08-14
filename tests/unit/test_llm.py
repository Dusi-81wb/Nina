"""Unit tests for the LLM Provider integration."""
import httpx
import respx

from nina.llm.ollama import OllamaProvider


@respx.mock
def test_ollama_generate_response_success():
    provider = OllamaProvider(base_url="http://localhost:11434", model_name="llama3")

    # Mock Ollama API response
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, json={
            "model": "llama3",
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help?"
            },
            "done": True
        })
    )

    messages = [{"role": "user", "content": "Hi there!"}]
    response = provider.generate_response(messages)

    assert response["content"] == "Hello! How can I help?"
    assert response["tool_calls"] == []

@respx.mock
def test_ollama_generate_response_with_tools():
    provider = OllamaProvider(base_url="http://localhost:11434", model_name="llama3")

    tool_call_payload = {
        "function": {
            "name": "get_weather",
            "arguments": {"location": "San Francisco"}
        }
    }

    # Mock Ollama API response with tool calls
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, json={
            "model": "llama3",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [tool_call_payload]
            },
            "done": True
        })
    )

    messages = [{"role": "user", "content": "What is the weather in SF?"}]
    tools_schema = [{"type": "function", "function": {"name": "get_weather"}}]

    response = provider.generate_response(messages, tools=tools_schema)

    assert response["content"] == ""
    assert len(response["tool_calls"]) == 1
    assert response["tool_calls"][0]["function"]["name"] == "get_weather"
