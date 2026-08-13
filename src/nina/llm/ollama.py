"""Ollama LLM Provider implementation."""
import httpx
from typing import List, Dict, Any, Generator
import json

from nina.llm.provider import LLMProvider
from nina.core.logging import logger

class OllamaProvider(LLMProvider):
    """Provider for local Ollama LLMs."""

    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "llama3"):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name

    def generate_response(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate a response using Ollama API."""
        url = f"{self.base_url}/api/chat"

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False
        }

        if tools:
            payload["tools"] = tools

        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()

                msg = data.get("message", {})
                content = msg.get("content", "")
                tool_calls = msg.get("tool_calls", [])

                return {
                    "content": content,
                    "tool_calls": tool_calls
                }
        except httpx.RequestError as e:
            logger.error(f"Failed to communicate with Ollama: {e}")
            raise
        except Exception as e:
            logger.error(f"Error parsing Ollama response: {e}")
            raise

    def stream_response(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        """Stream a response using Ollama API."""
        url = f"{self.base_url}/api/chat"

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True
        }

        try:
            with httpx.Client(timeout=60.0) as client:
                with client.stream("POST", url, json=payload) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line:
                            data = json.loads(line)
                            if "message" in data and "content" in data["message"]:
                                yield data["message"]["content"]
        except Exception as e:
            logger.error(f"Error streaming from Ollama: {e}")
            raise
