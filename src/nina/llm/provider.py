"""Base interfaces for LLM providers."""
from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any


class LLMProvider(ABC):
    """Abstract base class for LLM communication."""

    @abstractmethod
    def generate_response(self, messages: list[dict[str, str]], tools: list[dict[str, Any]] = None) -> dict[str, Any]:
        """
        Generate a response from the LLM given a conversation history.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            tools: Optional list of tool schemas available to the model

        Returns:
            Dictionary containing the response 'content' and optional 'tool_calls'
        """

    @abstractmethod
    def stream_response(self, messages: list[dict[str, str]]) -> Generator[str, None, None]:
        """Stream a response back."""
