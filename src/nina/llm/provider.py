"""Base interfaces for LLM providers."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Generator

class LLMProvider(ABC):
    """Abstract base class for LLM communication."""

    @abstractmethod
    def generate_response(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a response from the LLM given a conversation history.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            tools: Optional list of tool schemas available to the model

        Returns:
            Dictionary containing the response 'content' and optional 'tool_calls'
        """
        pass

    @abstractmethod
    def stream_response(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        """Stream a response back."""
        pass
