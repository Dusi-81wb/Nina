"""Memory subsystem for conversation context tracking."""
from typing import Any

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)

class ConversationMemory:
    """Tracks conversation history between user and assistant."""

    def __init__(self, max_history: int = 50) -> None:
        self.max_history = max_history
        self._messages: list[Message] = []

    def add_message(self, role: str, content: str, metadata: dict[str, Any] = None) -> None:
        """Add a message to the history."""
        if metadata is None:
            metadata = {}
        self._messages.append(Message(role=role, content=content, metadata=metadata))

        # Keep within max history
        if len(self._messages) > self.max_history:
            self._messages = self._messages[-self.max_history:]

    def get_history(self) -> list[dict[str, str]]:
        """Get full history as a list of dictionaries suitable for LLMs."""
        return [{"role": msg.role, "content": msg.content} for msg in self._messages]

    def get_context_window(self, max_tokens: int = 2000, chars_per_token: int = 4) -> list[dict[str, str]]:
        """
        Get a context window that roughly fits within max_tokens.
        Uses a simplistic heuristic of chars_per_token.
        """
        max_chars = max_tokens * chars_per_token
        current_chars = 0
        window_messages = []

        for msg in reversed(self._messages):
            msg_chars = len(msg.content)
            if current_chars + msg_chars <= max_chars:
                window_messages.insert(0, {"role": msg.role, "content": msg.content})
                current_chars += msg_chars
            else:
                break

        return window_messages

    def clear(self) -> None:
        """Clear the conversation history."""
        self._messages.clear()
