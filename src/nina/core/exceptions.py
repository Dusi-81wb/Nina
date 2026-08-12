"""Custom exception hierarchy for the Nina system.

All application-specific exceptions inherit from NinaException to support
granular error catching, structured error reporting, and diagnostic logging.
"""

from typing import Any


class NinaException(Exception):
    """Base exception class for all Nina system errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ConfigurationError(NinaException):
    """Raised when environment variables or configuration validation fails."""



class AudioError(NinaException):
    """Raised when audio capture, buffer creation, or signal validation fails."""



class SpeechToTextError(NinaException):
    """Raised when speech recognition transcription fails or model execution errors occur."""



class PreprocessingError(NinaException):
    """Raised when text preprocessing or token normalization fails."""



class EmotionClassificationError(NinaException):
    """Raised when emotion classification inference fails."""



class ModelLoadError(NinaException):
    """Raised when model weights or compute engine backends fail to load."""

