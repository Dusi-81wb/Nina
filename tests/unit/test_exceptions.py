"""Unit tests for exception hierarchy and formatting."""

import pytest

from nina.core.exceptions import (
    AudioError,
    ConfigurationError,
    EmotionClassificationError,
    ModelLoadError,
    NinaException,
    PreprocessingError,
    SpeechToTextError,
)


def test_exception_formatting() -> None:
    """Verify NinaException includes message and details dictionary."""
    exc = NinaException("Test error message", details={"code": 404})
    assert exc.message == "Test error message"
    assert exc.details == {"code": 404}
    assert "Test error message" in str(exc)
    assert "Details:" in str(exc)


def test_exception_inheritance() -> None:
    """Verify specific exceptions inherit from NinaException."""
    assert issubclass(ConfigurationError, NinaException)
    assert issubclass(AudioError, NinaException)
    assert issubclass(SpeechToTextError, NinaException)
    assert issubclass(PreprocessingError, NinaException)
    assert issubclass(EmotionClassificationError, NinaException)
    assert issubclass(ModelLoadError, NinaException)


def test_exception_raising() -> None:
    """Verify raising a specific exception can be caught by base NinaException."""
    with pytest.raises(NinaException):
        raise AudioError("Microphone input disconnected")
