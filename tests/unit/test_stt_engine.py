"""Unit tests for FasterWhisperSpeechToText initialization and contract handling."""

from unittest.mock import MagicMock, patch

import pytest

from nina.api.schemas import AudioInput
from nina.core.exceptions import SpeechToTextError
from nina.speech.engine import FasterWhisperSpeechToText


def test_stt_engine_initialization() -> None:
    """Verify FasterWhisperSpeechToText initializes and adjusts compute_type for CPU mode."""
    engine = FasterWhisperSpeechToText(model_size="base.en", device="cpu", compute_type="float16")
    assert engine.model_size == "base.en"
    assert engine.resolved_device == "cpu"
    assert engine.compute_type == "int8"  # Auto-adjusted from float16 to int8 for CPU
    assert engine.is_ready() is False


def test_stt_engine_model_info() -> None:
    """Verify get_model_info returns expected diagnostic parameters."""
    engine = FasterWhisperSpeechToText(model_size="tiny.en", device="cpu")
    info = engine.get_model_info()
    assert info["model_size"] == "tiny.en"
    assert info["resolved_device"] == "cpu"
    assert info["is_loaded"] is False


def test_stt_transcribe_missing_file_metadata() -> None:
    """Verify transcribe raises SpeechToTextError when file_path is missing in AudioInput metadata."""
    engine = FasterWhisperSpeechToText(model_size="base.en", device="cpu")
    audio = AudioInput(duration_seconds=2.0, metadata={})

    with pytest.raises(SpeechToTextError, match="file path is missing or inaccessible"):
        engine.transcribe(audio)


@patch("faster_whisper.WhisperModel")
def test_stt_transcribe_mocked_model(mock_whisper_class: MagicMock, tmp_path) -> None:
    """Verify transcribe calls underlying faster-whisper model and returns SpeechResult."""
    # Create dummy file path
    dummy_wav = tmp_path / "mock_audio.wav"
    dummy_wav.write_bytes(b"mock bytes")

    # Mock WhisperModel behavior
    mock_segment = MagicMock()
    mock_segment.text = "This is a test transcript"
    mock_info = MagicMock()
    mock_info.language = "en"

    mock_model_instance = MagicMock()
    mock_model_instance.transcribe.return_value = ([mock_segment], mock_info)
    mock_whisper_class.return_value = mock_model_instance

    engine = FasterWhisperSpeechToText(model_size="base.en", device="cpu")
    engine._model = mock_model_instance
    engine._is_loaded = True

    audio = AudioInput(
        source="file:mock_audio.wav",
        duration_seconds=3.0,
        metadata={"file_path": str(dummy_wav.resolve())},
    )

    result = engine.transcribe(audio)

    assert result.text == "This is a test transcript"
    assert result.language == "en"
    assert result.processing_time_ms > 0.0
