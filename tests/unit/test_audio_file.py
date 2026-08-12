"""Unit tests for FileAudioSource WAV reading, format validation, and error handling."""

import pytest

from nina.audio.source import FileAudioSource, create_synthetic_wav_file
from nina.core.exceptions import AudioError


def test_valid_synthetic_wav_file(tmp_path) -> None:
    """Verify FileAudioSource successfully reads and validates a synthetic 16kHz WAV file."""
    wav_path = tmp_path / "test_speech.wav"
    create_synthetic_wav_file(wav_path, duration_seconds=2.0, sample_rate=16000, frequency=440.0)

    source = FileAudioSource(wav_path)
    audio_input = source.get_audio_input()

    assert audio_input.sample_rate == 16000
    assert audio_input.duration_seconds == 2.0
    assert audio_input.signal_rms > 0.001
    assert audio_input.source == "file:test_speech.wav"


def test_missing_audio_file(tmp_path) -> None:
    """Verify missing audio file path raises AudioError."""
    missing_path = tmp_path / "non_existent.wav"
    source = FileAudioSource(missing_path)

    with pytest.raises(AudioError, match="Audio file not found"):
        source.get_audio_input()


def test_empty_zero_byte_audio_file(tmp_path) -> None:
    """Verify zero-byte audio file raises AudioError."""
    empty_path = tmp_path / "empty.wav"
    empty_path.write_bytes(b"")

    source = FileAudioSource(empty_path)
    with pytest.raises(AudioError, match="Audio file is empty"):
        source.get_audio_input()


def test_unsupported_audio_extension(tmp_path) -> None:
    """Verify unsupported file extension raises AudioError."""
    unsupported_path = tmp_path / "file.txt"
    unsupported_path.write_text("not audio data")

    source = FileAudioSource(unsupported_path)
    with pytest.raises(AudioError, match="Unsupported audio format"):
        source.get_audio_input()


def test_corrupted_wav_header(tmp_path) -> None:
    """Verify corrupted WAV header raises AudioError."""
    corrupted_path = tmp_path / "corrupted.wav"
    corrupted_path.write_bytes(b"RIFF1234WAVEfmt bad bytes data")

    source = FileAudioSource(corrupted_path)
    with pytest.raises(AudioError):
        source.get_audio_input()
