"""Unit tests for MicrophoneAudioSource and MicrophoneRecorder."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nina.audio.recorder import MicrophoneRecorder
from nina.audio.source import MicrophoneAudioSource
from nina.core.exceptions import AudioError


def test_microphone_source_invalid_duration() -> None:
    """Verify negative or 0 duration raises AudioError."""
    source = MicrophoneAudioSource(duration_seconds=0.0)
    with pytest.raises(AudioError, match="duration must be > 0s"):
        source.get_audio_input()


@patch("sounddevice.query_devices")
@patch("sounddevice.rec")
@patch("sounddevice.wait")
def test_microphone_source_recording_success(
    mock_wait: MagicMock, mock_rec: MagicMock, mock_query: MagicMock
) -> None:
    """Verify MicrophoneAudioSource records and returns valid AudioInput when sounddevice is mocked."""
    mock_query.return_value = [{"name": "Mock Mic", "max_input_channels": 2}]
    # Generate non-silent synthetic sine wave array (RMS > 0.001)
    sine_signal = (0.5 * np.sin(np.linspace(0, 20 * np.pi, 16000))).astype(np.float32)
    mock_rec.return_value = sine_signal.reshape(-1, 1)

    source = MicrophoneAudioSource(duration_seconds=1.0, sample_rate=16000)
    audio = source.get_audio_input()

    assert audio.source == "microphone"
    assert audio.duration_seconds == 1.0
    assert audio.signal_rms > 0.001


def test_microphone_recorder_init() -> None:
    """Verify MicrophoneRecorder initializes with correct settings."""
    recorder = MicrophoneRecorder(sample_rate=16000, channels=1)
    assert recorder.sample_rate == 16000
    assert recorder.channels == 1
