"""Unit tests for audio validator and signal energy calculations."""

import numpy as np
import pytest

from nina.api.schemas import AudioInput
from nina.audio.validator import AudioValidator
from nina.core.exceptions import AudioError


def test_rms_calculation() -> None:
    """Verify calculate_rms computes accurate Root Mean Square energy."""
    validator = AudioValidator()
    empty_signal = np.array([], dtype=np.float32)
    assert validator.calculate_rms(empty_signal) == 0.0

    sine_wave = np.sin(np.linspace(0, 2 * np.pi, 1000)).astype(np.float32)
    rms = validator.calculate_rms(sine_wave)
    assert 0.70 < rms < 0.72  # Sine wave RMS is ~0.707


def test_audio_input_validation_pass() -> None:
    """Verify valid AudioInput payload passes validation."""
    validator = AudioValidator(max_duration_seconds=30.0, min_rms_threshold=0.001)
    audio = AudioInput(
        duration_seconds=5.0,
        sample_rate=16000,
        signal_rms=0.05,
    )
    assert validator.validate_input(audio) is True


def test_audio_input_validation_silence_fail() -> None:
    """Verify near-silent audio raises AudioError."""
    validator = AudioValidator(min_rms_threshold=0.01)
    audio = AudioInput(
        duration_seconds=5.0,
        sample_rate=16000,
        signal_rms=0.0001,
    )
    with pytest.raises(AudioError, match="Input is silent"):
        validator.validate_input(audio)


def test_audio_input_validation_exceed_duration_fail() -> None:
    """Verify audio exceeding max duration threshold raises AudioError."""
    validator = AudioValidator(max_duration_seconds=10.0)
    audio = AudioInput(
        duration_seconds=15.0,
        sample_rate=16000,
        signal_rms=0.05,
    )
    with pytest.raises(AudioError, match="exceeds maximum allowed threshold"):
        validator.validate_input(audio)
