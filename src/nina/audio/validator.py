"""Audio input validator for quality, sample rate, and silence detection."""

import numpy as np

from nina.api.schemas import AudioInput
from nina.core.exceptions import AudioError
from nina.core.logging import logger


class AudioValidator:
    """Validates audio properties and calculates signal energy metrics."""

    def __init__(
        self,
        expected_sample_rate: int = 16000,
        max_duration_seconds: float = 30.0,
        min_rms_threshold: float = 0.001,
    ) -> None:
        self.expected_sample_rate = expected_sample_rate
        self.max_duration_seconds = max_duration_seconds
        self.min_rms_threshold = min_rms_threshold

    def validate_input(self, audio: AudioInput) -> bool:
        """Validate AudioInput object metadata.

        Args:
            audio: AudioInput payload.

        Returns:
            bool: True if valid.

        Raises:
            AudioError: If parameters violate bounds.
        """
        if audio.duration_seconds <= 0:
            raise AudioError("Audio duration must be greater than 0 seconds.")

        if audio.duration_seconds > self.max_duration_seconds:
            raise AudioError(
                f"Audio duration ({audio.duration_seconds:.1f}s) exceeds maximum allowed threshold ({self.max_duration_seconds}s)."
            )

        if audio.sample_rate != self.expected_sample_rate:
            logger.warning(
                f"Audio sample rate ({audio.sample_rate} Hz) differs from target ({self.expected_sample_rate} Hz)."
            )

        if audio.signal_rms > 0 and audio.signal_rms < self.min_rms_threshold:
            raise AudioError(
                f"Audio RMS signal energy ({audio.signal_rms:.6f}) is below minimum silence threshold ({self.min_rms_threshold}). Input is silent."
            )

        return True

    @staticmethod
    def calculate_rms(signal: np.ndarray) -> float:
        """Calculate Root Mean Square (RMS) energy level of a raw PCM audio signal array.

        Args:
            signal: 1D numpy array of audio samples.

        Returns:
            float: RMS value.
        """
        if len(signal) == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(signal))))
