"""Voice Activity Detection (VAD) engine using energy and zero-crossing heuristics."""

import numpy as np

from nina.core.logging import logger


class VoiceActivityDetector:
    """Detects speech start/end boundaries and filters out background silence."""

    def __init__(
        self,
        energy_threshold: float = 0.002,
        silence_duration_seconds: float = 1.5,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
    ) -> None:
        self.energy_threshold = energy_threshold
        self.silence_duration_seconds = silence_duration_seconds
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * (frame_duration_ms / 1000.0))

    def calculate_frame_rms(self, frame: np.ndarray) -> float:
        """Calculate Root Mean Square (RMS) energy for an audio frame."""
        if len(frame) == 0:
            return 0.0
        if np.issubdtype(frame.dtype, np.integer):
            max_val = float(np.iinfo(frame.dtype).max)
            samples = frame.astype(np.float32) / max_val
        else:
            samples = frame.astype(np.float32)

        return float(np.sqrt(np.mean(samples**2)))

    def is_speech_frame(self, frame: np.ndarray) -> bool:
        """Check if a single audio frame contains speech energy above threshold."""
        return self.calculate_frame_rms(frame) >= self.energy_threshold

    def extract_speech_segment(
        self,
        audio_array: np.ndarray,
        sample_rate: int = 16000,
    ) -> tuple[np.ndarray, bool]:
        """Extract continuous speech segment from audio array, trimming leading/trailing silence.

        Args:
            audio_array: 1D numpy array of audio samples.
            sample_rate: Audio sampling frequency in Hz.

        Returns:
            tuple[np.ndarray, bool]: (Trimmed audio array, True if valid speech detected).
        """
        if audio_array is None or len(audio_array) == 0:
            return np.array([], dtype=np.float32), False

        frame_len = int(sample_rate * 0.03)  # 30ms frames
        if len(audio_array) < frame_len:
            has_speech = self.is_speech_frame(audio_array)
            return audio_array, has_speech

        num_frames = len(audio_array) // frame_len
        speech_flags = []

        for i in range(num_frames):
            frame = audio_array[i * frame_len : (i + 1) * frame_len]
            speech_flags.append(self.is_speech_frame(frame))

        if not any(speech_flags):
            return np.array([], dtype=np.float32), False

        # Find first and last speech frame
        first_idx = speech_flags.index(True)
        last_idx = len(speech_flags) - 1 - speech_flags[::-1].index(True)

        # Pad 1 frame (30ms) before and after if possible
        start_sample = max(0, (first_idx - 1) * frame_len)
        end_sample = min(len(audio_array), (last_idx + 2) * frame_len)

        trimmed_audio = audio_array[start_sample:end_sample]
        logger.debug(
            f"VAD extracted {len(trimmed_audio)/sample_rate:.2f}s speech from {len(audio_array)/sample_rate:.2f}s total audio."
        )

        return trimmed_audio, True
