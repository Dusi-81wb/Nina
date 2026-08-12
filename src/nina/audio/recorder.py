"""Controlled microphone hardware recorder implementation."""

import wave
from pathlib import Path

from nina.api.schemas import AudioInput
from nina.audio.source import MicrophoneAudioSource
from nina.core.exceptions import AudioError
from nina.core.logging import logger


class MicrophoneRecorder:
    """High-level recorder for sampling hardware microphone and optionally persisting WAV files."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        device_index: int | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.device_index = device_index

    def record(self, duration_seconds: float = 5.0) -> AudioInput:
        """Record live audio for N seconds and return AudioInput data contract.

        Args:
            duration_seconds: Duration in seconds to sample microphone.

        Returns:
            AudioInput: Standardized audio contract payload.
        """
        source = MicrophoneAudioSource(
            duration_seconds=duration_seconds,
            sample_rate=self.sample_rate,
            channels=self.channels,
            device_index=self.device_index,
        )
        return source.get_audio_input()

    def record_to_wav(self, output_path: str | Path, duration_seconds: float = 5.0) -> Path:
        """Record live audio for N seconds and save output to a WAV file.

        Args:
            output_path: Target destination path for .wav file.
            duration_seconds: Duration in seconds to record.

        Returns:
            Path: Resolved path object of written WAV file.
        """
        dest_path = Path(output_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import sounddevice as sd
        except ImportError as e:
            raise AudioError("sounddevice package is required for microphone recording.") from e

        logger.info(f"Recording mic input for {duration_seconds}s to '{dest_path.name}'...")

        try:
            recording = sd.rec(
                int(duration_seconds * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
                device=self.device_index,
            )
            sd.wait()

            with wave.open(str(dest_path), "wb") as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(recording.tobytes())

            logger.info(f"Saved recording to '{dest_path.resolve()}'")
            return dest_path
        except Exception as e:
            raise AudioError(f"Failed to record audio to file '{dest_path.name}': {e!s}") from e
