"""Audio source abstractions for file and microphone audio ingestion."""

import math
import struct
import wave
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from nina.api.schemas import AudioInput
from nina.audio.validator import AudioValidator
from nina.core.exceptions import AudioError
from nina.core.logging import logger


class AudioSource(ABC):
    """Abstract interface defining the contract for audio input sources."""

    @abstractmethod
    def get_audio_input(self) -> AudioInput:
        """Fetch or capture audio and return standardized AudioInput metadata payload.

        Returns:
            AudioInput: Standardized container.

        Raises:
            AudioError: If audio ingestion fails.
        """


class FileAudioSource(AudioSource):
    """Reads and validates audio input from local audio files (e.g. WAV)."""

    def __init__(
        self,
        file_path: str | Path,
        target_sample_rate: int = 16000,
        validator: AudioValidator | None = None,
    ) -> None:
        self.file_path = Path(file_path)
        self.target_sample_rate = target_sample_rate
        self.validator = validator or AudioValidator(expected_sample_rate=target_sample_rate)

    def get_audio_input(self) -> AudioInput:
        """Read local audio file, validate parameters, and construct AudioInput contract.

        Returns:
            AudioInput: Standardized audio payload.

        Raises:
            AudioError: If file is missing, unreadable, corrupted, or violates bounds.
        """
        if not self.file_path.exists():
            raise AudioError(f"Audio file not found: '{self.file_path}'")

        if not self.file_path.is_file():
            raise AudioError(f"Specified path is not a file: '{self.file_path}'")

        if self.file_path.stat().st_size == 0:
            raise AudioError(f"Audio file is empty (0 bytes): '{self.file_path}'")

        suffix = self.file_path.suffix.lower()
        if suffix not in [".wav", ".flac", ".mp3", ".ogg"]:
            raise AudioError(f"Unsupported audio format '{suffix}'. Supported formats: .wav, .flac, .mp3, .ogg")

        try:
            return self._read_wav_file()
        except AudioError:
            raise
        except Exception as e:
            raise AudioError(f"Failed to read audio file '{self.file_path.name}': {e!s}") from e

    def _read_wav_file(self) -> AudioInput:
        """Internal helper to parse standard WAV file headers and sample data."""
        try:
            with wave.open(str(self.file_path), "rb") as wf:
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                framerate = wf.getframerate()
                n_frames = wf.getnframes()

                if n_frames == 0:
                    raise AudioError("WAV file contains 0 audio frames.")

                duration = n_frames / float(framerate)
                raw_bytes = wf.readframes(n_frames)

            # Convert raw bytes to numpy array based on sample width
            if sample_width == 2:  # 16-bit PCM
                signal = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            elif sample_width == 4:  # 32-bit float or PCM
                signal = np.frombuffer(raw_bytes, dtype=np.float32)
            elif sample_width == 1:  # 8-bit unsigned
                signal = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            else:
                signal = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            # Convert multi-channel to mono if necessary
            if channels > 1:
                signal = signal.reshape(-1, channels).mean(axis=1)

            rms = self.validator.calculate_rms(signal)

            audio_input = AudioInput(
                source=f"file:{self.file_path.name}",
                sample_rate=framerate,
                duration_seconds=round(duration, 2),
                channels=channels,
                signal_rms=round(rms, 6),
                metadata={
                    "file_path": str(self.file_path.resolve()),
                    "sample_width_bytes": sample_width,
                    "frame_count": n_frames,
                },
            )

            # Validate against thresholds
            self.validator.validate_input(audio_input)
            return audio_input

        except wave.Error as e:
            raise AudioError(f"Corrupted or invalid WAV header in '{self.file_path.name}': {e!s}") from e


class MicrophoneAudioSource(AudioSource):
    """Captures live audio from physical microphone hardware."""

    def __init__(
        self,
        duration_seconds: float = 5.0,
        sample_rate: int = 16000,
        channels: int = 1,
        device_index: int | None = None,
        validator: AudioValidator | None = None,
    ) -> None:
        self.duration_seconds = duration_seconds
        self.sample_rate = sample_rate
        self.channels = channels
        self.device_index = device_index
        self.validator = validator or AudioValidator(expected_sample_rate=sample_rate)

    def get_audio_input(self) -> AudioInput:
        """Record microphone input for configured duration and construct AudioInput contract.

        Returns:
            AudioInput: Standardized audio payload.

        Raises:
            AudioError: If microphone device is unavailable or recording fails.
        """
        if self.duration_seconds <= 0:
            raise AudioError(f"Recording duration must be > 0s, got {self.duration_seconds}s")

        try:
            import sounddevice as sd
        except ImportError as e:
            raise AudioError("sounddevice package is required for microphone recording.") from e

        logger.info(f"Recording microphone input for {self.duration_seconds}s at {self.sample_rate}Hz...")

        try:
            # Check device availability
            devices = sd.query_devices()
            if len(devices) == 0:
                raise AudioError("No audio input devices found on system.")

            # Record numpy array
            recording = sd.rec(
                int(self.duration_seconds * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                device=self.device_index,
            )
            sd.wait()  # Block until recording finishes

            signal = recording.flatten()
            rms = self.validator.calculate_rms(signal)

            audio_input = AudioInput(
                source="microphone",
                sample_rate=self.sample_rate,
                duration_seconds=round(self.duration_seconds, 2),
                channels=self.channels,
                signal_rms=round(rms, 6),
                metadata={
                    "device_index": self.device_index,
                    "sample_count": len(signal),
                },
            )

            self.validator.validate_input(audio_input)
            return audio_input

        except AudioError:
            raise
        except Exception as e:
            raise AudioError(f"Microphone recording failed: {e!s}") from e


def create_synthetic_wav_file(
    file_path: str | Path,
    duration_seconds: float = 2.0,
    sample_rate: int = 16000,
    frequency: float = 440.0,
) -> Path:
    """Helper utility to generate a synthetic WAV audio file fixture for automated tests.

    Args:
        file_path: Destination path for .wav file.
        duration_seconds: Audio duration.
        sample_rate: Sampling frequency Hz.
        frequency: Audio tone frequency in Hz.

    Returns:
        Path: Path object to generated fixture file.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_samples = int(sample_rate * duration_seconds)
    t = [i / sample_rate for i in range(n_samples)]
    # Generate 440Hz sine wave with non-zero amplitude (RMS > 0.001)
    amplitude = 16000.0
    waveform = [int(amplitude * math.sin(2 * math.pi * frequency * time_pt)) for time_pt in t]

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        packed_data = struct.pack(f"<{len(waveform)}h", *waveform)
        wf.writeframes(packed_data)

    return path
