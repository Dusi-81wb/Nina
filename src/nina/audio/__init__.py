"""Audio layer package containing source abstractions, validators, and hardware recorders."""

from nina.audio.interface import AudioCaptureInterface
from nina.audio.recorder import MicrophoneRecorder
from nina.audio.source import (
    AudioSource,
    FileAudioSource,
    MicrophoneAudioSource,
    create_synthetic_wav_file,
)
from nina.audio.validator import AudioValidator

__all__ = [
    "AudioCaptureInterface",
    "AudioSource",
    "AudioValidator",
    "FileAudioSource",
    "MicrophoneAudioSource",
    "MicrophoneRecorder",
    "create_synthetic_wav_file",
]
