"""Configuration settings management for Nina using Pydantic Settings."""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from nina.core.exceptions import ConfigurationError


class NinaSettings(BaseSettings):
    """Centralized configuration object for Nina system parameters."""

    model_config = SettingsConfigDict(
        env_prefix="NINA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Core Environment Settings
    env: Literal["development", "staging", "production"] = "development"
    debug: bool = True
    log_level: str = "INFO"
    device: str = "auto"  # Options: auto, cuda, cpu, mps

    # Storage & Model Paths
    model_dir: Path = Path("./models")
    huggingface_cache_dir: Path = Path("./huggingface_cache")

    # Audio Input Settings
    audio_sample_rate: int = 16000
    audio_channels: int = 1
    audio_record_seconds: float = 5.0
    audio_max_duration_seconds: float = 30.0
    audio_min_rms_threshold: float = 0.001
    audio_device_index: int = 0

    # Voice Activity Detection (VAD) Settings
    vad_energy_threshold: float = 0.002
    vad_silence_duration_seconds: float = 1.5
    vad_max_speech_duration_seconds: float = 15.0

    # Speech-to-Text (ASR) Settings
    stt_engine: str = "faster-whisper"  # Options: faster-whisper, whisper, stub
    stt_model_name: str = "faster-whisper"
    stt_model_size: str = "base.en"
    stt_compute_type: str = "float16"

    # Emotion Classification Settings
    emotion_engine: str = "roberta"  # Options: roberta, classical, stub
    emotion_model_name: str = "artifacts/models/distilbert_cuda_best/best_model"
    emotion_max_length: int = 64
    emotion_batch_size: int = 1

    # Optional Intensity Calculator Settings
    enable_intensity: bool = True
    intensity_low_threshold: float = 0.55
    intensity_high_threshold: float = 0.80

    # Decision Thresholds
    confidence_threshold: float = 0.50

    # API Settings
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # Assistant Loop Settings
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    wake_word_enabled: bool = True
    wake_word_threshold: float = 0.5
    tts_enabled: bool = True


def get_settings() -> NinaSettings:
    """Instantiate and validate settings from environment variables."""
    try:
        return NinaSettings()
    except Exception as e:
        raise ConfigurationError(
            message=f"Failed to load system configuration: {e!s}",
            details={"error": str(e)},
        ) from e
