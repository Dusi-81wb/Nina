"""Unit tests for configuration system management."""

import os

from nina.core.config import NinaSettings, get_settings


def test_default_settings() -> None:
    """Verify default settings load with valid fallback values."""
    settings = get_settings()
    assert settings.env in ["development", "staging", "production"]
    assert settings.audio_sample_rate == 16000
    assert settings.stt_model_name == "faster-whisper"
    assert "best_model" in str(settings.emotion_model_name) or "roberta" in str(settings.emotion_model_name)


def test_env_override(monkeypatch: os.MonkeyPatch) -> None:
    """Verify environment variables override default settings."""
    monkeypatch.setenv("NINA_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("NINA_DEVICE", "cpu")
    settings = NinaSettings()
    assert settings.log_level == "DEBUG"
    assert settings.device == "cpu"
