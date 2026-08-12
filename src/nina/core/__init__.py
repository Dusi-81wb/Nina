"""Core system utilities, configuration management, logging, and device diagnostics."""

from nina.core.config import NinaSettings, get_settings
from nina.core.device import DeviceInfo, get_device_info, resolve_device
from nina.core.exceptions import (
    AudioError,
    ConfigurationError,
    EmotionClassificationError,
    ModelLoadError,
    NinaException,
    PreprocessingError,
    SpeechToTextError,
)
from nina.core.logging import log_execution_time, logger, setup_logging

__all__ = [
    "AudioError",
    "ConfigurationError",
    "DeviceInfo",
    "EmotionClassificationError",
    "ModelLoadError",
    "NinaException",
    "NinaSettings",
    "PreprocessingError",
    "SpeechToTextError",
    "get_device_info",
    "get_settings",
    "log_execution_time",
    "logger",
    "resolve_device",
    "setup_logging",
]
