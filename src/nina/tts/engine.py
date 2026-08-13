"""Text-to-Speech (TTS) integration."""
from abc import ABC, abstractmethod
import pyttsx3

from nina.core.logging import logger

class TTSProvider(ABC):
    """Abstract base class for TTS engines."""

    @abstractmethod
    def speak(self, text: str) -> None:
        """Convert text to speech and play it."""
        pass

class LocalTTSProvider(TTSProvider):
    """Local TTS using pyttsx3."""

    def __init__(self, rate: int = 175, volume: float = 1.0):
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', rate)
            self.engine.setProperty('volume', volume)
            self._available = True
        except Exception as e:
            logger.error(f"Failed to initialize pyttsx3: {e}")
            self._available = False

    def speak(self, text: str) -> None:
        if not self._available:
            logger.warning("TTS is unavailable. Skipping speech output.")
            return

        if not text or not text.strip():
            return

        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            logger.error(f"Error during TTS playback: {e}")
