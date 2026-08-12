"""Speech-to-text layer package containing interfaces, engine bindings, and stubs."""

from nina.speech.engine import FasterWhisperSpeechToText
from nina.speech.interface import SpeechToTextEngine
from nina.speech.stubs import StubSpeechToTextEngine

__all__ = [
    "FasterWhisperSpeechToText",
    "SpeechToTextEngine",
    "StubSpeechToTextEngine",
]
