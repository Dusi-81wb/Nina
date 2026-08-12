"""Development test stub for Speech-to-Text Engine interface.

WARNING: This module is an explicit development stub for pipeline contract
testing without loading heavy neural network model weights.
IT IS NOT FOR PRODUCTION USE.
"""

from nina.api.schemas import AudioInput, SpeechResult
from nina.speech.interface import SpeechToTextEngine


class StubSpeechToTextEngine(SpeechToTextEngine):
    """Development stub implementing SpeechToTextEngine for test verification."""

    def __init__(self, stub_text: str = "I am very happy today") -> None:
        self.stub_text = stub_text

    def transcribe(self, audio: AudioInput, language: str | None = None) -> SpeechResult:
        """Return static transcript result without executing Whisper model weights.

        Args:
            audio: AudioInput object.
            language: Optional target language code.

        Returns:
            SpeechResult: Deterministic stub payload.
        """
        return SpeechResult(
            text=self.stub_text,
            language=language or "en",
            confidence=0.98,
            processing_time_ms=1.5,
        )

    def is_ready(self) -> bool:
        """Stub engine is always ready for test execution."""
        return True
