"""Abstract interface for Speech-to-Text (ASR) Engines."""

from abc import ABC, abstractmethod

from nina.api.schemas import AudioInput, SpeechResult


class SpeechToTextEngine(ABC):
    """Abstract interface defining the contract for speech recognition engines."""

    @abstractmethod
    def transcribe(self, audio: AudioInput, language: str | None = None) -> SpeechResult:
        """Convert raw audio signal input into structured SpeechResult transcript.

        Args:
            audio: AudioInput data model.
            language: Optional ISO language code.

        Returns:
            SpeechResult: Transcribed text string and metadata payload.

        Raises:
            SpeechToTextError: If transcription inference fails.
        """

    @abstractmethod
    def is_ready(self) -> bool:
        """Check if model weights and compute backend are loaded and ready.

        Returns:
            bool: True if ready for inference.
        """
