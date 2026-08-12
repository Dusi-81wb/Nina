"""Nina: Modular Voice-to-Text Emotion Detection Component for Parent Applications."""

from nina.api.schemas import EmotionResult, IntensityLevel, SupportedEmotion
from nina.engine import NinaEmotionEngine

__version__ = "0.1.0"


def process_text(text: str, include_intensity: bool = True) -> EmotionResult:
    """Convenience top-level programmatic function to process text and return EmotionResult."""
    engine = NinaEmotionEngine()
    return engine.process_text(text, include_intensity=include_intensity)


def process_file(file_path: str, include_intensity: bool = True) -> EmotionResult:
    """Convenience top-level programmatic function to process an audio file and return EmotionResult."""
    engine = NinaEmotionEngine()
    return engine.process_file(file_path, include_intensity=include_intensity)


__all__ = [
    "EmotionResult",
    "IntensityLevel",
    "NinaEmotionEngine",
    "SupportedEmotion",
    "process_file",
    "process_text",
]
