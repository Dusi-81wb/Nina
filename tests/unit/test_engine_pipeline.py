"""Unit tests for NinaEmotionEngine and programmatic API contracts."""


from nina import NinaEmotionEngine, process_text
from nina.api.schemas import EmotionResult, SupportedEmotion
from nina.emotion.stubs import StubEmotionClassifier
from nina.speech.stubs import StubSpeechToTextEngine


def test_nina_emotion_engine_process_text() -> None:
    """Verify process_text returns valid EmotionResult payload."""
    engine = NinaEmotionEngine(
        stt_engine=StubSpeechToTextEngine(),
        emotion_classifier=StubEmotionClassifier(),
    )

    res = engine.process_text("I am really happy today", include_intensity=True)

    assert isinstance(res, EmotionResult)
    assert res.text == "I am really happy today"
    assert res.emotion in SupportedEmotion
    assert 0.0 <= res.confidence <= 1.0
    assert len(res.probabilities) == 6
    assert res.intensity is not None
    assert res.processing_time_ms > 0.0


def test_nina_emotion_engine_optional_intensity() -> None:
    """Verify intensity field remains optional when include_intensity=False."""
    engine = NinaEmotionEngine(
        stt_engine=StubSpeechToTextEngine(),
        emotion_classifier=StubEmotionClassifier(),
    )

    res = engine.process_text("I am feeling fine", include_intensity=False)

    assert isinstance(res, EmotionResult)
    assert res.intensity is None
    assert res.intensity_level is None


def test_top_level_convenience_functions() -> None:
    """Verify process_text top-level convenience helper function."""
    res = process_text("Everything is awesome!", include_intensity=True)
    assert isinstance(res, EmotionResult)
    assert res.emotion in SupportedEmotion
