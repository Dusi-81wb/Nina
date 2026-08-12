"""Unit tests for core Pydantic data contract models."""

from nina.api.schemas import (
    AudioInput,
    EmotionPrediction,
    EmotionResult,
    IntensityLevel,
    SpeechResult,
    SupportedEmotion,
)


def test_audio_input_contract() -> None:
    """Verify AudioInput schema validation."""
    audio = AudioInput(
        source="microphone",
        sample_rate=16000,
        duration_seconds=5.0,
        channels=1,
        signal_rms=0.05,
    )
    assert audio.duration_seconds == 5.0
    assert audio.sample_rate == 16000


def test_speech_result_contract() -> None:
    """Verify SpeechResult schema validation."""
    speech = SpeechResult(
        text="Hello world",
        language="en",
        confidence=0.95,
        processing_time_ms=120.5,
    )
    assert speech.text == "Hello world"
    assert speech.confidence == 0.95


def test_emotion_prediction_contract() -> None:
    """Verify EmotionPrediction schema validation."""
    probs = {
        SupportedEmotion.HAPPY: 0.90,
        SupportedEmotion.SADNESS: 0.02,
        SupportedEmotion.ANGER: 0.02,
        SupportedEmotion.FEAR: 0.02,
        SupportedEmotion.LOVE: 0.02,
        SupportedEmotion.SURPRISE: 0.02,
    }
    pred = EmotionPrediction(
        emotion=SupportedEmotion.HAPPY,
        confidence=0.90,
        probabilities=probs,
        processing_time_ms=15.0,
    )
    assert pred.emotion == SupportedEmotion.HAPPY
    assert pred.probabilities[SupportedEmotion.HAPPY] == 0.90


def test_emotion_result_contract() -> None:
    """Verify EmotionResult schema validation."""
    probs = {
        SupportedEmotion.HAPPY: 0.90,
        SupportedEmotion.SADNESS: 0.02,
        SupportedEmotion.ANGER: 0.02,
        SupportedEmotion.FEAR: 0.02,
        SupportedEmotion.LOVE: 0.02,
        SupportedEmotion.SURPRISE: 0.02,
    }
    res = EmotionResult(
        text="I am very happy",
        emotion=SupportedEmotion.HAPPY,
        confidence=0.90,
        intensity=IntensityLevel.HIGH,
        probabilities=probs,
        processing_time_ms=150.0,
    )
    assert res.intensity == IntensityLevel.HIGH
    assert res.emotion == SupportedEmotion.HAPPY
