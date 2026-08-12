"""Unit tests for contract interfaces and stub implementations."""

from nina.api.schemas import AudioInput, SupportedEmotion
from nina.emotion.intensity import HeuristicIntensityCalculator
from nina.emotion.stubs import StubEmotionClassifier
from nina.preprocessing.interface import CleanedText
from nina.speech.stubs import StubSpeechToTextEngine


def test_stub_speech_to_text() -> None:
    """Verify StubSpeechToTextEngine produces valid SpeechResult."""
    stub_asr = StubSpeechToTextEngine(stub_text="Testing Nina speech engine")
    assert stub_asr.is_ready() is True

    audio = AudioInput(duration_seconds=3.0, sample_rate=16000)
    result = stub_asr.transcribe(audio)
    assert result.text == "Testing Nina speech engine"
    assert result.confidence == 0.98


def test_stub_emotion_classifier() -> None:
    """Verify StubEmotionClassifier produces valid EmotionPrediction."""
    stub_cls = StubEmotionClassifier(default_emotion=SupportedEmotion.SURPRISE, default_confidence=0.88)
    assert stub_cls.is_ready() is True

    pred = stub_cls.predict("I am amazed")
    assert pred.emotion == SupportedEmotion.SURPRISE
    assert pred.confidence == 0.88
    assert pred.probabilities[SupportedEmotion.SURPRISE] == 0.88


def test_heuristic_intensity_calculator() -> None:
    """Verify HeuristicIntensityCalculator derives low/medium/high levels accurately."""
    calc = HeuristicIntensityCalculator(low_threshold=0.55, high_threshold=0.80)
    stub_cls = StubEmotionClassifier(default_emotion=SupportedEmotion.HAPPY, default_confidence=0.95)
    pred = stub_cls.predict("I am extremely happy")

    text_high = CleanedText(
        raw_text="I am extremely happy",
        cleaned_text="I am extremely happy",
        tokens=["i", "am", "extremely", "happy"],
        intensifier_count=2,
    )
    intensity = calc.calculate_intensity(pred, text_high)
    assert intensity.value == "high"
