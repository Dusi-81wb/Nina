"""Integration test for end-to-end pipeline execution using development stubs."""

from nina.api.schemas import AudioInput, SupportedEmotion
from nina.emotion.intensity import HeuristicIntensityCalculator
from nina.emotion.stubs import StubEmotionClassifier
from nina.inference.pipeline import ModularEmotionPipeline
from nina.preprocessing.processor import DefaultTextPreprocessor
from nina.speech.stubs import StubSpeechToTextEngine


def test_pipeline_integration_with_stubs() -> None:
    """Verify ModularEmotionPipeline integrates all layers into a complete EmotionResult."""
    pipeline = ModularEmotionPipeline(
        stt_engine=StubSpeechToTextEngine(stub_text="I am deeply terrified"),
        classifier=StubEmotionClassifier(default_emotion=SupportedEmotion.FEAR, default_confidence=0.91),
        preprocessor=DefaultTextPreprocessor(),
        intensity_calculator=HeuristicIntensityCalculator(),
    )

    audio = AudioInput(
        source="unit_test",
        sample_rate=16000,
        duration_seconds=4.0,
        signal_rms=0.05,
    )

    result = pipeline.process(audio)

    assert result.text == "I am deeply terrified"
    assert result.emotion == SupportedEmotion.FEAR
    assert result.confidence == 0.91
    assert result.intensity.value in ["medium", "high"]
    assert result.processing_time_ms > 0.0


def test_pipeline_process_text_shortcut() -> None:
    """Verify process_text shortcut executes preprocessing -> classifier -> intensity correctly."""
    pipeline = ModularEmotionPipeline(
        stt_engine=StubSpeechToTextEngine(),
        classifier=StubEmotionClassifier(default_emotion=SupportedEmotion.LOVE, default_confidence=0.94),
        preprocessor=DefaultTextPreprocessor(),
        intensity_calculator=HeuristicIntensityCalculator(),
    )

    result = pipeline.process_text("I love this project so much")

    assert result.text == "I love this project so much"
    assert result.emotion == SupportedEmotion.LOVE
    assert result.confidence == 0.94
    assert result.processing_time_ms >= 0.0
