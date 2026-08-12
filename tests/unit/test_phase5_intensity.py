"""Unit and integration tests for Phase 5 Emotion Intensity Engine and acoustic feature extraction."""

import numpy as np

from nina.api.schemas import (
    EmotionPrediction,
    IntensityComponents,
    IntensityLevel,
    IntensityPrediction,
    SupportedEmotion,
)
from nina.audio.features import NinaAudioFeatures
from nina.emotion.intensity import DefaultIntensityCalculator
from nina.preprocessing.processor import DefaultTextPreprocessor


def test_confidence_vs_intensity_separation() -> None:
    """Verify model confidence and emotion intensity remain separate concepts."""
    # High confidence, low intensity text
    pred_high_conf = EmotionPrediction(
        emotion=SupportedEmotion.HAPPY,
        confidence=0.98,
        probabilities={
            SupportedEmotion.HAPPY: 0.98,
            SupportedEmotion.SADNESS: 0.004,
            SupportedEmotion.ANGER: 0.004,
            SupportedEmotion.FEAR: 0.004,
            SupportedEmotion.LOVE: 0.004,
            SupportedEmotion.SURPRISE: 0.004,
        },
    )

    preprocessor = DefaultTextPreprocessor()
    text_plain = preprocessor.preprocess("i am happy")
    text_intense = preprocessor.preprocess("I AM SOOO EXTREMELY DELIGHTED AND FANTASTIC TODAY!!!")

    calc = DefaultIntensityCalculator()

    res_plain = calc.calculate_composite_intensity(pred_high_conf, text_plain)
    res_intense = calc.calculate_composite_intensity(pred_high_conf, text_intense)

    # Confidence must remain identical
    assert res_plain.confidence == pred_high_conf.confidence == 0.98
    assert res_intense.confidence == pred_high_conf.confidence == 0.98

    # Intensity must differ based on linguistic markers
    assert res_intense.intensity > res_plain.intensity
    assert res_intense.components.intensifier_count > res_plain.components.intensifier_count


def test_audio_feature_extraction() -> None:
    """Verify NinaAudioFeatures extracts valid acoustic metrics from numpy PCM arrays."""
    # Generate 1 second of 16kHz sine wave audio
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sine_audio = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    extractor = NinaAudioFeatures(target_sample_rate=sr)
    metrics = extractor.extract_from_array(sine_audio, sample_rate=sr)

    assert metrics["has_audio"] == 1.0
    assert metrics["rms_energy"] > 0.10
    assert metrics["crest_factor"] > 1.0
    assert 0.0 <= metrics["zcr_rate"] <= 1.0
    assert 0.0 <= metrics["spectral_ratio"] <= 1.0


def test_missing_audio_fallback() -> None:
    """Verify composite intensity engine falls back gracefully when audio is absent."""
    pred = EmotionPrediction(
        emotion=SupportedEmotion.ANGER,
        confidence=0.85,
        probabilities={
            SupportedEmotion.ANGER: 0.85,
            SupportedEmotion.SADNESS: 0.05,
            SupportedEmotion.FEAR: 0.05,
            SupportedEmotion.HAPPY: 0.02,
            SupportedEmotion.LOVE: 0.01,
            SupportedEmotion.SURPRISE: 0.02,
        },
    )
    preprocessor = DefaultTextPreprocessor()
    text = preprocessor.preprocess("i am mad")

    calc = DefaultIntensityCalculator()

    # None audio features
    res_no_audio = calc.calculate_composite_intensity(pred, text, audio_features=None)
    assert res_no_audio.intensity > 0.0
    assert res_no_audio.components.audio_score == 0.0

    # Explicit missing audio flag
    empty_audio_features = NinaAudioFeatures.extract_from_array(np.array([], dtype=np.float32))
    res_empty_audio = calc.calculate_composite_intensity(pred, text, audio_features=empty_audio_features)

    assert res_empty_audio.intensity == res_no_audio.intensity


def test_composite_audio_plus_text_intensity() -> None:
    """Verify audio features boost composite intensity score when high acoustic energy is present."""
    pred = EmotionPrediction(
        emotion=SupportedEmotion.ANGER,
        confidence=0.88,
        probabilities={
            SupportedEmotion.ANGER: 0.88,
            SupportedEmotion.FEAR: 0.05,
            SupportedEmotion.SADNESS: 0.03,
            SupportedEmotion.HAPPY: 0.02,
            SupportedEmotion.LOVE: 0.01,
            SupportedEmotion.SURPRISE: 0.01,
        },
    )
    preprocessor = DefaultTextPreprocessor()
    text = preprocessor.preprocess("this is terrible")

    calc = DefaultIntensityCalculator()

    high_energy_audio = {
        "rms_energy": 0.18,
        "crest_factor": 4.5,
        "zcr_rate": 0.15,
        "spectral_ratio": 0.45,
        "has_audio": 1.0,
    }

    res_text_only = calc.calculate_composite_intensity(pred, text)
    res_audio_text = calc.calculate_composite_intensity(pred, text, audio_features=high_energy_audio)

    assert res_audio_text.components.audio_score > 0.50
    assert res_audio_text.intensity > res_text_only.intensity
    assert isinstance(res_audio_text.level, IntensityLevel)


def test_schema_serialization() -> None:
    """Verify IntensityPrediction Pydantic schema serializes and validates cleanly."""
    comp = IntensityComponents(
        text_score=0.75,
        audio_score=0.60,
        entropy=0.35,
        margin=0.70,
        intensifier_count=2,
        punctuation_score=0.50,
        rms_energy=0.12,
        zcr_rate=0.08,
        spectral_ratio=0.30,
    )
    pred = IntensityPrediction(
        intensity=78.5,
        level=IntensityLevel.HIGH,
        confidence=0.92,
        components=comp,
        processing_time_ms=1.25,
    )

    d = pred.model_dump()
    assert d["intensity"] == 78.5
    assert d["level"] == "high"
    assert d["confidence"] == 0.92
    assert d["components"]["intensifier_count"] == 2
