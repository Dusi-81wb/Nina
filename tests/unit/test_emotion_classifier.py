"""Unit and integration tests for Emotion Classifier, Label Mapper, and Evaluator."""

import pytest

from nina.api.schemas import SupportedEmotion
from nina.emotion.classical import ClassicalEmotionClassifier
from nina.emotion.evaluator import EmotionEvaluator
from nina.emotion.mapping import EmotionLabelMapper
from nina.preprocessing.processor import DefaultTextPreprocessor


def test_label_mapper_mapping() -> None:
    """Verify EmotionLabelMapper maps raw source labels to canonical SupportedEmotion."""
    assert EmotionLabelMapper.map_label("joy") == SupportedEmotion.HAPPY
    assert EmotionLabelMapper.map_label("happy") == SupportedEmotion.HAPPY
    assert EmotionLabelMapper.map_label("sad") == SupportedEmotion.SADNESS
    assert EmotionLabelMapper.map_label("angry") == SupportedEmotion.ANGER
    assert EmotionLabelMapper.map_label("scared") == SupportedEmotion.FEAR
    assert EmotionLabelMapper.map_label("loving") == SupportedEmotion.LOVE
    assert EmotionLabelMapper.map_label("surprised") == SupportedEmotion.SURPRISE


def test_label_mapper_probability_normalization() -> None:
    """Verify normalize_probabilities aggregates raw labels and returns valid 6-class softmax distribution."""
    raw_probs = {"joy": 0.80, "sadness": 0.10, "anger": 0.10}
    norm_probs = EmotionLabelMapper.normalize_probabilities(raw_probs)

    assert len(norm_probs) == 6
    assert norm_probs[SupportedEmotion.HAPPY] == 0.80
    assert norm_probs[SupportedEmotion.SADNESS] == 0.10
    assert norm_probs[SupportedEmotion.ANGER] == 0.10
    assert sum(norm_probs.values()) == pytest.approx(1.0, abs=0.01)


def test_classical_emotion_classifier_predict() -> None:
    """Verify ClassicalEmotionClassifier predicts canonical emotion with non-zero probabilities."""
    classifier = ClassicalEmotionClassifier()
    assert classifier.is_ready() is True

    preprocessor = DefaultTextPreprocessor()
    preprocessed = preprocessor.preprocess("I am so happy and delighted today!")

    res = classifier.predict(preprocessed)

    assert res.emotion == SupportedEmotion.HAPPY
    assert res.confidence > 0.40
    assert len(res.probabilities) == 6
    assert sum(res.probabilities.values()) == pytest.approx(1.0, abs=0.01)
    assert res.processing_time_ms > 0.0


def test_classical_emotion_classifier_different_emotions() -> None:
    """Verify ClassicalEmotionClassifier distinguishes different emotion prompts."""
    classifier = ClassicalEmotionClassifier()

    res_sad = classifier.predict("I am deeply sad and depressed")
    assert res_sad.emotion == SupportedEmotion.SADNESS

    res_anger = classifier.predict("I am furious and angry")
    assert res_anger.emotion == SupportedEmotion.ANGER

    res_fear = classifier.predict("I am terrified and scared")
    assert res_fear.emotion == SupportedEmotion.FEAR

    res_love = classifier.predict("I love you sweetheart")
    assert res_love.emotion == SupportedEmotion.LOVE

    res_surprise = classifier.predict("Woah I am completely surprised and amazed")
    assert res_surprise.emotion == SupportedEmotion.SURPRISE


def test_emotion_evaluator() -> None:
    """Verify EmotionEvaluator computes accuracy, macro F1, and confusion matrix."""
    classifier = ClassicalEmotionClassifier()
    evaluator = EmotionEvaluator(classifier)

    dataset = [
        ("I am so happy", "happy"),
        ("I am so sad", "sadness"),
        ("I am angry", "anger"),
        ("I am scared", "fear"),
        ("I love you", "love"),
        ("I am surprised", "surprise"),
    ]

    metrics = evaluator.evaluate(dataset)

    assert metrics.total_samples == 6
    assert metrics.accuracy == 1.0
    assert metrics.macro_f1 == 1.0
    assert "happy" in metrics.confusion_matrix
    assert metrics.confusion_matrix["happy"]["happy"] == 1
