"""Emotion classification and intensity calculation module."""

from nina.emotion.classical import ClassicalEmotionClassifier
from nina.emotion.evaluator import EmotionEvaluator, EvaluationMetrics
from nina.emotion.intensity import HeuristicIntensityCalculator
from nina.emotion.interface import EmotionClassifier, IntensityCalculator
from nina.emotion.mapping import EmotionLabelMapper
from nina.emotion.stubs import StubEmotionClassifier
from nina.emotion.transformer import TransformerEmotionClassifier

__all__ = [
    "ClassicalEmotionClassifier",
    "EmotionClassifier",
    "EmotionEvaluator",
    "EmotionLabelMapper",
    "EvaluationMetrics",
    "HeuristicIntensityCalculator",
    "IntensityCalculator",
    "StubEmotionClassifier",
    "TransformerEmotionClassifier",
]
