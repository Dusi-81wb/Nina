"""Abstract interfaces for Emotion Classifier and Intensity Calculator."""

from abc import ABC, abstractmethod

from nina.api.schemas import EmotionPrediction, IntensityLevel
from nina.preprocessing.interface import CleanedText


class EmotionClassifier(ABC):
    """Abstract interface defining the contract for ML/DL emotion classification engines."""

    @abstractmethod
    def predict(self, text: CleanedText | str) -> EmotionPrediction:
        """Predict emotion probabilities over the 6 supported emotion classes.

        Args:
            text: CleanedText object or raw input string.

        Returns:
            EmotionPrediction: Prediction payload containing top emotion and probability map.

        Raises:
            EmotionClassificationError: If model inference fails.
        """

    @abstractmethod
    def is_ready(self) -> bool:
        """Check if classifier model weights are loaded and ready.

        Returns:
            bool: True if ready.
        """


class IntensityCalculator(ABC):
    """Abstract interface defining the contract for emotion intensity estimation engines."""

    @abstractmethod
    def calculate_intensity(
        self,
        prediction: EmotionPrediction,
        text: CleanedText,
    ) -> IntensityLevel:
        """Derive emotional intensity (low, medium, high) based on model probabilities
        and linguistic modifiers.

        Args:
            prediction: EmotionPrediction result.
            text: Preprocessed text with extracted intensifiers.

        Returns:
            IntensityLevel: Derived level (LOW, MEDIUM, HIGH).
        """
