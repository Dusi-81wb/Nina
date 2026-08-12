"""Development test stub for Emotion Classifier interface.

WARNING: This module is an explicit development stub for pipeline contract
testing without loading heavy neural network model weights.
IT IS NOT FOR PRODUCTION USE.
"""


from nina.api.schemas import EmotionPrediction, SupportedEmotion
from nina.emotion.interface import EmotionClassifier
from nina.preprocessing.interface import CleanedText


class StubEmotionClassifier(EmotionClassifier):
    """Development stub implementing EmotionClassifier for test verification."""

    def __init__(
        self,
        default_emotion: SupportedEmotion = SupportedEmotion.HAPPY,
        default_confidence: float = 0.92,
    ) -> None:
        self.default_emotion = default_emotion
        self.default_confidence = default_confidence

    def predict(self, text: CleanedText | str) -> EmotionPrediction:
        """Return static emotion prediction without executing PyTorch model weights.

        Args:
            text: Input string or CleanedText.

        Returns:
            EmotionPrediction: Deterministic stub probability distribution payload.
        """
        probabilities: dict[SupportedEmotion, float] = {
            SupportedEmotion.HAPPY: 0.02,
            SupportedEmotion.SADNESS: 0.02,
            SupportedEmotion.ANGER: 0.02,
            SupportedEmotion.FEAR: 0.02,
            SupportedEmotion.LOVE: 0.02,
            SupportedEmotion.SURPRISE: 0.02,
        }
        probabilities[self.default_emotion] = self.default_confidence

        # Renormalize remaining remaining probabilities
        rem_prob = (1.0 - self.default_confidence) / 5.0
        for emo in probabilities:
            if emo != self.default_emotion:
                probabilities[emo] = round(rem_prob, 4)

        return EmotionPrediction(
            emotion=self.default_emotion,
            confidence=self.default_confidence,
            probabilities=probabilities,
            processing_time_ms=2.0,
        )

    def is_ready(self) -> bool:
        """Stub classifier is always ready for test execution."""
        return True
