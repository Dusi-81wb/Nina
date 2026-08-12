"""Classical ML baseline classifier using trained TF-IDF and Logistic Regression."""

import time
from pathlib import Path
from typing import Any, ClassVar

from joblib import load

from nina.api.schemas import EmotionPrediction, SupportedEmotion
from nina.core.logging import logger
from nina.emotion.interface import EmotionClassifier
from nina.emotion.mapping import EmotionLabelMapper
from nina.preprocessing.interface import CleanedText

MODEL_PATH = Path("artifacts/models/classical_baseline.joblib")


class ClassicalEmotionClassifier(EmotionClassifier):
    """Classical Machine Learning baseline emotion classifier using TF-IDF + Logistic Regression."""

    # Lexicon fallback rules for cold-start baseline
    LEXICON_RULES: ClassVar[dict[SupportedEmotion, set[str]]] = {
        SupportedEmotion.HAPPY: {
            "happy", "joy", "great", "awesome", "fantastic", "glad", "delighted", "cheerful", "smile", "good"
        },
        SupportedEmotion.SADNESS: {
            "sad", "depressed", "unhappy", "cry", "grief", "miserable", "heartbroken", "gloomy", "sorrow", "tear"
        },
        SupportedEmotion.ANGER: {
            "angry", "mad", "furious", "hate", "enraged", "irritated", "annoyed", "outraged", "rage"
        },
        SupportedEmotion.FEAR: {
            "afraid", "scared", "terrified", "fear", "anxious", "panic", "horrified", "dread", "frightened"
        },
        SupportedEmotion.LOVE: {
            "love", "adore", "cherish", "affection", "caring", "beloved", "sweetheart", "fond"
        },
        SupportedEmotion.SURPRISE: {
            "surprised", "shocked", "amazed", "astonished", "unexpected", "woah", "wow", "unbelievable"
        },
    }

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or MODEL_PATH
        self._vectorizer: Any = None
        self._classifier: Any = None
        self._is_loaded = False
        self.load_model()

    def load_model(self) -> None:
        """Load trained TF-IDF vectorizer and Logistic Regression classifier from artifact path if available."""
        if self.model_path.exists():
            try:
                bundle = load(self.model_path)
                self._vectorizer = bundle["vectorizer"]
                self._classifier = bundle["classifier"]
                self._is_loaded = True
                logger.info(f"Loaded trained Classical Baseline model from '{self.model_path}'.")
                return
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to load trained classical baseline: {e!s}. Falling back to lexicon rules.")

        self._is_loaded = True

    def is_ready(self) -> bool:
        """Classical baseline is ready."""
        return self._is_loaded

    def predict(self, text: CleanedText | str) -> EmotionPrediction:
        """Predict emotion probabilities using trained Logistic Regression or lexicon fallback.

        Args:
            text: Input string or CleanedText payload.

        Returns:
            EmotionPrediction: Prediction payload.
        """
        start_time = time.perf_counter()

        raw_str = text.cleaned_text if isinstance(text, CleanedText) else str(text)

        if not raw_str or not raw_str.strip():
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return EmotionPrediction(
                emotion=SupportedEmotion.HAPPY,
                confidence=0.1667,
                probabilities={emo: 0.1667 for emo in SupportedEmotion},
                processing_time_ms=round(elapsed_ms, 3),
            )

        # 1. Trained ML Model Inference
        if self._vectorizer is not None and self._classifier is not None:
            X_vec = self._vectorizer.transform([raw_str])
            probs_arr = self._classifier.predict_proba(X_vec)[0]
            classes = self._classifier.classes_

            id_to_label = {0: "happy", 1: "sadness", 2: "anger", 3: "fear", 4: "love", 5: "surprise"}
            raw_probs = {id_to_label[c]: float(probs_arr[i]) for i, c in enumerate(classes)}
            normalized_probs = EmotionLabelMapper.normalize_probabilities(raw_probs)

            top_emo = max(normalized_probs, key=normalized_probs.get)  # type: ignore
            top_confidence = normalized_probs[top_emo]

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            return EmotionPrediction(
                emotion=top_emo,
                confidence=top_confidence,
                probabilities=normalized_probs,
                processing_time_ms=round(elapsed_ms, 3),
            )

        # 2. Lexicon Rule Fallback
        tokens = [t.lower() for t in raw_str.split()]
        scores: dict[SupportedEmotion, float] = {emo: 0.1 for emo in SupportedEmotion}

        for token in tokens:
            for emo, keywords in self.LEXICON_RULES.items():
                if token in keywords:
                    scores[emo] += 2.5

        total_score = sum(scores.values())
        raw_probs_rule = {emo.value: score / total_score for emo, score in scores.items()}
        normalized_probs = EmotionLabelMapper.normalize_probabilities(raw_probs_rule)

        top_emo = max(normalized_probs, key=normalized_probs.get)  # type: ignore
        top_confidence = normalized_probs[top_emo]

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return EmotionPrediction(
            emotion=top_emo,
            confidence=top_confidence,
            probabilities=normalized_probs,
            processing_time_ms=round(elapsed_ms, 3),
        )

    def get_model_info(self) -> dict[str, Any]:
        """Return diagnostic baseline parameters."""
        return {
            "model_type": "Classical (TF-IDF + Logistic Regression)",
            "model_path": str(self.model_path),
            "is_trained": self._vectorizer is not None,
            "classes": [e.value for e in SupportedEmotion],
            "is_loaded": True,
        }
