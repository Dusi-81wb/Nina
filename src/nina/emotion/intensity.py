"""Emotion intensity calculator integrating text signals and acoustic prosodic features."""

import math
import time

from nina.api.schemas import (
    EmotionPrediction,
    IntensityComponents,
    IntensityLevel,
    IntensityPrediction,
    SupportedEmotion,
)
from nina.emotion.interface import IntensityCalculator
from nina.preprocessing.interface import CleanedText


class DefaultIntensityCalculator(IntensityCalculator):
    """Calculates continuous emotional intensity (0.0 to 100.0) combining text signals and acoustic prosody."""

    def __init__(
        self,
        text_weight: float = 0.60,
        audio_weight: float = 0.40,
        low_threshold: float = 45.0,
        high_threshold: float = 70.0,
    ) -> None:
        self.text_weight = text_weight
        self.audio_weight = audio_weight
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    @staticmethod
    def calculate_entropy(probabilities: dict[SupportedEmotion, float]) -> float:
        """Calculate normalized Shannon entropy over the 6 emotion probabilities (0.0 to 1.0)."""
        if not probabilities:
            return 1.0

        entropy_val = 0.0
        for p in probabilities.values():
            if p > 1e-7:
                entropy_val -= p * math.log2(p)

        # Max entropy for 6 uniform classes = log2(6) ≈ 2.58496
        max_entropy = math.log2(6.0)
        return min(1.0, max(0.0, entropy_val / max_entropy))

    def calculate_text_components(
        self,
        prediction: EmotionPrediction,
        text: CleanedText | str,
    ) -> tuple[float, float, float, int, float]:
        """Derive text-level intensity sub-score and component metrics.

        Returns:
            Tuple[text_score, entropy, margin, intensifier_count, punctuation_score]
        """
        confidence = prediction.confidence

        # Sort probabilities descending to find margin Δp
        sorted_probs = sorted(prediction.probabilities.values(), reverse=True)
        top_prob = sorted_probs[0] if len(sorted_probs) > 0 else confidence
        second_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
        delta_p = top_prob - second_prob

        entropy = self.calculate_entropy(prediction.probabilities)

        intensifier_count = 0
        punctuation_score = 0.0

        if isinstance(text, CleanedText):
            intensifier_count = text.intensifier_count
            punct = text.punctuation_features
            exclamation_cnt = punct.get("exclamation", 0)
            question_cnt = punct.get("question", 0)
            caps_ratio = float(punct.get("uppercase_words", 0)) / max(1, len(text.tokens))
            punctuation_score = min(1.0, (exclamation_cnt * 0.3) + (question_cnt * 0.1) + (caps_ratio * 0.4))
        elif isinstance(text, str):
            exclamation_cnt = text.count("!")
            question_cnt = text.count("?")
            words = text.split()
            caps_cnt = sum(1 for w in words if w.isupper() and len(w) > 1)
            caps_ratio = float(caps_cnt) / max(1, len(words))
            punctuation_score = min(1.0, (exclamation_cnt * 0.3) + (question_cnt * 0.1) + (caps_ratio * 0.4))

        intensifier_subscore = min(1.0, intensifier_count * 0.33)

        # Composite text intensity formula
        # S_text = 0.40 * confidence + 0.30 * margin + 0.15 * intensifiers + 0.15 * punctuation
        text_score = min(
            1.0,
            (0.40 * confidence) + (0.30 * delta_p) + (0.15 * intensifier_subscore) + (0.15 * punctuation_score),
        )

        return text_score, entropy, delta_p, intensifier_count, punctuation_score

    def calculate_composite_intensity(
        self,
        prediction: EmotionPrediction,
        text: CleanedText | str,
        audio_features: dict[str, float] | None = None,
    ) -> IntensityPrediction:
        """Calculate continuous numeric emotional intensity (0.0 to 100.0) with component breakdown."""
        start_t = time.perf_counter()

        text_score, entropy, delta_p, intensifier_cnt, punct_score = self.calculate_text_components(prediction, text)

        audio_score = 0.0
        rms = 0.0
        zcr = 0.0
        spectral_ratio = 0.0

        if audio_features and audio_features.get("has_audio", 0.0) > 0.0:
            rms = float(audio_features.get("rms_energy", 0.0))
            zcr = float(audio_features.get("zcr_rate", 0.0))
            spectral_ratio = float(audio_features.get("spectral_ratio", 0.0))

            # Normalize acoustic signals
            rms_norm = min(1.0, rms / 0.15)  # Nominal speech RMS ~0.05-0.15
            zcr_norm = min(1.0, zcr / 0.20)  # Nominal ZCR ~0.05-0.20
            spectral_norm = min(1.0, spectral_ratio / 0.50)  # Nominal high-freq ratio

            audio_score = min(1.0, (0.40 * rms_norm) + (0.30 * zcr_norm) + (0.30 * spectral_norm))

            raw_intensity = (self.text_weight * text_score) + (self.audio_weight * audio_score)
        else:
            raw_intensity = text_score

        intensity_100 = round(float(raw_intensity * 100.0), 2)

        # Derive qualitative level
        if intensity_100 >= self.high_threshold:
            level = IntensityLevel.HIGH
        elif intensity_100 >= self.low_threshold:
            level = IntensityLevel.MEDIUM
        else:
            level = IntensityLevel.LOW

        components = IntensityComponents(
            text_score=round(text_score, 4),
            audio_score=round(audio_score, 4),
            entropy=round(entropy, 4),
            margin=round(delta_p, 4),
            intensifier_count=intensifier_cnt,
            punctuation_score=round(punct_score, 4),
            rms_energy=round(rms, 6),
            zcr_rate=round(zcr, 6),
            spectral_ratio=round(spectral_ratio, 4),
        )

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        return IntensityPrediction(
            intensity=intensity_100,
            level=level,
            confidence=prediction.confidence,  # Separate concept!
            components=components,
            processing_time_ms=round(elapsed_ms, 3),
        )

    def calculate_intensity(
        self,
        prediction: EmotionPrediction,
        text: CleanedText,
    ) -> IntensityLevel:
        """Satisfy IntensityCalculator interface contract returning IntensityLevel."""
        res = self.calculate_composite_intensity(prediction, text)
        return res.level


class HeuristicIntensityCalculator(DefaultIntensityCalculator):
    """Backward compatible wrapper around DefaultIntensityCalculator."""
