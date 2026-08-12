"""Explicit label mapping layer for normalizing model taxonomy to Nina canonical emotions."""

from typing import ClassVar

from nina.api.schemas import SupportedEmotion


class EmotionLabelMapper:
    """Normalizes raw model label strings and aligns probability maps to Nina's 6 canonical emotions."""

    CANONICAL_LABEL_MAP: ClassVar[dict[str, SupportedEmotion]] = {
        "joy": SupportedEmotion.HAPPY,
        "happy": SupportedEmotion.HAPPY,
        "happiness": SupportedEmotion.HAPPY,
        "sadness": SupportedEmotion.SADNESS,
        "sad": SupportedEmotion.SADNESS,
        "anger": SupportedEmotion.ANGER,
        "angry": SupportedEmotion.ANGER,
        "annoyance": SupportedEmotion.ANGER,
        "fear": SupportedEmotion.FEAR,
        "fearful": SupportedEmotion.FEAR,
        "scared": SupportedEmotion.FEAR,
        "love": SupportedEmotion.LOVE,
        "loving": SupportedEmotion.LOVE,
        "surprise": SupportedEmotion.SURPRISE,
        "surprised": SupportedEmotion.SURPRISE,
        "astonishment": SupportedEmotion.SURPRISE,
    }

    @classmethod
    def map_label(cls, raw_label: str) -> SupportedEmotion:
        """Map raw dataset/model label string to canonical SupportedEmotion enum.

        Args:
            raw_label: Raw string label from dataset or Hugging Face model output.

        Returns:
            SupportedEmotion: Mapped canonical emotion.
        """
        clean_label = raw_label.strip().lower()
        if clean_label in cls.CANONICAL_LABEL_MAP:
            return cls.CANONICAL_LABEL_MAP[clean_label]

        # Substring fallback heuristics
        if "joy" in clean_label or "happ" in clean_label:
            return SupportedEmotion.HAPPY
        elif "sad" in clean_label or "grief" in clean_label:
            return SupportedEmotion.SADNESS
        elif "ang" in clean_label or "furi" in clean_label:
            return SupportedEmotion.ANGER
        elif "fear" in clean_label or "panic" in clean_label or "anxi" in clean_label:
            return SupportedEmotion.FEAR
        elif "lov" in clean_label or "affec" in clean_label:
            return SupportedEmotion.LOVE
        elif "surp" in clean_label or "wonder" in clean_label:
            return SupportedEmotion.SURPRISE

        # Default fallback
        return SupportedEmotion.HAPPY

    @classmethod
    def normalize_probabilities(
        cls, raw_probs: dict[str, float]
    ) -> dict[SupportedEmotion, float]:
        """Convert a raw string-keyed probability distribution to canonical 6-class normalized probabilities.

        Args:
            raw_probs: Dictionary mapping raw label strings to float probabilities.

        Returns:
            dict[SupportedEmotion, float]: Normalized canonical probability map summing to 1.0.
        """
        canonical_map: dict[SupportedEmotion, float] = {
            SupportedEmotion.HAPPY: 0.0,
            SupportedEmotion.SADNESS: 0.0,
            SupportedEmotion.ANGER: 0.0,
            SupportedEmotion.FEAR: 0.0,
            SupportedEmotion.LOVE: 0.0,
            SupportedEmotion.SURPRISE: 0.0,
        }

        # Aggregate mapped probabilities
        for label, prob in raw_probs.items():
            canonical_emo = cls.map_label(label)
            canonical_map[canonical_emo] += float(prob)

        # Softmax / Sum normalization
        total_sum = sum(canonical_map.values())
        if total_sum > 0:
            for emo, val in canonical_map.items():
                canonical_map[emo] = round(val / total_sum, 4)
        else:
            # Uniform fallback
            uniform_val = round(1.0 / len(canonical_map), 4)
            for emo in canonical_map:
                canonical_map[emo] = uniform_val

        return canonical_map
