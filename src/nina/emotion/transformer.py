"""Concrete Transformer-based emotion classifier adapter using Hugging Face Transformers."""

import time
from pathlib import Path
from typing import Any

from nina.api.schemas import EmotionPrediction, SupportedEmotion
from nina.core.config import NinaSettings, get_settings
from nina.core.device import get_actual_model_device, resolve_device
from nina.core.exceptions import EmotionClassificationError
from nina.core.logging import logger
from nina.emotion.interface import EmotionClassifier
from nina.emotion.mapping import EmotionLabelMapper
from nina.preprocessing.interface import CleanedText

LOCAL_FINE_TUNED_PATH = Path("artifacts/models/distilbert_cuda_best/best_model")


class TransformerEmotionClassifier(EmotionClassifier):
    """Transformer-based emotion classifier adapter supporting fine-tuned local checkpoints or Hugging Face models."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        cache_dir: str | Path | None = None,
        settings: NinaSettings | None = None,
    ) -> None:
        self.settings = settings or get_settings()

        # Prefer local fine-tuned checkpoint if available
        if model_name is None and LOCAL_FINE_TUNED_PATH.exists():
            self.model_name = str(LOCAL_FINE_TUNED_PATH.resolve())
        else:
            self.model_name = model_name or self.settings.emotion_model_name

        self.requested_device = device or self.settings.device
        self.cache_dir = Path(cache_dir or self.settings.huggingface_cache_dir)

        # Resolve target compute device using single authoritative resolver
        self.resolved_device = resolve_device(self.requested_device)

        self._pipeline: Any = None
        self._is_loaded = False
        self._actual_device_str: str = "unloaded"

    @property
    def actual_device(self) -> str:
        """Return actual PyTorch model parameter device string (e.g. 'cuda:0' or 'cpu')."""
        if self._is_loaded and self._pipeline is not None:
            return get_actual_model_device(self._pipeline)
        return self.resolved_device

    def load_model(self) -> None:
        """Explicitly load Hugging Face transformer model pipeline into memory.

        Raises:
            EmotionClassificationError: If model weights fail to load.
        """
        if self._is_loaded and self._pipeline is not None:
            return

        logger.info(
            f"Loading Transformer emotion classifier '{self.model_name}' on device '{self.resolved_device}'..."
        )

        try:
            from transformers import pipeline

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            device_id = 0 if self.resolved_device == "cuda" else -1

            self._pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                top_k=None,
                device=device_id,
            )
            self._is_loaded = True

            # Inspect actual PyTorch model parameter device
            self._actual_device_str = get_actual_model_device(self._pipeline)
            logger.info(
                f"Transformer model '{self.model_name}' loaded successfully. "
                f"Resolved device: '{self.resolved_device}', Actual model device: '{self._actual_device_str}'."
            )

        except Exception as e:
            self._is_loaded = False
            raise EmotionClassificationError(
                f"Failed to load Transformer emotion model '{self.model_name}': {e!s}",
                details={"model_name": self.model_name, "device": self.resolved_device},
            ) from e

    def is_ready(self) -> bool:
        """Check if model engine is loaded and ready for inference."""
        return self._is_loaded and self._pipeline is not None

    def predict(self, text: CleanedText | str) -> EmotionPrediction:
        """Predict probability distribution over the 6 canonical emotion classes.

        Args:
            text: CleanedText object or raw input string.

        Returns:
            EmotionPrediction: Prediction payload.

        Raises:
            EmotionClassificationError: If model inference fails.
        """
        if not self.is_ready():
            self.load_model()

        start_time = time.perf_counter()

        raw_str = text.cleaned_text if isinstance(text, CleanedText) else str(text)

        if not raw_str.strip():
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return EmotionPrediction(
                emotion=SupportedEmotion.HAPPY,
                confidence=0.1667,
                probabilities={emo: 0.1667 for emo in SupportedEmotion},
                processing_time_ms=round(elapsed_ms, 3),
            )

        try:
            # Model forward pass
            results = self._pipeline(raw_str)

            # Extract raw label-probability dictionary
            raw_scores: dict[str, float] = {}
            if results and isinstance(results, list):
                score_list = results[0] if isinstance(results[0], list) else results
                for item in score_list:
                    raw_scores[item["label"]] = float(item["score"])

            # Map raw labels to canonical 6-class normalized probabilities
            normalized_probs = EmotionLabelMapper.normalize_probabilities(raw_scores)

            top_emo = max(normalized_probs, key=normalized_probs.get)  # type: ignore
            top_confidence = normalized_probs[top_emo]

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            return EmotionPrediction(
                emotion=top_emo,
                confidence=top_confidence,
                probabilities=normalized_probs,
                processing_time_ms=round(elapsed_ms, 2),
            )

        except EmotionClassificationError:
            raise
        except Exception as e:
            raise EmotionClassificationError(
                f"Emotion classification inference failed: {e!s}",
                details={"input_text": raw_str},
            ) from e

    def get_model_info(self) -> dict[str, Any]:
        """Return diagnostic dictionary of model engine parameters."""
        gpu_alloc = 0.0
        gpu_res = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                gpu_alloc = round(torch.cuda.memory_allocated() / (1024**2), 2)
                gpu_res = round(torch.cuda.memory_reserved() / (1024**2), 2)
        except Exception:  # noqa: BLE001, S110
            pass

        return {
            "model_name": self.model_name,
            "requested_device": self.requested_device,
            "resolved_device": self.resolved_device,
            "actual_model_device": self.actual_device,
            "gpu_memory_allocated_mb": gpu_alloc,
            "gpu_memory_reserved_mb": gpu_res,
            "cache_dir": str(self.cache_dir),
            "is_loaded": self.is_ready(),
        }
