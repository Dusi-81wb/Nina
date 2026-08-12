"""Evaluation and error analysis suite for emotion classification models."""

import time
from dataclasses import dataclass
from typing import Any

from nina.api.schemas import SupportedEmotion
from nina.emotion.interface import EmotionClassifier
from nina.emotion.mapping import EmotionLabelMapper


@dataclass
class EvaluationMetrics:
    """Dataclass encapsulating multi-class evaluation results."""

    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    per_class_metrics: dict[str, dict[str, float]]
    confusion_matrix: dict[str, dict[str, int]]
    error_samples: list[dict[str, Any]]
    total_samples: int
    avg_latency_ms: float


class EmotionEvaluator:
    """Evaluates emotion classifiers and generates metrics and confusion matrices."""

    def __init__(self, classifier: EmotionClassifier) -> None:
        self.classifier = classifier

    def evaluate(self, dataset: list[tuple[str, str]]) -> EvaluationMetrics:
        """Run evaluation dataset through classifier and compute metrics.

        Args:
            dataset: List of (text_sample, ground_truth_label) tuples.

        Returns:
            EvaluationMetrics: Comprehensive metrics report.
        """
        y_true: list[SupportedEmotion] = []
        y_pred: list[SupportedEmotion] = []
        latencies: list[float] = []
        error_samples: list[dict[str, Any]] = []

        # Initialize confusion matrix
        emotions = [e.value for e in SupportedEmotion]
        confusion_matrix: dict[str, dict[str, int]] = {
            row: {col: 0 for col in emotions} for row in emotions
        }

        for text, raw_label in dataset:
            true_emo = EmotionLabelMapper.map_label(raw_label)

            start = time.perf_counter()
            pred_res = self.classifier.predict(text)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            pred_emo = pred_res.emotion

            y_true.append(true_emo)
            y_pred.append(pred_emo)
            latencies.append(elapsed_ms)

            confusion_matrix[true_emo.value][pred_emo.value] += 1

            if true_emo != pred_emo:
                error_samples.append({
                    "text": text,
                    "ground_truth": true_emo.value,
                    "predicted": pred_emo.value,
                    "confidence": pred_res.confidence,
                })

        total = len(dataset)
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        accuracy = round(correct / total, 4) if total > 0 else 0.0

        # Calculate per-class precision, recall, F1
        per_class: dict[str, dict[str, float]] = {}
        macro_p_sum = 0.0
        macro_r_sum = 0.0
        macro_f1_sum = 0.0

        for emo_enum in SupportedEmotion:
            emo = emo_enum.value
            tp = confusion_matrix[emo][emo]
            fp = sum(confusion_matrix[other][emo] for other in emotions if other != emo)
            fn = sum(confusion_matrix[emo][other] for other in emotions if other != emo)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            per_class[emo] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": tp + fn,
            }

            macro_p_sum += precision
            macro_r_sum += recall
            macro_f1_sum += f1

        num_classes = len(SupportedEmotion)
        macro_p = round(macro_p_sum / num_classes, 4)
        macro_r = round(macro_r_sum / num_classes, 4)
        macro_f1 = round(macro_f1_sum / num_classes, 4)

        # Weighted F1
        weighted_f1_sum = sum(
            per_class[emo]["f1"] * per_class[emo]["support"] for emo in emotions
        )
        weighted_f1 = round(weighted_f1_sum / total, 4) if total > 0 else 0.0
        avg_latency = round(sum(latencies) / total, 2) if total > 0 else 0.0

        return EvaluationMetrics(
            accuracy=accuracy,
            macro_precision=macro_p,
            macro_recall=macro_r,
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            per_class_metrics=per_class,
            confusion_matrix=confusion_matrix,
            error_samples=error_samples,
            total_samples=total,
            avg_latency_ms=avg_latency,
        )
