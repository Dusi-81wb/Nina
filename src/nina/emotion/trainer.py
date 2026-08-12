"""Training and evaluation pipeline for Classical Baseline and DistilBERT Emotion Classifiers."""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from nina.api.schemas import SupportedEmotion
from nina.core.exceptions import NinaException
from nina.core.logging import logger

ARTIFACTS_DIR = Path("artifacts")
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_TO_ID: dict[str, int] = {
    "happy": 0,
    "sadness": 1,
    "anger": 2,
    "fear": 3,
    "love": 4,
    "surprise": 5,
}

ID_TO_LABEL: dict[int, str] = {v: k for k, v in LABEL_TO_ID.items()}


def verify_dataset_integrity(
    train_path: Path = Path("data/processed/train.csv"),
    val_path: Path = Path("data/processed/val.csv"),
    test_path: Path = Path("data/processed/test.csv"),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Programmatically verify dataset splits and raise NinaException on integrity or leakage failure."""
    logger.info("Verifying dataset integrity across train, val, and test splits...")

    for p in [train_path, val_path, test_path]:
        if not p.exists():
            raise NinaException(f"Dataset split file missing: {p}")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    req_cols = {"id", "text", "emotion", "source_dataset", "source_label", "source_row_id"}
    for name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        if not req_cols.issubset(df.columns):
            raise NinaException(f"{name} split missing required columns: {req_cols - set(df.columns)}")

        # Check missing values
        if df["text"].isnull().any() or df["emotion"].isnull().any():
            raise NinaException(f"{name} split contains missing text or emotion entries!")

        # Check exact 6 canonical labels
        unique_emotions = set(df["emotion"].unique())
        canonical_set = {e.value for e in SupportedEmotion}
        if not unique_emotions.issubset(canonical_set):
            raise NinaException(f"{name} split contains invalid emotion labels: {unique_emotions - canonical_set}")

    # Check zero data leakage across splits
    train_texts = set(df_train["text"].str.strip().str.lower())
    val_texts = set(df_val["text"].str.strip().str.lower())
    test_texts = set(df_test["text"].str.strip().str.lower())

    val_overlap = train_texts.intersection(val_texts)
    test_overlap = train_texts.intersection(test_texts).union(val_texts.intersection(test_texts))

    if val_overlap:
        raise NinaException(f"DATA LEAKAGE DETECTED: {len(val_overlap)} text overlaps between train and val splits!")

    if test_overlap:
        raise NinaException(f"DATA LEAKAGE DETECTED: {len(test_overlap)} text overlaps between train/val and test splits!")

    logger.info(
        f"Dataset integrity verified successfully! Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)} (Zero Leakage)"
    )
    return df_train, df_val, df_test


def compute_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Compute Accuracy, Macro/Weighted Precision, Recall, F1, Per-class metrics, and Confusion Matrix."""
    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    # Per-class metrics
    p_per_class, r_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(6)), zero_division=0
    )

    per_class_dict = {}
    for idx, label_name in ID_TO_LABEL.items():
        per_class_dict[label_name] = {
            "precision": round(float(p_per_class[idx]), 4),
            "recall": round(float(r_per_class[idx]), 4),
            "f1": round(float(f1_per_class[idx]), 4),
            "support": int(support_per_class[idx]),
        }

    cm = confusion_matrix(y_true, y_pred, labels=list(range(6)))
    cm_dict = {
        ID_TO_LABEL[r]: {ID_TO_LABEL[c]: int(cm[r, c]) for c in range(6)} for r in range(6)
    }

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_precision": round(float(p_weighted), 4),
        "weighted_recall": round(float(r_weighted), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "per_class": per_class_dict,
        "confusion_matrix": cm_dict,
    }


def train_classical_baseline(
    df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame
) -> dict[str, Any]:
    """Train TF-IDF + Logistic Regression baseline model on training set ONLY and evaluate on test set."""
    logger.info("Training Model A: Classical TF-IDF + Logistic Regression Baseline...")

    start_train_time = time.perf_counter()

    # Fit TF-IDF Vectorizer ONLY on training split
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
    X_train = vectorizer.fit_transform(df_train["text"])
    y_train = np.array([LABEL_TO_ID[e] for e in df_train["emotion"]])

    X_val = vectorizer.transform(df_val["text"])
    y_val = np.array([LABEL_TO_ID[e] for e in df_val["emotion"]])

    X_test = vectorizer.transform(df_test["text"])
    y_test = np.array([LABEL_TO_ID[e] for e in df_test["emotion"]])

    # Fit Logistic Regression model
    clf = LogisticRegression(max_iter=1000, C=1.5, class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)

    train_duration = round(time.perf_counter() - start_train_time, 2)

    # Evaluate on Validation set
    y_val_pred = clf.predict(X_val)
    val_metrics = compute_comprehensive_metrics(y_val, y_val_pred)

    # Evaluate on Test set
    start_inference = time.perf_counter()
    y_test_pred = clf.predict(X_test)
    test_latency_ms = round(((time.perf_counter() - start_inference) * 1000.0) / len(df_test), 3)

    test_metrics = compute_comprehensive_metrics(y_test, y_test_pred)

    # Save model artifact
    artifact_path = MODELS_DIR / "classical_baseline.joblib"
    dump({"vectorizer": vectorizer, "classifier": clf, "label_map": LABEL_TO_ID}, artifact_path)
    model_size_mb = round(artifact_path.stat().st_size / (1024 * 1024), 2)

    logger.info(
        f"Classical Baseline Trained! Test Accuracy: {test_metrics['accuracy']*100:.2f}%, Macro F1: {test_metrics['macro_f1']:.4f}, Latency: {test_latency_ms} ms"
    )

    return {
        "model_name": "TF-IDF + Logistic Regression",
        "training_duration_s": train_duration,
        "model_size_mb": model_size_mb,
        "test_latency_ms": test_latency_ms,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "artifact_path": str(artifact_path),
    }


def train_distilbert_model(
    df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame
) -> dict[str, Any]:
    """Fine-tune DistilBERT (distilbert-base-uncased) on 6-class emotion dataset."""
    logger.info("Training Model B: DistilBERT (distilbert-base-uncased) Fine-Tuning...")

    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"DistilBERT compute target device: {device_str}")

    # Optimize sample size for CPU execution to ensure fast completion (~50 steps)
    if device_str == "cpu":
        logger.info("CPU mode detected: Subsampling stratified 3,200 training samples for fast execution...")
        df_train_sub, _ = train_test_split(
            df_train, train_size=3200, stratify=df_train["emotion"], random_state=42
        )
    else:
        df_train_sub = df_train

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Convert DataFrames to Hugging Face Datasets
    train_dataset = Dataset.from_pandas(pd.DataFrame({
        "text": df_train_sub["text"].values,
        "label": [LABEL_TO_ID[e] for e in df_train_sub["emotion"].values],
    }))
    val_dataset = Dataset.from_pandas(pd.DataFrame({
        "text": df_val["text"].values,
        "label": [LABEL_TO_ID[e] for e in df_val["emotion"].values],
    }))
    test_dataset = Dataset.from_pandas(pd.DataFrame({
        "text": df_test["text"].values,
        "label": [LABEL_TO_ID[e] for e in df_test["emotion"].values],
    }))

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=64, padding="max_length")

    train_tok = train_dataset.map(tokenize_fn, batched=True)
    val_tok = val_dataset.map(tokenize_fn, batched=True)
    test_tok = test_dataset.map(tokenize_fn, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=6,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    output_dir = MODELS_DIR / "distilbert_emotion"

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=64 if device_str == "cpu" else 16,
        per_device_eval_batch_size=64,
        num_train_epochs=1 if device_str == "cpu" else 3,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        logging_steps=25,
        seed=42,
    )

    def compute_trainer_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        m = compute_comprehensive_metrics(labels, preds)
        return {
            "eval_accuracy": m["accuracy"],
            "eval_macro_f1": m["macro_f1"],
            "eval_weighted_f1": m["weighted_f1"],
        }

    start_train_time = time.perf_counter()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        compute_metrics=compute_trainer_metrics,
    )

    trainer.train()
    train_duration = round(time.perf_counter() - start_train_time, 2)

    # Save best model checkpoint
    best_model_path = MODELS_DIR / "distilbert_emotion_best"
    trainer.save_model(str(best_model_path))
    tokenizer.save_pretrained(str(best_model_path))

    # Calculate model size
    total_bytes = sum(f.stat().st_size for f in best_model_path.glob("*") if f.is_file())
    model_size_mb = round(total_bytes / (1024 * 1024), 2)

    # Final Evaluation on FULL Held-out Test Set ONLY (8,716 samples)
    start_test_eval = time.perf_counter()
    test_preds_output = trainer.predict(test_tok)
    test_latency_ms = round(((time.perf_counter() - start_test_eval) * 1000.0) / len(df_test), 3)

    y_test_true = np.array([LABEL_TO_ID[e] for e in df_test["emotion"]])
    y_test_pred = np.argmax(test_preds_output.predictions, axis=-1)

    test_metrics = compute_comprehensive_metrics(y_test_true, y_test_pred)

    logger.info(
        f"DistilBERT Fine-Tuned! Test Accuracy: {test_metrics['accuracy']*100:.2f}%, Macro F1: {test_metrics['macro_f1']:.4f}, Latency: {test_latency_ms} ms"
    )

    return {
        "model_name": "DistilBERT (distilbert-base-uncased)",
        "training_duration_s": train_duration,
        "model_size_mb": model_size_mb,
        "test_latency_ms": test_latency_ms,
        "test_metrics": test_metrics,
        "artifact_path": str(best_model_path),
        "test_predictions": y_test_pred.tolist(),
        "test_true": y_test_true.tolist(),
    }


def run_training_pipeline() -> dict[str, Any]:
    """Execute complete dataset integrity verification, baseline training, DistilBERT fine-tuning, and test evaluation."""
    df_train, df_val, df_test = verify_dataset_integrity()

    # Train Model A (Classical Baseline)
    classical_results = train_classical_baseline(df_train, df_val, df_test)

    # Train Model B (DistilBERT Fine-Tuning)
    distilbert_results = train_distilbert_model(df_train, df_val, df_test)

    summary = {
        "classical": classical_results,
        "distilbert": distilbert_results,
    }

    # Save summary artifact
    with open(METRICS_DIR / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    run_training_pipeline()
