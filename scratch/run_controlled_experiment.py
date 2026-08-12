"""Controlled Retraining Experiment for Phase 4B.1 DistilBERT Validation.

Tests whether fine-tuning DistilBERT on 3,840 stratified training samples for 2 full epochs (120 steps)
resolves model collapse and improves Macro F1.
Does NOT overwrite production artifacts/models/classical_baseline.joblib.
"""

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from nina.api.schemas import SupportedEmotion
from nina.emotion.evaluator import EmotionEvaluator
from nina.emotion.trainer import (
    ID_TO_LABEL,
    LABEL_TO_ID,
    compute_comprehensive_metrics,
    verify_dataset_integrity,
)

EXPERIMENTAL_MODEL_DIR = Path("artifacts/models/experimental_distilbert_3epochs")
EXPERIMENTAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)


def run_controlled_experiment():
    print("==================================================")
    print("      PHASE 4B.1 CONTROLLED EXPERIMENT (DISTILBERT) ")
    print("==================================================\n")

    df_train, df_val, df_test = verify_dataset_integrity()

    # Use stratified 3,840 training samples (640 per class)
    df_train_sub, _ = train_test_split(
        df_train, train_size=3840, stratify=df_train["emotion"], random_state=42
    )

    print(f"Training Sample Size:   {len(df_train_sub)} samples (balanced across 6 emotions)")
    print(f"Validation Sample Size: {len(df_val)} samples")
    print(f"Test Sample Size:       {len(df_test)} samples\n")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=6,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    train_ds = Dataset.from_pandas(pd.DataFrame({
        "text": df_train_sub["text"].values,
        "label": [LABEL_TO_ID[e] for e in df_train_sub["emotion"].values],
    }))
    val_ds = Dataset.from_pandas(pd.DataFrame({
        "text": df_val["text"].values,
        "label": [LABEL_TO_ID[e] for e in df_val["emotion"].values],
    }))
    test_ds = Dataset.from_pandas(pd.DataFrame({
        "text": df_test["text"].values,
        "label": [LABEL_TO_ID[e] for e in df_test["emotion"].values],
    }))

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=64, padding="max_length")

    train_tok = train_ds.map(tokenize_fn, batched=True)
    val_tok = val_ds.map(tokenize_fn, batched=True)
    test_tok = test_ds.map(tokenize_fn, batched=True)

    training_args = TrainingArguments(
        output_dir=str(EXPERIMENTAL_MODEL_DIR / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=2,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        logging_steps=20,
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        compute_metrics=compute_trainer_metrics,
    )

    print("Executing 2-epoch fine-tuning pass (120 steps)...")
    start_t = time.perf_counter()
    trainer.train()
    train_duration = round(time.perf_counter() - start_t, 2)

    best_path = EXPERIMENTAL_MODEL_DIR / "best_model"
    trainer.save_model(str(best_path))
    tokenizer.save_pretrained(str(best_path))

    # Evaluate on FULL held-out test split (8,716 samples)
    print("\nEvaluating fine-tuned DistilBERT on held-out test set (8,716 samples)...")
    test_output = trainer.predict(test_tok)
    y_test_true = np.array([LABEL_TO_ID[e] for e in df_test["emotion"]])
    y_test_pred = np.argmax(test_output.predictions, axis=-1)

    test_metrics = compute_comprehensive_metrics(y_test_true, y_test_pred)

    print("\n==================================================")
    print("       CONTROLLED EXPERIMENT TEST RESULTS          ")
    print("==================================================")
    print(f"Experimental Test Accuracy:  {test_metrics['accuracy'] * 100:.2f}%")
    print(f"Experimental Macro F1:       {test_metrics['macro_f1']:.4f}")
    print(f"Experimental Weighted F1:    {test_metrics['weighted_f1']:.4f}")

    print("\nExperimental Per-Class Performance:")
    for emo, m in test_metrics["per_class"].items():
        print(f"  {emo:<10} Precision: {m['precision']:.4f}  Recall: {m['recall']:.4f}  F1: {m['f1']:.4f}")

    # Output prediction distribution
    pred_counts = pd.Series([ID_TO_LABEL[p] for p in y_test_pred]).value_counts().to_dict()
    print(f"\nPredicted Class Distribution on Test Set: {pred_counts}")

    summary = {
        "training_samples": len(df_train_sub),
        "epochs": 2,
        "train_duration_s": train_duration,
        "test_metrics": test_metrics,
        "predicted_distribution": pred_counts,
    }

    with open(EXPERIMENTAL_MODEL_DIR / "experiment_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    run_controlled_experiment()
