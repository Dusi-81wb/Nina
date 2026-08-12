"""Comprehensive PyTorch CUDA GPU verification, tiny overfit test, and 3-epoch DistilBERT fine-tuning pipeline for NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM)."""

import json
from pathlib import Path
import time
from typing import Any, Dict

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
from nina.core.exceptions import NinaException
from nina.core.logging import logger
from nina.emotion.evaluator import EmotionEvaluator
from nina.emotion.mapping import EmotionLabelMapper
from nina.emotion.trainer import (
    ID_TO_LABEL,
    LABEL_TO_ID,
    compute_comprehensive_metrics,
    verify_dataset_integrity,
)

ARTIFACTS_DIR = Path("artifacts")
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
CUDA_MODEL_DIR = MODELS_DIR / "distilbert_cuda_best"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
CUDA_MODEL_DIR.mkdir(parents=True, exist_ok=True)


def verify_cuda_environment() -> Dict[str, Any]:
    """Verify PyTorch CUDA hardware acceleration, device name, VRAM, and tensor operation on GPU."""
    print("==================================================")
    print("      STEP 1: PYTORCH CUDA HARDWARE VERIFICATION  ")
    print("==================================================\n")

    pytorch_ver = torch.__version__
    cuda_ver = torch.version.cuda
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0

    print(f"PyTorch Version:  {pytorch_ver}")
    print(f"CUDA Version:     {cuda_ver}")
    print(f"CUDA Available:   {cuda_available}")
    print(f"Device Count:     {device_count}")

    if not cuda_available:
        raise NinaException("FATAL: PyTorch CUDA acceleration is unavailable! Aborting GPU training.")

    gpu_name = torch.cuda.get_device_name(0)
    total_vram_bytes = torch.cuda.get_device_properties(0).total_memory
    vram_gb = round(total_vram_bytes / (1024**3), 2)

    print(f"GPU Device Name:  {gpu_name}")
    print(f"Total VRAM:       {vram_gb} GB\n")

    if "RTX 4050" not in gpu_name:
        logger.warning(f"Expected NVIDIA GeForce RTX 4050 Laptop GPU, detected: {gpu_name}")

    # CUDA Tensor test
    print("Testing CUDA Tensor operations on cuda:0...")
    t_cuda = torch.ones((10, 10), device="cuda:0") * 3.14159
    sum_val = float(t_cuda.sum().item())
    assert abs(sum_val - 314.159) < 1e-2, "CUDA Tensor computation mismatch!"
    print("CUDA Tensor Test Passed! [ OK ]\n")

    return {
        "pytorch_version": pytorch_ver,
        "cuda_version": cuda_ver,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
    }


def run_cuda_tiny_overfit_test(df_train: pd.DataFrame) -> Dict[str, Any]:
    """Run a controlled 10-epoch overfit sanity experiment on 120 samples strictly on CUDA:0."""
    print("==================================================")
    print("      STEP 2: CUDA TINY DATASET OVERFIT SANITY    ")
    print("==================================================\n")

    # Select 120 samples (20 per class)
    tiny_df, _ = train_test_split(df_train, train_size=120, stratify=df_train["emotion"], random_state=42)

    print(f"Tiny Overfit Dataset Size: 120 samples (20 per class)")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=6,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    # Move model to CUDA and verify parameter device
    model = model.to("cuda:0")
    param_device = next(model.parameters()).device
    print(f"Model Parameter Target Device: {param_device}")
    assert "cuda" in str(param_device), f"Model is not on CUDA device: {param_device}"

    initial_classifier_weight = model.classifier.weight.clone().detach().cpu()

    tiny_ds = Dataset.from_pandas(pd.DataFrame({
        "text": tiny_df["text"].values,
        "label": [LABEL_TO_ID[e] for e in tiny_df["emotion"].values],
    }))

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=64, padding="max_length")

    tiny_tok = tiny_ds.map(tokenize_fn, batched=True)

    training_args = TrainingArguments(
        output_dir="artifacts/sanity_cuda_overfit",
        eval_strategy="no",
        save_strategy="no",
        learning_rate=5e-4,
        per_device_train_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.0,
        fp16=True,
        logging_steps=5,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tiny_tok,
    )

    print("Running 10-epoch CUDA tiny overfit training pass...")
    train_res = trainer.train()

    # Weight update check on CUDA
    final_classifier_weight = model.classifier.weight.clone().detach().cpu()
    weight_diff = torch.sum(torch.abs(final_classifier_weight - initial_classifier_weight)).item()

    print(f"\nTiny CUDA Overfit Final Loss:  {train_res.training_loss:.4f}")
    print(f"Model Weight L1 Delta on CUDA: {weight_diff:.6f}")

    # Evaluate accuracy on CUDA
    preds_raw = trainer.predict(tiny_tok)
    preds = np.argmax(preds_raw.predictions, axis=-1)
    labels = np.array([LABEL_TO_ID[e] for e in tiny_df["emotion"].values])
    acc = (preds == labels).mean()

    print(f"Tiny Dataset Re-prediction Accuracy: {acc * 100:.2f}%")

    if acc >= 0.80 and weight_diff > 0.1:
        print("\nSANITY PASSED: CUDA backpropagation, gradients, and loss optimization working! [ OK ]\n")
    else:
        raise NinaException("CUDA Overfit Sanity Failed! Aborting full training.")

    return {
        "final_loss": train_res.training_loss,
        "weight_delta": weight_diff,
        "accuracy": acc,
    }


def train_full_distilbert_on_cuda(
    df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame
) -> Dict[str, Any]:
    """Fine-tune DistilBERT on CUDA GPU for 3 full epochs across the 69,726 training dataset."""
    print("==================================================")
    print("      STEP 3: FULL DISTILBERT FINE-TUNING ON CUDA ")
    print("==================================================\n")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=6,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    # Move model to CUDA
    model = model.to("cuda:0")
    print(f"Model moved to GPU: {next(model.parameters()).device}")

    train_ds = Dataset.from_pandas(pd.DataFrame({
        "text": df_train["text"].values,
        "label": [LABEL_TO_ID[e] for e in df_train["emotion"].values],
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

    print("Tokenizing train, val, and test splits (max_length=64)...")
    train_tok = train_ds.map(tokenize_fn, batched=True)
    val_tok = val_ds.map(tokenize_fn, batched=True)
    test_tok = test_ds.map(tokenize_fn, batched=True)

    # Training hyperparameter setup for RTX 4050 6GB VRAM
    epochs = 3
    batch_size = 8
    eval_batch_size = 16
    grad_accum_steps = 2
    lr = 2e-5

    training_args = TrainingArguments(
        output_dir=str(CUDA_MODEL_DIR / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        num_train_epochs=epochs,
        weight_decay=0.01,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        logging_steps=100,
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

    # Track GPU Memory prior to training
    torch.cuda.reset_peak_memory_stats(0)
    start_train_time = time.perf_counter()

    print(f"Executing 3-epoch CUDA fine-tuning ({len(df_train)} samples, batch_size={batch_size}, grad_accum={grad_accum_steps}, fp16=True)...")
    train_output = trainer.train()
    train_duration_s = round(time.perf_counter() - start_train_time, 2)

    peak_alloc_mb = round(torch.cuda.max_memory_allocated(0) / (1024 * 1024), 2)
    peak_res_mb = round(torch.cuda.max_memory_reserved(0) / (1024 * 1024), 2)

    print(f"\nCUDA Training Complete in {train_duration_s} seconds!")
    print(f"Peak VRAM Allocated: {peak_alloc_mb} MB ({peak_alloc_mb / 1024:.2f} GB)")
    print(f"Peak VRAM Reserved:  {peak_res_mb} MB ({peak_res_mb / 1024:.2f} GB)")

    # Extract training history from trainer log state
    history = []
    for log in trainer.state.log_history:
        if "eval_macro_f1" in log:
            history.append({
                "epoch": log.get("epoch"),
                "eval_loss": log.get("eval_loss"),
                "eval_accuracy": log.get("eval_accuracy"),
                "eval_macro_f1": log.get("eval_macro_f1"),
                "eval_weighted_f1": log.get("eval_weighted_f1"),
            })

    # Save best model checkpoint
    best_path = CUDA_MODEL_DIR / "best_model"
    trainer.save_model(str(best_path))
    tokenizer.save_pretrained(str(best_path))

    # Calculate model artifact size
    total_bytes = sum(f.stat().st_size for f in best_path.glob("*") if f.is_file())
    model_size_mb = round(total_bytes / (1024 * 1024), 2)

    # Final Evaluation on HELD-OUT TEST SET ONLY (8,716 samples)
    print("\nExecuting final evaluation on held-out test split (8,716 samples) on CUDA...")
    start_test_eval = time.perf_counter()
    test_preds_output = trainer.predict(test_tok)
    test_latency_gpu_ms = round(((time.perf_counter() - start_test_eval) * 1000.0) / len(df_test), 3)

    y_test_true = np.array([LABEL_TO_ID[e] for e in df_test["emotion"]])
    logits_test = test_preds_output.predictions
    
    # Calculate Softmax probabilities
    exp_logits = np.exp(logits_test - np.max(logits_test, axis=-1, keepdims=True))
    probs_test = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    y_test_pred = np.argmax(probs_test, axis=-1)

    # Verify probability distributions sum to ~1.0
    prob_sums = np.sum(probs_test, axis=-1)
    assert np.allclose(prob_sums, 1.0, atol=1e-4), "Softmax probabilities do not sum to 1.0!"

    test_metrics = compute_comprehensive_metrics(y_test_true, y_test_pred)

    print("\n==================================================")
    print("      HELD-OUT TEST SET RESULTS (GPU DISTILBERT)  ")
    print("==================================================")
    print(f"Test Accuracy:       {test_metrics['accuracy'] * 100:.2f}%")
    print(f"Test Macro F1:       {test_metrics['macro_f1']:.4f}")
    print(f"Test Weighted F1:    {test_metrics['weighted_f1']:.4f}")
    print(f"GPU Test Latency:    {test_latency_gpu_ms} ms / sample\n")

    pred_counts = pd.Series([ID_TO_LABEL[p] for p in y_test_pred]).value_counts().to_dict()
    print(f"Predicted Class Distribution on Test Set:\n{pred_counts}\n")

    history_payload = {
        "training_history": history,
        "peak_vram_allocated_mb": peak_alloc_mb,
        "peak_vram_reserved_mb": peak_res_mb,
        "training_duration_s": train_duration_s,
        "test_latency_gpu_ms": test_latency_gpu_ms,
        "model_size_mb": model_size_mb,
        "test_metrics": test_metrics,
        "predicted_distribution": pred_counts,
    }

    with open(METRICS_DIR / "distilbert_training_history.json", "w", encoding="utf-8") as f:
        json.dump(history_payload, f, indent=2)

    return history_payload


def test_checkpoint_reloading(df_test: pd.DataFrame) -> None:
    """Reload fine-tuned CUDA checkpoint from disk onto cuda:0 and verify inference."""
    print("==================================================")
    print("      STEP 4: MODEL CHECKPOINT RELOAD VERIFICATION")
    print("==================================================\n")

    best_path = CUDA_MODEL_DIR / "best_model"
    assert best_path.exists(), "CUDA model checkpoint directory missing!"

    tokenizer = AutoTokenizer.from_pretrained(str(best_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(best_path))

    model = model.to("cuda:0")
    model.eval()

    param_device = next(model.parameters()).device
    print(f"Reloaded Model Device: {param_device}")
    assert "cuda" in str(param_device), "Reloaded model is not on CUDA!"

    # Benchmark CPU vs GPU inference latency
    sample_texts = df_test["text"].head(50).tolist()

    # GPU Inference Benchmark
    start_gpu = time.perf_counter()
    with torch.no_grad():
        inputs = tokenizer(sample_texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
        inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
        logits_gpu = model(**inputs).logits
        preds_gpu = torch.argmax(logits_gpu, dim=-1).cpu().numpy()
    gpu_lat_ms = round(((time.perf_counter() - start_gpu) * 1000.0) / len(sample_texts), 3)

    # CPU Inference Benchmark
    model_cpu = model.to("cpu")
    start_cpu = time.perf_counter()
    with torch.no_grad():
        inputs_cpu = tokenizer(sample_texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
        logits_cpu = model_cpu(**inputs_cpu).logits
        preds_cpu = torch.argmax(logits_cpu, dim=-1).numpy()
    cpu_lat_ms = round(((time.perf_counter() - start_cpu) * 1000.0) / len(sample_texts), 3)

    print(f"Reload Benchmark (50 samples):")
    print(f"  CUDA GPU Inference Latency: {gpu_lat_ms} ms / sample")
    print(f"  CPU Inference Latency:      {cpu_lat_ms} ms / sample")

    assert np.array_equal(preds_gpu, preds_cpu), "GPU vs CPU prediction discrepancy detected!"
    print("STATUS: Checkpoint reload and prediction consistency verified! [ OK ]\n")


def run_complete_cuda_training_pipeline() -> Dict[str, Any]:
    """Execute complete PyTorch CUDA verification, tiny overfit test, full 3-epoch GPU fine-tuning, and reload test."""
    gpu_info = verify_cuda_environment()
    df_train, df_val, df_test = verify_dataset_integrity()
    overfit_info = run_cuda_tiny_overfit_test(df_train)
    full_info = train_full_distilbert_on_cuda(df_train, df_val, df_test)
    test_checkpoint_reloading(df_test)

    return {
        "gpu_info": gpu_info,
        "overfit_info": overfit_info,
        "training_info": full_info,
    }


if __name__ == "__main__":
    run_complete_cuda_training_pipeline()
