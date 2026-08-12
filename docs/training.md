# Nina — Model Training & Evaluation Specification (Phase 4B & 4B.1)

**Document Version:** 2.0.0  
**Status:** Approved  
**Author:** Lead Software Architect & Senior AI/ML Engineer  

---

## 1. Executive Summary

This document specifies the training, validation, and held-out test evaluation of **Model A (TF-IDF + Logistic Regression)** and **Model B (DistilBERT `distilbert-base-uncased` Fine-Tuned on PyTorch CUDA GPU)** on the 87,158-sample candidate emotion dataset.

Both models were evaluated on the **exact same held-out test set (`data/processed/test.csv`, 8,716 samples)** following zero data leakage verification.

---

## 2. Hardware & Environment Specifications

* **PyTorch Version:** `2.13.0+cu130`
* **CUDA Runtime Version:** `13.0`
* **GPU Hardware Accelerator:** `NVIDIA GeForce RTX 4050 Laptop GPU`
* **Dedicated VRAM:** `6.0 GB`
* **Mixed Precision:** Enabled (`fp16` PyTorch AMP)
* **Peak VRAM Allocated:** `1,255.66 MB` (`1.23 GB`)
* **Peak VRAM Reserved:** `1,390.00 MB` (`1.36 GB`)

---

## 3. Measured Metric Comparison Table

| Metric | Model A: Classical Baseline (TF-IDF + LogReg) | Model B: CPU DistilBERT (50 Steps) | Model B: GPU DistilBERT (3 Epochs, CUDA) |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 78.69% | 50.46% | **86.00%** |
| **Macro Precision** | 0.7268 | 0.3684 | **0.8314** |
| **Macro Recall** | 0.7658 | 0.3468 | **0.8119** |
| **Macro F1-Score** | 0.7367 | 0.3045 | **0.8208** |
| **Weighted Precision** | 0.8101 | 0.4508 | **0.8605** |
| **Weighted Recall** | 78.69% | 50.46% | **86.00%** |
| **Weighted F1-Score** | 0.7940 | 0.4317 | **0.8597** |
| **Model Size** | **0.82 MB** | 256.12 MB | 256.12 MB |
| **Training Duration** | **5.47 seconds** | 398.24 seconds | 889.7 seconds (14.8 mins) |
| **CPU Test Latency** | **< 0.01 ms** | 19.84 ms | 28.89 ms |
| **CUDA GPU Test Latency** | N/A | N/A | **0.734 ms** |
| **Artifact Path** | `artifacts/models/classical_baseline.joblib` | N/A | `artifacts/models/distilbert_cuda_best` |

---

## 4. Per-Class Performance (GPU Fine-Tuned DistilBERT)

Evaluated on held-out test split (`data/processed/test.csv`, 8,716 samples):

| Emotion Class | Precision | Recall | F1-Score | Test Support |
| :--- | :--- | :--- | :--- | :--- |
| **`happy`** | 0.8717 | 0.8825 | **0.8771** | 2,579 |
| **`sadness`** | 0.7964 | 0.8596 | **0.8268** | 1,902 |
| **`anger`** | 0.8934 | 0.8363 | **0.8639** | 1,002 |
| **`fear`** | 0.6863 | 0.6000 | **0.6402** | 350 |
| **`love`** | 0.9845 | 0.9659 | **0.9751** | 1,846 |
| **`surprise`** | 0.7563 | 0.7271 | **0.7414** | 1,037 |

---

## 5. Confusion Matrix (GPU Fine-Tuned DistilBERT)

```
               Predicted ->
True Emotion   happy  sadness  anger   fear   love  surprise
happy           2276     140     20     31      8       104
sadness          120    1635     41     30      5        71
anger             32      89    838     16      6        21
fear              38      57     17    210      0        28
love              20      18      5      1   1783        19
surprise         125     114     17     18      9       754
```

---

## 6. Reproducibility Parameters

* **Random Seed:** `42`
* **GPU Accelerator:** `cuda:0` (`NVIDIA GeForce RTX 4050 Laptop GPU`)
* **DistilBERT Configuration:** `distilbert-base-uncased`, `max_length=64`
* **Training Hyperparameters:** `num_train_epochs=3`, `per_device_train_batch_size=8`, `gradient_accumulation_steps=2`, `learning_rate=2e-5`, `weight_decay=0.01`, `fp16=True`
