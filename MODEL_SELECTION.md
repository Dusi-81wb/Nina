# Nina — Model Selection Specification

**Document Version:** 2.0.0  
**Status:** Approved  
**Author:** Lead Software Architect & Senior AI/ML Engineer  

---

## 1. Executive Summary

This document specifies the model selection rationale for **Nina Feature 1: Voice-Based Emotion Detection**.

Following empirical model training and evaluation on the 87,158-sample candidate dataset (`data/processed/train.csv`, `val.csv`, `test.csv`), both **Model A (TF-IDF + Logistic Regression)** and **Model B (GPU Fine-Tuned DistilBERT)** have been benchmarked on held-out test data.

---

## 2. Benchmark Comparison Table

| Feature / Metric | Model A: Classical Baseline (TF-IDF + LogReg) | Model B: Fine-Tuned GPU DistilBERT (`distilbert-base-uncased`) |
| :--- | :--- | :--- |
| **Test Accuracy** | 78.69% | **86.00%** |
| **Macro F1-Score** | 0.7367 | **0.8208** |
| **Weighted F1-Score** | 0.7940 | **0.8597** |
| **Model Size** | **0.82 MB** | 256.12 MB |
| **Training Duration** | **5.47 seconds** | 889.7 seconds (14.8 minutes) |
| **CPU Test Latency** | **< 0.01 ms / utterance** | 28.89 ms / utterance |
| **CUDA GPU Test Latency** | N/A | **0.734 ms / sample** |
| **Hardware Required** | CPU | NVIDIA CUDA GPU (Recommended) / CPU |

---

## 3. Environment-Based Selection Strategy

To balance accuracy and deployment flexibility:

1. **Primary Production Model for GPU / Server Environments:**  
   **Fine-Tuned GPU DistilBERT (`artifacts/models/distilbert_cuda_best/best_model`)**  
   * **Why:** Achieves superior **86.00% Test Accuracy** (+7.31% over baseline) and **0.8208 Macro F1** (+0.0841 over baseline) with sub-millisecond GPU inference latency (**0.734 ms**).

2. **Primary Production Model for CPU / Edge Environments:**  
   **Classical Baseline (`artifacts/models/classical_baseline.joblib`)**  
   * **Why:** Provides solid **78.69% Test Accuracy** and **0.7367 Macro F1** with **<0.01 ms** CPU inference latency and a minimal **0.82 MB** RAM footprint.
