# Nina — Model Evaluation Protocol Specification

**Document Version:** 2.0.0  
**Status:** Approved  
**Author:** Lead Software Architect & Senior AI/ML Engineer  

---

## 1. Executive Summary

This document specifies the evaluation protocol for measuring emotion classification accuracy, F1-scores, latency, and memory footprint across Nina's emotion engines.

---

## 2. Evaluation Metrics Protocol

Every model candidate is evaluated using:

1. **Overall Accuracy:** $\frac{\text{Correct Predictions}}{\text{Total Test Samples}}$
2. **Macro F1-Score:** Unweighted average of F1-scores across the 6 canonical classes (`happy`, `sadness`, `anger`, `fear`, `love`, `surprise`).
3. **Weighted F1-Score:** Class-support-weighted average of F1-scores.
4. **Per-Class Metrics:** Precision, Recall, F1-Score, and Support for each emotion.
5. **6x6 Confusion Matrix:** Row = True Emotion, Column = Predicted Emotion.
6. **Inference Latency:** Average milliseconds per utterance (measured on CPU and CUDA GPU).

---

## 3. Measured Evaluation Metrics

### Held-out Test Set (`data/processed/test.csv`, 8,716 samples)

| Metric | Classical Baseline (TF-IDF + LogReg) | Fine-Tuned GPU DistilBERT (`distilbert-base-uncased`) |
| :--- | :--- | :--- |
| **Accuracy** | 78.69% | **86.00%** |
| **Macro Precision** | 0.7268 | **0.8314** |
| **Macro Recall** | 0.7658 | **0.8119** |
| **Macro F1-Score** | 0.7367 | **0.8208** |
| **Weighted F1-Score** | 0.7940 | **0.8597** |
| **GPU Inference Latency** | N/A | **0.734 ms** |
| **CPU Inference Latency** | **< 0.01 ms** | 28.89 ms |
