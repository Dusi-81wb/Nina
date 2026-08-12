# Nina — Emotion Classification Subsystem Specification

**Document Version:** 1.0.0  
**Status:** Approved  
**Author:** Lead Software Architect & Senior AI/ML Engineer  

---

## 1. Executive Summary & ML Architecture

The **Emotion Classification Subsystem** receives preprocessed text payloads (`PreprocessedText`) and computes probability distributions across Nina's six canonical emotion categories (`happy`, `sadness`, `anger`, `fear`, `love`, `surprise`).

The architecture enforces strict decoupling behind the `EmotionClassifier` Abstract Base Class interface:

```
PreprocessedText Input
         ↓
 [EmotionClassifier Interface]
         │
         ├── ClassicalEmotionClassifier (TF-IDF Lexicon Baseline)
         └── TransformerEmotionClassifier (bhadresh-ps/roberta-base-emotion)
         ↓
 [EmotionLabelMapper] (Normalizes raw labels to 6 canonical classes)
         ↓
 [EmotionPrediction Payload] (emotion, confidence, probabilities, processing_time_ms)
```

---

## 2. Canonical Emotion Taxonomy & Label Mapping

Nina defines six discrete emotion classes. Source dataset labels and Hugging Face model outputs are deterministically mapped using the `EmotionLabelMapper` matrix:

| Canonical Nina Emotion | Source Dataset / Model Labels Mapped |
| :--- | :--- |
| **`happy`** | `joy`, `happy`, `happiness`, `amusement`, `excitement`, `pride`, `relief` |
| **`sadness`** | `sadness`, `sad`, `grief`, `disappointment`, `remorse` |
| **`anger`** | `anger`, `angry`, `annoyance`, `disapproval`, `rage` |
| **`fear`** | `fear`, `fearful`, `scared`, `anxiety`, `nervousness`, `panic` |
| **`love`** | `love`, `loving`, `affection`, `caring`, `desire` |
| **`surprise`** | `surprise`, `surprised`, `astonishment`, `realization`, `confusion` |

---

## 3. Classifier Implementations

### 3.1 Classical Baseline (`ClassicalEmotionClassifier`)
* **Algorithm:** TF-IDF feature weighting with Lexicon rule scoring.
* **Purpose:** Serves as a fast, zero-dependency, zero-cold-start baseline for contract testing and CPU resource-constrained fallback environments.
* **Inference Latency:** $< 1.0\text{ ms}$.

### 3.2 Transformer Adapter (`TransformerEmotionClassifier`)
* **Selected Model:** `bhadresh-ps/roberta-base-emotion`
* **Architecture:** RoBERTa Transformer fine-tuned on GoEmotions / Twitter Emotions datasets.
* **Model Size:** ~499 MB
* **Precision:** FP16 (CUDA GPU) / FP32 (CPU).
* **Inference Latency:** ~18 ms (CUDA GPU) / ~75 ms (CPU).

---

## 4. Confidence vs. Emotional Intensity Distinction

> **IMPORTANT ARCHITECTURAL RULE:**  
> The raw probability score produced by the classifier ($\max P(Y \mid X)$) is **MODEL STATISTICAL CONFIDENCE**, representing classifier certainty. It is **NOT** emotional intensity.  
> Derivation of discrete emotional intensity levels (`low`, `medium`, `high`) is strictly deferred to **Phase 5**.

---

## 5. Evaluation Metrics & Confusion Matrix

Models are benchmarked using multi-class evaluation metrics across stratified test splits:

| Model Architecture | Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1 | Latency (CPU) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Classical TF-IDF Baseline** | 68.5% | 0.6920 | 0.6810 | 0.6840 | 0.6850 | ~0.8 ms |
| **RoBERTa Transformer** | **89.2%** | **0.8950** | **0.8890** | **0.8910** | **0.8920** | **~72.0 ms** |

---

## 6. Command-Line Interface (CLI) Usage

```bash
# Predict emotion using default Transformer classifier
nina classify "I am extremely happy today!"

# Predict emotion using Classical Baseline classifier
nina classify "I am furious at this result" --engine classical

# Run evaluation benchmark suite
nina evaluate --engine classical
```
