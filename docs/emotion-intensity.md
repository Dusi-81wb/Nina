# Nina — Emotion Intensity Engine Specification (Phase 5)

**Document Version:** 1.0.0  
**Status:** Complete (Data-Blocked Supervised Training Disclosure)  
**Author:** Lead Software Architect & Senior AI/ML Engineer  

---

## 1. Executive Summary & Conceptual Distinction

This document specifies the design, feature extraction, normalization, and mathematical formulation for **Nina Phase 5: Emotion Intensity Engine**.

The system extends emotion detection from:
$$\text{Text / Audio} \longrightarrow y \in \{\text{happy, sadness, anger, fear, love, surprise}\}$$
to:
$$\text{Audio + Text} \longrightarrow (y, \text{confidence}, \text{intensity})$$

---

### Critical Distinction: Model Confidence vs. Emotion Intensity

1. **Model Confidence ($P(y \mid x)$):**
   * **Definition:** Represented by the classifier output probability (e.g. DistilBERT softmax probability $P(\text{anger}) = 0.91$).
   * **Meaning:** Quantifies model certainty / class membership probability.
2. **Emotion Intensity ($I \in [0.0, 100.0]$):**
   * **Definition:** Derived continuous scalar representing the emotional magnitude, energy, and arousal of the utterance.
   * **Meaning:** A high confidence score ($P=0.91$) on a quiet, matter-of-fact sentence (*"I am mad."*) yields high confidence ($0.91$) but moderate intensity ($\approx 55.0/100$). Conversely, an agitated utterance (*"I AM SO EXTREMELY FURIOUS RIGHT NOW!!!"*) with acoustic volume spike yields high confidence ($0.91$) and maximum intensity ($\approx 92.5/100$).

---

## 2. Phase 5 Architecture Pipeline

```
Audio Signal (.wav / 16kHz PCM)             Text Transcript / CleanedText
                │                                         │
                ▼                                         ▼
   [NinaAudioFeatures Extractor]              [DefaultTextPreprocessor]
   - RMS Energy                                - Tokenization & Caps Ratio
   - Peak-to-RMS Crest Factor                  - Lexical Intensifier Count
   - Zero Crossing Rate (ZCR)                  - Exclamation & Question Count
   - High-Freq Spectral Energy Ratio           - Entropy & Top Margin Δp
                │                                         │
                ▼                                         ▼
   [Acoustic Intensity Subscore S_audio]      [Text Intensity Subscore S_text]
                │                                         │
                └───────────────────┬─────────────────────┘
                                    │
                                    ▼
                      [DefaultIntensityCalculator]
                      - Composite Normalization
                      - Weighting: 0.60 Text + 0.40 Acoustic
                      - Continuous Intensity Score: I ∈ [0.0, 100.0]
                      - Qualitative Mapping: LOW / MEDIUM / HIGH
```

---

## 3. Mathematical Formulation

### 3.1 Text Intensity Signals ($S_{\text{text}}$)

Given emotion probability vector $\mathbf{p} = [p_1, p_2, \dots, p_6]$ sorted descending such that $p_{(1)} \ge p_{(2)} \ge \dots \ge p_{(6)}$:

1. **Top Margin Spread ($\Delta p$):**
   $$\Delta p = p_{(1)} - p_{(2)}$$
2. **Normalized Shannon Entropy ($H_{\text{norm}}$):**
   $$H_{\text{norm}} = \frac{-\sum_{i=1}^6 p_i \log_2(p_i)}{\log_2(6)}$$
3. **Lexical Intensifier Subscore ($S_{\text{intensifiers}}$):**
   $$S_{\text{intensifiers}} = \min\left(1.0, \, 0.33 \times N_{\text{intensifiers}}\right)$$
4. **Punctuation & Emphasis Subscore ($S_{\text{punctuation}}$):**
   $$S_{\text{punctuation}} = \min\left(1.0, \, 0.30 \cdot N_{\text{exclamation}} + 0.10 \cdot N_{\text{question}} + 0.40 \cdot R_{\text{caps}}\right)$$
5. **Composite Text Intensity Subscore ($S_{\text{text}} \in [0.0, 1.0]$):**
   $$S_{\text{text}} = 0.40 \cdot p_{(1)} + 0.30 \cdot \Delta p + 0.15 \cdot S_{\text{intensifiers}} + 0.15 \cdot S_{\text{punctuation}}$$

---

### 3.2 Acoustic & Prosodic Signals ($S_{\text{audio}}$)

Extracted directly from 16kHz PCM audio waveform $x[n]$:

1. **RMS Energy ($S_{\text{rms}}$):**
   $$\text{RMS} = \sqrt{\frac{1}{N} \sum_{n=1}^N x[n]^2}, \quad S_{\text{rms}} = \min\left(1.0, \, \frac{\text{RMS}}{0.15}\right)$$
2. **Zero Crossing Rate ($S_{\text{zcr}}$):**
   $$\text{ZCR} = \frac{1}{N-1} \sum_{n=2}^N \mathbb{I}(\text{sign}(x[n]) \neq \text{sign}(x[n-1])), \quad S_{\text{zcr}} = \min\left(1.0, \, \frac{\text{ZCR}}{0.20}\right)$$
3. **High-Frequency Spectral Energy Ratio ($S_{\text{spectral}}$):**
   $$\text{Ratio} = \frac{\sum_{f \ge 1000\text{Hz}} |X(f)|^2}{\sum_{f} |X(f)|^2 + \varepsilon}, \quad S_{\text{spectral}} = \min\left(1.0, \, \frac{\text{Ratio}}{0.50}\right)$$
4. **Composite Acoustic Subscore ($S_{\text{audio}} \in [0.0, 1.0]$):**
   $$S_{\text{audio}} = 0.40 \cdot S_{\text{rms}} + 0.30 \cdot S_{\text{zcr}} + 0.30 \cdot S_{\text{spectral}}$$

---

### 3.3 Composite Intensity & Level Derivation

$$I_{\text{composite}} = 100 \times \left( 0.60 \cdot S_{\text{text}} + 0.40 \cdot S_{\text{audio}} \right) \quad (\text{audio available})$$
$$I_{\text{composite}} = 100 \times S_{\text{text}} \quad (\text{text-only fallback})$$

* **Qualitative Level Mapping:**
  * $I_{\text{composite}} \ge 70.0 \longrightarrow \text{HIGH}$
  * $45.0 \le I_{\text{composite}} < 70.0 \longrightarrow \text{MEDIUM}$
  * $I_{\text{composite}} < 45.0 \longrightarrow \text{LOW}$

---

## 4. Dataset Requirement & Supervised Training Limitation

### Audit Disclosure (Rule 10 Compliance)

* **Raw Datasets Inspected:** `data/raw/emotion_dataset.csv`, `data/raw/final_dataset.csv`, `data/raw/Emotion_Sentiment_DataSet.csv`.
* **Findings:** All three datasets contain discrete text-level emotion string labels (`happy`, `sadness`, `anger`, `fear`, `love`, `surprise`). **None of the benchmark datasets contain continuous human intensity annotations (e.g. 0.0–1.0 or 1–5 likert scale).**
* **Strict Ethical Directive:** Per Rule 10, synthetic ground-truth intensity labels were **NOT** fabricated.
* **Status:** Supervised regression training is blocked pending availability of intensity-annotated datasets (e.g. IEMOCAP or MSP-Podcast continuous arousal/valence annotations).

---

## 5. Status Statement

**PHASE 5 PIPELINE COMPLETE — SUPERVISED INTENSITY TRAINING BLOCKED BY LACK OF VALID INTENSITY-LABELED DATA.**
