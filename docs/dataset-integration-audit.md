# Nina — Dataset Integration & Audit Specification

**Document Version:** 1.0.0  
**Status:** Approved  
**Author:** Lead Software Architect & Senior AI/ML Engineer  

---

## 1. Executive Summary

This document presents the programmatic audit, label mapping, duplicate analysis, conflict resolution, and data leakage verification for three newly acquired raw emotion datasets:

1. `data/raw/emotion_dataset.csv`
2. `data/raw/final_dataset.csv`
3. `data/raw/Emotion_Sentiment_DataSet.csv`

The objective is to establish a high-quality, deduplicated, 6-class canonical emotion dataset for Nina (`data/processed/candidate_all.csv`, `train.csv`, `val.csv`, `test.csv`) without modifying existing Phase 4 models or breaking the existing codebase.

---

## 2. Raw Dataset Inspection & Statistics

| Dataset File Name | Total Rows | Column Names | Malformed Rows | Exact Duplicate Rows | Unique Texts | Avg Word Length |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `emotion_dataset.csv` | 34,792 | `['', 'Emotion', 'Text', 'Clean_Text']` | 0 | 0 | 31,094 | 16.2 words |
| `final_dataset.csv` | 106,355 | `['text', 'emotion']` | 0 | 25,529 | 80,786 | 20.2 words |
| `Emotion_Sentiment_DataSet.csv` | 160,000 | `['', 'Text', 'Emotion']` | 8 | 0 | 87,982 | 32.1 words |
| **TOTAL** | **301,147** | — | **8** | **25,529** | **199,862** | **~22.8 words** |

---

## 3. Label Mapping Strategy & Audit

Nina enforces six canonical emotion categories: **`happy`**, **`sadness`**, **`anger`**, **`fear`**, **`love`**, **`surprise`**.

### 3.1 Direct Label Mappings (185,034 Records)
* `joy`, `happy`, `happiness` $\rightarrow$ **`happy`**
* `sadness`, `sad` $\rightarrow$ **`sadness`**
* `anger`, `angry` $\rightarrow$ **`anger`**
* `fear`, `fearful`, `scared` $\rightarrow$ **`fear`**
* `love`, `loving` $\rightarrow$ **`love`**
* `surprise`, `surprised` $\rightarrow$ **`surprise`**

### 3.2 Discarded / Ambiguous Labels (116,105 Records Discarded)
* `neutral` (12,252) & `normal` (16,343): Discarded (Neutral is out of discrete 6-class emotion taxonomy scope).
* `hate` (25,267): Discarded (Ambiguous between general anger vs toxic hate speech; blurs pure anger affect).
* `fun` (20,075), `enthusiasm` (10,000), `relief` (10,000): Discarded (Ambiguous affective states that blur pure happiness distinction).
* `depression` (10,333): Discarded (Clinical condition, not transient emotion affect).
* `worry` (4,475): Discarded (Ambiguous overlap between fear and sadness).
* `empty` (6,358), `shame` (146), `disgust` (856): Discarded (Out-of-scope taxonomy).

---

## 4. Duplicate & Conflict Resolution

* **Total Mapped Records Evaluated:** 185,034
* **Total Unique Normalized Texts:** 87,895
* **Label Conflict Instances:** 737 texts occurred with conflicting emotion labels across sources.
* **Resolution Action:** All 737 conflicting text instances were completely purged to ensure zero label noise.
* **Final Clean Candidate Dataset Size:** **87,158 unique records**.

---

## 5. Candidate Dataset 6-Class Distribution

| Canonical Emotion | Sample Count | Percentage | Class Balance Status |
| :--- | :--- | :--- | :--- |
| **`happy`** | 25,460 | 29.21% | Majority class |
| **`sadness`** | 18,972 | 21.77% | Well balanced |
| **`love`** | 18,471 | 21.19% | Well balanced |
| **`surprise`** | 10,537 | 12.09% | Moderate |
| **`anger`** | 10,179 | 11.68% | Moderate |
| **`fear`** | 3,539 | 4.06% | Minority class |
| **TOTAL** | **87,158** | **100.00%** | **Clean Candidate Dataset** |

---

## 6. Reproducible 80/10/10 Train / Validation / Test Splits

The dataset was partitioned using fixed seed (`seed=42`) into:

* `data/processed/train.csv`: 69,726 samples (80.0%)
* `data/processed/val.csv`: 8,716 samples (10.0%)
* `data/processed/test.csv`: 8,716 samples (10.0%)

### Zero Data Leakage Verification
* Train $\leftrightarrow$ Validation text overlap: **0**
* Train/Validation $\leftrightarrow$ Test text overlap: **0**

---

## 7. Model Recommendation for Future Retraining

### Recommended Primary Architecture: `DistilBERT` (`distilbert-base-uncased`)
* **Parameter Count:** ~66 Million (~260 MB model footprint).
* **Inference Speed:** ~35 ms (CPU) / ~8 ms (CUDA GPU).
* **Target Metric:** ~92% Macro F1.
* **Why Selected:** Delivers 97% of BERT's performance with 40% smaller footprint and 60% faster CPU inference speed, perfectly aligning with Nina's local-first deployment architecture.
