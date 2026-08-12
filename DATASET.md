# Nina — Dataset Specification & Data Management Strategy

**Document Version:** 1.2.0  
**Status:** Approved  
**Author:** Lead Software Architect & Senior AI/ML Engineer  

---

## 1. Primary Dataset Overview

Nina utilizes a clean, unified, multi-source emotion dataset (`data/processed/candidate_all.csv`, `train.csv`, `val.csv`, `test.csv`) constructed from three raw emotion corpora:
1. `data/raw/emotion_dataset.csv` (34,792 raw rows)
2. `data/raw/final_dataset.csv` (106,355 raw rows)
3. `data/raw/Emotion_Sentiment_DataSet.csv` (160,000 raw rows)

Following a comprehensive programmatic audit, label mapping, duplicate removal, and conflict purging, the final candidate dataset contains **87,158 clean, conflict-free, 6-class emotion text samples**.

---

## 2. Canonical Emotion Taxonomy & Distribution

| Canonical Emotion | Sample Count | Percentage | Class Balance Status |
| :--- | :--- | :--- | :--- |
| **`happy`** | 25,460 | 29.21% | Majority class |
| **`sadness`** | 18,972 | 21.77% | Well balanced |
| **`love`** | 18,471 | 21.19% | Well balanced |
| **`surprise`** | 10,537 | 12.09% | Moderate |
| **`anger`** | 10,179 | 11.68% | Moderate |
| **`fear`** | 3,539 | 4.06% | Minority class |
| **TOTAL** | **87,158** | **100.00%** | **Clean Candidate Corpus** |

---

## 3. Label Mapping & Discarded Categories

### 3.1 Mapped Direct Labels (185,034 Raw Records)
* `joy`, `happy`, `happiness` $\rightarrow$ **`happy`**
* `sadness`, `sad` $\rightarrow$ **`sadness`**
* `anger`, `angry` $\rightarrow$ **`anger`**
* `fear`, `fearful`, `scared` $\rightarrow$ **`fear`**
* `love`, `loving` $\rightarrow$ **`love`**
* `surprise`, `surprised` $\rightarrow$ **`surprise`**

### 3.2 Explicitly Discarded Categories (116,105 Records Discarded)
* `neutral` & `normal` (28,595): Discarded (Neutral is out of 6-class discrete emotion scope).
* `hate` (25,267): Discarded (Ambiguous between toxicity vs pure anger).
* `fun`, `enthusiasm`, `relief` (40,075): Discarded (Ambiguous affect blurs pure happiness).
* `depression` (10,333): Discarded (Clinical condition).
* `worry` (4,475): Discarded (Overlaps fear/sadness).
* `empty`, `shame`, `disgust` (7,360): Discarded (Out of taxonomy scope).

---

## 4. Reproducible 80/10/10 Train / Validation / Test Splits

* **Train Set (`data/processed/train.csv`):** 69,726 samples (80.0%)
* **Validation Set (`data/processed/val.csv`):** 8,716 samples (10.0%)
* **Test Set (`data/processed/test.csv`):** 8,716 samples (10.0%)
* **Data Leakage Check:** Verified **0** text overlap across train, validation, and test splits (fixed seed `42`).

For complete audit logs and conflict reports, see [docs/dataset-integration-audit.md](docs/dataset-integration-audit.md) and [docs/dataset-conflicts.md](docs/dataset-conflicts.md).
