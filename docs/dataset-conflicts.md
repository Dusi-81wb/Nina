# Nina — Dataset Duplicate & Conflict Resolution Specification

**Document Version:** 1.0.0  
**Status:** Approved  
**Author:** Lead Software Architect & Senior AI/ML Engineer  

---

## 1. Executive Summary

During the integration audit of `emotion_dataset.csv`, `final_dataset.csv`, and `Emotion_Sentiment_DataSet.csv`, multi-stage duplicate and label conflict detection was performed across 301,147 raw records.

To guarantee high model precision and eliminate label ambiguity during fine-tuning, all duplicate records were merged, and all instances of conflicting text labels were purged.

---

## 2. Duplicate Detection Stages

1. **Stage 1: Raw Exact Duplicate Rows**
   * Identified 25,529 exact duplicate CSV rows in `final_dataset.csv`.
   * Action: Deduplicated to 1 clean copy.

2. **Stage 2: Within-Dataset Duplicate Texts**
   * Identified 3,698 duplicate texts in `emotion_dataset.csv` and 72,018 duplicate texts in `Emotion_Sentiment_DataSet.csv`.
   * Action: Deduplicated to unique normalized text representations.

3. **Stage 3: Cross-Dataset Duplicate Texts**
   * Identified significant phrase overlap between `final_dataset.csv` and `Emotion_Sentiment_DataSet.csv`.
   * Action: Merged across datasets while tracking source attribution.

---

## 3. Label Conflict Resolution

When identical normalized text strings were mapped to conflicting emotion categories (e.g. `"I feel strange"` labelled as `sadness` in one source and `surprise` in another), a **Label Conflict** was flagged.

* **Total Label Conflict Instances Found:** 737 unique text strings.
* **Total Samples Affected:** 1,671 records.
* **Conflict Handling Strategy:** Rather than making subjective manual guesses, all 737 conflicting text groups were **purged from the candidate training corpus**.

### Example Conflict Instances Purged:
* `"i feel like a complete failure"` (Mapped as `sadness` vs `anger` across sources).
* `"i can't believe this happened"` (Mapped as `surprise` vs `fear` across sources).
* `"i am feeling really weird today"` (Mapped as `fear` vs `surprise` across sources).

---

## 4. Final Clean Corpus Integrity

By eliminating exact duplicates, cross-dataset duplicates, and 737 label conflict groups, the unified corpus was reduced from 301,147 raw records to **87,158 clean, conflict-free, multi-class samples** (`data/processed/candidate_all.csv`).
