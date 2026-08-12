# Nina — Model Error Analysis & Misclassification Specification

**Document Version:** 2.0.0  
**Status:** Approved  
**Author:** Lead Software Architect & Senior AI/ML Engineer  

---

## 1. Executive Summary

This document performs empirical error analysis on misclassified test samples from the held-out test evaluation (`data/processed/test.csv`, 8,716 samples) comparing **Model A (Classical Baseline)** against **Model B (Fine-Tuned GPU DistilBERT)**.

---

## 2. Model Error Comparison & Analysis

### GPU Fine-Tuned DistilBERT Error Analysis (86.00% Accuracy, 0.8208 Macro F1)

1. **`love` Highest Classification Performance (97.51% F1):**
   * Out of 1,846 true `love` samples in the test split, 1,783 were correctly classified (**96.59% recall**). Only 20 were misclassified as `happy`. Romantic/affectionate tokens (*love, adore, cherish, sweetheart*) form exceptionally distinct vector representations in transformer attention layers.

2. **`anger` High Precision (89.34% Precision, 86.39% F1):**
   * DistilBERT correctly identified 838 of 1,002 true `anger` test utterances. Aggressive phrasing (*enraged, furious, hate, outraged*) maps to distinct high-energy attention embeddings.

3. **`happy` $\leftrightarrow$ `surprise` Confusion (104 false surprises, 125 false happy):**
   * Astonishment tokens (*wow, amazed, unbelievable*) combined with positive sentiment often blur boundaries between `happy` and `surprise`.

4. **`fear` Minority Class Challenges (64.02% F1):**
   * `fear` represents only 5.1% of the dataset (350 test samples). While DistilBERT improved recall from 0.0% to 60.00%, vulnerable expressions (*scared, terrified, anxious*) are occasionally confused with `sadness` (57 false sadness) or `happy` (38 false happy).

---

## 3. Confusion Matrix Comparison

### Model A: Classical Baseline (78.69% Accuracy)

```
               Predicted ->
True Emotion   happy  sadness  anger   fear   love  surprise
happy           2007     100     69    137     42       224
sadness          116    1336     80    152     47       171
anger             32      63    788     52     18        49
fear              26      24     31    227      7        35
love              42      11     16      9   1733        35
surprise          81      79     29     60     20       768
```

### Model B: GPU Fine-Tuned DistilBERT (86.00% Accuracy)

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
