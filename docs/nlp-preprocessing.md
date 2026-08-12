# Nina — NLP Text Preprocessing Subsystem Specification

**Document Version:** 1.0.0  
**Status:** Approved  
**Author:** Lead Software Architect & Senior AI/ML Engineer  

---

## 1. Executive Summary & Design Philosophy

The **NLP Text Preprocessing Subsystem** prepares raw Speech-to-Text transcripts for downstream machine learning and deep learning emotion classification models.

### 1.1 Emotion Preservation vs. Generic NLP Cleaning
Traditional generic NLP pipelines aggressively strip punctuation, convert all text to lowercase, remove stopwords, and apply stemming. In emotion detection, **this naive approach destroys critical affective signals**.

* **Punctuation:** `"I am happy!"` vs `"I am happy!!!"` vs `"I am happy?"` convey dramatically different emotional affect and intensity.
* **Capitalization:** `"I LOVE THIS!"` carries strong emotional emphasis compared to `"i love this."`.
* **Negation:** Stripping stopwords would turn `"I am not happy"` into `"happy"`, completely inverting emotional polarity.
* **Emojis & Emoticons:** Express direct emotional state and are preserved.

---

## 2. Preprocessing Architecture & Pipeline Stages

Nina implements a modular, deterministic preprocessing pipeline via `DefaultTextPreprocessor`:

```
Raw Speech Transcript (str or SpeechResult)
                   ↓
      [1. Unicode Normalization] (NFKC)
                   ↓
    [2. Whitespace Normalization] (Collapse multi-space)
                   ↓
 [3. Contraction Expansion] (don't -> do not; preserving negation)
                   ↓
 [4. Character Lengthening Normalization] (sooooo -> soo)
                   ↓
   [5. Tokenization & Feature Extraction] (Intensifiers, Negations, Caps, !, ?)
                   ↓
        [PreprocessedText Payload]
```

---

## 3. Data Contracts & Configuration

### 3.1 PreprocessedText Data Model
The output payload satisfies the strongly typed `PreprocessedText` schema:

```json
{
  "raw_text": "I AM SOOO HAPPY!!!",
  "cleaned_text": "I AM SO HAPPY!!!",
  "tokens": ["I", "AM", "SO", "HAPPY", "!", "!", "!"],
  "intensifier_count": 1,
  "negation_count": 0,
  "punctuation_features": {
    "exclamations": 3,
    "questions": 0,
    "caps_words": 4
  },
  "processing_time_ms": 0.42,
  "metadata": {
    "token_count": 7
  }
}
```

### 3.2 Policy Configuration (`TextPreprocessorConfig`)

| Policy Setting | Default | Description |
| :--- | :--- | :--- |
| `normalize_unicode` | `True` | Applies NFKC normalization to standardize accents and quote styles. |
| `normalize_whitespace` | `True` | Trims leading/trailing spaces and collapses internal multi-space gaps. |
| `lowercase` | `False` | Preserves uppercase characters for emphasis detection. |
| `preserve_punctuation` | `True` | Preserves exclamation marks (`!`), question marks (`?`), and repeated punctuation. |
| `preserve_emojis` | `True` | Preserves unicode emoji representations (`❤️`, `😃`). |
| `expand_contractions` | `True` | Expands contractions (`don't` $\rightarrow$ `do not`), preserving explicit negation terms. |
| `handle_repeated_chars` | `True` | Normalizes excessive character lengthening (`sooooo` $\rightarrow$ `soo`). |
| `remove_stopwords` | `False` | MUST default to `False` to prevent erasing negation and intensity modifiers. |
| `apply_stemming` | `False` | MUST default to `False` to preserve word inflections for transformer tokenizers. |

---

## 4. Model Compatibility & Tokenization Strategy

### 4.1 Transformer Models (RoBERTa / DeBERTa)
* Transformer-based classification models incorporate dedicated byte-level BPE or WordPiece tokenizers (`roberta-base-tokenizer`).
* Nina's text preprocessor outputs `cleaned_text` directly to the transformer tokenizer without duplicating sub-word tokenization.

### 4.2 Classical ML Models (TF-IDF + SVM / XGBoost)
* For classical ML baselines, `PreprocessedText.tokens` provides pre-tokenized features, while `punctuation_features` (caps count, exclamations) supply explicit auxiliary dense features.

---

## 5. Data Leakage Prevention Guidelines

When training or fine-tuning models on reference emotion datasets (e.g. GoEmotions, CARER):
1. **Pipeline Consistency:** The exact same `TextPreprocessorConfig` policy must be executed during training, validation, testing, and live inference.
2. **Feature Fitting Isolation:** Any statistical transformation (e.g. TF-IDF vectorizers or feature scalers) MUST be fitted strictly on the training partition and transformed on test splits to prevent data leakage.

---

## 6. Command-Line Interface (CLI) Debugger

```bash
# Run preprocessing debugger on target text
nina preprocess "I AM SOOO HAPPY!!!"
```
