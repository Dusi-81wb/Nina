# Nina — System Architecture Specification

**Document Version:** 3.0.0  
**Status:** Approved & Complete  
**Author:** Lead Software Architect & Senior AI/ML Engineer  

---

## 1. Executive Summary & Project Responsibility

**Nina** is engineered as a modular, production-grade **Voice-to-Text Emotion Detection Component** for parent applications.

Nina is **NOT** a standalone chatbot or conversational voice assistant. Nina does **NOT** generate response text, host local LLMs (Ollama/llama), execute Text-to-Speech (pyttsx3/TTS), or maintain conversation memory.

---

## 2. Core Architecture Pipeline

```
USER VOICE / AUDIO SIGNAL
        ↓
MICROPHONE / WAV AUDIO FILE
        ↓
VOICE ACTIVITY DETECTION (VAD) — Trimming Background Silence
        ↓
SPEECH-TO-TEXT (FasterWhisper ASR) — Speech Transcription
        ↓
TEXT PREPROCESSING (NLP Normalizer) — Tokenization & Modifiers
        ↓
EMOTION CLASSIFICATION (CUDA DistilBERT / CPU TF-IDF Fallback)
        ↓
OPTIONAL INTENSITY ENGINE (Phase 5 Sub-Score Calculation)
        ↓
STRUCTURED RESULT (EmotionResult)
        ↓
PARENT PROJECT
```

---

## 3. Production Machine Learning Models

* **Primary Production GPU Model:** Fine-Tuned DistilBERT (`artifacts/models/distilbert_cuda_best/best_model`) loaded on `cuda:0` (**86.00% Test Accuracy**, **0.8208 Macro F1**, **0.734 ms GPU latency**).
* **Primary Production CPU Fallback Model:** TF-IDF + Logistic Regression (`artifacts/models/classical_baseline.joblib`, **78.69% Test Accuracy**, **0.7367 Macro F1**, **< 0.01 ms CPU latency**).
* **Canonical 6-Class Taxonomy:** `happy`, `sadness`, `anger`, `fear`, `love`, `surprise`.

---

## 4. Programmatic API Contract (`EmotionResult`)

Returned by `NinaEmotionEngine` (`process_audio()`, `process_file()`, `process_text()`):

```python
class EmotionResult(BaseModel):
    text: str                                    # Transcribed text
    emotion: SupportedEmotion                    # Primary predicted emotion
    confidence: float                            # Classification confidence (0.0 - 1.0)
    probabilities: dict[SupportedEmotion, float] # 6-class probability distribution
    intensity: float | None = None               # Optional Phase 5 intensity score (0 - 100)
    intensity_level: IntensityLevel | None = None# Optional intensity level (LOW/MED/HIGH)
    processing_time_ms: float
    metadata: dict[str, Any]
```
