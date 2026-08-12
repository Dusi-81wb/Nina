# Nina — Parent Project Integration Specification

**Document Version:** 2.0.0  
**Status:** Approved & Complete  
**Author:** Lead Software Architect & Senior AI/ML Engineer  

---

## 1. Integration Scope & Boundaries

**Nina** is engineered as a modular, production-grade **Voice-to-Text Emotion Detection Component** for parent applications.

Nina is **NOT** a standalone chatbot, conversational voice assistant, or LLM host. Nina does **NOT** generate dialogue responses, run local LLMs (Ollama/llama), execute Text-to-Speech (pyttsx3/TTS), or maintain conversation memory.

### Core Component Scope:

```
USER VOICE / AUDIO
        ↓
MICROPHONE / AUDIO FILE
        ↓
VOICE ACTIVITY DETECTION (VAD)
        ↓
SPEECH-TO-TEXT (FasterWhisper)
        ↓
TEXT PREPROCESSING (NLP Normalizer)
        ↓
EMOTION CLASSIFICATION (CUDA DistilBERT / CPU TF-IDF Fallback)
        ↓
STRUCTURED RESULT (EmotionResult)
        ↓
PARENT PROJECT
```

---

## 2. Programmatic API Integration

Parent applications consume Nina through top-level programmatic functions or the `NinaEmotionEngine` class:

### 2.1 Importing Nina

```python
from nina import NinaEmotionEngine, EmotionResult, SupportedEmotion, process_text, process_file
```

---

### 2.2 Text Emotion Detection Example

```python
from nina import process_text

# Process a text transcript
result = process_text("I am really happy today", include_intensity=True)

print(result.text)                 # "I am really happy today"
print(result.emotion)              # SupportedEmotion.HAPPY ("happy")
print(result.confidence)           # 0.9412
print(result.probabilities["happy"])# 0.9412
print(result.intensity)            # 88.50 (Optional Phase 5 intensity score)
print(result.intensity_level)      # IntensityLevel.HIGH
print(result.processing_time_ms)   # 1.25 ms
```

---

### 2.3 Audio File Emotion Detection Example

```python
from nina import process_file

# Process a WAV audio file
result = process_file("path/to/speech.wav", include_intensity=True)

print(f"Transcript: {result.text}")
print(f"Detected Emotion: {result.emotion.value}")
print(f"Confidence: {result.confidence:.4f}")
```

---

### 2.4 Persistent Engine Instance Example

For continuous streaming or multi-request backend integration:

```python
from nina import NinaEmotionEngine

# Initialize engine (automatically selects CUDA GPU DistilBERT if available, or CPU fallback)
engine = NinaEmotionEngine()

# Process multiple inputs efficiently
result1 = engine.process_text("I am feeling great!")
result2 = engine.process_file("user_recording.wav")
```

---

## 3. Output Contract (`EmotionResult`)

The primary contract returned by Nina to parent applications:

```python
class EmotionResult(BaseModel):
    text: str
    emotion: SupportedEmotion               # "happy", "sadness", "anger", "fear", "love", "surprise"
    confidence: float                       # 0.0 to 1.0
    probabilities: dict[SupportedEmotion, float] # Map across all 6 canonical emotions
    intensity: float | None = None          # Optional continuous intensity (0.0 to 100.0)
    intensity_level: IntensityLevel | None = None # Optional level ("low", "medium", "high")
    processing_time_ms: float
    metadata: dict[str, Any]
```

---

## 4. Environment & Hardware Requirements

* **Primary Production Model:** Fine-Tuned DistilBERT (`artifacts/models/distilbert_cuda_best/best_model`) loaded on `cuda:0` (**86.00% Test Accuracy**, **0.8208 Macro F1**).
* **CPU Fallback Model:** TF-IDF + Logistic Regression (`artifacts/models/classical_baseline.joblib`, **78.69% Test Accuracy**).
* **Hardware Acceleration:** Supports NVIDIA CUDA GPUs (verified on NVIDIA RTX 4050 Laptop GPU, 6GB VRAM) and automatically falls back to CPU when CUDA is unavailable.
* **Dependencies:** Zero LLM / Ollama / TTS dependencies.
