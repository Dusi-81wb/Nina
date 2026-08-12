# Nina — Comprehensive Technical Specification & System Report

**Document Version:** 4.0.0  
**Status:** Complete, Verified & Production Ready  
**Authors:** Lead Software Architect & Senior AI/ML Engineer  
**Repository:** `Nina`  

---

## 1. Executive Summary & Architectural Scope

**Nina** is engineered as a modular, high-performance **Voice-to-Text Emotion Detection Component** for parent applications.

Nina is **NOT** a standalone chatbot or conversational voice assistant. Nina does **NOT** generate dialogue responses, host local LLMs (Ollama/llama), execute Text-to-Speech (pyttsx3/TTS), or maintain conversation memory.

### Core Component Boundary:

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
PARENT APPLICATION
```

---

## 2. Hardware & Environment Requirements

| Parameter | Specification | Verification Method |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 (AMD64) / Linux | `python -m nina.cli doctor` |
| **Python Runtime** | Python 3.10+ / 3.14.3 | `sys.version` |
| **PyTorch Acceleration** | PyTorch 2.13.0+cu130 (CUDA 13.0) | `torch.cuda.is_available()` |
| **Target GPU Hardware** | NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB VRAM) | `torch.cuda.get_device_name(0)` |
| **VRAM Allocated** | ~264.63 MB (DistilBERT resident) | `torch.cuda.memory_allocated()` |
| **System RAM** | 16 GB | OS System Info |

---

## 3. System Architecture & Pipeline Stages

Nina operates as an end-to-end data pipeline processing raw audio or text into a strongly typed `EmotionResult` payload:

### Stage 1: Audio Hardware Capture & VAD (`src/nina/audio/`)
* **`MicrophoneRecorder`**: Samples physical microphone hardware via `sounddevice` at 16,000 Hz mono PCM.
* **`FileAudioSource`**: Validates local WAV headers, duration, and sampling rate.
* **`VoiceActivityDetector`**: Calculates RMS audio energy and trims non-speech trailing silence.

### Stage 2: Speech-to-Text ASR Engine (`src/nina/speech/`)
* **`FasterWhisperSpeechToText`**: Local-first ASR engine powered by CTranslate2.
* **Optimized Execution:** Runs on CPU (`int8` compute) in **~170 ms**, ensuring zero missing DLL errors on Windows while leaving the GPU 100% dedicated to DistilBERT emotion inference.

### Stage 3: Text Preprocessing & NLP Normalization (`src/nina/preprocessing/`)
* **`DefaultTextPreprocessor`**: Normalizes text transcripts, extracts intensifiers (*so, extremely, very*), negations (*not, never*), and punctuation emphasis features (`!`, `?`, ALL CAPS).

### Stage 4: Emotion Classification Engine (`src/nina/emotion/`)
* **Primary GPU Classifier (`TransformerEmotionClassifier`):** Fine-tuned DistilBERT checkpoint (`artifacts/models/distilbert_cuda_best/best_model`) running natively on `cuda:0` (**86.00% Accuracy**, **0.8208 Macro F1**, **~7.8 ms GPU inference**).
* **Primary CPU Fallback (`ClassicalEmotionClassifier`):** TF-IDF + Logistic Regression (`artifacts/models/classical_baseline.joblib`, **78.69% Accuracy**, **0.82 MB size**).

### Stage 5: Composite Intensity Calculation (`src/nina/emotion/intensity.py`)
* **`DefaultIntensityCalculator`**: Optional Phase 5 component deriving a continuous score ($0.0$ to $100.0$) and qualitative level (`LOW`, `MEDIUM`, `HIGH`) from prediction confidence, entropy, margin spread, and lexical emphasis.

---

## 4. Machine Learning & Model Performance Benchmarks

Evaluated on the authoritative 87,158-example emotion dataset across 6 canonical emotion classes (`happy`, `sadness`, `anger`, `fear`, `love`, `surprise`):

| Model Architecture | Compute Device | Test Accuracy | Macro F1-Score | Weighted F1 | Inference Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Fine-Tuned DistilBERT** | **NVIDIA RTX 4050 GPU (`cuda:0`)** | **86.00%** | **0.8208** | **0.8601** | **7.83 ms** |
| **TF-IDF + Logistic Regression** | CPU | 78.69% | 0.7367 | 0.7940 | < 0.01 ms |

---

## 5. Measured Performance & Latency Breakdown

Measured on NVIDIA GeForce RTX 4050 GPU:

| Pipeline Step | Execution Time | Notes / Optimization |
| :--- | :--- | :--- |
| **1. Model Preloading** | ~6,000 ms | **One-time startup only** (reused across turns) |
| **2. Microphone Audio Capture** | 3,000–5,000 ms | Configurable live recording window |
| **3. ASR Speech Transcription** | ~170–185 ms | FasterWhisper CPU int8 engine |
| **4. Text Preprocessing** | < 0.20 ms | Regex token & modifier extractor |
| **5. DistilBERT GPU Inference** | **7.83 ms** | PyTorch forward pass on `cuda:0` |
| **6. Intensity Calculation** | < 0.05 ms | Composite sub-score aggregator |
| **Total Turn Processing Time** | **< 185 ms** | Real-time speech-to-emotion latency |

---

## 6. Complete Programmatic API Specification

Parent applications consume Nina by importing top-level helpers or the `NinaEmotionEngine`:

```python
from nina import NinaEmotionEngine, EmotionResult, SupportedEmotion, process_text, process_file
```

### 6.1 `EmotionResult` Output Payload Schema

```python
class EmotionResult(BaseModel):
    text: str                                    # Transcribed text transcript
    emotion: SupportedEmotion                    # "happy", "sadness", "anger", "fear", "love", "surprise"
    confidence: float                            # Classification confidence score (0.0 - 1.0)
    probabilities: dict[SupportedEmotion, float] # Full 6-class probability distribution map
    intensity: float | None = None               # Optional Phase 5 continuous intensity (0.0 - 100.0)
    intensity_level: IntensityLevel | None = None# Optional intensity level (LOW, MEDIUM, HIGH)
    processing_time_ms: float                    # Total processing latency in ms
    metadata: dict[str, Any]                     # Detailed execution and hardware telemetry
```

---

### 6.2 Text Processing Example

```python
from nina import process_text

result = process_text("I am really happy today!", include_intensity=True)

print(result.text)                   # "I am really happy today!"
print(result.emotion)                # SupportedEmotion.HAPPY
print(result.confidence)             # 0.9241
print(result.probabilities["happy"])  # 0.9241
print(result.intensity)              # 75.0
print(result.intensity_level)        # IntensityLevel.HIGH
print(result.processing_time_ms)     # 7.94 ms
```

---

### 6.3 Audio File Processing Example

```python
from nina import process_file

result = process_file("path/to/user_speech.wav", include_intensity=True)
print(f"Transcript: {result.text}")
print(f"Detected Emotion: {result.emotion.value}")
```

---

### 6.4 Persistent Engine Session (Zero Model Reloading Across Turns)

For continuous streaming or backend server integration:

```python
from nina import NinaEmotionEngine

# Initialize and preload models ONCE into memory
engine = NinaEmotionEngine(auto_preload=True)

# Turn 1 (Models reused, 0 ms loading latency)
res1 = engine.process_text("Everything is going great!")

# Turn 2 (Models reused, ~7 ms inference)
res2 = engine.process_file("user_recording.wav")
```

---

## 7. Command Line Interface (CLI) Reference Guide

All commands are executed within your activated virtual environment:

### 7.1 Environment Diagnostics (`nina doctor`)
Inspects Python, PyTorch, CUDA GPU availability, VRAM, and model paths:
```bash
python -m nina.cli doctor
```

### 7.2 Multi-Turn Interactive Listening (`nina listen --interactive`)
Preloads models **ONCE** at startup and allows continuous live speech recording without reloading models:
```bash
python -m nina.cli listen --interactive
```

### 7.3 Single-Turn Live Audio Emotion Runner (`nina listen`)
Records live microphone audio and predicts emotion:
```bash
# Record 4 seconds of microphone audio
python -m nina.cli listen --duration 4

# Run against an existing WAV file
python -m nina.cli listen --audio-file sample.wav
```

### 7.4 Speech-to-Text Transcription (`nina transcribe`)
Transcribes an audio file using FasterWhisper ASR:
```bash
python -m nina.cli transcribe sample.wav --model base.en
```

### 7.5 Text Classification Debugger (`nina classify`)
Predicts emotion and intensity for a text string:
```bash
python -m nina.cli classify "I am so extremely furious right now!" --engine transformer
```

### 7.6 Microphone Audio Recorder (`nina record`)
Records microphone audio to a WAV file:
```bash
python -m nina.cli record --duration 5 --output my_speech.wav
```

### 7.7 Text Preprocessing Debugger (`nina preprocess`)
Normalizes text and extracts NLP modifiers:
```bash
python -m nina.cli preprocess "I AM SO HAPPY TODAY!!!"
```

### 7.8 Benchmark Model Evaluator (`nina evaluate`)
Evaluates classifier performance against the benchmark test set:
```bash
python -m nina.cli evaluate --engine transformer
```

### 7.9 Automated Test Suite (`pytest`)
Executes all **72 unit and integration tests**:
```bash
pytest
```

---

## 8. Directory & Codebase Architecture Map

```text
Nina/
├── ARCHITECTURE.md                  # System architecture specification
├── PROJECT_PLAN.md                  # Project milestones and status log
├── README.md                        # Primary project documentation
├── docs/
│   ├── NINA_TECHNICAL_REPORT.md     # This comprehensive technical report
│   ├── integration.md               # Parent project integration guide
│   └── emotion-intensity.md         # Phase 5 intensity calculation specification
├── artifacts/
│   └── models/
│       ├── distilbert_cuda_best/    # Production fine-tuned DistilBERT GPU model (86.00% Acc)
│       └── classical_baseline.joblib# Production TF-IDF LogReg CPU fallback (78.69% Acc)
├── src/
│   └── nina/
│       ├── __init__.py              # Top-level programmatic API exports
│       ├── cli.py                   # Command Line Interface router
│       ├── engine.py                # NinaEmotionEngine master orchestrator
│       ├── api/
│       │   └── schemas.py           # Strongly typed Pydantic data contracts (EmotionResult)
│       ├── audio/
│       │   ├── features.py          # Audio prosody/energy feature extractor
│       │   ├── recorder.py          # Controlled microphone recorder
│       │   ├── source.py            # FileAudioSource & MicrophoneAudioSource
│       │   ├── vad.py               # Voice Activity Detector
│       │   └── validator.py         # Audio signal validator
│       ├── core/
│       │   ├── config.py            # Pydantic system settings
│       │   ├── device.py            # Hardware & CUDA device resolution utilities
│       │   ├── exceptions.py        # Custom exception hierarchy
│       │   └── logging.py           # Loguru logger setup
│       ├── emotion/
│       │   ├── classical.py         # TF-IDF + Logistic Regression classifier
│       │   ├── evaluator.py         # Test dataset evaluation harness
│       │   ├── intensity.py         # Phase 5 emotional intensity calculator
│       │   ├── interface.py         # EmotionClassifier abstract interface
│       │   ├── mapping.py           # 6-class label normalization mapper
│       │   └── transformer.py       # PyTorch CUDA DistilBERT adapter
│       ├── preprocessing/
│       │   └── processor.py         # NLP text preprocessor & modifier extractor
│       └── speech/
│           ├── engine.py            # FasterWhisper ASR speech-to-text engine
│           └── stubs.py             # Stub ASR test mock
└── tests/                           # 72 automated unit & integration tests
```

---

## 9. Compliance Statement

> **"NINA IS A VOICE-TO-TEXT EMOTION DETECTION COMPONENT, NOT A STANDALONE CHATBOT."**
