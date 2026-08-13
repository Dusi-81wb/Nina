# Nina — Voice-to-Text Emotion Detection Component

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.14-blue.svg)](https://www.python.org/)
[![CUDA Acceleration](https://img.shields.io/badge/CUDA-13.0%20%7C%20RTX%204050-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Architecture: Decoupled](https://img.shields.io/badge/Architecture-Modular-green.svg)](ARCHITECTURE.md)
[![Technical Report: Full Specs](https://img.shields.io/badge/Documentation-Technical%20Report-purple.svg)](docs/NINA_TECHNICAL_REPORT.md)
[![Tests: 72%20Passed](https://img.shields.io/badge/Tests-72%20Passed%20%2865%25%20Cov%29-brightgreen.svg)](tests/)

---

## 1. What Nina Is

**Nina** is a modular, production-grade **Voice-to-Text Emotion Detection Component** for parent applications.

Nina is **NOT** a standalone chatbot or conversational voice assistant. Nina does **NOT** generate dialogue responses, host local LLMs (Ollama/llama), execute Text-to-Speech (pyttsx3/TTS), or maintain conversation memory.

### Core Component Architecture:

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

## 2. Programmatic Integration API

Parent applications consume Nina by importing top-level helpers or the `NinaEmotionEngine`:

```python
from nina import NinaEmotionEngine, EmotionResult, process_text, process_file

# 1. Text Emotion Detection
result = process_text("I am really happy today", include_intensity=True)
print(result.text)                   # "I am really happy today"
print(result.emotion)                # SupportedEmotion.HAPPY ("happy")
print(result.confidence)             # 0.9241
print(result.probabilities["happy"]) # 0.9241
print(result.intensity)              # 75.0 (Optional Phase 5 intensity score)

# 2. Audio File Emotion Detection
audio_result = process_file("path/to/speech.wav", include_intensity=True)
print(f"Transcript: {audio_result.text}")
print(f"Detected Emotion: {audio_result.emotion.value}")

# 3. Persistent Engine Instance (Zero Model Reloading Across Turns)
engine = NinaEmotionEngine(auto_preload=True)
res1 = engine.process_text("Everything is going great!")
res2 = engine.process_file("user_speech.wav")
```

---

## 3. Measured Model & Component Latency

| Stage | Measured Execution Duration | Component / Notes |
| :--- | :--- | :--- |
| **1. Model Preloading** | ~6,000 ms | **One-time startup only** (reused across turns) |
| **2. ASR Transcription** | ~170 ms | FasterWhisper CPU int8 engine |
| **3. Text Preprocessing** | < 0.20 ms | DefaultTextPreprocessor |
| **4. Emotion Classification** | **7.83 ms (CUDA GPU)** / 28.89 ms (CPU) | Fine-Tuned DistilBERT on `cuda:0` |
| **5. Intensity Engine** | < 0.05 ms | DefaultIntensityCalculator |
| **Total Turn Processing Time** | **< 185 ms** | Full Nina Real-Time Pipeline |

---

## 4. Quick Start & CLI Diagnostics

```bash
# 1. Run environment diagnostic check (Verifies RTX 4050 GPU & CUDA 13)
python -m nina.cli doctor

# 2. Run multi-turn interactive session (Preloads models once into VRAM)
python -m nina.cli listen --interactive

# 3. Single-turn live audio recording & emotion detection
python -m nina.cli listen --duration 4

# 4. Transcribe an audio file
python -m nina.cli transcribe sample.wav --model base.en

# 5. Classify emotion and intensity for a text string
python -m nina.cli classify "I AM SO EXTREMELY HAPPY TODAY!!!"

# 6. Record microphone audio to WAV file
python -m nina.cli record --duration 3 --output test.wav

# 7. Run complete automated test suite (72 tests)
pytest
```

---

## 5. Comprehensive Documentation

* 📖 **[Full Technical Report](docs/NINA_TECHNICAL_REPORT.md):** Complete hardware requirements, model benchmarks, pipeline stages, and full codebase specification.
* 🔌 **[Parent Project Integration Guide](docs/integration.md):** Programmatic API usage and `EmotionResult` data contracts for parent applications.
* 🏗️ **[System Architecture Specification](ARCHITECTURE.md):** Technical design document.
