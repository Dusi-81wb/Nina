# Nina — System Requirements Specification (SRS)

**Document Version:** 1.0.0  
**Status:** Approved  
**Author:** Lead Software Architect & Senior AI/ML Engineer  

---

## 1. Executive Summary & Scope

**Nina** is an artificial intelligence platform designed for modular expansion. In **Phase 1**, the scope is strictly restricted to **Voice-Based Emotion Detection**.

The voice emotion module receives real-time microphone speech input, transcribes the speech into text using an offline-capable Automatic Speech Recognition (ASR) engine, processes the transcript through a natural language processing (NLP) pipeline, classifies the emotion into one of six supported target categories, and estimates both statistical confidence and emotional intensity.

---

## 2. Target Scope & Emotion Taxonomy

### 2.1 Supported Emotions (6 Classes)
1. **`happy`**: Positive affect, joy, contentment, enthusiasm.
2. **`sadness`**: Negative affect, grief, disappointment, sorrow.
3. **`anger`**: Hostility, annoyance, irritation, rage.
4. **`fear`**: Anxiety, apprehension, dread, panic.
5. **`love`**: Affection, warmth, attachment, admiration.
6. **`surprise`**: Astonishment, wonder, unexpected realization.

---

## 3. Functional Requirements (FR)

| Requirement ID | Module | Description | Acceptance Criteria |
| :--- | :--- | :--- | :--- |
| **FR-101** | Audio Input | Capture raw microphone audio at 16kHz mono PCM. | Audio captured without clipping or dropping buffer frames for up to 30s. |
| **FR-102** | Audio Input | Validate minimum audio energy (RMS) before passing to ASR. | Reject silent or near-silent inputs with descriptive error (`AudioCaptureError`). |
| **FR-103** | Speech-to-Text | Transcribe spoken English audio into clean text transcripts. | Achieve < 15% WER on standard clear speech datasets locally. |
| **FR-104** | Text Preprocessing | Sanitize transcript while preserving emotion-bearing modifiers. | Retain intensifiers (e.g., "extremely", "slightly") and punctuation marks. |
| **FR-105** | Emotion Classifier | Predict probability distribution over the 6 supported emotion classes. | Output valid softmax probabilities summing to 1.0. |
| **FR-106** | Confidence | Extract model prediction confidence score ($0.0 - 1.0$). | Return top probability score $\max(P(Y \mid X))$. |
| **FR-107** | Intensity | Derive discrete emotional intensity (`low`, `medium`, `high`). | Intensity calculated via logit spread + linguistic modifier heuristics. |
| **FR-108** | API Contract | Expose structured JSON response complying with `EmotionResponse` schema. | Response contains text, emotion, confidence, intensity, and breakdown. |

---

## 4. Non-Functional Requirements (NFR)

### 4.1 Performance & Latency
* **NFR-201 (End-to-End Latency):** Complete pipeline processing (Audio Capture $\rightarrow$ Inference $\rightarrow$ Response) must execute within **$< 1.5$ seconds** on CUDA GPU and **$< 3.5$ seconds** on modern 8-core CPU.
* **NFR-202 (Throughput):** Support concurrent requests at single-batch inference without memory leaking.

### 4.2 Accuracy & Quality
* **NFR-203 (Classification Accuracy):** Emotion classification model must achieve **$\ge 85\%$ Macro F1-Score** on standard benchmark test split.
* **NFR-204 (ASR Accuracy):** Speech recognition must achieve **$\le 10\%$ WER** on clear microphone input.

### 4.3 Hardware & Resource Constraints
* **NFR-205 (VRAM Footprint):** GPU VRAM allocation must remain **$< 3.0$ GB** for both ASR and Emotion models combined.
* **NFR-206 (System Memory):** Total system RAM usage must not exceed **$4.0$ GB**.

### 4.4 Portability & Reliability
* **NFR-207 (Offline Capability):** System MUST run completely offline without requiring active cloud API calls or external network connectivity after initial model caching.
* **NFR-208 (Fault Tolerance):** Exception handling must prevent process crashes on corrupted audio or unparseable input.

---

## 5. System Hardware & Software Requirements

### 5.1 Operating System Compatibility
* Windows 10/11 (64-bit)
* Ubuntu 20.04/22.04 LTS (64-bit)
* macOS 12+ (Apple Silicon M-series supported)

### 5.2 Hardware Specifications
* **CPU:** Quad-Core Intel i5/i7 (10th gen+) or AMD Ryzen 5/7
* **RAM:** Minimum 8 GB (16 GB recommended)
* **GPU (Optional but Recommended):** NVIDIA GTX 1660 / RTX 2060 (6GB VRAM) or higher with CUDA 11.8 / 12.1 support
* **Audio Hardware:** Any Windows/ALSA compatible hardware microphone

### 5.3 Software & Environment
* **Python:** 3.10.x or 3.11.x
* **Deep Learning Framework:** PyTorch $\ge 2.0.0$
* **ASR Backend:** `faster-whisper` (CTranslate2)
* **NLP Framework:** Hugging Face `transformers` $\ge 4.36.0$

---

## 6. Out-of-Scope (Phase 1 Explicit Non-Goals)

To enforce strict project focus, the following capabilities are **explicitly out of scope**:
* ❌ GA-2 / General AI Assistant orchestration
* ❌ Conversational Chatbot or LLM dialog manager
* ❌ Retrieval-Augmented Generation (RAG) or Vector Databases
* ❌ Multi-agent systems
* ❌ Long-term memory or user profiling databases
* ❌ Text-to-Speech (TTS) voice response synthesis
* ❌ Computer Vision / Facial emotion analysis
* ❌ Telegram / Mobile app / Cloud deployment
