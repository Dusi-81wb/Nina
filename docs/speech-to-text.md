# Nina — Speech-to-Text (ASR) Subsystem Guide

**Document Version:** 1.0.0  
**Status:** Approved  
**Author:** Lead Software Architect & Senior AI/ML Engineer  

---

## 1. Subsystem Overview

The **Speech-to-Text (ASR)** subsystem converts spoken audio signals (from local audio files or live hardware microphone streams) into clean text transcripts.

In alignment with [MODEL_SELECTION.md](../MODEL_SELECTION.md), Nina integrates **`faster-whisper`**, a high-performance CTranslate2 re-implementation of OpenAI Whisper.

```
Audio Input (WAV / Microphone)
             ↓
    [FileAudioSource / MicrophoneAudioSource]
             ↓
     [AudioValidator] (16kHz PCM, RMS energy silence check)
             ↓
 [FasterWhisperSpeechToText Engine] (CTranslate2 Backend)
             ↓
    [SpeechResult] (text, language, duration, processing_time_ms, RTF)
```

---

## 2. ASR Engine & Model Strategy

### 2.1 Model Selection & Local Storage Strategy
* **Default Model:** `faster-whisper` (`base.en`)
* **Model Footprint:** ~142 MB
* **Local Storage Directory:** Model weights are downloaded directly to `./models/` on first execution.
* **Offline Execution Guarantee:** Once model weights exist in `./models/`, Nina runs **100% offline** without requiring external cloud API requests or active internet connections.

### 2.2 Device Allocation & Fallback Architecture
* **CUDA Mode:** On systems with NVIDIA GPUs and CUDA drivers, Nina initializes `faster-whisper` with `device="cuda"` and `compute_type="float16"`, yielding latency $< 150\text{ms}$.
* **CPU Fallback:** On systems without CUDA, `FasterWhisperSpeechToText` automatically resolves device to `"cpu"` and adjusts compute precision to `"int8"`, yielding latency $< 500\text{ms}$ on modern multi-core processors without process failure.

---

## 3. Audio Source Contracts & Validation

### 3.1 Supported Audio Formats
* **Primary Format:** `WAV` (16-bit PCM, 16,000 Hz, Mono).
* **Extended Formats:** `FLAC`, `MP3`, `OGG`.

### 3.2 Signal Quality & Silence Detection
Before passing audio signals to the ASR engine, `AudioValidator` verifies:
1. **Duration Bounds:** $0.1\text{s} \le t \le 30.0\text{s}$.
2. **File Quality:** Rejects empty (0-byte) or corrupted header files (`AudioError`).
3. **Silence Rejection:** Calculates Root Mean Square (RMS) signal energy:
   $$\text{RMS} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2}$$
   If $\text{RMS} < 0.001$, the input is flagged as silent and rejected.

---

## 4. Privacy & Security Guarantees

* **In-Memory Streaming:** Microphone audio recordings are sampled directly into temporary in-memory NumPy arrays.
* **Automatic Destruction:** Audio buffers are discarded immediately after transcription completion. No audio recordings are uploaded to external cloud services or telemetry databases.

---

## 5. Command-Line Interface (CLI) Usage

### 5.1 Transcribe Local WAV Audio File
```bash
# Transcribe WAV file using default faster-whisper base.en model
nina transcribe sample.wav

# Transcribe specifying custom model size and target language
nina transcribe speech.wav --model small.en --language en

# Transcribe using development test stub (for fast test verification)
nina transcribe sample.wav --stub
```

### 5.2 Record Live Microphone Input
```bash
# Record live microphone audio for 5 seconds into memory buffer
nina record --duration 5.0

# Record live microphone audio and save to a local WAV file
nina record --duration 5.0 --output my_speech.wav
```

---

## 6. Performance Metrics & Real-Time Factor (RTF)

Nina measures transcription performance using the **Real-Time Factor (RTF)** metric:

$$\text{RTF} = \frac{\text{Processing Duration (seconds)}}{\text{Audio Duration (seconds)}}$$

* $\mathbf{\text{RTF} < 1.0}$: Faster than real-time (target requirement).
* **Expected Benchmark SLAs:**
  * **CUDA GPU (FP16):** $\text{RTF} \approx 0.05 - 0.10$ (~250ms for 5s audio).
  * **CPU (INT8):** $\text{RTF} \approx 0.15 - 0.35$ (~800ms for 5s audio).

---

## 7. Troubleshooting Guide

| Issue / Error | Cause | Resolution |
| :--- | :--- | :--- |
| `AudioError: Audio file not found` | Invalid file path specified | Verify file existence and relative path. |
| `AudioError: Input is silent` | Microphone RMS energy $< 0.001$ | Check microphone gain, mute switch, or speak louder. |
| `sounddevice package is required` | `sounddevice` dependency missing | Run `pip install sounddevice`. |
| `CUDA requested but not available` | NVIDIA CUDA drivers missing | System automatically falls back to CPU INT8 mode. |
