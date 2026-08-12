# Architectural Decision Record (ADR) 0001

## Title
Voice-Based Emotion Detection Modular Architecture and Model Selection Strategy

## Context
Nina is designed as a production-grade AI system. In Phase 1, the system must support a single, dedicated core feature: **Voice-Based Emotion Detection**. 

The core user experience is defined as:
> "The user speaks a sentence through the microphone, and the system converts the speech into text using a speech-to-text module. The converted text is then passed to an ML/DL-based emotion classifier to detect emotions such as happy, sadness, anger, fear, love, or surprise, along with a confidence score to estimate emotion intensity."

Future phases will expand Nina to support GA-2, conversational AI, vector memory, RAG, and multimodal processing. Therefore, Phase 1 architecture must satisfy strict decoupling guarantees without introducing unnecessary abstraction bloat.

## Decision
1. **Modular Abstract Layering**: Define strict abstract base classes (interfaces) for:
   - `AudioRecorderInterface` (`nina.audio`)
   - `SpeechToTextInterface` (`nina.speech`)
   - `TextPreprocessorInterface` (`nina.preprocessing`)
   - `EmotionClassifierInterface` (`nina.emotion`)
   - `IntensityCalculatorInterface` (`nina.emotion`)
   - `EmotionPipelineInterface` (`nina.inference`)

2. **Model Selection Strategy**:
   - **ASR Layer**: Selected `faster-whisper` (`base.en` / `small.en`) using CTranslate2 backend. This delivers ~4x speedup over standard PyTorch Whisper with float16/int8 quantization, lower VRAM requirements (~1GB), and offline execution.
   - **Emotion Classifier Layer**: Selected Transformer architecture (`bhadresh-ps/roberta-base-emotion`). Fine-tuned specifically for 6 discrete emotion categories (`happy`, `sadness`, `anger`, `fear`, `love`, `surprise`) matching exact project requirements.

3. **Confidence vs. Intensity Distinction**:
   - Model confidence ($P(\hat{y} \mid x)$) represents classifier statistical certainty.
   - Emotional intensity (`low`, `medium`, `high`) is calculated as a composite metric combining:
     - Logit magnitude ratio / softmax distance from uniform distribution.
     - Linguistic amplifier/dimmer count detected during preprocessing (e.g., "extremely", "slightly").

4. **Extensibility Guarantee**:
   - The emotion detection pipeline exposes a single top-level interface `EmotionPipelineInterface`. Future voice assistant or agent orchestration modules can consume this interface directly as a standalone subservice.

## Consequences
### Positive
- Independent swapability of ASR or Emotion models without touching adjacent code.
- Predictable contract testing via mocked layer interfaces.
- Low local computational footprint capable of running on modest consumer GPUs/CPUs.

### Negative
- Text-only emotion classification omits acoustic prosody features (pitch, energy, tone) in Phase 1.
- Initial intensity model relies on heuristic rules overlaid on model logits until fine-tuned intensity annotated datasets are integrated.

## Status
Accepted

## Date
2026-08-11
