"""Master Nina Voice-to-Text Emotion Engine Orchestrator (Core Programmatic API)."""

import time
from pathlib import Path
from typing import Any

from nina.api.schemas import EmotionPrediction, EmotionResult, SpeechResult
from nina.audio.features import NinaAudioFeatures
from nina.audio.source import FileAudioSource
from nina.audio.vad import VoiceActivityDetector
from nina.core.config import get_settings
from nina.core.device import is_cuda_available
from nina.core.logging import logger
from nina.emotion.classical import ClassicalEmotionClassifier
from nina.emotion.intensity import DefaultIntensityCalculator
from nina.emotion.interface import EmotionClassifier
from nina.preprocessing.processor import DefaultTextPreprocessor
from nina.speech.engine import FasterWhisperSpeechToText
from nina.speech.interface import SpeechToTextEngine
from nina.speech.stubs import StubSpeechToTextEngine


class NinaEmotionEngine:
    """Production-grade voice-to-text emotion detection engine for parent applications."""

    def __init__(
        self,
        stt_engine: SpeechToTextEngine | None = None,
        emotion_classifier: EmotionClassifier | None = None,
        enable_vad: bool = True,
        enable_intensity: bool = True,
        auto_preload: bool = False,
    ) -> None:
        self.settings = get_settings()
        self.enable_vad = enable_vad
        self.enable_intensity = enable_intensity

        # 1. Voice Activity Detector
        self.vad = VoiceActivityDetector(
            energy_threshold=self.settings.vad_energy_threshold,
            silence_duration_seconds=self.settings.vad_silence_duration_seconds,
            sample_rate=self.settings.audio_sample_rate,
        )

        # 2. Speech-to-Text Engine (CPU mode for CTranslate2 ASR stability on Windows)
        if stt_engine:
            self.stt_engine = stt_engine
        elif self.settings.stt_engine == "stub":
            self.stt_engine = StubSpeechToTextEngine()
        else:
            self.stt_engine = FasterWhisperSpeechToText(
                model_size=self.settings.stt_model_size,
                device="cpu",
            )

        # 3. Preprocessor & Intensity Calculator
        self.preprocessor = DefaultTextPreprocessor()
        self.intensity_calc = DefaultIntensityCalculator()

        # 4. Emotion Classifier (CUDA DistilBERT / CPU Classical Fallback)
        if emotion_classifier:
            self.emotion_classifier = emotion_classifier
        elif is_cuda_available():
            try:
                from nina.emotion.transformer import TransformerEmotionClassifier

                self.emotion_classifier = TransformerEmotionClassifier(
                    model_name=str(self.settings.emotion_model_name),
                    device="cuda",
                )
                logger.info("NinaEmotionEngine initialized with CUDA GPU DistilBERT model.")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to load CUDA DistilBERT model: {e!s}. Using Classical Baseline fallback.")
                self.emotion_classifier = ClassicalEmotionClassifier()
        else:
            self.emotion_classifier = ClassicalEmotionClassifier()

        self._init_time_ms: float = 0.0
        self._is_preloaded: bool = False

        if auto_preload:
            self.preload_models()

    def preload_models(self) -> float:
        """Preload both ASR and Emotion Classifier models into memory ONCE.

        Returns:
            float: Initialization duration in milliseconds.
        """
        start_t = time.perf_counter()
        logger.info("Preloading Nina pipeline models into memory...")

        if hasattr(self.stt_engine, "load_model"):
            self.stt_engine.load_model()

        if hasattr(self.emotion_classifier, "load_model"):
            self.emotion_classifier.load_model()

        self._init_time_ms = round((time.perf_counter() - start_t) * 1000.0, 2)
        self._is_preloaded = True

        actual_device = getattr(self.emotion_classifier, "actual_device", "cpu")
        logger.info(f"Nina models preloaded in {self._init_time_ms} ms. Actual classifier device: '{actual_device}'.")
        return self._init_time_ms

    def process_text(
        self,
        text: str,
        include_intensity: bool = True,
    ) -> EmotionResult:
        """Process a text transcript and return structured EmotionResult.

        Args:
            text: Raw input text string.
            include_intensity: Whether to calculate optional Phase 5 intensity score.

        Returns:
            EmotionResult: Structured result payload.
        """
        start_t = time.perf_counter()

        init_ms = 0.0
        models_was_ready = getattr(self.emotion_classifier, "is_ready", lambda: True)()
        if not models_was_ready:
            t_init_start = time.perf_counter()
            if hasattr(self.emotion_classifier, "load_model"):
                self.emotion_classifier.load_model()
            init_ms = (time.perf_counter() - t_init_start) * 1000.0

        # 1. Preprocess Text
        t_prep_start = time.perf_counter()
        preprocessed = self.preprocessor.preprocess(text)
        prep_ms = (time.perf_counter() - t_prep_start) * 1000.0

        # 2. Predict Emotion
        t_emo_start = time.perf_counter()
        prediction: EmotionPrediction = self.emotion_classifier.predict(preprocessed)
        emo_ms = (time.perf_counter() - t_emo_start) * 1000.0

        # 3. Optional Intensity Calculation
        intensity_score = None
        intensity_level = None
        int_ms = 0.0

        if include_intensity and self.enable_intensity:
            t_int_start = time.perf_counter()
            intensity_res = self.intensity_calc.calculate_composite_intensity(prediction, preprocessed)
            intensity_score = intensity_res.intensity
            intensity_level = intensity_res.level
            int_ms = (time.perf_counter() - t_int_start) * 1000.0

        total_ms = (time.perf_counter() - start_t) * 1000.0

        actual_device = getattr(self.emotion_classifier, "actual_device", "cpu")
        gpu_alloc = 0.0
        try:
            import torch

            if torch.cuda.is_available():
                gpu_alloc = round(torch.cuda.memory_allocated() / (1024**2), 2)
        except Exception:  # noqa: BLE001, S110
            pass

        metadata = {
            "model_init_ms": round(init_ms, 3),
            "recording_time_ms": 0.0,
            "asr_inference_ms": 0.0,
            "preprocessor_ms": round(prep_ms, 3),
            "emotion_inference_ms": round(emo_ms, 3),
            "intensity_ms": round(int_ms, 3),
            "total_turn_ms": round(total_ms, 3),
            "models_reused": models_was_ready,
            "actual_model_device": actual_device,
            "gpu_memory_allocated_mb": gpu_alloc,
            "classifier": self.emotion_classifier.__class__.__name__,
        }

        return EmotionResult(
            text=text,
            emotion=prediction.emotion,
            confidence=prediction.confidence,
            probabilities=prediction.probabilities,
            intensity=intensity_score,
            intensity_level=intensity_level,
            processing_time_ms=round(total_ms, 2),
            metadata=metadata,
        )

    def process_file(
        self,
        file_path: Path | str,
        include_intensity: bool = True,
    ) -> EmotionResult:
        """Read a local audio file (e.g. WAV), transcribe, and predict emotion.

        Args:
            file_path: Path to target audio file.
            include_intensity: Whether to calculate optional Phase 5 intensity score.

        Returns:
            EmotionResult: Structured result payload.
        """
        source = FileAudioSource(file_path)
        return self.process_audio(source, include_intensity=include_intensity)

    def process_audio(
        self,
        audio_source: Any,
        include_intensity: bool = True,
    ) -> EmotionResult:
        """Process an audio source through ASR -> Preprocessor -> Emotion Classifier -> EmotionResult.

        Args:
            audio_source: AudioSource implementation (FileAudioSource, MicrophoneAudioSource, etc.).
            include_intensity: Whether to calculate optional Phase 5 intensity score.

        Returns:
            EmotionResult: Structured result payload.
        """
        start_t = time.perf_counter()

        init_ms = 0.0
        stt_ready = getattr(self.stt_engine, "is_ready", lambda: True)()
        emo_ready = getattr(self.emotion_classifier, "is_ready", lambda: True)()
        models_was_ready = stt_ready and emo_ready

        if not models_was_ready:
            t_init_start = time.perf_counter()
            if hasattr(self.stt_engine, "load_model"):
                self.stt_engine.load_model()
            if hasattr(self.emotion_classifier, "load_model"):
                self.emotion_classifier.load_model()
            init_ms = (time.perf_counter() - t_init_start) * 1000.0

        # 1. Speech-to-Text Transcription
        t_asr_start = time.perf_counter()
        speech_result: SpeechResult = self.stt_engine.transcribe(audio_source)
        asr_ms = (time.perf_counter() - t_asr_start) * 1000.0

        # 2. Extract Acoustic Features (if file exists)
        audio_features = None
        rec_time_ms = 0.0
        if hasattr(audio_source, "file_path") and Path(audio_source.file_path).exists():
            extractor = NinaAudioFeatures(target_sample_rate=self.settings.audio_sample_rate)
            audio_features = extractor.extract_from_file(audio_source.file_path)
            if hasattr(audio_source, "get_audio_input"):
                try:
                    rec_time_ms = round(audio_source.get_audio_input().duration_seconds * 1000.0, 2)
                except Exception:  # noqa: BLE001, S110
                    pass

        # 3. Preprocess Transcript
        t_prep_start = time.perf_counter()
        preprocessed = self.preprocessor.preprocess(speech_result.text)
        prep_ms = (time.perf_counter() - t_prep_start) * 1000.0

        # 4. Predict Emotion
        t_emo_start = time.perf_counter()
        prediction: EmotionPrediction = self.emotion_classifier.predict(preprocessed)
        emo_ms = (time.perf_counter() - t_emo_start) * 1000.0

        # 5. Optional Intensity Calculation
        intensity_score = None
        intensity_level = None
        int_ms = 0.0

        if include_intensity and self.enable_intensity:
            t_int_start = time.perf_counter()
            intensity_res = self.intensity_calc.calculate_composite_intensity(prediction, preprocessed, audio_features)
            intensity_score = intensity_res.intensity
            intensity_level = intensity_res.level
            int_ms = (time.perf_counter() - t_int_start) * 1000.0

        total_ms = (time.perf_counter() - start_t) * 1000.0

        actual_device = getattr(self.emotion_classifier, "actual_device", "cpu")
        gpu_alloc = 0.0
        try:
            import torch

            if torch.cuda.is_available():
                gpu_alloc = round(torch.cuda.memory_allocated() / (1024**2), 2)
        except Exception:  # noqa: BLE001, S110
            pass

        metadata = {
            "model_init_ms": round(init_ms, 3),
            "recording_time_ms": rec_time_ms,
            "asr_inference_ms": round(asr_ms, 3),
            "preprocessor_ms": round(prep_ms, 3),
            "emotion_inference_ms": round(emo_ms, 3),
            "intensity_ms": round(int_ms, 3),
            "total_turn_ms": round(total_ms, 3),
            "models_reused": models_was_ready,
            "actual_model_device": actual_device,
            "gpu_memory_allocated_mb": gpu_alloc,
            "stt_engine": self.stt_engine.__class__.__name__,
            "classifier": self.emotion_classifier.__class__.__name__,
        }

        return EmotionResult(
            text=speech_result.text,
            emotion=prediction.emotion,
            confidence=prediction.confidence,
            probabilities=prediction.probabilities,
            intensity=intensity_score,
            intensity_level=intensity_level,
            processing_time_ms=round(total_ms, 2),
            metadata=metadata,
        )
