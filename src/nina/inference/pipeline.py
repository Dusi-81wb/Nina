"""Modular pipeline orchestrator for end-to-end Voice Emotion Detection."""

import time
from abc import ABC, abstractmethod

from nina.api.schemas import AudioInput, EmotionResult
from nina.core.logging import log_execution_time
from nina.emotion.intensity import HeuristicIntensityCalculator
from nina.emotion.interface import EmotionClassifier, IntensityCalculator
from nina.preprocessing.interface import TextPreprocessor
from nina.preprocessing.processor import DefaultTextPreprocessor
from nina.speech.interface import SpeechToTextEngine


class EmotionPipeline(ABC):
    """Abstract interface defining the complete end-to-end voice emotion pipeline contract."""

    @abstractmethod
    def process(self, audio: AudioInput) -> EmotionResult:
        """Process AudioInput signal through STT -> Preprocessing -> Emotion -> Intensity.

        Args:
            audio: AudioInput model payload.

        Returns:
            EmotionResult: Complete pipeline result contract.
        """

    @abstractmethod
    def process_text(self, text: str) -> EmotionResult:
        """Shortcut pipeline execution skipping ASR step for direct text input.

        Args:
            text: Input text string.

        Returns:
            EmotionResult: Complete pipeline result contract.
        """


class ModularEmotionPipeline(EmotionPipeline):
    """Production pipeline implementation accepting injectable layer engines."""

    def __init__(
        self,
        stt_engine: SpeechToTextEngine,
        classifier: EmotionClassifier,
        preprocessor: TextPreprocessor | None = None,
        intensity_calculator: IntensityCalculator | None = None,
    ) -> None:
        self.stt_engine = stt_engine
        self.classifier = classifier
        self.preprocessor = preprocessor or DefaultTextPreprocessor()
        self.intensity_calculator = intensity_calculator or HeuristicIntensityCalculator()

    def process(self, audio: AudioInput) -> EmotionResult:
        """Process AudioInput signal through STT -> Preprocessing -> Emotion -> Intensity."""
        start_time = time.perf_counter()

        # Step 1: Speech-to-Text
        with log_execution_time("SpeechToText Transcription"):
            speech_result = self.stt_engine.transcribe(audio)

        # Step 2: Preprocessing
        with log_execution_time("Text Preprocessing"):
            cleaned_text = self.preprocessor.preprocess(speech_result.text)

        # Step 3: Emotion Classification
        with log_execution_time("Emotion Classification"):
            prediction = self.classifier.predict(cleaned_text)

        # Step 4: Intensity Calculation
        with log_execution_time("Intensity Calculation"):
            intensity = self.intensity_calculator.calculate_intensity(prediction, cleaned_text)

        total_duration_ms = (time.perf_counter() - start_time) * 1000.0

        return EmotionResult(
            text=cleaned_text.cleaned_text,
            emotion=prediction.emotion,
            confidence=prediction.confidence,
            intensity=intensity,
            probabilities=prediction.probabilities,
            processing_time_ms=round(total_duration_ms, 2),
        )

    def process_text(self, text: str) -> EmotionResult:
        """Shortcut pipeline execution skipping ASR step for direct text input."""
        start_time = time.perf_counter()

        cleaned_text = self.preprocessor.preprocess(text)
        prediction = self.classifier.predict(cleaned_text)
        intensity = self.intensity_calculator.calculate_intensity(prediction, cleaned_text)

        total_duration_ms = (time.perf_counter() - start_time) * 1000.0

        return EmotionResult(
            text=cleaned_text.cleaned_text,
            emotion=prediction.emotion,
            confidence=prediction.confidence,
            intensity=intensity,
            probabilities=prediction.probabilities,
            processing_time_ms=round(total_duration_ms, 2),
        )
