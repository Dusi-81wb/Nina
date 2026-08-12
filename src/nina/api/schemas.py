"""Strongly typed core data contracts for objects flowing through the Nina system."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SupportedEmotion(str, Enum):
    """Enumeration of the 6 supported emotion classes in Phase 1."""

    HAPPY = "happy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    LOVE = "love"
    SURPRISE = "surprise"


class IntensityLevel(str, Enum):
    """Enumeration of emotional intensity levels derived from confidence & linguistics."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AudioInput(BaseModel):
    """Data contract representing audio input signals passed into the pipeline."""

    source: str = Field(default="microphone", description="Audio input source (microphone, file, buffer)")
    sample_rate: int = Field(default=16000, description="Sampling rate in Hz")
    duration_seconds: float = Field(ge=0.1, description="Audio duration in seconds")
    channels: int = Field(default=1, description="Number of audio channels (1=mono)")
    signal_rms: float = Field(default=0.0, description="Root Mean Square audio energy level")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional audio hardware/file metadata")


class SpeechResult(BaseModel):
    """Data contract representing the transcript output from a Speech-to-Text engine."""

    text: str = Field(description="Transcribed text transcript")
    language: str = Field(default="en", description="Detected or configured language code")
    confidence: float = Field(ge=0.0, le=1.0, default=1.0, description="Average word confidence score")
    processing_time_ms: float = Field(default=0.0, description="ASR inference duration in milliseconds")


class PreprocessedText(BaseModel):
    """Data contract representing preprocessed text payload ready for downstream NLP models."""

    raw_text: str = Field(description="Original unfiltered raw transcript")
    cleaned_text: str = Field(description="Normalized model-ready text representation")
    tokens: list[str] = Field(default_factory=list, description="Extracted word and punctuation tokens")
    intensifier_count: int = Field(default=0, description="Count of emotional intensifiers")
    negation_count: int = Field(default=0, description="Count of negation modifiers")
    punctuation_features: dict[str, int] = Field(
        default_factory=dict, description="Extracted punctuation and emphasis counts (!, ?, caps)"
    )
    processing_time_ms: float = Field(default=0.0, description="Preprocessing execution duration in ms")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Preprocessing policy and configuration metadata")


class EmotionPrediction(BaseModel):
    """Data contract representing raw emotion classifier outputs before intensity derivation."""

    emotion: SupportedEmotion = Field(description="Primary predicted emotion class")
    confidence: float = Field(ge=0.0, le=1.0, description="Classification confidence score")
    probabilities: dict[SupportedEmotion, float] = Field(
        description="Probability distribution across all 6 supported emotions"
    )
    processing_time_ms: float = Field(default=0.0, description="Model inference duration in milliseconds")


class IntensityComponents(BaseModel):
    """Granular sub-score components contributing to the overall emotional intensity score."""

    text_score: float = Field(ge=0.0, le=1.0, description="Normalized text-level intensity sub-score")
    audio_score: float = Field(ge=0.0, le=1.0, default=0.0, description="Normalized acoustic/prosodic sub-score")
    entropy: float = Field(ge=0.0, description="Normalized prediction distribution entropy")
    margin: float = Field(ge=0.0, le=1.0, description="Top-1 to Top-2 probability margin spread")
    intensifier_count: int = Field(ge=0, description="Count of lexical intensifier words")
    punctuation_score: float = Field(ge=0.0, le=1.0, description="Punctuation & capitalization emphasis sub-score")
    rms_energy: float = Field(ge=0.0, default=0.0, description="Audio Root Mean Square energy")
    zcr_rate: float = Field(ge=0.0, default=0.0, description="Audio Zero Crossing Rate")
    spectral_ratio: float = Field(ge=0.0, default=0.0, description="Audio high-frequency spectral energy ratio")


class IntensityPrediction(BaseModel):
    """Data contract representing emotional intensity calculation results (0.0 to 100.0)."""

    intensity: float = Field(ge=0.0, le=100.0, description="Overall emotional intensity score (0 to 100)")
    level: IntensityLevel = Field(description="Derived qualitative intensity level (low, medium, high)")
    confidence: float = Field(ge=0.0, le=1.0, description="Classifier model confidence (kept strictly separate!)")
    components: IntensityComponents = Field(description="Breakdown of text and acoustic component sub-scores")
    processing_time_ms: float = Field(default=0.0, description="Intensity calculation duration in milliseconds")


class EmotionResult(BaseModel):
    """Structured result returned by Nina to parent applications."""

    text: str = Field(description="Transcribed text transcript")
    emotion: SupportedEmotion = Field(description="Primary predicted emotion class")
    confidence: float = Field(ge=0.0, le=1.0, description="Classification confidence score")
    probabilities: dict[SupportedEmotion, float] = Field(
        description="Probability distribution across all 6 supported emotions"
    )
    intensity: IntensityLevel | float | None = Field(default=None, description="Optional intensity level or float score")
    intensity_level: IntensityLevel | None = Field(default=None, description="Optional qualitative intensity level")
    processing_time_ms: float = Field(default=0.0, description="Total pipeline processing duration in milliseconds")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Pipeline latency and execution metadata")
