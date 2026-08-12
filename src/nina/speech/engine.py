"""Concrete implementation of Speech-to-Text engine using faster-whisper (CTranslate2)."""

import time
from pathlib import Path
from typing import Any

from nina.api.schemas import AudioInput, SpeechResult
from nina.core.config import NinaSettings, get_settings
from nina.core.exceptions import SpeechToTextError
from nina.core.logging import logger
from nina.speech.interface import SpeechToTextEngine


class FasterWhisperSpeechToText(SpeechToTextEngine):
    """Local-first Speech-to-Text engine using faster-whisper (CTranslate2)."""

    def __init__(
        self,
        model_size: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
        download_root: str | Path | None = None,
        settings: NinaSettings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.model_size = model_size or self.settings.stt_model_size
        self.download_root = Path(download_root or self.settings.model_dir)

        # ASR Speech-to-Text compute configuration.
        # Default to "cpu" int8 for CTranslate2 ASR to avoid CUDA 12 cuBLAS DLL missing errors on Windows,
        # while PyTorch DistilBERT emotion inference runs natively on CUDA GPU.
        self.requested_device = device or "cpu"
        self.resolved_device = self.requested_device
        self.compute_type = compute_type or ("int8" if self.resolved_device == "cpu" else "float16")

        if self.resolved_device == "cpu" and self.compute_type == "float16":
            self.compute_type = "int8"
            logger.info("Device is CPU. Automatically adjusted compute_type from 'float16' to 'int8'.")

        self._model: Any = None
        self._is_loaded = False

    def load_model(self) -> None:
        """Explicitly load faster-whisper CTranslate2 model weights into memory with automatic CPU fallback.

        Raises:
            SpeechToTextError: If model initialization or weight download fails.
        """
        if self._is_loaded and self._model is not None:
            return

        logger.info(
            f"Loading ASR model '{self.model_size}' on device '{self.resolved_device}' ({self.compute_type})..."
        )

        try:
            from faster_whisper import WhisperModel

            self.download_root.mkdir(parents=True, exist_ok=True)

            try:
                self._model = WhisperModel(
                    model_size_or_path=self.model_size,
                    device=self.resolved_device,
                    compute_type=self.compute_type,
                    download_root=str(self.download_root.resolve()),
                )
                self._is_loaded = True
                logger.info(f"ASR model '{self.model_size}' loaded successfully on device '{self.resolved_device}'.")
            except Exception as cuda_err:
                if self.resolved_device != "cpu":
                    logger.warning(
                        f"Failed to initialize faster-whisper on '{self.resolved_device}': {cuda_err!s}. "
                        "Falling back to CPU (int8)."
                    )
                    self.resolved_device = "cpu"
                    self.compute_type = "int8"
                    self._model = WhisperModel(
                        model_size_or_path=self.model_size,
                        device="cpu",
                        compute_type="int8",
                        download_root=str(self.download_root.resolve()),
                    )
                    self._is_loaded = True
                    logger.info(f"ASR model '{self.model_size}' loaded successfully on CPU fallback.")
                else:
                    raise

        except Exception as e:
            self._is_loaded = False
            raise SpeechToTextError(
                f"Failed to load faster-whisper ASR model '{self.model_size}': {e!s}",
                details={"model_size": self.model_size, "device": self.resolved_device},
            ) from e

    def is_ready(self) -> bool:
        """Check if model engine is loaded and ready for inference."""
        return self._is_loaded and self._model is not None

    def transcribe(self, audio: Any, language: str | None = None) -> SpeechResult:
        """Convert input AudioInput or AudioSource into clean transcribed text and metadata.

        Args:
            audio: AudioInput, AudioSource, Path, or str.
            language: Optional ISO language code (e.g. 'en'). Defaults to auto-detection.

        Returns:
            SpeechResult: Transcribed text string and latency metrics.

        Raises:
            SpeechToTextError: If model transcription fails.
        """
        if not self.is_ready():
            self.load_model()

        start_time = time.perf_counter()

        file_path_str = None
        duration_sec = 1.0

        if isinstance(audio, (str, Path)):
            file_path_str = str(Path(audio).resolve())
        elif hasattr(audio, "file_path"):
            file_path_str = str(Path(audio.file_path).resolve())
            if hasattr(audio, "get_audio_input"):
                try:
                    duration_sec = audio.get_audio_input().duration_seconds
                except Exception:  # noqa: BLE001, S110
                    pass
        elif isinstance(audio, AudioInput):
            file_path_str = audio.metadata.get("file_path")
            duration_sec = audio.duration_seconds
        elif hasattr(audio, "get_audio_input"):
            try:
                inp = audio.get_audio_input()
                file_path_str = inp.metadata.get("file_path")
                duration_sec = inp.duration_seconds
            except Exception:  # noqa: BLE001, S110
                pass

        if not file_path_str or not Path(file_path_str).exists():
            raise SpeechToTextError(
                "Audio file path is missing or inaccessible in AudioInput metadata.",
                details={"source": str(audio)},
            )

        try:
            try:
                segments, info = self._model.transcribe(
                    file_path_str,
                    language=language,
                    beam_size=5,
                    vad_filter=True,
                )
            except Exception as err:
                # If CUDA execution fails at inference time (e.g. missing cublas dll), fallback to CPU ASR
                if self.resolved_device != "cpu":
                    logger.warning(
                        f"Inference error on device '{self.resolved_device}': {err!s}. Reloading ASR model on CPU fallback."
                    )
                    from faster_whisper import WhisperModel
                    self.resolved_device = "cpu"
                    self.compute_type = "int8"
                    self._model = WhisperModel(
                        model_size_or_path=self.model_size,
                        device="cpu",
                        compute_type="int8",
                        download_root=str(self.download_root.resolve()),
                    )
                    segments, info = self._model.transcribe(
                        file_path_str,
                        language=language,
                        beam_size=5,
                        vad_filter=True,
                    )
                else:
                    raise

            # Combine output text segments
            transcript_text = " ".join([segment.text.strip() for segment in segments]).strip()

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            rtf = (elapsed_ms / 1000.0) / max(duration_sec, 0.1)

            logger.info(
                f"Transcribed audio ({duration_sec:.2f}s) in {elapsed_ms:.2f}ms (RTF: {rtf:.2f})"
            )

            return SpeechResult(
                text=transcript_text,
                language=info.language if hasattr(info, "language") else "en",
                confidence=0.95,
                processing_time_ms=round(elapsed_ms, 2),
            )

        except SpeechToTextError:
            raise
        except Exception as e:
            raise SpeechToTextError(
                f"Transcription error during inference: {e!s}",
                details={"file_path": file_path_str},
            ) from e

    def get_model_info(self) -> dict[str, Any]:
        """Return diagnostic dictionary of model engine parameters."""
        return {
            "model_size": self.model_size,
            "requested_device": self.requested_device,
            "resolved_device": self.resolved_device,
            "compute_type": self.compute_type,
            "download_root": str(self.download_root),
            "is_loaded": self.is_ready(),
        }
