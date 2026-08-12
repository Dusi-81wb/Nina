"""Acoustic and prosodic feature extraction engine for Nina Phase 5 Intensity Calculation."""

import wave
from pathlib import Path

import numpy as np

from nina.core.exceptions import AudioError
from nina.core.logging import logger


class NinaAudioFeatures:
    """Extracts normalized acoustic and prosodic features (RMS energy, crest factor, ZCR, spectral ratio)."""

    def __init__(self, target_sample_rate: int = 16000) -> None:
        self.target_sample_rate = target_sample_rate

    @staticmethod
    def extract_from_array(audio_array: np.ndarray, sample_rate: int = 16000) -> dict[str, float]:
        """Extract acoustic features from a 1D float/int PCM numpy array.

        Args:
            audio_array: 1D numpy array of audio samples.
            sample_rate: Audio sampling frequency in Hz.

        Returns:
            dict[str, float]: Extracted acoustic feature values.
        """
        if audio_array is None or len(audio_array) == 0:
            return {
                "rms_energy": 0.0,
                "crest_factor": 0.0,
                "zcr_rate": 0.0,
                "spectral_ratio": 0.0,
                "has_audio": 0.0,
            }

        # Convert int16/int32 to float32 normalized [-1.0, 1.0]
        if np.issubdtype(audio_array.dtype, np.integer):
            max_val = float(np.iinfo(audio_array.dtype).max)
            audio = audio_array.astype(np.float32) / max_val
        else:
            audio = audio_array.astype(np.float32)

        # Flatten if 2D (mono selection)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        num_samples = len(audio)
        if num_samples == 0:
            return {
                "rms_energy": 0.0,
                "crest_factor": 0.0,
                "zcr_rate": 0.0,
                "spectral_ratio": 0.0,
                "has_audio": 0.0,
            }

        # 1. RMS Energy
        rms = float(np.sqrt(np.mean(audio**2)))

        # 2. Crest Factor (Peak to RMS ratio)
        peak = float(np.max(np.abs(audio)))
        crest = float(peak / (rms + 1e-7))

        # 3. Zero Crossing Rate (ZCR)
        zero_crossings = np.diff(np.signbit(audio))
        zcr = float(np.sum(zero_crossings) / max(1, num_samples - 1))

        # 4. High-Frequency Spectral Energy Ratio (Energy > 1000 Hz / Total Energy)
        try:
            fft_vals = np.abs(np.fft.rfft(audio))
            fft_freqs = np.fft.rfftfreq(num_samples, d=1.0 / sample_rate)
            total_spectral_power = float(np.sum(fft_vals**2)) + 1e-7
            hf_mask = fft_freqs >= 1000.0
            hf_spectral_power = float(np.sum(fft_vals[hf_mask] ** 2))
            spectral_ratio = float(hf_spectral_power / total_spectral_power)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Spectral analysis failed: {e!s}. Defaulting ratio to 0.0.")
            spectral_ratio = 0.0

        return {
            "rms_energy": round(rms, 6),
            "crest_factor": round(crest, 4),
            "zcr_rate": round(zcr, 6),
            "spectral_ratio": round(spectral_ratio, 4),
            "has_audio": 1.0,
        }

    def extract_from_file(self, file_path: Path | str) -> dict[str, float]:
        """Read a WAV audio file and extract acoustic features.

        Args:
            file_path: Path to WAV audio file.

        Returns:
            dict[str, float]: Extracted feature metrics.
        """
        p = Path(file_path)
        if not p.exists():
            raise AudioError(f"Audio file for feature extraction not found: {p}")

        try:
            with wave.open(str(p), "rb") as wf:
                sr = wf.getframerate()
                n_frames = wf.getnframes()
                raw_bytes = wf.readframes(n_frames)
                sampwidth = wf.getsampwidth()

                if sampwidth == 2:
                    dtype = np.int16
                elif sampwidth == 4:
                    dtype = np.int32
                else:
                    dtype = np.uint8

                audio_arr = np.frombuffer(raw_bytes, dtype=dtype)
                if wf.getnchannels() > 1:
                    audio_arr = audio_arr.reshape(-1, wf.getnchannels()).mean(axis=1)

                return self.extract_from_array(audio_arr, sample_rate=sr)

        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to read WAV file '{p}' for acoustic extraction: {e!s}")
            return {
                "rms_energy": 0.0,
                "crest_factor": 0.0,
                "zcr_rate": 0.0,
                "spectral_ratio": 0.0,
                "has_audio": 0.0,
            }
