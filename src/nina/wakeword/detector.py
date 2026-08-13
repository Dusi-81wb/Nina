"""Wake word detection subsystem."""
import numpy as np


class WakeWordDetector:
    """Detects wake words in audio streams (Stub implementation)."""

    def __init__(self, wake_word: str = "nina", sensitivity: float = 0.5):
        self.wake_word = wake_word.lower()
        self.sensitivity = sensitivity

    def detect_wake_word(self, audio_buffer: np.ndarray, sample_rate: int = 16000) -> bool:
        """
        Detect if the wake word is present in the audio buffer.

        Note: This is a stub implementation. In a production environment,
        this would use a specialized engine like Porcupine, PocketSphinx,
        or an energy-based acoustic model.
        For now, we simulate detection based on a simple energy threshold
        and random probability to allow testing the pipeline.
        """
        if len(audio_buffer) == 0:
            return False

        # Calculate RMS energy
        if np.issubdtype(audio_buffer.dtype, np.integer):
            max_val = float(np.iinfo(audio_buffer.dtype).max)
            samples = audio_buffer.astype(np.float32) / max_val
        else:
            samples = audio_buffer.astype(np.float32)

        energy = float(np.sqrt(np.mean(samples**2)))

        # Simple heuristic: If there is significant energy, we "detected" the wake word
        # In real life, this would actually decode the phonemes or use a wake word model
        if energy > 0.05:
            # Add a slight random element to simulate the "Wake Word" firing
            # only sometimes when there is noise, just for testing purposes.
            # Real implementation would be deterministic based on acoustic features.
            import random
            if random.random() < self.sensitivity:
                return True

        return False
