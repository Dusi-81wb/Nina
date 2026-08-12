"""Abstract interface for Audio Capture Components."""

from abc import ABC, abstractmethod

from nina.api.schemas import AudioInput


class AudioCaptureInterface(ABC):
    """Abstract interface defining the contract for audio hardware/file input readers."""

    @abstractmethod
    def capture(self, duration_seconds: float) -> AudioInput:
        """Capture audio signal and return standardized AudioInput metadata payload.

        Args:
            duration_seconds: Recording duration in seconds.

        Returns:
            AudioInput: Populated data contract.

        Raises:
            AudioError: If input stream fails or yields corrupted frames.
        """
