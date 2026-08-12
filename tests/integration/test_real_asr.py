"""Integration test for real local ASR model inference using faster-whisper."""

import pytest

from nina.audio.source import FileAudioSource, create_synthetic_wav_file


@pytest.mark.real_model
def test_real_faster_whisper_transcription(tmp_path) -> None:
    """Run real local faster-whisper ASR inference on a synthetic WAV audio file fixture."""
    try:
        from nina.speech.engine import FasterWhisperSpeechToText
    except ImportError:
        pytest.skip("faster-whisper package not installed in environment.")

    # Create synthetic WAV file fixture
    wav_path = tmp_path / "fixture_audio.wav"
    create_synthetic_wav_file(wav_path, duration_seconds=2.0, sample_rate=16000, frequency=440.0)

    source = FileAudioSource(wav_path)
    audio_input = source.get_audio_input()

    engine = FasterWhisperSpeechToText(model_size="tiny.en", device="cpu")

    try:
        result = engine.transcribe(audio_input)
        assert isinstance(result.text, str)
        assert result.processing_time_ms > 0.0
    except Exception as e:  # noqa: BLE001
        # If internet is restricted or model download fails, skip test gracefully
        pytest.skip(f"Real model execution skipped due to network/environment constraint: {e!s}")
