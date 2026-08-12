"""Unit tests for CUDA device resolution, actual model parameter device inspection, model preloading & reuse, and timing breakdown metrics."""


from nina.core.device import get_actual_model_device, is_cuda_available, resolve_device
from nina.emotion.classical import ClassicalEmotionClassifier
from nina.emotion.stubs import StubEmotionClassifier
from nina.engine import NinaEmotionEngine
from nina.speech.stubs import StubSpeechToTextEngine


def test_device_resolution_logic() -> None:
    """Verify resolve_device logic for auto, cuda, and cpu settings."""
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto", cuda_available=False) == "cpu"
    assert resolve_device("auto", cuda_available=True) == "cuda"
    assert resolve_device("cuda", cuda_available=False) == "cpu"
    assert resolve_device("cuda", cuda_available=True) == "cuda"


def test_is_cuda_available_returns_bool() -> None:
    """Verify is_cuda_available returns boolean value without crashing."""
    res = is_cuda_available()
    assert isinstance(res, bool)


def test_get_actual_model_device_helper() -> None:
    """Verify get_actual_model_device returns expected string for non-PyTorch models."""
    stub = StubEmotionClassifier()
    assert get_actual_model_device(stub) == "unknown"


def test_engine_preload_and_model_reuse() -> None:
    """Verify NinaEmotionEngine preloads models ONCE and reuses them across multiple turns."""
    engine = NinaEmotionEngine(
        stt_engine=StubSpeechToTextEngine(),
        emotion_classifier=StubEmotionClassifier(),
    )

    init_ms = engine.preload_models()
    assert init_ms >= 0.0
    assert engine._is_preloaded is True

    # Turn 1
    res1 = engine.process_text("I am feeling great!")
    assert res1.metadata["models_reused"] is True
    assert res1.metadata["model_init_ms"] == 0.0
    assert "emotion_inference_ms" in res1.metadata
    assert "total_turn_ms" in res1.metadata

    # Turn 2 (Models reused, zero reloading time)
    res2 = engine.process_text("I am feeling sad today.")
    assert res2.metadata["models_reused"] is True
    assert res2.metadata["model_init_ms"] == 0.0


def test_cpu_fallback_graceful_handling() -> None:
    """Verify NinaEmotionEngine runs cleanly under classical CPU fallback."""
    engine = NinaEmotionEngine(
        stt_engine=StubSpeechToTextEngine(),
        emotion_classifier=ClassicalEmotionClassifier(),
    )

    res = engine.process_text("Classical fallback test")
    assert res.emotion in ["happy", "sadness", "anger", "fear", "love", "surprise"]
    assert res.metadata["classifier"] == "ClassicalEmotionClassifier"
