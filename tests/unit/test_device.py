"""Unit tests for device detection and CUDA/CPU fallback logic."""

from nina.core.device import get_device_info, resolve_device


def test_device_info_structure() -> None:
    """Verify get_device_info returns populated DeviceInfo dataclass."""
    info = get_device_info("auto")
    assert isinstance(info.python_version, str)
    assert isinstance(info.os_info, str)
    assert isinstance(info.cuda_available, bool)
    assert info.selected_device in ["cuda", "cpu", "mps"]


def test_resolve_device_fallback() -> None:
    """Verify requested CUDA falls back to CPU when CUDA is unavailable."""
    assert resolve_device("auto", cuda_available=False) == "cpu"
    assert resolve_device("auto", cuda_available=True) == "cuda"
    assert resolve_device("cuda", cuda_available=False) == "cpu"
    assert resolve_device("cuda", cuda_available=True) == "cuda"
    assert resolve_device("cpu", cuda_available=True) == "cpu"
