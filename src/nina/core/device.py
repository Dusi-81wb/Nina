"""Device detection utility for PyTorch CUDA/CPU hardware acceleration."""

import platform
import sys
from dataclasses import dataclass
from typing import Any

from nina.core.logging import logger


@dataclass
class DeviceInfo:
    """Dataclass holding system environment and hardware accelerator details."""

    python_version: str
    os_info: str
    cpu_architecture: str
    pytorch_version: str
    cuda_available: bool
    device_name: str
    vram_gb: float
    selected_device: str


def is_cuda_available() -> bool:
    """Check if PyTorch CUDA acceleration is available."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def get_device_info(requested_device: str = "auto") -> DeviceInfo:
    """Inspect system environment and detect CUDA / CPU acceleration capabilities.

    Args:
        requested_device: User-requested device ("auto", "cuda", "cpu", "mps").

    Returns:
        DeviceInfo: Dataclass populated with diagnostic metrics.
    """
    python_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    os_info = f"{platform.system()} {platform.release()} ({platform.machine()})"
    cpu_arch = platform.processor() or platform.machine()

    pytorch_ver = "Not Installed"
    cuda_available = False
    device_name = "CPU"
    vram_gb = 0.0

    try:
        import torch

        pytorch_ver = torch.__version__
        cuda_available = torch.cuda.is_available()

        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            total_vram_bytes = torch.cuda.get_device_properties(0).total_memory
            vram_gb = round(total_vram_bytes / (1024**3), 2)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_name = "Apple Silicon MPS"
            cuda_available = False
    except ImportError:
        logger.warning("PyTorch is not installed in the active environment.")

    # Resolve target device
    selected_device = resolve_device(requested_device, cuda_available=cuda_available)

    return DeviceInfo(
        python_version=python_ver,
        os_info=os_info,
        cpu_architecture=cpu_arch,
        pytorch_version=pytorch_ver,
        cuda_available=cuda_available,
        device_name=device_name,
        vram_gb=vram_gb,
        selected_device=selected_device,
    )


def resolve_device(requested_device: str = "auto", cuda_available: bool | None = None) -> str:
    """Determine final compute device string based on hardware availability.

    Args:
        requested_device: Requested string ("auto", "cuda", "cpu", "mps").
        cuda_available: Optional boolean override. If None, checks `is_cuda_available()`.

    Returns:
        str: Resolved device name ("cuda" or "cpu" or "mps").
    """
    if cuda_available is None:
        cuda_available = is_cuda_available()

    requested = requested_device.lower()

    if requested == "auto":
        if cuda_available:
            return "cuda"
        return "cpu"

    if requested == "cuda":
        if not cuda_available:
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        return "cuda"

    return requested


def get_actual_model_device(model: Any) -> str:
    """Inspect actual PyTorch parameter device of a model object.

    Args:
        model: PyTorch nn.Module or wrapper object.

    Returns:
        str: Device string representation (e.g. 'cuda:0' or 'cpu').
    """
    try:
        model_obj = getattr(model, "model", model)
        param = next(model_obj.parameters(), None)
        if param is not None:
            return str(param.device)
    except Exception:  # noqa: BLE001, S110
        pass
    return "unknown"
