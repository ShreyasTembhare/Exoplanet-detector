"""
Centralized hardware detection for the TESS exoplanet pipeline.
Auto-selects CUDA > MPS > CPU and provides AMP / JAX backend configs.
"""

import logging
import platform

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def get_device() -> "torch.device":
    """Return the best available torch device: CUDA > MPS > CPU."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required. Install with: pip install torch")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_amp_config(device: "torch.device") -> dict:
    """
    Return AMP configuration dict for the given device.

    Keys:
        enabled     -- whether autocast is supported
        device_type -- string to pass to torch.autocast()
        dtype       -- torch dtype for mixed precision
        use_scaler  -- whether GradScaler is supported (CUDA only)
    """
    if not TORCH_AVAILABLE:
        return {"enabled": False, "device_type": "cpu", "dtype": None, "use_scaler": False}

    if device.type == "cuda":
        return {
            "enabled": True,
            "device_type": "cuda",
            "dtype": torch.float16,
            "use_scaler": True,
        }
    if device.type == "mps":
        return {
            "enabled": True,
            "device_type": "mps",
            "dtype": torch.float16,
            "use_scaler": False,
        }
    return {"enabled": False, "device_type": "cpu", "dtype": None, "use_scaler": False}


def get_jax_backend() -> str:
    """Return 'gpu' if JAX sees a GPU, otherwise 'cpu'."""
    try:
        import jax
        devices = jax.devices()
        for d in devices:
            if d.platform in ("gpu", "cuda"):
                return "gpu"
    except Exception:
        pass
    return "cpu"


def print_hw_report():
    """Log a one-time hardware summary at startup."""
    lines = [f"OS: {platform.system()} {platform.machine()}"]

    if TORCH_AVAILABLE:
        device = get_device()
        lines.append(f"PyTorch: {torch.__version__}  Device: {device}")
        amp_cfg = get_amp_config(device)
        amp_str = "enabled" if amp_cfg["enabled"] else "disabled"
        scaler_str = "+ GradScaler" if amp_cfg["use_scaler"] else "(no scaler)"
        lines.append(f"AMP: {amp_str} {scaler_str}")
        if device.type == "cuda":
            lines.append(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        lines.append("PyTorch: not installed")

    jax_backend = get_jax_backend()
    lines.append(f"JAX BLS backend: {jax_backend}")

    header = "--- Hardware Report ---"
    logger.info(header)
    for line in lines:
        logger.info("  %s", line)
    logger.info("-" * len(header))
