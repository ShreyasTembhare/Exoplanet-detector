"""
ResNet-1D classifier: two-channel input (global 2048 + local 256 padded to 2048)
and centroid offset scalar. Binary: planet vs false positive.
Target: <1M parameters, 10-15 conv layers.
Supports loading public pretrained checkpoints with key-mapping adapter.
"""

import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None  # type: ignore

logger = logging.getLogger(__name__)

GLOBAL_LEN = 2048
LOCAL_LEN = 256

INPUT_CONTRACT = {
    "global_len": GLOBAL_LEN,
    "local_len": LOCAL_LEN,
    "channels": 2,
    "centroid": True,
    "normalization": "median_divide",
}


def normalize_view(arr: np.ndarray) -> np.ndarray:
    """Canonical normalization: divide by median, subtract 1 (so baseline ~ 0)."""
    arr = np.asarray(arr, dtype=np.float32)
    med = np.nanmedian(arr)
    if med != 0 and not np.isnan(med):
        arr = arr / med - 1.0
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def _pad_local_to_global(local: np.ndarray) -> np.ndarray:
    """Pad local view (256) to 2048 (zero-pad at end)."""
    out = np.zeros(GLOBAL_LEN, dtype=np.float32)
    n = min(len(local), GLOBAL_LEN)
    out[:n] = local[:n]
    return out


def make_two_channel(global_view: np.ndarray, local_view: np.ndarray) -> np.ndarray:
    """
    Stack global and padded local into (2, 2048). Shape suitable for 1D conv.
    Applies canonical normalization to both views.
    """
    global_arr = normalize_view(np.asarray(global_view, dtype=np.float32))
    if len(global_arr) != GLOBAL_LEN:
        global_arr = np.interp(
            np.linspace(0, len(global_arr) - 1, GLOBAL_LEN),
            np.arange(len(global_arr)),
            global_arr,
        ).astype(np.float32)
    local_arr = normalize_view(np.asarray(local_view, dtype=np.float32))
    local_padded = _pad_local_to_global(local_arr)
    return np.stack([global_arr, local_padded], axis=0)  # (2, 2048)


class ResidualBlock1D(nn.Module if TORCH_AVAILABLE else object):
    """Residual block: two 1D convs with BatchNorm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for ResNet1D")
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out


class ResNet1DClassifier(nn.Module if TORCH_AVAILABLE else object):
    """
    ResNet-1D for exoplanet classification.
    Input: (batch, 2, 2048) (global + local channels).
    Centroid offset (scalar) is concatenated to pooled features before the dense head.
    Output: binary logits (planet vs false positive).
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 32,
        num_blocks: int = 4,
        kernel_size: int = 5,
        use_centroid: bool = True,
        dropout: float = 0.3,
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for ResNet1DClassifier")
        super().__init__()
        self.use_centroid = use_centroid
        layers = []
        ch = base_channels
        layers.append(
            nn.Sequential(
                nn.Conv1d(in_channels, ch, kernel_size=7, padding=3),
                nn.BatchNorm1d(ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
            )
        )
        # ~4 blocks of 2 convs each -> ~10 conv layers; add more blocks for 15
        for _ in range(num_blocks):
            layers.append(ResidualBlock1D(ch, ch, kernel_size))
            layers.append(nn.MaxPool1d(2))
        self.features = nn.Sequential(*layers)
        # After pooling: 2048 -> 1024 -> 512 -> 256 -> 128 -> 64 -> 32 (6 pools)
        # With 4 blocks + initial pool: 2048/2 = 1024, then 4 more /2 -> 64
        self.pool = nn.AdaptiveAvgPool1d(1)
        feat_size = ch
        if use_centroid:
            feat_size += 1  # centroid offset scalar
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_size, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 2),  # binary
        )

    def forward(self, x, centroid_offset: Optional[torch.Tensor] = None):
        """
        x: (B, 2, 2048), centroid_offset: (B,) or None.
        """
        feat = self.features(x)  # (B, C, L)
        feat = self.pool(feat)  # (B, C, 1)
        feat = feat.flatten(1)  # (B, C)
        if self.use_centroid:
            if centroid_offset is None:
                centroid_offset = torch.zeros(x.size(0), device=feat.device, dtype=feat.dtype).unsqueeze(1)
            else:
                centroid_offset = centroid_offset.to(feat.device).float().unsqueeze(1)
            feat = torch.cat([feat, centroid_offset], dim=1)
        return self.classifier(feat)


def count_parameters(model: "ResNet1DClassifier") -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _map_checkpoint_keys(state_dict: dict, model: "ResNet1DClassifier") -> dict:
    """
    Attempt to map external checkpoint keys to our model's keys.
    Handles common naming differences (e.g. 'backbone.' prefix, 'fc.' vs 'classifier.').
    Returns mapped state_dict (best-effort; loads what matches).
    """
    model_keys = set(model.state_dict().keys())
    if set(state_dict.keys()) == model_keys:
        return state_dict

    mapped = {}
    replacements = [
        ("backbone.", "features."),
        ("fc.", "classifier."),
        ("head.", "classifier."),
        ("stem.", "features.0."),
    ]
    for k, v in state_dict.items():
        new_k = k
        for old_prefix, new_prefix in replacements:
            if new_k.startswith(old_prefix):
                new_k = new_prefix + new_k[len(old_prefix):]
        if new_k in model_keys:
            mapped[new_k] = v
        elif k in model_keys:
            mapped[k] = v

    loaded = len(mapped)
    total = len(model_keys)
    logger.info(f"Checkpoint key mapping: {loaded}/{total} keys matched")
    return mapped


def load_checkpoint(
    checkpoint_path: str,
    model: Optional["ResNet1DClassifier"] = None,
    device: Optional["torch.device"] = None,
    strict: bool = False,
) -> "ResNet1DClassifier":
    """
    Load a checkpoint (local or pretrained) into a ResNet1DClassifier.
    Handles our own format (dict with 'model' key) and raw state_dicts.
    Falls back to partial loading if keys don't match exactly.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = ResNet1DClassifier(use_centroid=True)

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and any(k.startswith("features") or k.startswith("classifier") for k in ckpt):
        state_dict = ckpt
    else:
        state_dict = ckpt

    mapped = _map_checkpoint_keys(state_dict, model)
    model.load_state_dict(mapped, strict=strict)
    model.to(device)
    model.eval()
    logger.info(f"Loaded checkpoint from {path} ({count_parameters(model):,} params)")
    return model
