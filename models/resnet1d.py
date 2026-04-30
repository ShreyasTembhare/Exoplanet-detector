"""
Two-tower ResNet-1D classifier (Phase 4).

Architecture:

::

    global (1, 2048) -> 5 ResBlocks (32 -> 64 -> 128 -> 128 -> 256) -> AdaptiveAvgPool -> 256-d
    local  (1,  256) -> 3 ResBlocks (32 -> 64 -> 128)              -> AdaptiveAvgPool -> 128-d
    centroid scalar  -> normalized via running mean/std              -> 1-d
                                                                      v
                                            concat(256 + 128 + 1) -> Linear(385, 128)
                                                                  -> ReLU -> Dropout
                                                                  -> Linear(128, 2)

Why this is better than the previous "stack global+local as 2 channels":
  * Global and local views have different physical meanings; convolving them
    together as channels makes the network learn meaningless cross-channel
    couplings. AstroNet (Shallue & Vanderburg 2018) and ExoMiner both use
    separate towers.
  * Channel count grows with depth (32 -> 64 -> 128 -> 256), giving the
    representation more capacity in deeper layers, where the previous
    fixed-32-channels variant gave up its representational power.
  * The local view is no longer zero-padded to 2048 (which polluted batch
    norm with 1792 zeros per sample); it gets its own input shape and tower.
  * Centroid normalization is learned (running mean/std maintained on a
    BatchNorm1d layer) so the scalar can't dominate or be ignored.

The legacy ``ResNet1DClassifier`` class is preserved as
:class:`SingleTowerResNet1D` for backward-compatible checkpoint loading.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
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
    "two_tower": True,
}


# ---------------------------------------------------------------------------
# Input prep helpers
# ---------------------------------------------------------------------------


def normalize_view(arr: np.ndarray) -> np.ndarray:
    """Divide by median, subtract 1 so baseline ~ 0."""
    arr = np.asarray(arr, dtype=np.float32)
    med = np.nanmedian(arr)
    if med != 0 and not np.isnan(med):
        arr = arr / med - 1.0
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def make_two_channel(global_view: np.ndarray, local_view: np.ndarray) -> np.ndarray:
    """
    BACKCOMPAT: produces the legacy (2, 2048) shape for the
    :class:`SingleTowerResNet1D` checkpoint format. New training uses
    :func:`make_two_tower_inputs`.
    """
    g = normalize_view(np.asarray(global_view, dtype=np.float32))
    if len(g) != GLOBAL_LEN:
        g = np.interp(np.linspace(0, len(g) - 1, GLOBAL_LEN), np.arange(len(g)), g).astype(np.float32)
    l = normalize_view(np.asarray(local_view, dtype=np.float32))
    out_l = np.zeros(GLOBAL_LEN, dtype=np.float32)
    out_l[: min(len(l), GLOBAL_LEN)] = l[: min(len(l), GLOBAL_LEN)]
    return np.stack([g, out_l], axis=0)


def make_two_tower_inputs(
    global_view: np.ndarray, local_view: np.ndarray,
) -> tuple:
    """Return (global_tensor_shape (1, 2048), local_tensor_shape (1, 256))."""
    g = normalize_view(np.asarray(global_view, dtype=np.float32))
    if len(g) != GLOBAL_LEN:
        g = np.interp(np.linspace(0, len(g) - 1, GLOBAL_LEN), np.arange(len(g)), g).astype(np.float32)
    l = normalize_view(np.asarray(local_view, dtype=np.float32))
    if len(l) != LOCAL_LEN:
        l = np.interp(np.linspace(0, len(l) - 1, LOCAL_LEN), np.arange(len(l)), l).astype(np.float32)
    return g[None, :], l[None, :]


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ResidualBlock1D(nn.Module if TORCH_AVAILABLE else object):
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
        return self.relu(out)


def _make_tower(channels: list, kernel_size: int = 5) -> "nn.Sequential":
    """Build a sequence of ResidualBlock1D + MaxPool1d(2). channels[0] is the
    input channel count (always 1 here)."""
    layers = [
        nn.Conv1d(channels[0], channels[1], kernel_size=7, padding=3),
        nn.BatchNorm1d(channels[1]),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(2),
    ]
    for i in range(1, len(channels) - 1):
        layers.append(ResidualBlock1D(channels[i], channels[i + 1], kernel_size))
        layers.append(nn.MaxPool1d(2))
    layers.append(nn.AdaptiveAvgPool1d(1))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Two-tower model
# ---------------------------------------------------------------------------


class TwoTowerResNet1D(nn.Module if TORCH_AVAILABLE else object):
    """
    Two-tower ResNet-1D with growing channels and a normalized centroid scalar.
    """

    def __init__(
        self,
        global_channels=(1, 32, 64, 128, 128, 256),
        local_channels=(1, 32, 64, 128),
        kernel_size: int = 5,
        dropout: float = 0.3,
        use_centroid: bool = True,
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required")
        super().__init__()
        self.use_centroid = use_centroid

        self.global_tower = _make_tower(list(global_channels), kernel_size)
        self.local_tower = _make_tower(list(local_channels), kernel_size)

        self.centroid_norm = nn.BatchNorm1d(1) if use_centroid else None

        feat_dim = global_channels[-1] + local_channels[-1] + (1 if use_centroid else 0)
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, global_x, local_x, centroid_offset=None):
        """
        global_x: (B, 1, 2048)
        local_x : (B, 1, 256)
        centroid_offset: (B,) or None
        """
        g_feat = self.global_tower(global_x).squeeze(-1)  # (B, 256)
        l_feat = self.local_tower(local_x).squeeze(-1)    # (B, 128)
        feats = [g_feat, l_feat]
        if self.use_centroid:
            if centroid_offset is None:
                centroid_offset = torch.zeros(global_x.size(0), device=g_feat.device, dtype=g_feat.dtype)
            co = centroid_offset.float().unsqueeze(1).unsqueeze(2)  # (B, 1, 1)
            co_norm = self.centroid_norm(co).squeeze(-1)
            feats.append(co_norm)
        return self.head(torch.cat(feats, dim=1))


# ---------------------------------------------------------------------------
# Backward-compatible single-tower model (legacy checkpoints)
# ---------------------------------------------------------------------------


class ResNet1DClassifier(nn.Module if TORCH_AVAILABLE else object):
    """
    Legacy single-tower architecture. Kept so existing checkpoints
    (``models/checkpoints/resnet1d.pt``) keep loading. New runs prefer
    :class:`TwoTowerResNet1D`.
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
            raise RuntimeError("PyTorch required")
        super().__init__()
        self.use_centroid = use_centroid
        layers = [
            nn.Sequential(
                nn.Conv1d(in_channels, base_channels, kernel_size=7, padding=3),
                nn.BatchNorm1d(base_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
            )
        ]
        ch = base_channels
        for _ in range(num_blocks):
            layers.append(ResidualBlock1D(ch, ch, kernel_size))
            layers.append(nn.MaxPool1d(2))
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        feat_size = ch + (1 if use_centroid else 0)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_size, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x, centroid_offset=None):
        feat = self.pool(self.features(x)).flatten(1)
        if self.use_centroid:
            if centroid_offset is None:
                centroid_offset = torch.zeros(x.size(0), device=feat.device, dtype=feat.dtype).unsqueeze(1)
            else:
                centroid_offset = centroid_offset.to(feat.device).float().unsqueeze(1)
            feat = torch.cat([feat, centroid_offset], dim=1)
        return self.classifier(feat)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Checkpoint loading (legacy + new)
# ---------------------------------------------------------------------------


def _map_checkpoint_keys(state_dict: dict, model) -> dict:
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
    logger.info("Checkpoint key mapping: %d/%d keys matched", len(mapped), len(model_keys))
    return mapped


def load_checkpoint(
    checkpoint_path: str,
    model=None,
    device=None,
    strict: bool = False,
):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required")
    if device is None:
        from device_util import get_device
        device = get_device()
    if model is None:
        # Detect arch from the checkpoint itself.
        ckpt_peek = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        sd = ckpt_peek.get("model", ckpt_peek) if isinstance(ckpt_peek, dict) else ckpt_peek
        if isinstance(sd, dict) and any(k.startswith("global_tower") for k in sd):
            model = TwoTowerResNet1D(use_centroid=True)
        else:
            model = ResNet1DClassifier(use_centroid=True)

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
    mapped = _map_checkpoint_keys(state_dict, model)
    model.load_state_dict(mapped, strict=strict)
    model.to(device)
    model.eval()
    logger.info(f"Loaded checkpoint from {path} ({count_parameters(model):,} params)")
    return model
