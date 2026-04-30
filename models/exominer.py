"""
ExoMiner++-style multi-input vetter (Phase 6).

References:
  * Valizadegan et al. 2022 (ExoMiner)         -- arXiv:2111.10009
  * Valizadegan et al. 2025 (ExoMiner++)       -- arXiv:2502.09790

Architecture (late-fusion):

::

    global_view   (1, 2048) -> CNN tower -> 256-d ┐
    local_view    (1,  256) -> CNN tower -> 128-d ┤
    local_center  (1,  128) -> CNN tower -> 128-d ┤
    periodogram   (1, 1024) -> CNN tower -> 128-d ┤   concat -> MLP(704, 256, 2)
    diff_image    (1, 11, 11) -> 2D CNN  ->  64-d ┤
    scalar feats  (~25 numbers) -> MLP    ->  64-d ┘

The scalar branch consumes the Phase 3 + Phase 5 vetting outputs:
depth, duration, SDE_BLS, SDE_TLS, odd-even ratio, secondary
significance, V/U, n_transits, T_mag, Teff, log g, rho_star,
crowdsap, n_gaia_neighbors, etc.

This is meaningfully more capacity (~5 M params) than the two-tower
ResNet-1D (~1 M), so it needs more data: train against TOI labels +
injection-recovery + Kepler transfer learning. Phase 8 SSL pretraining
helps when labels are limited.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False
    nn = None  # type: ignore


# ---------------------------------------------------------------------------
# Building blocks (reuse style from models/resnet1d)
# ---------------------------------------------------------------------------


def _resblock_1d(in_ch, out_ch, kernel=5):
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv1d(out_ch, out_ch, kernel, padding=kernel // 2),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
    )


class _Conv1DTower(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, channels: List[int], kernel: int = 5):
        super().__init__()
        layers = [
            nn.Conv1d(1, channels[0], 7, padding=3),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        ]
        for i in range(len(channels) - 1):
            layers.append(_resblock_1d(channels[i], channels[i + 1], kernel))
            layers.append(nn.MaxPool1d(2))
        layers.append(nn.AdaptiveAvgPool1d(1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class _DiffImageTower(nn.Module if TORCH_AVAILABLE else object):
    """Tiny 2D CNN for the difference image branch (default 11x11)."""

    def __init__(self, in_size: int = 11, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(32, out_dim)

    def forward(self, x):
        h = self.net(x).flatten(1)
        return self.proj(h)


class _ScalarTower(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, n_features: int, out_dim: int = 64):
        super().__init__()
        self.bn = nn.BatchNorm1d(n_features)
        self.net = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(64, out_dim), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(self.bn(x))


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


SCALAR_FEATURE_NAMES = [
    "depth", "duration_hours", "n_transits", "sde_bls", "sde_tls",
    "odd_even_ratio", "secondary_significance", "v_shape_score",
    "centroid_offset_px", "diff_image_peak_px", "gaia_neighbors_within_30arcsec",
    "tmag", "teff", "logg", "radius", "crowdsap", "cadence_minutes",
    "rotation_period_d", "cdpp_ppm", "period_days", "best_phase",
    "depth_to_noise", "duration_to_period", "n_sectors",
]


class ExoMinerVetter(nn.Module if TORCH_AVAILABLE else object):
    """Late-fusion multi-input vetter."""

    def __init__(
        self,
        scalar_dim: int = len(SCALAR_FEATURE_NAMES),
        diff_image_size: int = 11,
        global_channels=(32, 64, 128, 256),
        local_channels=(32, 64, 128),
        local_centered_channels=(32, 64, 128),
        periodogram_channels=(32, 64, 128),
        dropout: float = 0.3,
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required")
        super().__init__()
        self.global_tower = _Conv1DTower(list(global_channels))
        self.local_tower = _Conv1DTower(list(local_channels))
        self.local_centered_tower = _Conv1DTower(list(local_centered_channels))
        self.periodogram_tower = _Conv1DTower(list(periodogram_channels))
        self.diff_tower = _DiffImageTower(in_size=diff_image_size, out_dim=64)
        self.scalar_tower = _ScalarTower(n_features=scalar_dim, out_dim=64)

        feat_dim = (
            global_channels[-1] + local_channels[-1] +
            local_centered_channels[-1] + periodogram_channels[-1] +
            64 + 64
        )
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, global_x, local_x, local_centered_x,
                periodogram_x, diff_image_x, scalars):
        g = self.global_tower(global_x)
        l = self.local_tower(local_x)
        lc = self.local_centered_tower(local_centered_x)
        pg = self.periodogram_tower(periodogram_x)
        di = self.diff_tower(diff_image_x)
        sc = self.scalar_tower(scalars)
        return self.head(torch.cat([g, l, lc, pg, di, sc], dim=1))


# ---------------------------------------------------------------------------
# Input prep
# ---------------------------------------------------------------------------


def normalize_view(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    med = np.nanmedian(arr)
    if med != 0 and not np.isnan(med):
        arr = arr / med - 1.0
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _resample(arr: np.ndarray, target_len: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if len(arr) == target_len:
        return arr
    if len(arr) == 0:
        return np.zeros(target_len, dtype=np.float32)
    return np.interp(np.linspace(0, len(arr) - 1, target_len),
                     np.arange(len(arr)), arr).astype(np.float32)


def make_exominer_inputs(
    global_view: np.ndarray,
    local_view: np.ndarray,
    local_centered: Optional[np.ndarray] = None,
    periodogram: Optional[np.ndarray] = None,
    diff_image: Optional[np.ndarray] = None,
    scalars_dict: Optional[dict] = None,
    diff_image_size: int = 11,
):
    """Return (global, local, local_centered, periodogram, diff_image, scalars)
    as ``(C, L)`` / ``(C, H, W)`` numpy arrays (channels-first).
    """
    g = normalize_view(_resample(global_view, 2048))[None, :]
    l = normalize_view(_resample(local_view, 256))[None, :]
    lc = normalize_view(_resample(local_centered if local_centered is not None else local_view, 128))[None, :]
    pg_raw = periodogram if periodogram is not None else np.zeros(1024, dtype=np.float32)
    pg = _resample(pg_raw, 1024)[None, :]
    if diff_image is None:
        di = np.zeros((1, diff_image_size, diff_image_size), dtype=np.float32)
    else:
        img = np.asarray(diff_image, dtype=np.float32)
        if img.ndim == 2:
            img = img[None, :, :]
        di = img
    scalars = np.array(
        [float((scalars_dict or {}).get(name, 0.0) or 0.0) for name in SCALAR_FEATURE_NAMES],
        dtype=np.float32,
    )
    return g, l, lc, pg, di, scalars


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
