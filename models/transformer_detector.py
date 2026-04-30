"""
Transformer detector for raw TESS light curves (Phase 7).

Reference: arXiv:2502.07542 (period-agnostic transit detection).

Key idea: instead of phase-folding on a candidate BLS period (which assumes
periodicity and good period recovery), feed the raw, detrended LC directly
to a sequence model that predicts a per-cadence transit probability. This
catches:

  * Single-transit events from long-period planets that BLS can't resolve.
  * Transit-timing-variation (TTV) systems where folding smears the dip.
  * Highly-non-uniform sampling patterns where periodograms underperform.

Architecture: encoder-only transformer with sinusoidal position encoding,
~30 M params at default settings. Tokens are fixed-length flux chunks
(default ``chunk_size=8`` cadences -> 8 floats per token), so a 27-day
2-min sector becomes ~1700 tokens.

Training:
  * ``Phase 4`` injection-recovery + TOI labels provide per-transit masks.
  * ``Phase 8`` masked-modeling SSL pretrains the encoder on unlabeled
    TESS LCs first; this script just defines the architecture.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False
    nn = None  # type: ignore


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


def make_token_inputs(
    flux: np.ndarray, chunk_size: int = 8, max_tokens: int = 2048,
) -> np.ndarray:
    """
    Chunk the (median-divided, detrended) ``flux`` array into fixed-length
    tokens. Returns ``(T, chunk_size)`` float32 array, padded or truncated to
    ``max_tokens``.
    """
    f = np.asarray(flux, dtype=np.float32)
    f = np.nan_to_num(f, nan=0.0)
    n = len(f)
    pad = (chunk_size - (n % chunk_size)) % chunk_size
    if pad:
        f = np.concatenate([f, np.zeros(pad, dtype=np.float32)])
    tokens = f.reshape(-1, chunk_size)
    if len(tokens) >= max_tokens:
        return tokens[:max_tokens]
    pad_tokens = np.zeros((max_tokens - len(tokens), chunk_size), dtype=np.float32)
    return np.concatenate([tokens, pad_tokens], axis=0)


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------


class SinusoidalPositionalEncoding(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class TransformerDetector(nn.Module if TORCH_AVAILABLE else object):
    """Encoder-only transformer with per-token transit probability head."""

    def __init__(
        self,
        chunk_size: int = 8,
        d_model: int = 192,
        n_heads: int = 6,
        n_layers: int = 6,
        dim_ff: int = 768,
        dropout: float = 0.1,
        max_tokens: int = 2048,
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required")
        super().__init__()
        self.chunk_size = chunk_size
        self.embed = nn.Linear(chunk_size, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_tokens)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.transit_head = nn.Linear(d_model, 1)
        # Optional pooled-sequence head for whole-LC binary classification.
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(
        self, tokens: "torch.Tensor", attention_mask: Optional["torch.Tensor"] = None,
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        tokens: (B, T, chunk_size) float
        attention_mask: (B, T) bool — True = attend, False = pad

        Returns:
          per_cadence_logit: (B, T) -- transit probability for each token
          whole_lc_logit:    (B, 2) -- planet-vs-no-planet for the whole LC
        """
        x = self.embed(tokens)
        x = self.pos(x)
        # Build src_key_padding_mask: True at positions to ignore.
        if attention_mask is not None:
            key_padding = ~attention_mask
        else:
            key_padding = None
        h = self.encoder(x, src_key_padding_mask=key_padding)
        per_cadence = self.transit_head(h).squeeze(-1)
        # Mean-pool over non-padded positions for whole-LC classification.
        if attention_mask is not None:
            mask = attention_mask.float().unsqueeze(-1)
            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = h.mean(dim=1)
        return per_cadence, self.cls_head(pooled)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def find_transit_events(
    per_cadence_prob: np.ndarray, time: np.ndarray,
    threshold: float = 0.9, min_gap: int = 3, chunk_size: int = 8,
):
    """
    Convert a per-token transit probability sequence into a list of transit
    event timestamps. ``min_gap`` (in tokens) deduplicates adjacent
    above-threshold tokens that belong to the same physical transit.

    Returns list of (t_start, t_end, peak_prob) tuples.
    """
    p = np.asarray(per_cadence_prob)
    t = np.asarray(time, dtype=np.float64)
    n_tokens = len(p)
    cadences_per_token = chunk_size

    events = []
    i = 0
    while i < n_tokens:
        if p[i] < threshold:
            i += 1
            continue
        start = i
        peak_p = float(p[i])
        while i < n_tokens and p[i] >= threshold:
            peak_p = max(peak_p, float(p[i]))
            i += 1
        end = i
        # Convert token range to time range.
        t_lo_idx = start * cadences_per_token
        t_hi_idx = min(end * cadences_per_token, len(t) - 1)
        if t_hi_idx >= len(t):
            break
        events.append((float(t[t_lo_idx]), float(t[t_hi_idx]), peak_p))
        i += min_gap
    return events


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
