"""
Temperature scaling (Guo et al. 2017).

A classifier trained with focal loss and class weights ranks well but its
softmax is not a probability: 0.9 does not mean nine in ten such signals are
planets. One scalar ``T`` fitted on held-out logits by minimising the
negative log-likelihood fixes that without changing the ranking, so the
desk threshold and the candidate score mean what they say.
"""

from __future__ import annotations

import numpy as np


def _nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    z = logits / max(float(temperature), 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    log_p = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-np.mean(log_p[np.arange(len(labels)), labels]))


def fit_temperature(logits, labels, lo: float = 0.05, hi: float = 20.0) -> float:
    """Return the NLL-optimal temperature (1.0 when there is nothing to fit)."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if logits.ndim != 2 or len(labels) < 2 or len(np.unique(labels)) < 2:
        return 1.0
    # Coarse log-grid, then golden-section refinement in log T.
    grid = np.exp(np.linspace(np.log(lo), np.log(hi), 60))
    losses = [_nll(logits, labels, t) for t in grid]
    i = int(np.argmin(losses))
    a = np.log(grid[max(i - 1, 0)])
    b = np.log(grid[min(i + 1, len(grid) - 1)])
    phi = (np.sqrt(5) - 1) / 2
    for _ in range(40):
        c = b - phi * (b - a)
        d = a + phi * (b - a)
        if _nll(logits, labels, np.exp(c)) < _nll(logits, labels, np.exp(d)):
            b = d
        else:
            a = c
    return float(np.exp((a + b) / 2))


def calibrated_probs(logits, temperature: float = 1.0) -> np.ndarray:
    """Positive-class probability after temperature scaling."""
    z = np.asarray(logits, dtype=np.float64) / max(float(temperature), 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p[:, 1] / p.sum(axis=1)


def expected_calibration_error(probs, labels, n_bins: int = 10) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if len(probs) == 0:
        return float("nan")
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (probs >= lo) & (probs < hi if hi < 1 else probs <= hi)
        if sel.any():
            ece += sel.mean() * abs(probs[sel].mean() - labels[sel].mean())
    return float(ece)
