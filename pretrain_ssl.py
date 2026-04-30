#!/usr/bin/env python3
"""
Self-supervised pretraining (Phase 8).

Two pretraining objectives, each writes a checkpoint that downstream
:func:`train_classifier.run_train` can load via ``--pretrained``:

1. **Masked modeling** for the :class:`models.transformer_detector.TransformerDetector`.
   Mask 15% of input tokens; predict the masked values. Encoder learns a
   general representation of TESS light curves without needing labels.

2. **SimCLR contrastive** for the :class:`models.resnet1d.TwoTowerResNet1D`
   towers. Two augmented views of the same star are positives; views from
   different stars are negatives. Encoder learns transit-relevant features.

Inputs are unlabeled cached Phase 1 LCs, so the corpus scales with however
many stars you've ever processed (the broken Phase 0 cache is now fixed
and persists across runs).

Both pretrainers stream from disk via a Phase-1 cache reader, so the corpus
can be much larger than RAM.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time as _time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Unlabeled corpus loader: walk cache/phase1/*.npz
# ---------------------------------------------------------------------------


class Phase1Corpus(Dataset):
    """Stream cached Phase 1 light curves from disk."""

    def __init__(self, cache_dir: str = "cache/phase1", target_len: int = 16384):
        self.paths = sorted(Path(cache_dir).glob("*.npz"))
        self.target_len = target_len
        if not self.paths:
            logger.warning("No cached LCs found in %s", cache_dir)

    def __len__(self):
        return len(self.paths)

    def _load_flux(self, path: Path) -> np.ndarray:
        try:
            data = np.load(path, allow_pickle=True)
            f = np.asarray(data.get("flux", np.array([])), dtype=np.float32)
            if len(f) == 0:
                return np.zeros(self.target_len, dtype=np.float32)
            f = np.nan_to_num(f, nan=0.0)
            f = f / (np.median(f) or 1.0) - 1.0
            if len(f) >= self.target_len:
                start = np.random.randint(0, len(f) - self.target_len + 1)
                return f[start:start + self.target_len]
            out = np.zeros(self.target_len, dtype=np.float32)
            out[: len(f)] = f
            return out
        except Exception:
            return np.zeros(self.target_len, dtype=np.float32)

    def __getitem__(self, idx):
        return self._load_flux(self.paths[idx])


# ---------------------------------------------------------------------------
# Augmentations for SimCLR
# ---------------------------------------------------------------------------


def _augment_lc(flux: np.ndarray) -> np.ndarray:
    rng = np.random
    out = flux.copy()
    # Gaussian noise scaled to per-LC sigma.
    sigma = max(np.std(flux), 1e-5)
    out = out + rng.normal(0, sigma * 0.2, size=flux.shape).astype(np.float32)
    # Random crop / pad.
    n = len(out)
    crop = rng.randint(int(n * 0.85), n + 1)
    start = rng.randint(0, n - crop + 1)
    cropped = out[start:start + crop]
    if len(cropped) < n:
        cropped = np.concatenate([cropped, np.zeros(n - len(cropped), dtype=np.float32)])
    return cropped[:n]


# ---------------------------------------------------------------------------
# Masked-modeling pretraining for the transformer
# ---------------------------------------------------------------------------


def pretrain_transformer_mlm(
    cache_dir: str = "cache/phase1",
    out: str = "models/checkpoints/transformer_mlm.pt",
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-4,
    chunk_size: int = 8,
    max_tokens: int = 2048,
    mask_frac: float = 0.15,
):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required")
    from device_util import get_device
    from models.transformer_detector import TransformerDetector, make_token_inputs

    device = get_device()
    model = TransformerDetector(chunk_size=chunk_size, max_tokens=max_tokens).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    target_len = chunk_size * max_tokens
    corpus = Phase1Corpus(cache_dir, target_len=target_len)
    if len(corpus) == 0:
        logger.error("Empty corpus; populate cache/phase1/ via the hunter first.")
        return None
    loader = DataLoader(corpus, batch_size=batch_size, shuffle=True, num_workers=0)

    logger.info("MLM pretraining: %d LCs, %d tokens/LC, %d epochs", len(corpus), max_tokens, epochs)
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss, n_batch = 0.0, 0
        t0 = _time.time()
        for fluxes in loader:
            fluxes = fluxes.numpy()
            tokens = np.stack([make_token_inputs(f, chunk_size, max_tokens) for f in fluxes], axis=0)
            tokens = torch.from_numpy(tokens).to(device)
            B, T, C = tokens.shape

            mask = torch.rand(B, T, device=device) < mask_frac
            target = tokens.clone()
            tokens_masked = tokens.clone()
            tokens_masked[mask] = 0.0

            optimizer.zero_grad()
            x = model.embed(tokens_masked)
            x = model.pos(x)
            h = model.encoder(x)
            # Reuse linear head as a token-reconstruction projector.
            reproj = nn.functional.linear(
                h.reshape(-1, h.shape[-1]),
                model.embed.weight,
                bias=None,
            ).reshape(B, T, C)
            loss = F.mse_loss(reproj[mask], target[mask])
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            n_batch += 1

        avg = total_loss / max(n_batch, 1)
        logger.info("MLM epoch %d/%d  loss=%.5f  time=%.1fs", epoch + 1, epochs, avg, _time.time() - t0)
        torch.save({"model": model.state_dict(), "epoch": epoch, "objective": "mlm"}, out)
    logger.info("Saved MLM checkpoint to %s", out)
    return out


# ---------------------------------------------------------------------------
# SimCLR pretraining for the ResNet towers
# ---------------------------------------------------------------------------


def info_nce(z1, z2, temperature: float = 0.1):
    """Symmetric InfoNCE / NT-Xent loss."""
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    z = torch.cat([z1, z2], dim=0)
    sim = z @ z.t() / temperature
    n = z1.size(0)
    mask = torch.eye(2 * n, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float("-inf"))
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, targets)


def pretrain_two_tower_simclr(
    cache_dir: str = "cache/phase1",
    out: str = "models/checkpoints/twotower_simclr.pt",
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-4,
):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required")
    from device_util import get_device
    from models.resnet1d import TwoTowerResNet1D, make_two_tower_inputs
    from pipeline.fold_features import global_view, local_view

    device = get_device()
    model = TwoTowerResNet1D(use_centroid=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    corpus = Phase1Corpus(cache_dir, target_len=16384)
    if len(corpus) == 0:
        logger.error("Empty corpus.")
        return None
    loader = DataLoader(corpus, batch_size=batch_size, shuffle=True, num_workers=0)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    logger.info("SimCLR pretraining: %d LCs, %d epochs", len(corpus), epochs)

    for epoch in range(epochs):
        model.train()
        total, n = 0.0, 0
        t0 = _time.time()
        for fluxes in loader:
            f_np = fluxes.numpy()
            view1, view2 = [], []
            for f in f_np:
                # Two augmented views of the same LC.
                a = _augment_lc(f)
                b = _augment_lc(f)
                # Re-create the global and local 1D inputs.
                t = np.linspace(0, 27, len(a), dtype=np.float64)
                gv1 = global_view(t, a, period=3.0, epoch=0.0)
                lv1 = local_view(t, a, period=3.0, epoch=0.0)
                gv2 = global_view(t, b, period=3.0, epoch=0.0)
                lv2 = local_view(t, b, period=3.0, epoch=0.0)
                g1, l1 = make_two_tower_inputs(gv1, lv1)
                g2, l2 = make_two_tower_inputs(gv2, lv2)
                view1.append((g1, l1))
                view2.append((g2, l2))

            g1 = torch.from_numpy(np.stack([v[0] for v in view1])).float().to(device)
            l1 = torch.from_numpy(np.stack([v[1] for v in view1])).float().to(device)
            g2 = torch.from_numpy(np.stack([v[0] for v in view2])).float().to(device)
            l2 = torch.from_numpy(np.stack([v[1] for v in view2])).float().to(device)

            # Use only the towers (skip head).
            z1 = torch.cat([
                model.global_tower(g1).squeeze(-1),
                model.local_tower(l1).squeeze(-1),
            ], dim=1)
            z2 = torch.cat([
                model.global_tower(g2).squeeze(-1),
                model.local_tower(l2).squeeze(-1),
            ], dim=1)

            optimizer.zero_grad()
            loss = info_nce(z1, z2)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            n += 1

        logger.info("SimCLR epoch %d/%d  loss=%.4f  time=%.1fs",
                    epoch + 1, epochs, total / max(n, 1), _time.time() - t0)
        torch.save({"model": model.state_dict(), "epoch": epoch, "objective": "simclr"}, out)
    logger.info("Saved SimCLR checkpoint to %s", out)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Self-supervised pretraining for TESS exoplanet models")
    sub = parser.add_subparsers(dest="objective", required=True)

    mlm = sub.add_parser("mlm", help="Masked-modeling pretrain for the transformer")
    mlm.add_argument("--cache-dir", default="cache/phase1")
    mlm.add_argument("--out", default="models/checkpoints/transformer_mlm.pt")
    mlm.add_argument("--epochs", type=int, default=10)
    mlm.add_argument("--batch-size", type=int, default=16)
    mlm.add_argument("--lr", type=float, default=1e-4)
    mlm.add_argument("--max-tokens", type=int, default=2048)
    mlm.add_argument("--mask-frac", type=float, default=0.15)

    simclr = sub.add_parser("simclr", help="Contrastive pretrain for the two-tower ResNet")
    simclr.add_argument("--cache-dir", default="cache/phase1")
    simclr.add_argument("--out", default="models/checkpoints/twotower_simclr.pt")
    simclr.add_argument("--epochs", type=int, default=10)
    simclr.add_argument("--batch-size", type=int, default=16)
    simclr.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()
    if args.objective == "mlm":
        return pretrain_transformer_mlm(
            cache_dir=args.cache_dir, out=args.out, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr, max_tokens=args.max_tokens,
            mask_frac=args.mask_frac,
        )
    if args.objective == "simclr":
        return pretrain_two_tower_simclr(
            cache_dir=args.cache_dir, out=args.out, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr,
        )


if __name__ == "__main__":
    main()
