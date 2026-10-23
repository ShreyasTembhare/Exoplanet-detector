"""
ExoMiner second-stage vetter.

The two-tower ResNet triages every SDE-pass peak; when an ExoMiner
checkpoint exists (``EXOMINER_CHECKPOINT`` or
``models/checkpoints/exominer.pt``) the survivors are re-scored with the
multi-input model, which also sees the centred local view, periodogram and
vetting scalars. The two calibrated probabilities are combined with a
geometric mean so either model can veto. ``train_exominer`` trains it on the
same manifest the triage model uses (``pipeline.training_data``), holding
out whole sectors and fitting a temperature on them.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = "models/checkpoints/exominer.pt"


def checkpoint_path() -> Path:
    return Path(os.environ.get("EXOMINER_CHECKPOINT", DEFAULT_CHECKPOINT))


def scalars_from_row(row: dict) -> dict:
    vetting = dict(row.get("vetting") or {})
    out = dict(vetting)
    out.setdefault("sde_bls", row.get("sde"))
    out.setdefault("period_days", row.get("period", row.get("best_period")))
    out.setdefault("centroid_offset_px", row.get("centroid_offset"))
    period, dur = out.get("period_days"), out.get("duration_hours")
    try:
        out.setdefault("duration_to_period", float(dur) / 24.0 / float(period))
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    return out


def load_vetter(path: Optional[str | Path] = None):
    """(model, device, temperature) or None when no checkpoint / torch."""
    path = Path(path) if path else checkpoint_path()
    if not path.is_file():
        return None
    try:
        import torch

        from device_util import get_device
        from models.exominer import ExoMinerVetter
    except ImportError:
        return None
    device = get_device()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ExoMinerVetter()
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    model.to(device).eval()
    logger.info("ExoMiner second stage loaded from %s", path)
    return model, device, float(ckpt.get("temperature", 1.0))


def _batch_inputs(rows: List[dict]):
    import torch

    from models.exominer import make_exominer_inputs

    parts = [[] for _ in range(6)]
    for row in rows:
        inputs = make_exominer_inputs(
            row["global_view"], row["local_view"],
            local_centered=row.get("local_centered"),
            periodogram=row.get("periodogram"),
            diff_image=row.get("diff_image"),
            scalars_dict=row.get("scalars") or scalars_from_row(row),
        )
        for bucket, arr in zip(parts, inputs):
            bucket.append(arr)
    return [torch.from_numpy(np.stack(b)).float() for b in parts]


def vet(bundle, rows: List[dict]) -> List[Optional[float]]:
    """Calibrated planet probability per row (None when it cannot be scored)."""
    if bundle is None or not rows:
        return [None] * len(rows)
    import torch

    model, device, temperature = bundle
    ok = [i for i, r in enumerate(rows) if r.get("global_view") is not None and r.get("local_view") is not None]
    out: List[Optional[float]] = [None] * len(rows)
    if not ok:
        return out
    tensors = [t.to(device) for t in _batch_inputs([rows[i] for i in ok])]
    with torch.inference_mode():
        probs = torch.softmax(model(*tensors) / max(temperature, 1e-3), dim=1)[:, 1].cpu().numpy()
    for i, p in zip(ok, probs):
        out[i] = float(p)
    return out


def combine(primary: Optional[float], second: Optional[float]) -> Optional[float]:
    if second is None:
        return primary
    if primary is None:
        return second
    return float(np.sqrt(max(primary, 0.0) * max(second, 0.0)))


def _load_manifest(manifest: str) -> List[dict]:
    import pandas as pd

    from models.exominer import SCALAR_FEATURE_NAMES

    rows = []
    for rec in pd.read_csv(manifest).to_dict("records"):
        data = np.load(rec["path"], allow_pickle=True)
        scalars = {}
        if "scalars" in data and "scalar_names" in data:
            scalars = {str(k): float(v) for k, v in zip(data["scalar_names"], data["scalars"])
                       if str(k) in SCALAR_FEATURE_NAMES}
        rows.append({
            "global_view": data["global_view"], "local_view": data["local_view"],
            "local_centered": data["local_centered"] if "local_centered" in data else None,
            "periodogram": data["periodogram"] if "periodogram" in data else None,
            "scalars": scalars,
            "label": int(rec.get("label", data["label"] if "label" in data else 0)),
            "sector": int(rec.get("sector", -1) if rec.get("sector") == rec.get("sector") else -1),
        })
    return rows


def train_exominer(manifest: str, out: str = DEFAULT_CHECKPOINT, epochs: int = 20,
                   batch_size: int = 32, lr: float = 5e-4, seed: int = 0) -> dict:
    import torch
    import torch.nn as nn

    from models.calibration import fit_temperature
    from models.exominer import ExoMinerVetter
    from train_classifier import sector_disjoint_split

    torch.manual_seed(seed)
    np.random.seed(seed)
    rows = _load_manifest(manifest)
    labels = np.array([r["label"] for r in rows])
    train_idx, val_idx = sector_disjoint_split(len(rows), [r["sector"] for r in rows], None)
    model = ExoMinerVetter()
    n_pos = max(int(labels[train_idx].sum()), 1)
    weight = torch.tensor([1.0, max(1.0, (len(train_idx) - n_pos) / n_pos)])
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def batches(idx, shuffle):
        idx = np.random.permutation(idx) if shuffle else np.asarray(idx)
        for s in range(0, len(idx), batch_size):
            chunk = idx[s:s + batch_size]
            if shuffle and len(chunk) < 2:
                continue
            yield _batch_inputs([rows[i] for i in chunk]), torch.from_numpy(labels[chunk]).long()

    def val_logits():
        model.eval()
        outs = []
        with torch.inference_mode():
            for xs, _ in batches(val_idx, False):
                outs.append(model(*xs).numpy())
        return np.concatenate(outs) if outs else np.zeros((0, 2))

    best, best_state = -1.0, None
    for epoch in range(epochs):
        model.train()
        for xs, y in batches(train_idx, True):
            opt.zero_grad()
            loss = loss_fn(model(*xs), y)
            loss.backward()
            opt.step()
        logits = val_logits()
        acc = float(np.mean(logits.argmax(1) == labels[val_idx])) if len(logits) else 0.0
        if acc >= best:
            best, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
        logger.info("ExoMiner epoch %d/%d val_acc=%.3f", epoch + 1, epochs, acc)
    if best_state is not None:
        model.load_state_dict(best_state)
    temperature = fit_temperature(val_logits(), labels[val_idx])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "temperature": temperature,
                "val_acc": best, "arch": "exominer"}, out)
    return {"checkpoint": str(out), "val_acc": best, "temperature": temperature,
            "n_train": len(train_idx), "n_val": len(val_idx)}


def main(argv=None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Train the ExoMiner second-stage vetter")
    parser.add_argument("--data", required=True, help="Manifest from pipeline.training_data / train_triage_v2")
    parser.add_argument("--out", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    print(json.dumps(train_exominer(args.data, args.out, args.epochs, args.batch_size), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
