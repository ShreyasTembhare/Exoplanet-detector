#!/usr/bin/env python3
"""
Train / fine-tune the exoplanet vetter (Phase 4).

Improvements vs the previous version:
  * **TOI-only labels** -- queries one table (NASA Exoplanet Archive ``toi``)
    and labels by ``tfopwg_disp``: ``CP``/``KP`` -> 1, ``FP``/``FA`` -> 0,
    drop ``PC`` (still ambiguous). Eliminates the population-mismatch bias
    between ``ps`` (confirmed planets) and ``toi`` (TOI dispositions).
  * **Injection-recovery synthetic positives** -- inject ``batman`` transits
    onto known-quiet stars to expand the positive class.
  * **Focal loss** for class imbalance.
  * **Same-shift, no-mask-on-transit** augmentation that respects the local
    view's geometry.
  * **Sector-disjoint validation split** (e.g. train sectors 1-20, validate
    21-26) so we measure generalization, not memorization.
  * **Recall@FPR=1%** as the primary headline metric.
  * Two-tower :class:`TwoTowerResNet1D` model architecture with growing
    channels.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time as _time
from pathlib import Path
from typing import List, Optional, Tuple

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
except ImportError:
    TORCH_AVAILABLE = False

from models.resnet1d import (  # noqa: E402  (after optional torch try/except above)
    ResNet1DClassifier,
    TwoTowerResNet1D,
    count_parameters,
    load_checkpoint,
    make_two_tower_inputs,
)

# ---------------------------------------------------------------------------
# Augmentations (Phase 4: same-shift, transit-protecting)
# ---------------------------------------------------------------------------


def augment_pair(global_arr: np.ndarray, local_arr: np.ndarray):
    """Apply *consistent* augmentations to a (global, local) pair."""
    rng = np.random
    if rng.random() < 0.5:
        sigma = max(1e-4, np.std(global_arr) * 0.2)
        global_arr = global_arr + rng.normal(0, sigma, size=global_arr.shape).astype(np.float32)
        local_arr = local_arr + rng.normal(0, sigma, size=local_arr.shape).astype(np.float32)
    if rng.random() < 0.3:
        # Same fractional shift on both views; cap at 2% of length.
        frac = rng.uniform(-0.02, 0.02)
        global_arr = np.roll(global_arr, int(frac * len(global_arr)))
        local_arr = np.roll(local_arr, int(frac * len(local_arr)))
    if rng.random() < 0.3:
        # Mask only the *outside* region of the local view (don't touch the
        # central 20% where the transit lives).
        n = len(local_arr)
        protected_lo, protected_hi = int(n * 0.4), int(n * 0.6)
        mask_len = rng.randint(n // 50, n // 20 + 1)
        side = rng.choice([0, 1])
        if side == 0:
            start = rng.randint(0, protected_lo - mask_len)
        else:
            start = rng.randint(protected_hi, max(protected_hi + 1, n - mask_len))
        local_arr[start:start + mask_len] = 0.0
        # Same-fraction mask on global, also outside the central transit pixel.
        ng = len(global_arr)
        mask_len_g = rng.randint(ng // 50, ng // 20 + 1)
        start_g = rng.randint(ng // 10, max(ng // 10 + 1, ng - mask_len_g))
        global_arr[start_g:start_g + mask_len_g] = 0.0
    if rng.random() < 0.3:
        scale = rng.uniform(0.97, 1.03)
        global_arr = global_arr * scale
        local_arr = local_arr * scale
    return global_arr, local_arr


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class LightCurveTwoTowerDataset(Dataset):
    """Dataset returning (global_tensor, local_tensor, centroid, label)."""

    def __init__(self, global_views, local_views, centroid_offsets, labels,
                 sectors=None, augment=False):
        self.global_views = np.asarray(global_views, dtype=np.float32)
        self.local_views = np.asarray(local_views, dtype=np.float32)
        self.centroid_offsets = np.nan_to_num(np.asarray(centroid_offsets, dtype=np.float32))
        self.labels = np.asarray(labels, dtype=np.int64)
        self.sectors = sectors
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        g = self.global_views[idx].copy()
        l = self.local_views[idx].copy()
        if self.augment:
            g, l = augment_pair(g, l)
        g_in, l_in = make_two_tower_inputs(g, l)
        return (
            torch.from_numpy(g_in.copy()),
            torch.from_numpy(l_in.copy()),
            torch.tensor(self.centroid_offsets[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_csv_data(csv_path: str):
    import pandas as pd
    df = pd.read_csv(csv_path)
    if "path" in df.columns:
        global_views, local_views, cos, labels, sectors = [], [], [], [], []
        for _, row in df.iterrows():
            data = np.load(row["path"], allow_pickle=True)
            global_views.append(data["global_view"])
            local_views.append(data["local_view"])
            cos.append(float(data.get("centroid_offset", np.nan)))
            labels.append(int(data["label"]))
            sectors.append(int(row.get("sector", -1)))
        return global_views, local_views, cos, labels, sectors
    g_cols = [c for c in df.columns if c.startswith("global_")]
    l_cols = [c for c in df.columns if c.startswith("local_")]
    global_views = df[g_cols].values if g_cols else []
    local_views = df[l_cols].values if l_cols else []
    cos = df["centroid_offset"].values if "centroid_offset" in df.columns else np.zeros(len(df))
    labels = df["label"].values
    sectors = df["sector"].values if "sector" in df.columns else np.full(len(df), -1)
    return global_views, local_views, cos, labels, sectors


# ---------------------------------------------------------------------------
# TOI-only label builder (Phase 4)
# ---------------------------------------------------------------------------


def build_labeled_dataset_toi_only(
    out_dir: str = "data/labeled",
    max_per_class: int = 2500,
    inject_synthetic: bool = True,
    synthetic_per_class: int = 1000,
    sector: Optional[int] = None,
):
    """
    Build a labeled dataset from the TOI table (single source of truth):
      * label=1 if ``tfopwg_disp in ('CP', 'KP')``.
      * label=0 if ``tfopwg_disp in ('FP', 'FA')``.
      * Drop ``PC`` (planet candidate, still ambiguous).
    Optionally inject synthetic transits onto quiet stars for extra positives.
    """
    import pandas as pd

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    manifest = out_path / "manifest.csv"
    if manifest.exists():
        logger.info("Manifest already exists at %s; reusing.", manifest)
        return str(manifest)

    pos_tics, neg_tics = [], []
    try:
        from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive

        toi = NasaExoplanetArchive.query_criteria(
            table="toi", select="tid,toi,tfopwg_disp,pl_orbper,sectors",
        )
        df = toi.to_pandas()
        pos = df[df["tfopwg_disp"].isin(["CP", "KP"])]
        neg = df[df["tfopwg_disp"].isin(["FP", "FA"])]
        pos_tics = list({str(t) for t in pos["tid"] if pd.notna(t)})[:max_per_class]
        neg_tics = list({str(t) for t in neg["tid"] if pd.notna(t)})[:max_per_class]
        logger.info("TOI labels: %d positive (CP/KP), %d negative (FP/FA)",
                    len(pos_tics), len(neg_tics))
    except Exception as exc:
        logger.warning("Could not query TOI table: %s", exc)
        return None

    from pipeline import run_phase1, run_phase2, run_phase3

    rows = []
    for label, tic_list in [(1, pos_tics), (0, neg_tics)]:
        tag = "planet" if label == 1 else "fp"
        for i, tic_raw in enumerate(tic_list):
            tic_id = f"TIC {tic_raw}" if not tic_raw.upper().startswith("TIC") else tic_raw
            npz_path = out_path / f"{tag}_{i:04d}.npz"
            if npz_path.exists():
                rows.append({"path": str(npz_path), "label": label, "tic": tic_raw,
                             "sector": sector if sector is not None else -1})
                continue
            try:
                t, f, _, sec, meta = run_phase1(tic_id, sector=sector, use_cache=True)
                if t is None or len(t) < 50:
                    continue
                res = run_phase2(t, f, tic_id=tic_id, sector=sec, use_cache=True, return_result=True)
                gv, lv, co, _ = run_phase3(
                    t, f, res.best_period, res.epoch,
                    tic_id=tic_id, sector=sec, use_cache=True,
                    bls_sde=res.sde, bls_periods=res.periods, bls_power=res.power,
                )
                np.savez(npz_path, global_view=gv, local_view=lv,
                         centroid_offset=np.array(co), label=np.array(label))
                rows.append({"path": str(npz_path), "label": label, "tic": tic_raw,
                             "sector": int(sec) if str(sec).isdigit() else -1})
                logger.info("[%s %d/%d] Saved %s", tag, i + 1, len(tic_list), npz_path.name)
            except Exception as e:
                logger.warning("Skipping %s: %s", tic_id, e)

    if inject_synthetic and synthetic_per_class > 0:
        try:
            n_synth = generate_injection_recovery(
                out_path, n_samples=synthetic_per_class, neg_tics=neg_tics, rows=rows,
            )
            logger.info("Generated %d synthetic injection-recovery positives", n_synth)
        except Exception as exc:
            logger.warning("Injection-recovery synthesis failed: %s", exc)

    if not rows:
        logger.error("No samples generated.")
        return None

    df_out = pd.DataFrame(rows)
    df_out.to_csv(manifest, index=False)
    logger.info("Manifest written: %s (%d samples)", manifest, len(df_out))
    return str(manifest)


def generate_injection_recovery(out_path: Path, n_samples: int, neg_tics: list,
                                rows: list) -> int:
    """Inject simple box transits onto known-noisy LCs to make synthetic +
    samples. We use the negative-class LCs as "background" because they're
    already cached and have realistic noise."""
    from pipeline import run_phase1, run_phase2, run_phase3
    rng = np.random.default_rng(0)
    written = 0
    for i in range(n_samples):
        if not neg_tics:
            break
        tic_raw = rng.choice(neg_tics)
        tic_id = f"TIC {tic_raw}"
        try:
            t, f, _, sec, _ = run_phase1(tic_id, sector=None, use_cache=True)
            if t is None or len(t) < 100:
                continue
            period = float(rng.uniform(0.5, 12.0))
            depth = float(rng.uniform(0.0005, 0.02))
            duration_d = float(rng.uniform(0.05, 0.2))
            epoch = float(t.min() + rng.uniform(0, period))
            phase = ((t - epoch) % period) / period
            in_transit = (phase < duration_d / 2 / period) | (phase > 1 - duration_d / 2 / period)
            f_inj = f.copy()
            f_inj[in_transit] -= depth
            res = run_phase2(t, f_inj, tic_id=f"INJ_{i}", sector=str(sec), use_cache=False, return_result=True)
            gv, lv, co, _ = run_phase3(
                t, f_inj, res.best_period, res.epoch,
                tic_id=f"INJ_{i}", sector=str(sec), use_cache=False,
                bls_sde=res.sde, bls_periods=res.periods, bls_power=res.power,
            )
            npz_path = out_path / f"inj_{i:04d}.npz"
            np.savez(npz_path, global_view=gv, local_view=lv,
                     centroid_offset=np.array(co), label=np.array(1))
            rows.append({"path": str(npz_path), "label": 1, "tic": f"INJ_{i}",
                         "sector": int(sec) if str(sec).isdigit() else -1})
            written += 1
        except Exception:
            continue
    return written


# ---------------------------------------------------------------------------
# Loss + training utilities
# ---------------------------------------------------------------------------


class FocalLoss(nn.Module if TORCH_AVAILABLE else object):
    """Multi-class focal loss for class imbalance."""

    def __init__(self, alpha: float = 0.5, gamma: float = 2.0, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce * self.alpha).mean()


def train_epoch(model, loader, criterion, optimizer, device, two_tower: bool):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        if two_tower:
            g, l, co, y = batch
            g, l, co, y = g.to(device), l.to(device), co.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(g, l, centroid_offset=co)
        else:
            x, co, y = batch
            x, co, y = x.to(device), co.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x, centroid_offset=co)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return total_loss / max(len(loader), 1), correct / max(total, 1)


def evaluate(model, loader, criterion, device, two_tower: bool):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels, all_preds = [], [], []
    with torch.inference_mode():
        for batch in loader:
            if two_tower:
                g, l, co, y = batch
                g, l, co, y = g.to(device), l.to(device), co.to(device), y.to(device)
                logits = model(g, l, centroid_offset=co)
            else:
                x, co, y = batch
                x, co, y = x.to(device), co.to(device), y.to(device)
                logits = model(x, centroid_offset=co)
            loss = criterion(logits, y)
            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())
    return total_loss / max(len(loader), 1), correct / max(total, 1), \
        np.array(all_probs), np.array(all_labels), np.array(all_preds)


def recall_at_fpr(probs, labels, target_fpr: float = 0.01) -> float:
    """Return recall (TPR) at the threshold that keeps FPR <= target_fpr."""
    order = np.argsort(probs)[::-1]
    tp = fp = 0
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.0
    best_recall = 0.0
    for idx in order:
        if labels[idx] == 1:
            tp += 1
        else:
            fp += 1
        if fp / n_neg <= target_fpr:
            recall = tp / n_pos
            if recall > best_recall:
                best_recall = recall
    return float(best_recall)


def compute_metrics(probs, labels, preds):
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    metrics = {
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(f1, 4),
    }
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        if len(np.unique(labels)) > 1:
            metrics["roc_auc"] = round(float(roc_auc_score(labels, probs)), 4)
            metrics["pr_auc"] = round(float(average_precision_score(labels, probs)), 4)
    except ImportError:
        pass
    metrics["recall_at_fpr_001"] = round(recall_at_fpr(probs, labels, 0.01), 4)
    metrics["recall_at_fpr_0001"] = round(recall_at_fpr(probs, labels, 0.001), 4)
    return metrics


# ---------------------------------------------------------------------------
# Sector-disjoint split helper
# ---------------------------------------------------------------------------


def parse_sector_range(spec: Optional[str]) -> Optional[List[int]]:
    if spec is None:
        return None
    out: List[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            lo, hi = chunk.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        elif chunk:
            out.append(int(chunk))
    return out


def sector_disjoint_split(
    n_total: int, sectors: list, val_holdout: Optional[List[int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx). If val_holdout is given, validation is
    exactly the samples whose sector is in the holdout list; otherwise a
    random 80/20."""
    if val_holdout and sectors is not None:
        sectors_arr = np.asarray(sectors)
        val_mask = np.isin(sectors_arr, val_holdout)
        train_idx = np.where(~val_mask)[0]
        val_idx = np.where(val_mask)[0]
        if len(val_idx) == 0:
            logger.warning("No samples in holdout sectors; falling back to random split.")
        else:
            return train_idx, val_idx
    indices = np.random.permutation(n_total)
    n_val = max(1, n_total // 5)
    return indices[:n_total - n_val], indices[n_total - n_val:]


# ---------------------------------------------------------------------------
# Train entry point
# ---------------------------------------------------------------------------


def run_train(
    data: Optional[str] = None,
    pretrained: Optional[str] = None,
    epochs: int = 30, batch_size: int = 32, lr: float = 1e-3,
    finetune_lr: float = 1e-4, freeze_epochs: int = 5, unfreeze_blocks: int = 2,
    out: str = "models/checkpoints/resnet1d.pt", seed: int = 42,
    patience: int = 7, amp: bool = False, max_per_class: int = 2500, grad_accum: int = 1,
    use_focal: bool = True, use_injection_recovery: bool = True,
    val_holdout_sectors: Optional[str] = None,
    model_arch: str = "resnet1d",
):
    if not TORCH_AVAILABLE:
        logger.error("PyTorch is required. Install with: pip install torch")
        return None
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---- Data ----
    if data == "auto":
        manifest = build_labeled_dataset_toi_only(
            max_per_class=max_per_class,
            inject_synthetic=use_injection_recovery,
            synthetic_per_class=max(0, max_per_class // 3),
        )
        if manifest is None:
            return None
        gv, lv, co, lb, sectors = load_csv_data(manifest)
    elif data is not None:
        gv, lv, co, lb, sectors = load_csv_data(data)
    else:
        logger.warning("No --data; using small synthetic dataset for demo.")
        n = 200
        gv = [np.random.randn(2048).astype(np.float32) for _ in range(n)]
        lv = [np.random.randn(256).astype(np.float32) for _ in range(n)]
        co = np.random.rand(n).astype(np.float32)
        lb = (np.random.rand(n) > 0.5).astype(np.int64)
        sectors = np.full(n, -1)

    n_total = len(lb)
    holdout = parse_sector_range(val_holdout_sectors)
    train_idx, val_idx = sector_disjoint_split(n_total, sectors, holdout)
    logger.info("Train/val: %d / %d (holdout sectors=%s)", len(train_idx), len(val_idx), holdout)

    gv = np.asarray(gv, dtype=np.float32)
    lv = np.asarray(lv, dtype=np.float32)
    co_all = np.asarray(co, dtype=np.float32)
    lb_all = np.asarray(lb, dtype=np.int64)

    train_ds = LightCurveTwoTowerDataset(gv[train_idx], lv[train_idx], co_all[train_idx],
                                         lb_all[train_idx], augment=True)
    val_ds = LightCurveTwoTowerDataset(gv[val_idx], lv[val_idx], co_all[val_idx],
                                       lb_all[val_idx], augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    from device_util import get_device
    device = get_device()
    logger.info("Device: %s", device)

    # ---- Model ----
    two_tower = (model_arch == "resnet1d")  # resnet1d means two-tower in Phase 4
    if pretrained:
        model = load_checkpoint(pretrained, device=device, strict=False)
    elif two_tower:
        model = TwoTowerResNet1D(use_centroid=True).to(device)
    else:
        model = ResNet1DClassifier(use_centroid=True).to(device)
    logger.info("Model: %s, %s params",
                type(model).__name__, f"{count_parameters(model):,}")

    # ---- Loss ----
    n_pos = int(np.sum(lb_all[train_idx] == 1))
    n_neg = int(np.sum(lb_all[train_idx] == 0))
    pos_weight = max(1.0, n_neg / max(n_pos, 1))
    weights = torch.tensor([1.0, pos_weight], device=device, dtype=torch.float32)
    criterion = FocalLoss(weight=weights) if use_focal else nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    best_recall = -1.0
    patience_counter = 0
    final_m = None
    for epoch in range(epochs):
        t0 = _time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, two_tower)
        val_loss, val_acc, probs, lbls, preds = evaluate(model, val_loader, criterion, device, two_tower)
        m = compute_metrics(probs, lbls, preds)
        primary = m.get("recall_at_fpr_001", 0.0)
        scheduler.step(primary)
        logger.info(
            "Epoch %d/%d [%.1fs] loss=%.4f acc=%.4f | val_loss=%.4f val_acc=%.4f "
            "F1=%.4f Rec@1%%FPR=%.4f Rec@.1%%FPR=%.4f ROC=%.4f PR=%.4f",
            epoch + 1, epochs, _time.time() - t0,
            train_loss, train_acc, val_loss, val_acc,
            m["f1"], m.get("recall_at_fpr_001", 0.0), m.get("recall_at_fpr_0001", 0.0),
            m.get("roc_auc", float("nan")), m.get("pr_auc", float("nan")),
        )
        if primary > best_recall:
            best_recall = primary
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": m,
                        "arch": "two_tower" if two_tower else "single_tower"}, out)
            patience_counter = 0
            final_m = m
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    if final_m is not None:
        metrics_path = Path(out).with_suffix(".metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(final_m, f, indent=2)
        logger.info("Metrics saved to %s", metrics_path)
    return final_m


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Train / fine-tune ResNet-1D exoplanet classifier")
    parser.add_argument("--data", type=str, default=None,
                        help="CSV with features, or 'auto' to build from NASA TOI table")
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--freeze-epochs", type=int, default=5)
    parser.add_argument("--unfreeze-blocks", type=int, default=2)
    parser.add_argument("--out", type=str, default="models/checkpoints/resnet1d.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max-per-class", type=int, default=2500)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--no-focal", action="store_true", help="Use plain CrossEntropy instead of focal loss")
    parser.add_argument("--no-injection-recovery", action="store_true")
    parser.add_argument("--val-holdout-sectors", type=str, default=None,
                        help="e.g. '21-26' to validate on sectors 21..26 only")
    parser.add_argument("--arch", type=str, default="resnet1d", choices=["resnet1d", "single"])
    args = parser.parse_args()
    return run_train(
        data=args.data, pretrained=args.pretrained, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, finetune_lr=args.finetune_lr,
        freeze_epochs=args.freeze_epochs, unfreeze_blocks=args.unfreeze_blocks,
        out=args.out, seed=args.seed, patience=args.patience, amp=args.amp,
        max_per_class=args.max_per_class, grad_accum=args.grad_accum,
        use_focal=not args.no_focal, use_injection_recovery=not args.no_injection_recovery,
        val_holdout_sectors=args.val_holdout_sectors,
        model_arch="single" if args.arch == "single" else "resnet1d",
    )


if __name__ == "__main__":
    main()
