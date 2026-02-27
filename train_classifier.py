#!/usr/bin/env python3
"""
Train / fine-tune ResNet-1D exoplanet classifier.

Supports:
  - Loading a pretrained checkpoint and fine-tuning (freeze backbone -> unfreeze last blocks).
  - Building a labeled dataset from NASA Exoplanet Archive dispositions.
  - Mixed-precision training for VRAM-limited laptop GPUs.
  - Early stopping with best-checkpoint selection.
  - Full evaluation metrics: Precision, Recall, F1, ROC-AUC, PR-AUC, confusion matrix.
"""

import argparse
import json
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
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from models.resnet1d import (
    ResNet1DClassifier,
    make_two_channel,
    count_parameters,
    load_checkpoint,
)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def augment_sample(global_arr: np.ndarray, local_arr: np.ndarray):
    """Apply random augmentations to a single (global, local) pair."""
    rng = np.random
    if rng.random() < 0.5:
        global_arr = global_arr + rng.normal(0, 0.002, size=global_arr.shape).astype(np.float32)
        local_arr = local_arr + rng.normal(0, 0.002, size=local_arr.shape).astype(np.float32)
    if rng.random() < 0.3:
        shift_g = rng.randint(-len(global_arr) // 20, len(global_arr) // 20 + 1)
        shift_l = rng.randint(-len(local_arr) // 20, len(local_arr) // 20 + 1)
        global_arr = np.roll(global_arr, shift_g)
        local_arr = np.roll(local_arr, shift_l)
    if rng.random() < 0.3:
        mask_len = rng.randint(len(global_arr) // 50, len(global_arr) // 20 + 1)
        start = rng.randint(0, len(global_arr) - mask_len)
        global_arr[start:start + mask_len] = 0.0
    if rng.random() < 0.3:
        scale = rng.uniform(0.95, 1.05)
        global_arr = global_arr * scale
        local_arr = local_arr * scale
    return global_arr, local_arr


class LightCurveDataset(Dataset):
    """Dataset of (global_view, local_view, centroid_offset, label)."""

    def __init__(self, global_views, local_views, centroid_offsets, labels, augment=False):
        self.global_views = np.asarray(global_views, dtype=np.float32)
        self.local_views = np.asarray(local_views, dtype=np.float32)
        self.centroid_offsets = np.asarray(centroid_offsets, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        assert len(self.global_views) == len(self.labels)
        self.centroid_offsets = np.nan_to_num(self.centroid_offsets, nan=0.0)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        g = self.global_views[idx].copy()
        l = self.local_views[idx].copy()
        if self.augment:
            g, l = augment_sample(g, l)
        x = make_two_channel(g, l)
        x = torch.from_numpy(x).unsqueeze(0)
        co = torch.tensor(self.centroid_offsets[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, co, y


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_csv_data(csv_path: str):
    """Load precomputed features from CSV (columns or .npz paths)."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    if "path" in df.columns:
        global_views, local_views, centroid_offsets, labels = [], [], [], []
        for p in df["path"]:
            data = np.load(p, allow_pickle=True)
            global_views.append(data["global_view"])
            local_views.append(data["local_view"])
            centroid_offsets.append(float(data.get("centroid_offset", np.nan)))
            labels.append(int(data["label"]))
        return global_views, local_views, centroid_offsets, labels
    g_cols = [c for c in df.columns if c.startswith("global_")]
    l_cols = [c for c in df.columns if c.startswith("local_")]
    global_views = df[g_cols].values if g_cols else []
    local_views = df[l_cols].values if l_cols else []
    centroid_offsets = df["centroid_offset"].values if "centroid_offset" in df.columns else np.zeros(len(df))
    labels = df["label"].values
    return global_views, local_views, centroid_offsets, labels


def build_labeled_dataset_from_archive(
    out_dir: str = "data/labeled",
    max_per_class: int = 250,
    sector: int = None,
):
    """
    Build a labeled dataset by querying the NASA Exoplanet Archive for
    confirmed planets (label=1) and known false positives (label=0),
    then running Phase 1-3 on each to produce feature .npz files.
    Returns path to a CSV manifest.
    """
    import pandas as pd

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    manifest = out_path / "manifest.csv"
    if manifest.exists():
        logger.info("Manifest already exists at %s; reusing.", manifest)
        return str(manifest)

    try:
        from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
        confirmed = NasaExoplanetArchive.query_criteria(
            table="ps", select="tic_id,disc_facility",
            where="disc_facility like '%TESS%' AND default_flag=1",
        )
        confirmed_tics = list(set(str(t) for t in confirmed["tic_id"] if t))[:max_per_class]
    except Exception as e:
        logger.warning("Could not query NASA archive for confirmed planets: %s", e)
        confirmed_tics = []

    try:
        from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
        fps = NasaExoplanetArchive.query_criteria(
            table="toi", select="tid,tfopwg_disp",
            where="tfopwg_disp='FP'",
        )
        fp_tics = list(set(str(t) for t in fps["tid"] if t))[:max_per_class]
    except Exception as e:
        logger.warning("Could not query NASA archive for false positives: %s", e)
        fp_tics = []

    if not confirmed_tics and not fp_tics:
        logger.error("No labeled TICs obtained. Provide --data CSV or check network.")
        return None

    from pipeline import run_phase1, run_phase2, run_phase3

    rows = []
    for label, tic_list in [(1, confirmed_tics), (0, fp_tics)]:
        tag = "planet" if label == 1 else "fp"
        for i, tic_raw in enumerate(tic_list):
            tic_id = f"TIC {tic_raw}" if not tic_raw.upper().startswith("TIC") else tic_raw
            npz_path = out_path / f"{tag}_{i:04d}.npz"
            if npz_path.exists():
                rows.append({"path": str(npz_path), "label": label, "tic": tic_raw})
                continue
            try:
                time, flux, _, sec = run_phase1(tic_id, sector=sector, use_cache=True)
                if time is None or len(time) < 50:
                    continue
                _, _, best_period, epoch = run_phase2(
                    time, flux, tic_id=tic_id, sector=sec, use_cache=True,
                )
                gv, lv, co = run_phase3(
                    time, flux, best_period, epoch,
                    tic_id=tic_id, sector=sec, use_cache=True,
                )
                np.savez(npz_path, global_view=gv, local_view=lv,
                         centroid_offset=np.array(co), label=np.array(label))
                rows.append({"path": str(npz_path), "label": label, "tic": tic_raw})
                logger.info("[%s %d/%d] Saved %s", tag, i + 1, len(tic_list), npz_path.name)
            except Exception as e:
                logger.warning("Skipping %s: %s", tic_id, e)

    if not rows:
        logger.error("No samples generated.")
        return None

    df = pd.DataFrame(rows)
    df.to_csv(manifest, index=False)
    logger.info("Manifest written: %s (%d samples)", manifest, len(df))
    return str(manifest)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_epoch(model, loader, criterion, optimizer, device, scaler=None, amp_device_type=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, co, y in loader:
        x = x.squeeze(1).to(device)
        co = co.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        if amp_device_type is not None:
            with torch.autocast(device_type=amp_device_type, dtype=torch.float16):
                logits = model(x, centroid_offset=co)
                loss = criterion(logits, y)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            logits = model(x, centroid_offset=co)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return total_loss / max(len(loader), 1), correct / max(total, 1)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels, all_preds = [], [], []
    with torch.inference_mode():
        for x, co, y in loader:
            x = x.squeeze(1).to(device)
            co = co.to(device)
            y = y.to(device)
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
    avg_loss = total_loss / max(len(loader), 1)
    acc = correct / max(total, 1)
    return avg_loss, acc, np.array(all_probs), np.array(all_labels), np.array(all_preds)


def compute_metrics(probs, labels, preds):
    """Compute precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix."""
    metrics = {}
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    metrics["confusion_matrix"] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    metrics["precision"] = round(precision, 4)
    metrics["recall"] = round(recall, 4)
    metrics["f1"] = round(f1, 4)

    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        if len(np.unique(labels)) > 1:
            metrics["roc_auc"] = round(float(roc_auc_score(labels, probs)), 4)
            metrics["pr_auc"] = round(float(average_precision_score(labels, probs)), 4)
    except ImportError:
        sorted_idx = np.argsort(-probs)
        sorted_labels = labels[sorted_idx]
        cum_tp = np.cumsum(sorted_labels)
        cum_fp = np.cumsum(1 - sorted_labels)
        rec = cum_tp / max(np.sum(labels), 1)
        prec = cum_tp / (cum_tp + cum_fp + 1e-10)
        metrics["pr_auc"] = round(float(np.trapz(prec, rec)), 4)
    return metrics


def freeze_backbone(model):
    """Freeze all feature layers; only classifier head is trainable."""
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.pool.parameters():
        param.requires_grad = False
    logger.info("Backbone frozen. Trainable params: %d", count_parameters(model))


def unfreeze_last_n_blocks(model, n: int = 2):
    """Unfreeze the last N residual blocks in the feature extractor."""
    blocks = [m for m in model.features if hasattr(m, 'conv1')]
    for block in blocks[-n:]:
        for param in block.parameters():
            param.requires_grad = True
    logger.info("Unfroze last %d blocks. Trainable params: %d", n, count_parameters(model))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train / fine-tune ResNet-1D exoplanet classifier")
    parser.add_argument("--data", type=str, default=None,
                        help="CSV with features, or 'auto' to build from NASA archive")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained checkpoint to fine-tune from")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=1e-4,
                        help="LR for stage-2 (backbone unfreeze)")
    parser.add_argument("--freeze-epochs", type=int, default=5,
                        help="Epochs to train with frozen backbone before unfreezing")
    parser.add_argument("--unfreeze-blocks", type=int, default=2,
                        help="Number of residual blocks to unfreeze in stage 2")
    parser.add_argument("--out", type=str, default="models/checkpoints/resnet1d.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--amp", action="store_true", help="Use mixed-precision (AMP)")
    parser.add_argument("--max-per-class", type=int, default=250,
                        help="Max samples per class when auto-building dataset")
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Gradient accumulation steps (for small VRAM)")
    args = parser.parse_args()

    if not TORCH_AVAILABLE:
        logger.error("PyTorch is required. Install with: pip install torch")
        sys.exit(1)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Data ----
    if args.data == "auto":
        manifest = build_labeled_dataset_from_archive(max_per_class=args.max_per_class)
        if manifest is None:
            sys.exit(1)
        global_views, local_views, centroid_offsets, labels = load_csv_data(manifest)
    elif args.data is not None:
        path = Path(args.data)
        if path.suffix.lower() == ".csv":
            global_views, local_views, centroid_offsets, labels = load_csv_data(str(path))
        else:
            logger.error("--data must be a CSV file or 'auto'")
            sys.exit(1)
    else:
        logger.warning("No --data provided; using small synthetic dataset for demo.")
        n = 200
        global_views = [np.random.randn(2048).astype(np.float32) for _ in range(n)]
        local_views = [np.random.randn(256).astype(np.float32) for _ in range(n)]
        centroid_offsets = np.random.rand(n).astype(np.float32)
        labels = (np.random.rand(n) > 0.5).astype(np.int64)

    n_total = len(labels) if hasattr(labels, '__len__') else len(global_views)
    n_val = max(1, n_total // 5)
    n_train = n_total - n_val
    indices = np.random.permutation(n_total)
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    gv, lv = np.asarray(global_views, dtype=np.float32), np.asarray(local_views, dtype=np.float32)
    co_all = np.asarray(centroid_offsets, dtype=np.float32)
    lb = np.asarray(labels, dtype=np.int64)
    train_ds = LightCurveDataset(gv[train_idx], lv[train_idx], co_all[train_idx], lb[train_idx], augment=True)
    val_ds = LightCurveDataset(gv[val_idx], lv[val_idx], co_all[val_idx], lb[val_idx], augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    from device_util import get_device, get_amp_config
    device = get_device()
    logger.info("Device: %s", device)

    # ---- Model ----
    if args.pretrained:
        model = load_checkpoint(args.pretrained, device=device, strict=False)
        logger.info("Loaded pretrained checkpoint from %s", args.pretrained)
    else:
        model = ResNet1DClassifier(use_centroid=True).to(device)
    logger.info("Model parameters: %s", f"{count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    amp_cfg = get_amp_config(device)
    scaler = torch.GradScaler(device.type) if (args.amp and amp_cfg["enabled"] and amp_cfg["use_scaler"]) else None
    amp_device_type = amp_cfg["device_type"] if (args.amp and amp_cfg["enabled"]) else None

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    patience_counter = 0

    # ---- Stage 1: frozen backbone (if fine-tuning) ----
    stage1_epochs = args.freeze_epochs if args.pretrained else 0
    if stage1_epochs > 0:
        freeze_backbone(model)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
        )
        logger.info("=== Stage 1: frozen backbone for %d epochs ===", stage1_epochs)
        for epoch in range(stage1_epochs):
            t0 = _time.time()
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler, amp_device_type)
            val_loss, val_acc, probs, lbls, preds = evaluate(model, val_loader, criterion, device)
            m = compute_metrics(probs, lbls, preds)
            elapsed = _time.time() - t0
            logger.info(
                "S1 Epoch %d/%d [%.1fs] train_loss=%.4f train_acc=%.4f | "
                "val_loss=%.4f val_acc=%.4f F1=%.4f P=%.4f R=%.4f",
                epoch + 1, stage1_epochs, elapsed,
                train_loss, train_acc, val_loss, val_acc,
                m["f1"], m["precision"], m["recall"],
            )
            if m["f1"] > best_f1:
                best_f1 = m["f1"]
                _save_checkpoint(model, args.out, epoch, m)
                patience_counter = 0
            else:
                patience_counter += 1

    # ---- Stage 2: unfreeze last blocks ----
    if args.pretrained:
        unfreeze_last_n_blocks(model, args.unfreeze_blocks)
    for param in model.parameters():
        param.requires_grad = True
    lr2 = args.finetune_lr if args.pretrained else args.lr
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr2
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    remaining_epochs = args.epochs - stage1_epochs
    logger.info("=== Stage 2: full training for %d epochs (lr=%.1e) ===", remaining_epochs, lr2)
    patience_counter = 0

    for epoch in range(remaining_epochs):
        t0 = _time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler, amp_device_type)
        val_loss, val_acc, probs, lbls, preds = evaluate(model, val_loader, criterion, device)
        m = compute_metrics(probs, lbls, preds)
        scheduler.step(m["f1"])
        elapsed = _time.time() - t0
        extra = ""
        if "roc_auc" in m:
            extra = f" ROC-AUC={m['roc_auc']:.4f} PR-AUC={m['pr_auc']:.4f}"
        logger.info(
            "S2 Epoch %d/%d [%.1fs] train_loss=%.4f train_acc=%.4f | "
            "val_loss=%.4f val_acc=%.4f F1=%.4f P=%.4f R=%.4f%s",
            epoch + 1, remaining_epochs, elapsed,
            train_loss, train_acc, val_loss, val_acc,
            m["f1"], m["precision"], m["recall"], extra,
        )
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            _save_checkpoint(model, args.out, stage1_epochs + epoch, m)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch + 1, args.patience)
                break

    # ---- Final evaluation ----
    best_ckpt = torch.load(args.out, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model"])
    _, _, probs, lbls, preds = evaluate(model, val_loader, criterion, device)
    final_m = compute_metrics(probs, lbls, preds)
    logger.info("=== Final metrics (best checkpoint) ===")
    logger.info("  F1=%.4f  Precision=%.4f  Recall=%.4f", final_m["f1"], final_m["precision"], final_m["recall"])
    if "roc_auc" in final_m:
        logger.info("  ROC-AUC=%.4f  PR-AUC=%.4f", final_m["roc_auc"], final_m["pr_auc"])
    cm = final_m["confusion_matrix"]
    logger.info("  Confusion: TP=%d FP=%d FN=%d TN=%d", cm["tp"], cm["fp"], cm["fn"], cm["tn"])

    metrics_path = Path(args.out).with_suffix(".metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(final_m, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)


def _save_checkpoint(model, path, epoch, metrics):
    torch.save({
        "model": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
    }, path)
    logger.info("Saved best checkpoint to %s (F1=%.4f)", path, metrics["f1"])


if __name__ == "__main__":
    main()
