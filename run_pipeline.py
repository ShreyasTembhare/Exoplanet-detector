#!/usr/bin/env python3
"""
Run TESS exoplanet pipeline: Phase 1 (preprocess) -> Phase 2 (GPU BLS) -> Phase 3 (fold + centroid).
Uses cache at each step so reruns resume from the last cached stage.
Optional: load ResNet-1D and predict planet vs false positive.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run(tic_id: str, sector: int = None, period_min: float = 0.5, period_max: float = 20.0, nperiods: int = 5000, use_cache: bool = True):
    from pipeline import run_phase1, run_phase2, run_phase3

    logger.info(f"Running pipeline for TIC {tic_id} (sector={sector})")
    # Phase 1: TESS + flatten + 3σ
    time, flux, flux_err, sector_label = run_phase1(tic_id, sector=sector, use_cache=use_cache)
    logger.info(f"Phase 1 done: {len(time)} cadences, sector={sector_label}")

    # Phase 2: GPU BLS
    periods, power, best_period, epoch = run_phase2(
        time, flux, tic_id=tic_id, sector=sector_label,
        period_min=period_min, period_max=period_max, nperiods=nperiods,
        use_cache=use_cache,
    )
    logger.info(f"Phase 2 done: best_period={best_period:.4f} d, epoch={epoch:.4f}")

    # Phase 3: fold -> global 2048, local 256, centroid (no TPF here; centroid will be NaN)
    global_vec, local_vec, centroid_offset = run_phase3(
        time, flux, best_period, epoch,
        tic_id=tic_id, sector=sector_label, tpf=None, use_cache=use_cache,
    )
    logger.info(f"Phase 3 done: global_view={global_vec.shape}, local_view={local_vec.shape}, centroid_offset={centroid_offset}")

    return {
        "tic_id": tic_id,
        "sector": sector_label,
        "time": time,
        "flux": flux,
        "periods": periods,
        "power": power,
        "best_period": best_period,
        "epoch": epoch,
        "global_view": global_vec,
        "local_view": local_vec,
        "centroid_offset": centroid_offset,
    }


def predict(result: dict, checkpoint_path: str = None):
    """Run ResNet-1D on pipeline result. Returns class (0 or 1) and probability."""
    try:
        import torch
        from models.resnet1d import load_checkpoint, make_two_channel
    except ImportError:
        logger.warning("PyTorch or models not available; skipping prediction")
        return None
    if checkpoint_path is None:
        checkpoint_path = "models/checkpoints/resnet1d.pt"
    path = Path(checkpoint_path)
    if not path.exists():
        logger.warning(f"Checkpoint not found: {path}; skipping prediction")
        return None
    from device_util import get_device
    device = get_device()
    model = load_checkpoint(checkpoint_path, device=device, strict=False)
    x = make_two_channel(result["global_view"], result["local_view"])
    x = torch.from_numpy(x).float().unsqueeze(0).to(device)
    co_val = result["centroid_offset"]
    co_val = 0.0 if (co_val != co_val or co_val is None) else co_val
    co = torch.tensor([co_val], device=device)
    with torch.inference_mode():
        logits = model(x, centroid_offset=co)
        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1).item()
    return {"class": pred, "prob_planet": probs[0, 1].item(), "prob_fp": probs[0, 0].item()}


def main():
    parser = argparse.ArgumentParser(description="Run TESS exoplanet pipeline (Phase 1 -> 2 -> 3) with cache")
    parser.add_argument("tic", type=str, help="TIC ID (e.g. 441462736)")
    parser.add_argument("--sector", type=int, default=None, help="TESS sector (default: first available or stitched)")
    parser.add_argument("--period-min", type=float, default=0.5)
    parser.add_argument("--period-max", type=float, default=20.0)
    parser.add_argument("--nperiods", type=int, default=5000)
    parser.add_argument("--no-cache", action="store_true", help="Disable cache (run from scratch)")
    parser.add_argument("--predict", action="store_true", help="Run classifier if checkpoint exists")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    result = run(
        args.tic,
        sector=args.sector,
        period_min=args.period_min,
        period_max=args.period_max,
        nperiods=args.nperiods,
        use_cache=not args.no_cache,
    )
    logger.info(f"Best period: {result['best_period']:.4f} d")

    if args.predict:
        pred = predict(result, args.checkpoint)
        if pred is not None:
            logger.info(f"Prediction: class={pred['class']} (0=FP, 1=planet), prob_planet={pred['prob_planet']:.3f}")

    return result


if __name__ == "__main__":
    main()
