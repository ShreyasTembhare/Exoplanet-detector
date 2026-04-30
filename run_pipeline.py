#!/usr/bin/env python3
"""
Run TESS exoplanet pipeline: Phase 1 -> Phase 2 BLS -> Phase 3 fold + centroid + vetting.
Uses the cache at each step so reruns resume from the last completed stage.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run(
    tic_id: str,
    sector: int = None,
    period_min: float = 0.5,
    period_max: float = 20.0,
    nperiods: int = 5000,
    use_cache: bool = True,
    with_tpf: bool = True,
    backend: str = None,
):
    from pipeline import fetch_tpf_arrays, run_phase1, run_phase2, run_phase3

    logger.info(f"Running pipeline for TIC {tic_id} (sector={sector})")

    time, flux, _, sector_label, meta = run_phase1(
        tic_id, sector=sector, use_cache=use_cache, with_tpf=with_tpf,
    )
    logger.info(f"Phase 1 done: {len(time)} cadences, sector={sector_label}, cad={meta.get('cadence_minutes', '?'):.2f}min")

    res = run_phase2(
        time, flux, tic_id=tic_id, sector=sector_label,
        period_min=period_min, period_max=period_max, nperiods=nperiods,
        use_cache=use_cache, backend=backend, return_result=True,
    )
    logger.info(
        f"Phase 2 done: best_period={res.best_period:.4f} d, epoch={res.epoch:.4f}, "
        f"SDE={res.sde:.2f}, peaks={len(res.peaks)}"
    )

    tpf_data = None
    if with_tpf:
        tpf_data = fetch_tpf_arrays(tic_id, sector=sector, use_cache=use_cache)

    global_vec, local_vec, centroid_offset, extras = run_phase3(
        time, flux, res.best_period, res.epoch,
        tic_id=tic_id, sector=sector_label, use_cache=use_cache,
        bls_sde=res.sde, bls_periods=res.periods, bls_power=res.power,
        tpf_time=tpf_data[0] if tpf_data else None,
        tpf_flux_cube=tpf_data[1] if tpf_data else None,
    )
    logger.info(
        f"Phase 3 done: global={global_vec.shape}, local={local_vec.shape}, "
        f"centroid_offset={centroid_offset}"
    )
    if "vetting" in extras:
        v = extras["vetting"]
        logger.info(
            "  Vetting: depth=%.5f dur=%.2fh n_tx=%d odd_even=%.2f sec_sig=%.2f V/U=%.2f",
            v.get("depth", float("nan")),
            v.get("duration_hours", float("nan")),
            v.get("n_transits", 0),
            v.get("odd_even_ratio", float("nan")),
            v.get("secondary_significance", float("nan")),
            v.get("v_shape_score", float("nan")),
        )

    return {
        "tic_id": tic_id,
        "sector": sector_label,
        "time": time,
        "flux": flux,
        "periods": res.periods,
        "power": res.power,
        "best_period": res.best_period,
        "epoch": res.epoch,
        "sde": res.sde,
        "peaks": [
            {"period": p.period, "epoch": p.epoch, "power": p.power, "sde": p.sde}
            for p in res.peaks
        ],
        "global_view": global_vec,
        "local_view": local_vec,
        "centroid_offset": centroid_offset,
        "extras": extras,
        "meta": meta,
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
    parser.add_argument("--sector", type=int, default=None)
    parser.add_argument("--period-min", type=float, default=0.5)
    parser.add_argument("--period-max", type=float, default=20.0)
    parser.add_argument("--nperiods", type=int, default=5000)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-tpf", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--backend", type=str, default=None,
                        choices=[None, "jax", "numpy", "astropy", "cuvarbase"])
    args = parser.parse_args()

    result = run(
        args.tic,
        sector=args.sector,
        period_min=args.period_min,
        period_max=args.period_max,
        nperiods=args.nperiods,
        use_cache=not args.no_cache,
        with_tpf=not args.no_tpf,
        backend=args.backend,
    )
    logger.info(f"Best period: {result['best_period']:.4f} d  SDE: {result['sde']:.2f}")

    if args.predict:
        pred = predict(result, args.checkpoint)
        if pred is not None:
            logger.info(f"Prediction: class={pred['class']} (0=FP, 1=planet), prob_planet={pred['prob_planet']:.3f}")

    return result


if __name__ == "__main__":
    main()
