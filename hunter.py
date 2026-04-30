#!/usr/bin/env python3
"""
TESS Hunter: autonomous "forever" pipeline.

Iterates over stars in a TESS sector, runs Phase 1 -> 2 (BLS) -> [SDE gate]
-> Phase 3 -> ResNet-1D, logs results to ``processed_stars*.txt``, saves
candidate plots + evidence JSON to ``candidates/``.

Improvements vs the previous version:
  * **SDE gating** instead of raw BLS power — invariant to cadence count and
    noise level; the standard Kepler/TESS pipeline cutoff is SDE > 7.
  * **TPF wired through** so centroid offset is real (not always NaN).
  * **flock-protected append** for the progress log instead of read-rewrite-rename
    (the old code was O(n^2) over star count).
  * **Prefilter** drops faint / contaminated / too-noisy stars before BLS.
  * Per-candidate JSON now stores the full vetting metrics dict.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time as _time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
SECTOR_TO_HUNT = 15
PROBABILITY_THRESHOLD = 0.85
SDE_THRESHOLD = 7.0
BLS_POWER_THRESHOLD = 0.0  # legacy raw-power gate; default off
LOG_FILE = "processed_stars.txt"
CANDIDATE_DIR = "candidates"
TARGET_LIMIT = 10000
DEFAULT_CHECKPOINT = "models/checkpoints/resnet1d.pt"

STAGE_DONE = "DONE"
STAGES = ["phase1", "phase2", "phase3", "predict", STAGE_DONE]


def _normalize_tic(tic_id: str) -> str:
    s = str(tic_id).strip().upper()
    if s.startswith("TIC"):
        s = s[3:].strip()
    return s


# ---------------------------------------------------------------------------
# Log: append under flock; no quadratic rewrites.
# ---------------------------------------------------------------------------

LOG_HEADER = "TIC_ID,LAST_STAGE,STATUS,BEST_PERIOD,SDE,UPDATED_AT,ERROR\n"


def _ensure_setup(log_file: str, candidate_dir: str):
    os.makedirs(candidate_dir, exist_ok=True)
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write(LOG_HEADER)


def _append_log(
    log_file: str, tic_id: str, last_stage: str, status: str,
    best_period=None, sde: float = float("nan"), error: str = "",
):
    """Append a row to the progress log under fcntl.flock."""
    bp_str = f"{best_period:.6f}" if (best_period is not None and best_period == best_period) else ""
    sde_str = f"{sde:.3f}" if (sde == sde and sde != float("inf")) else ""
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    error_clean = str(error).replace(",", ";").replace("\n", " ")[:200]
    line = f"{tic_id},{last_stage},{status},{bp_str},{sde_str},{ts},{error_clean}\n"

    try:
        import fcntl
        with open(log_file, "a") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.write(line)
                f.flush()
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception:
        with open(log_file, "a") as f:
            f.write(line)


def _is_nan(val):
    if val is None:
        return True
    try:
        import math
        return math.isnan(float(val))
    except (ValueError, TypeError):
        return False


def _load_progress(log_file: str) -> dict:
    """Load per-TIC progress: {normalized_tic: {last_stage, status, best_period, sde, error}}."""
    import pandas as pd
    if not os.path.exists(log_file):
        return {}
    try:
        df = pd.read_csv(log_file)
        if "TIC_ID" not in df.columns:
            return {}
        progress = {}
        for _, row in df.iterrows():
            tic = _normalize_tic(str(row["TIC_ID"]))
            progress[tic] = {
                "last_stage": str(row.get("LAST_STAGE", "")),
                "status": str(row.get("STATUS", "")),
                "best_period": float(row["BEST_PERIOD"]) if "BEST_PERIOD" in row and not _is_nan(row.get("BEST_PERIOD")) else None,
                "sde": float(row["SDE"]) if "SDE" in row and not _is_nan(row.get("SDE")) else float("nan"),
                "error": str(row.get("ERROR", "")) if not _is_nan(row.get("ERROR")) else "",
            }
        return progress
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Sector target listing
# ---------------------------------------------------------------------------


def _get_sector_target_list(sector: int, limit: int, tic_list_path: str = None):
    if tic_list_path and os.path.exists(tic_list_path):
        with open(tic_list_path) as f:
            lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
        tics = [_normalize_tic(t) for t in lines]
        return list(dict.fromkeys(tics))[:limit]

    try:
        from astroquery.mast import Observations
        logger.info("Querying MAST for sector %d timeseries observations...", sector)
        obs = Observations.query_criteria(
            obs_collection="TESS",
            sequence_number=sector,
            dataproduct_type="timeseries",
        )
        if obs is not None and len(obs) > 0:
            col = "target_name" if "target_name" in obs.colnames else obs.colnames[0]
            tics = [_normalize_tic(str(n)) for n in obs[col]]
            unique = list(dict.fromkeys(t for t in tics if t and t.isdigit()))
            logger.info("MAST returned %d numeric TICs for sector %d", len(unique), sector)
            return unique[:limit]
    except Exception as e:
        logger.warning("MAST query failed (%s); trying lightkurve fallback.", e)

    try:
        import lightkurve as lk
        search = lk.search_lightcurve("TESS", mission="TESS", sector=sector)
        if search is not None and len(search) > 0 and hasattr(search, "table"):
            col = "target_name" if "target_name" in search.table.colnames else search.table.colnames[0]
            tics = [_normalize_tic(str(n)) for n in search.table[col]]
            return list(dict.fromkeys(t for t in tics if t and t.isdigit()))[:limit]
    except Exception as e:
        logger.warning("lightkurve fallback also failed (%s).", e)

    logger.error("Could not fetch targets for sector %d. Provide --tic-list.", sector)
    return []


# ---------------------------------------------------------------------------
# Candidate persistence
# ---------------------------------------------------------------------------


def _save_candidate_plot(time, flux, best_period, epoch, tic_id: str, prob_planet: float, candidate_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    phase = ((time - epoch) % best_period) / best_period
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(phase, flux, s=1, alpha=0.7)
    axes[0].set_xlabel("Phase")
    axes[0].set_ylabel("Flux")
    axes[0].set_title(f"Folded scatter | TIC {tic_id}")
    n_bins = 200
    bins = np.linspace(0, 1, n_bins + 1)
    binned = np.array([
        np.nanmean(flux[(phase >= bins[j]) & (phase < bins[j + 1])])
        if np.sum((phase >= bins[j]) & (phase < bins[j + 1])) > 0 else np.nan
        for j in range(n_bins)
    ])
    axes[1].plot(np.linspace(0, 1, n_bins), binned, "k-", lw=1)
    axes[1].set_xlabel("Phase")
    axes[1].set_ylabel("Binned flux")
    axes[1].set_title(f"Binned view | Prob: {prob_planet:.2%}")
    fig.suptitle(f"CANDIDATE: TIC {tic_id} | P={best_period:.4f} d", fontsize=13)
    fig.tight_layout()
    safe_tic = str(tic_id).replace(" ", "_").replace("/", "_")
    path = os.path.join(candidate_dir, f"TIC_{safe_tic}_p{best_period:.2f}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _save_candidate_evidence(
    candidate_dir: str, tic_id: str, best_period: float, prob_planet: float,
    bls_power: float, sde: float, centroid_offset: float,
    plot_path: str, vetting: dict, peaks: list, strategy_profile: str,
):
    safe_tic = str(tic_id).replace(" ", "_").replace("/", "_")
    meta = {
        "tic_id": tic_id,
        "best_period": best_period,
        "prob_planet": prob_planet,
        "bls_max_power": bls_power,
        "sde": sde,
        "centroid_offset": centroid_offset if centroid_offset == centroid_offset else None,
        "vetting": vetting,
        "alt_peaks": peaks,
        "strategy_profile": strategy_profile,
        "plot": plot_path,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    json_path = os.path.join(candidate_dir, f"TIC_{safe_tic}_p{best_period:.2f}.json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def _load_model(checkpoint_path: str):
    try:
        from models.resnet1d import load_checkpoint
    except ImportError:
        return None, None
    path = Path(checkpoint_path)
    if not path.exists():
        logger.warning("Checkpoint not found: %s; candidates will not use AI.", path)
        return None, None
    from device_util import get_device
    device = get_device()
    try:
        model = load_checkpoint(checkpoint_path, device=device, strict=False)
        return model, device
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        return None, None


def _predict_batch(model, device, items):
    """Run batched inference. items: list of dicts with global_view, local_view, centroid_offset."""
    if model is None:
        return [None] * len(items)
    try:
        import torch

        from models.resnet1d import make_two_channel
    except ImportError:
        return [None] * len(items)
    xs, cos = [], []
    for item in items:
        gv, lv, co = item["global_view"], item["local_view"], item["centroid_offset"]
        co_val = float(co) if co == co and co is not None else 0.0
        xs.append(make_two_channel(gv, lv))
        cos.append(co_val)
    x = torch.from_numpy(np.stack(xs, axis=0)).float().to(device)
    co_t = torch.tensor(cos, device=device, dtype=torch.float32)
    with torch.inference_mode():
        logits = model(x, centroid_offset=co_t)
        probs = torch.softmax(logits, dim=1)
    return [probs[i, 1].item() for i in range(len(items))]


def _should_skip_stage(progress: dict, tic_norm: str, stage: str) -> bool:
    if tic_norm not in progress:
        return False
    last = progress[tic_norm].get("last_stage", "")
    if last == STAGE_DONE or last == "predict":
        return True
    try:
        last_idx = STAGES.index(last)
        this_idx = STAGES.index(stage)
        return this_idx <= last_idx
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Core hunt loop
# ---------------------------------------------------------------------------


def run_hunt(
    sector=SECTOR_TO_HUNT, limit=TARGET_LIMIT, threshold=PROBABILITY_THRESHOLD,
    bls_threshold=BLS_POWER_THRESHOLD, sde_threshold=SDE_THRESHOLD,
    checkpoint=DEFAULT_CHECKPOINT, log_file=LOG_FILE, candidate_dir=CANDIDATE_DIR,
    tic_list=None, infer_batch_size=32, period_min=0.5, period_max=15.0,
    nperiods=10000, strategy_profile=None, with_tpf=True, prefilter=True,
    backend=None, adaptive_duration=True, use_tls=False,
):
    """Core hunt logic, callable without CLI arg parsing."""
    candidates_found = []
    _ensure_setup(log_file, candidate_dir)
    progress = _load_progress(log_file)
    logger.info("Loaded progress for %d TICs from %s", len(progress), log_file)

    target_list = _get_sector_target_list(sector, limit, tic_list)
    if not target_list:
        logger.error("No targets for sector %s.", sector)
        return {"completed": 0, "candidates": candidates_found}
    logger.info("Sector %s: %d unique targets (limit %d)", sector, len(target_list), limit)

    model, device = _load_model(checkpoint)
    if model is None:
        logger.warning("Running without classifier; all SDE-pass stars will be logged as CLEARED.")

    from pipeline import (
        fetch_tpf_arrays,
        prefilter_target,
        run_phase1,
        run_phase3,
    )
    from pipeline.bls_gpu import compute_bls_periodogram
    from pipeline.physical_priors import derive_rho_star

    total = len(target_list)
    hunt_start = _time.time()
    completed_count = 0
    timings_csv = Path("hunter_timings.csv")
    if not timings_csv.exists():
        with open(timings_csv, "w") as f:
            f.write("TIC_ID,PHASE1_MS,PHASE2_MS,PHASE3_MS,PREDICT_MS,TOTAL_MS,SDE,STATUS\n")

    predict_queue = []

    def _flush_predict_queue():
        nonlocal completed_count
        if not predict_queue:
            return
        t0 = _time.time()
        probs = _predict_batch(model, device, predict_queue)
        batch_ms = (_time.time() - t0) * 1000
        per_item_ms = batch_ms / len(predict_queue) if predict_queue else 0

        for item, prob_planet in zip(predict_queue, probs):
            item["timings"]["predict"] = per_item_ms
            tic_norm_q = item["tic_norm"]
            tic_id_q = item["tic_id"]
            best_period_q = item["best_period"]

            if prob_planet is None:
                status_q = "CLEARED"
            elif prob_planet >= threshold:
                status_q = f"CANDIDATE ({prob_planet:.2f})"
                plot_path = _save_candidate_plot(
                    item["time_arr"], item["flux_arr"], best_period_q,
                    item["epoch"], tic_id_q, prob_planet, candidate_dir,
                )
                _save_candidate_evidence(
                    candidate_dir, tic_id_q, best_period_q, prob_planet,
                    item["max_power"] or 0.0, item["sde"], item["centroid_offset"] or 0.0,
                    plot_path, vetting=item["vetting"], peaks=item["peaks"],
                    strategy_profile=strategy_profile or "",
                )
                candidates_found.append({
                    "tic_id": tic_id_q, "period": best_period_q,
                    "prob_planet": prob_planet, "sde": item["sde"],
                    "strategy_profile": strategy_profile,
                })
            else:
                status_q = "CLEARED"

            total_ms = (_time.time() - item["tic_start"]) * 1000
            _append_log(log_file, tic_norm_q, STAGE_DONE, status_q,
                        best_period=best_period_q, sde=item["sde"])
            progress[tic_norm_q] = {"last_stage": STAGE_DONE, "status": status_q}
            t = item["timings"]
            logger.info("  -> %s [SDE=%.2f, %.0f ms total: p1=%.0f p2=%.0f p3=%.0f pred=%.0f]",
                        status_q, item["sde"], total_ms, t["phase1"], t["phase2"], t["phase3"], t["predict"])

            with open(timings_csv, "a") as f:
                f.write(f"{tic_norm_q},{t['phase1']:.0f},{t['phase2']:.0f},"
                        f"{t['phase3']:.0f},{t['predict']:.0f},{total_ms:.0f},"
                        f"{item['sde']:.3f},{status_q}\n")

            completed_count += 1
            elapsed_hours = (_time.time() - hunt_start) / 3600
            if elapsed_hours > 0:
                rate = completed_count / elapsed_hours
                logger.info("  Throughput: %.1f stars/hour (%d done in %.1f h)", rate, completed_count, elapsed_hours)

        predict_queue.clear()
        gc.collect()

    for i, tic_raw in enumerate(target_list):
        tic_id = tic_raw if tic_raw.upper().startswith("TIC") else f"TIC {tic_raw}"
        tic_norm = _normalize_tic(tic_id)

        if _should_skip_stage(progress, tic_norm, "predict"):
            continue

        logger.info("[%s/%s] Processing %s...", i + 1, total, tic_id)
        time_arr, flux_arr = None, None
        best_period, epoch, max_power, sde_val = None, None, None, float("nan")
        global_view, local_view, centroid_offset = None, None, None
        peaks_list: list = []
        vetting_dict: dict = {}
        timings = {"phase1": 0, "phase2": 0, "phase3": 0, "predict": 0}
        tic_start = _time.time()

        try:
            # --- Phase 1 ---
            t0 = _time.time()
            time_arr, flux_arr, _, sector_label, meta = run_phase1(
                tic_id, sector=sector, use_cache=True, with_tpf=with_tpf,
            )
            timings["phase1"] = (_time.time() - t0) * 1000
            if time_arr is None or len(time_arr) < 50:
                status = "ERROR: Empty or too short light curve"
                _append_log(log_file, tic_norm, "phase1", status, error=status)
                progress[tic_norm] = {"last_stage": STAGE_DONE, "status": status}
                continue

            # Prefilter
            if prefilter:
                pf = prefilter_target(time_arr, flux_arr, meta=meta)
                if not pf.passed:
                    status = f"PREFILTERED ({','.join(pf.reasons)})"
                    _append_log(log_file, tic_norm, STAGE_DONE, status)
                    progress[tic_norm] = {"last_stage": STAGE_DONE, "status": status}
                    logger.info("  -> %s", status)
                    continue

            # --- Phase 2 (BLS) ---
            t0 = _time.time()
            rho_star = derive_rho_star(meta) if adaptive_duration else 1.0
            # We bypass run_phase2's defaults here so we can pass backend +
            # adaptive duration; cache key is unchanged.
            from pipeline.cache_io import get_phase2, set_phase2
            res = None
            cached = get_phase2(tic_id, sector_label, period_min, period_max, nperiods)
            if cached is not None:
                periods, power, best_p, epoch_c, peaks_c = cached
                from pipeline.bls_gpu import BLSResult
                from pipeline.bls_gpu import compute_sde as _compute_sde
                sde_arr = _compute_sde(power)
                res = BLSResult(periods=periods, power=power, best_period=best_p,
                                epoch=epoch_c, sde=float(np.max(sde_arr)), peaks=peaks_c)
            else:
                res = compute_bls_periodogram(
                    time_arr, flux_arr,
                    period_min=period_min, period_max=period_max, nperiods=nperiods,
                    duration_phase=None if adaptive_duration else 0.05,
                    backend=backend, rho_star_solar=rho_star,
                )
                set_phase2(tic_id, sector_label, period_min, period_max, nperiods,
                           res.periods, res.power, res.best_period, res.epoch, peaks=res.peaks)
            timings["phase2"] = (_time.time() - t0) * 1000
            best_period = res.best_period
            epoch = res.epoch
            max_power = float(np.max(res.power))
            sde_val = res.sde
            peaks_list = [
                {"period": p.period, "epoch": p.epoch, "power": p.power, "sde": p.sde}
                for p in res.peaks
            ]

            # SDE gate (cadence-invariant) + legacy power gate (default off)
            if sde_val < sde_threshold or max_power < bls_threshold:
                status = f"NO_SIGNAL (SDE={sde_val:.2f})"
                _append_log(log_file, tic_norm, STAGE_DONE, status,
                            best_period=best_period, sde=sde_val)
                progress[tic_norm] = {"last_stage": STAGE_DONE, "status": status}
                logger.info("  -> %s (SDE=%.2f < %.2f)", status, sde_val, sde_threshold)
                continue

            # Optional TLS refinement on top peaks (slow but small-planet friendly)
            tls_refinements = []
            if use_tls:
                from pipeline import best_refinement, refine_peaks
                tls_refinements = refine_peaks(
                    time_arr, flux_arr, res.peaks, max_peaks=3,
                )
                br = best_refinement(tls_refinements)
                if br is not None:
                    logger.info("  TLS refined: P=%.4f, depth=%.5f, SDE=%.2f, SNR=%.2f",
                                br.period, br.depth, br.sde, br.snr)
                    if br.sde > sde_val:
                        # Use TLS period if it found a stronger signal.
                        best_period = br.period
                        epoch = br.epoch
                        sde_val = max(sde_val, br.sde)

            # --- Phase 3 + vetting ---
            t0 = _time.time()
            tpf_data = None
            if with_tpf:
                tpf_data = fetch_tpf_arrays(tic_id, sector=sector, use_cache=True)
            tpf_time = tpf_data[0] if tpf_data else None
            tpf_flux_cube = tpf_data[1] if tpf_data else None

            global_view, local_view, centroid_offset, extras = run_phase3(
                time_arr, flux_arr, best_period, epoch,
                tic_id=tic_id, sector=sector_label, use_cache=True,
                bls_sde=sde_val, bls_periods=res.periods, bls_power=res.power,
                tpf_time=tpf_time, tpf_flux_cube=tpf_flux_cube,
            )
            timings["phase3"] = (_time.time() - t0) * 1000
            vetting_dict = extras.get("vetting", {}) if isinstance(extras, dict) else {}

            predict_queue.append({
                "tic_norm": tic_norm, "tic_id": tic_id,
                "time_arr": time_arr, "flux_arr": flux_arr,
                "best_period": best_period, "epoch": epoch,
                "max_power": max_power, "sde": sde_val,
                "global_view": global_view, "local_view": local_view,
                "centroid_offset": centroid_offset,
                "peaks": peaks_list, "vetting": vetting_dict,
                "timings": timings, "tic_start": tic_start,
            })
            if len(predict_queue) >= infer_batch_size:
                _flush_predict_queue()

        except Exception as e:
            status = f"ERROR: {str(e)}"
            logger.exception("Failed for %s", tic_id)
            total_ms = (_time.time() - tic_start) * 1000
            _append_log(log_file, tic_norm, STAGE_DONE, status,
                        best_period=best_period, sde=sde_val)
            progress[tic_norm] = {"last_stage": STAGE_DONE, "status": status}
            logger.info("  -> %s [%.0f ms]", status, total_ms)
            with open(timings_csv, "a") as f:
                f.write(f"{tic_norm},{timings['phase1']:.0f},{timings['phase2']:.0f},"
                        f"{timings['phase3']:.0f},0,{total_ms:.0f},{sde_val:.3f},{status}\n")
            completed_count += 1

    _flush_predict_queue()
    logger.info("Hunter run complete. Processed %d stars.", completed_count)
    return {"completed": completed_count, "candidates": candidates_found}


def main():
    parser = argparse.ArgumentParser(description="TESS Hunter: autonomous sector pipeline")
    parser.add_argument("--sector", type=int, default=SECTOR_TO_HUNT)
    parser.add_argument("--limit", type=int, default=TARGET_LIMIT)
    parser.add_argument("--threshold", type=float, default=PROBABILITY_THRESHOLD)
    parser.add_argument("--bls-threshold", type=float, default=BLS_POWER_THRESHOLD,
                        help="Legacy raw-power gate (default 0 = disabled). Use --sde-threshold instead.")
    parser.add_argument("--sde-threshold", type=float, default=SDE_THRESHOLD,
                        help="Signal Detection Efficiency gate (default 7).")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--log-file", type=str, default=LOG_FILE)
    parser.add_argument("--candidate-dir", type=str, default=CANDIDATE_DIR)
    parser.add_argument("--tic-list", type=str, default=None)
    parser.add_argument("--infer-batch-size", type=int, default=32)
    parser.add_argument("--period-min", type=float, default=0.5)
    parser.add_argument("--period-max", type=float, default=15.0)
    parser.add_argument("--nperiods", type=int, default=10000)
    parser.add_argument("--strategy-profile", type=str, default=None)
    parser.add_argument("--no-tpf", action="store_true", help="Skip TPF download (faster but disables centroid vetting)")
    parser.add_argument("--no-prefilter", action="store_true", help="Skip cheap prefilter (CDPP/mag/contam)")
    parser.add_argument("--backend", type=str, default=None,
                        choices=[None, "jax", "numpy", "astropy", "cuvarbase"],
                        help="BLS backend (default: auto)")
    parser.add_argument("--no-adaptive-duration", action="store_true",
                        help="Disable Kepler's-3rd-law transit duration prior")
    parser.add_argument("--use-tls", action="store_true",
                        help="Run Transit Least Squares refinement on BLS top peaks")
    args = parser.parse_args()
    return run_hunt(
        sector=args.sector, limit=args.limit, threshold=args.threshold,
        bls_threshold=args.bls_threshold, sde_threshold=args.sde_threshold,
        checkpoint=args.checkpoint, log_file=args.log_file,
        candidate_dir=args.candidate_dir, tic_list=args.tic_list,
        infer_batch_size=args.infer_batch_size,
        period_min=args.period_min, period_max=args.period_max,
        nperiods=args.nperiods, strategy_profile=args.strategy_profile,
        with_tpf=not args.no_tpf, prefilter=not args.no_prefilter,
        backend=args.backend, adaptive_duration=not args.no_adaptive_duration,
        use_tls=args.use_tls,
    )


if __name__ == "__main__":
    main()
