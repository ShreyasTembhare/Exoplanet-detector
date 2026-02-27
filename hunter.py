#!/usr/bin/env python3
"""
TESS Hunter: autonomous "forever" pipeline.
Iterates over stars in a TESS sector, runs Phase 1 -> 2 (BLS) -> [gate] -> Phase 3 -> ResNet-1D,
logs results to processed_stars.txt, saves candidate plots + evidence JSON to candidates/.

Stage-level resume: tracks LAST_STAGE per TIC so a restart continues from the
exact phase where it left off (not from scratch).
Timing telemetry: logs per-phase durations and rolling throughput.
Atomic writes: log and cache use temp-file + rename to prevent corruption.
"""

import argparse
import gc
import json
import logging
import os
import sys
import tempfile
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
BLS_THRESHOLD = 0.001
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


def _ensure_setup(log_file: str, candidate_dir: str):
    os.makedirs(candidate_dir, exist_ok=True)
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("TIC_ID,LAST_STAGE,STATUS,BEST_PERIOD,UPDATED_AT,ERROR\n")


def _load_progress(log_file: str) -> dict:
    """Load per-TIC progress: {normalized_tic: {last_stage, status, best_period, error}}."""
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
                "error": str(row.get("ERROR", "")) if not _is_nan(row.get("ERROR")) else "",
            }
        return progress
    except Exception:
        return {}


def _is_nan(val):
    if val is None:
        return True
    try:
        import math
        return math.isnan(float(val))
    except (ValueError, TypeError):
        return False


def _append_log_atomic(log_file: str, tic_id: str, last_stage: str, status: str,
                       best_period=None, error: str = ""):
    """Append a row to the progress log atomically."""
    bp_str = f"{best_period:.6f}" if best_period is not None else ""
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    error_clean = str(error).replace(",", ";").replace("\n", " ")[:200]
    line = f"{tic_id},{last_stage},{status},{bp_str},{ts},{error_clean}\n"

    fd, tmp = tempfile.mkstemp(suffix=".log.tmp", dir=os.path.dirname(log_file) or ".")
    try:
        os.close(fd)
        if os.path.exists(log_file):
            with open(log_file, "r") as src:
                content = src.read()
        else:
            content = "TIC_ID,LAST_STAGE,STATUS,BEST_PERIOD,UPDATED_AT,ERROR\n"
        with open(tmp, "w") as dst:
            dst.write(content)
            dst.write(line)
        os.replace(tmp, log_file)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        with open(log_file, "a") as f:
            f.write(line)


def _get_sector_target_list(sector: int, limit: int, tic_list_path: str = None):
    if tic_list_path and os.path.exists(tic_list_path):
        with open(tic_list_path) as f:
            lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
        tics = [_normalize_tic(t) for t in lines]
        return list(dict.fromkeys(tics))[:limit]

    # Primary: query MAST for all timeseries observations in this sector
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
            unique = list(dict.fromkeys(t for t in tics if t))
            logger.info("MAST returned %d unique TICs for sector %d", len(unique), sector)
            return unique[:limit]
    except Exception as e:
        logger.warning("MAST query failed (%s); trying lightkurve fallback.", e)

    # Fallback: lightkurve (requires a target, so try a wide-field search)
    try:
        import lightkurve as lk
        search = lk.search_lightcurve("TESS", mission="TESS", sector=sector)
        if search is not None and len(search) > 0 and hasattr(search, "table"):
            col = "target_name" if "target_name" in search.table.colnames else search.table.colnames[0]
            tics = [_normalize_tic(str(n)) for n in search.table[col]]
            return list(dict.fromkeys(t for t in tics if t))[:limit]
    except Exception as e:
        logger.warning("lightkurve fallback also failed (%s).", e)

    logger.error("Could not fetch targets for sector %d. Provide --tic-list.", sector)
    return []


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
    binned = np.array([np.nanmean(flux[(phase >= bins[j]) & (phase < bins[j + 1])]) if np.sum((phase >= bins[j]) & (phase < bins[j + 1])) > 0 else np.nan for j in range(n_bins)])
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


def _save_candidate_evidence(candidate_dir: str, tic_id: str, best_period: float,
                             prob_planet: float, bls_power: float, centroid_offset: float,
                             plot_path: str):
    """Write a sidecar JSON with model score, BLS power, and metadata."""
    safe_tic = str(tic_id).replace(" ", "_").replace("/", "_")
    meta = {
        "tic_id": tic_id,
        "best_period": best_period,
        "prob_planet": prob_planet,
        "bls_max_power": bls_power,
        "centroid_offset": centroid_offset if centroid_offset == centroid_offset else None,
        "plot": plot_path,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    json_path = os.path.join(candidate_dir, f"TIC_{safe_tic}_p{best_period:.2f}.json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)


def _load_model(checkpoint_path: str):
    try:
        import torch
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


def _predict(model, device, global_view, local_view, centroid_offset, checkpoint_path: str):
    if model is None:
        return None
    try:
        import torch
        from models.resnet1d import make_two_channel
    except ImportError:
        return None
    co_val = float(centroid_offset) if centroid_offset == centroid_offset and centroid_offset is not None else 0.0
    x = make_two_channel(global_view, local_view)
    x = torch.from_numpy(x).float().unsqueeze(0).to(device)
    co = torch.tensor([co_val], device=device, dtype=torch.float32)
    with torch.inference_mode():
        logits = model(x, centroid_offset=co)
        probs = torch.softmax(logits, dim=1)
    return probs[0, 1].item()


def _predict_batch(model, device, items):
    """
    Run batched inference on a list of dicts with keys
    'global_view', 'local_view', 'centroid_offset'.
    Returns list of prob_planet floats (or None per item if model is None).
    """
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
    co = torch.tensor(cos, device=device, dtype=torch.float32)
    with torch.inference_mode():
        logits = model(x, centroid_offset=co)
        probs = torch.softmax(logits, dim=1)
    return [probs[i, 1].item() for i in range(len(items))]


def _should_skip_stage(progress: dict, tic_norm: str, stage: str) -> bool:
    """Return True if this TIC has already completed this stage or a later one."""
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


def run_hunt(sector=SECTOR_TO_HUNT, limit=TARGET_LIMIT, threshold=PROBABILITY_THRESHOLD,
             bls_threshold=BLS_THRESHOLD, checkpoint=DEFAULT_CHECKPOINT,
             log_file=LOG_FILE, candidate_dir=CANDIDATE_DIR, tic_list=None,
             infer_batch_size=32, period_min=1.0, period_max=15.0, nperiods=10000,
             strategy_profile=None):
    """Core hunt logic, callable without CLI arg parsing.

    Returns dict with 'completed' count and 'candidates' list.
    """
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
        logger.warning("Running without classifier; all BLS-pass stars will be logged as CLEARED.")

    from pipeline import run_phase1, run_phase2, run_phase3

    total = len(target_list)
    hunt_start = _time.time()
    completed_count = 0
    timings_csv = Path("hunter_timings.csv")
    if not timings_csv.exists():
        with open(timings_csv, "w") as f:
            f.write("TIC_ID,PHASE1_MS,PHASE2_MS,PHASE3_MS,PREDICT_MS,TOTAL_MS,STATUS\n")

    predict_queue = []

    def _flush_predict_queue():
        """Run batched inference on queued items and finalize each star."""
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
                    item["max_power"] or 0.0, item["centroid_offset"] or 0.0, plot_path,
                )
                candidates_found.append({
                    "tic_id": tic_id_q, "period": best_period_q,
                    "prob_planet": prob_planet, "strategy_profile": strategy_profile,
                })
            else:
                status_q = "CLEARED"

            total_ms = (_time.time() - item["tic_start"]) * 1000
            _append_log_atomic(log_file, tic_norm_q, STAGE_DONE, status_q, best_period=best_period_q)
            progress[tic_norm_q] = {"last_stage": STAGE_DONE, "status": status_q}
            t = item["timings"]
            logger.info("  -> %s [%.0f ms total: p1=%.0f p2=%.0f p3=%.0f pred=%.0f]",
                         status_q, total_ms, t["phase1"], t["phase2"], t["phase3"], t["predict"])

            with open(timings_csv, "a") as f:
                f.write(f"{tic_norm_q},{t['phase1']:.0f},{t['phase2']:.0f},"
                        f"{t['phase3']:.0f},{t['predict']:.0f},{total_ms:.0f},{status_q}\n")

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
        best_period, epoch, max_power = None, None, None
        global_view, local_view, centroid_offset = None, None, None
        timings = {"phase1": 0, "phase2": 0, "phase3": 0, "predict": 0}
        tic_start = _time.time()

        try:
            # --- Phase 1 ---
            if not _should_skip_stage(progress, tic_norm, "phase1"):
                t0 = _time.time()
                time_arr, flux_arr, flux_err, sector_label = run_phase1(tic_id, sector=sector, use_cache=True)
                timings["phase1"] = (_time.time() - t0) * 1000
                if time_arr is None or len(time_arr) < 50:
                    status = "ERROR: Empty or too short light curve"
                    _append_log_atomic(log_file, tic_norm, "phase1", status, error=status)
                    progress[tic_norm] = {"last_stage": STAGE_DONE, "status": status}
                    continue
                _append_log_atomic(log_file, tic_norm, "phase1", "IN_PROGRESS")
                progress[tic_norm] = {"last_stage": "phase1", "status": "IN_PROGRESS"}
            else:
                time_arr, flux_arr, flux_err, sector_label = run_phase1(tic_id, sector=sector, use_cache=True)

            # --- Phase 2 (BLS) ---
            if not _should_skip_stage(progress, tic_norm, "phase2"):
                t0 = _time.time()
                periods, power, best_period, epoch = run_phase2(
                    time_arr, flux_arr, tic_id=tic_id, sector=sector_label,
                    period_min=period_min, period_max=period_max, nperiods=nperiods, use_cache=True,
                )
                timings["phase2"] = (_time.time() - t0) * 1000
                max_power = float(np.max(power))
                if max_power < bls_threshold:
                    status = "NO_SIGNAL"
                    _append_log_atomic(log_file, tic_norm, STAGE_DONE, status, best_period=best_period)
                    progress[tic_norm] = {"last_stage": STAGE_DONE, "status": status}
                    logger.info("  -> %s (BLS power=%.6f < %.6f)", status, max_power, bls_threshold)
                    continue
                _append_log_atomic(log_file, tic_norm, "phase2", "IN_PROGRESS", best_period=best_period)
                progress[tic_norm] = {"last_stage": "phase2", "status": "IN_PROGRESS", "best_period": best_period}
            else:
                cached_bp = progress.get(tic_norm, {}).get("best_period")
                periods, power, best_period, epoch = run_phase2(
                    time_arr, flux_arr, tic_id=tic_id, sector=sector_label,
                    period_min=period_min, period_max=period_max, nperiods=nperiods, use_cache=True,
                )
                max_power = float(np.max(power))

            # --- Phase 3 ---
            if not _should_skip_stage(progress, tic_norm, "phase3"):
                t0 = _time.time()
                global_view, local_view, centroid_offset = run_phase3(
                    time_arr, flux_arr, best_period, epoch,
                    tic_id=tic_id, sector=sector_label, tpf=None, use_cache=True,
                )
                timings["phase3"] = (_time.time() - t0) * 1000
                _append_log_atomic(log_file, tic_norm, "phase3", "IN_PROGRESS", best_period=best_period)
                progress[tic_norm] = {"last_stage": "phase3", "status": "IN_PROGRESS", "best_period": best_period}
            else:
                global_view, local_view, centroid_offset = run_phase3(
                    time_arr, flux_arr, best_period, epoch,
                    tic_id=tic_id, sector=sector_label, tpf=None, use_cache=True,
                )

            # --- Queue for batched prediction ---
            predict_queue.append({
                "tic_norm": tic_norm, "tic_id": tic_id,
                "time_arr": time_arr, "flux_arr": flux_arr,
                "best_period": best_period, "epoch": epoch,
                "max_power": max_power,
                "global_view": global_view, "local_view": local_view,
                "centroid_offset": centroid_offset,
                "timings": timings, "tic_start": tic_start,
            })
            if len(predict_queue) >= infer_batch_size:
                _flush_predict_queue()

        except Exception as e:
            status = f"ERROR: {str(e)}"
            logger.exception("Failed for %s", tic_id)
            total_ms = (_time.time() - tic_start) * 1000
            _append_log_atomic(log_file, tic_norm, STAGE_DONE, status, best_period=best_period)
            progress[tic_norm] = {"last_stage": STAGE_DONE, "status": status}
            logger.info("  -> %s [%.0f ms]", status, total_ms)
            with open(timings_csv, "a") as f:
                f.write(f"{tic_norm},{timings['phase1']:.0f},{timings['phase2']:.0f},"
                        f"{timings['phase3']:.0f},0,{total_ms:.0f},{status}\n")
            completed_count += 1

    _flush_predict_queue()
    logger.info("Hunter run complete. Processed %d stars.", completed_count)
    return {"completed": completed_count, "candidates": candidates_found}


def main():
    parser = argparse.ArgumentParser(description="TESS Hunter: autonomous sector pipeline with stage-level resume")
    parser.add_argument("--sector", type=int, default=SECTOR_TO_HUNT)
    parser.add_argument("--limit", type=int, default=TARGET_LIMIT)
    parser.add_argument("--threshold", type=float, default=PROBABILITY_THRESHOLD)
    parser.add_argument("--bls-threshold", type=float, default=BLS_THRESHOLD)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--log-file", type=str, default=LOG_FILE)
    parser.add_argument("--candidate-dir", type=str, default=CANDIDATE_DIR)
    parser.add_argument("--tic-list", type=str, default=None,
                        help="File with one TIC ID per line (fallback if sector search fails)")
    parser.add_argument("--infer-batch-size", type=int, default=32,
                        help="Batch size for AI inference (default 32)")
    parser.add_argument("--period-min", type=float, default=1.0)
    parser.add_argument("--period-max", type=float, default=15.0)
    parser.add_argument("--nperiods", type=int, default=10000)
    parser.add_argument("--strategy-profile", type=str, default=None)
    args = parser.parse_args()
    return run_hunt(
        sector=args.sector, limit=args.limit, threshold=args.threshold,
        bls_threshold=args.bls_threshold, checkpoint=args.checkpoint,
        log_file=args.log_file, candidate_dir=args.candidate_dir,
        tic_list=args.tic_list, infer_batch_size=args.infer_batch_size,
        period_min=args.period_min, period_max=args.period_max,
        nperiods=args.nperiods, strategy_profile=args.strategy_profile,
    )


if __name__ == "__main__":
    main()
