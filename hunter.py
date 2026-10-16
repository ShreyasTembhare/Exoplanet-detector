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


def _records_from_tic_file(path: str):
    """A TIC file is either one id per line or a CSV with catalog columns."""
    from pipeline.target_rank import TargetRecord, records_from_rows

    with open(path) as handle:
        raw = handle.read()
    lines = [line.strip() for line in raw.splitlines() if line.strip() and not line.strip().startswith("#")]
    if not lines:
        return []
    header = lines[0].lower()
    if "," in lines[0] and any(token in header for token in ("tic", "radius", "tmag", "crowdsap", "disposition")):
        import pandas as pd
        frame = pd.read_csv(path, comment="#")
        frame.columns = [str(c).strip().lower() for c in frame.columns]
        return records_from_rows(frame.to_dict(orient="records"))
    records = []
    seen = set()
    for line in lines:
        tic = _normalize_tic(line.split(",")[0].split()[0])
        if tic and tic not in seen:
            seen.add(tic)
            records.append(TargetRecord(tic_id=tic))
    return records


def _fetch_sector_records(sector: int, cap: int):
    """MAST/lightkurve target list. Columns are kept when the table has them.

    The list is not cut to ``cap`` until after ranking. ``cap`` only bounds
    the catalog pull so a sector query cannot be unbounded.
    """
    from pipeline.target_rank import TargetRecord, records_from_rows

    try:
        from astroquery.mast import Observations
        logger.info("Querying MAST for sector %d timeseries observations...", sector)
        obs = Observations.query_criteria(
            obs_collection="TESS",
            sequence_number=sector,
            dataproduct_type="timeseries",
        )
        if obs is not None and len(obs) > 0:
            rows = []
            take = min(len(obs), cap)
            for i in range(take):
                row = {name: obs[name][i] for name in obs.colnames}
                name = row.get("target_name", row.get(obs.colnames[0]))
                tic = _normalize_tic(str(name))
                if not tic or not tic.isdigit():
                    continue
                rows.append({
                    "tic_id": tic,
                    "ra": row.get("s_ra", float("nan")),
                    "dec": row.get("s_dec", float("nan")),
                    "tmag": row.get("t_mag", row.get("tess_mag", float("nan"))),
                })
            # Unique, preserving pull order for ties.
            seen = set()
            unique_rows = []
            for row in rows:
                if row["tic_id"] in seen:
                    continue
                seen.add(row["tic_id"])
                unique_rows.append(row)
            logger.info("MAST returned %d numeric TICs for sector %d", len(unique_rows), sector)
            return records_from_rows(unique_rows)
    except Exception as e:
        logger.warning("MAST query failed (%s); trying lightkurve fallback.", e)

    try:
        import lightkurve as lk
        search = lk.search_lightcurve("TESS", mission="TESS", sector=sector)
        if search is not None and len(search) > 0 and hasattr(search, "table"):
            col = "target_name" if "target_name" in search.table.colnames else search.table.colnames[0]
            tics = [_normalize_tic(str(n)) for n in search.table[col]]
            seen = set()
            records = []
            for tic in tics:
                if tic and tic.isdigit() and tic not in seen:
                    seen.add(tic)
                    records.append(TargetRecord(tic_id=tic))
                    if len(records) >= cap:
                        break
            return records
    except Exception as e:
        logger.warning("lightkurve fallback also failed (%s).", e)
    return []


def _attach_toi_dispositions(records):
    """Mark known CP/KP hosts and FP/FA using the TOI table, when it answers.

    This is one catalog query, before any light curve download. A failure
    leaves dispositions blank; ranking does not use sky position.
    """
    if not records:
        return records
    try:
        from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
        toi = NasaExoplanetArchive.query_criteria(
            table="toi",
            select="tid,tfopwg_disp,pl_orbper,pl_tranmid,st_rad,st_tmag",
        )
        frame = toi.to_pandas()
    except Exception as exc:
        logger.info("TOI table unavailable (%s); known-host queue stays empty.", exc)
        return records

    by_tic = {}
    for _, row in frame.iterrows():
        try:
            tic = str(int(row["tid"]))
        except (TypeError, ValueError):
            continue
        by_tic[tic] = row
    for rec in records:
        row = by_tic.get(rec.tic_norm)
        if row is None:
            continue
        disp = str(row.get("tfopwg_disp", "") or "").strip().upper()
        if disp and disp != "NAN":
            rec.disposition = disp
        for attr, column in (
            ("known_period", "pl_orbper"),
            ("known_epoch", "pl_tranmid"),
            ("radius", "st_rad"),
            ("tmag", "st_tmag"),
        ):
            try:
                value = float(row.get(column))
            except (TypeError, ValueError):
                continue
            if value == value:
                current = getattr(rec, attr)
                if attr in ("radius", "tmag") and current == current:
                    continue
                setattr(rec, attr, value)
    return records


def _get_ranked_queue(
    sector: int, limit: int, tic_list_path: str = None,
    period_min: float = 0.5, period_max: float = 15.0,
):
    """Best-N primary targets, then a masked second pass for known CP/KP hosts."""
    from pipeline.target_rank import rank_targets
    from pipeline.tic_enrich import enrich_records

    if tic_list_path and os.path.exists(tic_list_path):
        records = _records_from_tic_file(tic_list_path)
    else:
        # Pull a wider pool than ``limit`` so ranking, not MAST order, chooses.
        records = _fetch_sector_records(sector, cap=max(int(limit) * 20, int(limit)))
        if not records:
            logger.error("Could not fetch targets for sector %d. Provide --tic-list.", sector)
            from pipeline.target_rank import RankedQueue
            return RankedQueue(primary=[], second_pass=[])
        records = _attach_toi_dispositions(records)
    # Bare MAST TIC lists lack radius / n_sectors / CROWDSAP; enrich before rank.
    enrich_records(records)
    return rank_targets(
        records, limit=limit, period_min=period_min, period_max=period_max,
    )


def _get_sector_target_list(
    sector: int, limit: int, tic_list_path: str = None,
    period_min: float = 0.5, period_max: float = 15.0,
):
    """TIC ids in search order: best N by expected SNR, then known-host second pass.

    ``limit`` is the best N primary targets, not the first N MAST returned.
    """
    queue = _get_ranked_queue(sector, limit, tic_list_path, period_min, period_max)
    return [rec.tic_norm for rec in queue.search_order()]


# ---------------------------------------------------------------------------
# Candidate persistence
# ---------------------------------------------------------------------------


def _save_candidate_plot(time, flux, best_period, epoch, tic_id: str, prob_planet: float, candidate_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    period_ok = best_period is not None and best_period == best_period and best_period > 0
    epoch_ok = epoch is not None and epoch == epoch
    if period_ok and epoch_ok:
        phase = ((time - epoch) % best_period) / best_period
    else:
        phase = np.zeros_like(np.asarray(time, dtype=np.float64))
        best_period = 0.0 if not period_ok else best_period
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
    if prob_planet is None or prob_planet != prob_planet:
        prob_txt = "SDE (no classifier)"
    else:
        prob_txt = f"{float(prob_planet):.2%}"
    axes[1].set_title(f"Binned view | Prob: {prob_txt}")
    fig.suptitle(f"CANDIDATE: TIC {tic_id} | P={best_period:.4f} d", fontsize=13)
    fig.tight_layout()
    safe_tic = str(tic_id).replace(" ", "_").replace("/", "_")
    path = os.path.join(candidate_dir, f"TIC_{safe_tic}_p{best_period:.2f}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def status_for_probability(prob_planet, threshold: float) -> str:
    """SDE-pass stars stay candidates when there is no checkpoint."""
    if prob_planet is None:
        return "CANDIDATE_SDE"
    if float(prob_planet) >= float(threshold):
        return f"CANDIDATE ({float(prob_planet):.2f})"
    return "CLEARED"


def _save_candidate_evidence(
    candidate_dir: str, tic_id: str, best_period: float, prob_planet: float,
    bls_power: float, sde: float, centroid_offset: float,
    plot_path: str, vetting: dict, peaks: list, strategy_profile: str,
    epoch: float = None,
):
    safe_tic = str(tic_id).replace(" ", "_").replace("/", "_")
    vetting = dict(vetting or {})
    if epoch is not None and epoch == epoch:
        vetting.setdefault("transit_center_btjd", float(epoch))
    meta = {
        "tic_id": tic_id,
        "best_period": best_period,
        "epoch": None if epoch is None or epoch != epoch else float(epoch),
        "prob_planet": prob_planet,
        "bls_max_power": bls_power,
        "sde": sde,
        "centroid_offset": centroid_offset if centroid_offset == centroid_offset else None,
        "vetting": vetting,
        "hard_flags": vetting.get("hard_flags", []),
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
    """Batched inference with the same contract as ``train_classifier``.

    Two-tower checkpoints are called as
    ``TwoTowerResNet1D.forward(global, local, centroid_offset=...)`` on
    ``make_two_tower_inputs``. A legacy single-tower checkpoint still accepts
    the stacked two-channel tensor.
    """
    if model is None or not items:
        return [None] * len(items)
    try:
        import torch

        from models.resnet1d import TwoTowerResNet1D, make_two_channel, make_two_tower_inputs
    except ImportError:
        return [None] * len(items)

    two_tower = isinstance(model, TwoTowerResNet1D) or hasattr(model, "global_tower")
    cos = []
    if two_tower:
        globals_, locals_ = [], []
        for item in items:
            global_in, local_in = make_two_tower_inputs(item["global_view"], item["local_view"])
            globals_.append(global_in)
            locals_.append(local_in)
            co = item.get("centroid_offset")
            cos.append(float(co) if co == co and co is not None else 0.0)
        global_t = torch.from_numpy(np.stack(globals_, axis=0)).float().to(device)
        local_t = torch.from_numpy(np.stack(locals_, axis=0)).float().to(device)
        co_t = torch.tensor(cos, device=device, dtype=torch.float32)
        with torch.inference_mode():
            logits = model(global_t, local_t, centroid_offset=co_t)
            probs = torch.softmax(logits, dim=1)
        return [probs[i, 1].item() for i in range(len(items))]

    xs = []
    for item in items:
        xs.append(make_two_channel(item["global_view"], item["local_view"]))
        co = item.get("centroid_offset")
        cos.append(float(co) if co == co and co is not None else 0.0)
    stacked = torch.from_numpy(np.stack(xs, axis=0)).float().to(device)
    co_t = torch.tensor(cos, device=device, dtype=torch.float32)
    with torch.inference_mode():
        logits = model(stacked, centroid_offset=co_t)
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
    backend=None, adaptive_duration=True, use_tls=True,
):
    """Core hunt logic, callable without CLI arg parsing."""
    candidates_found = []
    _ensure_setup(log_file, candidate_dir)
    progress = _load_progress(log_file)
    logger.info("Loaded progress for %d TICs from %s", len(progress), log_file)

    ranked = _get_ranked_queue(sector, limit, tic_list, period_min, period_max)
    target_list = ranked.search_order()
    if not target_list:
        logger.error("No targets for sector %s.", sector)
        return {"completed": 0, "candidates": candidates_found}
    logger.info(
        "Sector %s: %d primary + %d known-host second pass (limit %d means best N)",
        sector, len(ranked.primary), len(ranked.second_pass), limit,
    )

    model, device = _load_model(checkpoint)
    if model is None:
        logger.warning(
            "No checkpoint at %s. SDE-pass stars will be saved as CANDIDATE_SDE with vetting JSON.",
            checkpoint,
        )

    from pipeline import (
        fetch_tpf_arrays,
        prefilter_target,
        run_phase1,
    )
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

            status_q = status_for_probability(prob_planet, threshold)
            if status_q == "CANDIDATE_SDE" or status_q.startswith("CANDIDATE ("):
                plot_path = _save_candidate_plot(
                    item["time_arr"], item["flux_arr"], best_period_q,
                    item["epoch"], tic_id_q, prob_planet, candidate_dir,
                )
                _save_candidate_evidence(
                    candidate_dir, tic_id_q, best_period_q, prob_planet,
                    item["max_power"] or 0.0, item["sde"], item["centroid_offset"] or 0.0,
                    plot_path, vetting=item["vetting"], peaks=item["peaks"],
                    strategy_profile=strategy_profile or "",
                    epoch=item["epoch"],
                )
                candidates_found.append({
                    "tic_id": tic_id_q, "period": best_period_q,
                    "prob_planet": prob_planet, "sde": item["sde"],
                    "epoch": item["epoch"],
                    "strategy_profile": strategy_profile,
                    "status": status_q,
                })

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

    for i, target in enumerate(target_list):
        tic_raw = target.tic_norm
        tic_id = tic_raw if str(tic_raw).upper().startswith("TIC") else f"TIC {tic_raw}"
        tic_norm = _normalize_tic(tic_id)

        if _should_skip_stage(progress, tic_norm, "predict"):
            continue

        logger.info("[%s/%s] Processing %s...", i + 1, total, tic_id)
        time_arr, flux_arr = None, None
        best_period, epoch, max_power, sde_val = None, None, None, float("nan")
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

            # --- Phase 2–3: BLS, vet every SDE-pass peak, mask, MES ---
            t0 = _time.time()
            rho_star = derive_rho_star(meta) if adaptive_duration else 1.0
            tpf_data = None
            if with_tpf:
                tpf_data = fetch_tpf_arrays(tic_id, sector=sector, use_cache=True)
            tpf_time = tpf_data[0] if tpf_data else None
            tpf_flux_cube = tpf_data[1] if tpf_data else None
            known_period = target.known_period if target.queue == "second_pass" else None
            known_epoch = target.known_epoch if target.queue == "second_pass" else None
            from pipeline.planet_search import search_with_masking
            found = search_with_masking(
                time_arr, flux_arr,
                tic_id=tic_id, sector=str(sector_label),
                period_min=period_min, period_max=period_max, nperiods=nperiods,
                backend=backend, rho_star_solar=rho_star,
                adaptive_duration=adaptive_duration,
                sde_threshold=sde_threshold,
                use_cache=True, use_tls=use_tls, max_planets=3,
                tpf_time=tpf_time, tpf_flux_cube=tpf_flux_cube,
                ra=float(meta.get("ra", float("nan"))) if isinstance(meta, dict) else float("nan"),
                dec=float(meta.get("dec", float("nan"))) if isinstance(meta, dict) else float("nan"),
                known_period=known_period if known_period == known_period else None,
                known_epoch=known_epoch if known_epoch == known_epoch else None,
                run_triceratops=True,
            )
            timings["phase2"] = (_time.time() - t0) * 1000
            res = found.get("bls")
            signals = list(found.get("signals") or [])
            mono = found.get("monotransit")
            if res is None or (not signals and mono is None):
                sde_val = float(res.sde) if res is not None else float("nan")
                best_period = float(res.best_period) if res is not None else None
                status = f"NO_SIGNAL (SDE={sde_val:.2f})" if sde_val == sde_val else "NO_SIGNAL"
                if res is not None and float(np.max(res.power)) < bls_threshold:
                    status = f"NO_SIGNAL (SDE={sde_val:.2f})"
                _append_log(log_file, tic_norm, STAGE_DONE, status,
                            best_period=best_period, sde=sde_val)
                progress[tic_norm] = {"last_stage": STAGE_DONE, "status": status}
                logger.info("  -> %s", status)
                continue
            if mono is not None:
                signals.append({
                    "period": float("nan"),
                    "epoch": mono["epoch"],
                    "power": 0.0,
                    "sde": mono["mes"],
                    "vetting": {
                        "depth": mono["depth"],
                        "duration_hours": mono["duration_days"] * 24.0,
                        "n_transits": 1,
                        "transit_center_btjd": mono["epoch"],
                        "mes": mono["mes"],
                        "kind": "monotransit",
                    },
                    "hard_flags": ["n_transits"],
                    "global_view": np.zeros(2048),
                    "local_view": np.zeros(256),
                    "centroid_offset": float("nan"),
                    "kind": "monotransit",
                })
            peaks_list = [
                {"period": s["period"], "epoch": s["epoch"], "power": s["power"], "sde": s["sde"]}
                for s in signals
            ]
            for signal in signals:
                vetting_dict = dict(signal.get("vetting") or {})
                vetting_dict["hard_flags"] = list(signal.get("hard_flags") or [])
                if signal.get("diff_image"):
                    vetting_dict["diff_image"] = signal["diff_image"]
                if signal.get("triceratops"):
                    vetting_dict["triceratops"] = signal["triceratops"]
                best_period = signal["period"]
                epoch = signal["epoch"]
                sde_val = signal["sde"]
                max_power = signal.get("power") or 0.0
                predict_queue.append({
                    "tic_norm": tic_norm, "tic_id": tic_id,
                    "time_arr": time_arr, "flux_arr": flux_arr,
                    "best_period": best_period, "epoch": epoch,
                    "max_power": max_power, "sde": sde_val,
                    "global_view": signal.get("global_view"),
                    "local_view": signal.get("local_view"),
                    "centroid_offset": signal.get("centroid_offset"),
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
                        help="TLS on SDE-pass peaks (this is already the default).")
    parser.add_argument("--no-tls", action="store_true",
                        help="Do not run TLS, even on SDE-pass peaks.")
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
        use_tls=not args.no_tls,
    )


if __name__ == "__main__":
    main()
