#!/usr/bin/env python3
"""
Parallel sector hunter.

Architecture:

::

    [TIC list] -> ThreadPool[N]      bounded Queue       ProcessPool[M]
                  download/preprocess  ---->  prefilter+BLS  ---->  fold+vet
                                       Queue                 Queue
                                                                      |
                                                                      v
                                                                ThreadPool[1]
                                                                 batched infer
                                                                      |
                                                                      v
                                                                 candidate save

Why threads vs processes?
  * **Phase 1 (download/preprocess)** is I/O-bound waiting on MAST. Use
    ``ThreadPoolExecutor`` so we can parallelize ~16 downloads per CPU core
    without the IPC tax of multiprocessing.
  * **Phase 2 (BLS)** is CPU/GPU bound. With JAX it's already vectorized on
    the GPU, so we only need one BLS worker; a process pool of 2-4 workers
    on CPU machines is the alternative.
  * **Phase 3 (folding/vetting)** is pure NumPy and benefits from a small
    process pool (``--cpu-workers``) to bypass the GIL.
  * **Phase 4 (inference)** stays in a single thread that drains a queue
    in batches sized for the GPU; the existing batched-inference path in
    ``hunter.py`` is unchanged.

Bounded ``queue.Queue`` between stages keeps RAM bounded. Telemetry: per-stage
queue depth and per-stage throughput are written to ``hunter_timings.csv``.
"""

from __future__ import annotations

import argparse
import collections
import gc
import logging
import queue
import sys
import threading
import time as _time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Sentinel for queue shutdown.
SENTINEL = object()


# ---------------------------------------------------------------------------
# Stage workers
# ---------------------------------------------------------------------------


def _phase1_worker(tic_id: str, sector: int, with_tpf: bool) -> Optional[dict]:
    """I/O-bound: download + preprocess. Runs in a thread."""
    try:
        from pipeline import fetch_tpf_arrays, run_phase1

        t0 = _time.time()
        time_arr, flux_arr, _, sector_label, meta = run_phase1(
            tic_id, sector=sector, use_cache=True, with_tpf=with_tpf,
        )
        if time_arr is None or len(time_arr) < 50:
            return None

        tpf_data = None
        if with_tpf:
            tpf_data = fetch_tpf_arrays(tic_id, sector=sector, use_cache=True)
        return {
            "tic_id": tic_id,
            "sector": sector_label,
            "time": time_arr,
            "flux": flux_arr,
            "meta": meta,
            "tpf_time": tpf_data[0] if tpf_data else None,
            "tpf_flux_cube": tpf_data[1] if tpf_data else None,
            "phase1_ms": (_time.time() - t0) * 1000,
        }
    except Exception as exc:
        logger.warning("Phase 1 failed for %s: %s", tic_id, exc)
        return None


def _phase2_worker(item: dict, period_min: float, period_max: float, nperiods: int,
                   sde_threshold: float, prefilter_on: bool, backend: Optional[str],
                   adaptive_duration: bool, use_tls: bool = True,
                   run_triceratops: bool = False) -> Optional[dict]:
    """Prefilter, cached BLS, vet SDE-pass peaks, mask-and-research, monotransit MES.

    The phase-2 cache is consulted first so a second pass over the same
    TIC / sector / grid does not recompute BLS. TLS runs only inside
    ``search_with_masking`` after the SDE gate.
    """
    try:
        from pipeline import prefilter_target
        from pipeline.physical_priors import derive_rho_star
        from pipeline.planet_search import search_with_masking

        time_arr = item["time"]
        flux_arr = item["flux"]
        meta = item["meta"] or {}

        if prefilter_on:
            pf = prefilter_target(time_arr, flux_arr, meta=meta)
            if not pf.passed:
                item["status"] = f"PREFILTERED ({','.join(pf.reasons)})"
                return item

        rho = derive_rho_star(meta) if adaptive_duration else 1.0
        t0 = _time.time()
        known_period = item.get("mask_period")
        known_epoch = item.get("mask_epoch")
        found = search_with_masking(
            time_arr, flux_arr,
            tic_id=item["tic_id"], sector=str(item["sector"]),
            period_min=period_min, period_max=period_max, nperiods=nperiods,
            backend=backend, rho_star_solar=rho,
            adaptive_duration=adaptive_duration,
            sde_threshold=sde_threshold,
            use_cache=known_period is None,
            use_tls=use_tls,
            tpf_time=item.get("tpf_time"),
            tpf_flux_cube=item.get("tpf_flux_cube"),
            ra=item.get("ra", float("nan")),
            dec=item.get("dec", float("nan")),
            known_period=known_period,
            known_epoch=known_epoch,
            run_triceratops=run_triceratops,
        )
        item["phase2_ms"] = (_time.time() - t0) * 1000
        item["bls"] = found.get("bls")
        item["signals"] = found.get("signals") or []
        item["monotransit"] = found.get("monotransit")
        if not item["signals"] and item["monotransit"] is None:
            sde = item["bls"].sde if item["bls"] is not None else float("nan")
            item["status"] = f"NO_SIGNAL (SDE={sde:.2f})" if sde == sde else "NO_SIGNAL"
        return item
    except Exception as exc:
        logger.warning("Phase 2 failed for %s: %s", item.get("tic_id"), exc)
        item["status"] = f"ERROR: {exc}"
        return item


def _phase3_worker(item: dict) -> Optional[dict]:
    """Folding already happened on each SDE-pass peak inside phase 2.

    This stage only stamps timing so the process-pool slot still exists for
    debugging. Signals and the monotransit statistic are already on ``item``.
    """
    if "status" in item and not item.get("signals") and not item.get("monotransit"):
        return item
    item["phase3_ms"] = item.get("phase3_ms", 0.0)
    return item


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_parallel_hunt(
    sector: int,
    tics: Optional[List[str]] = None,
    *,
    targets: Optional[list] = None,
    period_min: float = 0.5,
    period_max: float = 15.0,
    nperiods: int = 10000,
    sde_threshold: float = 7.0,
    threshold: float = 0.85,
    checkpoint: Optional[str] = "models/checkpoints/resnet1d.pt",
    candidate_dir: str = "candidates",
    log_file: Optional[str] = None,
    download_workers: int = 8,
    cpu_workers: int = 4,
    bls_backend: Optional[str] = None,
    with_tpf: bool = True,
    prefilter: bool = True,
    adaptive_duration: bool = True,
    infer_batch_size: int = 32,
    queue_depth: int = 32,
    use_tls: bool = True,
    strategy_profile: str = "",
    resume: bool = True,
    run_triceratops: bool = False,
) -> Dict[str, Any]:
    """Run a sector hunt with the producer-consumer pipeline above.

    With ``resume`` (default) stars already finished in ``log_file`` are
    skipped and errored stars are retried up to ``MAX_ERROR_ATTEMPTS`` times.
    """
    from hunter import (
        STAGE_DONE,
        _append_log,
        _ensure_setup,
        _load_model,
        _load_progress,
        _normalize_tic,
        _predict_batch,
        _save_candidate_evidence,
        _save_candidate_plot,
        _should_skip_stage,
        sky_coords,
        status_for_probability,
    )
    from pipeline.target_rank import TargetRecord

    if targets is None:
        targets = [TargetRecord(tic_id=str(tic)) for tic in (tics or [])]
    log_file = log_file or f"processed_stars_s{sector:03d}.txt"
    _ensure_setup(log_file, candidate_dir)
    skipped = 0
    if resume:
        progress = _load_progress(log_file)
        pending = [t for t in targets if not _should_skip_stage(progress, t.tic_norm, "predict")]
        skipped = len(targets) - len(pending)
        if skipped:
            logger.info("Resume: %d of %d stars already done in %s", skipped, len(targets), log_file)
        targets = pending

    model, device = _load_model(checkpoint) if checkpoint else (None, None)
    if model is None:
        logger.warning(
            "No checkpoint at %s. SDE-pass stars will be saved as CANDIDATE_SDE with vetting JSON.",
            checkpoint,
        )

    q_phase2 = queue.Queue(maxsize=queue_depth)
    q_phase3 = queue.Queue(maxsize=queue_depth)
    q_predict = queue.Queue(maxsize=queue_depth)

    candidates: List[dict] = []
    completed = 0
    hunt_start = _time.time()

    timings_path = Path("hunter_timings.csv")
    if not timings_path.exists():
        timings_path.write_text("TIC_ID,PHASE1_MS,PHASE2_MS,PHASE3_MS,PREDICT_MS,TOTAL_MS,SDE,STATUS\n")

    # --- Stage 1: download pool ---
    def _handle_download(fut, target):
        try:
            item = fut.result()
        except Exception as exc:
            logger.warning("download fail %s: %s", target.tic_norm, exc)
            item = None
        if item is None:
            _append_log(log_file, target.tic_norm, STAGE_DONE, "ERROR: phase1 failed")
            return
        if target.queue == "second_pass" and target.known_period == target.known_period:
            item["mask_period"] = float(target.known_period)
            item["mask_epoch"] = (
                float(target.known_epoch) if target.known_epoch == target.known_epoch else None
            )
        item["ra"], item["dec"] = sky_coords(target, item.get("meta"))
        item["tic_start"] = _time.time()
        q_phase2.put(item)

    def producer():
        # Bounded in-flight downloads: submitting the whole sector at once
        # spikes memory and MAST load while BLS is the real bottleneck.
        max_inflight = max(download_workers * 2, 1)
        with ThreadPoolExecutor(max_workers=download_workers) as ex:
            inflight: "collections.deque" = collections.deque()
            for target in targets:
                inflight.append((ex.submit(_phase1_worker, f"TIC {target.tic_norm}", sector, with_tpf), target))
                while len(inflight) >= max_inflight:
                    _handle_download(*inflight.popleft())
            while inflight:
                _handle_download(*inflight.popleft())
        q_phase2.put(SENTINEL)

    # --- Stage 2: BLS. One worker owns the GPU; CPU backends get several. ---
    from pipeline.bls_gpu import _select_backend

    gpu_backend = _select_backend(0, bls_backend) in ("jax", "cuvarbase")
    n_bls_workers = 1 if gpu_backend else max(int(cpu_workers), 1)
    bls_done = {"count": 0}
    bls_lock = threading.Lock()

    def bls_consumer():
        while True:
            item = q_phase2.get()
            if item is SENTINEL:
                q_phase2.put(SENTINEL)
                with bls_lock:
                    bls_done["count"] += 1
                    last = bls_done["count"] == n_bls_workers
                if last:
                    q_phase3.put(SENTINEL)
                return
            item = _phase2_worker(
                item, period_min, period_max, nperiods,
                sde_threshold, prefilter, bls_backend, adaptive_duration, use_tls,
                run_triceratops,
            )
            if "status" in item:
                _drain_failed(item, log_file, timings_path)
                continue
            q_phase3.put(item)

    # --- Stage 3: folding + vetting already ran per peak inside stage 2, so
    # this stage streams items through instead of pickling them to a pool.
    def fold_consumer():
        while True:
            item = q_phase3.get()
            if item is SENTINEL:
                break
            try:
                result_item = _phase3_worker(item)
            except Exception as exc:
                logger.warning("fold worker error: %s", exc)
                continue
            if "status" in result_item:
                _drain_failed(result_item, log_file, timings_path)
                continue
            q_predict.put(result_item)
        q_predict.put(SENTINEL)

    # --- Stage 4: batched inference on GPU ---
    def predict_consumer():
        nonlocal completed
        batch: List[dict] = []
        while True:
            item = q_predict.get()
            if item is SENTINEL:
                if batch:
                    _flush_batch(batch)
                return
            batch.append(item)
            if len(batch) >= infer_batch_size:
                _flush_batch(batch)
                batch = []

    def _rows_for_item(item: dict) -> List[dict]:
        signals = list(item.get("signals") or [])
        mono = item.get("monotransit")
        if mono is not None:
            signals.append({
                "period": float("nan"),
                "epoch": mono["epoch"],
                "sde": mono["mes"],
                "power": 0.0,
                "global_view": mono.get("global_view"),
                "local_view": mono.get("local_view"),
                "skip_classifier": True,
                "centroid_offset": float("nan"),
                "vetting": {
                    "depth": mono["depth"],
                    "duration_hours": mono["duration_days"] * 24.0,
                    "n_transits": 1,
                    "transit_center_btjd": mono["epoch"],
                    "mes": mono["mes"],
                    "kind": "monotransit",
                },
                "hard_flags": ["n_transits"],
                "kind": "monotransit",
            })
        rows = []
        for sig in signals:
            vetting = dict(sig.get("vetting") or {})
            vetting["hard_flags"] = list(sig.get("hard_flags") or [])
            rows.append({
                "global_view": sig.get("global_view"),
                "local_view": sig.get("local_view"),
                "skip_classifier": bool(sig.get("skip_classifier")),
                "centroid_offset": sig.get("centroid_offset"),
                "period": sig.get("period"),
                "epoch": sig.get("epoch"),
                "sde": sig.get("sde"),
                "power": sig.get("power") or 0.0,
                "vetting": vetting,
                "tic_id": item["tic_id"],
                "time": item["time"],
                "flux": item["flux"],
                "phase1_ms": item.get("phase1_ms", 0.0),
                "phase2_ms": item.get("phase2_ms", 0.0),
                "phase3_ms": item.get("phase3_ms", 0.0),
                "tic_start": item.get("tic_start", _time.time()),
                "peaks": [
                    {"period": s.get("period"), "epoch": s.get("epoch"),
                     "power": s.get("power"), "sde": s.get("sde")}
                    for s in signals
                ],
            })
        return rows

    def _flush_batch(batch: List[dict]):
        nonlocal completed
        rows = []
        for item in batch:
            rows.extend(_rows_for_item(item))
        if not rows:
            return
        predictable = [
            row for row in rows
            if row["global_view"] is not None and row["local_view"] is not None
            and not row.get("skip_classifier")
        ]
        t0 = _time.time()
        if model is None or not predictable:
            pred_probs = [None] * len(predictable)
        else:
            pred_probs = _predict_batch(model, device, predictable)
        per_item_ms = (_time.time() - t0) * 1000 / max(len(rows), 1)
        prob_by_id = {id(row): prob for row, prob in zip(predictable, pred_probs)}
        seen_tics = set()
        for row in rows:
            prob = prob_by_id.get(id(row))
            status = status_for_probability(prob, threshold)
            tic_norm = _normalize_tic(row["tic_id"])
            if status == "CANDIDATE_SDE" or status.startswith("CANDIDATE ("):
                plot_path = _save_candidate_plot(
                    row["time"], row["flux"], row["period"], row["epoch"],
                    row["tic_id"], prob, candidate_dir,
                )
                _save_candidate_evidence(
                    candidate_dir, row["tic_id"], row["period"] if row["period"] == row["period"] else 0.0,
                    prob, float(row["power"] or 0.0), float(row["sde"]),
                    row["centroid_offset"] if row["centroid_offset"] == row["centroid_offset"] else 0.0,
                    plot_path, vetting=row["vetting"], peaks=row["peaks"],
                    strategy_profile=strategy_profile or "balanced", epoch=row["epoch"],
                )
                candidates.append({
                    "tic_id": row["tic_id"], "period": row["period"],
                    "prob_planet": prob, "sde": row["sde"], "epoch": row["epoch"],
                    "status": status,
                })
            total_ms = (_time.time() - row["tic_start"]) * 1000
            _append_log(log_file, tic_norm, STAGE_DONE, status,
                        best_period=row["period"], sde=row["sde"])
            if tic_norm not in seen_tics:
                with open(timings_path, "a") as handle:
                    handle.write(
                        f"{tic_norm},{row['phase1_ms']:.0f},{row['phase2_ms']:.0f},"
                        f"{row['phase3_ms']:.0f},{per_item_ms:.0f},{total_ms:.0f},"
                        f"{float(row['sde']):.3f},{status}\n"
                    )
                seen_tics.add(tic_norm)
                completed += 1
        gc.collect()

    threads = [
        threading.Thread(target=producer, name="downloader"),
        *[threading.Thread(target=bls_consumer, name=f"bls-{i}") for i in range(n_bls_workers)],
        threading.Thread(target=fold_consumer, name="fold"),
        threading.Thread(target=predict_consumer, name="predict"),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    elapsed = _time.time() - hunt_start
    rate = completed / max(elapsed / 3600.0, 1e-6)
    logger.info("Parallel hunt complete: %d stars in %.1f min (%.0f stars/h, %d candidates)",
                completed, elapsed / 60, rate, len(candidates))
    return {
        "completed": completed, "skipped": skipped, "candidates": candidates,
        "elapsed_seconds": elapsed, "stars_per_hour": rate,
    }


def _drain_failed(item: dict, log_file: str, timings_path: Path):
    from hunter import STAGE_DONE, _append_log, _normalize_tic

    tic_norm = _normalize_tic(item.get("tic_id", ""))
    status = item.get("status", "ERROR")
    sde = float("nan")
    bp = None
    if "bls" in item and item["bls"] is not None:
        sde = item["bls"].sde
        bp = item["bls"].best_period
    _append_log(log_file, tic_norm, STAGE_DONE, status, best_period=bp, sde=sde)
    p1 = item.get("phase1_ms", 0)
    p2 = item.get("phase2_ms", 0)
    with open(timings_path, "a") as f:
        f.write(f"{tic_norm},{p1:.0f},{p2:.0f},0,0,{(p1 + p2):.0f},{sde:.3f},{status}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Parallel sector hunter")
    parser.add_argument("--sector", type=int, required=True)
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--tic-list", type=str, default=None)
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--cpu-workers", type=int, default=4)
    parser.add_argument("--queue-depth", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--sde-threshold", type=float, default=None)
    parser.add_argument("--period-min", type=float, default=None)
    parser.add_argument("--period-max", type=float, default=None)
    parser.add_argument("--nperiods", type=int, default=None)
    parser.add_argument("--strategy-profile", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/resnet1d.pt")
    parser.add_argument("--candidate-dir", type=str, default="candidates")
    parser.add_argument("--no-tpf", action="store_true")
    parser.add_argument("--no-prefilter", action="store_true")
    parser.add_argument("--no-adaptive-duration", action="store_true")
    parser.add_argument("--backend", type=str, default=None,
                        choices=[None, "jax", "numpy", "astropy", "cuvarbase"])
    args = parser.parse_args()

    from hunter import _get_ranked_queue
    from strategy_profiles import DEFAULT_PROFILE, apply_profile_to_hunt_kwargs

    search = apply_profile_to_hunt_kwargs(args.strategy_profile or DEFAULT_PROFILE, {
        "threshold": args.threshold, "sde_threshold": args.sde_threshold,
        "period_min": args.period_min, "period_max": args.period_max,
        "nperiods": args.nperiods,
    })
    ranked = _get_ranked_queue(
        args.sector, args.limit, args.tic_list, search["period_min"], search["period_max"],
    )
    targets = ranked.search_order()
    if not targets:
        logger.error("No targets for sector %d.", args.sector)
        sys.exit(1)
    return run_parallel_hunt(
        sector=args.sector,
        targets=targets,
        period_min=search["period_min"],
        period_max=search["period_max"],
        nperiods=search["nperiods"],
        sde_threshold=search["sde_threshold"],
        threshold=search["threshold"],
        strategy_profile=args.strategy_profile or DEFAULT_PROFILE,
        checkpoint=args.checkpoint,
        candidate_dir=args.candidate_dir,
        download_workers=args.download_workers,
        cpu_workers=args.cpu_workers,
        bls_backend=args.backend,
        with_tpf=not args.no_tpf,
        prefilter=not args.no_prefilter,
        adaptive_duration=not args.no_adaptive_duration,
        queue_depth=args.queue_depth,
    )


if __name__ == "__main__":
    main()
