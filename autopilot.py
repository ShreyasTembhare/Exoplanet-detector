#!/usr/bin/env python3
"""
Autopilot: fully autonomous multi-sector exoplanet hunting.

1. Bootstraps a trained model if none exists (auto-builds dataset from NASA Archive).
2. Iterates through TESS sectors, running the hunter on each.
3. Cross-matches candidates against the NASA Exoplanet Archive TOI list.
4. Persists progress to autopilot_state.json so restarts resume cleanly.
5. Catches SIGINT for graceful shutdown.
"""

import argparse
import json
import logging
import os
import signal
import sys
import time as _time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = "models/checkpoints/resnet1d.pt"
STATE_FILE = "autopilot_state.json"
CANDIDATE_DIR = "candidates"
MAX_SECTOR = 100

_shutdown_requested = False


def _handle_sigint(signum, frame):
    """Signal-safe handler: only set a flag. Logging from inside a signal
    handler is not async-signal-safe and can deadlock the logging lock."""
    global _shutdown_requested
    if _shutdown_requested:
        sys.exit(1)
    _shutdown_requested = True


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def _load_state(state_file: str) -> dict:
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            return json.load(f)
    return {"completed_sectors": [], "current_sector": None}


def _save_state(state_file: str, state: dict):
    tmp = state_file + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, state_file)


# ---------------------------------------------------------------------------
# Bootstrap: train model if checkpoint missing
# ---------------------------------------------------------------------------

def _bootstrap_model(checkpoint: str, max_per_class: int = 250, epochs: int = 30):
    if os.path.exists(checkpoint):
        logger.info("Checkpoint found at %s — skipping training.", checkpoint)
        return True

    logger.info("=" * 60)
    logger.info("NO MODEL FOUND — bootstrapping from NASA Exoplanet Archive")
    logger.info("This downloads light curves and trains a classifier.")
    logger.info("It only happens once. Please be patient.")
    logger.info("=" * 60)

    from services import TrainConfig, run_train

    try:
        run_train(TrainConfig(
            data="auto",
            max_per_class=max_per_class,
            epochs=epochs,
            out=checkpoint,
            amp=True,
        ))
    except Exception as exc:
        logger.error("Bootstrap training failed: %s", exc)
        return False

    if os.path.exists(checkpoint):
        logger.info("Bootstrap complete. Checkpoint saved to %s", checkpoint)
        return True

    logger.error("Bootstrap training failed — no checkpoint produced.")
    return False


# ---------------------------------------------------------------------------
# Cross-match candidates against NASA TOI list
# ---------------------------------------------------------------------------

def _load_toi_tics():
    """Known TOI ephemerides from the NASA Exoplanet Archive, keyed by TIC."""
    from pipeline.known_planets import load_known_ephemerides

    return load_known_ephemerides()


def _cross_match_candidates(candidate_dir: str, known, since: float = 0.0):
    """Tag candidate JSONs written after ``since`` by orbit, not by host star.

    Returns ``(known_count, new_count)`` for those files only, so a sector
    summary does not re-count earlier sectors.
    """
    from pipeline.atomic import atomic_write_json
    from pipeline.known_planets import classify_candidate

    if not known:
        return 0, 0
    known_count, new_count = 0, 0
    for jf in Path(candidate_dir).glob("TIC_*.json"):
        try:
            if jf.stat().st_mtime < since:
                continue
            with open(jf, "r") as f:
                meta = json.load(f)
        except Exception:
            continue
        status, match = classify_candidate(
            meta.get("tic_id", ""), meta.get("best_period"), meta.get("epoch"), known,
        )
        meta["discovery_status"] = status
        meta["known_match"] = match
        if status == "KNOWN":
            known_count += 1
        else:
            new_count += 1
        atomic_write_json(jf, meta)
    return known_count, new_count


# ---------------------------------------------------------------------------
# Run hunter on a single sector
# ---------------------------------------------------------------------------

def _run_sector(sector: int, limit: int, checkpoint: str, threshold: float,
                bls_threshold: float, sde_threshold: float, candidate_dir: str,
                period_min: float = 0.5, period_max: float = 15.0,
                nperiods: int = 10000, strategy_profile: str = None,
                parallel: bool = False, download_workers: int = 8,
                cpu_workers: int = 4, bls_backend: str = None):
    """Run the hunter on a single sector via the callable hunter API.

    If ``parallel`` is True, dispatches to ``parallel_hunter.run_parallel_hunt``
    (producer-consumer pipeline with parallel MAST downloads and CPU folding
    workers); otherwise uses the sequential ``hunter.run_hunt``. Both write
    the same ``processed_stars_sNNN.txt`` (flock-protected) and share the
    Phase 1/2/3 cache, so resume keeps working across mode changes.
    """
    log_file = f"processed_stars_s{sector:03d}.txt"
    if strategy_profile:
        from strategy_profiles import apply_profile_to_hunt_kwargs

        tuned = apply_profile_to_hunt_kwargs(strategy_profile)
        period_min, period_max = tuned["period_min"], tuned["period_max"]
        nperiods, sde_threshold = tuned["nperiods"], tuned["sde_threshold"]
        threshold = tuned["threshold"]
    try:
        if parallel:
            from hunter import _get_ranked_queue
            from parallel_hunter import run_parallel_hunt
            ranked = _get_ranked_queue(sector, limit, None, period_min, period_max)
            targets = ranked.search_order()
            if not targets:
                logger.error("No targets for sector %d.", sector)
                return {"completed": 0, "candidates": [], "no_targets": True}
            return run_parallel_hunt(
                sector=sector, targets=targets,
                period_min=period_min, period_max=period_max, nperiods=nperiods,
                sde_threshold=sde_threshold, threshold=threshold,
                checkpoint=checkpoint, candidate_dir=candidate_dir,
                log_file=log_file,
                download_workers=download_workers, cpu_workers=cpu_workers,
                bls_backend=bls_backend, strategy_profile=strategy_profile or "",
            )
        from hunter import run_hunt
        return run_hunt(
            sector=sector, limit=limit, threshold=threshold,
            bls_threshold=bls_threshold, sde_threshold=sde_threshold,
            checkpoint=checkpoint,
            log_file=log_file, candidate_dir=candidate_dir,
            period_min=period_min, period_max=period_max,
            nperiods=nperiods, strategy_profile=strategy_profile,
        )
    except Exception as e:
        logger.error("Sector %d failed: %s", sector, e)
        return {"completed": 0, "candidates": [], "error": str(e)}


MAX_SECTOR_ATTEMPTS = 3


def sector_outcome(result: dict) -> str:
    """``done`` when the hunt ran; ``retry`` when it crashed or did no work."""
    if not isinstance(result, dict) or result.get("error"):
        return "retry"
    worked = int(result.get("completed", 0)) + int(result.get("skipped", 0))
    if worked == 0 and not result.get("no_targets"):
        return "retry"
    return "done"


# ---------------------------------------------------------------------------
# Main autopilot loop
# ---------------------------------------------------------------------------

def run_autopilot(start_sector=1, end_sector=MAX_SECTOR, limit=10000,
                  threshold=0.85, bls_threshold=0.0, sde_threshold=7.0,
                  checkpoint=DEFAULT_CHECKPOINT,
                  max_per_class=2500, train_epochs=30, candidate_dir=CANDIDATE_DIR,
                  state_file=STATE_FILE, strategy_profile=None,
                  period_min=0.5, period_max=15.0, nperiods=10000,
                  parallel=True, download_workers=8, cpu_workers=4,
                  bls_backend=None):
    """Core autopilot logic, callable without CLI arg parsing.

    Returns summary dict with sectors_done, total_candidates, elapsed_hours.
    """
    signal.signal(signal.SIGINT, _handle_sigint)

    from device_util import print_hw_report
    print_hw_report()

    if not _bootstrap_model(checkpoint, max_per_class, train_epochs):
        logger.warning(
            "No trained model; hunting anyway. SDE-pass stars are saved as "
            "CANDIDATE_SDE and ranked by the vetting score instead of a classifier.",
        )

    logger.info("Fetching known TOI list for cross-matching...")
    known_tics = _load_toi_tics()
    logger.info("Loaded %d known TOI TICs.", len(known_tics))

    state = _load_state(state_file)
    completed = set(state.get("completed_sectors", []))
    os.makedirs(candidate_dir, exist_ok=True)

    autopilot_start = _time.time()
    sectors_done_this_run = 0
    total_candidates = 0

    logger.info("=" * 60)
    logger.info("AUTOPILOT ENGAGED — sectors %d through %d", start_sector, end_sector)
    if parallel:
        logger.info("Mode: PARALLEL (download_workers=%d, cpu_workers=%d)",
                    download_workers, cpu_workers)
    else:
        logger.info("Mode: sequential")
    logger.info("=" * 60)

    for sector in range(start_sector, end_sector + 1):
        if _shutdown_requested:
            logger.info("Shutdown requested — stopping before sector %d.", sector)
            break

        if sector in completed or sector in set(state.get("failed_sectors", [])):
            logger.info("Sector %d already completed or parked — skipping.", sector)
            continue

        logger.info("")
        logger.info("=" * 60)
        logger.info("SECTOR %d", sector)
        logger.info("=" * 60)

        state["current_sector"] = sector
        _save_state(state_file, state)

        sector_start = _time.time()
        result = _run_sector(
            sector=sector, limit=limit, checkpoint=checkpoint,
            threshold=threshold, bls_threshold=bls_threshold,
            sde_threshold=sde_threshold,
            candidate_dir=candidate_dir,
            period_min=period_min, period_max=period_max,
            nperiods=nperiods, strategy_profile=strategy_profile,
            parallel=parallel,
            download_workers=download_workers,
            cpu_workers=cpu_workers,
            bls_backend=bls_backend,
        )
        sector_elapsed = _time.time() - sector_start

        known_count, new_count = _cross_match_candidates(
            candidate_dir, known_tics, since=sector_start - 1.0,
        )
        total_candidates += known_count + new_count

        attempts = state.setdefault("sector_attempts", {})
        if sector_outcome(result) == "done":
            completed.add(sector)
            attempts.pop(str(sector), None)
        else:
            attempts[str(sector)] = int(attempts.get(str(sector), 0)) + 1
            if attempts[str(sector)] >= MAX_SECTOR_ATTEMPTS:
                failed = set(state.get("failed_sectors", []))
                failed.add(sector)
                state["failed_sectors"] = sorted(failed)
                logger.error("Sector %d failed %d times; parking it in failed_sectors.",
                             sector, attempts[str(sector)])
            else:
                logger.warning("Sector %d did not finish (%s); it will be retried.",
                               sector, result.get("error", "no stars completed"))
        state["completed_sectors"] = sorted(completed)
        state["current_sector"] = None
        _save_state(state_file, state)

        sectors_done_this_run += 1
        total_elapsed_h = (_time.time() - autopilot_start) / 3600

        logger.info("")
        logger.info("--- Sector %d Summary ---", sector)
        logger.info("  Time: %.1f minutes", sector_elapsed / 60)
        logger.info("  Candidates this sector: %d (%d match known planets, %d new)",
                     known_count + new_count, known_count, new_count)
        if new_count > 0:
            logger.info("  *** %d NEW CANDIDATE(s) not in TOI list! Check candidates/ ***", new_count)
        logger.info("  Sectors completed this run: %d (%.1f hours elapsed)",
                     sectors_done_this_run, total_elapsed_h)
        logger.info("-" * 40)

    elapsed_hours = (_time.time() - autopilot_start) / 3600
    logger.info("")
    logger.info("=" * 60)
    logger.info("AUTOPILOT COMPLETE")
    logger.info("  Sectors processed this run: %d", sectors_done_this_run)
    logger.info("  Total candidates found: %d", total_candidates)
    logger.info("  Total time: %.1f hours", elapsed_hours)
    logger.info("=" * 60)

    return {
        "sectors_done": sectors_done_this_run,
        "total_candidates": total_candidates,
        "elapsed_hours": round(elapsed_hours, 2),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Autopilot: autonomous multi-sector exoplanet hunter"
    )
    parser.add_argument("--start-sector", type=int, default=1,
                        help="First sector to process (default: 1)")
    parser.add_argument("--end-sector", type=int, default=MAX_SECTOR,
                        help="Last sector to try (default: 100)")
    parser.add_argument("--limit", type=int, default=10000,
                        help="Max stars per sector (default: 10000)")
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="AI probability threshold for candidates")
    parser.add_argument("--bls-threshold", type=float, default=0.0,
                        help="Legacy raw-power gate (default 0 = off).")
    parser.add_argument("--sde-threshold", type=float, default=7.0,
                        help="Signal Detection Efficiency gate (default 7).")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--max-per-class", type=int, default=2500,
                        help="Training samples per class during bootstrap")
    parser.add_argument("--train-epochs", type=int, default=30,
                        help="Training epochs during bootstrap")
    parser.add_argument("--candidate-dir", type=str, default=CANDIDATE_DIR)
    parser.add_argument("--state-file", type=str, default=STATE_FILE)
    parser.add_argument("--strategy-profile", type=str, default=None)
    parser.add_argument("--period-min", type=float, default=0.5)
    parser.add_argument("--period-max", type=float, default=15.0)
    parser.add_argument("--nperiods", type=int, default=10000)
    parser.add_argument("--parallel", action="store_true",
                        help="Use the parallel hunter (this is the default).")
    parser.add_argument("--sequential", action="store_true",
                        help="Use sequential hunter.py for debugging.")
    parser.add_argument("--download-workers", type=int, default=8,
                        help="Parallel MAST download threads (only with --parallel).")
    parser.add_argument("--cpu-workers", type=int, default=4,
                        help="CPU folding/vetting workers (only with --parallel).")
    parser.add_argument("--bls-backend", type=str, default=None,
                        choices=[None, "jax", "numpy", "astropy", "cuvarbase"],
                        help="BLS backend (only with --parallel; default: auto).")
    args = parser.parse_args()
    return run_autopilot(
        start_sector=args.start_sector, end_sector=args.end_sector,
        limit=args.limit, threshold=args.threshold,
        bls_threshold=args.bls_threshold, sde_threshold=args.sde_threshold,
        checkpoint=args.checkpoint,
        max_per_class=args.max_per_class, train_epochs=args.train_epochs,
        candidate_dir=args.candidate_dir, state_file=args.state_file,
        strategy_profile=args.strategy_profile,
        period_min=args.period_min, period_max=args.period_max,
        nperiods=args.nperiods,
        parallel=not args.sequential,
        download_workers=args.download_workers,
        cpu_workers=args.cpu_workers,
        bls_backend=args.bls_backend,
    )


if __name__ == "__main__":
    main()
