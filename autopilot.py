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
    global _shutdown_requested
    if _shutdown_requested:
        logger.warning("Second interrupt — forcing exit.")
        sys.exit(1)
    _shutdown_requested = True
    logger.info("Shutdown requested. Will finish current star and stop.")


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
    logger.info("This downloads ~500 light curves and trains a classifier.")
    logger.info("It only happens once. Please be patient.")
    logger.info("=" * 60)

    sys.argv = [
        "train_classifier.py",
        "--data", "auto",
        "--max-per-class", str(max_per_class),
        "--epochs", str(epochs),
        "--out", checkpoint,
        "--amp",
    ]

    try:
        from train_classifier import main as train_main
        train_main()
    except SystemExit:
        pass

    if os.path.exists(checkpoint):
        logger.info("Bootstrap complete. Checkpoint saved to %s", checkpoint)
        return True

    logger.error("Bootstrap training failed — no checkpoint produced.")
    return False


# ---------------------------------------------------------------------------
# Cross-match candidates against NASA TOI list
# ---------------------------------------------------------------------------

def _load_toi_tics() -> set:
    """Fetch known TOI TIC IDs from the NASA Exoplanet Archive."""
    try:
        from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
        toi_table = NasaExoplanetArchive.query_criteria(
            table="toi", select="tid",
        )
        return set(str(t) for t in toi_table["tid"] if t)
    except Exception as e:
        logger.warning("Could not fetch TOI list for cross-matching: %s", e)
        return set()


def _cross_match_candidates(candidate_dir: str, known_tics: set):
    """Tag each candidate JSON as KNOWN or NEW_CANDIDATE."""
    if not known_tics:
        return 0, 0
    known_count, new_count = 0, 0
    json_files = list(Path(candidate_dir).glob("TIC_*.json"))
    for jf in json_files:
        try:
            with open(jf, "r") as f:
                meta = json.load(f)
        except Exception:
            continue
        if "discovery_status" in meta:
            if meta["discovery_status"] == "KNOWN":
                known_count += 1
            else:
                new_count += 1
            continue
        tic_raw = str(meta.get("tic_id", ""))
        tic_norm = tic_raw.strip().upper()
        if tic_norm.startswith("TIC"):
            tic_norm = tic_norm[3:].strip()
        if tic_norm in known_tics:
            meta["discovery_status"] = "KNOWN"
            known_count += 1
        else:
            meta["discovery_status"] = "NEW_CANDIDATE"
            new_count += 1
        with open(jf, "w") as f:
            json.dump(meta, f, indent=2)
    return known_count, new_count


# ---------------------------------------------------------------------------
# Run hunter on a single sector
# ---------------------------------------------------------------------------

def _run_sector(sector: int, limit: int, checkpoint: str, threshold: float,
                bls_threshold: float, candidate_dir: str):
    """Run the hunter main() for a single sector by injecting sys.argv."""
    log_file = f"processed_stars_s{sector:03d}.txt"
    sys.argv = [
        "hunter.py",
        "--sector", str(sector),
        "--limit", str(limit),
        "--checkpoint", checkpoint,
        "--threshold", str(threshold),
        "--bls-threshold", str(bls_threshold),
        "--log-file", log_file,
        "--candidate-dir", candidate_dir,
    ]

    try:
        import importlib
        import hunter as hunter_mod
        importlib.reload(hunter_mod)
        hunter_mod.main()
    except SystemExit:
        pass
    except Exception as e:
        logger.error("Sector %d failed: %s", sector, e)


# ---------------------------------------------------------------------------
# Main autopilot loop
# ---------------------------------------------------------------------------

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
    parser.add_argument("--bls-threshold", type=float, default=0.001,
                        help="BLS power threshold to pass to AI")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--max-per-class", type=int, default=250,
                        help="Training samples per class during bootstrap")
    parser.add_argument("--train-epochs", type=int, default=30,
                        help="Training epochs during bootstrap")
    parser.add_argument("--candidate-dir", type=str, default=CANDIDATE_DIR)
    parser.add_argument("--state-file", type=str, default=STATE_FILE)
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_sigint)

    from device_util import print_hw_report
    print_hw_report()

    # --- Bootstrap model ---
    if not _bootstrap_model(args.checkpoint, args.max_per_class, args.train_epochs):
        logger.error("Cannot proceed without a trained model. Exiting.")
        sys.exit(1)

    # --- Load TOI list for cross-matching ---
    logger.info("Fetching known TOI list for cross-matching...")
    known_tics = _load_toi_tics()
    logger.info("Loaded %d known TOI TICs.", len(known_tics))

    # --- Sector loop ---
    state = _load_state(args.state_file)
    completed = set(state.get("completed_sectors", []))
    os.makedirs(args.candidate_dir, exist_ok=True)

    autopilot_start = _time.time()
    sectors_done_this_run = 0
    total_candidates = 0

    logger.info("=" * 60)
    logger.info("AUTOPILOT ENGAGED — sectors %d through %d", args.start_sector, args.end_sector)
    logger.info("=" * 60)

    for sector in range(args.start_sector, args.end_sector + 1):
        if _shutdown_requested:
            logger.info("Shutdown requested — stopping before sector %d.", sector)
            break

        if sector in completed:
            logger.info("Sector %d already completed — skipping.", sector)
            continue

        logger.info("")
        logger.info("=" * 60)
        logger.info("SECTOR %d", sector)
        logger.info("=" * 60)

        state["current_sector"] = sector
        _save_state(args.state_file, state)

        sector_start = _time.time()
        _run_sector(
            sector=sector,
            limit=args.limit,
            checkpoint=args.checkpoint,
            threshold=args.threshold,
            bls_threshold=args.bls_threshold,
            candidate_dir=args.candidate_dir,
        )
        sector_elapsed = _time.time() - sector_start

        # Cross-match candidates
        known_count, new_count = _cross_match_candidates(args.candidate_dir, known_tics)
        total_candidates += known_count + new_count

        # Mark sector done
        completed.add(sector)
        state["completed_sectors"] = sorted(completed)
        state["current_sector"] = None
        _save_state(args.state_file, state)

        sectors_done_this_run += 1
        total_elapsed_h = (_time.time() - autopilot_start) / 3600

        logger.info("")
        logger.info("--- Sector %d Summary ---", sector)
        logger.info("  Time: %.1f minutes", sector_elapsed / 60)
        logger.info("  Candidates (cumulative): %d total (%d known TOIs, %d new)",
                     known_count + new_count, known_count, new_count)
        if new_count > 0:
            logger.info("  *** %d NEW CANDIDATE(s) not in TOI list! Check candidates/ ***", new_count)
        logger.info("  Sectors completed this run: %d (%.1f hours elapsed)",
                     sectors_done_this_run, total_elapsed_h)
        logger.info("-" * 40)

    logger.info("")
    logger.info("=" * 60)
    logger.info("AUTOPILOT COMPLETE")
    logger.info("  Sectors processed this run: %d", sectors_done_this_run)
    logger.info("  Total candidates found: %d", total_candidates)
    logger.info("  Total time: %.1f hours", (_time.time() - autopilot_start) / 3600)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
