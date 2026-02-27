"""
Orchestration service layer for the TESS Exoplanet Detector.

Provides typed config dataclasses and callable entry-points so both
the CLI (run.py) and the Streamlit dashboard (app.py) share the same
execution path without sys.argv manipulation.
"""

from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ScanConfig:
    tic_id: str
    sector: Optional[int] = None
    period_min: float = 0.5
    period_max: float = 20.0
    nperiods: int = 5000
    use_cache: bool = True
    predict: bool = False
    checkpoint: Optional[str] = None


@dataclass
class HuntConfig:
    sector: int = 15
    limit: int = 10000
    threshold: float = 0.85
    bls_threshold: float = 0.001
    checkpoint: str = "models/checkpoints/resnet1d.pt"
    log_file: str = "processed_stars.txt"
    candidate_dir: str = "candidates"
    tic_list: Optional[str] = None
    infer_batch_size: int = 32
    period_min: float = 1.0
    period_max: float = 15.0
    nperiods: int = 10000
    strategy_profile: str = "balanced"


@dataclass
class TrainConfig:
    data: Optional[str] = None
    pretrained: Optional[str] = None
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    finetune_lr: float = 1e-4
    freeze_epochs: int = 5
    unfreeze_blocks: int = 2
    out: str = "models/checkpoints/resnet1d.pt"
    seed: int = 42
    patience: int = 7
    amp: bool = False
    max_per_class: int = 250
    grad_accum: int = 1


@dataclass
class AutopilotConfig:
    start_sector: int = 1
    end_sector: int = 100
    limit: int = 10000
    threshold: float = 0.85
    bls_threshold: float = 0.001
    checkpoint: str = "models/checkpoints/resnet1d.pt"
    max_per_class: int = 250
    train_epochs: int = 30
    candidate_dir: str = "candidates"
    state_file: str = "autopilot_state.json"
    strategy_profile: str = "balanced"
    period_min: float = 1.0
    period_max: float = 15.0
    nperiods: int = 10000


# ---------------------------------------------------------------------------
# Service functions
# ---------------------------------------------------------------------------

def run_scan(config: ScanConfig) -> dict:
    """Run the full pipeline on a single TIC target and optionally classify."""
    from run_pipeline import run, predict

    result = run(
        config.tic_id,
        sector=config.sector,
        period_min=config.period_min,
        period_max=config.period_max,
        nperiods=config.nperiods,
        use_cache=config.use_cache,
    )

    if config.predict:
        pred = predict(result, config.checkpoint)
        if pred is not None:
            result["prediction"] = pred

    return result


def run_hunt(config: HuntConfig) -> dict:
    """Run the sector hunter. Returns stats dict."""
    from strategy_profiles import apply_profile_to_hunt_kwargs
    from hunter import run_hunt as _hunt

    profile_kw = apply_profile_to_hunt_kwargs(config.strategy_profile)

    return _hunt(
        sector=config.sector,
        limit=config.limit,
        threshold=profile_kw.get("threshold", config.threshold),
        bls_threshold=profile_kw.get("bls_threshold", config.bls_threshold),
        checkpoint=config.checkpoint,
        log_file=config.log_file,
        candidate_dir=config.candidate_dir,
        tic_list=config.tic_list,
        infer_batch_size=config.infer_batch_size,
        period_min=profile_kw.get("period_min", config.period_min),
        period_max=profile_kw.get("period_max", config.period_max),
        nperiods=profile_kw.get("nperiods", config.nperiods),
        strategy_profile=config.strategy_profile,
    )


def run_train(config: TrainConfig) -> dict:
    """Train / fine-tune the classifier. Returns final metrics dict."""
    from train_classifier import run_train as _train

    return _train(
        data=config.data,
        pretrained=config.pretrained,
        epochs=config.epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        finetune_lr=config.finetune_lr,
        freeze_epochs=config.freeze_epochs,
        unfreeze_blocks=config.unfreeze_blocks,
        out=config.out,
        seed=config.seed,
        patience=config.patience,
        amp=config.amp,
        max_per_class=config.max_per_class,
        grad_accum=config.grad_accum,
    )


def run_autopilot(config: AutopilotConfig) -> dict:
    """Run multi-sector autopilot. Returns summary dict."""
    from autopilot import run_autopilot as _autopilot

    return _autopilot(
        start_sector=config.start_sector,
        end_sector=config.end_sector,
        limit=config.limit,
        threshold=config.threshold,
        bls_threshold=config.bls_threshold,
        checkpoint=config.checkpoint,
        max_per_class=config.max_per_class,
        train_epochs=config.train_epochs,
        candidate_dir=config.candidate_dir,
        state_file=config.state_file,
        strategy_profile=config.strategy_profile,
        period_min=config.period_min,
        period_max=config.period_max,
        nperiods=config.nperiods,
    )


# ---------------------------------------------------------------------------
# Candidate & state readers  (used by Streamlit dashboard)
# ---------------------------------------------------------------------------

def load_candidates(candidate_dir: str = "candidates") -> list:
    """Read all candidate JSON files and return as a list of dicts."""
    import json
    from pathlib import Path

    cdir = Path(candidate_dir)
    if not cdir.exists():
        return []
    candidates = []
    for jf in sorted(cdir.glob("TIC_*.json")):
        try:
            with open(jf) as f:
                meta = json.load(f)
            meta["_json_path"] = str(jf)
            png = jf.with_suffix(".png")
            if png.exists():
                meta["_plot_path"] = str(png)
            candidates.append(meta)
        except Exception:
            continue
    return candidates


def load_autopilot_state(state_file: str = "autopilot_state.json") -> dict:
    """Load persisted autopilot state."""
    import json, os
    if not os.path.exists(state_file):
        return {"completed_sectors": [], "current_sector": None}
    with open(state_file) as f:
        return json.load(f)


def load_hunt_progress(log_file: str = "processed_stars.txt"):
    """Load hunter progress CSV into a DataFrame."""
    import os
    try:
        import pandas as pd
    except ImportError:
        return None
    if not os.path.exists(log_file):
        return None
    try:
        return pd.read_csv(log_file)
    except Exception:
        return None


def load_training_metrics(metrics_path: str = "models/checkpoints/resnet1d.metrics.json") -> Optional[dict]:
    """Load saved training metrics JSON."""
    import json, os
    if not os.path.exists(metrics_path):
        return None
    with open(metrics_path) as f:
        return json.load(f)
