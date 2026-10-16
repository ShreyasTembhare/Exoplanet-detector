"""
Orchestration service layer for the TESS Exoplanet Detector.

Typed configs and callables shared by the CLI (run.py) and the Transit
desk. The desk calls these functions in-process. It does not shell out.
"""

import logging
from dataclasses import dataclass
from typing import Optional

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
    with_tpf: bool = True
    predict: bool = False
    checkpoint: Optional[str] = None
    backend: Optional[str] = None  # jax|numpy|astropy|cuvarbase


@dataclass
class HuntConfig:
    sector: int = 15
    limit: int = 10000
    threshold: float = 0.85
    bls_threshold: float = 0.0
    sde_threshold: float = 7.0
    checkpoint: str = "models/checkpoints/resnet1d.pt"
    log_file: str = "processed_stars.txt"
    candidate_dir: str = "candidates"
    tic_list: Optional[str] = None
    infer_batch_size: int = 32
    period_min: float = 0.5
    period_max: float = 15.0
    nperiods: int = 10000
    strategy_profile: str = "balanced"
    with_tpf: bool = True
    prefilter: bool = True
    sequential: bool = False


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
    max_per_class: int = 2500
    grad_accum: int = 1
    use_focal: bool = True
    use_injection_recovery: bool = True
    val_holdout_sectors: Optional[str] = None  # e.g. "21-26"
    model_arch: str = "resnet1d"  # resnet1d | exominer | transformer


@dataclass
class AutopilotConfig:
    start_sector: int = 1
    end_sector: int = 100
    limit: int = 10000
    threshold: float = 0.85
    bls_threshold: float = 0.0
    sde_threshold: float = 7.0
    checkpoint: str = "models/checkpoints/resnet1d.pt"
    max_per_class: int = 2500
    train_epochs: int = 30
    candidate_dir: str = "candidates"
    state_file: str = "autopilot_state.json"
    strategy_profile: str = "balanced"
    period_min: float = 0.5
    period_max: float = 15.0
    nperiods: int = 10000
    parallel: bool = True
    download_workers: int = 8
    cpu_workers: int = 4
    bls_backend: Optional[str] = None


# ---------------------------------------------------------------------------
# Service functions
# ---------------------------------------------------------------------------

def run_scan(config: ScanConfig) -> dict:
    """Run the full pipeline on a single TIC target and optionally classify."""
    from run_pipeline import predict, run

    result = run(
        config.tic_id,
        sector=config.sector,
        period_min=config.period_min,
        period_max=config.period_max,
        nperiods=config.nperiods,
        use_cache=config.use_cache,
        with_tpf=config.with_tpf,
        backend=config.backend,
    )

    if config.predict:
        pred = predict(result, config.checkpoint)
        if pred is not None:
            result["prediction"] = pred

    return result


def run_hunt(config: HuntConfig) -> dict:
    """Run the sector hunter. Parallel search is the default."""
    from strategy_profiles import apply_profile_to_hunt_kwargs

    profile_kw = apply_profile_to_hunt_kwargs(config.strategy_profile)
    period_min = profile_kw.get("period_min", config.period_min)
    period_max = profile_kw.get("period_max", config.period_max)
    if config.sequential:
        from hunter import run_hunt as _hunt
        return _hunt(
            sector=config.sector,
            limit=config.limit,
            threshold=profile_kw.get("threshold", config.threshold),
            bls_threshold=profile_kw.get("bls_threshold", config.bls_threshold),
            sde_threshold=profile_kw.get("sde_threshold", config.sde_threshold),
            checkpoint=config.checkpoint,
            log_file=config.log_file,
            candidate_dir=config.candidate_dir,
            tic_list=config.tic_list,
            infer_batch_size=config.infer_batch_size,
            period_min=period_min,
            period_max=period_max,
            nperiods=profile_kw.get("nperiods", config.nperiods),
            strategy_profile=config.strategy_profile,
            with_tpf=config.with_tpf,
            prefilter=config.prefilter,
        )

    from hunter import _get_ranked_queue
    from parallel_hunter import run_parallel_hunt

    ranked = _get_ranked_queue(
        config.sector, config.limit, config.tic_list, period_min, period_max,
    )
    return run_parallel_hunt(
        sector=config.sector,
        targets=ranked.search_order(),
        period_min=period_min,
        period_max=period_max,
        nperiods=profile_kw.get("nperiods", config.nperiods),
        sde_threshold=profile_kw.get("sde_threshold", config.sde_threshold),
        threshold=profile_kw.get("threshold", config.threshold),
        checkpoint=config.checkpoint,
        candidate_dir=config.candidate_dir,
        log_file=config.log_file,
        with_tpf=config.with_tpf,
        prefilter=config.prefilter,
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
        use_focal=config.use_focal,
        use_injection_recovery=config.use_injection_recovery,
        val_holdout_sectors=config.val_holdout_sectors,
        model_arch=config.model_arch,
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
        sde_threshold=config.sde_threshold,
        checkpoint=config.checkpoint,
        max_per_class=config.max_per_class,
        train_epochs=config.train_epochs,
        candidate_dir=config.candidate_dir,
        state_file=config.state_file,
        strategy_profile=config.strategy_profile,
        period_min=config.period_min,
        period_max=config.period_max,
        nperiods=config.nperiods,
        parallel=config.parallel,
        download_workers=config.download_workers,
        cpu_workers=config.cpu_workers,
        bls_backend=config.bls_backend,
    )


# ---------------------------------------------------------------------------
# Candidate & state readers
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
    import json
    import os
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
    import json
    import os
    if not os.path.exists(metrics_path):
        return None
    with open(metrics_path) as f:
        return json.load(f)


def mast_requirement() -> dict:
    """Whether a sector search can reach MAST.

    A missing client is reported as a hard stop. The desk must not invent
    targets or candidates when the catalog cannot be queried.
    """
    import importlib.util

    missing = [name for name in ("lightkurve", "astroquery") if importlib.util.find_spec(name) is None]
    if missing:
        listed = " and ".join(missing) if len(missing) < 3 else ", ".join(missing)
        verb = "is" if len(missing) == 1 else "are"
        return {
            "available": False,
            "missing": missing,
            "message": (
                "This search needs MAST to list sector targets and download light curves. "
                f"{listed} {verb} not installed, so no light curves were fetched and no candidates were written."
            ),
        }
    return {
        "available": True,
        "missing": [],
        "message": "MAST clients are installed. A sector search downloads light curves.",
    }


def run_local_lightcurve(
    time,
    flux,
    tic_id: str,
    candidate_dir: str,
    *,
    period_min: float = 0.5,
    period_max: float = 15.0,
    nperiods: int = 800,
    sde_threshold: float = 7.0,
    disposition: str = "",
) -> dict:
    """BLS and vetting on an operator-supplied curve. No MAST and no classifier.

    A star is written only when the SDE gate passes. The result is labeled as
    a local light curve, not as a MAST discovery.
    """
    import os

    import numpy as np

    from pipeline import run_phase2, run_phase3
    from pipeline.hard_flags import hard_flag_reasons

    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    finite = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[finite], flux[finite]
    if len(time) < 30:
        raise ValueError("Need at least 30 finite time, flux samples.")

    os.makedirs(candidate_dir, exist_ok=True)
    res = run_phase2(
        time, flux, tic_id=str(tic_id), sector="local",
        period_min=period_min, period_max=period_max, nperiods=int(nperiods),
        use_cache=False, two_pass=True, return_result=True,
    )
    sde = float(res.sde) if res.sde == res.sde else float("nan")
    period = float(res.best_period) if res.best_period == res.best_period else float("nan")
    epoch = float(res.epoch) if res.epoch == res.epoch else float("nan")
    if not (sde == sde and sde >= float(sde_threshold) and period == period and period > 0 and epoch == epoch):
        return {
            "saved": False,
            "sde": None if sde != sde else sde,
            "period_days": None if period != period else period,
            "epoch_btjd": None if epoch != epoch else epoch,
            "message": (
                f"No SDE pass (SDE {sde:.2f} vs gate {float(sde_threshold):.1f}). "
                "No candidate was written."
            ) if sde == sde else "BLS did not return a finite SDE. No candidate was written.",
        }

    _global, _local, centroid, extras = run_phase3(
        time, flux, period, epoch,
        tic_id=str(tic_id), sector="local", use_cache=False,
        bls_sde=sde, bls_periods=res.periods, bls_power=res.power,
        skip_gaia=True,
    )
    vetting = dict(extras.get("vetting") or {})
    flags = hard_flag_reasons(vetting)
    vetting["hard_flags"] = flags
    peaks = [
        {
            "period": float(p.period),
            "epoch": float(p.epoch),
            "power": float(p.power),
            "sde": float(p.sde),
        }
        for p in (res.peaks or [])
    ]
    power = float(np.nanmax(res.power)) if len(res.power) else float("nan")
    from hunter import _save_candidate_evidence, _save_candidate_plot

    try:
        plot_path = _save_candidate_plot(time, flux, period, epoch, str(tic_id), None, candidate_dir)
    except Exception as exc:
        logger.warning("Candidate plot skipped: %s", exc)
        plot_path = ""
    _save_candidate_evidence(
        candidate_dir, str(tic_id), period, None, power, sde, centroid,
        plot_path, vetting, peaks, "balanced", epoch=epoch,
    )
    safe = str(tic_id).replace(" ", "_").replace("/", "_")
    json_path = os.path.join(candidate_dir, f"TIC_{safe}_p{period:.2f}.json")
    import json
    with open(json_path) as handle:
        payload = json.load(handle)
    disposition = (disposition or "").strip().upper()
    if disposition in {"CP", "KP"}:
        discovery = "KNOWN"
    else:
        discovery = "LOCAL_CURVE"
    payload.update({
        "source": "local_light_curve",
        "discovery_status": discovery,
        "status": "CANDIDATE_SDE",
        "phase_curve": _phase_curve(time, flux, period, epoch),
        "hard_flags": flags,
    })
    with open(json_path, "w") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return {
        "saved": True,
        "id": os.path.splitext(os.path.basename(json_path))[0],
        "sde": sde,
        "period_days": period,
        "epoch_btjd": epoch,
        "message": "SDE pass saved as a candidate. No classifier checkpoint was used.",
        "path": json_path,
    }


def _phase_curve(time, flux, period: float, epoch: float, n_bins: int = 64) -> dict:
    import numpy as np

    phase = ((np.asarray(time, dtype=np.float64) - epoch) % period) / period
    flux = np.asarray(flux, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers, values = [], []
    for i in range(n_bins):
        mask = (phase >= edges[i]) & (phase < edges[i + 1]) & np.isfinite(flux)
        if not np.any(mask):
            continue
        centers.append(float((edges[i] + edges[i + 1]) / 2.0))
        values.append(float(np.nanmedian(flux[mask])))
    return {"phase": centers, "flux": values}
