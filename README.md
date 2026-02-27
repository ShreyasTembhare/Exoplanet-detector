# TESS Exoplanet Detector (GPU BLS + ResNet-1D)

Autonomous exoplanet detection pipeline for NASA TESS data. Preprocesses light curves, runs GPU-accelerated BLS periodograms, extracts phase-folded features, and classifies candidates with a ResNet-1D neural network. Includes a fully autonomous **autopilot** mode that trains a model, sweeps all TESS sectors, and cross-matches discoveries against known catalogs.

## Quick Start

```bash
uv venv && source .venv/bin/activate
uv pip install --native-tls -r requirements.txt

# --- Streamlit Control Center (recommended) ---
streamlit run app.py

# --- CLI (same functionality) ---
python run.py autopilot                           # fully autonomous
python run.py scan  "TIC 441462736"              # analyze one star
python run.py hunt  --sector 15 --limit 100       # sweep a sector
python run.py train --data auto --epochs 30       # train the classifier
```

Hardware is auto-detected: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU.

## Architecture

- **Phase 1 (CPU):** TESS light curves via lightkurve; Savitzky-Golay flatten (`window_length=101`); 3-sigma sigma-clipping.
- **Phase 2 (GPU):** Two-pass BLS periodogram (coarse 2000 periods -> refine 3000 around peak) with JAX or NumPy fallback. Auto-selects backend based on cadence count.
- **Phase 3:** Fold on best period -> global view (2048 bins), local transit view (256 bins); centroid offset from TPF.
- **Phase 4:** ResNet-1D classifier: two channels (global + local), centroid scalar; binary planet vs false positive.

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install --native-tls -r requirements.txt
```

Optional: Install JAX with GPU support for faster BLS (see [JAX install](https://github.com/google/jax#installation)).

## Usage

All commands go through `run.py`, which auto-detects hardware and prints a report at startup.

### Autopilot (fully autonomous)

Trains the model if needed, then sweeps all TESS sectors indefinitely. Survives restarts (resumes from where it left off). Cross-matches candidates against the NASA TOI list.

```bash
python run.py autopilot
python run.py autopilot --start-sector 15
python run.py autopilot --limit 100               # cap stars per sector (for testing)
```

New candidates are saved to `candidates/` with plots and evidence JSON. Each candidate is tagged as `KNOWN` (matches a TOI) or `NEW_CANDIDATE`.

### Single target

```bash
python run.py scan "TIC 441462736"
python run.py scan "TIC 441462736" --predict
```

### Sector sweep

```bash
python run.py hunt --sector 15 --limit 10000 --threshold 0.85
python run.py hunt --sector 15 --strategy-profile ultra_short_period
```

Stage-level resume: if your laptop crashes, each TIC resumes from its last completed phase.

### Train / fine-tune classifier

```bash
python run.py train --data auto --max-per-class 250 --epochs 30 --amp
python run.py train --pretrained models/checkpoints/pretrained.pt --epochs 30 --freeze-epochs 5
python run.py train --data path/to/features.csv --epochs 30
```

Training includes data augmentation (Gaussian noise, phase jitter, random masking, amplitude scaling) applied automatically to the training split.

### Python API

```python
from pipeline import run_phase1, run_phase2, run_phase3

time, flux, flux_err, sector = run_phase1("441462736", use_cache=True)
periods, power, best_period, epoch = run_phase2(time, flux, tic_id="441462736", sector=sector, use_cache=True)
global_view, local_view, centroid_offset = run_phase3(time, flux, best_period, epoch, tic_id="441462736", sector=sector, use_cache=True)
```

### Service layer (programmatic access)

```python
from services import run_scan, ScanConfig

result = run_scan(ScanConfig(tic_id="TIC 441462736", predict=True))
print(result["best_period"], result.get("prediction"))
```

## Strategy Profiles

Select a detection niche to tune BLS parameters, thresholds, and scoring heuristics. Available via `--strategy-profile` on CLI or the sidebar dropdown in the Streamlit dashboard.

| Profile | Period Range | BLS Threshold | Best For |
|---------|-------------|---------------|----------|
| `balanced` | 1 – 15 d | 0.001 | General-purpose search |
| `ultra_short_period` | 0.2 – 1 d | 0.002 | Hot, close-in planets |
| `single_transit_long_period` | 10 – 100 d | 0.0005 | Single/few-transit events |
| `low_snr_m_dwarf` | 0.5 – 20 d | 0.0003 | Faint M-dwarf hosts |

## Streamlit Dashboard

Launch the full control center:

```bash
streamlit run app.py
```

Tabs: **Scan** (single target), **Hunt** (sector sweep), **Train** (model training), **Autopilot** (autonomous multi-sector), **Candidates** (browse/rank/export), **Logs** (timing, state, task output).

Long-running tasks (Hunt, Train, Autopilot) run as background processes. Progress is polled from state files. All strategy profiles are selectable from the sidebar.

## Tuning Guide

| Parameter | Where | Default | Notes |
|-----------|-------|---------|-------|
| `--strategy-profile` | hunter / autopilot | balanced | Niche preset: balanced, ultra_short_period, single_transit_long_period, low_snr_m_dwarf |
| `--bls-threshold` | hunter / autopilot | 0.001 | Lower = more stars pass to AI (overridden by profile) |
| `--threshold` | hunter / autopilot | 0.85 | Probability cutoff for "CANDIDATE" (overridden by profile) |
| `coarse_nperiods` | bls_gpu.py | 2000 | More = slower coarse pass but fewer missed peaks |
| `refine_nperiods` | bls_gpu.py | 3000 | More = finer period resolution |
| `downsample_limit` | bls_gpu.py | 80000 | Cadences above this are downsampled before BLS |
| `--freeze-epochs` | train_classifier.py | 5 | Epochs with frozen backbone during fine-tuning |
| `--patience` | train_classifier.py | 7 | Early stopping patience |

## Project Layout

```
Exoplanet-detector/
├── app.py                   # Streamlit dashboard (streamlit run app.py)
├── services.py              # Orchestration service layer (typed configs + runners)
├── strategy_profiles.py     # Niche detection strategy profiles + scoring
├── run.py                   # Unified CLI launcher (scan / hunt / train / autopilot)
├── autopilot.py             # Multi-sector autonomous loop + TOI cross-matching
├── device_util.py           # Auto hardware detection (CUDA / MPS / CPU)
├── hunter.py                # Single-sector hunter with batched inference
├── run_pipeline.py          # Single-target pipeline runner
├── train_classifier.py      # Train / fine-tune with augmentation + metrics
├── pipeline/
│   ├── __init__.py
│   ├── cache_io.py          # Phase 1/2/3 cache with atomic writes
│   ├── phase1_preprocess.py
│   ├── bls_gpu.py           # Two-pass BLS with JAX/NumPy auto backend
│   └── fold_features.py
├── models/
│   └── resnet1d.py          # ResNet-1D classifier + checkpoint adapter
├── candidates/              # Candidate plots + evidence JSON (created at runtime)
├── requirements.txt
└── README.md
```

Runtime artifacts (not committed): `cache/`, `candidates/`, `autopilot_state.json`, `processed_stars*.txt`, `hunter_timings.csv`, `data/`.

## Dependencies

- lightkurve, astropy, numpy, pandas, scipy
- jax, jaxlib (optional; NumPy fallback for BLS)
- torch (ResNet-1D; supports CUDA and MPS)
- astroquery (MAST sector queries + NASA Exoplanet Archive)
- scikit-learn (ROC-AUC / PR-AUC metrics)
- matplotlib (candidate plots)
- streamlit (web dashboard)
- tqdm

## References

- BLS: Kovacs, Zucker & Mazeh (2002); Astropy BoxLeastSquares.
- Centroid vetting: e.g. SSDataLab/vetting.
- NASA TESS / lightkurve / MAST.

## License

MIT (see LICENSE).
