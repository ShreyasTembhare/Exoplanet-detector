# TESS Exoplanet Detector (GPU BLS + ResNet-1D)

TESS-only exoplanet detection pipeline: Phase 1 (preprocess), Phase 2 (GPU BLS periodogram), Phase 3 (folded views + centroid vetting), Phase 4 (ResNet-1D classifier). Caching at each phase so reruns resume from the last cached stage.

## Architecture

- **Phase 1 (CPU):** TESS light curves via lightkurve; Savitzky-Golay flatten (`window_length=101`); 3-sigma sigma-clipping.
- **Phase 2 (GPU):** Two-pass BLS periodogram (coarse 2000 periods -> refine 3000 around peak) with JAX or NumPy fallback. Auto-selects backend based on cadence count. Optional downsampling for very long light curves.
- **Phase 3:** Fold on best period -> global view (2048 bins), local transit view (256 bins); centroid offset from TPF (optional).
- **Phase 4:** ResNet-1D classifier: two channels (global + local), centroid scalar; binary: Confirmed Planet vs False Positive. Supports loading pretrained checkpoints with automatic key mapping.

## Setup

```bash
pip install -r requirements.txt
```

Optional: Install JAX with GPU support for faster BLS (see [JAX install](https://github.com/google/jax#installation)).

## Usage

### Run pipeline (single target)

```bash
python run_pipeline.py "TIC 441462736"
python run_pipeline.py "TIC 441462736" --predict --checkpoint models/checkpoints/resnet1d.pt
```

Options: `--sector`, `--period-min`, `--period-max`, `--nperiods`, `--no-cache`, `--predict`, `--checkpoint`.

Reruns use cached Phase 1/2/3 results when available.

### Train / fine-tune classifier

**From scratch (synthetic demo):**
```bash
python train_classifier.py --epochs 30 --out models/checkpoints/resnet1d.pt
```

**Fine-tune from pretrained checkpoint:**
```bash
python train_classifier.py --pretrained models/checkpoints/pretrained.pt --epochs 30 --freeze-epochs 5 --unfreeze-blocks 2 --finetune-lr 1e-4 --amp
```

**Auto-build labeled dataset from NASA Exoplanet Archive:**
```bash
python train_classifier.py --data auto --max-per-class 250 --epochs 30 --amp
```

**From your own CSV:**
```bash
python train_classifier.py --data path/to/features.csv --epochs 30 --out models/checkpoints/resnet1d.pt
```

CSV can have columns `global_0`..`global_2047`, `local_0`..`local_255`, `centroid_offset`, `label` (0=FP, 1=planet), or a `path` column pointing to `.npz` files.

Training outputs:
- Best checkpoint: `models/checkpoints/resnet1d.pt`
- Metrics JSON: `models/checkpoints/resnet1d.metrics.json` (Precision, Recall, F1, ROC-AUC, PR-AUC, confusion matrix)

**Fine-tuning strategy:**
1. Stage 1 (freeze-epochs): backbone frozen, only classifier head trains at `--lr`.
2. Stage 2: last N residual blocks unfrozen at `--finetune-lr` with LR scheduler and early stopping.
3. Mixed precision (`--amp`) recommended for laptop GPUs with limited VRAM.

### TESS Hunter (autonomous pipeline)

Run a sector overnight: fetch stars, run Phase 1 -> BLS -> (pre-filter) -> Phase 3 -> ResNet-1D, log results, and save candidate plots + evidence JSON.

```bash
python hunter.py --sector 15 --limit 10000 --threshold 0.85
```

**Stage-level resume:** The hunter tracks progress per TIC at the phase level. If your laptop crashes or restarts, it resumes each TIC from the exact phase where it left off (not from scratch).

**Progress log** (`processed_stars.txt`):
```
TIC_ID,LAST_STAGE,STATUS,BEST_PERIOD,UPDATED_AT,ERROR
441462736,DONE,CLEARED,3.456789,2026-02-12T10:30:00+00:00,
```

Stages: `phase1` -> `phase2` -> `phase3` -> `predict` -> `DONE`.

**Timing telemetry:** Per-TIC phase durations and rolling throughput (stars/hour) are logged to console and `hunter_timings.csv`.

**Candidate evidence:** For each candidate, the hunter saves:
- Folded scatter plot + binned view: `candidates/TIC_<id>_p<period>.png`
- Sidecar JSON with model score, BLS power, centroid offset: `candidates/TIC_<id>_p<period>.json`

**Options:** `--sector`, `--limit`, `--threshold`, `--bls-threshold`, `--checkpoint`, `--log-file`, `--candidate-dir`, `--tic-list`.

### Python API

```python
from pipeline import run_phase1, run_phase2, run_phase3

time, flux, flux_err, sector = run_phase1("441462736", use_cache=True)
periods, power, best_period, epoch = run_phase2(time, flux, tic_id="441462736", sector=sector, use_cache=True)
global_view, local_view, centroid_offset = run_phase3(time, flux, best_period, epoch, tic_id="441462736", sector=sector, use_cache=True)
```

## Tuning Guide

| Parameter | Where | Default | Notes |
|-----------|-------|---------|-------|
| `--bls-threshold` | hunter.py | 0.001 | Lower = more stars pass to AI; calibrate from pilot run |
| `--threshold` | hunter.py | 0.85 | Probability cutoff for "CANDIDATE" |
| `coarse_nperiods` | bls_gpu.py | 2000 | More = slower coarse pass but less likely to miss peaks |
| `refine_nperiods` | bls_gpu.py | 3000 | More = finer period resolution |
| `downsample_limit` | bls_gpu.py | 80000 | Cadences above this are downsampled before BLS |
| `--freeze-epochs` | train_classifier.py | 5 | Epochs with frozen backbone during fine-tuning |
| `--patience` | train_classifier.py | 7 | Early stopping patience |

## Project layout

```
Exoplanet-detector/
├── pipeline/
│   ├── __init__.py
│   ├── cache_io.py          # Phase 1/2/3 get/set with atomic writes
│   ├── phase1_preprocess.py
│   ├── bls_gpu.py           # Two-pass BLS with auto backend policy
│   └── fold_features.py
├── models/
│   └── resnet1d.py          # ResNet-1D + pretrained checkpoint adapter
├── train_classifier.py      # Train / fine-tune with metrics
├── run_pipeline.py
├── hunter.py                # TESS Hunter with stage-level resume + telemetry
├── processed_stars.txt      # Hunter progress log (created on first run)
├── hunter_timings.csv       # Per-TIC timing telemetry (created on first run)
├── candidates/              # Candidate plots + evidence JSON
├── data/                    # Auto-built labeled datasets (created by --data auto)
├── requirements.txt
└── README.md
```

Cache directory: `cache/` (phase1, phase2, phase3). Add `cache/` to `.gitignore` (already included).

## Dependencies

- lightkurve, astropy, numpy, pandas, scipy
- jax, jaxlib (optional; NumPy fallback for BLS)
- torch (ResNet-1D)
- astroquery (optional, for auto-building labeled datasets)
- scikit-learn (optional, for ROC-AUC/PR-AUC metrics)
- matplotlib (candidate plots)
- tqdm

## References

- BLS: Kovacs, Zucker & Mazeh (2002); Astropy BoxLeastSquares.
- Centroid vetting: e.g. SSDataLab/vetting.
- NASA TESS / lightkurve.

## License

MIT (see LICENSE).
