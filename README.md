# TESS Exoplanet Detector

End-to-end TESS pipeline: download light curves, detrend, run BLS (and optionally TLS) periodograms, fold + vet candidates, classify with a two-tower ResNet-1D (and optionally an ExoMiner++-style multi-input vetter or a transformer detector for raw light curves), and export TFOP-compatible discovery packets.

## Run on your computer

Plain `python3` and `pip` — no `uv` required.

```bash
# One-time install (creates .venv and installs requirements)
make bootstrap

# Default: one process, built UI, continuous search auto-starts
make desk
# then open http://127.0.0.1:8765
```

`make desk` is the local operator UI. Continuous search starts with the desk process. Progress stays visible on every page. Pause and Stop still work; Stop turns auto-start off until you Start or Resume again.

For Vite hot reload during UI work:

```bash
make dev
# API on :8765, Vite on :5173 — open http://localhost:5173
```

CLI (same venv):

```bash
source .venv/bin/activate
python run.py scan      "TIC 441462736" --predict
python run.py hunt      --sector 15 --limit 100
python run.py phunt     --sector 15 --limit 1000 --download-workers 8 --cpu-workers 4
python run.py train     --data auto --epochs 30 --max-per-class 2500
python run.py pretrain  mlm --epochs 10
python run.py autopilot --start-sector 1 --end-sector 26
```

The desk ranks a TIC table, keeps searching (sectors, ranked queue, or local `time,flux` CSVs), reviews SDE-pass candidates, and exports one TFOP packet. Training, autopilot, and SSL pretrain stay on the CLI. A sector search needs MAST (`lightkurve` and `astroquery`). Without them the Live Run log says so and no invented stars are written.

Hardware is auto-detected: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU.

### Classifier checkpoint (optional)

Without `models/checkpoints/resnet1d.pt`, SDE-pass stars are kept as candidates. To train from catalog ephemerides after a Zenodo / MAST fetch:

```bash
python train_triage_v2.py --help
python train_triage_v2.py --fixture-train   # proves the path offline
python train_triage_v2.py --manifest data/triage_v2/manifest.csv --epochs 30
```

Do not invent a science checkpoint from Gaussian noise — the trainer refuses that path.

## Architecture

```
[MAST / lightkurve] -> Phase 1: download + quality_bitmask + same-author stitch + cadence-aware Savitzky-Golay flatten + 3-sigma clip + TPF cache
                    -> Prefilter: CDPP, Tmag, contamination, sector count
                    -> Phase 2: GPU BLS (JAX-jit / cuvarbase / Astropy / NumPy backends), top-k peaks with NMS, SDE > 7 gate
                    -> Phase 5: optional TLS refinement on BLS top peaks
                    -> Phase 3: fold (global 2048 + local 256 + local-centered 128 + periodogram 1024) + centroid + difference image + vetting metrics (SDE, depth, duration, n_transits, odd/even, secondary, V/U, Gaia neighbors, TOI ephemeris match)
                    -> Phase 4 model: TwoTowerResNet1D (1.07M params) -> binary planet/FP
                    -> Phase 6 model: ExoMinerVetter (multi-input late-fusion, 1.3M params)
                    -> Phase 7 model: TransformerDetector (per-cadence, period-agnostic; for single-transit + long-period discovery)
                    -> Save candidates with full vetting JSON; export TFOP discovery packets
```

Self-supervised pretraining (Phase 8) of the two-tower or transformer encoders is supported via `python run.py pretrain {mlm,simclr}`.

## Pipeline modules

| Module | Purpose |
|---|---|
| [`pipeline/phase1_preprocess.py`](pipeline/phase1_preprocess.py) | Download + quality + same-author stitch + cadence-aware flatten + TPF |
| [`pipeline/prefilter.py`](pipeline/prefilter.py) | Cheap O(N) gates on CDPP / magnitude / contamination |
| [`pipeline/bls_gpu.py`](pipeline/bls_gpu.py) | Multi-backend BLS, top-k peaks, SDE |
| [`pipeline/physical_priors.py`](pipeline/physical_priors.py) | Kepler-3rd-law adaptive transit duration grid |
| [`pipeline/tls_refine.py`](pipeline/tls_refine.py) | Phase 5 Transit Least Squares refiner |
| [`pipeline/fold_features.py`](pipeline/fold_features.py) | Global / local / local-centered / periodogram views |
| [`pipeline/vetting.py`](pipeline/vetting.py) | Depth, duration, odd/even, secondary, V/U, centroid, Gaia, TOI match |
| [`pipeline/centroid_vetting.py`](pipeline/centroid_vetting.py) | Optional `tesscentroidvetting` wrapper |
| [`pipeline/tic_enrich.py`](pipeline/tic_enrich.py) | Cache radius / n_sectors / CROWDSAP / Tmag before ranking |
| [`pipeline/catalog_labels.py`](pipeline/catalog_labels.py) | Fold training views at catalog ephemerides |
| [`pipeline/cache_io.py`](pipeline/cache_io.py) | Atomic-write cache for all phases (correctness-fixed in Phase 0) |
| [`models/resnet1d.py`](models/resnet1d.py) | TwoTowerResNet1D + legacy single-tower |
| [`models/exominer.py`](models/exominer.py) | ExoMiner++-style multi-input vetter |
| [`models/transformer_detector.py`](models/transformer_detector.py) | Encoder-only transformer for raw LC |
| [`hunter.py`](hunter.py) | Sequential sector hunter, SDE-gated, TPF-aware |
| [`parallel_hunter.py`](parallel_hunter.py) | Producer-consumer parallel hunter |
| [`autopilot.py`](autopilot.py) | Multi-sector loop with TOI cross-match |
| [`train_classifier.py`](train_classifier.py) | Catalog-ephemeris labels, focal loss, sector-disjoint val, two-tower head |
| [`train_triage_v2.py`](train_triage_v2.py) | Triage-v2 / catalog download → train path |
| [`pretrain_ssl.py`](pretrain_ssl.py) | Masked-modeling and SimCLR pretraining |
| [`exports/tfop_packet.py`](exports/tfop_packet.py) | TFOP-compatible JSON exporter |

## Strategy profiles

| Profile | Period range | SDE gate | Best for |
|---|---|---|---|
| `balanced` | 0.5 – 15 d | 7.0 | General-purpose |
| `ultra_short_period` | 0.2 – 1 d | 8.0 | Hot, close-in planets |
| `single_transit_long_period` | 10 – 100 d | 6.0 | Few-transit / long-period |
| `low_snr_m_dwarf` | 0.5 – 20 d | 6.0 | Faint M-dwarf hosts |

## Tests

```bash
make test            # full pytest suite
make smoke           # cache + BLS + vetting tests only
```

## Reproducibility

```bash
make bootstrap       # python3 -m venv + pip install -r requirements.txt
make desk            # local Transit desk (default)
make hunt            # one sector, sequential, low limit
make phunt           # one sector, parallel, larger limit
make train           # full TOI / Triage training run
make autopilot       # autonomous multi-sector
```

## License

MIT (see LICENSE).
