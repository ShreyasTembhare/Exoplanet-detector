# TESS Exoplanet Detector

End-to-end TESS pipeline: download light curves, detrend, run BLS (and optionally TLS) periodograms, fold + vet candidates, classify with a two-tower ResNet-1D (and optionally an ExoMiner++-style multi-input vetter or a transformer detector for raw light curves), and export TFOP-compatible discovery packets.

## Quick start

```bash
uv venv && source .venv/bin/activate
uv pip install --native-tls -r requirements.txt

# Streamlit dashboard
streamlit run app.py

# CLI
python run.py scan      "TIC 441462736" --predict
python run.py hunt      --sector 15 --limit 100
python run.py phunt     --sector 15 --limit 1000 --download-workers 8 --cpu-workers 4
python run.py train     --data auto --epochs 30 --max-per-class 2500
python run.py pretrain  mlm --epochs 10
python run.py autopilot --start-sector 1 --end-sector 26
```

Hardware is auto-detected: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU.

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
| [`pipeline/cache_io.py`](pipeline/cache_io.py) | Atomic-write cache for all phases (correctness-fixed in Phase 0) |
| [`models/resnet1d.py`](models/resnet1d.py) | TwoTowerResNet1D + legacy single-tower |
| [`models/exominer.py`](models/exominer.py) | ExoMiner++-style multi-input vetter |
| [`models/transformer_detector.py`](models/transformer_detector.py) | Encoder-only transformer for raw LC |
| [`hunter.py`](hunter.py) | Sequential sector hunter, SDE-gated, TPF-aware |
| [`parallel_hunter.py`](parallel_hunter.py) | Producer-consumer parallel hunter |
| [`autopilot.py`](autopilot.py) | Multi-sector loop with TOI cross-match |
| [`train_classifier.py`](train_classifier.py) | TOI-only labels, focal loss, sector-disjoint val, two-tower head |
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
make bootstrap       # create venv + install deps
make hunt            # one sector, sequential, low limit
make phunt           # one sector, parallel, larger limit
make train           # full TOI training run
make autopilot       # autonomous multi-sector
```

## Recent changes (modernization plan)

The pipeline was overhauled in 9 phases. Headline changes:

- **Phase 0**: fixed a silent cache-write bug (every cache file was 0 bytes); BLS depth clamp; wrap-around fix; SDE gate replaces fixed power threshold; cadence-aware flatten; same-author stitching; flock-protected log; pytest + ruff suite.
- **Phase 1**: BLS JAX-jit + chunked grids + float32, top-k peaks with harmonic NMS, Astropy oracle, cuvarbase optional backend, physical-prior duration grid.
- **Phase 2**: parallel hunter (producer-consumer pipeline with download / GPU BLS / CPU folding / batched inference stages).
- **Phase 3**: full vetting suite (centroid via TPF, odd/even, secondary, V/U, Gaia, ephemeris match) persisted to candidate JSON.
- **Phase 4**: TOI-only labels, injection-recovery, focal loss, two-tower ResNet, sector-disjoint validation, Recall@FPR=1% as primary metric.
- **Phase 5**: optional TLS second-stage refiner.
- **Phase 6**: ExoMiner++-style multi-input vetter with periodogram + diff-image + scalar branches.
- **Phase 7**: transformer detector for raw light curves (single-transit + long-period discovery path).
- **Phase 8**: self-supervised pretraining (masked modeling + SimCLR).
- **Phase 9**: pytest + GitHub CI + Streamlit Scan as background task + TFOP discovery packet exporter + Makefile.

## License

MIT (see LICENSE).
