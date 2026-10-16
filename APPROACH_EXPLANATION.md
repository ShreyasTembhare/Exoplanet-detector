# TESS search scope

This repository searches TESS light curves for transiting planets. The live path is:

1. Rank the sector TIC queue before download (stellar radius, sector baseline, crowding, expected SNR). Known CP/KP hosts are a masked second pass. The ranker does not use a sky map of published planets.
2. Parallel hunt (`parallel_hunter.py`) is the default search. `hunter.py` remains for debugging (`python run.py hunt --sequential`).
3. BLS on the fastest installed backend (JAX, otherwise Astropy, otherwise NumPy). Astropy stays the test reference. A phase-2 cache skips a second pass over the same TIC, sector, and grid.
4. Durations come from `adaptive_duration_grid` (about 3–4 per period). Every kept top-k peak is refined, folded, and vetted. An accepted repeating transit is masked and the curve is searched again. A single-event MES covers monotransits. TLS runs only on SDE-pass peaks.
5. Without a classifier checkpoint, an SDE-pass star is `CANDIDATE_SDE` with vetting JSON and the Phase 2 epoch. Inference uses `make_two_tower_inputs` and `TwoTowerResNet1D`. Training refuses to write `resnet1d.pt` from Gaussian noise and folds labels at catalog ephemerides.
6. The shortlist gets hard flags (odd/even, secondary, centroid, n_transits), Gaia, one difference-image score, and an optional TRICERATOPS FPP/NFPP. `batman` injection-recovery on quiet stars reports completeness against period, radius, Tmag, and CDPP.

A real Astronet-Triage-v2 light-curve download is still required before the two-tower checkpoint is a science model. The transformer in `models/transformer_detector.py` is not on the hunt path.
