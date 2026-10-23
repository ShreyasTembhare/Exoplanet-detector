.PHONY: bootstrap test lint web-test ci hunt phunt scan train pretrain autopilot clean smoke tool desk dev completeness

PY := .venv/bin/python
PIP := .venv/bin/pip

bootstrap:
	test -d .venv || python3 -m venv .venv
	$(PIP) install -r requirements.txt

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .

web-test:
	cd web && npm test && npm run build

# Everything CI runs.
ci: lint test web-test

completeness:
	$(PY) -m pipeline.completeness --out var/desk/completeness.json

scan:
	$(PY) run.py scan "TIC 441462736" --predict

hunt:
	$(PY) run.py hunt --sector 15 --limit 100

phunt:
	$(PY) run.py phunt --sector 15 --limit 1000 --download-workers 8 --cpu-workers 4

train:
	$(PY) train_triage_v2.py --fetch --limit 2000 --injections 500 --epochs 30

train-labels:
	$(PY) train_classifier.py --labels var/desk

pretrain-mlm:
	$(PY) run.py pretrain mlm --epochs 10

pretrain-simclr:
	$(PY) run.py pretrain simclr --epochs 10

autopilot:
	$(PY) run.py autopilot --start-sector 1 --end-sector 26

smoke:
	$(PY) -m pytest tests/test_cache.py tests/test_bls.py tests/test_vetting.py -q

# Default local run: one process, built UI, continuous search auto-starts.
desk: bootstrap
	cd web && npm install && npm run build
	$(PY) -m tool --host 127.0.0.1 --port 8765 --auto-start

tool: desk

dev: bootstrap
	cd web && npm install
	$(PY) -m tool --host 127.0.0.1 --port 8765 --auto-start &
	cd web && npm run dev

clean:
	rm -rf cache/phase1/*.npz cache/phase2/*.npz cache/phase3/*.npz \
	       candidates/*.png candidates/*.json \
	       processed_stars*.txt hunter_timings.csv \
	       autopilot_state.json .streamlit_tasks.json .task_*.log .scan_result.json
