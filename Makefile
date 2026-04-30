.PHONY: bootstrap test lint hunt phunt scan train pretrain autopilot clean smoke

PY := .venv/bin/python

bootstrap:
	test -d .venv || uv venv
	uv pip install --native-tls -r requirements.txt

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check . || true
	$(PY) -m ruff format --check . || true

scan:
	$(PY) run.py scan "TIC 441462736" --predict

hunt:
	$(PY) run.py hunt --sector 15 --limit 100

phunt:
	$(PY) run.py phunt --sector 15 --limit 1000 --download-workers 8 --cpu-workers 4

train:
	$(PY) run.py train --data auto --epochs 30 --max-per-class 2500 --val-holdout-sectors 21-26

pretrain-mlm:
	$(PY) run.py pretrain mlm --epochs 10

pretrain-simclr:
	$(PY) run.py pretrain simclr --epochs 10

autopilot:
	$(PY) run.py autopilot --start-sector 1 --end-sector 26

smoke:
	$(PY) -m pytest tests/test_cache.py tests/test_bls.py tests/test_vetting.py -q

clean:
	rm -rf cache/phase1/*.npz cache/phase2/*.npz cache/phase3/*.npz \
	       candidates/*.png candidates/*.json \
	       processed_stars*.txt hunter_timings.csv \
	       autopilot_state.json .streamlit_tasks.json .task_*.log .scan_result.json
