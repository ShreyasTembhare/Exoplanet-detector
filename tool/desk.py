"""Run state, ranked queue, and candidate inbox for the Transit desk."""

from __future__ import annotations

import copy
import csv
import io
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pipeline.atomic import atomic_write_json, read_json
from services import HuntConfig, mast_requirement, run_hunt, run_local_lightcurve
from tool.candidate_index import CandidateIndex
from tool.continuous import ContinuousSearch
from tool.state import StateStore


def _json_ready(value):
    """Replace NaN so the packet can be sent as JSON."""
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and value != value:
        return None
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def parse_table(text: str) -> list:
    """CSV with a header, or one TIC id per line."""
    sample = (text or "").strip()
    if not sample:
        return []
    known = {
        "tic", "tic_id", "tid", "radius", "rad", "st_rad", "n_sectors", "sectors",
        "crowdsap", "tmag", "disposition", "tfopwg_disp", "cdpp", "cdpp_ppm",
        "known_period", "period", "pl_orbper",
    }
    reader = csv.DictReader(io.StringIO(sample))
    fields = [(name or "").strip().lower() for name in (reader.fieldnames or [])]
    if not any(field in known for field in fields):
        rows = []
        for line in sample.splitlines():
            tic = line.split(",")[0].strip()
            if tic and not tic.startswith("#"):
                rows.append({"tic_id": tic})
        return rows
    rows = []
    for raw in reader:
        row = {}
        for key, value in raw.items():
            name = (key or "").strip().lower()
            if not name:
                continue
            row[name] = value.strip() if isinstance(value, str) else value
        if any(row.get(name) for name in ("tic", "tic_id", "tid")):
            rows.append(row)
    return rows


def read_time_flux(text: str):
    import numpy as np

    lines = [line.strip() for line in (text or "").splitlines() if line.strip() and not line.startswith("#")]
    if not lines:
        raise ValueError("Light curve file is empty.")
    start = 0
    header = lines[0].lower().replace(" ", "")
    if "time" in header and "flux" in header:
        start = 1
    time, flux = [], []
    for line in lines[start:]:
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        time.append(float(parts[0]))
        flux.append(float(parts[1]))
    if len(time) < 30:
        raise ValueError("Need at least 30 time, flux rows.")
    return np.asarray(time, dtype=np.float64), np.asarray(flux, dtype=np.float64)


def _provided_fields(row: dict) -> dict:
    def has(*names):
        return any(name in row and str(row.get(name) or "").strip() != "" for name in names)

    return {
        "radius": has("radius", "rad", "st_rad"),
        "n_sectors": has("n_sectors", "sectors"),
        "crowdsap": has("crowdsap"),
        "tmag": has("tmag"),
    }


def _why(record, provided: dict) -> str:
    bits = []
    radius = _finite(record.radius)
    if provided.get("radius") and radius is not None and radius > 0:
        bits.append(f"{radius:.2f} Rsun")
    if provided.get("n_sectors") and record.n_sectors:
        n = int(record.n_sectors)
        bits.append(f"{n} sector" if n == 1 else f"{n} sectors")
    crowdsap = _finite(record.crowdsap)
    if provided.get("crowdsap") and crowdsap is not None:
        bits.append(f"CROWDSAP {crowdsap:.2f}")
    tmag = _finite(record.tmag)
    if provided.get("tmag") and tmag is not None:
        bits.append(f"Tmag {tmag:.1f}")
    score = _finite(record.score)
    snr = f"expected SNR {score:.1f}" if score is not None else "expected SNR —"
    if not bits:
        return snr
    return " · ".join(bits + [snr])


def _public_target(record, provided: dict) -> dict:
    return {
        "tic_id": record.tic_norm,
        "radius_rsun": _finite(record.radius),
        "n_sectors": int(record.n_sectors) if provided.get("n_sectors") and record.n_sectors else None,
        "crowdsap": _finite(record.crowdsap) if provided.get("crowdsap") else None,
        "tmag": _finite(record.tmag) if provided.get("tmag") else None,
        "expected_snr": _finite(record.score),
        "disposition": record.disposition or "",
        "queue": record.queue,
        "known_period": _finite(record.known_period),
        "why": _why(record, provided),
    }


def candidate_label(meta: dict) -> str:
    status = str(meta.get("discovery_status") or "")
    vetting = meta.get("vetting") or {}
    if status == "KNOWN" or vetting.get("matches_toi"):
        return "Known TOI"
    if status == "NEW_ON_KNOWN_HOST":
        return "New on known host"
    if status == "LOCAL_CURVE" or meta.get("source") == "local_light_curve":
        return "Local curve"
    return "New"


def present_candidate(meta: dict) -> dict:
    from exports.tfop_packet import _is_real_epoch, phase2_epoch
    from pipeline.hard_flags import hard_flag_reasons

    vetting = meta.get("vetting") or {}
    flags = list(meta.get("hard_flags") or hard_flag_reasons(vetting))
    epoch = phase2_epoch(meta)
    depth = _finite(vetting.get("depth"))
    sde = _finite(meta.get("sde"))
    period = _finite(meta.get("best_period"))
    diff = meta.get("difference_image")
    plot = meta.get("plot") or meta.get("_plot_path")
    prob = _finite(meta.get("prob_planet"))
    return {
        "id": meta.get("_id"),
        "tic_id": str(meta.get("tic_id") or ""),
        "period_days": period,
        "epoch_btjd": epoch if _is_real_epoch(epoch) else None,
        "depth": depth,
        "depth_ppm": None if depth is None else depth * 1e6,
        "sde": sde,
        "prob_planet": prob,
        "model_score": prob,
        "hard_flags": flags,
        "label": candidate_label(meta),
        "status": meta.get("status") or ("CANDIDATE_SDE" if meta.get("prob_planet") is None else ""),
        "source": meta.get("source") or "",
        "has_difference_image": bool(diff) and Path(str(diff)).is_file(),
        "has_phase_curve": bool((meta.get("phase_curve") or {}).get("phase")),
        "has_plot": bool(plot) and Path(str(plot)).is_file(),
        "odd_even_ratio": _finite(vetting.get("odd_even_ratio")),
        "secondary_significance": _finite(vetting.get("secondary_significance")),
        "centroid_offset_px": _finite(vetting.get("centroid_offset_px", meta.get("centroid_offset"))),
        "n_transits": vetting.get("n_transits"),
        "duration_hours": _finite(vetting.get("duration_hours")),
        "score": _finite(meta.get("score")),
        "demotion": meta.get("demotion") or vetting.get("demotion"),
        "discovery_status": meta.get("discovery_status") or "",
        "fpp": _finite(vetting.get("fpp")),
        "kind": vetting.get("kind") or "periodic",
        "timestamp": meta.get("timestamp") or "",
    }


class Desk:
    def __init__(
        self,
        root: Path,
        *,
        enable_auto_start: bool = True,
        allowed_dirs: Optional[list] = None,
    ):
        """``allowed_dirs=None`` lets path inputs read anywhere (local laptop use).

        A list jails every path input to the data directory plus those
        directories; :mod:`tool.__main__` turns this on whenever the desk is
        bound to a non-loopback address.
        """
        self.root = Path(root).expanduser().resolve()
        self.candidates = self.root / "candidates"
        self.exports = self.root / "exports"
        self.candidates.mkdir(parents=True, exist_ok=True)
        self.exports.mkdir(parents=True, exist_ok=True)
        self.allowed_dirs = None if allowed_dirs is None else [
            Path(p).expanduser().resolve() for p in [self.root, *allowed_dirs]
        ]
        self.store = StateStore(self.root / "state.db")
        self.store.release_stale()
        self.lock = threading.Lock()
        self.queue: Optional[dict] = read_json(self.root / "queue.json")
        self.run: Optional[dict] = self.store.latest_run()
        if self.run and self.run.get("status") == "running":
            # The process that owned this run is gone; say so instead of spinning.
            self.run.update(status="failed", error="Interrupted by a restart.", finished_at=_now(),
                            message="Search was interrupted by a restart.")
            self.store.save_run(self.run)
        self._index = CandidateIndex(self.candidates)
        self.continuous = ContinuousSearch(self, enable_auto_start=enable_auto_start)

    def resolve_user_path(self, raw: str, *, must_be_dir: bool = False) -> Path:
        path = Path(str(raw)).expanduser()
        if not path.is_absolute():
            path = self.root / path
        path = path.resolve()
        if self.allowed_dirs is not None and not any(
            path == base or base in path.parents for base in self.allowed_dirs
        ):
            raise ValueError("Path is outside the directories this desk may read.")
        if must_be_dir and not path.is_dir():
            raise ValueError(f"Local curve directory not found: {raw}")
        if not must_be_dir and not path.is_file():
            raise ValueError(f"File not found: {raw}")
        return path

    def read_user_file(self, raw: str) -> str:
        file = self.resolve_user_path(raw)
        if file.stat().st_size > 20_000_000:
            raise ValueError("File is larger than 20 MB.")
        return file.read_text()

    def shutdown(self, timeout: Optional[float] = None):
        self.continuous.shutdown(timeout)

    def health(self) -> dict:
        checkpoint = Path("models/checkpoints/resnet1d.pt")
        mast = mast_requirement()
        progress = self.continuous.progress()
        return {
            "ok": True,
            "checkpoint": checkpoint.is_file(),
            "mast": mast["available"],
            "mast_message": mast["message"],
            "continuous_status": progress.get("status") or self.continuous.state.get("status"),
            "auto_start": bool(progress.get("auto_start", True)),
        }

    def rank(self, rows: list, *, limit: int, period_min: float, period_max: float) -> dict:
        from pipeline.target_rank import rank_targets, records_from_rows
        from pipeline.tic_enrich import enrich_records

        cache = self.root / "tic_enrich.json"
        records = records_from_rows(rows)
        enrich_records(records, cache_path=cache)
        provided = {}
        for record, row in zip(records, rows):
            flags = _provided_fields(row)
            # After enrichment, treat filled fields as provided for the why string.
            if _finite(record.radius):
                flags["radius"] = True
            if record.n_sectors and int(record.n_sectors) > 0:
                flags["n_sectors"] = True
            if _finite(record.crowdsap):
                flags["crowdsap"] = True
            if _finite(record.tmag):
                flags["tmag"] = True
            record.extra = flags
            provided[id(record)] = flags
        ranked = rank_targets(
            records, limit=int(limit), period_min=period_min, period_max=period_max,
        )
        payload = {
            "limit": int(limit),
            "period_min": period_min,
            "period_max": period_max,
            "primary": [_public_target(rec, provided.get(id(rec), {})) for rec in ranked.primary],
            "second_pass": [_public_target(rec, provided.get(id(rec), {})) for rec in ranked.second_pass],
            "note": "limit is the best N, not the first N. Sky position is not used.",
        }
        self.queue = payload
        atomic_write_json(self.root / "queue.json", payload)
        return payload

    def list_candidates(self, **query) -> list:
        return self.query_candidates(**query)["candidates"]

    def query_candidates(
        self,
        *,
        q: str = "",
        status: str = "",
        label: str = "",
        flags: str = "",
        min_sde: Optional[float] = None,
        min_prob: Optional[float] = None,
        min_period: Optional[float] = None,
        max_period: Optional[float] = None,
        min_depth_ppm: Optional[float] = None,
        max_depth_ppm: Optional[float] = None,
        sort: str = "score",
        order: str = "desc",
        offset: int = 0,
        limit: int = 0,
    ) -> dict:
        labels = self.store.labels()
        rows = []
        for meta in self._index.refresh():
            row = present_candidate(meta)
            row["review_label"] = (labels.get(row["id"]) or {}).get("label")
            rows.append(row)

        def keep(row) -> bool:
            if q and q.strip().lower() not in f"{row['tic_id']} {row['id']}".lower():
                return False
            if status and status.lower() not in str(row.get("status") or "").lower():
                return False
            if label:
                wanted = label.lower()
                if wanted == "unreviewed":
                    if row.get("review_label"):
                        return False
                elif wanted not in {str(row.get("label") or "").lower(), str(row.get("review_label") or "").lower(),
                                    str(row.get("discovery_status") or "").lower()}:
                    return False
            if flags == "none" and row["hard_flags"]:
                return False
            if flags == "any" and not row["hard_flags"]:
                return False
            for key, bound, cmp in (
                ("sde", min_sde, "ge"), ("prob_planet", min_prob, "ge"),
                ("period_days", min_period, "ge"), ("period_days", max_period, "le"),
                ("depth_ppm", min_depth_ppm, "ge"), ("depth_ppm", max_depth_ppm, "le"),
            ):
                if bound is None:
                    continue
                value = row.get(key)
                if value is None:
                    return False
                if cmp == "ge" and value < bound:
                    return False
                if cmp == "le" and value > bound:
                    return False
            return True

        rows = [row for row in rows if keep(row)]
        sortable = {"score", "sde", "prob_planet", "period_days", "depth_ppm", "tic_id", "updated"}
        field = sort if sort in sortable else "score"
        missing_last = -1e30 if order != "asc" else 1e30

        def sort_key(row):
            value = row.get(field)
            if field == "tic_id":
                return (str(value or ""), row["id"])
            if field == "updated":
                return (str(row.get("timestamp") or ""), row["id"])
            if field == "score" and value is None:
                value = row.get("sde")
            return (value if value is not None else missing_last, row["tic_id"])

        rows.sort(key=sort_key, reverse=(order != "asc"))
        total = len(rows)
        if limit and limit > 0:
            rows = rows[max(offset, 0): max(offset, 0) + limit]
        elif offset:
            rows = rows[offset:]
        return {"candidates": [_json_ready(row) for row in rows], "total": total}

    def get_candidate(self, candidate_id: str) -> Optional[dict]:
        meta = self._index.get(candidate_id)
        if meta is None:
            return None
        public = present_candidate(meta)
        public["phase_curve"] = meta.get("phase_curve") or {"phase": [], "flux": []}
        public["light_curve"] = meta.get("light_curve") or {"time": [], "flux": []}
        public["periodogram"] = meta.get("periodogram") or {"period": [], "power": []}
        public["odd_even_curve"] = meta.get("odd_even_curve") or {}
        public["secondary_curve"] = meta.get("secondary_curve") or {}
        public["vetting"] = meta.get("vetting") or {}
        public["alt_peaks"] = meta.get("alt_peaks") or []
        public["known_match"] = meta.get("known_match")
        public["review"] = self.store.labels().get(candidate_id)
        return _json_ready(public)

    def label_candidate(self, candidate_id: str, label: str, note: str = "") -> dict:
        allowed = {"planet", "false_positive", "unsure"}
        if label not in allowed:
            raise ValueError(f"label must be one of {sorted(allowed)}")
        meta = self._index.get(candidate_id)
        if meta is None:
            raise KeyError(candidate_id)
        snapshot = {
            "tic_id": meta.get("tic_id"), "best_period": meta.get("best_period"),
            "epoch": meta.get("epoch"), "sde": meta.get("sde"), "json_path": meta.get("_json_path"),
        }
        self.store.set_label(candidate_id, label, note, snapshot)
        return {"candidate_id": candidate_id, "label": label, "note": note}

    def candidate_file(self, candidate_id: str, kind: str) -> Optional[Path]:
        meta = self._index.get(candidate_id)
        if meta is None:
            return None
        if kind == "plot":
            raw = meta.get("plot") or meta.get("_plot_path")
        elif kind == "export":
            raw = str(self.exports / f"{candidate_id}_tfop.json")
        else:
            raw = meta.get("difference_image")
        if not raw:
            return None
        path = Path(str(raw))
        if not path.is_absolute():
            path = Path.cwd() / path
        path = path.resolve()
        if path.is_file() and self._allowed(path):
            return path
        return None

    def _allowed(self, path: Path) -> bool:
        root = self.root.resolve()
        try:
            path.relative_to(root)
        except ValueError:
            return False
        return True

    def raw_candidate(self, candidate_id: str) -> Optional[dict]:
        return self._index.get(candidate_id)

    def export_candidate(self, candidate_id: str) -> dict:
        from exports.tfop_packet import (
            TFOP_SCHEMA_VERSION,
            _is_real_epoch,
            candidate_json_to_tfop,
            phase2_epoch,
        )

        meta = self.raw_candidate(candidate_id)
        if meta is None:
            raise KeyError(candidate_id)
        epoch = phase2_epoch(meta)
        if not _is_real_epoch(epoch):
            raise ValueError("No transit epoch on this candidate. Refusing to write t0 = 0.")
        cand = candidate_json_to_tfop(meta)
        item = {
            "packet_id": cand.packet_id(),
            "tic_id": cand.tic_id,
            "period_days": cand.period_days,
            "epoch_btjd": epoch,
            "depth_ppm": cand.depth_ppm,
            "duration_hours": cand.duration_hours,
            "sde": cand.sde,
            "n_transits": cand.n_transits,
            "odd_even_ratio": cand.odd_even_ratio,
            "secondary_significance": cand.secondary_significance,
            "v_shape_score": cand.v_shape_score,
            "centroid_offset_arcsec": cand.centroid_offset_arcsec,
            "prob_planet": cand.prob_planet,
            "discovery_status": cand.discovery_status,
            "plot_path": cand.plot_path,
        }
        packet = _json_ready({
            "schema_version": TFOP_SCHEMA_VERSION,
            "submitter": cand.submitter,
            "strategy_profile": str(meta.get("strategy_profile") or "balanced"),
            "generated_at": _now(),
            "n_candidates": 1,
            "candidates": [item],
        })
        out = self.exports / f"{candidate_id}_tfop.json"
        atomic_write_json(out, packet)
        return {
            "epoch_btjd": epoch, "path": str(out), "packet": packet,
            "download_url": f"/api/candidates/{candidate_id}/export.json",
        }

    def current_run(self) -> Optional[dict]:
        with self.lock:
            return copy.deepcopy(self.run)

    def start_run(self, payload: dict) -> dict:
        mode = payload.get("mode") or "sector"
        if mode not in {"sector", "local"}:
            raise ValueError("mode must be sector or local")
        with self.lock:
            if self.continuous.is_running() or (self.run and self.run.get("status") == "running"):
                raise RuntimeError("A search is already running.")
            run = {
                "id": "run-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"),
                "status": "running",
                "mode": mode,
                "message": "Search started.",
                "error": None,
                "needs_mast": False,
                "started_at": _now(),
                "finished_at": None,
                "candidate_ids": [],
            }
            self.run = run
            self.store.save_run(run)
            snapshot = copy.deepcopy(run)
        thread = threading.Thread(target=self._execute, args=(run["id"], payload), name=run["id"])
        thread.start()
        return snapshot

    def _execute(self, run_id: str, payload: dict):
        try:
            if payload.get("mode") == "local":
                self._execute_local(run_id, payload)
            else:
                self._execute_sector(run_id, payload)
        except Exception as exc:
            self._finish(run_id, "failed", error=str(exc), message="Search failed.")

    def _execute_sector(self, run_id: str, payload: dict):
        mast = mast_requirement()
        if not mast["available"]:
            self._finish(
                run_id, "failed", error=mast["message"], needs_mast=True,
                message="Search needs MAST.",
            )
            return
        tic_list = None
        text = payload.get("tic_csv") or ""
        if payload.get("tic_path"):
            text = self.read_user_file(payload["tic_path"])
        if text.strip():
            path = self.root / "tic_list.csv"
            path.write_text(text)
            tic_list = str(path)
        config = HuntConfig(
            sector=int(payload.get("sector") or 15),
            limit=int(payload.get("limit") or 20),
            sde_threshold=float(payload.get("sde_threshold") or 7.0),
            period_min=float(payload.get("period_min") or 0.5),
            period_max=float(payload.get("period_max") or 15.0),
            nperiods=int(payload.get("nperiods") or 2000),
            checkpoint=payload.get("checkpoint") or "models/checkpoints/resnet1d.pt",
            candidate_dir=str(self.candidates),
            log_file=str(self.root / "processed_stars.txt"),
            tic_list=tic_list,
            with_tpf=bool(payload.get("with_tpf", True)),
            prefilter=True,
            sequential=False,
        )
        self._set(run_id, message="Downloading light curves and running BLS.")
        result = run_hunt(config)
        found = result.get("candidates") if isinstance(result, dict) else None
        self._finish(
            run_id, "done",
            message="Sector search finished.",
            candidate_ids=list(found or []),
        )

    def _execute_local(self, run_id: str, payload: dict):
        text = payload.get("light_curve_csv") or ""
        if payload.get("light_curve_path"):
            text = self.read_user_file(payload["light_curve_path"])
        if not text.strip():
            raise ValueError("A local search needs a time, flux file.")
        time, flux = read_time_flux(text)
        tic_id = str(payload.get("tic_id") or "local").strip() or "local"
        self._set(run_id, message=f"BLS on local light curve {tic_id}. No MAST download.")
        result = run_local_lightcurve(
            time, flux, tic_id, str(self.candidates),
            period_min=float(payload.get("period_min") or 0.5),
            period_max=float(payload.get("period_max") or 15.0),
            nperiods=int(payload.get("nperiods") or 800),
            sde_threshold=float(payload.get("sde_threshold") or 7.0),
            disposition=str(payload.get("disposition") or ""),
        )
        ids = [result["id"]] if result.get("saved") else []
        self._finish(run_id, "done", message=result["message"], candidate_ids=ids)

    def _set(self, run_id: str, **fields):
        with self.lock:
            if self.run and self.run.get("id") == run_id:
                self.run.update(fields)
                self.store.save_run(_json_ready(self.run))

    def _finish(self, run_id: str, status: str, **fields):
        fields["status"] = status
        fields["finished_at"] = _now()
        self._set(run_id, **fields)


