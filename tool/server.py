"""FastAPI surface for the Transit desk."""

from __future__ import annotations

import hmac
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from tool.desk import Desk, parse_table


class RankBody(BaseModel):
    csv: str | None = None
    path: str | None = None
    rows: list[dict] | None = None
    limit: int = 20
    period_min: float = 0.5
    period_max: float = 15.0


class ContinuousBody(BaseModel):
    local_dir: str | None = None
    mode: str | None = None
    rescan: bool = False
    strategy_profile: str | None = None
    sector: int = 1
    end_sector: int | None = None
    limit: int = 20
    period_min: float = 0.5
    period_max: float = 15.0
    nperiods: int = Field(default=400, le=20000, ge=32)
    sde_threshold: float = 7.0


class RunBody(BaseModel):
    mode: str = "sector"
    sector: int = 15
    limit: int = 20
    period_min: float = 0.5
    period_max: float = 15.0
    nperiods: int = Field(default=800, le=20000, ge=50)
    sde_threshold: float = 7.0
    tic_csv: str | None = None
    tic_path: str | None = None
    light_curve_csv: str | None = None
    light_curve_path: str | None = None
    tic_id: str = "local"
    disposition: str = ""


class LabelBody(BaseModel):
    label: str
    note: str = ""


def create_app(
    data_dir: str | Path = "var/desk",
    *,
    enable_auto_start: bool = True,
    allowed_dirs: list | None = None,
    token: str | None = None,
    checkpoint_minutes: float = 0.0,
) -> FastAPI:
    """``token`` requires ``Authorization: Bearer <token>`` (or ``?token=`` for
    EventSource and image URLs) on every ``/api`` route except health."""
    app = FastAPI(title="Transit desk")
    app.state.desk = Desk(Path(data_dir), enable_auto_start=enable_auto_start, allowed_dirs=allowed_dirs)
    app.state.checkpointer = None

    if token:
        expected = token.encode()

        @app.middleware("http")
        async def require_token(request: Request, call_next):
            path = request.url.path
            if path.startswith("/api/") and path != "/api/health":
                header = request.headers.get("authorization", "")
                supplied = header[7:] if header.lower().startswith("bearer ") else request.query_params.get("token", "")
                if not hmac.compare_digest(supplied.encode(), expected):
                    return JSONResponse({"detail": "A desk token is required."}, status_code=401,
                                        headers={"WWW-Authenticate": "Bearer"})
            return await call_next(request)

    @app.on_event("startup")
    def _start_checkpoints():
        if checkpoint_minutes and checkpoint_minutes > 0:
            from tool.checkpoint import PeriodicCheckpoint

            app.state.checkpointer = PeriodicCheckpoint(app.state.desk, checkpoint_minutes * 60.0)
            app.state.checkpointer.start()

    @app.on_event("shutdown")
    def _drain():
        app.state.desk.shutdown(timeout=None)
        if app.state.checkpointer is not None:
            app.state.checkpointer.stop(final_push=True)

    @app.get("/api/health")
    def health():
        return {**app.state.desk.health(), "auth_required": bool(token)}

    @app.get("/api/queue")
    def get_queue():
        return {"queue": app.state.desk.queue}

    @app.post("/api/queue/rank")
    def rank(body: RankBody):
        rows = list(body.rows or [])
        text = body.csv or ""
        if body.path:
            try:
                text = app.state.desk.read_user_file(body.path)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        if text.strip():
            rows = parse_table(text)
        if not rows:
            raise HTTPException(status_code=400, detail="Provide a TIC table before ranking.")
        if body.limit < 1:
            raise HTTPException(status_code=400, detail="limit is the best N and must be at least 1.")
        return app.state.desk.rank(
            rows, limit=body.limit, period_min=body.period_min, period_max=body.period_max,
        )

    @app.get("/api/continuous")
    def continuous_state(after_id: int = 0, limit: int = Query(default=500, ge=1, le=5000)):
        return app.state.desk.continuous.snapshot(after_id=after_id, limit=limit)

    @app.get("/api/continuous/events")
    def continuous_events(after_id: int = 0, limit: int = Query(default=200, ge=1, le=5000), tail: bool = False,
                          before_id: int | None = None):
        store = app.state.desk.store
        return {"lines": store.events(after_id=after_id, limit=limit, tail=tail, before_id=before_id),
                "last_event_id": store.last_event_id(), "total": store.event_count()}

    @app.get("/api/stats")
    def stats():
        return app.state.desk.stats()

    @app.get("/api/stream")
    def stream(after_id: int = 0):
        """Server-sent events: new log lines and a progress frame whenever something changes."""
        cont = app.state.desk.continuous

        def frames():
            cursor = int(after_id)
            yield "retry: 3000\n\n"
            while not cont.shutdown_event.is_set():
                lines = cont.read_lines(after_id=cursor, limit=500) if cursor else cont.read_lines(limit=200)
                if lines:
                    cursor = int(lines[-1]["id"])
                    yield f"event: lines\ndata: {json.dumps(lines)}\n\n"
                yield f"event: progress\ndata: {json.dumps(cont.progress())}\n\n"
                cont.wait_for_event(cursor, timeout=10.0)

        return StreamingResponse(frames(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/work")
    def work_items(status: str | None = None, plan: str | None = None,
                   limit: int = Query(default=100, ge=1, le=1000), offset: int = 0):
        store = app.state.desk.store
        return {"items": store.items(plan=plan, status=status, limit=limit, offset=offset),
                "counts": store.counts(plan)}

    @app.post("/api/work/{item_id}/retry")
    def retry_item(item_id: int):
        store = app.state.desk.store
        if not store.retry_item(item_id):
            raise HTTPException(status_code=409, detail="Only finished or failed items can be retried.")
        return {"item": store.get_item(item_id)}

    @app.post("/api/work/retry-failed")
    def retry_failed(plan: str | None = None):
        store = app.state.desk.store
        retried = [item["id"] for item in store.items(plan=plan, status="failed", limit=10000)
                   if store.retry_item(item["id"])]
        return {"retried": len(retried)}

    @app.get("/api/continuous/progress")
    def continuous_progress():
        return app.state.desk.continuous.progress()

    @app.post("/api/continuous/start")
    def continuous_start(body: ContinuousBody):
        try:
            return app.state.desk.continuous.start(body.model_dump(exclude_none=True))
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/continuous/pause")
    def continuous_pause():
        try:
            return app.state.desk.continuous.pause()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/continuous/stop")
    def continuous_stop():
        try:
            return app.state.desk.continuous.stop()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/runs/current")
    def current_run():
        return {"run": app.state.desk.current_run()}

    @app.post("/api/runs")
    def start_run(body: RunBody):
        try:
            run = app.state.desk.start_run(body.model_dump())
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"run": run}

    @app.get("/api/candidates")
    def candidates(
        q: str = "", status: str = "", label: str = "", flags: str = "",
        min_sde: float | None = None, min_prob: float | None = None,
        min_period: float | None = None, max_period: float | None = None,
        min_depth_ppm: float | None = None, max_depth_ppm: float | None = None,
        sort: str = "score", order: str = "desc",
        offset: int = Query(default=0, ge=0), limit: int = Query(default=0, ge=0, le=5000),
    ):
        return app.state.desk.query_candidates(
            q=q, status=status, label=label, flags=flags, min_sde=min_sde, min_prob=min_prob,
            min_period=min_period, max_period=max_period,
            min_depth_ppm=min_depth_ppm, max_depth_ppm=max_depth_ppm,
            sort=sort, order=order, offset=offset, limit=limit,
        )

    @app.post("/api/candidates/{candidate_id}/label")
    def label_candidate(candidate_id: str, body: LabelBody):
        try:
            return app.state.desk.label_candidate(candidate_id, body.label, body.note)
        except KeyError:
            raise HTTPException(status_code=404, detail="Candidate not found.") from None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/candidates/{candidate_id}/export.json")
    def export_download(candidate_id: str):
        path = app.state.desk.candidate_file(candidate_id, "export")
        if path is None:
            raise HTTPException(status_code=404, detail="Export this candidate first.")
        return FileResponse(path, media_type="application/json", filename=path.name)

    @app.get("/api/candidates/{candidate_id}")
    def candidate(candidate_id: str):
        found = app.state.desk.get_candidate(candidate_id)
        if found is None:
            raise HTTPException(status_code=404, detail="Candidate not found.")
        return found

    @app.get("/api/candidates/{candidate_id}/plot")
    def plot(candidate_id: str):
        path = app.state.desk.candidate_file(candidate_id, "plot")
        if path is None:
            raise HTTPException(status_code=404, detail="No phase plot was saved.")
        return FileResponse(path)

    @app.get("/api/candidates/{candidate_id}/difference-image")
    def difference_image(candidate_id: str):
        path = app.state.desk.candidate_file(candidate_id, "difference")
        if path is None:
            raise HTTPException(status_code=404, detail="No difference image for this candidate.")
        return FileResponse(path)

    @app.post("/api/candidates/{candidate_id}/export")
    def export_one(candidate_id: str):
        try:
            return app.state.desk.export_candidate(candidate_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Candidate not found.") from None
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    static = Path(__file__).resolve().parent / "static"
    index = static / "index.html"
    if index.is_file():
        app.mount("/", StaticFiles(directory=static, html=True), name="ui")
    else:
        @app.get("/")
        def ui_missing():
            return PlainTextResponse(
                "UI is not built. From the repo root run: make tool",
                status_code=503,
            )

    return app
