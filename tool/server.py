"""FastAPI surface for the Transit desk."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
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
    sector: int = 1
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


def create_app(data_dir: str | Path = "var/desk") -> FastAPI:
    app = FastAPI(title="Transit desk")
    app.state.desk = Desk(Path(data_dir))

    @app.get("/api/health")
    def health():
        return app.state.desk.health()

    @app.get("/api/queue")
    def get_queue():
        return {"queue": app.state.desk.queue}

    @app.post("/api/queue/rank")
    def rank(body: RankBody):
        rows = list(body.rows or [])
        text = body.csv or ""
        if body.path:
            from tool.desk import _read_user_file
            text = _read_user_file(body.path)
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
    def continuous_state():
        return app.state.desk.continuous.snapshot()

    @app.post("/api/continuous/start")
    def continuous_start(body: ContinuousBody):
        try:
            return app.state.desk.continuous.start(body.model_dump())
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
    def candidates():
        return {"candidates": app.state.desk.list_candidates()}

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
