"""FastAPI application serving the local benchtop dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from benchtop.server.runs import Run, RunNotFound, RunStore

PACKAGE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = PACKAGE_DIR / "templates"
STATIC_DIR = PACKAGE_DIR / "static"


def _pct(x: float | None) -> str:
    return "--" if x is None else f"{100 * x:.1f}%"


def _num(x: float | None, digits: int = 3) -> str:
    return "--" if x is None else f"{x:.{digits}f}"


def intervals_overlap(a: Run, b: Run) -> bool:
    """Whether two runs' 95% success-rate intervals overlap at all.

    Non-overlap is the honest headline of the compare view: if they overlap, the
    observed difference is not distinguishable from noise at this episode count.
    """
    return (
        a.metrics.success_rate_ci_low <= b.metrics.success_rate_ci_high
        and b.metrics.success_rate_ci_low <= a.metrics.success_rate_ci_high
    )


def create_app(runs_dir: Path | str) -> FastAPI:
    store = RunStore(Path(runs_dir))
    app = FastAPI(title="benchtop dash")
    app.state.store = store

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    templates.env.filters["pct"] = _pct
    templates.env.filters["num"] = _num
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    def _get(run_id: str) -> Run:
        try:
            return store.get(run_id)
        except RunNotFound:
            raise HTTPException(status_code=404, detail=f"no such run: {run_id}") from None

    def _page(request: Request, name: str, ctx: dict[str, Any]) -> HTMLResponse:
        return templates.TemplateResponse(request, name, {"runs_dir": str(store.runs_dir), **ctx})

    # ---------------------------------------------------------------- API

    @app.get("/api/runs")
    def api_runs() -> dict[str, Any]:
        return {"runs": [r.summary_dict() for r in store.list_runs()]}

    @app.get("/api/runs/{run_id}")
    def api_run(run_id: str) -> dict[str, Any]:
        return _get(run_id).detail_dict()

    # --------------------------------------------------------------- pages

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        return _page(request, "index.html", {"runs": store.list_runs()})

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def run_detail(request: Request, run_id: str) -> HTMLResponse:
        run = _get(run_id)
        return _page(request, "run.html", {"run": run, "all_runs": store.list_runs()})

    @app.get("/compare", response_class=HTMLResponse)
    def compare(request: Request, a: str | None = None, b: str | None = None) -> HTMLResponse:
        runs = store.list_runs()
        left = _get(a) if a else None
        right = _get(b) if b else None
        overlap = intervals_overlap(left, right) if left and right else None
        return _page(
            request,
            "compare.html",
            {"all_runs": runs, "left": left, "right": right, "overlap": overlap},
        )

    @app.get("/runs/{run_id}/videos/{name}")
    def video(run_id: str, name: str) -> FileResponse:
        run = _get(run_id)
        path = run.videos_dir / name
        if "/" in name or "\\" in name or not path.is_file():
            raise HTTPException(status_code=404, detail=f"no such video: {name}")
        return FileResponse(path, media_type="video/mp4")

    return app
