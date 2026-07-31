"""`benchtop dash` -- serve the local dashboard over a runs directory."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
import uvicorn

from benchtop.server import create_app

app = typer.Typer(help="Serve the local dashboard over existing runs.")

DEFAULT_RUNS_DIR = Path("runs")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    runs_dir: Annotated[
        Path, typer.Option("--runs-dir", help="Directory containing run directories.")
    ] = DEFAULT_RUNS_DIR,
    host: Annotated[str, typer.Option("--host", help="Interface to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Port to bind.")] = 8000,
    reload: Annotated[
        bool, typer.Option("--reload", help="Reload on code changes (development).")
    ] = False,
) -> None:
    """Serve the local dashboard over existing runs."""
    if ctx.invoked_subcommand is not None:
        return

    runs_dir = runs_dir.expanduser()
    if not runs_dir.is_dir():
        typer.echo(f"runs directory does not exist: {runs_dir}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"benchtop dash: serving {runs_dir} at http://{host}:{port}")
    uvicorn.run(create_app(runs_dir), host=host, port=port, reload=reload)
