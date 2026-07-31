"""`benchtop dash` -- stub, implemented in Phase 1."""

from __future__ import annotations

import typer

app = typer.Typer(help="Serve the local dashboard over existing runs.")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Serve the local dashboard over existing runs."""
    if ctx.invoked_subcommand is None:
        typer.echo("benchtop dash: not implemented")
