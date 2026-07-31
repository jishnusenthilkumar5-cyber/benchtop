"""`benchtop collect` -- stub, implemented in Phase 1."""

from __future__ import annotations

import typer

app = typer.Typer(help="Collect expert demonstrations into a LeRobotDataset.")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Collect expert demonstrations into a LeRobotDataset."""
    if ctx.invoked_subcommand is None:
        typer.echo("benchtop collect: not implemented")
