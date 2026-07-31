"""`benchtop train` -- stub, implemented in Phase 1."""

from __future__ import annotations

import typer

app = typer.Typer(help="Train a policy on a demonstration dataset.")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Train a policy on a demonstration dataset."""
    if ctx.invoked_subcommand is None:
        typer.echo("benchtop train: not implemented")
