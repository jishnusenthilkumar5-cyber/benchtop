"""`benchtop eval` -- stub, implemented in Phase 1."""

from __future__ import annotations

import typer

app = typer.Typer(help="Evaluate a policy against a suite and write a run directory.")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Evaluate a policy against a suite and write a run directory."""
    if ctx.invoked_subcommand is None:
        typer.echo("benchtop eval: not implemented")
