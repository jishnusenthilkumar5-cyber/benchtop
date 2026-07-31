"""`benchtop eval`: run a policy over a suite and write a run directory."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Annotated

import typer

from benchtop.core.types import MANIFEST_FILENAME
from benchtop.eval.artifacts import read_run
from benchtop.eval.registry import PolicyUnavailableError, registered
from benchtop.eval.runner import EvalConfig, run_suite
from benchtop.eval.suite import load_suite

app = typer.Typer(help="Evaluate a policy against a suite and write a run directory.")

DEFAULT_SUITE = Path("suites/pick_cube_v0.yaml")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    suite: Annotated[
        Path, typer.Option("--suite", help="Suite YAML defining task, seeds and episode cap.")
    ] = DEFAULT_SUITE,
    policy: Annotated[
        str,
        typer.Option(
            "--policy",
            help=f"Policy selector: {', '.join(registered())} (lerobot takes :<checkpoint>).",
        ),
    ] = "random",
    runs_dir: Annotated[
        Path, typer.Option("--runs-dir", help="Directory run directories are written under.")
    ] = Path("runs"),
    episodes: Annotated[
        int | None, typer.Option("--episodes", help="Evaluate only the first N suite seeds.")
    ] = None,
    video: Annotated[
        bool | None,
        typer.Option("--video/--no-video", help="Override the suite's video capture flag."),
    ] = None,
    run_id: Annotated[str | None, typer.Option("--run-id", help="Name the run directory.")] = None,
) -> None:
    """Evaluate a policy against a suite and write a run directory."""
    if ctx.invoked_subcommand is not None:
        return

    try:
        loaded = load_suite(suite)
    except FileNotFoundError:
        typer.echo(f"suite not found: {suite}", err=True)
        raise typer.Exit(2) from None
    except ValueError as exc:
        typer.echo(f"invalid suite {suite}: {exc}", err=True)
        raise typer.Exit(2) from None

    if episodes is not None:
        if episodes < 1:
            typer.echo("--episodes must be at least 1", err=True)
            raise typer.Exit(2)
        loaded = replace(loaded, seeds=loaded.seeds[:episodes])

    typer.echo(f"evaluating {policy} on {loaded.name}: {len(loaded.seeds)} episodes")
    try:
        run_dir = run_suite(
            loaded,
            EvalConfig(
                policy_spec=policy,
                runs_dir=runs_dir,
                video=video,
                run_id=run_id,
                progress=lambda line: typer.echo(line),
            ),
        )
    except (PolicyUnavailableError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from None

    metrics = read_run(run_dir).metrics
    typer.echo(
        f"success {metrics.successes}/{metrics.episodes} = {metrics.success_rate:.1%} "
        f"[95% CI {metrics.success_rate_ci_low:.1%}, {metrics.success_rate_ci_high:.1%}]"
    )
    typer.echo(f"wrote {run_dir / MANIFEST_FILENAME}")
