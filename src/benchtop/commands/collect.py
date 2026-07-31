"""`benchtop collect` -- scripted expert demonstrations into a LeRobotDataset."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(help="Collect expert demonstrations into a LeRobotDataset.")

DEFAULT_OUT = Path("datasets/pick_cube_v0_expert")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    episodes: Annotated[int, typer.Option(help="Number of episodes to attempt.")] = 100,
    seed_start: Annotated[int, typer.Option(help="First episode seed; seeds run upward.")] = 0,
    out: Annotated[Path, typer.Option(help="Dataset directory to create.")] = DEFAULT_OUT,
    repo_id: Annotated[str | None, typer.Option(help="LeRobot repo id for the dataset.")] = None,
) -> None:
    """Roll out the scripted expert and write the successful episodes.

    Failed rollouts are discarded -- a demonstration dataset should not teach
    a policy how to fail -- and counted in the summary.
    """
    if ctx.invoked_subcommand is not None:
        return

    # Imported here so `benchtop --help` does not pay for mujoco and lerobot.
    from benchtop.data.collect import DEFAULT_REPO_ID, Episode, collect_dataset

    if episodes <= 0:
        raise typer.BadParameter("--episodes must be positive")

    typer.echo(f"collecting {episodes} expert episodes from seed {seed_start} into {out}")

    def report_episode(episode: Episode) -> None:
        mark = "ok  " if episode.success else "drop"
        typer.echo(f"  seed {episode.seed:>5}  {mark}  {episode.steps:>3} steps")

    report = collect_dataset(
        out,
        episodes,
        seed_start,
        repo_id=repo_id or DEFAULT_REPO_ID,
        on_episode=report_episode,
    )
    typer.echo(report.summary())
    if report.kept == 0:
        raise typer.Exit(code=1)
