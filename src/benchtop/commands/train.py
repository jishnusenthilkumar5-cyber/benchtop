"""`benchtop train` -- train a state-only ACT policy on a demonstration dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from benchtop.train.config import DEFAULT_REPO_ID, TrainConfig
from benchtop.train.runner import run_training

app = typer.Typer(help="Train a policy on a demonstration dataset.")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    dataset: Annotated[
        Path | None,
        typer.Option("--dataset", help="LeRobotDataset directory written by `benchtop collect`."),
    ] = None,
    out: Annotated[
        Path,
        typer.Option("--out", help="Output directory for checkpoints and logs."),
    ] = Path("outputs/train/act_pick_cube_v0"),
    steps: Annotated[int, typer.Option(help="Total training steps.")] = TrainConfig.steps,
    batch_size: Annotated[int, typer.Option(help="Batch size.")] = TrainConfig.batch_size,
    save_freq: Annotated[
        int, typer.Option(help="Checkpoint every N steps. Smaller is cheaper insurance on CPU.")
    ] = TrainConfig.save_freq,
    num_workers: Annotated[int, typer.Option(help="Dataloader workers.")] = TrainConfig.num_workers,
    seed: Annotated[int, typer.Option(help="Training seed.")] = TrainConfig.seed,
    repo_id: Annotated[
        str, typer.Option(help="Dataset id recorded in the run config.")
    ] = DEFAULT_REPO_ID,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume/--no-resume",
            help="Continue the run in --out from its last checkpoint instead of starting over.",
        ),
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print the lerobot-train command and exit.")
    ] = False,
) -> None:
    """Train ACT on `--dataset`, writing checkpoints under `--out`.

    State-only and CPU-only by construction: expect ~3 hours for the default
    30k steps on 8 CPU cores (see `benchtop.train.config` for the measurement).
    Interrupted runs resume from the last checkpoint with the same command plus
    `--resume`; `--dataset` is then read back from the checkpoint's config.
    """
    if ctx.invoked_subcommand is not None:
        return
    if dataset is None and not resume:
        raise typer.BadParameter("--dataset is required unless --resume is passed")

    config = TrainConfig(
        dataset=Path(dataset) if dataset is not None else Path(),
        output_dir=out,
        repo_id=repo_id,
        steps=steps,
        batch_size=batch_size,
        save_freq=save_freq,
        num_workers=num_workers,
        seed=seed,
        resume=resume,
    )
    code = run_training(config, dry_run=dry_run)
    if code != 0:
        raise typer.Exit(code)
