"""Run `lerobot-train` for a benchtop `TrainConfig`.

Training is launched as a subprocess rather than by importing lerobot's train
entrypoint: that entrypoint is a draccus-wrapped CLI that parses `sys.argv` and
installs its own logging and signal handling, so calling it in-process would
mean fighting it. A subprocess also means a crashed or killed run cannot take
the CLI's own state with it -- and the checkpoint on disk is what matters.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from benchtop.train.config import TrainConfig

#: Console script installed by the `lerobot[training]` extra.
LEROBOT_TRAIN_BIN = "lerobot-train"
#: Module fallback, for the case where the venv's bin dir is not on PATH.
LEROBOT_TRAIN_MODULE = "lerobot.scripts.lerobot_train"


def train_command(config: TrainConfig) -> list[str]:
    """The full command line that `run_training` would execute."""
    executable = shutil.which(LEROBOT_TRAIN_BIN)
    launcher = [executable] if executable else [sys.executable, "-m", LEROBOT_TRAIN_MODULE]
    return [*launcher, *config.to_cli_args()]


def resolve_resume(config: TrainConfig) -> TrainConfig:
    """Turn an explicit `--resume` into a resumable config, or explain why not.

    CPU training is the long pole in this project: a run that cannot be resumed
    is a run that has to start over, so resuming is a first-class path rather
    than an afterthought.
    """
    if not config.resume:
        return config
    if not config.has_resumable_checkpoint():
        raise FileNotFoundError(
            f"--resume was requested but {config.resume_config_path} does not exist. "
            f"Either {config.output_dir} is not a previous run, or it never reached its "
            f"first checkpoint (save_freq={config.save_freq})."
        )
    return config


def run_training(config: TrainConfig, *, dry_run: bool = False) -> int:
    """Execute (or, with `dry_run`, just print) the training run.

    Returns the subprocess exit code; 0 for a dry run.
    """
    config = resolve_resume(config)
    command = train_command(config)
    if dry_run:
        print(" ".join(command))
        return 0

    if not config.resume:
        dataset = Path(config.dataset)
        if not dataset.exists():
            raise FileNotFoundError(
                f"dataset {dataset} does not exist -- run `benchtop collect` first"
            )
        if config.output_dir.exists():
            raise FileExistsError(
                f"{config.output_dir} already exists. Pass --resume to continue that run, "
                f"or point --out somewhere else; lerobot-train refuses to overwrite it."
            )
        # Only the parent: lerobot-train creates the run dir itself and treats a
        # pre-existing one as a run it would clobber.
        config.output_dir.parent.mkdir(parents=True, exist_ok=True)

    return _run(command)


def _run(command: Sequence[str]) -> int:
    """Stream the trainer's output through, so a long run stays observable."""
    return subprocess.run(list(command), check=False).returncode
