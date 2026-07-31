from __future__ import annotations

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from benchtop.cli import app
from benchtop.train.config import TrainConfig, last_checkpoint_config
from benchtop.train.runner import resolve_resume, run_training, train_command

runner = CliRunner()

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _unwrap(output: str) -> str:
    """Colour codes and box drawing removed, line wrapping undone."""
    return " ".join(_ANSI.sub("", output).replace("│", " ").split())


def make_config(tmp_path: Path, **kwargs) -> TrainConfig:
    return TrainConfig(dataset=tmp_path / "dataset", output_dir=tmp_path / "out", **kwargs)


def test_cli_args_are_state_only_cpu_and_checkpointing(tmp_path):
    args = make_config(tmp_path).to_cli_args()
    joined = " ".join(args)
    assert "--policy.type=act" in args
    assert "--policy.device=cpu" in args
    assert "--policy.pretrained_backbone_weights=null" in args
    assert "--batch_size=64" in args
    assert "--steps=30000" in args
    assert "--save_checkpoint=true" in args
    assert "--wandb.enable=false" in args
    # Nothing may quietly pull a vision backbone or a GPU back in.
    assert "resnet" not in joined
    assert "cuda" not in joined


def test_action_chunking_config_is_internally_consistent(tmp_path):
    with pytest.raises(ValueError):
        make_config(tmp_path, chunk_size=10, n_action_steps=20)


def test_resume_rebuilds_the_run_from_the_last_checkpoint(tmp_path):
    config = make_config(tmp_path, resume=True)
    with pytest.raises(FileNotFoundError):
        resolve_resume(config)

    config_path = last_checkpoint_config(config.output_dir)
    config_path.parent.mkdir(parents=True)
    config_path.write_text("{}")

    args = resolve_resume(config).to_cli_args()
    assert f"--config_path={config_path}" in args
    assert "--resume=true" in args
    # Resuming must not respecify the dataset or the architecture: the
    # checkpoint's own config is the source of truth for those.
    assert not any(a.startswith("--dataset.") or a.startswith("--policy.") for a in args)


def test_train_command_invokes_lerobot_train(tmp_path):
    command = train_command(make_config(tmp_path))
    assert "lerobot" in command[0] or command[:3] == [
        command[0],
        "-m",
        "lerobot.scripts.lerobot_train",
    ]


def test_run_training_refuses_a_missing_dataset(tmp_path):
    with pytest.raises(FileNotFoundError, match="benchtop collect"):
        run_training(make_config(tmp_path))


def test_cli_dry_run_prints_the_command(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    result = runner.invoke(
        app,
        ["train", "--dataset", str(dataset), "--out", str(tmp_path / "out"), "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert "lerobot" in result.output
    assert f"--dataset.root={dataset}" in result.output


def test_cli_requires_a_dataset_unless_resuming():
    result = runner.invoke(app, ["train"])
    assert result.exit_code != 0
    # Typer renders errors in a box that wraps to the terminal width, which is
    # narrower on CI than locally.
    assert "--dataset is required" in _unwrap(result.output)
