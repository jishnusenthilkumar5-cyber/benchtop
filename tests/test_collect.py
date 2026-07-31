from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from benchtop.cli import app
from benchtop.core.types import (
    ACTION_DIM,
    ACTION_KEY,
    CONTROL_HZ,
    ENV_STATE_DIM,
    EVAL_SEED_START,
    OBS_KEY_ENV_STATE,
    OBS_KEY_STATE,
    STATE_DIM,
)
from benchtop.data.collect import (
    CollectionReport,
    collect_dataset,
    collect_dataset_from_seeds,
    dataset_features,
)

runner = CliRunner()


def test_features_mirror_the_frozen_spec():
    features = dataset_features()
    assert set(features) == {OBS_KEY_STATE, OBS_KEY_ENV_STATE, ACTION_KEY}
    assert features[OBS_KEY_STATE]["shape"] == (STATE_DIM,)
    assert features[OBS_KEY_ENV_STATE]["shape"] == (ENV_STATE_DIM,)
    assert features[ACTION_KEY]["shape"] == (ACTION_DIM,)
    for spec in features.values():
        assert spec["dtype"] == "float32"
        assert len(spec["names"]) == spec["shape"][0]


def test_collection_refuses_evaluation_seeds(tmp_path: Path):
    with pytest.raises(ValueError, match="held out"):
        collect_dataset(tmp_path / "ds", episodes=1, seed_start=EVAL_SEED_START)


def test_collection_refuses_to_overwrite(tmp_path: Path):
    existing = tmp_path / "ds"
    (existing / "data").mkdir(parents=True)
    (existing / "data" / "file.parquet").touch()
    with pytest.raises(FileExistsError):
        collect_dataset(existing, episodes=1)


def test_report_counts_kept_and_dropped():
    report = CollectionReport(
        path=Path("x"), repo_id="r", fps=CONTROL_HZ, kept_seeds=[0, 1, 2], dropped_seeds=[3]
    )
    assert (report.attempted, report.kept, report.dropped) == (4, 3, 1)
    assert report.success_rate == pytest.approx(0.75)
    assert "dropped 1" in report.summary()


def test_dataset_round_trips_through_lerobot(tmp_path: Path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out = tmp_path / "pick_cube_v0_expert"
    report = collect_dataset_from_seeds(out, [0, 1], repo_id="benchtop/test_expert")
    assert report.kept == 2, report.summary()
    assert report.frames > 0

    dataset = LeRobotDataset("benchtop/test_expert", root=out)
    assert dataset.num_episodes == 2
    assert dataset.num_frames == report.frames
    assert dataset.fps == CONTROL_HZ

    frame = dataset[0]
    assert frame[OBS_KEY_STATE].shape == (STATE_DIM,)
    assert frame[OBS_KEY_ENV_STATE].shape == (ENV_STATE_DIM,)
    assert frame[ACTION_KEY].shape == (ACTION_DIM,)
    assert np.isfinite(np.asarray(frame[ACTION_KEY])).all()


def test_cli_collects_into_the_requested_directory(tmp_path: Path):
    out = tmp_path / "demos"
    result = runner.invoke(
        app,
        ["collect", "--episodes", "1", "--seed-start", "0", "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert "kept 1/1 episodes" in result.output
    assert (out / "meta" / "info.json").exists()


def test_cli_rejects_a_non_positive_episode_count(tmp_path: Path):
    result = runner.invoke(app, ["collect", "--episodes", "0", "--out", str(tmp_path / "d")])
    assert result.exit_code != 0
