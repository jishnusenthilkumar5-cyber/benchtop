from __future__ import annotations

import re

import pytest
from typer.testing import CliRunner

from benchtop.cli import app
from benchtop.core.types import MANIFEST_FILENAME, METRICS_FILENAME
from benchtop.eval.artifacts import read_run

runner = CliRunner()


def test_eval_help_documents_the_selectors():
    # Rich wraps and colours help text to the terminal width, which differs
    # between a local shell and CI; pin the width and strip the escapes.
    result = runner.invoke(app, ["eval", "--help"], env={"COLUMNS": "200", "NO_COLOR": "1"})
    assert result.exit_code == 0
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout)
    for token in ("--suite", "--policy", "random", "lerobot"):
        assert token in plain


def test_missing_suite_exits_nonzero(tmp_path):
    result = runner.invoke(app, ["eval", "--suite", str(tmp_path / "nope.yaml")])
    assert result.exit_code == 2


def test_suite_with_demo_seeds_is_refused(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: bad\ntask: pick_cube-v0\nseeds: [0, 1]\n")
    result = runner.invoke(app, ["eval", "--suite", str(bad)])
    assert result.exit_code == 2


@pytest.mark.render
@pytest.mark.slow
def test_eval_random_end_to_end(tmp_path, suite_path):
    """The real thing: MuJoCo rollouts, video capture and a run directory."""
    result = runner.invoke(
        app,
        [
            "eval",
            "--suite",
            str(suite_path),
            "--policy",
            "random",
            "--episodes",
            "2",
            "--runs-dir",
            str(tmp_path),
            "--run-id",
            "e2e",
        ],
    )
    assert result.exit_code == 0, result.output
    run = read_run(tmp_path / "e2e")
    assert run.metrics.episodes == 2
    assert [e.seed for e in run.episodes] == [10000, 10001]
    assert all((tmp_path / "e2e" / "videos" / e.video).stat().st_size > 0 for e in run.episodes)
    assert (tmp_path / "e2e" / MANIFEST_FILENAME).exists()
    assert (tmp_path / "e2e" / METRICS_FILENAME).exists()


@pytest.mark.slow
def test_eval_random_without_video(tmp_path, suite_path):
    """No GL context needed: the same loop with video capture off."""
    result = runner.invoke(
        app,
        [
            "eval",
            "--suite",
            str(suite_path),
            "--policy",
            "noop",
            "--episodes",
            "1",
            "--no-video",
            "--runs-dir",
            str(tmp_path),
            "--run-id",
            "novideo",
        ],
    )
    assert result.exit_code == 0, result.output
    run = read_run(tmp_path / "novideo")
    assert run.metrics.episodes == 1
    assert run.manifest.policy.type == "noop"
