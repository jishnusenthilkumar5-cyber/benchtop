from __future__ import annotations

import pathlib

import yaml
from typer.testing import CliRunner

from benchtop.cli import app
from benchtop.core.types import TASK_ID, assert_held_out

runner = CliRunner()
SUITE = pathlib.Path(__file__).resolve().parents[1] / "suites" / "pick_cube_v0.yaml"


def test_help_lists_every_verb():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for verb in ("collect", "train", "eval", "dash"):
        assert verb in result.stdout


def test_stubs_report_not_implemented():
    for verb in ("collect", "eval", "dash"):
        result = runner.invoke(app, [verb])
        assert result.exit_code == 0
        assert "not implemented" in result.stdout


def test_suite_uses_held_out_seeds():
    suite = yaml.safe_load(SUITE.read_text())
    assert suite["task"] == TASK_ID
    seeds = range(suite["seed_start"], suite["seed_start"] + suite["episodes"])
    assert_held_out(seeds)
