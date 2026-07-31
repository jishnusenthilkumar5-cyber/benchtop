from __future__ import annotations

import pathlib

import pytest

from benchtop.eval.artifacts import read_run

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURE_RUN_DIR = REPO_ROOT / "tests" / "fixtures" / "runs" / "20260730T120000Z-random-pick_cube_v0"
SUITE_PATH = REPO_ROOT / "suites" / "pick_cube_v0.yaml"


@pytest.fixture
def fixture_run():
    """The committed reference run every artifact writer must conform to."""
    return read_run(FIXTURE_RUN_DIR)


@pytest.fixture
def suite_path():
    return SUITE_PATH
