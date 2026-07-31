"""The spec is frozen: these tests exist to make a change to it loud."""

from __future__ import annotations

import json

import pytest

from benchtop.core import types
from benchtop.core.types import (
    EpisodeResult,
    PolicyDescriptor,
    RunManifest,
    RunMetrics,
    assert_held_out,
    read_episodes,
)

FIXTURE_RUN = (
    __import__("pathlib").Path(__file__).parent
    / "fixtures"
    / "runs"
    / "20260730T120000Z-random-pick_cube_v0"
)


def test_frozen_dimensions():
    assert (types.STATE_DIM, types.ENV_STATE_DIM, types.ACTION_DIM) == (15, 10, 8)
    assert len(types.STATE_NAMES) == types.STATE_DIM
    assert len(types.ENV_STATE_NAMES) == types.ENV_STATE_DIM
    assert len(types.ACTION_NAMES) == types.ACTION_DIM
    assert types.TASK_ID == "pick_cube-v0"
    assert types.CONTROL_HZ == 50
    assert types.PHYSICS_DT * types.CONTROL_DECIMATION == pytest.approx(1 / types.CONTROL_HZ)


def test_seed_ranges_are_disjoint():
    assert types.DEMO_SEED_END <= types.EVAL_SEED_START
    assert types.is_demo_seed(0) and types.is_demo_seed(99)
    assert not types.is_demo_seed(10_000)
    assert types.is_eval_seed(10_000) and not types.is_eval_seed(99)
    assert_held_out([10_000, 10_042])
    with pytest.raises(ValueError):
        assert_held_out([10_000, 42])


def test_round_trip():
    episode = EpisodeResult(10_001, True, True, 233, 223, 0.011, "ep001.mp4")
    assert EpisodeResult.from_dict(json.loads(json.dumps(episode.to_dict()))) == episode

    failed = EpisodeResult(10_000, False, False, 500, None, 0.3)
    assert "video" not in failed.to_dict()
    assert EpisodeResult.from_dict(failed.to_dict()) == failed


def test_fixture_run_conforms_to_schema():
    manifest = RunManifest.from_dict(json.loads((FIXTURE_RUN / "manifest.json").read_text()))
    assert manifest.task == types.TASK_ID
    assert manifest.schema_version == types.ARTIFACT_SCHEMA_VERSION
    assert isinstance(manifest.policy, PolicyDescriptor)

    metrics = RunMetrics.from_dict(json.loads((FIXTURE_RUN / "metrics.json").read_text()))
    episodes = list(read_episodes(FIXTURE_RUN / "episodes.jsonl"))
    assert metrics.episodes == len(episodes)
    assert metrics.successes == sum(e.success for e in episodes)
    assert metrics.success_rate_ci_low <= metrics.success_rate <= metrics.success_rate_ci_high

    assert_held_out(e.seed for e in episodes)
    for i, episode in enumerate(episodes):
        assert episode.video == types.video_filename(i)
        assert (FIXTURE_RUN / types.VIDEOS_DIRNAME / episode.video).is_file()
