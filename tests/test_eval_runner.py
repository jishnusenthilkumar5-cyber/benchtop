from __future__ import annotations

import json

import numpy as np
import pytest

from benchtop.core.types import (
    ACTION_DIM,
    DTYPE,
    ENV_STATE_DIM,
    EPISODES_FILENAME,
    MANIFEST_FILENAME,
    METRICS_FILENAME,
    STATE_DIM,
    TASK_ID,
    VIDEOS_DIRNAME,
)
from benchtop.eval import artifacts
from benchtop.eval.runner import EvalConfig, rollout, run_suite
from benchtop.eval.suite import Suite


class StubEnv:
    """A PickCubeV0 stand-in: same API, no MuJoCo and no GL context.

    Succeeds on even seeds after 20 steps, so the loop's success latching,
    episode caps and video wiring are testable in the default test run.
    """

    def __init__(self, success_step: int = 20, cap: int = 500):
        self.success_step = success_step
        self.cap = cap
        bounds = {"low": np.zeros(ACTION_DIM), "high": np.ones(ACTION_DIM)}
        self.action_space = type("Box", (), bounds)()
        self.seeds: list[int] = []
        self.closed = False
        self._steps = 0
        self._seed = 0

    def reset(self, *, seed=None, options=None):
        self.seeds.append(seed)
        self._seed = seed or 0
        self._steps = 0
        return self._obs(), self._info()

    def step(self, action):
        assert np.asarray(action).shape == (ACTION_DIM,)
        self._steps += 1
        info = self._info()
        terminated = info["is_success"]
        truncated = self._steps >= self.cap and not terminated
        return self._obs(), float(terminated), terminated, truncated, info

    def render(self):
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def close(self):
        self.closed = True

    def _succeeds(self):
        return self._seed % 2 == 0 and self._steps >= self.success_step

    def _obs(self):
        return {
            "state": np.zeros(STATE_DIM, dtype=DTYPE),
            "environment_state": np.zeros(ENV_STATE_DIM, dtype=DTYPE),
        }

    def _info(self):
        return {
            "is_success": self._succeeds(),
            "lifted": self._seed % 3 == 0,
            "distance": 0.5 if not self._succeeds() else 0.01,
            "steps": self._steps,
        }


def suite(seeds=(10000, 10001, 10002), video=False, max_steps=50):
    return Suite(name="stub", task=TASK_ID, seeds=tuple(seeds), max_steps=max_steps, video=video)


def test_rollout_latches_first_success_step():
    env = StubEnv(success_step=5)
    policy = _random_policy()
    result, frames = rollout(env, policy, seed=10000, max_steps=50)
    assert result.success is True
    assert result.time_to_success == 5
    assert result.steps == 5  # env terminates on success
    assert frames == []


def test_rollout_honours_the_episode_cap():
    env = StubEnv(success_step=10_000, cap=10_000)
    result, _ = rollout(env, _random_policy(), seed=10001, max_steps=7)
    assert result.steps == 7
    assert result.success is False
    assert result.time_to_success is None


def test_rollout_captures_one_frame_per_step_plus_the_reset_frame():
    env = StubEnv(success_step=4)
    _, frames = rollout(env, _random_policy(), seed=10000, max_steps=50, capture=True)
    assert len(frames) == 5


def test_run_writes_a_fixture_conformant_run_directory(tmp_path):
    env = StubEnv(success_step=5)
    run_dir = run_suite(
        suite(),
        EvalConfig(policy_spec="random", runs_dir=tmp_path, run_id="testrun"),
        env=env,
    )
    assert run_dir == tmp_path / "testrun"
    assert {p.name for p in run_dir.iterdir()} == {
        MANIFEST_FILENAME,
        METRICS_FILENAME,
        EPISODES_FILENAME,
    }

    run = artifacts.read_run(run_dir)
    assert [e.seed for e in run.episodes] == [10000, 10001, 10002]
    assert run.metrics.episodes == 3
    assert run.metrics.successes == 2
    assert run.metrics.success_rate_ci_low < run.metrics.success_rate
    assert run.manifest.task == TASK_ID
    assert run.manifest.policy.spec == "random"
    assert run.manifest.suite["seeds"] == [10000, 10001, 10002]
    assert run.manifest.git_sha and run.manifest.benchtop_version == "0.1.0"
    assert run.manifest.duration_s >= 0.0
    assert run.manifest.extra["video"] is False
    assert env.seeds == [10000, 10001, 10002]
    assert env.closed is False  # a caller-supplied env is the caller's to close


def test_run_directory_matches_the_committed_fixture_shape(tmp_path, fixture_run):
    run_dir = run_suite(
        suite(video=True),
        EvalConfig(policy_spec="random", runs_dir=tmp_path, run_id="testrun"),
        env=StubEnv(success_step=5),
    )
    written = json.loads((run_dir / MANIFEST_FILENAME).read_text())
    assert set(written) == set(json.loads((fixture_run.path / MANIFEST_FILENAME).read_text()))
    assert set(written["policy"]) <= {"spec", "type", "checkpoint_path", "checkpoint_sha256"}
    assert set(json.loads((run_dir / METRICS_FILENAME).read_text())) == set(
        json.loads((fixture_run.path / METRICS_FILENAME).read_text())
    )
    episode = json.loads((run_dir / EPISODES_FILENAME).read_text().splitlines()[0])
    fixture_episode = json.loads((fixture_run.path / EPISODES_FILENAME).read_text().splitlines()[1])
    assert set(episode) == set(fixture_episode)


def test_video_capture_writes_playable_clips_named_by_episode_index(tmp_path):
    run_dir = run_suite(
        suite(seeds=(10000, 10001), video=True),
        EvalConfig(policy_spec="random", runs_dir=tmp_path, run_id="vid"),
        env=StubEnv(success_step=3),
    )
    videos = sorted(p.name for p in (run_dir / VIDEOS_DIRNAME).iterdir())
    assert videos == ["ep000.mp4", "ep001.mp4"]
    assert all(p.stat().st_size > 0 for p in (run_dir / VIDEOS_DIRNAME).iterdir())
    run = artifacts.read_run(run_dir)
    assert [e.video for e in run.episodes] == videos


def test_video_flag_can_be_overridden_per_run(tmp_path):
    run_dir = run_suite(
        suite(video=True),
        EvalConfig(policy_spec="random", runs_dir=tmp_path, run_id="novid", video=False),
        env=StubEnv(),
    )
    assert not (run_dir / VIDEOS_DIRNAME).exists()
    run = artifacts.read_run(run_dir)
    assert run.manifest.suite["video"] is False
    assert all(e.video is None for e in run.episodes)


def test_episodes_are_appended_while_the_run_is_in_flight(tmp_path):
    seen: list[int] = []

    def progress(_line: str) -> None:
        seen.append(len((tmp_path / "live" / EPISODES_FILENAME).read_text().splitlines()))

    run_suite(
        suite(),
        EvalConfig(policy_spec="random", runs_dir=tmp_path, run_id="live", progress=progress),
        env=StubEnv(),
    )
    assert seen == [1, 2, 3]


def test_run_is_deterministic_for_a_seed_list(tmp_path):
    dirs = [
        run_suite(
            suite(),
            EvalConfig(policy_spec="random", runs_dir=tmp_path, run_id=f"r{i}"),
            env=StubEnv(),
        )
        for i in range(2)
    ]
    assert (dirs[0] / EPISODES_FILENAME).read_text() == (dirs[1] / EPISODES_FILENAME).read_text()


def test_unknown_policy_fails_before_the_environment_is_built(tmp_path):
    with pytest.raises(ValueError, match="unknown policy"):
        run_suite(suite(), EvalConfig(policy_spec="nope", runs_dir=tmp_path), env=StubEnv())


def test_run_ids_are_sortable_and_descriptive():
    run_id = artifacts.make_run_id("lerobot:/tmp/ckpt-1", "pick_cube_v0")
    assert run_id.endswith("-lerobot_tmp_ckpt_1-pick_cube_v0")
    assert run_id[8] == "T"


def _random_policy():
    from benchtop.policies.simple import RandomPolicy

    return RandomPolicy(np.zeros(ACTION_DIM), np.ones(ACTION_DIM), seed=0)
