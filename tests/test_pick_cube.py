from __future__ import annotations

import numpy as np
import pytest

from benchtop.core.types import ACTION_DIM, DTYPE, ENV_STATE_DIM, MAX_EPISODE_STEPS, STATE_DIM
from benchtop.envs.pick_cube import (
    CUBE_REST_Z,
    CUBE_X_RANGE,
    CUBE_Y_RANGE,
    LIFT_HEIGHT,
    SUCCESS_HOLD_STEPS,
    SUCCESS_XY_TOL,
    TARGET_X_RANGE,
    TARGET_Y_RANGE,
    PickCubeV0,
)
from benchtop.policies.base import Policy
from benchtop.policies.simple import NoopPolicy, RandomPolicy


@pytest.fixture(scope="module")
def env():
    e = PickCubeV0()
    yield e
    e.close()


def test_observation_shapes_and_dtypes(env):
    obs, info = env.reset(seed=10_000)
    assert obs["state"].shape == (STATE_DIM,) and obs["state"].dtype == DTYPE
    assert obs["environment_state"].shape == (ENV_STATE_DIM,)
    assert obs["environment_state"].dtype == DTYPE
    assert env.observation_space.contains(obs)
    assert env.action_space.shape == (ACTION_DIM,)
    assert info["steps"] == 0 and not info["is_success"]


def test_reset_is_deterministic_and_seed_dependent(env):
    a, _ = env.reset(seed=7)
    b, _ = env.reset(seed=7)
    c, _ = env.reset(seed=8)
    assert np.array_equal(a["environment_state"], b["environment_state"])
    assert not np.array_equal(a["environment_state"], c["environment_state"])


def test_randomization_stays_in_range(env):
    for seed in range(20):
        obs, _ = env.reset(seed=seed)
        cube = obs["environment_state"][:3]
        target = obs["environment_state"][7:10]
        assert CUBE_X_RANGE[0] <= cube[0] <= CUBE_X_RANGE[1]
        assert CUBE_Y_RANGE[0] <= cube[1] <= CUBE_Y_RANGE[1]
        assert cube[2] == pytest.approx(CUBE_REST_Z, abs=1e-6)
        assert TARGET_X_RANGE[0] <= target[0] <= TARGET_X_RANGE[1]
        assert TARGET_Y_RANGE[0] <= target[1] <= TARGET_Y_RANGE[1]
        # The cube never starts inside the target zone.
        assert np.linalg.norm(cube[:2] - target[:2]) > SUCCESS_XY_TOL


def test_step_is_deterministic(env):
    def rollout(seed):
        env.reset(seed=seed)
        rng = np.random.default_rng(0)
        out = []
        for _ in range(10):
            action = rng.uniform(env.action_space.low, env.action_space.high)
            out.append(env.step(action)[0]["state"].copy())
        return np.stack(out)

    assert np.array_equal(rollout(10_000), rollout(10_000))


def test_success_detector_by_teleporting_the_cube(env):
    env.reset(seed=10_000)
    on_target = env.target_pos.copy()
    on_target[2] = CUBE_REST_Z

    # Placed but not yet held long enough.
    for _ in range(SUCCESS_HOLD_STEPS - 1):
        env.set_cube_pos(on_target)
    assert not env.is_success()
    env.set_cube_pos(on_target)
    assert env.is_success()

    # Leaving the target zone drops the hold counter immediately.
    away = on_target + np.array([0.0, 2 * SUCCESS_XY_TOL, 0.0])
    env.set_cube_pos(away)
    assert not env.is_success()

    # In the zone but hovering above the table is not a placement.
    hovering = on_target + np.array([0.0, 0.0, 0.2])
    for _ in range(SUCCESS_HOLD_STEPS + 1):
        env.set_cube_pos(hovering)
    assert not env.is_success()


def test_lift_event_latches(env):
    env.reset(seed=10_001)
    assert not env.lifted
    pos = env.cube_pos.copy()
    env.set_cube_pos(pos + np.array([0.0, 0.0, LIFT_HEIGHT + 0.02]))
    assert env.lifted
    env.set_cube_pos(pos)
    assert env.lifted
    env.reset(seed=10_001)
    assert not env.lifted


def test_truncates_at_the_episode_cap():
    env = PickCubeV0()
    try:
        obs, _ = env.reset(seed=10_002)
        policy = NoopPolicy()
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated):
            obs, _reward, terminated, truncated, _info = env.step(policy.act(obs))
            steps += 1
        assert (steps, truncated, terminated) == (MAX_EPISODE_STEPS, True, False)
    finally:
        env.close()


@pytest.mark.render
def test_render_returns_a_frame(env):
    env.reset(seed=10_000)
    frame = env.render()
    assert frame.shape == (480, 640, 3) and frame.dtype == np.uint8
    assert frame.std() > 0


def test_policies_satisfy_the_protocol(env):
    obs, _ = env.reset(seed=10_000)
    for policy in (
        RandomPolicy(env.action_space.low, env.action_space.high, seed=0),
        NoopPolicy(),
    ):
        assert isinstance(policy, Policy)
        policy.reset(0)
        action = policy.act(obs)
        assert action.shape == (ACTION_DIM,) and action.dtype == DTYPE
        assert env.action_space.contains(action)


def test_random_policy_is_seeded(env):
    obs, _ = env.reset(seed=10_000)
    p = RandomPolicy(env.action_space.low, env.action_space.high)
    p.reset(3)
    first = [p.act(obs) for _ in range(3)]
    p.reset(3)
    again = [p.act(obs) for _ in range(3)]
    assert all(np.array_equal(a, b) for a, b in zip(first, again, strict=True))
