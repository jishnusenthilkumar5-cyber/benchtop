from __future__ import annotations

import numpy as np
import pytest

from benchtop.core.types import ACTION_DIM, DTYPE, MAX_EPISODE_STEPS
from benchtop.envs.pick_cube import PickCubeV0
from benchtop.policies.base import Policy
from benchtop.policies.expert import GRIPPER_CLOSED, GRIPPER_OPEN, ExpertPolicy

#: Expected floor over the collection seeds; the plan's acceptance bar is 90%.
SUCCESS_BAR = 0.9


@pytest.fixture(scope="module")
def env() -> PickCubeV0:
    environment = PickCubeV0()
    yield environment
    environment.close()


@pytest.fixture
def expert(env: PickCubeV0) -> ExpertPolicy:
    return ExpertPolicy(env.model)


def run_episode(env: PickCubeV0, expert: ExpertPolicy, seed: int) -> dict:
    obs, _ = env.reset(seed=seed)
    expert.reset(seed)
    info: dict = {"is_success": False}
    for _ in range(MAX_EPISODE_STEPS):
        obs, _, terminated, truncated, info = env.step(expert.act(obs))
        if terminated or truncated:
            break
    return info


def test_expert_satisfies_the_policy_protocol(expert: ExpertPolicy):
    assert isinstance(expert, Policy)


def test_actions_match_the_frozen_spec(env: PickCubeV0, expert: ExpertPolicy):
    obs, _ = env.reset(seed=0)
    expert.reset(0)
    action = expert.act(obs)
    assert action.shape == (ACTION_DIM,)
    assert action.dtype == DTYPE
    assert env.action_space.contains(action)


def test_actions_stay_inside_the_action_space(env: PickCubeV0, expert: ExpertPolicy):
    obs, _ = env.reset(seed=3)
    expert.reset(3)
    for _ in range(120):
        action = expert.act(obs)
        assert env.action_space.contains(action), f"out of bounds in phase {expert.phase}"
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break


def test_gripper_command_is_open_or_closed(env: PickCubeV0, expert: ExpertPolicy):
    obs, _ = env.reset(seed=1)
    expert.reset(1)
    for _ in range(200):
        action = expert.act(obs)
        assert action[7] in (GRIPPER_OPEN, GRIPPER_CLOSED)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break


def test_phases_run_in_order(env: PickCubeV0, expert: ExpertPolicy):
    obs, _ = env.reset(seed=2)
    expert.reset(2)
    seen: list[str] = []
    for _ in range(MAX_EPISODE_STEPS):
        action = expert.act(obs)
        if not seen or seen[-1] != expert.phase:
            seen.append(expert.phase)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    assert (
        seen
        == [
            "pregrasp",
            "grasp",
            "close",
            "lift",
            "transit",
            "move",
            "place",
            "open",
            "retreat",
        ][: len(seen)]
    )
    assert "open" in seen, "the expert never got as far as releasing the cube"


def test_expert_is_deterministic_for_a_seed(env: PickCubeV0, expert: ExpertPolicy):
    def actions(seed: int) -> np.ndarray:
        obs, _ = env.reset(seed=seed)
        expert.reset(seed)
        out = []
        for _ in range(40):
            action = expert.act(obs)
            out.append(action)
            obs, _, _, _, _ = env.step(action)
        return np.asarray(out)

    assert np.array_equal(actions(7), actions(7))


def test_expert_picks_and_places_one_cube(env: PickCubeV0, expert: ExpertPolicy):
    info = run_episode(env, expert, seed=0)
    assert info["is_success"]
    assert info["lifted"]


def test_reset_replans_for_the_new_episode(env: PickCubeV0, expert: ExpertPolicy):
    assert run_episode(env, expert, seed=4)["is_success"]
    assert run_episode(env, expert, seed=5)["is_success"]


@pytest.mark.slow
def test_expert_clears_the_success_bar_on_collection_seeds(env: PickCubeV0, expert: ExpertPolicy):
    seeds = range(0, 50)
    successes = sum(run_episode(env, expert, seed)["is_success"] for seed in seeds)
    rate = successes / len(seeds)
    assert rate >= SUCCESS_BAR, f"expert success rate {rate:.0%} below the {SUCCESS_BAR:.0%} bar"
