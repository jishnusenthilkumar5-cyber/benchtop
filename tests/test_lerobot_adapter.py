from __future__ import annotations

import numpy as np
import pytest

from benchtop.adapters.lerobot_policy import (
    LeRobotACTPolicy,
    resolve_checkpoint_dir,
    save_untrained_checkpoint,
)
from benchtop.core.types import ACTION_DIM, DTYPE, ENV_STATE_DIM, STATE_DIM, Observation

#: Small enough that building one costs a second, large enough to be a real ACT.
TINY = {
    "chunk_size": 10,
    "n_action_steps": 10,
    "dim_model": 32,
    "n_heads": 4,
    "dim_feedforward": 64,
    "n_encoder_layers": 1,
    "n_decoder_layers": 1,
    "n_vae_encoder_layers": 1,
}


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    return save_untrained_checkpoint(tmp_path_factory.mktemp("act") / "pretrained_model", **TINY)


@pytest.fixture(scope="module")
def policy(checkpoint):
    return LeRobotACTPolicy(checkpoint)


def fabricate_obs(rng: np.random.Generator) -> Observation:
    return {
        "state": rng.normal(size=STATE_DIM).astype(DTYPE),
        "environment_state": rng.normal(size=ENV_STATE_DIM).astype(DTYPE),
    }


def test_act_returns_a_spec_shaped_action(policy):
    rng = np.random.default_rng(0)
    policy.reset(0)
    action = policy.act(fabricate_obs(rng))
    assert action.shape == (ACTION_DIM,)
    assert action.dtype == DTYPE
    assert np.isfinite(action).all()


def test_action_chunk_is_consumed_then_refilled(policy):
    """`n_action_steps` actions come from one forward pass, the next from another."""
    rng = np.random.default_rng(1)
    policy.reset(0)
    obs = fabricate_obs(rng)
    actions = [policy.act(obs) for _ in range(policy.n_action_steps + 1)]
    # Same observation, so a re-query returns the head of a fresh chunk: the
    # queue was drained, not silently reused.
    assert np.allclose(actions[0], actions[policy.n_action_steps])
    assert not np.allclose(actions[0], actions[1])


def test_reset_drops_the_previous_episodes_actions(policy):
    rng = np.random.default_rng(2)
    obs = fabricate_obs(rng)
    policy.reset(0)
    first = policy.act(obs)
    policy.act(obs)
    policy.reset(1)
    assert np.allclose(policy.act(obs), first)


def test_resolve_accepts_run_dir_checkpoint_dir_and_model_dir(tmp_path, checkpoint):
    run_dir = tmp_path / "run"
    step_dir = run_dir / "checkpoints" / "000100"
    step_dir.mkdir(parents=True)
    (step_dir / "pretrained_model").symlink_to(checkpoint, target_is_directory=True)
    (run_dir / "checkpoints" / "last").symlink_to(step_dir, target_is_directory=True)

    expected = checkpoint.resolve()
    assert resolve_checkpoint_dir(run_dir) == expected
    assert resolve_checkpoint_dir(step_dir) == expected
    assert resolve_checkpoint_dir(step_dir / "pretrained_model") == expected


def test_resolve_rejects_a_directory_that_is_not_a_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint_dir(tmp_path)
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint_dir(tmp_path / "nope")


def test_rejects_a_checkpoint_that_disagrees_with_the_frozen_spec(tmp_path):
    from lerobot.configs.types import FeatureType, PolicyFeature

    from benchtop.core.types import OBS_KEY_STATE

    directory = save_untrained_checkpoint(
        tmp_path / "wrong",
        input_features={
            OBS_KEY_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
            "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(4,)),
        },
        **TINY,
    )
    with pytest.raises(ValueError, match="environment_state"):
        LeRobotACTPolicy(directory)


def test_eval_registry_resolves_a_lerobot_selector(checkpoint):
    """`--policy lerobot:<path>` reaches this adapter without WI-2 knowing it exists."""
    from benchtop.envs.pick_cube import PickCubeV0
    from benchtop.eval.registry import describe, resolve_policy

    spec = f"lerobot:{checkpoint}"
    assert describe(spec).type == "lerobot"

    env = PickCubeV0()
    try:
        resolved = resolve_policy(spec, env, seed=10_000)
    finally:
        env.close()
    assert isinstance(resolved, LeRobotACTPolicy)


def test_twenty_step_rollout_in_pick_cube(policy):
    from benchtop.envs.pick_cube import PickCubeV0

    env = PickCubeV0()
    try:
        obs, _ = env.reset(seed=10_000)
        policy.reset(10_000)
        for _ in range(20):
            action = policy.act(obs)
            assert env.action_space.contains(
                np.clip(action, env.action_space.low, env.action_space.high)
            )
            obs, _reward, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                break
    finally:
        env.close()
