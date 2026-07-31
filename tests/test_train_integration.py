"""End-to-end: train a few steps, resume, then load the checkpoint as a Policy.

Marked `slow`: it shells out to `lerobot-train` twice. It is the only test that
proves the three pieces of WI-4 actually fit together -- the CLI's argument
list, what `lerobot-train` writes to disk, and what the adapter reads back.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchtop.adapters.lerobot_policy import LeRobotACTPolicy
from benchtop.core.types import (
    ACTION_DIM,
    ACTION_KEY,
    DTYPE,
    ENV_STATE_DIM,
    OBS_KEY_ENV_STATE,
    OBS_KEY_STATE,
    STATE_DIM,
)
from benchtop.train.config import TrainConfig
from benchtop.train.runner import run_training

pytestmark = pytest.mark.slow

STEPS = 4
EPISODES = 2
FRAMES_PER_EPISODE = 20

#: Smallest ACT that still exercises every code path.
TINY_ARCH = {
    "chunk_size": 5,
    "n_action_steps": 5,
    "dim_model": 32,
    "n_heads": 4,
    "dim_feedforward": 64,
    "n_encoder_layers": 1,
    "n_decoder_layers": 1,
}


def synthetic_dataset(root, repo_id: str):
    """A spec-shaped LeRobotDataset of random frames.

    Deliberately not `benchtop collect`: this test is about the training and
    checkpoint plumbing, and it must not wait on a MuJoCo rollout to run.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        OBS_KEY_STATE: {"dtype": "float32", "shape": (STATE_DIM,), "names": None},
        OBS_KEY_ENV_STATE: {"dtype": "float32", "shape": (ENV_STATE_DIM,), "names": None},
        ACTION_KEY: {"dtype": "float32", "shape": (ACTION_DIM,), "names": None},
    }
    dataset = LeRobotDataset.create(
        repo_id=repo_id, fps=50, root=root, features=features, use_videos=False
    )
    rng = np.random.default_rng(0)
    for _ in range(EPISODES):
        for _ in range(FRAMES_PER_EPISODE):
            dataset.add_frame(
                {
                    OBS_KEY_STATE: rng.normal(size=STATE_DIM).astype(DTYPE),
                    OBS_KEY_ENV_STATE: rng.normal(size=ENV_STATE_DIM).astype(DTYPE),
                    ACTION_KEY: rng.normal(size=ACTION_DIM).astype(DTYPE),
                    "task": "pick the cube",
                }
            )
        dataset.save_episode()
    dataset.finalize()
    return root


def test_train_resume_and_load_the_checkpoint(tmp_path):
    dataset = synthetic_dataset(tmp_path / "dataset", "benchtop/test_pick_cube_v0")
    output_dir = tmp_path / "out"
    config = TrainConfig(
        dataset=dataset,
        output_dir=output_dir,
        repo_id="benchtop/test_pick_cube_v0",
        batch_size=4,
        steps=STEPS,
        num_workers=0,
        save_freq=STEPS,
        **TINY_ARCH,
    )
    assert run_training(config) == 0
    assert config.has_resumable_checkpoint()

    # Resuming picks up where the run stopped rather than restarting it: the
    # extra steps land in a checkpoint the first run never wrote.
    resumed = TrainConfig(
        dataset=dataset,
        output_dir=output_dir,
        steps=STEPS * 2,
        save_freq=STEPS,
        resume=True,
    )
    assert run_training(resumed) == 0
    steps_dir = output_dir / "checkpoints"
    assert {p.name for p in steps_dir.iterdir()} >= {f"{STEPS:06d}", f"{STEPS * 2:06d}"}

    policy = LeRobotACTPolicy(output_dir)
    policy.reset(0)
    action = policy.act(
        {
            "state": np.zeros(STATE_DIM, dtype=DTYPE),
            "environment_state": np.zeros(ENV_STATE_DIM, dtype=DTYPE),
        }
    )
    assert action.shape == (ACTION_DIM,)
    assert action.dtype == DTYPE
    assert np.isfinite(action).all()
