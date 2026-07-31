"""Training entrypoint: state-only ACT on CPU, wrapping `lerobot-train`."""

from __future__ import annotations

from benchtop.train.config import TrainConfig, last_checkpoint_config
from benchtop.train.runner import run_training, train_command

__all__ = ["TrainConfig", "last_checkpoint_config", "run_training", "train_command"]
