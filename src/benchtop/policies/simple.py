"""Reference policies: the floor (`random`) and a do-nothing control (`noop`)."""

from __future__ import annotations

import numpy as np

from benchtop.core.types import ACTION_DIM, DTYPE, Observation


class RandomPolicy:
    """Uniform samples from the action bounds. The baseline floor."""

    def __init__(self, low: np.ndarray, high: np.ndarray, seed: int | None = None) -> None:
        self.low = np.asarray(low, dtype=DTYPE)
        self.high = np.asarray(high, dtype=DTYPE)
        self._rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, obs: Observation) -> np.ndarray:
        return self._rng.uniform(self.low, self.high).astype(DTYPE)


class NoopPolicy:
    """Holds the joint targets at the current joint positions, gripper open.

    Not a zero action: zeros are a valid joint-target command and would make
    the arm move. This one is the closest thing to "do nothing".
    """

    #: Gripper actuator is remapped to 0-255 in panda.xml; 255 is fully open.
    GRIPPER_OPEN = 255.0

    def reset(self, seed: int | None = None) -> None:
        return None

    def act(self, obs: Observation) -> np.ndarray:
        action = np.empty(ACTION_DIM, dtype=DTYPE)
        action[:7] = obs["state"][:7]
        action[7] = self.GRIPPER_OPEN
        return action
