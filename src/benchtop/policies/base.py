"""The Policy protocol every evaluated policy implements."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from benchtop.core.types import Observation


@runtime_checkable
class Policy(Protocol):
    """Maps observations to actions.

    `reset` is called once per episode, before the first `act`, with the
    episode's seed so that stochastic policies are reproducible too.
    """

    def reset(self, seed: int | None = None) -> None: ...

    def act(self, obs: Observation) -> np.ndarray:
        """Return an 8-dim float32 action for the given observation."""
        ...
