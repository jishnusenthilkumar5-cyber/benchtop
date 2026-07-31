"""Aggregation of per-episode results into a scorecard.

The Wilson score interval is used rather than the normal approximation: at the
success rates and episode counts benchtop deals with (a policy that succeeds 3
times in 20), the normal interval is badly wrong and can extend below zero.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from benchtop.core.types import EpisodeResult, RunMetrics

#: Two-sided 95% normal quantile.
Z_95 = 1.959963984540054


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval on a binomial proportion, clamped to [0, 1].

    With no observations the interval is the whole unit interval.
    """
    if successes < 0 or n < 0 or successes > n:
        raise ValueError(f"invalid counts: {successes} successes out of {n}")
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1.0 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z**2 / (4 * n**2)) / denom
    # The interval is exactly [0, x] at zero successes and [x, 1] at all of
    # them; rounding would otherwise leave a bound a few ulps off the endpoint.
    low = 0.0 if successes == 0 else max(0.0, centre - half)
    high = 1.0 if successes == n else min(1.0, centre + half)
    return low, high


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def aggregate(episodes: Sequence[EpisodeResult], *, ndigits: int = 6) -> RunMetrics:
    """Summarise evaluated episodes. Raises on an empty run."""
    n = len(episodes)
    if n == 0:
        raise ValueError("cannot aggregate an empty run")
    successes = [e for e in episodes if e.success]
    steps_to_success = [
        float(e.time_to_success) for e in successes if e.time_to_success is not None
    ]
    low, high = wilson_interval(len(successes), n)
    return RunMetrics(
        episodes=n,
        successes=len(successes),
        success_rate=len(successes) / n,
        success_rate_ci_low=round(low, ndigits),
        success_rate_ci_high=round(high, ndigits),
        lift_rate=sum(1 for e in episodes if e.lifted) / n,
        mean_steps_to_success=_mean(steps_to_success) if steps_to_success else None,
        median_steps_to_success=_median(steps_to_success) if steps_to_success else None,
        mean_final_distance=round(_mean([e.final_distance for e in episodes]), ndigits),
    )
