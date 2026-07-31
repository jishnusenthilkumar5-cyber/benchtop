from __future__ import annotations

import math

import pytest

from benchtop.core.types import EpisodeResult
from benchtop.eval.metrics import Z_95, aggregate, wilson_interval


def episode(seed: int, success: bool, lifted: bool = True, steps: int = 500, tts=None, dist=0.1):
    return EpisodeResult(
        seed=seed,
        success=success,
        lifted=lifted,
        steps=steps,
        time_to_success=tts,
        final_distance=dist,
    )


# Reference values obtained independently of the implementation, as the roots
# of (p_hat - p)^2 = z^2 p (1 - p) / n -- the score equation the Wilson
# interval solves -- to 6 dp.
@pytest.mark.parametrize(
    ("successes", "n", "low", "high"),
    [
        (0, 10, 0.0, 0.277533),
        (10, 10, 0.722467, 1.0),
        (5, 10, 0.236593, 0.763407),
        (1, 20, 0.008881, 0.236131),
        (3, 20, 0.052369, 0.360419),
        (50, 100, 0.403832, 0.596168),
        (0, 1, 0.0, 0.793451),
        (1, 1, 0.206549, 1.0),
    ],
)
def test_wilson_matches_reference_values(successes, n, low, high):
    got_low, got_high = wilson_interval(successes, n)
    assert got_low == pytest.approx(low, abs=1e-6)
    assert got_high == pytest.approx(high, abs=1e-6)


def test_wilson_matches_closed_form():
    """The implementation agrees with the textbook formula written out directly."""
    for successes, n in [(0, 7), (2, 7), (7, 7), (13, 40), (99, 100)]:
        p = successes / n
        z = Z_95
        centre = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = (z / (1 + z * z / n)) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        assert wilson_interval(successes, n) == pytest.approx((centre - half, centre + half))


def test_wilson_is_clamped_and_ordered():
    for n in range(1, 30):
        for k in range(n + 1):
            low, high = wilson_interval(k, n)
            assert 0.0 <= low <= k / n <= high <= 1.0


def test_wilson_narrows_with_more_data():
    widths = [
        wilson_interval(n // 2, n)[1] - wilson_interval(n // 2, n)[0] for n in (10, 100, 1000)
    ]
    assert widths == sorted(widths, reverse=True)


def test_wilson_is_asymmetric_at_the_boundary():
    """Unlike the normal approximation, the interval never leaves [0, 1] at p=0."""
    low, high = wilson_interval(0, 30)
    assert low == 0.0
    assert 0.0 < high < 0.2


def test_wilson_no_observations_is_the_unit_interval():
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_rejects_impossible_counts():
    with pytest.raises(ValueError):
        wilson_interval(3, 2)
    with pytest.raises(ValueError):
        wilson_interval(-1, 2)


def test_aggregate_matches_the_committed_fixture(fixture_run):
    metrics = aggregate(list(fixture_run.episodes))
    assert metrics.to_dict() == fixture_run.metrics.to_dict()


def test_aggregate_success_and_lift_rates():
    episodes = [
        episode(10000, True, lifted=True, steps=100, tts=90, dist=0.01),
        episode(10001, False, lifted=True, dist=0.2),
        episode(10002, False, lifted=False, dist=0.3),
        episode(10003, True, lifted=True, steps=200, tts=150, dist=0.02),
    ]
    m = aggregate(episodes)
    assert (m.episodes, m.successes) == (4, 2)
    assert m.success_rate == 0.5
    assert m.lift_rate == 0.75
    assert m.mean_steps_to_success == 120.0
    assert m.median_steps_to_success == 120.0
    assert m.mean_final_distance == pytest.approx(0.1325)
    assert m.success_rate_ci_low < m.success_rate < m.success_rate_ci_high


def test_aggregate_without_successes_has_no_step_stats():
    m = aggregate([episode(10000, False), episode(10001, False)])
    assert m.successes == 0
    assert m.mean_steps_to_success is None
    assert m.median_steps_to_success is None
    assert m.success_rate_ci_low == 0.0


def test_aggregate_median_over_even_count():
    episodes = [
        episode(10000, True, tts=10),
        episode(10001, True, tts=20),
        episode(10002, True, tts=30),
        episode(10003, True, tts=100),
    ]
    m = aggregate(episodes)
    assert m.median_steps_to_success == 25.0
    assert m.mean_steps_to_success == 40.0


def test_aggregate_rejects_empty_run():
    with pytest.raises(ValueError):
        aggregate([])
