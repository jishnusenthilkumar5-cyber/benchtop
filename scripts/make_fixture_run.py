"""Regenerate the committed fixture run under tests/fixtures/runs/.

The fixture exists so the dashboard and artifact readers can be developed and
tested without running an eval. It is hand-rolled here (rather than produced
by the eval runner, which does not exist yet in Phase 0) and conforms to the
frozen schema in `benchtop.core.types`.

    uv run python scripts/make_fixture_run.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from benchtop.core.types import (
    EPISODES_FILENAME,
    MANIFEST_FILENAME,
    METRICS_FILENAME,
    TASK_ID,
    VIDEOS_DIRNAME,
    EpisodeResult,
    PolicyDescriptor,
    RunManifest,
    RunMetrics,
    video_filename,
)

RUN_ID = "20260730T120000Z-random-pick_cube_v0"
OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "runs" / RUN_ID

EPISODES = [
    EpisodeResult(10000, False, True, 500, None, 0.184, video_filename(0)),
    EpisodeResult(10001, True, True, 233, 223, 0.011, video_filename(1)),
    EpisodeResult(10002, False, False, 500, None, 0.262, video_filename(2)),
    EpisodeResult(10003, False, True, 500, None, 0.097, video_filename(3)),
    EpisodeResult(10004, False, False, 500, None, 0.301, video_filename(4)),
    EpisodeResult(10005, False, False, 500, None, 0.244, video_filename(5)),
]


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def metrics(episodes: list[EpisodeResult]) -> RunMetrics:
    n = len(episodes)
    successes = [e for e in episodes if e.success]
    tts = [e.time_to_success for e in successes if e.time_to_success is not None]
    low, high = wilson(len(successes), n)
    return RunMetrics(
        episodes=n,
        successes=len(successes),
        success_rate=len(successes) / n,
        success_rate_ci_low=round(low, 6),
        success_rate_ci_high=round(high, 6),
        lift_rate=sum(e.lifted for e in episodes) / n,
        mean_steps_to_success=float(np.mean(tts)) if tts else None,
        median_steps_to_success=float(np.median(tts)) if tts else None,
        mean_final_distance=round(float(np.mean([e.final_distance for e in episodes])), 6),
    )


def placeholder_video(path: Path, seed: int) -> None:
    """A 10-frame 64x48 clip. Small enough to commit, real enough to play."""
    rng = np.random.default_rng(seed)
    base = rng.integers(40, 90, size=(48, 64, 3), dtype=np.uint8)
    frames = []
    for i in range(10):
        frame = base.copy()
        x = 6 + 4 * i
        frame[20:28, x : x + 8] = (200, 60, 50)
        frames.append(frame)
    iio.imwrite(path, np.stack(frames), fps=10, codec="libx264", macro_block_size=1)


def main() -> None:
    (OUT / VIDEOS_DIRNAME).mkdir(parents=True, exist_ok=True)

    manifest = RunManifest(
        run_id=RUN_ID,
        task=TASK_ID,
        policy=PolicyDescriptor(spec="random", type="random"),
        suite={
            "name": "pick_cube_v0",
            "task": TASK_ID,
            "max_steps": 500,
            "video": True,
            "seeds": [e.seed for e in EPISODES],
        },
        git_sha="0000000000000000000000000000000000000000",
        benchtop_version="0.1.0",
        platform="Linux-x86_64-py3.12",
        started_at="2026-07-30T12:00:00+00:00",
        finished_at="2026-07-30T12:03:20+00:00",
        duration_s=200.0,
        extra={"fixture": True, "video": True},
    )
    (OUT / MANIFEST_FILENAME).write_text(json.dumps(manifest.to_dict(), indent=2) + "\n")
    (OUT / METRICS_FILENAME).write_text(json.dumps(metrics(EPISODES).to_dict(), indent=2) + "\n")
    (OUT / EPISODES_FILENAME).write_text("".join(json.dumps(e.to_dict()) + "\n" for e in EPISODES))
    for i, episode in enumerate(EPISODES):
        placeholder_video(OUT / VIDEOS_DIRNAME / video_filename(i), episode.seed)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
