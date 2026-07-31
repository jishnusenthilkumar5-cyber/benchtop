"""The episode loop: seeded resets, policy stepping, optional video capture."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from benchtop.core.types import (
    CONTROL_HZ,
    VIDEOS_DIRNAME,
    EpisodeResult,
    PolicyDescriptor,
    RunManifest,
    RunMetrics,
    video_filename,
)
from benchtop.eval import artifacts
from benchtop.eval.metrics import aggregate
from benchtop.eval.registry import describe, resolve_policy
from benchtop.eval.suite import Suite

#: A policy is rebuilt per run, not per episode; `reset(seed)` reseeds it.
PolicyFactory = Callable[[Any], Any]


@dataclass(frozen=True, slots=True)
class EvalConfig:
    """Everything a run needs beyond the suite itself."""

    policy_spec: str
    runs_dir: Path
    #: None means "whatever the suite says".
    video: bool | None = None
    run_id: str | None = None
    #: Seed handed to the policy constructor; episodes reseed it per seed.
    policy_seed: int | None = 0
    progress: Callable[[str], None] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def rollout(
    env: Any,
    policy: Any,
    seed: int,
    max_steps: int,
    capture: bool = False,
) -> tuple[EpisodeResult, list[np.ndarray]]:
    """Run one episode and return its result plus any captured frames."""
    obs, info = env.reset(seed=seed)
    policy.reset(seed)

    frames: list[np.ndarray] = []
    if capture:
        frames.append(env.render())

    time_to_success: int | None = None
    steps = 0
    for step in range(1, max_steps + 1):
        obs, _reward, terminated, truncated, info = env.step(policy.act(obs))
        steps = step
        if capture:
            frames.append(env.render())
        if info["is_success"] and time_to_success is None:
            time_to_success = step
        if terminated or truncated:
            break

    result = EpisodeResult(
        seed=seed,
        success=bool(info["is_success"]),
        lifted=bool(info["lifted"]),
        steps=steps,
        time_to_success=time_to_success,
        final_distance=round(float(info["distance"]), 6),
    )
    return result, frames


def _make_env(video: bool) -> Any:
    from benchtop.envs.pick_cube import PickCubeV0

    return PickCubeV0(render_mode="rgb_array" if video else None)


def run_suite(
    suite: Suite,
    config: EvalConfig,
    env: Any | None = None,
) -> Path:
    """Evaluate a policy over a suite and write a run directory. Returns its path."""
    video = suite.video if config.video is None else config.video
    run_id = config.run_id or artifacts.make_run_id(config.policy_spec, suite.name)
    run_dir = Path(config.runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if video:
        (run_dir / VIDEOS_DIRNAME).mkdir(exist_ok=True)

    descriptor: PolicyDescriptor = describe(config.policy_spec)
    owns_env = env is None
    env = env if env is not None else _make_env(video)

    started = datetime.now(UTC)
    clock = time.perf_counter()
    episodes: list[EpisodeResult] = []
    try:
        policy = resolve_policy(config.policy_spec, env, config.policy_seed)
        for index, seed in enumerate(suite.seeds):
            result, frames = rollout(env, policy, seed, suite.max_steps, capture=video)
            if video and frames:
                artifacts.write_video(
                    run_dir / VIDEOS_DIRNAME / video_filename(index), frames, CONTROL_HZ
                )
                result = _with_video(result, video_filename(index))
            episodes.append(result)
            artifacts.append_episode(run_dir, result)
            if config.progress is not None:
                config.progress(_progress_line(index, len(suite.seeds), result))
    finally:
        if owns_env:
            env.close()

    finished = datetime.now(UTC)
    metrics: RunMetrics = aggregate(episodes)
    artifacts.write_episodes(run_dir, episodes)
    artifacts.write_metrics(run_dir, metrics)
    artifacts.write_manifest(
        run_dir,
        RunManifest(
            run_id=run_id,
            task=suite.task,
            policy=descriptor,
            suite={**suite.to_dict(), "video": video},
            git_sha=artifacts.git_sha(),
            benchtop_version=artifacts.benchtop_version(),
            platform=artifacts.platform_string(),
            started_at=started.isoformat(),
            finished_at=finished.isoformat(),
            duration_s=round(time.perf_counter() - clock, 3),
            extra={"video": video, **config.extra},
        ),
    )
    return run_dir


def _with_video(result: EpisodeResult, name: str) -> EpisodeResult:
    return EpisodeResult(
        seed=result.seed,
        success=result.success,
        lifted=result.lifted,
        steps=result.steps,
        time_to_success=result.time_to_success,
        final_distance=result.final_distance,
        video=name,
    )


def _progress_line(index: int, total: int, result: EpisodeResult) -> str:
    mark = "success" if result.success else "failure"
    return (
        f"[{index + 1}/{total}] seed {result.seed}: {mark} "
        f"in {result.steps} steps, distance {result.final_distance:.3f}"
    )
