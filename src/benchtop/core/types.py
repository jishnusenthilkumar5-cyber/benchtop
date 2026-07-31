"""Frozen observation/action spec and run-artifact schemas.

Everything in this module is a contract that other packages build against:
the environment produces it, policies consume it, the eval runner serialises
it and the dashboard reads it back. It is FROZEN. If a work item seems to
need a change here, stop and flag it -- do not edit.

Task versions are immutable in the same way: the semantics of `pick_cube-v0`
never change. Behaviour changes mean a new version, because scores across
versions must never be silently comparable.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

# --------------------------------------------------------------------------
# Observation / action spec
# --------------------------------------------------------------------------

#: 7 joint positions + 7 joint velocities + gripper width.
STATE_DIM = 15
#: cube position (3) + cube quaternion, wxyz (4) + target position (3).
ENV_STATE_DIM = 10
#: 7 joint position targets + 1 gripper command, matching panda.xml actuators.
ACTION_DIM = 8

OBS_KEY_STATE = "observation.state"
OBS_KEY_ENV_STATE = "observation.environment_state"
ACTION_KEY = "action"

#: Every array crossing the spec boundary is float32.
DTYPE = np.float32

#: Human-readable names, index-aligned with `observation.state`.
STATE_NAMES: tuple[str, ...] = (
    *(f"joint{i}_pos" for i in range(1, 8)),
    *(f"joint{i}_vel" for i in range(1, 8)),
    "gripper_width",
)

#: Human-readable names, index-aligned with `observation.environment_state`.
ENV_STATE_NAMES: tuple[str, ...] = (
    "cube_x",
    "cube_y",
    "cube_z",
    "cube_qw",
    "cube_qx",
    "cube_qy",
    "cube_qz",
    "target_x",
    "target_y",
    "target_z",
)

#: Human-readable names, index-aligned with `action`.
ACTION_NAMES: tuple[str, ...] = (
    *(f"joint{i}_target" for i in range(1, 8)),
    "gripper_command",
)


class Observation(TypedDict):
    """Flat dict of float32 arrays, as returned by `PickCubeV0`."""

    state: np.ndarray
    environment_state: np.ndarray


# --------------------------------------------------------------------------
# Task + seed protocol
# --------------------------------------------------------------------------

TASK_ID = "pick_cube-v0"

#: Physics timestep (s) and control decimation: 50 Hz control over 2 ms physics.
PHYSICS_DT = 0.002
CONTROL_DECIMATION = 10
CONTROL_HZ = 50
#: Episode cap in control steps (10 s at 50 Hz).
MAX_EPISODE_STEPS = 500

#: Demo collection uses seeds [0, 100). Evaluation uses seeds >= 10_000.
#: The two ranges are disjoint by protocol and must stay that way.
DEMO_SEED_START = 0
DEMO_SEED_END = 100
EVAL_SEED_START = 10_000


def is_demo_seed(seed: int) -> bool:
    return DEMO_SEED_START <= seed < DEMO_SEED_END


def is_eval_seed(seed: int) -> bool:
    return seed >= EVAL_SEED_START


def assert_held_out(seeds: Iterable[int]) -> None:
    """Raise if any seed would evaluate a policy on data it may have trained on."""
    bad = sorted(s for s in seeds if not is_eval_seed(s))
    if bad:
        raise ValueError(
            f"evaluation seeds must be >= {EVAL_SEED_START} (demo seeds are "
            f"{DEMO_SEED_START}-{DEMO_SEED_END - 1}); got {bad}"
        )


# --------------------------------------------------------------------------
# Run artifact schema
# --------------------------------------------------------------------------

MANIFEST_FILENAME = "manifest.json"
METRICS_FILENAME = "metrics.json"
EPISODES_FILENAME = "episodes.jsonl"
VIDEOS_DIRNAME = "videos"

#: Bumped when the on-disk layout of a run directory changes incompatibly.
ARTIFACT_SCHEMA_VERSION = 1


def video_filename(episode_index: int) -> str:
    return f"ep{episode_index:03d}.mp4"


def _drop_none(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """One evaluated episode. Serialised as a line of `episodes.jsonl`."""

    seed: int
    success: bool
    lifted: bool
    steps: int
    #: Control step at which success was first latched; None if never.
    time_to_success: int | None
    #: Cube-to-target XY distance (m) at the final step.
    final_distance: float
    video: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _drop_none(asdict(self))

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> EpisodeResult:
        return cls(
            seed=int(d["seed"]),
            success=bool(d["success"]),
            lifted=bool(d["lifted"]),
            steps=int(d["steps"]),
            time_to_success=None if d.get("time_to_success") is None else int(d["time_to_success"]),
            final_distance=float(d["final_distance"]),
            video=d.get("video"),
        )


@dataclass(frozen=True, slots=True)
class RunMetrics:
    """Aggregate scorecard. Serialised as `metrics.json`."""

    episodes: int
    successes: int
    success_rate: float
    #: 95% Wilson score interval on the success rate.
    success_rate_ci_low: float
    success_rate_ci_high: float
    lift_rate: float
    #: Steps to success, over successful episodes only; None if there were none.
    mean_steps_to_success: float | None
    median_steps_to_success: float | None
    mean_final_distance: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> RunMetrics:
        return cls(
            episodes=int(d["episodes"]),
            successes=int(d["successes"]),
            success_rate=float(d["success_rate"]),
            success_rate_ci_low=float(d["success_rate_ci_low"]),
            success_rate_ci_high=float(d["success_rate_ci_high"]),
            lift_rate=float(d["lift_rate"]),
            mean_steps_to_success=(
                None
                if d.get("mean_steps_to_success") is None
                else float(d["mean_steps_to_success"])
            ),
            median_steps_to_success=(
                None
                if d.get("median_steps_to_success") is None
                else float(d["median_steps_to_success"])
            ),
            mean_final_distance=float(d["mean_final_distance"]),
        )


@dataclass(frozen=True, slots=True)
class PolicyDescriptor:
    """What was evaluated, precisely enough to find it again."""

    #: Selector string as passed on the CLI, e.g. `random` or `lerobot:<path>`.
    spec: str
    #: Registry name of the resolved policy, e.g. `random`, `noop`, `lerobot`.
    type: str
    checkpoint_path: str | None = None
    #: sha256 of the checkpoint, so a renamed file cannot masquerade as another.
    checkpoint_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _drop_none(asdict(self))

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> PolicyDescriptor:
        return cls(
            spec=str(d["spec"]),
            type=str(d["type"]),
            checkpoint_path=d.get("checkpoint_path"),
            checkpoint_sha256=d.get("checkpoint_sha256"),
        )


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Provenance for a run. Serialised as `manifest.json`."""

    run_id: str
    task: str
    policy: PolicyDescriptor
    #: The suite config as resolved (not as written), including the seed list.
    suite: dict[str, Any]
    git_sha: str
    benchtop_version: str
    platform: str
    started_at: str
    finished_at: str
    duration_s: float
    schema_version: int = ARTIFACT_SCHEMA_VERSION
    #: Free-form, e.g. whether video capture was on.
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["policy"] = self.policy.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> RunManifest:
        return cls(
            run_id=str(d["run_id"]),
            task=str(d["task"]),
            policy=PolicyDescriptor.from_dict(d["policy"]),
            suite=dict(d["suite"]),
            git_sha=str(d["git_sha"]),
            benchtop_version=str(d["benchtop_version"]),
            platform=str(d["platform"]),
            started_at=str(d["started_at"]),
            finished_at=str(d["finished_at"]),
            duration_s=float(d["duration_s"]),
            schema_version=int(d.get("schema_version", ARTIFACT_SCHEMA_VERSION)),
            extra=dict(d.get("extra", {})),
        )


def read_episodes(path: Path) -> Iterator[EpisodeResult]:
    """Read an `episodes.jsonl` file."""
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield EpisodeResult.from_dict(json.loads(line))
