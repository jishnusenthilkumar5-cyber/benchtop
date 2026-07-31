"""Expert rollouts written out as a LeRobotDataset.

Only successful episodes are kept: a demonstration dataset is a training
signal, and a failed pick teaches the wrong thing. The number dropped is
reported rather than swallowed -- it is the expert's success rate on the
collection seeds, which is worth knowing before training on the result.

`lerobot` imports live here and in `adapters/`; nothing else in the codebase
should import it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from benchtop.core.types import (
    ACTION_DIM,
    ACTION_KEY,
    ACTION_NAMES,
    CONTROL_HZ,
    DTYPE,
    ENV_STATE_DIM,
    ENV_STATE_NAMES,
    EVAL_SEED_START,
    MAX_EPISODE_STEPS,
    OBS_KEY_ENV_STATE,
    OBS_KEY_STATE,
    STATE_DIM,
    STATE_NAMES,
    TASK_ID,
    is_eval_seed,
)
from benchtop.envs.pick_cube import PickCubeV0
from benchtop.policies.expert import ExpertPolicy

#: Natural-language task string stored with every frame.
TASK_DESCRIPTION = "Pick up the cube and place it on the target."

DEFAULT_REPO_ID = "benchtop/pick_cube_v0_expert"


def dataset_features() -> dict[str, dict]:
    """LeRobot feature spec, mirroring the frozen observation/action spec."""
    return {
        OBS_KEY_STATE: {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": list(STATE_NAMES),
        },
        OBS_KEY_ENV_STATE: {
            "dtype": "float32",
            "shape": (ENV_STATE_DIM,),
            "names": list(ENV_STATE_NAMES),
        },
        ACTION_KEY: {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": list(ACTION_NAMES),
        },
    }


@dataclass(frozen=True, slots=True)
class Episode:
    """One expert rollout, held in memory until we know whether it succeeded."""

    seed: int
    success: bool
    lifted: bool
    frames: list[dict[str, np.ndarray]]

    @property
    def steps(self) -> int:
        return len(self.frames)


@dataclass(frozen=True, slots=True)
class CollectionReport:
    """What was written, and what was thrown away."""

    path: Path
    repo_id: str
    fps: int
    kept_seeds: list[int] = field(default_factory=list)
    dropped_seeds: list[int] = field(default_factory=list)
    frames: int = 0

    @property
    def attempted(self) -> int:
        return len(self.kept_seeds) + len(self.dropped_seeds)

    @property
    def kept(self) -> int:
        return len(self.kept_seeds)

    @property
    def dropped(self) -> int:
        return len(self.dropped_seeds)

    @property
    def success_rate(self) -> float:
        return self.kept / self.attempted if self.attempted else 0.0

    def summary(self) -> str:
        return (
            f"kept {self.kept}/{self.attempted} episodes "
            f"({self.success_rate:.0%} expert success), dropped {self.dropped} "
            f"failed rollout(s), {self.frames} frames at {self.fps} Hz -> {self.path}"
        )


def rollout(env: PickCubeV0, policy: ExpertPolicy, seed: int) -> Episode:
    """Run one expert episode, buffering its frames without writing anything."""
    obs, _ = env.reset(seed=seed)
    policy.reset(seed)
    frames: list[dict[str, np.ndarray]] = []
    info: dict = {"is_success": False, "lifted": False}
    for _ in range(MAX_EPISODE_STEPS):
        action = np.asarray(policy.act(obs), dtype=DTYPE)
        frames.append(
            {
                OBS_KEY_STATE: np.asarray(obs["state"], dtype=DTYPE),
                OBS_KEY_ENV_STATE: np.asarray(obs["environment_state"], dtype=DTYPE),
                ACTION_KEY: action,
            }
        )
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    return Episode(
        seed=seed,
        success=bool(info["is_success"]),
        lifted=bool(info["lifted"]),
        frames=frames,
    )


def collect_dataset(
    out: str | Path,
    episodes: int,
    seed_start: int = 0,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    fps: int = CONTROL_HZ,
    task_description: str = TASK_DESCRIPTION,
    on_episode: Callable[[Episode], None] | None = None,
) -> CollectionReport:
    """Collect `episodes` expert rollouts into a LeRobotDataset at `out`.

    Seeds run `seed_start, seed_start + 1, ...`. Evaluation seeds are refused:
    training on them would quietly destroy the hold-out that makes an eval
    score mean anything.
    """
    seeds = list(range(seed_start, seed_start + episodes))
    return collect_dataset_from_seeds(
        out,
        seeds,
        repo_id=repo_id,
        fps=fps,
        task_description=task_description,
        on_episode=on_episode,
    )


def collect_dataset_from_seeds(
    out: str | Path,
    seeds: Sequence[int],
    *,
    repo_id: str = DEFAULT_REPO_ID,
    fps: int = CONTROL_HZ,
    task_description: str = TASK_DESCRIPTION,
    on_episode: Callable[[Episode], None] | None = None,
) -> CollectionReport:
    """As `collect_dataset`, for an explicit seed list."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    _reject_eval_seeds(seeds)
    root = Path(out)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            f"{root} already exists and is not empty; collection never overwrites a "
            "dataset, since a half-replaced one is indistinguishable from a whole one"
        )

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=dataset_features(),
        root=root,
        robot_type="panda",
        use_videos=False,
    )

    env = PickCubeV0()
    policy = ExpertPolicy(env.model)
    kept: list[int] = []
    dropped: list[int] = []
    frames = 0
    try:
        for seed in seeds:
            episode = rollout(env, policy, seed)
            if on_episode is not None:
                on_episode(episode)
            if not episode.success:
                dropped.append(seed)
                continue
            for frame in episode.frames:
                dataset.add_frame({**frame, "task": task_description})
            dataset.save_episode()
            kept.append(seed)
            frames += episode.steps
    finally:
        env.close()
        dataset.finalize()

    return CollectionReport(
        path=root,
        repo_id=repo_id,
        fps=fps,
        kept_seeds=kept,
        dropped_seeds=dropped,
        frames=frames,
    )


def _reject_eval_seeds(seeds: Iterable[int]) -> None:
    bad = sorted(s for s in seeds if is_eval_seed(s))
    if bad:
        raise ValueError(
            f"refusing to collect demonstrations on evaluation seeds (>= {EVAL_SEED_START}): "
            f"{bad}. Held-out seeds stay held out -- {TASK_ID} scores depend on it."
        )
