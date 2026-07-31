"""Suite files: the versioned description of what an eval run does."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from benchtop.core.types import MAX_EPISODE_STEPS, TASK_ID, assert_held_out


@dataclass(frozen=True, slots=True)
class Suite:
    """A resolved suite: an explicit seed list, not a recipe for one."""

    name: str
    task: str
    seeds: tuple[int, ...]
    max_steps: int = MAX_EPISODE_STEPS
    video: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "task": self.task,
            "max_steps": self.max_steps,
            "video": self.video,
            "seeds": list(self.seeds),
        }


def _resolve_seeds(raw: dict[str, Any]) -> tuple[int, ...]:
    if "seeds" in raw:
        seeds = [int(s) for s in raw["seeds"]]
    else:
        start = int(raw["seed_start"])
        seeds = list(range(start, start + int(raw["episodes"])))
    if not seeds:
        raise ValueError("suite defines no seeds")
    assert_held_out(seeds)
    return tuple(seeds)


def parse_suite(raw: dict[str, Any]) -> Suite:
    task = str(raw.get("task", TASK_ID))
    if task != TASK_ID:
        raise ValueError(f"unknown task {task!r}; v0 evaluates {TASK_ID!r} only")
    return Suite(
        name=str(raw["name"]),
        task=task,
        seeds=_resolve_seeds(raw),
        max_steps=int(raw.get("max_steps", MAX_EPISODE_STEPS)),
        video=bool(raw.get("video", True)),
    )


def load_suite(path: Path) -> Suite:
    return parse_suite(yaml.safe_load(Path(path).read_text()))
