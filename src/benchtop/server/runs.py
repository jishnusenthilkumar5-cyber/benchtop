"""Read run directories off disk, through the frozen artifact schema."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchtop.core.types import (
    EPISODES_FILENAME,
    MANIFEST_FILENAME,
    METRICS_FILENAME,
    VIDEOS_DIRNAME,
    EpisodeResult,
    RunManifest,
    RunMetrics,
    read_episodes,
)


class RunNotFound(Exception):
    """No readable run with that id under the runs directory."""


@dataclass(frozen=True, slots=True)
class Run:
    """A run directory, parsed."""

    run_id: str
    path: Path
    manifest: RunManifest
    metrics: RunMetrics
    episodes: tuple[EpisodeResult, ...]

    @property
    def videos_dir(self) -> Path:
        return self.path / VIDEOS_DIRNAME

    def video_names(self) -> list[str]:
        """Video files that exist on disk, in episode order."""
        return [
            ep.video for ep in self.episodes if ep.video and (self.videos_dir / ep.video).is_file()
        ]

    def summary_dict(self) -> dict[str, Any]:
        m, mf = self.metrics, self.manifest
        return {
            "run_id": self.run_id,
            "task": mf.task,
            "policy": mf.policy.to_dict(),
            "finished_at": mf.finished_at,
            "started_at": mf.started_at,
            "duration_s": mf.duration_s,
            "git_sha": mf.git_sha,
            "episodes": m.episodes,
            "successes": m.successes,
            "success_rate": m.success_rate,
            "success_rate_ci_low": m.success_rate_ci_low,
            "success_rate_ci_high": m.success_rate_ci_high,
        }

    def detail_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "manifest": self.manifest.to_dict(),
            "metrics": self.metrics.to_dict(),
            "episodes": [ep.to_dict() for ep in self.episodes],
            "videos": self.video_names(),
        }


def is_run_dir(path: Path) -> bool:
    return (path / MANIFEST_FILENAME).is_file() and (path / METRICS_FILENAME).is_file()


def load_run(path: Path) -> Run:
    """Parse a single run directory."""
    manifest = RunManifest.from_dict(json.loads((path / MANIFEST_FILENAME).read_text()))
    metrics = RunMetrics.from_dict(json.loads((path / METRICS_FILENAME).read_text()))
    episodes_path = path / EPISODES_FILENAME
    episodes = tuple(read_episodes(episodes_path)) if episodes_path.is_file() else ()
    return Run(
        run_id=manifest.run_id or path.name,
        path=path,
        manifest=manifest,
        metrics=metrics,
        episodes=episodes,
    )


class RunStore:
    """Runs are read fresh on every request, so a running dash sees new evals."""

    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = Path(runs_dir)

    def list_runs(self) -> list[Run]:
        """All readable runs, newest first."""
        if not self.runs_dir.is_dir():
            return []
        runs = []
        for child in sorted(self.runs_dir.iterdir()):
            if child.is_dir() and is_run_dir(child):
                try:
                    runs.append(load_run(child))
                except (KeyError, ValueError, json.JSONDecodeError):
                    continue
        runs.sort(key=lambda r: (r.manifest.finished_at, r.run_id), reverse=True)
        return runs

    def get(self, run_id: str) -> Run:
        path = self.runs_dir / run_id
        # Reject anything that escapes the runs directory or names a nested path.
        if "/" in run_id or "\\" in run_id or run_id in ("", ".", ".."):
            raise RunNotFound(run_id)
        if not path.is_dir() or not is_run_dir(path):
            raise RunNotFound(run_id)
        return load_run(path)
