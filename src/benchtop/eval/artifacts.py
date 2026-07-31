"""Reading and writing run directories.

A run directory is the unit benchtop publishes:

    <run_id>/
      manifest.json    provenance: git sha, resolved suite, policy descriptor
      metrics.json     the scorecard
      episodes.jsonl   one EpisodeResult per line, in evaluation order
      videos/          ep000.mp4, ... (only when video capture is on)
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from benchtop.core.types import (
    EPISODES_FILENAME,
    MANIFEST_FILENAME,
    METRICS_FILENAME,
    VIDEOS_DIRNAME,
    EpisodeResult,
    RunManifest,
    RunMetrics,
    read_episodes,
    video_filename,
)

_SLUG = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True)
class Run:
    """A run directory read back off disk."""

    path: Path
    manifest: RunManifest
    metrics: RunMetrics
    episodes: tuple[EpisodeResult, ...]


def slugify(value: str) -> str:
    return _SLUG.sub("_", value.lower()).strip("_") or "run"


def make_run_id(policy_spec: str, suite_name: str, when: datetime | None = None) -> str:
    stamp = (when or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{slugify(policy_spec)}-{slugify(suite_name)}"


def git_sha(repo: Path | None = None) -> str:
    """The commit that produced a run, or `unknown` outside a checkout."""
    root = repo or Path(__file__).resolve().parents[3]
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def benchtop_version() -> str:
    try:
        return version("benchtop")
    except PackageNotFoundError:  # pragma: no cover - editable installs always resolve
        return "0.0.0+unknown"


def platform_string() -> str:
    py = f"py{sys.version_info.major}.{sys.version_info.minor}"
    return f"{platform.system()}-{platform.machine()}-{py}"


def write_video(path: Path, frames: Sequence[np.ndarray], fps: int) -> None:
    """Encode captured RGB frames to mp4."""
    import imageio.v3 as iio

    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, np.stack(frames), fps=fps, codec="libx264", macro_block_size=1)


def episode_video_path(run_dir: Path, episode_index: int) -> Path:
    return run_dir / VIDEOS_DIRNAME / video_filename(episode_index)


def write_manifest(run_dir: Path, manifest: RunManifest) -> Path:
    return _write_json(run_dir / MANIFEST_FILENAME, manifest.to_dict())


def write_metrics(run_dir: Path, metrics: RunMetrics) -> Path:
    return _write_json(run_dir / METRICS_FILENAME, metrics.to_dict())


def write_episodes(run_dir: Path, episodes: Iterable[EpisodeResult]) -> Path:
    path = run_dir / EPISODES_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(e.to_dict()) + "\n" for e in episodes))
    return path


def append_episode(run_dir: Path, episode: EpisodeResult) -> None:
    """Append one episode, so a long run is inspectable while it is running."""
    path = run_dir / EPISODES_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(episode.to_dict()) + "\n")


def read_run(run_dir: Path) -> Run:
    run_dir = Path(run_dir)
    manifest = RunManifest.from_dict(json.loads((run_dir / MANIFEST_FILENAME).read_text()))
    metrics = RunMetrics.from_dict(json.loads((run_dir / METRICS_FILENAME).read_text()))
    episodes = tuple(read_episodes(run_dir / EPISODES_FILENAME))
    return Run(path=run_dir, manifest=manifest, metrics=metrics, episodes=episodes)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path
