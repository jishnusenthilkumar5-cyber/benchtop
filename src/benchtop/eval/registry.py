"""Resolution of `--policy` selector strings into policies.

Selectors are `random`, `noop`, `expert` and `lerobot:<checkpoint path>`.
Builders are looked up in a registry and only imported when selected, so a
selector whose module does not exist yet fails at use, not at import: the
scripted expert and the lerobot adapter plug in by registering here or by
simply existing at the module path below.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from benchtop.core.types import PolicyDescriptor

if TYPE_CHECKING:  # pragma: no cover - typing only
    import gymnasium as gym

    from benchtop.policies.base import Policy

#: A builder takes the environment (for action bounds and task metadata), the
#: argument after the colon in the selector, if any, and a seed.
PolicyBuilder = Callable[["gym.Env", str | None, int | None], "Policy"]

_REGISTRY: dict[str, PolicyBuilder] = {}


class PolicyUnavailableError(RuntimeError):
    """A known selector whose implementation is not installed."""


def register(name: str, builder: PolicyBuilder) -> None:
    """Register a policy builder under a selector name."""
    _REGISTRY[name] = builder


def registered() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def _build_random(env: gym.Env, arg: str | None, seed: int | None) -> Policy:
    from benchtop.policies.simple import RandomPolicy

    return RandomPolicy(env.action_space.low, env.action_space.high, seed=seed)


def _build_noop(env: gym.Env, arg: str | None, seed: int | None) -> Policy:
    from benchtop.policies.simple import NoopPolicy

    return NoopPolicy()


def _build_expert(env: gym.Env, arg: str | None, seed: int | None) -> Policy:
    try:
        from benchtop.policies.expert import ScriptedExpertPolicy
    except ImportError as exc:  # pragma: no cover - depends on WI-1 landing
        raise PolicyUnavailableError("the scripted expert is not available in this build") from exc
    return ScriptedExpertPolicy(env)


def _build_lerobot(env: gym.Env, arg: str | None, seed: int | None) -> Policy:
    if not arg:
        raise ValueError("lerobot selector needs a checkpoint path: lerobot:<path>")
    try:
        from benchtop.adapters.lerobot_policy import LeRobotPolicy
    except ImportError as exc:  # pragma: no cover - depends on WI-4 landing
        raise PolicyUnavailableError("the lerobot adapter is not available in this build") from exc
    return LeRobotPolicy(Path(arg))


register("random", _build_random)
register("noop", _build_noop)
register("expert", _build_expert)
register("lerobot", _build_lerobot)


def split_spec(spec: str) -> tuple[str, str | None]:
    name, sep, arg = spec.partition(":")
    return name.strip(), (arg.strip() if sep else None)


def sha256_of(path: Path) -> str:
    """Hash a checkpoint: the file itself, or every file in a checkpoint dir."""
    digest = hashlib.sha256()
    paths = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    for p in paths:
        digest.update(str(p.relative_to(path) if path.is_dir() else p.name).encode())
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def describe(spec: str) -> PolicyDescriptor:
    """Provenance for a selector, resolved without building the policy."""
    name, arg = split_spec(spec)
    if name not in _REGISTRY:
        raise ValueError(f"unknown policy {spec!r}; known: {', '.join(registered())}")
    checkpoint = Path(arg).expanduser() if arg else None
    return PolicyDescriptor(
        spec=spec,
        type=name,
        checkpoint_path=str(checkpoint) if checkpoint else None,
        checkpoint_sha256=(sha256_of(checkpoint) if checkpoint and checkpoint.exists() else None),
    )


def resolve_policy(spec: str, env: gym.Env, seed: int | None = None) -> Policy:
    """Build the policy a selector names."""
    name, arg = split_spec(spec)
    if name not in _REGISTRY:
        raise ValueError(f"unknown policy {spec!r}; known: {', '.join(registered())}")
    return _REGISTRY[name](env, arg, seed)
