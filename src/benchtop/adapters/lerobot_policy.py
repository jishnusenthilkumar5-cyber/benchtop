"""Load a LeRobot ACT checkpoint and expose it as a benchtop `Policy`.

Everything lerobot-specific about inference lives here: the mapping from our
observation dict to lerobot's `observation.state` / `observation.environment_state`
keys, the normalisation statistics baked into a checkpoint's processor pipelines,
and ACT's action chunking. Inference is CPU-only -- there is no GPU anywhere in
v0 -- and lerobot imports are deferred to call time so that importing benchtop
(and hence the CLI) does not pay for importing torch.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from benchtop.core.types import (
    ACTION_DIM,
    ACTION_KEY,
    DTYPE,
    ENV_STATE_DIM,
    OBS_KEY_ENV_STATE,
    OBS_KEY_STATE,
    STATE_DIM,
    Observation,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from lerobot.policies.act.configuration_act import ACTConfig

#: Sub-directory names written by `lerobot-train` under a run's output dir.
CHECKPOINTS_DIRNAME = "checkpoints"
LAST_CHECKPOINT_LINK = "last"
PRETRAINED_MODEL_DIRNAME = "pretrained_model"

#: A directory is a loadable checkpoint if it holds the policy config.
CONFIG_FILENAME = "config.json"

_LAST_PRETRAINED = Path(CHECKPOINTS_DIRNAME) / LAST_CHECKPOINT_LINK / PRETRAINED_MODEL_DIRNAME

DEVICE = "cpu"


def resolve_checkpoint_dir(path: str | Path) -> Path:
    """Return the `pretrained_model` directory a user meant by `path`.

    Accepts any of the three things someone reasonably points at after a
    training run, so `--policy lerobot:outputs/train/act` just works:

    - the run output dir            (`.../act`, resolved via `checkpoints/last`)
    - a single checkpoint dir       (`.../checkpoints/010000`)
    - the pretrained model dir      (`.../checkpoints/010000/pretrained_model`)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such checkpoint path: {path}")

    candidates = [
        path,
        path / PRETRAINED_MODEL_DIRNAME,
        path / _LAST_PRETRAINED,
    ]
    for candidate in candidates:
        if (candidate / CONFIG_FILENAME).is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        f"{path} does not look like a lerobot checkpoint: none of "
        f"{[str(c) for c in candidates]} contains {CONFIG_FILENAME}"
    )


def _check_feature(config: ACTConfig, key: str, expected_dim: int, *, required: bool) -> None:
    feature = config.input_features.get(key) or config.output_features.get(key)
    if feature is None:
        if required:
            raise ValueError(
                f"checkpoint does not consume {key!r}; benchtop policies are trained on the "
                f"frozen pick_cube-v0 spec (input features: {sorted(config.input_features)})"
            )
        return
    (dim,) = feature.shape
    if dim != expected_dim:
        raise ValueError(
            f"checkpoint {key!r} is {dim}-dim, the pick_cube-v0 spec is {expected_dim}-dim"
        )


class LeRobotACTPolicy:
    """A trained ACT checkpoint, behind the benchtop `Policy` protocol.

    ACT predicts a chunk of `chunk_size` actions per forward pass and consumes
    `n_action_steps` of them before running the model again; lerobot's
    `select_action` owns that queue, and `reset` drains it so an episode never
    starts on actions predicted for the previous one.
    """

    def __init__(self, checkpoint: str | Path) -> None:
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        self.checkpoint_dir = resolve_checkpoint_dir(checkpoint)
        policy = ACTPolicy.from_pretrained(self.checkpoint_dir)
        policy.to(DEVICE)
        policy.eval()

        config = policy.config
        if config.image_features:
            raise ValueError(
                "benchtop v0 policies are state-only; this checkpoint expects camera input "
                f"({sorted(config.image_features)})"
            )
        _check_feature(config, OBS_KEY_STATE, STATE_DIM, required=True)
        _check_feature(config, OBS_KEY_ENV_STATE, ENV_STATE_DIM, required=True)
        _check_feature(config, ACTION_KEY, ACTION_DIM, required=True)

        # The processors carry the dataset normalisation statistics saved alongside
        # the weights: the preprocessor normalises the observation, the
        # postprocessor unnormalises the predicted action back into joint targets.
        self._preprocessor, self._postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=str(self.checkpoint_dir),
            preprocessor_overrides={"device_processor": {"device": DEVICE}},
        )
        self._policy = policy
        self.config = config

    @property
    def chunk_size(self) -> int:
        return int(self.config.chunk_size)

    @property
    def n_action_steps(self) -> int:
        return int(self.config.n_action_steps)

    def reset(self, seed: int | None = None) -> None:
        """Drop any actions left over from the previous episode.

        ACT is deterministic at inference time, so `seed` is unused.
        """
        self._policy.reset()

    def act(self, obs: Observation) -> np.ndarray:
        import torch

        batch: dict[str, Any] = {
            OBS_KEY_STATE: torch.from_numpy(np.asarray(obs["state"], dtype=DTYPE)),
            OBS_KEY_ENV_STATE: torch.from_numpy(np.asarray(obs["environment_state"], dtype=DTYPE)),
        }
        processed = self._preprocessor(batch)
        action = self._policy.select_action(processed)
        action = self._postprocessor(action)
        action = action.squeeze(0).to("cpu").numpy().astype(DTYPE, copy=False)
        if action.shape != (ACTION_DIM,):
            raise ValueError(f"expected an {ACTION_DIM}-dim action, got shape {action.shape}")
        return action


def build_act_config(**overrides: Any) -> ACTConfig:
    """The state-only ACT configuration benchtop trains and evaluates.

    Features come straight from the frozen spec, and there is no vision
    backbone: v0 policies see proprioception plus object poses, which is what
    makes training feasible on CPU.
    """
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig

    kwargs: dict[str, Any] = {
        "input_features": {
            OBS_KEY_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
            OBS_KEY_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(ENV_STATE_DIM,)),
        },
        "output_features": {
            ACTION_KEY: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))
        },
        "device": DEVICE,
        # No images -> no backbone is ever built; keep the weights off the machine too.
        "pretrained_backbone_weights": None,
    }
    kwargs.update(overrides)
    return ACTConfig(**kwargs)


def save_untrained_checkpoint(directory: str | Path, **config_overrides: Any) -> Path:
    """Write a randomly-initialised, state-only ACT checkpoint to `directory`.

    Not a training shortcut: this is how the adapter is exercised without a
    dataset -- weights are random, but the on-disk layout, the config and the
    processor pipelines are exactly what `lerobot-train` writes.
    """
    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    directory = Path(directory)
    config = build_act_config(**config_overrides)
    policy = ACTPolicy(config)

    # Identity statistics: with no dataset to measure, normalisation must at
    # least be well-defined, or the processors refuse to run.
    stats = {
        OBS_KEY_STATE: {"mean": torch.zeros(STATE_DIM), "std": torch.ones(STATE_DIM)},
        OBS_KEY_ENV_STATE: {"mean": torch.zeros(ENV_STATE_DIM), "std": torch.ones(ENV_STATE_DIM)},
        ACTION_KEY: {"mean": torch.zeros(ACTION_DIM), "std": torch.ones(ACTION_DIM)},
    }
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg=config, dataset_stats=stats)

    directory.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(directory)
    preprocessor.save_pretrained(directory)
    postprocessor.save_pretrained(directory)
    return directory
