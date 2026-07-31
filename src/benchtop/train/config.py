"""Training configuration for state-only ACT on CPU.

The defaults here are the ones benchtop trains `pick_cube-v0` with. They are
deliberately CPU-sized: there is no GPU anywhere in v0, so the transformer is
smaller than ACT's bimanual-Aloha defaults and the batch/step budget is chosen
to finish overnight rather than in an hour on an A100.

Measured wall clock (8-vCPU x86 Linux VM, torch CPU wheels, batch 64, the
defaults below -- 7.4M parameters, no vision backbone):

    ~2.8 optimiser steps/s  ->  ~6 min per 1000 steps
    30_000 steps            ->  ~3 hours

Scale that by core count: it is compute-bound, not IO-bound, because a
state-only dataset is a few MB and never touches a video decoder. The default
`save_freq=1000` therefore costs a checkpoint every ~6 minutes, which is the
most you can lose to a killed run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

#: `lerobot-train` needs a dataset identifier even for a purely local dataset;
#: `--dataset.root` is what actually selects the directory on disk.
DEFAULT_REPO_ID = "benchtop/pick_cube_v0_expert"

#: Sub-paths written by `lerobot-train` inside an output dir.
CHECKPOINTS_DIRNAME = "checkpoints"
LAST_CHECKPOINT_LINK = "last"
PRETRAINED_MODEL_DIRNAME = "pretrained_model"
TRAIN_CONFIG_FILENAME = "train_config.json"


def last_checkpoint_config(output_dir: Path) -> Path:
    """Path to the `train_config.json` a resumed run is restarted from."""
    return (
        output_dir
        / CHECKPOINTS_DIRNAME
        / LAST_CHECKPOINT_LINK
        / PRETRAINED_MODEL_DIRNAME
        / TRAIN_CONFIG_FILENAME
    )


@dataclass(frozen=True)
class TrainConfig:
    """Everything `benchtop train` needs to invoke `lerobot-train`.

    Notes on the CPU-sized choices:

    - No vision backbone is ever built: the policy's only inputs are
      `observation.state` and `observation.environment_state`, so ACT skips the
      ResNet entirely. That is the single biggest reason this is trainable here.
    - `dim_model=256` / `dim_feedforward=1024` is a quarter of ACT's default
      parameter count. On 15+10-dim inputs the default 512/3200 transformer is
      mostly wasted capacity, and it costs ~4x the wall clock per step.
    - `save_freq` is small on purpose. A CPU run is measured in hours; losing
      one to a disconnect because checkpoints were an hour apart is the
      failure mode worth engineering against.
    """

    dataset: Path
    output_dir: Path
    repo_id: str = DEFAULT_REPO_ID
    job_name: str = "benchtop_act_pick_cube_v0"

    # Optimisation budget.
    batch_size: int = 64
    steps: int = 30_000
    seed: int = 1000
    num_workers: int = 4
    log_freq: int = 200
    save_freq: int = 1_000

    # ACT architecture / inference behaviour.
    chunk_size: int = 50
    n_action_steps: int = 50
    dim_model: int = 256
    n_heads: int = 8
    dim_feedforward: int = 1_024
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1
    optimizer_lr: float = 1e-4

    resume: bool = False
    #: Extra `lerobot-train` flags, passed through verbatim.
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot exceed "
                f"chunk_size ({self.chunk_size})"
            )
        if self.steps <= 0 or self.batch_size <= 0:
            raise ValueError("steps and batch_size must be positive")

    @property
    def resume_config_path(self) -> Path:
        return last_checkpoint_config(self.output_dir)

    def has_resumable_checkpoint(self) -> bool:
        return self.resume_config_path.is_file()

    def to_cli_args(self) -> list[str]:
        """Render the `lerobot-train` argument list.

        Resuming is a different invocation, not a different flag: lerobot
        rebuilds the run from the checkpoint's own `train_config.json`, and only
        `--steps` is overridden so the budget can still be extended mid-run.
        """
        if self.resume:
            if not self.has_resumable_checkpoint():
                raise FileNotFoundError(
                    f"nothing to resume from: {self.resume_config_path} does not exist"
                )
            return [
                f"--config_path={self.resume_config_path}",
                "--resume=true",
                f"--steps={self.steps}",
            ]

        return [
            f"--dataset.repo_id={self.repo_id}",
            f"--dataset.root={self.dataset}",
            "--policy.type=act",
            "--policy.device=cpu",
            "--policy.push_to_hub=false",
            "--policy.pretrained_backbone_weights=null",
            f"--policy.chunk_size={self.chunk_size}",
            f"--policy.n_action_steps={self.n_action_steps}",
            f"--policy.dim_model={self.dim_model}",
            f"--policy.n_heads={self.n_heads}",
            f"--policy.dim_feedforward={self.dim_feedforward}",
            f"--policy.n_encoder_layers={self.n_encoder_layers}",
            f"--policy.n_decoder_layers={self.n_decoder_layers}",
            f"--policy.optimizer_lr={self.optimizer_lr}",
            f"--output_dir={self.output_dir}",
            f"--job_name={self.job_name}",
            f"--batch_size={self.batch_size}",
            f"--steps={self.steps}",
            f"--seed={self.seed}",
            f"--num_workers={self.num_workers}",
            f"--log_freq={self.log_freq}",
            f"--save_freq={self.save_freq}",
            "--save_checkpoint=true",
            # No simulation eval inside training: benchtop's own eval runner is
            # the scorecard, and spinning up envs mid-run would only steal CPU.
            "--env_eval_freq=0",
            "--wandb.enable=false",
            *self.extra_args,
        ]
