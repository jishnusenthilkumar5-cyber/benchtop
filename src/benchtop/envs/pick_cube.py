"""`pick_cube-v0`: a Franka Panda picks a cube and places it in a target zone.

The semantics of this task version are frozen. Anything that would move a
success rate -- scene geometry, randomisation ranges, the success criterion --
means a new task version (`pick_cube-v1`), never an edit here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from benchtop.core.types import (
    ACTION_DIM,
    CONTROL_DECIMATION,
    DTYPE,
    ENV_STATE_DIM,
    MAX_EPISODE_STEPS,
    STATE_DIM,
    TASK_ID,
    Observation,
)

_ASSETS_DIR = Path(__file__).parent / "assets"
_VENDOR_PANDA_DIR = (
    Path(__file__).resolve().parents[3] / "vendor" / "mujoco_menagerie" / "franka_emika_panda"
)

_SCENE_XML = _ASSETS_DIR / "pick_cube_v0.xml"

# Randomisation ranges (metres), seeded per episode. Cube and target regions
# are disjoint in y so the cube never starts inside the target zone.
CUBE_X_RANGE = (0.35, 0.55)
CUBE_Y_RANGE = (-0.25, -0.05)
TARGET_X_RANGE = (0.35, 0.55)
TARGET_Y_RANGE = (0.05, 0.25)

TABLE_HEIGHT = 0.4
CUBE_HALF_SIZE = 0.025
#: z of the cube centre when it is resting on the table.
CUBE_REST_Z = TABLE_HEIGHT + CUBE_HALF_SIZE

# Success criterion.
SUCCESS_XY_TOL = 0.03
SUCCESS_Z_TOL = 0.01
SUCCESS_SPEED_TOL = 0.02
SUCCESS_HOLD_STEPS = 10
#: The cube counts as lifted once its centre clears the resting height by this.
LIFT_HEIGHT = 0.05

RENDER_WIDTH = 640
RENDER_HEIGHT = 480
RENDER_CAMERA = "eval"


def _panda_assets() -> dict[str, bytes]:
    """In-memory asset map for the vendored Panda.

    MuJoCo resolves `meshdir` relative to the *main* model file, so including
    the vendored `panda.xml` by relative path breaks its mesh references.
    Loading from a string with an explicit asset map sidesteps that: the keys
    are exactly the paths MuJoCo asks for.
    """
    assets: dict[str, bytes] = {"panda.xml": (_VENDOR_PANDA_DIR / "panda.xml").read_bytes()}
    for mesh in sorted((_VENDOR_PANDA_DIR / "assets").iterdir()):
        if mesh.is_file():
            assets[f"assets/{mesh.name}"] = mesh.read_bytes()
    return assets


class PickCubeV0(gym.Env):
    """Gymnasium environment for `pick_cube-v0`.

    Observations are a flat dict of float32 arrays: `state` (15-dim
    proprioception) and `environment_state` (10-dim object/target poses).
    Actions are 8-dim: 7 joint position targets plus a gripper command, in the
    units of the Panda's position actuators.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}
    task_id = TASK_ID

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()
        self.model = mujoco.MjModel.from_xml_string(_SCENE_XML.read_text(), _panda_assets())
        self.model.opt.timestep = 0.002
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self._renderer: mujoco.Renderer | None = None

        self._arm_joint_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
                for i in range(1, 8)
            ]
        )
        self._arm_qpos_adr = np.array([self.model.jnt_qposadr[j] for j in self._arm_joint_ids])
        self._arm_qvel_adr = np.array([self.model.jnt_dofadr[j] for j in self._arm_joint_ids])
        self._finger_qpos_adr = np.array(
            [
                self.model.jnt_qposadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
                for name in ("finger_joint1", "finger_joint2")
            ]
        )
        cube_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_freejoint")
        self._cube_qpos_adr = self.model.jnt_qposadr[cube_joint]
        self._cube_qvel_adr = self.model.jnt_dofadr[cube_joint]
        self._cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self._target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self._target_mocap_id = self.model.body_mocapid[self._target_body_id]
        self._home_key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")

        low = self.model.actuator_ctrlrange[:, 0].astype(DTYPE)
        high = self.model.actuator_ctrlrange[:, 1].astype(DTYPE)
        self.action_space = spaces.Box(low=low, high=high, shape=(ACTION_DIM,), dtype=DTYPE)
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(-np.inf, np.inf, shape=(STATE_DIM,), dtype=DTYPE),
                "environment_state": spaces.Box(
                    -np.inf, np.inf, shape=(ENV_STATE_DIM,), dtype=DTYPE
                ),
            }
        )

        self._steps = 0
        self._hold = 0
        self._lifted = False

    # -- gymnasium API ----------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key)

        cube_xy = self._sample_xy(CUBE_X_RANGE, CUBE_Y_RANGE)
        target_xy = self._sample_xy(TARGET_X_RANGE, TARGET_Y_RANGE)
        self._set_cube_pose(np.array([cube_xy[0], cube_xy[1], CUBE_REST_Z]))
        self.data.mocap_pos[self._target_mocap_id] = [
            target_xy[0],
            target_xy[1],
            TABLE_HEIGHT + 0.001,
        ]

        mujoco.mj_forward(self.model, self.data)
        self._steps = 0
        self._hold = 0
        self._lifted = False
        return self._observation(), self._info(success=False)

    def step(self, action: np.ndarray) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        ctrl = np.clip(
            np.asarray(action, dtype=np.float64).reshape(ACTION_DIM),
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1],
        )
        self.data.ctrl[:] = ctrl
        for _ in range(CONTROL_DECIMATION):
            mujoco.mj_step(self.model, self.data)

        self._steps += 1
        self._update_events()
        success = self.is_success()
        terminated = success
        truncated = self._steps >= MAX_EPISODE_STEPS and not terminated
        return self._observation(), float(success), terminated, truncated, self._info(success)

    def render(self) -> np.ndarray:
        """Offscreen RGB frame from the fixed evaluation camera."""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=RENDER_HEIGHT, width=RENDER_WIDTH)
        self._renderer.update_scene(self.data, camera=RENDER_CAMERA)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # -- task state -------------------------------------------------------

    @property
    def cube_pos(self) -> np.ndarray:
        return self.data.qpos[self._cube_qpos_adr : self._cube_qpos_adr + 3].copy()

    @property
    def cube_quat(self) -> np.ndarray:
        return self.data.qpos[self._cube_qpos_adr + 3 : self._cube_qpos_adr + 7].copy()

    @property
    def cube_speed(self) -> float:
        return float(np.linalg.norm(self.data.qvel[self._cube_qvel_adr : self._cube_qvel_adr + 3]))

    @property
    def target_pos(self) -> np.ndarray:
        return self.data.mocap_pos[self._target_mocap_id].copy()

    @property
    def distance_to_target(self) -> float:
        """Cube-to-target distance in the table plane (m)."""
        return float(np.linalg.norm(self.cube_pos[:2] - self.target_pos[:2]))

    @property
    def lifted(self) -> bool:
        return self._lifted

    def is_success(self) -> bool:
        """Cube placed on the target: settled, on the table, held for 10 steps."""
        return self._hold >= SUCCESS_HOLD_STEPS

    def _placed(self) -> bool:
        return (
            self.distance_to_target <= SUCCESS_XY_TOL
            and abs(self.cube_pos[2] - CUBE_REST_Z) <= SUCCESS_Z_TOL
            and self.cube_speed <= SUCCESS_SPEED_TOL
        )

    def _update_events(self) -> None:
        if self.cube_pos[2] > CUBE_REST_Z + LIFT_HEIGHT:
            self._lifted = True
        self._hold = self._hold + 1 if self._placed() else 0

    def set_cube_pos(self, pos: np.ndarray) -> None:
        """Teleport the cube. For tests and debugging, not for policies."""
        self._set_cube_pose(np.asarray(pos, dtype=np.float64))
        mujoco.mj_forward(self.model, self.data)
        self._update_events()

    # -- internals --------------------------------------------------------

    def _sample_xy(self, x_range: tuple[float, float], y_range: tuple[float, float]) -> np.ndarray:
        return np.array(
            [self.np_random.uniform(*x_range), self.np_random.uniform(*y_range)], dtype=np.float64
        )

    def _set_cube_pose(self, pos: np.ndarray, quat: np.ndarray | None = None) -> None:
        adr = self._cube_qpos_adr
        self.data.qpos[adr : adr + 3] = pos
        self.data.qpos[adr + 3 : adr + 7] = np.array([1.0, 0.0, 0.0, 0.0]) if quat is None else quat
        self.data.qvel[self._cube_qvel_adr : self._cube_qvel_adr + 6] = 0.0

    def _observation(self) -> Observation:
        gripper_width = float(self.data.qpos[self._finger_qpos_adr].sum())
        state = np.concatenate(
            [
                self.data.qpos[self._arm_qpos_adr],
                self.data.qvel[self._arm_qvel_adr],
                [gripper_width],
            ]
        ).astype(DTYPE)
        environment_state = np.concatenate([self.cube_pos, self.cube_quat, self.target_pos]).astype(
            DTYPE
        )
        return {"state": state, "environment_state": environment_state}

    def _info(self, success: bool) -> dict[str, Any]:
        return {
            "is_success": success,
            "lifted": self._lifted,
            "distance": self.distance_to_target,
            "steps": self._steps,
        }
