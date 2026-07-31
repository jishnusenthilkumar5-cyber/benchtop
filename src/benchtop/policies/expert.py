"""Scripted pick-and-place expert for `pick_cube-v0`.

A fixed waypoint sequence -- pregrasp, grasp, close, lift, move, place, open,
retreat -- solved through damped-least-squares IK and emitted as the frozen
8-dim action (7 joint position targets + gripper command). It is the source of
demonstrations for training, and doubles as the ceiling baseline in evaluation.

The plan is laid out once, from the first observation of an episode: the cube
is static until the gripper touches it, so there is nothing to react to before
the grasp, and after it the cube is held. Phases advance on joint-target
convergence, with a step budget each so a failed grasp cannot stall the
episode.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from benchtop.core.types import ACTION_DIM, DTYPE, Observation
from benchtop.ik import solve_ik

#: Gripper command bounds: panda.xml remaps the fingers to 0-255.
GRIPPER_OPEN = 255.0
GRIPPER_CLOSED = 0.0

#: Tool centre point in the hand frame: between the fingertips.
TCP_OFFSET = np.array([0.0, 0.0, 0.1034])

#: Hand orientation held throughout: z down, fingers closing along world x.
GRASP_MAT = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])

# Waypoint heights, relative to the cube centre / table (metres).
PREGRASP_HEIGHT = 0.10
GRASP_HEIGHT = 0.005
LIFT_HEIGHT = 0.14
PLACE_HEIGHT = 0.005
RETREAT_HEIGHT = 0.15

#: Joint-space tolerance (rad, max over joints) for calling a waypoint reached.
JOINT_TOLERANCE = 0.02


@dataclass(frozen=True, slots=True)
class Waypoint:
    """One leg of the script: hold a joint target and a gripper command."""

    name: str
    qpos: np.ndarray | None
    gripper: float
    #: Steps to hold once converged (or once the budget runs out).
    settle_steps: int
    max_steps: int


class ExpertPolicy:
    """Scripted expert. Implements the `Policy` protocol.

    Owns a private `MjModel`/`MjData` pair for IK. Pass the environment's model
    to avoid loading the scene twice; the solve only uses kinematics, so
    sharing the model with a running env is safe.
    """

    def __init__(self, model: mujoco.MjModel | None = None) -> None:
        if model is None:
            from benchtop.envs.pick_cube import PickCubeV0

            model = PickCubeV0().model
        self.model = model
        self._ik_data = mujoco.MjData(model)
        self._hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self._arm_joint_ids = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}") for i in range(1, 8)]
        )
        self._arm_qpos_adr = np.array([model.jnt_qposadr[j] for j in self._arm_joint_ids])
        self._arm_dof_adr = np.array([model.jnt_dofadr[j] for j in self._arm_joint_ids])
        self._home_qpos = model.key_qpos[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        ].copy()

        self._waypoints: list[Waypoint] = []
        self._index = 0
        self._steps_in_phase = 0
        self._settled = 0
        self._last_qpos: np.ndarray | None = None

    # -- Policy protocol ---------------------------------------------------

    def reset(self, seed: int | None = None) -> None:
        """Discard the plan; the next `act` replans from the observation."""
        self._waypoints = []
        self._index = 0
        self._steps_in_phase = 0
        self._settled = 0
        self._last_qpos = None

    def act(self, obs: Observation) -> np.ndarray:
        if not self._waypoints:
            self._plan(obs)
        self._advance(obs)
        waypoint = self._waypoints[min(self._index, len(self._waypoints) - 1)]
        qpos = self._home_qpos[self._arm_qpos_adr] if waypoint.qpos is None else waypoint.qpos
        action = np.empty(ACTION_DIM, dtype=DTYPE)
        action[:7] = qpos
        action[7] = waypoint.gripper
        return action

    @property
    def phase(self) -> str:
        """Name of the waypoint currently being executed (`""` before reset)."""
        if not self._waypoints:
            return ""
        return self._waypoints[min(self._index, len(self._waypoints) - 1)].name

    # -- planning ----------------------------------------------------------

    def _plan(self, obs: Observation) -> None:
        env_state = np.asarray(obs["environment_state"], dtype=np.float64)
        cube = env_state[:3]
        target = env_state[7:10]
        cube_z = float(cube[2])
        table_z = cube_z - 0.025

        # Waypoints are solved in sequence, each warm-started from the last, so
        # the arm follows a continuous joint path rather than jumping between
        # unrelated IK branches for nearby Cartesian poses.
        self._ik_data.qpos[:] = self._home_qpos
        pregrasp = self._solve([cube[0], cube[1], cube_z + PREGRASP_HEIGHT])
        grasp = self._solve([cube[0], cube[1], cube_z + GRASP_HEIGHT])
        lift = self._solve([cube[0], cube[1], cube_z + LIFT_HEIGHT])
        # Transport is split in two: one long joint interpolation across the
        # workspace accelerates hard enough to shake the cube out of the grip.
        midpoint = self._solve(
            [(cube[0] + target[0]) / 2, (cube[1] + target[1]) / 2, cube_z + LIFT_HEIGHT]
        )
        move = self._solve([target[0], target[1], cube_z + LIFT_HEIGHT])
        place = self._solve([target[0], target[1], table_z + 0.025 + PLACE_HEIGHT])
        retreat = self._solve([target[0], target[1], table_z + RETREAT_HEIGHT])

        self._waypoints = [
            Waypoint("pregrasp", pregrasp, GRIPPER_OPEN, settle_steps=2, max_steps=70),
            Waypoint("grasp", grasp, GRIPPER_OPEN, settle_steps=4, max_steps=60),
            Waypoint("close", grasp, GRIPPER_CLOSED, settle_steps=20, max_steps=20),
            Waypoint("lift", lift, GRIPPER_CLOSED, settle_steps=2, max_steps=60),
            Waypoint("transit", midpoint, GRIPPER_CLOSED, settle_steps=2, max_steps=50),
            Waypoint("move", move, GRIPPER_CLOSED, settle_steps=2, max_steps=60),
            Waypoint("place", place, GRIPPER_CLOSED, settle_steps=6, max_steps=60),
            Waypoint("open", place, GRIPPER_OPEN, settle_steps=15, max_steps=15),
            Waypoint("retreat", retreat, GRIPPER_OPEN, settle_steps=60, max_steps=80),
        ]

    def _solve(self, position: list[float] | np.ndarray) -> np.ndarray:
        result = solve_ik(
            self.model,
            self._ik_data,
            self._hand_id,
            np.asarray(position, dtype=np.float64),
            GRASP_MAT,
            point_offset=TCP_OFFSET,
            dof_ids=self._arm_dof_adr,
            qpos_ids=self._arm_qpos_adr,
        )
        return result.qpos[self._arm_qpos_adr].copy()

    # -- execution ---------------------------------------------------------

    def _advance(self, obs: Observation) -> None:
        """Move to the next waypoint once this one is reached and settled."""
        if self._index >= len(self._waypoints) - 1:
            return
        waypoint = self._waypoints[self._index]
        joints = np.asarray(obs["state"][:7], dtype=np.float64)
        target = self._home_qpos[self._arm_qpos_adr] if waypoint.qpos is None else waypoint.qpos
        reached = float(np.max(np.abs(joints - target))) <= JOINT_TOLERANCE

        self._steps_in_phase += 1
        self._settled = self._settled + 1 if reached else 0
        done = self._settled >= waypoint.settle_steps or self._steps_in_phase >= waypoint.max_steps
        if done:
            self._index += 1
            self._steps_in_phase = 0
            self._settled = 0
