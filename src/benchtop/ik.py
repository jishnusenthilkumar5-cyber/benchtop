"""Damped-least-squares inverse kinematics over MuJoCo jacobians.

Used by the scripted expert to turn Cartesian waypoints into the 7 joint
position targets the Panda's position actuators take. Kinematics only: the
solver iterates on a scratch `MjData` with `mj_kinematics`, so it never
disturbs a running simulation and costs no physics steps.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


def damped_least_squares(jac: np.ndarray, error: np.ndarray, damping: float = 0.05) -> np.ndarray:
    """Least-norm joint step for a task-space error, regularised near singularities.

    Solves `dq = J^T (J J^T + damping^2 I)^-1 e`, which degrades gracefully
    where the plain pseudo-inverse blows up.
    """
    jac = np.asarray(jac, dtype=np.float64)
    error = np.asarray(error, dtype=np.float64)
    n = jac.shape[0]
    lhs = jac @ jac.T + (damping**2) * np.eye(n)
    return jac.T @ np.linalg.solve(lhs, error)


def pose_error(
    current_pos: np.ndarray,
    current_mat: np.ndarray,
    target_pos: np.ndarray,
    target_mat: np.ndarray | None,
) -> np.ndarray:
    """6-vector (position, rotation) error taking the current pose to the target.

    The rotational part is the axis-angle of `target_mat @ current_mat^T`,
    expressed in world frame; zero-length if no target orientation is given.
    """
    pos_err = np.asarray(target_pos, dtype=np.float64) - np.asarray(current_pos, dtype=np.float64)
    if target_mat is None:
        return pos_err
    r_err = (
        np.asarray(target_mat, dtype=np.float64).reshape(3, 3)
        @ np.asarray(current_mat, dtype=np.float64).reshape(3, 3).T
    )
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, r_err.reshape(9))
    rot_err = np.empty(3)
    mujoco.mju_quat2Vel(rot_err, quat, 1.0)
    return np.concatenate([pos_err, rot_err])


@dataclass(frozen=True, slots=True)
class IKResult:
    """Outcome of an IK solve. `qpos` is the full model qpos, not just the arm."""

    qpos: np.ndarray
    success: bool
    pos_error: float
    rot_error: float
    iterations: int


def solve_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
    target_pos: np.ndarray,
    target_mat: np.ndarray | None = None,
    *,
    point_offset: np.ndarray | None = None,
    dof_ids: np.ndarray | None = None,
    qpos_ids: np.ndarray | None = None,
    max_iterations: int = 200,
    pos_tolerance: float = 1e-3,
    rot_tolerance: float = 1e-2,
    damping: float = 0.05,
    step_scale: float = 0.5,
    max_step: float = 0.2,
) -> IKResult:
    """Drive a body-fixed point to a Cartesian pose by iterated DLS steps.

    `data` is used as scratch and is left holding the solution; pass a copy if
    that matters. Its `qpos` seeds the solve, so warm-starting from the current
    configuration keeps successive waypoints on a continuous joint path.

    `point_offset` is the tool point in the body's local frame (e.g. the
    grasp centre relative to the hand). Only the DOFs in `dof_ids` move, and
    joint limits are respected throughout.
    """
    offset = np.zeros(3) if point_offset is None else np.asarray(point_offset, dtype=np.float64)
    dof_ids = np.arange(model.nv) if dof_ids is None else np.asarray(dof_ids)
    qpos_ids = dof_ids if qpos_ids is None else np.asarray(qpos_ids)
    lower = model.jnt_range[model.dof_jntid[dof_ids], 0]
    upper = model.jnt_range[model.dof_jntid[dof_ids], 1]
    limited = model.jnt_limited[model.dof_jntid[dof_ids]].astype(bool)

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    rotational = target_mat is not None

    pos_err_norm = np.inf
    rot_err_norm = np.inf
    for iteration in range(1, max_iterations + 1):
        mujoco.mj_kinematics(model, data)
        # mj_jac reads cdof/subtree_com, which mj_kinematics alone leaves stale.
        mujoco.mj_comPos(model, data)
        point = data.xpos[body_id] + data.xmat[body_id].reshape(3, 3) @ offset
        err = pose_error(point, data.xmat[body_id], target_pos, target_mat)
        pos_err_norm = float(np.linalg.norm(err[:3]))
        rot_err_norm = float(np.linalg.norm(err[3:])) if rotational else 0.0
        if pos_err_norm <= pos_tolerance and rot_err_norm <= rot_tolerance:
            return IKResult(data.qpos.copy(), True, pos_err_norm, rot_err_norm, iteration)

        mujoco.mj_jac(model, data, jacp, jacr, point, body_id)
        jac = np.vstack([jacp, jacr])[:, dof_ids] if rotational else jacp[:, dof_ids]
        dq = damped_least_squares(jac, err, damping) * step_scale
        norm = float(np.linalg.norm(dq))
        if norm > max_step:
            dq *= max_step / norm
        q = data.qpos[qpos_ids] + dq
        data.qpos[qpos_ids] = np.where(limited, np.clip(q, lower, upper), q)

    return IKResult(data.qpos.copy(), False, pos_err_norm, rot_err_norm, max_iterations)
