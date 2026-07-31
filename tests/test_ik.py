from __future__ import annotations

import mujoco
import numpy as np
import pytest

from benchtop.envs.pick_cube import PickCubeV0
from benchtop.ik import damped_least_squares, pose_error, solve_ik
from benchtop.policies.expert import GRASP_MAT, TCP_OFFSET


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    env = PickCubeV0()
    return env.model


@pytest.fixture
def arm(model: mujoco.MjModel):
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}") for i in range(1, 8)]
    data = mujoco.MjData(model)
    data.qpos[:] = model.key_qpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")]
    return (
        data,
        np.array([model.jnt_dofadr[j] for j in joints]),
        np.array([model.jnt_qposadr[j] for j in joints]),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand"),
    )


def test_dls_solves_a_well_conditioned_system():
    jac = np.eye(3)
    dq = damped_least_squares(jac, np.array([1.0, 2.0, 3.0]), damping=1e-6)
    assert np.allclose(dq, [1.0, 2.0, 3.0], atol=1e-5)


def test_dls_stays_bounded_at_a_singularity():
    # A rank-deficient jacobian: the pseudo-inverse diverges, DLS must not.
    jac = np.array([[1.0, 1.0], [0.0, 0.0]])
    dq = damped_least_squares(jac, np.array([1.0, 1.0]), damping=0.1)
    assert np.all(np.isfinite(dq))
    assert np.linalg.norm(dq) < 100.0


def test_pose_error_is_zero_at_the_target():
    err = pose_error(np.zeros(3), np.eye(3), np.zeros(3), np.eye(3))
    assert np.allclose(err, np.zeros(6), atol=1e-9)


def test_pose_error_reports_the_rotation_that_closes_the_gap():
    half_turn_z = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    err = pose_error(np.zeros(3), np.eye(3), np.zeros(3), half_turn_z)
    assert np.allclose(np.abs(err[3:]), [0.0, 0.0, np.pi], atol=1e-6)


def test_position_only_error_has_no_rotational_part():
    err = pose_error(np.zeros(3), np.eye(3), np.array([1.0, 0.0, 0.0]), None)
    assert err.shape == (3,)


def test_solve_ik_reaches_a_reachable_grasp_pose(model, arm):
    data, dof_ids, qpos_ids, hand = arm
    target = np.array([0.45, -0.15, 0.53])
    result = solve_ik(
        model,
        data,
        hand,
        target,
        GRASP_MAT,
        point_offset=TCP_OFFSET,
        dof_ids=dof_ids,
        qpos_ids=qpos_ids,
    )
    assert result.success
    assert result.pos_error <= 1e-3

    mujoco.mj_kinematics(model, data)
    tcp = data.xpos[hand] + data.xmat[hand].reshape(3, 3) @ TCP_OFFSET
    assert np.allclose(tcp, target, atol=1e-3)


def test_solve_ik_respects_joint_limits(model, arm):
    data, dof_ids, qpos_ids, hand = arm
    # Deliberately unreachable: the solver should saturate, not escape limits.
    solve_ik(
        model,
        data,
        hand,
        np.array([3.0, 3.0, 3.0]),
        None,
        point_offset=TCP_OFFSET,
        dof_ids=dof_ids,
        qpos_ids=qpos_ids,
        max_iterations=50,
    )
    joint_ids = model.dof_jntid[dof_ids]
    lower, upper = model.jnt_range[joint_ids, 0], model.jnt_range[joint_ids, 1]
    assert np.all(data.qpos[qpos_ids] >= lower - 1e-9)
    assert np.all(data.qpos[qpos_ids] <= upper + 1e-9)


def test_solve_ik_reports_failure_for_an_unreachable_target(model, arm):
    data, dof_ids, qpos_ids, hand = arm
    result = solve_ik(
        model,
        data,
        hand,
        np.array([5.0, 0.0, 0.5]),
        None,
        point_offset=TCP_OFFSET,
        dof_ids=dof_ids,
        qpos_ids=qpos_ids,
        max_iterations=50,
    )
    assert not result.success
    assert result.pos_error > 1e-3


def test_solve_ik_moves_only_the_requested_dofs(model, arm):
    data, dof_ids, qpos_ids, hand = arm
    before = data.qpos.copy()
    solve_ik(
        model,
        data,
        hand,
        np.array([0.45, -0.15, 0.53]),
        GRASP_MAT,
        point_offset=TCP_OFFSET,
        dof_ids=dof_ids,
        qpos_ids=qpos_ids,
    )
    untouched = np.setdiff1d(np.arange(model.nq), qpos_ids)
    assert np.array_equal(data.qpos[untouched], before[untouched])
