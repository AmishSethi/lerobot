# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy.spatial.transform import Rotation

from lerobot.robots.bi_yam.config_bi_yam import YAM_SCALAR_KEYS as DRIVER_YAM_SCALAR_KEYS
from lerobot.robots.bi_yam.umi_retargeting import (
    CURRENT_RELATIVE_R6D_SE3,
    CURRENT_RELATIVE_R6D_SE3_JAW_DELTA,
    CURRENT_RELATIVE_SE3,
    DUAL_LIDAR_UMI_MECHANICAL_LOGGED_FROM_TCP,
    EPISODE_START_ABSOLUTE,
    IK_SOLVER_BOUNDED_LEAST_SQUARES,
    IK_SOLVER_PLACO,
    LEFT_GRIPPER_KEY,
    LEFT_JOINT_KEYS,
    RIGHT_GRIPPER_KEY,
    RIGHT_JOINT_KEYS,
    YAM_SCALAR_KEYS,
    YAM_TCP_FROM_UMI_TCP,
    ArmCalibration,
    ArmTrajectory,
    CurrentRelativeGripperMap,
    EpisodeFrames,
    EventAwareSmoothingConfig,
    GripperMap,
    RigidTransformCalibration,
    SafetyLimits,
    UmiGripperEndpointCalibration,
    YamArmKinematics,
    YamUmiEeAdapter,
    arm_calibration_from_transforms,
    calibration_basis_report,
    calibrations_from_json,
    current_relative_gripper_map_from_json,
    detect_gripper_motion_events,
    dual_lidar_umi_f544_candidate_arm_calibration,
    load_raw_umi_episode,
    matrix_to_r6d_pose,
    pose_to_vec,
    r6d_pose_to_matrix,
    reanchor_pose_track,
    relativize_umi_action_chunk,
    resolve_current_relative_r6d_action_chunk,
    resolve_current_relative_r6d_jaw_delta_action_chunk,
    resolve_umi_action_chunk,
    smooth_arm_trajectory_between_gripper_events,
    time_parameterize_arm_trajectory,
    validate_trusted_yam_urdf,
    vec_to_pose,
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _transform(position=(0.0, 0.0, 0.0), rotvec=(0.0, 0.0, 0.0)) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    transform[:3, 3] = position
    return transform


def _verified_current_relative_gripper_map() -> CurrentRelativeGripperMap:
    def endpoint(
        *, dataset_device: str, assigned_arm: str, device_id: str, closed: float, opened: float
    ) -> UmiGripperEndpointCalibration:
        return UmiGripperEndpointCalibration(
            closed_width_mm=closed,
            open_width_mm=opened,
            verified=True,
            dataset_device=dataset_device,
            assigned_arm=assigned_arm,
            device_id=device_id,
            evidence_uri=f"lab://endpoint-sweeps/2026-08-09/{device_id}",
            evidence_sha256=_sha256_text(f"{device_id}-evidence"),
            detector_config_id="aruco-detector-v3",
            detector_config_sha256=_sha256_text("aruco-detector-v3"),
            fisheye_calibration_id=f"fisheye-{device_id}-2026-08-09",
            fisheye_calibration_sha256=_sha256_text(f"fisheye-{device_id}"),
            geometry_config_id="umi-jaw-tags-v2",
            geometry_config_sha256=_sha256_text("umi-jaw-tags-v2"),
        )

    return CurrentRelativeGripperMap(
        left=endpoint(
            dataset_device="umi1",
            assigned_arm="left",
            device_id="UMI-AIRY-SN-001",
            closed=25.0,
            opened=125.0,
        ),
        right=endpoint(
            dataset_device="umi2",
            assigned_arm="right",
            device_id="UMI-AIRY-SN-002",
            closed=20.0,
            opened=120.0,
        ),
    )


def _write_minimal_trusted_yam_urdf(path, *, fingertip_z: float = -0.0566829) -> None:
    arm_limits = (
        (-2.61799, 3.14159),
        (-8.88178e-16, 3.66519),
        (0.0, 3.14159),
        (-1.69297, 1.5708),
        (-1.5708, 1.5708),
        (-2.0944, 2.0944),
    )
    arm_joints = "".join(
        f'<joint name="joint{index}" type="revolute"><limit lower="{lower}" upper="{upper}"/></joint>'
        for index, (lower, upper) in enumerate(arm_limits, start=1)
    )
    path.write_text(
        '<robot name="yam">'
        + arm_joints
        + f"""
        <joint name="joint7" type="prismatic">
          <origin xyz="-0.0239931 0.0445119 {fingertip_z}" rpy="0 0 0"/>
          <axis xyz="0 1 0"/><parent link="gripper"/><child link="tip_left"/>
          <limit lower="-0.04695" upper="0"/>
        </joint>
        <joint name="joint8" type="prismatic">
          <origin xyz="0.0239931 -0.0445119 {fingertip_z}" rpy="0 0 0"/>
          <axis xyz="0 -1 0"/><parent link="gripper"/><child link="tip_right"/>
          <limit lower="-0.04695" upper="0"/>
        </joint>
        </robot>
        """
    )


def test_retargeting_command_order_exactly_matches_biyam_driver_order() -> None:
    assert YAM_SCALAR_KEYS == DRIVER_YAM_SCALAR_KEYS
    assert (
        *(f"left_joint_{index}.pos" for index in range(6)),
        "left_gripper.pos",
        *(f"right_joint_{index}.pos" for index in range(6)),
        "right_gripper.pos",
    ) == YAM_SCALAR_KEYS


def test_trusted_urdf_matches_tcp_aperture_and_configured_operational_limits(tmp_path) -> None:
    urdf = tmp_path / "yam.urdf"
    _write_minimal_trusted_yam_urdf(urdf)
    left_limits = [(-0.5, 0.5), (0.0, 0.6), (0.0, 0.7), (-0.8, 0.8), (-0.9, 0.9), (-1.0, 1.0)]
    right_limits = [(-0.4, 0.4), (0.0, 0.5), (0.0, 0.6), (-0.7, 0.7), (-0.8, 0.8), (-0.9, 0.9)]

    validate_trusted_yam_urdf(
        urdf,
        operational_joint_limits={"left": left_limits, "right": right_limits},
    )

    right_limits[0] = (-3.0, 0.4)
    with pytest.raises(ValueError, match="right operational limits exceed.*joints.*1"):
        validate_trusted_yam_urdf(
            urdf,
            operational_joint_limits={"left": left_limits, "right": right_limits},
        )


def test_trusted_urdf_rejects_stale_tcp_constant(tmp_path) -> None:
    urdf = tmp_path / "changed-yam.urdf"
    _write_minimal_trusted_yam_urdf(urdf, fingertip_z=-0.05)

    with pytest.raises(ValueError, match="midpoint disagrees"):
        validate_trusted_yam_urdf(urdf)


def test_hardware_urdf_validation_rejects_non_pinned_file_identity(tmp_path) -> None:
    urdf = tmp_path / "yam.urdf"
    _write_minimal_trusted_yam_urdf(urdf)

    with pytest.raises(ValueError, match="does not match pinned"):
        validate_trusted_yam_urdf(urdf, expected_sha256="0" * 64)


def test_reanchor_pose_track_uses_se3_composition() -> None:
    start = _transform((0.2, -0.1, 0.3), (0.4, -0.2, 0.7))
    relative = _transform((0.05, -0.03, 0.08), (-0.5, 0.2, 0.1))
    absolute = start @ relative
    track = np.stack((pose_to_vec(start), pose_to_vec(absolute)))

    reanchored = reanchor_pose_track(track)

    np.testing.assert_allclose(reanchored[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(vec_to_pose(reanchored[1]), relative, atol=1e-12)
    # Component subtraction is the raw storage law, but is not the relative SE(3) pose.
    assert not np.allclose(track[1] - track[0], reanchored[1], atol=1e-3)


def test_existing_umi_checkpoint_actions_remain_episode_start_absolute_by_default() -> None:
    chunk = np.arange(2 * 14, dtype=np.float32).reshape(2, 14) / 100.0

    resolved = resolve_umi_action_chunk(chunk)

    assert resolved.dtype == np.float32
    np.testing.assert_array_equal(resolved, chunk)
    assert not np.shares_memory(resolved, chunk)
    with pytest.raises(ValueError, match="require the measured policy state"):
        resolve_umi_action_chunk(chunk, semantics=CURRENT_RELATIVE_SE3)


def test_current_relative_action_chunk_is_se3_anchored_not_subtracted_or_recursive() -> None:
    current = np.zeros(14, dtype=np.float64)
    current[0:6] = pose_to_vec(_transform((0.3, -0.2, 0.4), (0.2, -0.3, np.pi / 2)))
    current[6] = 0.8
    current[7:13] = pose_to_vec(_transform((-0.1, 0.5, 0.2), (-0.4, 0.1, -0.2)))
    current[13] = 0.7

    left_deltas = (
        _transform((0.05, 0.0, 0.01), (0.1, 0.0, -0.2)),
        _transform((0.10, -0.02, 0.03), (-0.1, 0.2, 0.3)),
    )
    right_deltas = (
        _transform((-0.02, 0.04, 0.0), (0.0, 0.2, 0.1)),
        _transform((-0.06, 0.08, 0.02), (0.2, -0.1, 0.0)),
    )
    absolute = np.zeros((2, 14), dtype=np.float64)
    current_left = vec_to_pose(current[0:6])
    current_right = vec_to_pose(current[7:13])
    for index in range(2):
        absolute[index, 0:6] = pose_to_vec(current_left @ left_deltas[index])
        absolute[index, 7:13] = pose_to_vec(current_right @ right_deltas[index])
    absolute[:, 6] = (0.6, 0.4)
    absolute[:, 13] = (0.5, 0.3)

    relative = relativize_umi_action_chunk(absolute, current)
    resolved = resolve_umi_action_chunk(
        relative,
        semantics=CURRENT_RELATIVE_SE3,
        measured_policy_state=current,
    )

    for index in range(2):
        np.testing.assert_allclose(vec_to_pose(relative[index, 0:6]), left_deltas[index], atol=1e-7)
        np.testing.assert_allclose(vec_to_pose(relative[index, 7:13]), right_deltas[index], atol=1e-7)
    np.testing.assert_allclose(resolved, absolute, atol=2e-7)
    assert relative.dtype == np.float32
    np.testing.assert_allclose(relative[:, (6, 13)], absolute[:, (6, 13)], atol=1e-7)
    # Component subtraction is not an SE(3) relative pose for a rotated anchor.
    assert not np.allclose(relative[0, 0:6], absolute[0, 0:6] - current[0:6], atol=1e-3)
    # Row 1 is independently current->future[1], not future[0]->future[1].
    assert not np.allclose(
        relative[1, 0:6],
        pose_to_vec(np.linalg.inv(vec_to_pose(absolute[0, 0:6])) @ vec_to_pose(absolute[1, 0:6])),
        atol=1e-3,
    )


def _yam_observation(
    *,
    left_x: float = 0.0,
    left_yaw: float = 0.0,
    right_x: float = 0.0,
    right_yaw: float = 0.0,
) -> dict[str, float]:
    observation = dict.fromkeys((*LEFT_JOINT_KEYS, *RIGHT_JOINT_KEYS), 0.0)
    observation[LEFT_JOINT_KEYS[0]] = left_x
    observation[LEFT_JOINT_KEYS[1]] = left_yaw
    observation[RIGHT_JOINT_KEYS[0]] = right_x
    observation[RIGHT_JOINT_KEYS[1]] = right_yaw
    observation[LEFT_GRIPPER_KEY] = 1.0
    observation[RIGHT_GRIPPER_KEY] = 1.0
    return observation


class _MeasuredPoseKinematics:
    @staticmethod
    def fk(joints: np.ndarray, flange_to_target: np.ndarray | None = None) -> np.ndarray:
        pose = _transform((float(joints[0]), 0.0, 0.0), (0.0, 0.0, float(joints[1])))
        return pose if flange_to_target is None else pose @ flange_to_target


def test_current_relative_r6d_matches_training_layout_and_fixed_axis_conjugation() -> None:
    identity_r6d = np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    np.testing.assert_allclose(matrix_to_r6d_pose(np.eye(4))[3:], identity_r6d, atol=1e-12)

    left_delta = _transform((0.02, 0.03, 0.04), (0.2, -0.1, 0.3))
    right_delta = _transform((-0.05, 0.01, 0.02), (-0.1, 0.4, -0.2))
    chunk = np.zeros((2, 20), dtype=np.float32)
    chunk[:, 0:9] = matrix_to_r6d_pose(left_delta)
    chunk[:, 9] = (0.7, 0.6)
    chunk[:, 10:19] = matrix_to_r6d_pose(right_delta)
    chunk[:, 19] = (0.5, 0.4)
    left_anchor = _transform((0.3, -0.2, 0.4), (0.1, 0.2, -0.3))
    right_anchor = _transform((-0.2, 0.1, 0.5), (-0.2, 0.1, 0.4))

    resolved = resolve_current_relative_r6d_action_chunk(
        chunk,
        measured_yam_tcp={"left": left_anchor, "right": right_anchor},
    )

    expected_left_delta = YAM_TCP_FROM_UMI_TCP @ left_delta @ YAM_TCP_FROM_UMI_TCP
    expected_right_delta = YAM_TCP_FROM_UMI_TCP @ right_delta @ YAM_TCP_FROM_UMI_TCP
    np.testing.assert_allclose(vec_to_pose(resolved[0, 0:6]), left_anchor @ expected_left_delta, atol=1e-7)
    np.testing.assert_allclose(vec_to_pose(resolved[1, 0:6]), left_anchor @ expected_left_delta, atol=1e-7)
    np.testing.assert_allclose(vec_to_pose(resolved[0, 7:13]), right_anchor @ expected_right_delta, atol=1e-7)
    np.testing.assert_allclose(resolved[:, (6, 13)], chunk[:, (9, 19)], atol=1e-7)
    np.testing.assert_allclose(expected_left_delta[:3, 3], (-0.02, 0.03, -0.04), atol=1e-12)


def test_jaw_delta_rows_share_one_query_anchor_and_clip_without_integration() -> None:
    identity_pose = matrix_to_r6d_pose(np.eye(4))
    chunk = np.zeros((3, 20), dtype=np.float32)
    chunk[:, 0:9] = identity_pose
    chunk[:, 10:19] = identity_pose
    chunk[:, 9] = (0.10, 0.15, 0.80)
    chunk[:, 19] = (-0.20, -0.10, -1.00)

    resolved = resolve_current_relative_r6d_jaw_delta_action_chunk(
        chunk,
        measured_yam_tcp={"left": np.eye(4), "right": np.eye(4)},
        measured_umi_gripper={"left": 0.40, "right": 0.80},
    )

    # Row 1 is 0.40 + 0.15, not (0.40 + 0.10) + 0.15.
    np.testing.assert_allclose(resolved[:, 6], (0.50, 0.55, 1.00), atol=1e-7)
    np.testing.assert_allclose(resolved[:, 13], (0.60, 0.70, 0.00), atol=1e-7)


def test_jaw_delta_adapter_refreshes_calibrated_per_arm_anchors_on_requery() -> None:
    adapter = YamUmiEeAdapter(
        left=_MeasuredPoseKinematics(),
        right=_MeasuredPoseKinematics(),
        action_semantics=CURRENT_RELATIVE_R6D_SE3_JAW_DELTA,
        current_relative_gripper=_verified_current_relative_gripper_map(),
    )
    identity_pose = matrix_to_r6d_pose(np.eye(4))
    chunk = np.zeros((1, 20), dtype=np.float32)
    chunk[:, 0:9] = identity_pose
    chunk[:, 10:19] = identity_pose
    chunk[0, 9] = 0.10
    chunk[0, 19] = -0.05

    first_query = _yam_observation()
    first_query[LEFT_GRIPPER_KEY] = 0.0
    first_query[RIGHT_GRIPPER_KEY] = 0.0
    second_query = _yam_observation()
    second_query[LEFT_GRIPPER_KEY] = 0.5
    second_query[RIGHT_GRIPPER_KEY] = 0.25

    first = adapter.resolve_action_chunk(chunk, first_query)
    second = adapter.resolve_action_chunk(chunk, second_query)

    # Left is umi1 (normalized endpoints 0.20..1.00); right is umi2 (0.16..0.96).
    assert first[0, 6] == pytest.approx(0.30)
    assert first[0, 13] == pytest.approx(0.11)
    assert second[0, 6] == pytest.approx(0.70)
    assert second[0, 13] == pytest.approx(0.31)


def test_current_relative_r6d_state_is_identity_first_then_measured_tcp_history() -> None:
    adapter = YamUmiEeAdapter(
        left=_MeasuredPoseKinematics(),
        right=_MeasuredPoseKinematics(),
        action_semantics=CURRENT_RELATIVE_R6D_SE3,
        current_relative_gripper=_verified_current_relative_gripper_map(),
    )
    current = _yam_observation(left_x=0.20, right_x=-0.10)
    current[LEFT_GRIPPER_KEY] = 0.8
    current[RIGHT_GRIPPER_KEY] = 0.25
    previous = _yam_observation(left_x=0.15, right_x=-0.04)

    first = adapter.observation_to_policy_state(current)
    history = adapter.observation_to_policy_state(current, previous_observation=previous)

    np.testing.assert_allclose(first[[0, 1, 2, 10, 11, 12]], 0.0, atol=1e-12)
    np.testing.assert_allclose(first[3:9], (1.0, 0.0, 0.0, 0.0, 1.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(first[13:19], (1.0, 0.0, 0.0, 0.0, 1.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(history[0:3], (0.05, 0.0, 0.0), atol=1e-7)
    np.testing.assert_allclose(history[10:13], (-0.06, 0.0, 0.0), atol=1e-7)
    np.testing.assert_allclose(r6d_pose_to_matrix(history[0:9])[:3, :3], np.eye(3), atol=1e-7)
    assert first[9] == pytest.approx(0.84)
    assert first[19] == pytest.approx(0.36)
    assert first.shape == history.shape == (20,)


def test_current_relative_gripper_uses_verified_tag_width_endpoints_bidirectionally() -> None:
    mapping = _verified_current_relative_gripper_map()

    assert mapping.umi_to_yam(0.8, arm="left") == pytest.approx(0.75)
    assert mapping.umi_to_yam(0.4, arm="right") == pytest.approx(0.3)
    assert mapping.umi_to_yam(-0.1, arm="left") == 0.0
    assert mapping.umi_to_yam(1.1, arm="right") == 1.0
    assert mapping.yam_to_umi(0.75, arm="left") == pytest.approx(0.8)
    assert mapping.yam_to_umi(0.3, arm="right") == pytest.approx(0.4)


def test_yam_ik_refreshes_fk_cache_before_every_inverse_solve() -> None:
    class _CacheSensitiveKinematics:
        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple[float, ...]]] = []
            self.last_fk: tuple[float, ...] | None = None

        def forward_kinematics(self, joints_degrees):
            joints = tuple(np.asarray(joints_degrees, dtype=np.float64))
            self.calls.append(("fk", joints))
            self.last_fk = joints
            pose = np.eye(4)
            pose[0, 3] = 1.0  # Keep every attempt outside the residual tolerance.
            return pose

        def inverse_kinematics(self, joints_degrees, _target, **_weights):
            joints = tuple(np.asarray(joints_degrees, dtype=np.float64))
            self.calls.append(("ik", joints))
            if self.last_fk != joints:
                raise AssertionError("inverse solve used a stale kinematic linearization")
            return np.asarray(joints) + 1.0

    backend = _CacheSensitiveKinematics()
    solver = object.__new__(YamArmKinematics)
    solver._kinematics = backend
    solver.n_joints = 2
    solver.lower = np.full(2, -1.0)
    solver.upper = np.full(2, 1.0)
    solver.position_weight = 1.0
    solver.orientation_weight = 1.0
    solver.max_iterations = 2
    solver.random_restarts = 1
    solver.restart_seed = 7
    solver.max_position_residual = 1e-3
    solver.max_orientation_residual = 1e-3

    result = solver.solve(np.eye(4), np.zeros(2))

    assert not result.converged
    # The bounded projection adds FK-only residual evaluations between the
    # continuity attempt and restart. Every Placo inverse call must still be
    # immediately preceded by FK at that exact linearization point.
    inverse_indices = [index for index, (name, _) in enumerate(backend.calls) if name == "ik"]
    assert len(inverse_indices) == 4
    for index in inverse_indices:
        assert backend.calls[index - 1][0] == "fk"
        assert backend.calls[index - 1][1] == backend.calls[index][1]


class _LinearBoundedFallbackKinematics:
    """A radians-linear FK whose deliberately stalled Placo solve needs fallback."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def forward_kinematics(self, joints_degrees):
        self.calls.append("fk")
        pose = np.eye(4)
        pose[0, 3] = np.deg2rad(np.asarray(joints_degrees, dtype=np.float64)[0])
        return pose

    def inverse_kinematics(self, joints_degrees, _target, **_weights):
        self.calls.append("ik")
        return np.asarray(joints_degrees, dtype=np.float64)


def _linear_bounded_fallback_solver(*, random_restarts: int) -> YamArmKinematics:
    solver = object.__new__(YamArmKinematics)
    solver._kinematics = _LinearBoundedFallbackKinematics()
    solver.n_joints = 2
    solver.lower = np.asarray((-0.2, 0.0))
    solver.upper = np.asarray((0.3, 0.4))
    solver.position_weight = 1.0
    solver.orientation_weight = 1.0
    solver.max_iterations = 1
    solver.random_restarts = random_restarts
    solver.restart_seed = 7
    solver.max_position_residual = 1e-3
    solver.max_orientation_residual = 1e-3
    return solver


def test_yam_ik_uses_bounded_fallback_before_random_restarts(monkeypatch) -> None:
    solver = _linear_bounded_fallback_solver(random_restarts=2)
    target = _transform(position=(-0.1, 0.0, 0.0))
    seed = solver.lower.copy()
    captured = {}

    def fake_least_squares(fun, x0, **kwargs):
        captured["calls_before"] = list(solver._kinematics.calls)
        captured["x0"] = np.asarray(x0).copy()
        captured.update(kwargs)
        solution = np.asarray((-0.1, 0.0))
        np.testing.assert_allclose(fun(solution), np.zeros(6), atol=1e-12)
        return SimpleNamespace(x=solution, nfev=3)

    monkeypatch.setattr(
        "lerobot.robots.bi_yam.umi_retargeting.least_squares",
        fake_least_squares,
    )

    result = solver.solve(target, seed)

    assert result.converged
    assert result.solver == IK_SOLVER_BOUNDED_LEAST_SQUARES
    assert result.restart_index == 0
    assert captured["calls_before"].count("ik") == 1
    assert solver._kinematics.calls.count("ik") == 1
    np.testing.assert_array_equal(captured["bounds"][0], solver.lower)
    np.testing.assert_array_equal(captured["bounds"][1], solver.upper)
    assert np.all(captured["x0"] > solver.lower)
    assert np.all(captured["x0"] < solver.upper)
    assert captured["max_nfev"] == 24


def test_yam_ik_bounded_fallback_keeps_unreachable_target_fail_closed() -> None:
    solver = _linear_bounded_fallback_solver(random_restarts=0)
    target = _transform(position=(1.0, 0.0, 0.0))

    result = solver.solve(target, solver.lower.copy())

    assert not result.converged
    assert result.solver == IK_SOLVER_BOUNDED_LEAST_SQUARES
    assert result.position_error >= 0.7 - 1e-8
    assert np.all(result.joints >= solver.lower)
    assert np.all(result.joints <= solver.upper)


def test_yam_ik_bounded_fallback_preserves_diagnostic_orientation_weight(monkeypatch) -> None:
    solver = _linear_bounded_fallback_solver(random_restarts=0)
    target = _transform(position=(-0.1, 0.0, 0.0), rotvec=(0.0, 0.0, 0.01))
    captured = {}

    def fake_least_squares(fun, _x0, **_kwargs):
        solution = np.asarray((-0.1, 0.0))
        captured["normalized"] = fun(solution)
        return SimpleNamespace(x=solution, nfev=1)

    monkeypatch.setattr(
        "lerobot.robots.bi_yam.umi_retargeting.least_squares",
        fake_least_squares,
    )

    result = solver.solve(target, solver.lower.copy(), orientation_weight=0.02)

    np.testing.assert_allclose(captured["normalized"][:3], 0.0, atol=1e-12)
    assert np.linalg.norm(captured["normalized"][3:]) == pytest.approx(0.2)
    assert not result.converged


def test_current_relative_gripper_json_requires_verified_per_device_endpoints() -> None:
    def entry(device: str, arm: str, *, verified: bool) -> dict:
        return {
            "closed_width_mm": 25.0 if device == "umi1" else 20.0,
            "open_width_mm": 125.0 if device == "umi1" else 120.0,
            "verified": verified,
            "dataset_device": device,
            "assigned_arm": arm,
            "device_id": f"UMI-AIRY-SN-{device[-1]}",
            "evidence_uri": f"lab://endpoint-sweeps/2026-08-09/{device}",
            "evidence_sha256": _sha256_text(f"{device}-evidence"),
            "detector_config_id": "aruco-detector-v3",
            "detector_config_sha256": _sha256_text("aruco-detector-v3"),
            "fisheye_calibration_id": f"fisheye-{device}-2026-08-09",
            "fisheye_calibration_sha256": _sha256_text(f"fisheye-{device}"),
            "geometry_config_id": "umi-jaw-tags-v2",
            "geometry_config_sha256": _sha256_text("umi-jaw-tags-v2"),
        }

    payload = {
        "schema_version": 2,
        "umi1": entry("umi1", "left", verified=True),
        "umi2": entry("umi2", "right", verified=False),
    }

    mapping = current_relative_gripper_map_from_json(payload)
    with pytest.raises(RuntimeError, match="right"):
        mapping.assert_hardware_ready()
    payload["umi2"]["verified"] = True
    current_relative_gripper_map_from_json(payload).assert_hardware_ready()


def test_current_relative_gripper_placeholder_provenance_cannot_be_blessed_by_verified_flag() -> None:
    digest = hashlib.sha256(b"real-config").hexdigest()
    payload = {
        "schema_version": 2,
        "umi1": {
            "closed_width_mm": 25.0,
            "open_width_mm": 125.0,
            "verified": True,
            "dataset_device": "umi1",
            "assigned_arm": "left",
            "device_id": "REPLACE_UMI1_DEVICE_ID",
            "evidence_uri": "REPLACE_ENDPOINT_EVIDENCE",
            "evidence_sha256": "0" * 64,
            "detector_config_id": "REPLACE_DETECTOR_CONFIG",
            "detector_config_sha256": digest,
            "fisheye_calibration_id": "REPLACE_FISHEYE_CALIBRATION",
            "fisheye_calibration_sha256": digest,
            "geometry_config_id": "REPLACE_GEOMETRY_CONFIG",
            "geometry_config_sha256": digest,
        },
        "umi2": {
            "closed_width_mm": 20.0,
            "open_width_mm": 120.0,
            "verified": True,
            "dataset_device": "umi2",
            "assigned_arm": "right",
            "device_id": "UMI-AIRY-SN-002",
            "evidence_uri": "lab://endpoint-sweeps/2026-08-09/umi2",
            "evidence_sha256": hashlib.sha256(b"umi2-evidence").hexdigest(),
            "detector_config_id": "aruco-detector-v3",
            "detector_config_sha256": digest,
            "fisheye_calibration_id": "fisheye-umi2-2026-08-09",
            "fisheye_calibration_sha256": digest,
            "geometry_config_id": "umi-jaw-tags-v2",
            "geometry_config_sha256": digest,
        },
    }

    with pytest.raises(RuntimeError, match="placeholder"):
        current_relative_gripper_map_from_json(payload).assert_hardware_ready()


def test_example_gripper_endpoints_remain_blocked_if_only_verified_is_flipped() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "umi_yam"
        / "currentrel_gripper_endpoints.example.json"
    )
    payload = json.loads(path.read_text())
    payload["umi1"]["verified"] = True
    payload["umi2"]["verified"] = True

    with pytest.raises(RuntimeError, match="placeholder"):
        current_relative_gripper_map_from_json(payload).assert_hardware_ready()


def test_current_relative_gripper_requires_explicit_umi1_left_umi2_right_assignment() -> None:
    mapping = _verified_current_relative_gripper_map()
    payload = {
        "schema_version": 2,
        "umi1": {**mapping.left.__dict__, "assigned_arm": "right"},
        "umi2": {**mapping.right.__dict__, "assigned_arm": "left"},
    }

    with pytest.raises(RuntimeError, match="umi1->left assignment"):
        current_relative_gripper_map_from_json(payload).assert_hardware_ready()


def test_current_relative_adapter_blocks_missing_endpoint_calibration() -> None:
    adapter = YamUmiEeAdapter(
        left=_MeasuredPoseKinematics(),
        right=_MeasuredPoseKinematics(),
        action_semantics=CURRENT_RELATIVE_R6D_SE3,
    )

    with pytest.raises(RuntimeError, match="closed/open tag-width endpoints"):
        adapter.assert_hardware_ready()

    jaw_delta_adapter = YamUmiEeAdapter(
        left=_MeasuredPoseKinematics(),
        right=_MeasuredPoseKinematics(),
        action_semantics=CURRENT_RELATIVE_R6D_SE3_JAW_DELTA,
    )
    with pytest.raises(RuntimeError, match="closed/open tag-width endpoints"):
        jaw_delta_adapter.assert_hardware_ready()


def test_current_relative_command_is_absolute_driver_order_with_endpoint_mapped_grippers() -> None:
    adapter = YamUmiEeAdapter(
        left=_MeasuredPoseKinematics(),
        right=_MeasuredPoseKinematics(),
        action_semantics=CURRENT_RELATIVE_R6D_SE3,
        current_relative_gripper=_verified_current_relative_gripper_map(),
        limits=SafetyLimits(max_joint_delta=1.0, max_gripper_delta=1.0),
    )
    solved_targets_are_absolute = []

    def solve(arm, _pose, current_joints, *, target_is_base_tcp=False):
        solved_targets_are_absolute.append((arm, target_is_base_tcp))
        return current_joints + 0.01, object()

    adapter._solve_arm = solve
    observation = _yam_observation()
    row = np.zeros(14, dtype=np.float64)
    row[6] = 0.8
    row[13] = 0.25

    command = adapter.action_row_to_joint_command(row, observation)

    assert list(command) == list(YAM_SCALAR_KEYS)
    assert tuple(command[key] for key in LEFT_JOINT_KEYS) == pytest.approx((0.01,) * 6)
    assert tuple(command[key] for key in RIGHT_JOINT_KEYS) == pytest.approx((0.01,) * 6)
    assert command[LEFT_GRIPPER_KEY] == pytest.approx(0.75)
    assert command[RIGHT_GRIPPER_KEY] == pytest.approx(0.1125)
    assert solved_targets_are_absolute == [("left", True), ("right", True)]


def test_adapter_freezes_current_relative_chunk_at_inference_observation() -> None:
    adapter = YamUmiEeAdapter(
        left=_MeasuredPoseKinematics(),
        right=_MeasuredPoseKinematics(),
        action_semantics=CURRENT_RELATIVE_SE3,
    )
    adapter.capture_episode_start(_yam_observation())
    inference_observation = _yam_observation(
        left_x=0.20,
        left_yaw=0.35,
        right_x=-0.15,
        right_yaw=-0.20,
    )
    chunk = np.zeros((2, 14), dtype=np.float32)
    chunk[:, 0] = (0.03, 0.06)
    chunk[:, 7] = (-0.02, -0.04)
    chunk[:, 6] = 0.7
    chunk[:, 13] = 0.6

    resolved_at_inference = adapter.resolve_action_chunk(chunk, inference_observation)
    expected = resolve_umi_action_chunk(
        chunk,
        semantics=CURRENT_RELATIVE_SE3,
        measured_policy_state=adapter.observation_to_policy_state(inference_observation),
    )
    np.testing.assert_allclose(resolved_at_inference, expected, atol=1e-7)

    # The scheduler queues `resolved_at_inference`. Re-resolving the same rows at a
    # later measured pose would drift the targets and is deliberately a different result.
    later_observation = _yam_observation(
        left_x=0.25,
        left_yaw=0.35,
        right_x=-0.10,
        right_yaw=-0.20,
    )
    incorrectly_reanchored_later = adapter.resolve_action_chunk(chunk, later_observation)
    assert not np.allclose(incorrectly_reanchored_later, resolved_at_inference, atol=1e-4)
    np.testing.assert_allclose(resolved_at_inference, expected, atol=1e-7)


def test_adapter_default_does_not_reinterpret_existing_absolute_checkpoint_chunk() -> None:
    adapter = YamUmiEeAdapter(left=_MeasuredPoseKinematics(), right=_MeasuredPoseKinematics())
    assert adapter.action_semantics == EPISODE_START_ABSOLUTE
    chunk = np.arange(28, dtype=np.float32).reshape(2, 14) / 100.0

    # Absolute checkpoint output does not depend on a per-chunk measured anchor.
    np.testing.assert_array_equal(adapter.resolve_action_chunk(chunk, {}), chunk)


def test_primary_axis_mapping_keeps_up_and_approach_without_episode_fitting() -> None:
    # Measured URDF zero-joint flange orientation (rounded only at 1e-7 here).
    base_to_flange = np.eye(4)
    base_to_flange[:3, :3] = np.asarray([[0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]])
    base_to_flange[:3, 3] = (0.1106, 0.0, 0.1735)
    diagnostic = dual_lidar_umi_f544_candidate_arm_calibration()
    frames = EpisodeFrames(calibrations={"left": diagnostic, "right": diagnostic})
    frames.capture("left", base_to_flange)
    start_tcp = frames.target_tcp("left", np.zeros(6))

    displacements = []
    for axis in range(3):
        pose = np.zeros(6)
        pose[axis] = 0.1
        displacements.append(frames.target_tcp("left", pose)[:3, 3] - start_tcp[:3, 3])

    # UMI +X up maps to base +Z; +Y right maps to base -Y; +Z approach
    # maps to base +X at the trusted zero-joint YAM pose.
    np.testing.assert_allclose(
        displacements,
        ((0.0, 0.0, 0.1), (0.0, -0.1, 0.0), (0.1, 0.0, 0.0)),
        atol=1e-7,
    )


def test_primary_axis_and_mechanical_lever_compose_in_the_declared_direction() -> None:
    calibration = dual_lidar_umi_f544_candidate_arm_calibration()
    expected = np.asarray(
        (
            (-1.0, 0.0, 0.0, -0.10479),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, -1.0, 0.22244),
            (0.0, 0.0, 0.0, 1.0),
        )
    )

    np.testing.assert_allclose(
        calibration.yam_tcp_from_logged,
        YAM_TCP_FROM_UMI_TCP @ np.linalg.inv(DUAL_LIDAR_UMI_MECHANICAL_LOGGED_FROM_TCP),
        atol=1e-12,
    )
    np.testing.assert_allclose(calibration.yam_tcp_from_logged, expected, atol=1e-12)
    assert calibration.yam_tcp_from_umi_tcp.verified
    assert not calibration.logged_from_umi_tcp.verified


def test_supplied_umi_state_to_tcp_transform_determines_axis_mapping() -> None:
    state_to_tcp = _transform(rotvec=(0.0, 0.0, np.pi / 2))
    calibration = arm_calibration_from_transforms(
        state_to_tcp,
        np.eye(4),
        logged_from_umi_tcp_verified=True,
        logged_from_umi_tcp_source="unit-test tracker CAD",
        yam_tcp_from_umi_tcp_verified=True,
        yam_tcp_from_umi_tcp_source="unit-test isolated-axis measurement",
    )
    frames = EpisodeFrames(calibrations={"left": calibration, "right": calibration})
    frames.capture("left", np.eye(4))

    policy_move = np.zeros(6)
    policy_move[0] = 0.02
    displacement = (
        frames.target_tcp("left", policy_move)[:3, 3] - frames.target_tcp("left", np.zeros(6))[:3, 3]
    )

    # X^-1 * (+state-X) * X: a +90 deg state->TCP yaw maps it to TCP -Y.
    np.testing.assert_allclose(displacement, (0.0, -0.02, 0.0), atol=1e-9)
    report = calibration_basis_report(calibration)
    np.testing.assert_allclose(
        report["translate_x_20mm"]["tcp_translation_m"],
        (0.0, -0.02, 0.0),
        atol=1e-9,
    )
    np.testing.assert_allclose(
        report["rotate_x_10deg"]["tcp_rotvec_rad"],
        (0.0, -np.deg2rad(10.0), 0.0),
        atol=1e-9,
    )


def test_episode_frames_round_trip_with_per_arm_tcp_and_policy_transforms() -> None:
    flange_to_tcp_left = _transform((0.01, -0.02, 0.08), (0.1, 0.2, -0.1))
    tcp_to_policy_left = _transform((-0.03, 0.04, 0.01), (-0.2, 0.3, 0.5))
    flange_to_tcp_right = _transform((-0.02, 0.01, 0.06), (-0.1, 0.0, 0.2))
    tcp_to_policy_right = _transform((0.01, 0.02, -0.03), (0.4, -0.2, 0.1))

    def measured(transform: np.ndarray) -> RigidTransformCalibration:
        return RigidTransformCalibration(transform, verified=True, source="unit-test measurement")

    frames = EpisodeFrames(
        calibrations={
            "left": ArmCalibration(
                measured(flange_to_tcp_left), measured(np.eye(4)), measured(tcp_to_policy_left)
            ),
            "right": ArmCalibration(
                measured(flange_to_tcp_right), measured(np.eye(4)), measured(tcp_to_policy_right)
            ),
        }
    )
    starts = {
        "left": _transform((0.3, -0.2, 0.4), (0.2, -0.4, 0.1)),
        "right": _transform((-0.1, 0.5, 0.2), (-0.3, 0.1, 0.6)),
    }
    current = {
        "left": _transform((0.4, 0.1, 0.5), (-0.1, 0.2, 0.3)),
        "right": _transform((0.2, -0.3, 0.6), (0.4, -0.2, -0.1)),
    }
    for arm in ("left", "right"):
        frames.capture(arm, starts[arm])
        policy_pose = frames.to_policy(arm, current[arm])
        np.testing.assert_allclose(frames.target_flange(arm, policy_pose), current[arm], atol=1e-9)


def test_hardware_gate_lists_every_unmeasured_transform() -> None:
    frames = EpisodeFrames()
    with pytest.raises(RuntimeError, match="left.logged_from_umi_tcp") as error:
        frames.assert_hardware_ready()
    assert "right.logged_from_umi_tcp" in str(error.value)
    assert "yam_tcp_from_umi_tcp" not in str(error.value)
    assert "flange_to_tcp" not in str(error.value)

    estimated = dual_lidar_umi_f544_candidate_arm_calibration()
    measured = ArmCalibration(
        RigidTransformCalibration(
            estimated.flange_to_tcp.transform, verified=True, source="authoritative URDF"
        ),
        RigidTransformCalibration(
            estimated.logged_from_umi_tcp.transform,
            verified=True,
            source="measured tracker to jaw",
        ),
        RigidTransformCalibration(
            estimated.yam_tcp_from_umi_tcp.transform,
            verified=True,
            source="measured isolated axes",
        ),
    )
    EpisodeFrames(calibrations={"left": measured, "right": measured}).assert_hardware_ready()


def test_calibration_json_requires_independent_boolean_provenance() -> None:
    rigid = {
        "transform": np.eye(4).tolist(),
        "verified": True,
        "source": "unit-test measurement",
    }
    payload = {
        "schema_version": 1,
        "umi1": {
            "logged_from_umi_tcp": rigid,
            "yam_tcp_from_umi_tcp": rigid,
        },
        "umi2": {
            "logged_from_umi_tcp": rigid,
            "yam_tcp_from_umi_tcp": rigid,
        },
    }
    calibrations = calibrations_from_json(payload)
    assert calibrations["left"].hardware_ready
    assert calibrations["right"].hardware_ready

    payload["umi2"]["yam_tcp_from_umi_tcp"] = {**rigid, "verified": "false"}
    with pytest.raises(ValueError, match="JSON boolean"):
        calibrations_from_json(payload)


def test_safety_limits_apply_rate_and_operational_bounds() -> None:
    limits = SafetyLimits(max_joint_delta=0.02, max_gripper_delta=0.05)
    current = np.asarray((3.14, 0.01, 0.01, 0.0, 0.0, 0.0))
    target = np.asarray((9.0, -9.0, 9.0, -9.0, 9.0, -9.0))
    command = limits.clamp_joints(target, current)
    assert np.max(np.abs(command - current)) <= 0.0200001
    assert command[0] <= limits.joint_upper[0]
    assert command[1] >= limits.joint_lower[1]
    assert limits.clamp_gripper(0.0, 1.0) == pytest.approx(0.95)


def test_safety_limits_accept_exact_per_arm_kinematics_bounds() -> None:
    limits = SafetyLimits(max_joint_delta=10.0)
    current = np.zeros(6)
    target = np.ones(6)
    left_lower = np.full(6, -0.4)
    left_upper = np.full(6, 0.4)
    right_lower = np.full(6, -0.2)
    right_upper = np.full(6, 0.2)

    left = limits.clamp_joints(
        target,
        current,
        joint_lower=left_lower,
        joint_upper=left_upper,
    )
    right = limits.clamp_joints(
        target,
        current,
        joint_lower=right_lower,
        joint_upper=right_upper,
    )

    np.testing.assert_allclose(left, left_upper)
    np.testing.assert_allclose(right, right_upper)


def test_gripper_map_uses_physical_aperture_without_scene_fit() -> None:
    mapping = GripperMap()
    closed_umi = mapping.yam_to_umi(0.0, arm="left")
    open_umi = mapping.yam_to_umi(1.0, arm="right")

    assert closed_umi == pytest.approx(4.88 / 125.0)
    assert open_umi == pytest.approx(89.02 / 125.0)
    assert mapping.umi_to_yam(closed_umi, arm="right") == pytest.approx(0.0)
    assert mapping.umi_to_yam(open_umi, arm="left") == pytest.approx(1.0)
    assert mapping.umi_to_yam(1.0, arm="left") == 1.0


def test_gripper_event_detection_rejects_noise_and_keeps_open_close_motion() -> None:
    signal = np.ones(60)
    signal[10:16] = np.linspace(1.0, 0.6, 6)
    signal[16:30] = 0.6
    signal[30:38] = np.linspace(0.6, 1.0, 8)
    signal[45] -= 1e-3
    config = EventAwareSmoothingConfig(event_padding_s=0.0)

    event_mask = detect_gripper_motion_events(signal, fps=30.0, config=config)

    assert event_mask[10:16].all()
    assert event_mask[30:38].all()
    assert not event_mask[:10].any()
    assert not event_mask[40:].any()


def test_event_aware_smoothing_preserves_events_and_bounds_fk_deviation() -> None:
    frame_count = 60
    time = np.linspace(0.0, 1.0, frame_count)
    joints = np.zeros((frame_count, 6))
    joints[:, 0] = 0.15 * np.sin(2 * np.pi * time)
    joints[1:-1:2, 0] += 0.004
    poses = np.repeat(np.eye(4)[None], frame_count, axis=0)
    poses[:, 0, 3] = joints[:, 0]
    gripper = np.ones(frame_count)
    gripper[10:16] = np.linspace(1.0, 0.6, 6)
    gripper[16:30] = 0.6
    gripper[30:38] = np.linspace(0.6, 1.0, 8)
    trajectory = ArmTrajectory(
        arm="left",
        joints=joints,
        requested_joints=joints,
        gripper=gripper,
        target_tcp=poses,
        achieved_tcp=poses,
        ik_position_error=np.zeros(frame_count),
        ik_orientation_error=np.zeros(frame_count),
        position_error=np.zeros(frame_count),
        orientation_error=np.zeros(frame_count),
        ik_converged=np.ones(frame_count, dtype=bool),
        rate_limited=np.zeros(frame_count, dtype=bool),
        restart_index=np.zeros(frame_count, dtype=np.int64),
    )

    class LinearFakeKinematics:
        lower = np.full(6, -1.0)
        upper = np.full(6, 1.0)

        @staticmethod
        def fk(sample, flange_to_tcp):
            del flange_to_tcp
            pose = np.eye(4)
            pose[0, 3] = sample[0]
            return pose

    config = EventAwareSmoothingConfig(
        event_padding_s=0.0,
        max_position_deviation_m=10e-3,
        acceleration_weight=30.0,
        jerk_weight=3.0,
    )
    result = smooth_arm_trajectory_between_gripper_events(
        trajectory,
        LinearFakeKinematics(),
        np.eye(4),
        source_gripper_signal=gripper,
        fps=30.0,
        config=config,
    )

    np.testing.assert_array_equal(result.trajectory.joints[result.event_mask], joints[result.event_mask])
    np.testing.assert_array_equal(result.trajectory.joints[[0, -1]], joints[[0, -1]])
    assert set(result.trajectory.ik_solver) == {IK_SOLVER_PLACO}
    assert result.trajectory.summary()["bounded_least_squares_frames"] == 0
    assert np.max(result.position_deviation) <= config.max_position_deviation_m + 1e-12
    assert (
        result.summary()["smoothed_jerk_norm_rad_s3_p90"] < result.summary()["reference_jerk_norm_rad_s3_p90"]
    )


def test_time_parameterization_inserts_samples_instead_of_clipping_path() -> None:
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    joints = np.asarray(
        (
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.10, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.12, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
    )
    trajectory = ArmTrajectory(
        arm="left",
        joints=joints,
        requested_joints=joints,
        gripper=np.asarray((1.0, 0.5, 0.5)),
        target_tcp=poses,
        achieved_tcp=poses,
        ik_position_error=np.zeros(3),
        ik_orientation_error=np.zeros(3),
        position_error=np.zeros(3),
        orientation_error=np.zeros(3),
        ik_converged=np.ones(3, dtype=bool),
        rate_limited=np.zeros(3, dtype=bool),
        restart_index=np.zeros(3, dtype=np.int64),
    )

    class LinearFakeKinematics:
        @staticmethod
        def fk(sample, flange_to_tcp):
            del sample, flange_to_tcp
            return np.eye(4)

    retimed = time_parameterize_arm_trajectory(
        trajectory,
        LinearFakeKinematics(),
        np.eye(4),
        source_fps=30.0,
        output_fps=30.0,
        max_joint_velocity=0.6,
        max_gripper_velocity=1.5,
    )

    assert len(retimed.joints) > len(trajectory.joints)
    assert np.max(np.abs(np.diff(retimed.joints, axis=0))) <= 0.6 / 30.0 + 1e-12
    assert np.max(np.abs(np.diff(retimed.gripper))) <= 1.5 / 30.0 + 1e-12
    np.testing.assert_allclose(retimed.joints[-1], trajectory.joints[-1])
    assert retimed.source_frame[-1] == pytest.approx(2.0)


def _write_raw_dataset(root, *, corrupt_action: bool = False) -> None:
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"fps": 30}))

    states = []
    actions = []
    gripper1 = []
    gripper2 = []
    timestamps = []
    frame_indices = []
    episode_indices = []
    for episode in (0, 1):
        base = np.zeros((3, 12), dtype=np.float32)
        base[:, 0] = (episode, episode + 0.01, episode + 0.03)
        base[:, 7] = (0.4 + episode, 0.42 + episode, 0.47 + episode)
        base[:, 9] = (0.1, 0.2, 0.3)
        delta = np.zeros_like(base)
        delta[:-1] = base[1:] - base[:-1]
        states.append(base)
        actions.append(delta)
        gripper1.extend((125.0, 100.0, 125.0))
        gripper2.extend((118.0, 90.0, 118.0))
        timestamps.extend(np.arange(3) / 30.0)
        frame_indices.extend(range(3))
        episode_indices.extend((episode,) * 3)
    states_array = np.concatenate(states)
    actions_array = np.concatenate(actions)
    if corrupt_action:
        actions_array[3, 0] += np.float32(0.001)

    table = pa.table(
        {
            "observation.state": pa.FixedSizeListArray.from_arrays(
                pa.array(states_array.ravel(), type=pa.float32()), 12
            ),
            "action": pa.FixedSizeListArray.from_arrays(
                pa.array(actions_array.ravel(), type=pa.float32()), 12
            ),
            "observation.gripper_width.umi1": pa.array(gripper1, type=pa.float32()),
            "observation.gripper_width.umi2": pa.array(gripper2, type=pa.float32()),
            "timestamp": pa.array(timestamps, type=pa.float32()),
            "frame_index": pa.array(frame_indices, type=pa.int64()),
            "episode_index": pa.array(episode_indices, type=pa.int64()),
        }
    )
    pq.write_table(table, root / "data" / "chunk-000" / "file-000.parquet")


def test_raw_loader_selects_episode_by_column_and_reconstructs_policy_targets(tmp_path) -> None:
    _write_raw_dataset(tmp_path)

    episode = load_raw_umi_episode(tmp_path, 1)

    assert episode.episode_index == 1
    assert episode.fps == 30
    np.testing.assert_array_equal(episode.frame_index, (0, 1, 2))
    np.testing.assert_allclose(episode.policy_state[0, :6], 0.0)
    np.testing.assert_allclose(episode.policy_state[0, 7:13], 0.0)
    np.testing.assert_array_equal(episode.policy_action[:-1], episode.policy_state[1:])
    np.testing.assert_array_equal(episode.policy_action[-1], episode.policy_state[-1])
    assert episode.policy_state[0, 6] == pytest.approx(1.0)
    assert episode.policy_state[0, 13] == pytest.approx(118.0 / 125.0)


def test_raw_loader_rejects_a_broken_delta_contract(tmp_path) -> None:
    _write_raw_dataset(tmp_path, corrupt_action=True)
    with pytest.raises(ValueError, match=r"action\[t\]"):
        load_raw_umi_episode(tmp_path, 1)
