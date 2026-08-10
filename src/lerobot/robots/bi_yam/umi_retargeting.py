# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Transport-independent retargeting from bimanual UMI poses to YAM joints.

The UMI pose stream and a YAM arm do not share a coordinate frame.  This module
therefore keeps three concerns separate:

* :class:`ArmCalibration` contains measured rigid transforms and their provenance;
* :class:`EpisodeFrames` performs only SE(3) frame conversion;
* :class:`YamArmKinematics` performs sequential, residual-checked FK/IK.

Action chunks have an equally explicit contract.  The deployed 14-D checkpoint
uses episode-start absolute targets.  A separately retrained current-relative
checkpoint may opt into ``current_relative_se3``; its whole chunk is resolved
once against the measured inference observation before any row is scheduled.

Neither an inter-arm baseline nor a table pose belongs in per-arm IK.  Those
quantities are needed by a shared-world collision/task scene and must be supplied
by the rig calibration rather than inferred from independently-zeroed UMI tracks.

All public joint angles are radians.  ``RobotKinematics`` uses degrees internally,
so the conversion is confined to :class:`YamArmKinematics`.
"""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from scipy import sparse
from scipy.optimize import least_squares
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation

from lerobot.model import RobotKinematics

from .config_bi_yam import YAM_JOINT_LIMITS

LEFT_JOINT_KEYS = tuple(f"left_joint_{index}.pos" for index in range(6))
RIGHT_JOINT_KEYS = tuple(f"right_joint_{index}.pos" for index in range(6))
LEFT_GRIPPER_KEY = "left_gripper.pos"
RIGHT_GRIPPER_KEY = "right_gripper.pos"
YAM_SCALAR_KEYS = (*LEFT_JOINT_KEYS, LEFT_GRIPPER_KEY, *RIGHT_JOINT_KEYS, RIGHT_GRIPPER_KEY)

LEFT_POSE_SLICE = slice(0, 6)
LEFT_GRIPPER_INDEX = 6
RIGHT_POSE_SLICE = slice(7, 13)
RIGHT_GRIPPER_INDEX = 13

LEFT_R6D_POSE_SLICE = slice(0, 9)
LEFT_R6D_GRIPPER_INDEX = 9
RIGHT_R6D_POSE_SLICE = slice(10, 19)
RIGHT_R6D_GRIPPER_INDEX = 19
CURRENT_RELATIVE_R6D_DIM = 20

YAM_URDF_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 7))
YAM_FLANGE_FRAME = "gripper"
PINNED_I2RT_COMMIT = "e599e9db90d644db6efe0b16bb6fd7cb87c4272f"
PINNED_I2RT_YAM_URDF_SHA256 = "5b9ccf966aa353ccf1e82ad6e751b899333fab7c0b4d1071316172201da68c4e"

# The midpoint of the two URDF fingertip-link origins.  Their prismatic motion is
# symmetric, so this offset is invariant to jaw opening.  The YAM URDF is the
# deployment source of truth for this project.
YAM_FLANGE_TO_TCP = np.eye(4, dtype=np.float64)
YAM_FLANGE_TO_TCP[2, 3] = -0.056683

# Physical jaw aperture derived from joint7/joint8 in the trusted YAM URDF.
YAM_CLOSED_APERTURE_MM = 4.88
YAM_OPEN_APERTURE_MM = 89.02
UMI_WIDTH_NORMALIZER_MM = 125.0

# A later capture revision measured this fixed mechanical transform for two UMI
# devices.  In the coordinate convention used by the older f544 dataset it is
# ^logged T_umi_tcp: the jaw-TCP frame expressed in the logged LiDAR-pose frame.
# The later capture used different physical Airy units, so this is useful as a
# mechanical-design candidate but is deliberately NOT deployment-verified for f544.
DUAL_LIDAR_UMI_MECHANICAL_LOGGED_FROM_TCP = np.eye(4, dtype=np.float64)
DUAL_LIDAR_UMI_MECHANICAL_LOGGED_FROM_TCP[:3, 3] = (-0.10479, 0.0, 0.22244)

# Primary model definitions, independent of any task scene: robot-umi defines
# +X up, +Y right and +Z approach; the YAM midpoint frame inherited from the
# trusted `gripper` URDF link is X down, Y right and Z back.  Ry(pi) is therefore
# the unique right-handed axis mapping from the UMI jaw TCP to the YAM midpoint.
YAM_TCP_FROM_UMI_TCP = np.eye(4, dtype=np.float64)
YAM_TCP_FROM_UMI_TCP[:3, :3] = Rotation.from_euler("y", np.pi).as_matrix()

EPISODE_START_ABSOLUTE = "episode_start_absolute"
CURRENT_RELATIVE_SE3 = "current_relative_se3"
CURRENT_RELATIVE_R6D_SE3 = "current_relative_r6d_se3"
CURRENT_RELATIVE_R6D_SE3_JAW_DELTA = "current_relative_r6d_se3_jaw_delta"
GRIPPER_ACTION_QUERY_ANCHOR_DELTA = "query_anchor_delta_normalized_width"
UmiActionSemantics = Literal[
    "episode_start_absolute",
    "current_relative_se3",
    "current_relative_r6d_se3",
    "current_relative_r6d_se3_jaw_delta",
]
_UMI_ACTION_SEMANTICS = {
    EPISODE_START_ABSOLUTE,
    CURRENT_RELATIVE_SE3,
    CURRENT_RELATIVE_R6D_SE3,
    CURRENT_RELATIVE_R6D_SE3_JAW_DELTA,
}


def _is_current_relative_r6d_semantics(semantics: UmiActionSemantics) -> bool:
    return semantics in {CURRENT_RELATIVE_R6D_SE3, CURRENT_RELATIVE_R6D_SE3_JAW_DELTA}


IK_SOLVER_PLACO = "placo"
IK_SOLVER_BOUNDED_LEAST_SQUARES = "bounded_least_squares"
IkSolver = Literal["placo", "bounded_least_squares"]
_BOUNDED_LEAST_SQUARES_MAX_NFEV = 24


def validate_trusted_yam_urdf(
    urdf_path: str | Path,
    *,
    operational_joint_limits: Mapping[str, Sequence[tuple[float, float]]] | None = None,
    expected_sha256: str | None = None,
) -> None:
    """Verify duplicated TCP/aperture constants against the deployment URDF.

    The configured operational ranges may be tighter than the URDF, but never
    wider. The gripper checks derive the midpoint of the ``joint7``/``joint8``
    child-link origins at both prismatic endpoints; they do not inspect meshes.
    """

    path = Path(urdf_path)
    if expected_sha256 is not None:
        expected_sha256 = expected_sha256.strip().lower()
        if len(expected_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in expected_sha256
        ):
            raise ValueError("expected YAM URDF SHA-256 must contain 64 hexadecimal characters")
        try:
            actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            raise ValueError(f"cannot read trusted YAM URDF {path}: {error}") from error
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"YAM URDF SHA-256 {actual_sha256} does not match pinned {expected_sha256} "
                f"from i2rt {PINNED_I2RT_COMMIT}"
            )
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as error:
        raise ValueError(f"cannot read trusted YAM URDF {path}: {error}") from error
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    required = {*(f"joint{index}" for index in range(1, 9))}
    missing = sorted(required - joints.keys())
    if missing:
        raise ValueError(f"trusted YAM URDF is missing required joints: {missing}")

    urdf_arm_limits: list[tuple[float, float]] = []
    for index in range(1, 7):
        joint = joints[f"joint{index}"]
        limit = joint.find("limit")
        if joint.get("type") != "revolute" or limit is None:
            raise ValueError(f"trusted YAM URDF joint{index} must be a bounded revolute joint")
        try:
            lower = float(limit.attrib["lower"])
            upper = float(limit.attrib["upper"])
        except (KeyError, ValueError) as error:
            raise ValueError(f"trusted YAM URDF joint{index} has invalid limits") from error
        if not np.isfinite((lower, upper)).all() or lower >= upper:
            raise ValueError(f"trusted YAM URDF joint{index} has invalid limits")
        urdf_arm_limits.append((lower, upper))

    for side, configured in (operational_joint_limits or {}).items():
        configured_array = np.asarray(configured, dtype=np.float64)
        urdf_array = np.asarray(urdf_arm_limits, dtype=np.float64)
        if configured_array.shape != urdf_array.shape or not np.isfinite(configured_array).all():
            raise ValueError(f"{side} operational joint limits must have shape (6, 2)")
        if np.any(configured_array[:, 0] >= configured_array[:, 1]):
            raise ValueError(f"{side} operational joint limits must be increasing")
        outside = np.flatnonzero(
            (configured_array[:, 0] < urdf_array[:, 0] - 1e-9)
            | (configured_array[:, 1] > urdf_array[:, 1] + 1e-9)
        )
        if outside.size:
            raise ValueError(
                f"{side} operational limits exceed trusted YAM URDF at joints {(outside + 1).tolist()}"
            )

    endpoints: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name in ("joint7", "joint8"):
        joint = joints[name]
        origin = joint.find("origin")
        axis = joint.find("axis")
        limit = joint.find("limit")
        parent = joint.find("parent")
        if (
            joint.get("type") != "prismatic"
            or origin is None
            or axis is None
            or limit is None
            or parent is None
            or parent.get("link") != YAM_FLANGE_FRAME
        ):
            raise ValueError(f"trusted YAM URDF {name} must be a bounded gripper prismatic joint")
        try:
            xyz = np.asarray([float(value) for value in origin.attrib["xyz"].split()])
            rpy = np.asarray([float(value) for value in origin.attrib.get("rpy", "0 0 0").split()])
            direction = np.asarray([float(value) for value in axis.attrib["xyz"].split()])
            lower = float(limit.attrib["lower"])
            upper = float(limit.attrib["upper"])
        except (KeyError, ValueError) as error:
            raise ValueError(f"trusted YAM URDF {name} has invalid geometry") from error
        values = (xyz, rpy, direction, np.asarray((lower, upper)))
        if any(value.shape != (3,) for value in (xyz, rpy, direction)) or not all(
            np.isfinite(value).all() for value in values
        ):
            raise ValueError(f"trusted YAM URDF {name} has invalid geometry")
        rotation = Rotation.from_euler("xyz", rpy).as_matrix()
        if not np.allclose(rotation, np.eye(3), atol=1e-5):
            raise ValueError(f"trusted YAM URDF {name} no longer inherits the flange TCP orientation")
        parent_axis = rotation @ direction
        axis_norm = float(np.linalg.norm(parent_axis))
        if not np.isclose(axis_norm, 1.0, atol=1e-6):
            raise ValueError(f"trusted YAM URDF {name} axis must be a unit vector")
        endpoints[name] = (
            xyz + parent_axis * lower,
            xyz + parent_axis * upper,
            parent_axis / axis_norm,
        )

    closed_positions = np.stack((endpoints["joint7"][0], endpoints["joint8"][0]))
    open_positions = np.stack((endpoints["joint7"][1], endpoints["joint8"][1]))
    closed_midpoint = closed_positions.mean(axis=0)
    open_midpoint = open_positions.mean(axis=0)
    if not np.allclose(closed_midpoint, open_midpoint, atol=5e-6) or not np.allclose(
        open_midpoint, YAM_FLANGE_TO_TCP[:3, 3], atol=5e-6
    ):
        raise ValueError("trusted YAM URDF fingertip midpoint disagrees with YAM_FLANGE_TO_TCP")
    aperture_axis = endpoints["joint7"][2]
    if not np.allclose(aperture_axis, -endpoints["joint8"][2], atol=1e-6):
        raise ValueError("trusted YAM URDF fingertip prismatic axes must be antiparallel")
    closed_aperture_mm = float(abs(np.dot(closed_positions[0] - closed_positions[1], aperture_axis)) * 1e3)
    open_aperture_mm = float(abs(np.dot(open_positions[0] - open_positions[1], aperture_axis)) * 1e3)
    if not np.isclose(closed_aperture_mm, YAM_CLOSED_APERTURE_MM, atol=0.02) or not np.isclose(
        open_aperture_mm, YAM_OPEN_APERTURE_MM, atol=0.02
    ):
        raise ValueError("trusted YAM URDF fingertip separation disagrees with aperture constants")


def _as_transform(value: np.ndarray, *, name: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"{name} must be a finite 4x4 transform")
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
        raise ValueError(f"{name} has an invalid homogeneous bottom row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-7
    ):
        raise ValueError(f"{name} rotation must be orthonormal with determinant +1")
    return transform.copy()


def pose_to_vec(transform: np.ndarray) -> np.ndarray:
    """Convert a homogeneous transform to ``xyz + rotation-vector``."""

    transform = _as_transform(transform, name="pose")
    return np.concatenate((transform[:3, 3], Rotation.from_matrix(transform[:3, :3]).as_rotvec())).astype(
        np.float64
    )


def vec_to_pose(vector: np.ndarray) -> np.ndarray:
    """Convert ``xyz + rotation-vector`` to a homogeneous transform."""

    vector = np.asarray(vector, dtype=np.float64)
    if vector.shape != (6,) or not np.isfinite(vector).all():
        raise ValueError("pose vector must be finite with shape (6,)")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_rotvec(vector[3:]).as_matrix()
    transform[:3, 3] = vector[:3]
    return transform


def rotation_6d_to_matrix(rotation_6d: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    """Decode the training artifact's first-two-column Rotation6D convention."""

    vector = np.asarray(rotation_6d, dtype=np.float64)
    if vector.shape != (6,) or not np.isfinite(vector).all():
        raise ValueError("Rotation6D must be finite with shape (6,)")
    first = vector[:3]
    second = vector[3:]
    first_norm = float(np.linalg.norm(first))
    if first_norm < eps:
        raise ValueError("Rotation6D first column is degenerate")
    first = first / first_norm
    second = second - float(first @ second) * first
    second_norm = float(np.linalg.norm(second))
    if second_norm < eps:
        raise ValueError("Rotation6D columns are degenerate")
    second = second / second_norm
    return np.stack((first, second, np.cross(first, second)), axis=1)


def matrix_to_rotation_6d(rotation: np.ndarray) -> np.ndarray:
    """Encode a rotation as ``[R00,R10,R20,R01,R11,R21]``."""

    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("rotation must be finite with shape (3, 3)")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = matrix
    _as_transform(transform, name="rotation")
    return np.concatenate((matrix[:, 0], matrix[:, 1]))


def r6d_pose_to_matrix(vector: np.ndarray) -> np.ndarray:
    """Decode ``xyz + Rotation6D`` into an SE(3) transform."""

    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (9,) or not np.isfinite(value).all():
        raise ValueError("R6D pose must be finite with shape (9,)")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_6d_to_matrix(value[3:])
    transform[:3, 3] = value[:3]
    return transform


def matrix_to_r6d_pose(transform: np.ndarray) -> np.ndarray:
    """Encode an SE(3) transform as ``xyz + Rotation6D``."""

    transform = _as_transform(transform, name="R6D pose")
    return np.concatenate((transform[:3, 3], matrix_to_rotation_6d(transform[:3, :3])))


def umi_delta_to_yam_delta(delta_umi_tcp: np.ndarray) -> np.ndarray:
    """Conjugate one UMI jaw-frame motion into the trusted YAM jaw frame."""

    delta = _as_transform(delta_umi_tcp, name="UMI TCP delta")
    return YAM_TCP_FROM_UMI_TCP @ delta @ invert_pose(YAM_TCP_FROM_UMI_TCP)


def yam_delta_to_umi_delta(delta_yam_tcp: np.ndarray) -> np.ndarray:
    """Inverse coordinate-basis conversion used to build measured policy state."""

    delta = _as_transform(delta_yam_tcp, name="YAM TCP delta")
    mapping_inverse = invert_pose(YAM_TCP_FROM_UMI_TCP)
    return mapping_inverse @ delta @ YAM_TCP_FROM_UMI_TCP


def invert_pose(transform: np.ndarray) -> np.ndarray:
    """Invert a homogeneous rigid transform without a generic matrix inverse."""

    transform = _as_transform(transform, name="pose")
    rotation = transform[:3, :3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ transform[:3, 3]
    return inverse


def reanchor_pose_track(pose_vectors: np.ndarray) -> np.ndarray:
    """Express an absolute pose track relative to its first pose in SE(3).

    The raw dataset stores rotation vectors and also stores their component-wise
    next-frame differences.  Those differences reconstruct the recorded pose
    vectors exactly, but they are not themselves SE(3) transforms.  Reanchoring
    therefore happens after converting every recorded pose to a matrix.
    """

    pose_vectors = np.asarray(pose_vectors, dtype=np.float64)
    if pose_vectors.ndim != 2 or pose_vectors.shape[1] != 6 or len(pose_vectors) == 0:
        raise ValueError("pose track must have shape (frames, 6) and be non-empty")
    if not np.isfinite(pose_vectors).all():
        raise ValueError("pose track contains non-finite values")
    poses = [vec_to_pose(vector) for vector in pose_vectors]
    first_inverse = invert_pose(poses[0])
    result = np.stack([pose_to_vec(first_inverse @ pose) for pose in poses])
    result[0] = 0.0
    return result


def _as_umi_policy_state(state: np.ndarray, *, name: str) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64)
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError(f"{name} must be finite with shape (14,)")
    return state.copy()


def _as_umi_action_chunk(chunk: np.ndarray, *, name: str) -> np.ndarray:
    chunk = np.asarray(chunk, dtype=np.float64)
    if chunk.ndim != 2 or chunk.shape[0] < 1 or chunk.shape[1] != 14 or not np.isfinite(chunk).all():
        raise ValueError(f"{name} must be finite with shape (horizon, 14) and a non-empty horizon")
    return chunk.copy()


def relativize_umi_action_chunk(
    episode_start_targets: np.ndarray,
    measured_policy_state: np.ndarray,
) -> np.ndarray:
    """Express absolute UMI targets relative to one measured-current snapshot in SE(3).

    Every output pose row is ``T_current^-1 @ T_future``.  Rows are therefore
    independent look-ahead targets anchored to the *same* current observation;
    they are not recursive per-step deltas.  Gripper entries remain absolute jaw
    openings.  This is the representation to use only when creating data for an
    explicitly retrained current-relative checkpoint.
    """

    targets = _as_umi_action_chunk(episode_start_targets, name="episode-start action chunk")
    current = _as_umi_policy_state(measured_policy_state, name="measured policy state")
    relative = targets.copy()
    for pose_slice in (LEFT_POSE_SLICE, RIGHT_POSE_SLICE):
        current_inverse = invert_pose(vec_to_pose(current[pose_slice]))
        for row_index in range(len(relative)):
            relative[row_index, pose_slice] = pose_to_vec(
                current_inverse @ vec_to_pose(targets[row_index, pose_slice])
            )
    return relative.astype(np.float32)


def resolve_umi_action_chunk(
    chunk: np.ndarray,
    *,
    semantics: UmiActionSemantics = EPISODE_START_ABSOLUTE,
    measured_policy_state: np.ndarray | None = None,
) -> np.ndarray:
    """Materialize a policy chunk as episode-start absolute pose targets.

    ``episode_start_absolute`` is intentionally the default because that is the
    representation used by the existing ``dual-lidar-umi-14d`` dataset and its
    MolmoAct2 checkpoint.  It returns the checkpoint output unchanged.

    ``current_relative_se3`` is opt-in for a checkpoint retrained on
    :func:`relativize_umi_action_chunk`.  All rows are composed onto one measured
    state snapshot here, before scheduling.  Calling this again against the
    changing robot state for each executed row would introduce target drift.
    Gripper entries are absolute under both representations.
    """

    if semantics not in _UMI_ACTION_SEMANTICS:
        raise ValueError(
            f"unsupported UMI action semantics {semantics!r}; expected one of {sorted(_UMI_ACTION_SEMANTICS)}"
        )
    if _is_current_relative_r6d_semantics(semantics):
        raise ValueError(
            f"{semantics} requires measured YAM TCP anchors; use resolve_current_relative_r6d_action_chunk"
        )
    resolved = _as_umi_action_chunk(chunk, name="UMI action chunk")
    if semantics == EPISODE_START_ABSOLUTE:
        return resolved.astype(np.float32)
    if measured_policy_state is None:
        raise ValueError("current_relative_se3 actions require the measured policy state for this chunk")

    current = _as_umi_policy_state(measured_policy_state, name="measured policy state")
    for pose_slice in (LEFT_POSE_SLICE, RIGHT_POSE_SLICE):
        current_pose = vec_to_pose(current[pose_slice])
        for row_index in range(len(resolved)):
            resolved[row_index, pose_slice] = pose_to_vec(
                current_pose @ vec_to_pose(resolved[row_index, pose_slice])
            )
    return resolved.astype(np.float32)


def resolve_current_relative_r6d_action_chunk(
    chunk: np.ndarray,
    *,
    measured_yam_tcp: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Freeze a 20-D query-relative chunk into absolute base-frame TCP targets.

    Each predicted pose is a UMI-TCP motion from the same query anchor. The
    coordinate change is therefore a conjugation, ``C @ delta @ C^-1``, and the
    absolute target is ``measured_tcp_at_query @ delta_yam``. The returned 14-D
    rows are an internal scheduling representation: two ``xyz + rotvec``
    base-frame TCP targets plus the two unchanged physical gripper scalars.
    """

    values = np.asarray(chunk, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[0] < 1
        or values.shape[1] != CURRENT_RELATIVE_R6D_DIM
        or not np.isfinite(values).all()
    ):
        raise ValueError(
            f"current-relative R6D action chunk must be finite with shape (horizon, "
            f"{CURRENT_RELATIVE_R6D_DIM})"
        )
    if set(measured_yam_tcp) != {"left", "right"}:
        raise ValueError("measured_yam_tcp must contain exactly 'left' and 'right'")

    anchors = {
        arm: _as_transform(measured_yam_tcp[arm], name=f"measured {arm} YAM TCP") for arm in ("left", "right")
    }
    resolved = np.empty((len(values), 14), dtype=np.float64)
    layouts = (
        ("left", LEFT_R6D_POSE_SLICE, LEFT_R6D_GRIPPER_INDEX, LEFT_POSE_SLICE, LEFT_GRIPPER_INDEX),
        (
            "right",
            RIGHT_R6D_POSE_SLICE,
            RIGHT_R6D_GRIPPER_INDEX,
            RIGHT_POSE_SLICE,
            RIGHT_GRIPPER_INDEX,
        ),
    )
    for arm, source_pose, source_gripper, target_pose, target_gripper in layouts:
        for row_index, row in enumerate(values):
            delta_yam = umi_delta_to_yam_delta(r6d_pose_to_matrix(row[source_pose]))
            resolved[row_index, target_pose] = pose_to_vec(anchors[arm] @ delta_yam)
        resolved[:, target_gripper] = values[:, source_gripper]
    return resolved.astype(np.float32)


def resolve_current_relative_r6d_jaw_delta_action_chunk(
    chunk: np.ndarray,
    *,
    measured_yam_tcp: Mapping[str, np.ndarray],
    measured_umi_gripper: Mapping[str, float],
) -> np.ndarray:
    """Resolve v4 pose deltas and same-query-anchor jaw deltas.

    Pose rows use the existing query-relative SE(3) resolution. Jaw rows are
    ``clip(u_query + delta_u[k], 0, 1)`` for every ``k``. The same freshly
    measured, endpoint-calibrated ``u_query`` is used throughout the chunk; no
    row is integrated from the preceding row.
    """

    if set(measured_umi_gripper) != {"left", "right"}:
        raise ValueError("measured_umi_gripper must contain exactly 'left' and 'right'")
    anchors = {arm: float(measured_umi_gripper[arm]) for arm in ("left", "right")}
    if not np.isfinite(tuple(anchors.values())).all():
        raise ValueError("measured UMI gripper query anchors must be finite")

    resolved = resolve_current_relative_r6d_action_chunk(
        chunk,
        measured_yam_tcp=measured_yam_tcp,
    )
    values = np.asarray(chunk, dtype=np.float64)
    for arm, source_gripper, target_gripper in (
        ("left", LEFT_R6D_GRIPPER_INDEX, LEFT_GRIPPER_INDEX),
        ("right", RIGHT_R6D_GRIPPER_INDEX, RIGHT_GRIPPER_INDEX),
    ):
        resolved[:, target_gripper] = np.clip(
            anchors[arm] + values[:, source_gripper],
            0.0,
            1.0,
        )
    return resolved


@dataclass(frozen=True)
class RigidTransformCalibration:
    """A rigid transform plus explicit deployment-verification provenance."""

    transform: np.ndarray
    verified: bool
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "transform", _as_transform(self.transform, name="calibration transform"))
        if not self.source.strip():
            raise ValueError("calibration source must be non-empty")


@dataclass(frozen=True)
class ArmCalibration:
    """Per-arm frame calibration.

    Transform names use ``A_from_B``: the matrix maps coordinates in frame B to
    coordinates in frame A.

    ``flange_to_tcp`` is ``^flange T_yam_tcp`` and comes from the trusted YAM
    URDF. ``logged_from_umi_tcp`` is ``^logged T_umi_tcp`` and corrects the
    tracked LiDAR body to the physical UMI jaw midpoint. ``yam_tcp_from_umi_tcp``
    is ``^yam_tcp T_umi_tcp`` and aligns the two embodiments' jaw axes. These last
    two transforms are physically distinct and must never share one verification
    flag.

    For a logged relative motion ``P``, the corresponding YAM-TCP motion is
    ``C @ P @ inverse(C)``, where
    ``C = yam_tcp_from_umi_tcp @ inverse(logged_from_umi_tcp)``.
    """

    flange_to_tcp: RigidTransformCalibration
    logged_from_umi_tcp: RigidTransformCalibration
    yam_tcp_from_umi_tcp: RigidTransformCalibration

    @property
    def yam_tcp_from_logged(self) -> np.ndarray:
        """Coordinate-basis transform used to conjugate logged UMI motion."""

        return self.yam_tcp_from_umi_tcp.transform @ invert_pose(self.logged_from_umi_tcp.transform)

    @property
    def hardware_ready(self) -> bool:
        return (
            self.flange_to_tcp.verified
            and self.logged_from_umi_tcp.verified
            and self.yam_tcp_from_umi_tcp.verified
        )

    def unconfirmed_fields(self, arm: str) -> list[str]:
        missing = []
        if not self.flange_to_tcp.verified:
            missing.append(f"{arm}.flange_to_tcp ({self.flange_to_tcp.source})")
        if not self.logged_from_umi_tcp.verified:
            missing.append(f"{arm}.logged_from_umi_tcp ({self.logged_from_umi_tcp.source})")
        if not self.yam_tcp_from_umi_tcp.verified:
            missing.append(f"{arm}.yam_tcp_from_umi_tcp ({self.yam_tcp_from_umi_tcp.source})")
        return missing


def dual_lidar_umi_f544_candidate_arm_calibration() -> ArmCalibration:
    """Return the partly verified calibration used for f544 visualization.

    The LiDAR lever arm is transferred from a later pair of the same mechanical
    design, but not the physical f544 capture units. The jaw-axis alignment is
    derived from primary UMI/YAM model definitions rather than task motion. The
    unconfirmed per-device lever arm alone keeps hardware execution blocked.
    """

    return ArmCalibration(
        flange_to_tcp=RigidTransformCalibration(
            YAM_FLANGE_TO_TCP,
            verified=True,
            source="authoritative YAM URDF fingertip midpoint",
        ),
        logged_from_umi_tcp=RigidTransformCalibration(
            DUAL_LIDAR_UMI_MECHANICAL_LOGGED_FROM_TCP,
            verified=False,
            source=(
                "mechanical-design transfer from later robot-umi PR20 calibration; f544 Airy serials differ"
            ),
        ),
        yam_tcp_from_umi_tcp=RigidTransformCalibration(
            YAM_TCP_FROM_UMI_TCP,
            verified=True,
            source=(
                "primary robot-umi +X-up/+Y-right/+Z-approach and trusted "
                "YAM midpoint X-down/Y-right/Z-back definitions"
            ),
        ),
    )


def unverified_identity_arm_calibration() -> ArmCalibration:
    """Assume only that the logged UMI origin is already its jaw midpoint."""

    return ArmCalibration(
        flange_to_tcp=RigidTransformCalibration(
            YAM_FLANGE_TO_TCP,
            verified=True,
            source="authoritative YAM URDF fingertip midpoint",
        ),
        logged_from_umi_tcp=RigidTransformCalibration(
            np.eye(4),
            verified=False,
            source="identity hypothesis: UMI logged origin may already be its jaw midpoint",
        ),
        yam_tcp_from_umi_tcp=RigidTransformCalibration(
            YAM_TCP_FROM_UMI_TCP,
            verified=True,
            source=(
                "primary robot-umi +X-up/+Y-right/+Z-approach and trusted "
                "YAM midpoint X-down/Y-right/Z-back definitions"
            ),
        ),
    )


def arm_calibration_from_transforms(
    logged_from_umi_tcp: np.ndarray,
    yam_tcp_from_umi_tcp: np.ndarray,
    *,
    logged_from_umi_tcp_verified: bool,
    logged_from_umi_tcp_source: str,
    yam_tcp_from_umi_tcp_verified: bool,
    yam_tcp_from_umi_tcp_source: str,
) -> ArmCalibration:
    """Build an arm calibration without conflating tracker and embodiment axes.

    ``logged_from_umi_tcp`` corrects the pose's tracked point and axes to the UMI
    jaw TCP. ``yam_tcp_from_umi_tcp`` then maps those jaw axes into the trusted
    YAM TCP convention. Supplying one does not verify the other.
    """

    return ArmCalibration(
        flange_to_tcp=RigidTransformCalibration(
            YAM_FLANGE_TO_TCP,
            verified=True,
            source="authoritative YAM URDF fingertip midpoint",
        ),
        logged_from_umi_tcp=RigidTransformCalibration(
            logged_from_umi_tcp,
            verified=logged_from_umi_tcp_verified,
            source=logged_from_umi_tcp_source,
        ),
        yam_tcp_from_umi_tcp=RigidTransformCalibration(
            yam_tcp_from_umi_tcp,
            verified=yam_tcp_from_umi_tcp_verified,
            source=yam_tcp_from_umi_tcp_source,
        ),
    )


def _calibration_entry(
    entry: object,
    *,
    name: str,
) -> RigidTransformCalibration:
    if not isinstance(entry, Mapping):
        raise ValueError(f"{name} must be an object with transform, verified, and source")
    try:
        transform = np.asarray(entry["transform"], dtype=np.float64)
        verified = entry["verified"]
        source = entry["source"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must define transform, verified, and source") from error
    if not isinstance(verified, bool):
        raise ValueError(f"{name}.verified must be a JSON boolean")
    if not isinstance(source, str):
        raise ValueError(f"{name}.source must be a string")
    return RigidTransformCalibration(transform, verified=verified, source=source)


def calibrations_from_json(payload: object) -> dict[str, ArmCalibration]:
    """Parse the strict, provenance-bearing two-device calibration schema."""

    if not isinstance(payload, Mapping):
        raise ValueError("calibration JSON root must be an object")
    if payload.get("schema_version") != 1:
        raise ValueError("calibration JSON schema_version must be 1")
    calibrations: dict[str, ArmCalibration] = {}
    for arm, device in (("left", "umi1"), ("right", "umi2")):
        entry = payload.get(device)
        if not isinstance(entry, Mapping):
            raise ValueError(f"calibration JSON must define object {device!r}")
        logged = _calibration_entry(
            entry.get("logged_from_umi_tcp"),
            name=f"{device}.logged_from_umi_tcp",
        )
        axes = _calibration_entry(
            entry.get("yam_tcp_from_umi_tcp"),
            name=f"{device}.yam_tcp_from_umi_tcp",
        )
        calibrations[arm] = ArmCalibration(
            flange_to_tcp=RigidTransformCalibration(
                YAM_FLANGE_TO_TCP,
                verified=True,
                source="authoritative YAM URDF fingertip midpoint",
            ),
            logged_from_umi_tcp=logged,
            yam_tcp_from_umi_tcp=axes,
        )
    return calibrations


def load_calibrations_json(path: str | Path) -> dict[str, ArmCalibration]:
    """Load :func:`calibrations_from_json` from disk."""

    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid calibration JSON {path}: {error}") from error
    return calibrations_from_json(payload)


def calibration_basis_report(calibration: ArmCalibration) -> dict[str, dict[str, list[float]]]:
    """Report how six standardized logged-frame motions act in the jaw-TCP frame."""

    yam_tcp_from_logged = calibration.yam_tcp_from_logged
    logged_from_yam_tcp = invert_pose(yam_tcp_from_logged)
    report: dict[str, dict[str, list[float]]] = {}
    for index, axis in enumerate("xyz"):
        policy_motion = np.eye(4)
        policy_motion[index, 3] = 0.02
        tcp_motion = yam_tcp_from_logged @ policy_motion @ logged_from_yam_tcp
        report[f"translate_{axis}_20mm"] = {
            "tcp_translation_m": tcp_motion[:3, 3].tolist(),
            "tcp_rotvec_rad": Rotation.from_matrix(tcp_motion[:3, :3]).as_rotvec().tolist(),
        }
    for index, axis in enumerate("xyz"):
        policy_motion = np.eye(4)
        rotvec = np.zeros(3)
        rotvec[index] = np.deg2rad(10.0)
        policy_motion[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
        tcp_motion = yam_tcp_from_logged @ policy_motion @ logged_from_yam_tcp
        report[f"rotate_{axis}_10deg"] = {
            "tcp_translation_m": tcp_motion[:3, 3].tolist(),
            "tcp_rotvec_rad": Rotation.from_matrix(tcp_motion[:3, :3]).as_rotvec().tolist(),
        }
    return report


@dataclass
class EpisodeFrames:
    """Per-arm episode anchors with independent, provenance-bearing calibration."""

    calibrations: dict[str, ArmCalibration] = field(
        default_factory=lambda: {
            "left": unverified_identity_arm_calibration(),
            "right": unverified_identity_arm_calibration(),
        }
    )
    _base_to_policy_start: dict[str, np.ndarray] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if set(self.calibrations) != {"left", "right"}:
            raise ValueError("calibrations must contain exactly 'left' and 'right'")

    def reset(self) -> None:
        self._base_to_policy_start.clear()

    def capture(self, arm: str, base_to_flange: np.ndarray) -> None:
        calibration = self._calibration(arm)
        base_to_flange = _as_transform(base_to_flange, name=f"{arm} base_to_flange")
        base_to_tcp = base_to_flange @ calibration.flange_to_tcp.transform
        self._base_to_policy_start[arm] = base_to_tcp @ calibration.yam_tcp_from_logged

    @property
    def captured(self) -> bool:
        return set(self._base_to_policy_start) == {"left", "right"}

    def to_policy(self, arm: str, base_to_flange: np.ndarray) -> np.ndarray:
        """Map a measured base->flange pose to the episode-relative policy pose."""

        calibration = self._calibration(arm)
        anchor = self._anchor(arm)
        base_to_tcp = _as_transform(base_to_flange, name=f"{arm} base_to_flange") @ (
            calibration.flange_to_tcp.transform
        )
        base_to_policy = base_to_tcp @ calibration.yam_tcp_from_logged
        return pose_to_vec(invert_pose(anchor) @ base_to_policy)

    def target_tcp(self, arm: str, policy_pose: np.ndarray) -> np.ndarray:
        """Map an episode-relative policy pose to a desired base->TCP pose."""

        calibration = self._calibration(arm)
        anchor = self._anchor(arm)
        base_to_policy = anchor @ vec_to_pose(policy_pose)
        return base_to_policy @ invert_pose(calibration.yam_tcp_from_logged)

    def target_flange(self, arm: str, policy_pose: np.ndarray) -> np.ndarray:
        """Map an episode-relative policy pose to a desired base->flange pose."""

        calibration = self._calibration(arm)
        return self.target_tcp(arm, policy_pose) @ invert_pose(calibration.flange_to_tcp.transform)

    def assert_hardware_ready(self) -> None:
        missing = [
            field_name
            for arm, calibration in self.calibrations.items()
            for field_name in calibration.unconfirmed_fields(arm)
        ]
        if missing:
            raise RuntimeError(
                "hardware execution is blocked by estimated calibration: " + "; ".join(missing)
            )

    def _calibration(self, arm: str) -> ArmCalibration:
        try:
            return self.calibrations[arm]
        except KeyError as error:
            raise ValueError(f"unknown arm {arm!r}") from error

    def _anchor(self, arm: str) -> np.ndarray:
        if arm not in self._base_to_policy_start:
            raise RuntimeError(f"no episode anchor for {arm!r}; capture it at reset")
        return self._base_to_policy_start[arm]


@dataclass(frozen=True)
class IkResult:
    joints: np.ndarray
    position_error: float
    orientation_error: float
    converged: bool
    iterations: int
    restart_index: int
    max_joint_change: float
    solver: IkSolver

    def __post_init__(self) -> None:
        object.__setattr__(self, "joints", np.asarray(self.joints, dtype=np.float64).copy())
        if self.solver not in {IK_SOLVER_PLACO, IK_SOLVER_BOUNDED_LEAST_SQUARES}:
            raise ValueError(f"unsupported IK solver provenance {self.solver!r}")


class IkResidualError(RuntimeError):
    """Raised when no IK attempt reaches the declared TCP tolerance."""


class YamArmKinematics:
    """Sequential, residual-checked YAM FK/IK with a radians-only public API."""

    def __init__(
        self,
        urdf_path: str,
        *,
        target_frame_name: str = YAM_FLANGE_FRAME,
        joint_names: tuple[str, ...] = YAM_URDF_JOINT_NAMES,
        joint_limits: tuple[tuple[float, float], ...] = YAM_JOINT_LIMITS,
        position_weight: float = 1.0,
        orientation_weight: float = 1.0,
        max_iterations: int = 12,
        random_restarts: int = 4,
        restart_seed: int = 0,
        max_position_residual: float = 2e-3,
        max_orientation_residual: float = np.deg2rad(1.0),
    ) -> None:
        if max_iterations < 1 or random_restarts < 0:
            raise ValueError("max_iterations must be positive and random_restarts non-negative")
        if len(joint_names) != len(joint_limits):
            raise ValueError("joint_names and joint_limits must have the same length")
        limits = np.asarray(joint_limits, dtype=np.float64)
        if limits.shape != (len(joint_names), 2) or not np.isfinite(limits).all():
            raise ValueError("joint_limits must contain one finite (lower, upper) pair per joint")
        if np.any(limits[:, 0] >= limits[:, 1]):
            raise ValueError("joint_limits must contain increasing intervals")
        residual_limits = np.asarray(
            (max_position_residual, max_orientation_residual),
            dtype=np.float64,
        )
        if not np.isfinite(residual_limits).all() or np.any(residual_limits <= 0):
            raise ValueError("IK residual tolerances must be finite and positive")
        self._kinematics = RobotKinematics(
            urdf_path=urdf_path,
            target_frame_name=target_frame_name,
            joint_names=list(joint_names),
        )
        self.n_joints = len(joint_names)
        self.lower = limits[:, 0].copy()
        self.upper = limits[:, 1].copy()
        self.position_weight = float(position_weight)
        self.orientation_weight = float(orientation_weight)
        self.max_iterations = int(max_iterations)
        self.random_restarts = int(random_restarts)
        self.restart_seed = int(restart_seed)
        self.max_position_residual = float(max_position_residual)
        self.max_orientation_residual = float(max_orientation_residual)

    def _validate_joints(self, joints: np.ndarray, *, name: str) -> np.ndarray:
        joints = np.asarray(joints, dtype=np.float64)
        if joints.shape != (self.n_joints,) or not np.isfinite(joints).all():
            raise ValueError(f"{name} must be finite with shape ({self.n_joints},)")
        if np.abs(joints).max(initial=0.0) > 2 * np.pi:
            raise ValueError(f"{name} look like degrees; this API takes radians")
        return joints

    def fk(self, joints: np.ndarray, flange_to_target: np.ndarray | None = None) -> np.ndarray:
        joints = self._validate_joints(joints, name="joints")
        flange = np.asarray(self._kinematics.forward_kinematics(np.rad2deg(joints)), dtype=np.float64)
        if flange_to_target is None:
            return flange
        return flange @ _as_transform(flange_to_target, name="flange_to_target")

    def residual(
        self,
        joints: np.ndarray,
        target_pose: np.ndarray,
        *,
        flange_to_target: np.ndarray | None = None,
    ) -> tuple[float, float]:
        target_pose = _as_transform(target_pose, name="target_pose")
        reached = self.fk(joints, flange_to_target)
        position_error = float(np.linalg.norm(reached[:3, 3] - target_pose[:3, 3]))
        orientation_error = float(
            np.linalg.norm(Rotation.from_matrix(reached[:3, :3].T @ target_pose[:3, :3]).as_rotvec())
        )
        return position_error, orientation_error

    def solve(
        self,
        target_pose: np.ndarray,
        seed_joints: np.ndarray,
        *,
        flange_to_target: np.ndarray | None = None,
        orientation_weight: float | None = None,
    ) -> IkResult:
        """Solve a base->target pose, retrying deterministically only on failure."""

        target_pose = _as_transform(target_pose, name="target_pose")
        seed_joints = self._validate_joints(seed_joints, name="seed_joints")
        offset = (
            np.eye(4)
            if flange_to_target is None
            else _as_transform(flange_to_target, name="flange_to_target")
        )
        target_flange = target_pose @ invert_pose(offset)
        weight = self.orientation_weight if orientation_weight is None else float(orientation_weight)
        if weight < 0 or not np.isfinite(weight):
            raise ValueError("orientation_weight must be finite and non-negative")

        attempts: list[IkResult] = []
        continuity_result = self._iterate(
            target_pose,
            target_flange,
            offset,
            np.clip(seed_joints, self.lower, self.upper),
            seed_joints,
            weight,
            0,
        )
        attempts.append(continuity_result)
        if continuity_result.converged:
            return continuity_result

        # Placo can stall at a one-sided joint limit even when the target still
        # satisfies the declared Cartesian tolerances. Project from the same
        # continuity seed with exact operational bounds before considering any
        # unrelated restart. This is deterministic and contains no scene signal.
        bounded_result = self._bounded_least_squares(
            target_pose,
            offset,
            seed_joints,
            orientation_weight=weight,
        )
        attempts.append(bounded_result)
        if bounded_result.converged:
            return bounded_result

        rng = np.random.default_rng(self.restart_seed)
        for restart_index in range(1, self.random_restarts + 1):
            result = self._iterate(
                target_pose,
                target_flange,
                offset,
                rng.uniform(self.lower, self.upper),
                seed_joints,
                weight,
                restart_index,
            )
            attempts.append(result)

        converged = [attempt for attempt in attempts if attempt.converged]
        if converged:
            return min(converged, key=lambda attempt: (attempt.max_joint_change, attempt.iterations))
        return min(
            attempts,
            key=lambda attempt: (
                attempt.position_error / self.max_position_residual
                + attempt.orientation_error / self.max_orientation_residual,
                attempt.max_joint_change,
            ),
        )

    def _bounded_least_squares(
        self,
        target_pose: np.ndarray,
        flange_to_target: np.ndarray,
        continuity_seed: np.ndarray,
        *,
        orientation_weight: float,
    ) -> IkResult:
        """Deterministically project one target inside the exact joint bounds.

        The normalized residuals make the declared Cartesian tolerances the
        unit scales. An explicit diagnostic orientation-weight override is
        retained here rather than silently replacing that mode's objective;
        convergence is still decided by the separate, unweighted tolerances.
        """

        # scipy shifts an exactly bounded x0 into the interior. Supplying a tiny,
        # deterministic inset makes that behavior explicit while leaving the
        # optimizer's actual bounds equal to the operational limits.
        span = self.upper - self.lower
        inset = np.minimum(1e-10, span / 4.0)
        initial = np.clip(continuity_seed, self.lower + inset, self.upper - inset)

        def normalized_pose_residual(joints: np.ndarray) -> np.ndarray:
            reached = self.fk(joints, flange_to_target)
            position = (reached[:3, 3] - target_pose[:3, 3]) / self.max_position_residual
            orientation = (
                Rotation.from_matrix(reached[:3, :3].T @ target_pose[:3, :3]).as_rotvec()
                / self.max_orientation_residual
                * orientation_weight
            )
            return np.concatenate((position, orientation))

        projection = least_squares(
            normalized_pose_residual,
            initial,
            bounds=(self.lower, self.upper),
            x_scale="jac",
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            max_nfev=_BOUNDED_LEAST_SQUARES_MAX_NFEV,
        )
        joints = np.asarray(projection.x, dtype=np.float64)
        position_error, orientation_error = self.residual(
            joints,
            target_pose,
            flange_to_target=flange_to_target,
        )
        return IkResult(
            joints=joints,
            position_error=position_error,
            orientation_error=orientation_error,
            converged=(
                position_error <= self.max_position_residual
                and orientation_error <= self.max_orientation_residual
            ),
            iterations=int(projection.nfev),
            restart_index=0,
            max_joint_change=float(np.abs(joints - continuity_seed).max(initial=0.0)),
            solver=IK_SOLVER_BOUNDED_LEAST_SQUARES,
        )

    def ik(
        self,
        target_pose: np.ndarray,
        seed_joints: np.ndarray,
        *,
        flange_to_target: np.ndarray | None = None,
        check: bool = True,
    ) -> np.ndarray:
        """Compatibility wrapper returning joints and optionally raising on residual."""

        result = self.solve(target_pose, seed_joints, flange_to_target=flange_to_target)
        if check and not result.converged:
            raise IkResidualError(
                f"IK residual {result.position_error * 1e3:.2f} mm / "
                f"{np.rad2deg(result.orientation_error):.2f} deg exceeds "
                f"{self.max_position_residual * 1e3:.2f} mm / "
                f"{np.rad2deg(self.max_orientation_residual):.2f} deg"
            )
        return result.joints.copy()

    def _iterate(
        self,
        target_pose: np.ndarray,
        target_flange: np.ndarray,
        flange_to_target: np.ndarray,
        attempt_seed: np.ndarray,
        continuity_seed: np.ndarray,
        orientation_weight: float,
        restart_index: int,
    ) -> IkResult:
        joints = np.clip(attempt_seed, self.lower, self.upper)
        best: IkResult | None = None
        for iteration in range(1, self.max_iterations + 1):
            # RobotKinematics does not update its cached linearization after setting
            # the seed.  FK immediately before every solve is therefore mandatory.
            self._kinematics.forward_kinematics(np.rad2deg(joints))
            solution_degrees = self._kinematics.inverse_kinematics(
                np.rad2deg(joints),
                target_flange,
                position_weight=self.position_weight,
                orientation_weight=orientation_weight,
            )
            joints = np.clip(
                np.deg2rad(np.asarray(solution_degrees, dtype=np.float64)[: self.n_joints]),
                self.lower,
                self.upper,
            )
            position_error, orientation_error = self.residual(
                joints, target_pose, flange_to_target=flange_to_target
            )
            converged = (
                position_error <= self.max_position_residual
                and orientation_error <= self.max_orientation_residual
            )
            candidate = IkResult(
                joints=joints,
                position_error=position_error,
                orientation_error=orientation_error,
                converged=converged,
                iterations=iteration,
                restart_index=restart_index,
                max_joint_change=float(np.abs(joints - continuity_seed).max(initial=0.0)),
                solver=IK_SOLVER_PLACO,
            )
            if best is None or (
                candidate.position_error / self.max_position_residual
                + candidate.orientation_error / self.max_orientation_residual
            ) < (
                best.position_error / self.max_position_residual
                + best.orientation_error / self.max_orientation_residual
            ):
                best = candidate
            if converged:
                return candidate
        assert best is not None
        return best


@dataclass(frozen=True)
class GripperMap:
    """Object-independent conversion between physical aperture conventions.

    The UMI policy scalar is jaw width divided by 125 mm.  The YAM command is a
    normalized position between the trusted URDF's closed and open apertures.
    Values beyond the YAM's physical opening are clipped; no task outcome or
    scene geometry is used.
    """

    umi_width_normalizer_mm: float = UMI_WIDTH_NORMALIZER_MM
    yam_closed_aperture_mm: float = YAM_CLOSED_APERTURE_MM
    yam_open_aperture_mm: float = YAM_OPEN_APERTURE_MM

    def __post_init__(self) -> None:
        if self.umi_width_normalizer_mm <= 0:
            raise ValueError("UMI width normalizer must be positive")
        if self.yam_open_aperture_mm <= self.yam_closed_aperture_mm:
            raise ValueError("YAM open aperture must exceed closed aperture")

    def umi_to_yam(self, value: float, *, arm: str) -> float:
        self._validate_arm(arm)
        width_mm = float(value) * self.umi_width_normalizer_mm
        command = (width_mm - self.yam_closed_aperture_mm) / (
            self.yam_open_aperture_mm - self.yam_closed_aperture_mm
        )
        return float(np.clip(command, 0.0, 1.0))

    def yam_to_umi(self, value: float, *, arm: str) -> float:
        self._validate_arm(arm)
        width_mm = self.yam_closed_aperture_mm + float(value) * (
            self.yam_open_aperture_mm - self.yam_closed_aperture_mm
        )
        return float(width_mm / self.umi_width_normalizer_mm)

    @staticmethod
    def _validate_arm(arm: str) -> None:
        if arm not in {"left", "right"}:
            raise ValueError(f"unknown arm {arm!r}")


@dataclass(frozen=True)
class UmiGripperEndpointCalibration:
    """Measured UMI tag separations at the gripper's physical endpoints.

    The current-relative dataset scalar is ``clip(tag_separation_mm / 125, 0, 1)``.
    BiYAM instead reports and accepts a motor-endpoint fraction.  Sharing the same
    gripper mechanism makes the relationship monotonic, but it does not make these
    two numbers identical: the ArUco tags have a fixed mounting offset.  This
    endpoint calibration is therefore intentionally independent of task objects.
    """

    closed_width_mm: float
    open_width_mm: float
    verified: bool
    dataset_device: str
    assigned_arm: str
    device_id: str
    evidence_uri: str
    evidence_sha256: str
    detector_config_id: str
    detector_config_sha256: str
    fisheye_calibration_id: str
    fisheye_calibration_sha256: str
    geometry_config_id: str
    geometry_config_sha256: str

    def __post_init__(self) -> None:
        closed = float(self.closed_width_mm)
        opened = float(self.open_width_mm)
        if not np.isfinite((closed, opened)).all() or closed < 0 or opened <= closed:
            raise ValueError("UMI gripper endpoints must be finite with 0 <= closed < open")
        if not isinstance(self.verified, bool):
            raise TypeError("UMI gripper endpoint verified flag must be a boolean")
        for name in (
            "dataset_device",
            "assigned_arm",
            "device_id",
            "evidence_uri",
            "detector_config_id",
            "fisheye_calibration_id",
            "geometry_config_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"UMI gripper endpoint {name} must be a non-empty string")
        if self.dataset_device not in {"umi1", "umi2"}:
            raise ValueError("UMI gripper endpoint dataset_device must be umi1 or umi2")
        if self.assigned_arm not in {"left", "right"}:
            raise ValueError("UMI gripper endpoint assigned_arm must be left or right")
        for name in (
            "evidence_sha256",
            "detector_config_sha256",
            "fisheye_calibration_sha256",
            "geometry_config_sha256",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdefABCDEF" for character in value)
            ):
                raise ValueError(f"UMI gripper endpoint {name} must contain 64 hexadecimal characters")
        normalized_closed = float(np.clip(closed / UMI_WIDTH_NORMALIZER_MM, 0.0, 1.0))
        normalized_open = float(np.clip(opened / UMI_WIDTH_NORMALIZER_MM, 0.0, 1.0))
        if normalized_open <= normalized_closed:
            raise ValueError("UMI gripper endpoints collapse after the dataset's /125 clipping")

    def hardware_readiness_errors(self) -> list[str]:
        """Return provenance failures that a Boolean flag alone cannot override."""

        errors = []
        if not self.verified:
            errors.append("verified is false")
        placeholder_tokens = ("replace", "placeholder", "pending", "unknown", "todo", "changeme")
        for name in (
            "device_id",
            "evidence_uri",
            "detector_config_id",
            "fisheye_calibration_id",
            "geometry_config_id",
        ):
            value = getattr(self, name).strip().casefold()
            if any(token in value for token in placeholder_tokens):
                errors.append(f"{name} is a placeholder")
        for name in (
            "evidence_sha256",
            "detector_config_sha256",
            "fisheye_calibration_sha256",
            "geometry_config_sha256",
        ):
            if len(set(getattr(self, name).casefold())) == 1:
                errors.append(f"{name} is a placeholder digest")
        return errors

    @property
    def normalized_closed(self) -> float:
        return float(np.clip(self.closed_width_mm / UMI_WIDTH_NORMALIZER_MM, 0.0, 1.0))

    @property
    def normalized_open(self) -> float:
        return float(np.clip(self.open_width_mm / UMI_WIDTH_NORMALIZER_MM, 0.0, 1.0))


@dataclass(frozen=True)
class CurrentRelativeGripperMap:
    """Bidirectional map between model tag-width scalars and BiYAM commands."""

    left: UmiGripperEndpointCalibration
    right: UmiGripperEndpointCalibration

    def assert_hardware_ready(self) -> None:
        expected_assignments = {"left": "umi1", "right": "umi2"}
        failures = []
        for arm, dataset_device in expected_assignments.items():
            calibration = self._calibration(arm)
            if calibration.dataset_device != dataset_device or calibration.assigned_arm != arm:
                failures.append(
                    f"{dataset_device}->{arm} assignment is not explicit "
                    f"(got {calibration.dataset_device}->{calibration.assigned_arm})"
                )
            failures.extend(
                f"{dataset_device}->{arm}: {error}" for error in calibration.hardware_readiness_errors()
            )
        if self.left.device_id.casefold() == self.right.device_id.casefold():
            failures.append("umi1 and umi2 must identify distinct physical devices")
        if failures:
            raise RuntimeError(
                "hardware execution is blocked by incomplete UMI gripper endpoint provenance: "
                + "; ".join(failures)
            )

    def umi_to_yam(self, value: float, *, arm: str) -> float:
        calibration = self._calibration(arm)
        scalar = float(value)
        if not np.isfinite(scalar):
            raise ValueError("current-relative gripper command must be finite")
        command = (scalar - calibration.normalized_closed) / (
            calibration.normalized_open - calibration.normalized_closed
        )
        return float(np.clip(command, 0.0, 1.0))

    def yam_to_umi(self, value: float, *, arm: str) -> float:
        calibration = self._calibration(arm)
        command = float(value)
        if not np.isfinite(command):
            raise ValueError("measured YAM gripper position must be finite")
        command = float(np.clip(command, 0.0, 1.0))
        return float(
            calibration.normalized_closed
            + command * (calibration.normalized_open - calibration.normalized_closed)
        )

    def _calibration(self, arm: str) -> UmiGripperEndpointCalibration:
        if arm == "left":
            return self.left
        if arm == "right":
            return self.right
        raise ValueError(f"unknown arm {arm!r}")


def current_relative_gripper_map_from_json(payload: object) -> CurrentRelativeGripperMap:
    """Parse the strict, provenance-bearing UMI endpoint calibration schema."""

    if not isinstance(payload, Mapping):
        raise ValueError("gripper calibration JSON root must be an object")
    if payload.get("schema_version") != 2:
        raise ValueError("gripper calibration JSON schema_version must be 2")

    calibrations: dict[str, UmiGripperEndpointCalibration] = {}
    for arm, device in (("left", "umi1"), ("right", "umi2")):
        entry = payload.get(device)
        if not isinstance(entry, Mapping):
            raise ValueError(f"gripper calibration JSON must define object {device!r}")
        try:
            closed = float(entry["closed_width_mm"])
            opened = float(entry["open_width_mm"])
            verified = entry["verified"]
            string_fields = {
                name: entry[name]
                for name in (
                    "dataset_device",
                    "assigned_arm",
                    "device_id",
                    "evidence_uri",
                    "evidence_sha256",
                    "detector_config_id",
                    "detector_config_sha256",
                    "fisheye_calibration_id",
                    "fisheye_calibration_sha256",
                    "geometry_config_id",
                    "geometry_config_sha256",
                )
            }
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"{device} must define endpoint widths, verification, device/arm identity, "
                "endpoint evidence, and exact detector/fisheye/geometry config identities"
            ) from error
        if not isinstance(verified, bool):
            raise ValueError(f"{device}.verified must be a JSON boolean")
        if any(not isinstance(value, str) for value in string_fields.values()):
            raise ValueError(f"{device} provenance fields must be strings")
        calibrations[arm] = UmiGripperEndpointCalibration(
            closed_width_mm=closed,
            open_width_mm=opened,
            verified=verified,
            **string_fields,
        )
    return CurrentRelativeGripperMap(**calibrations)


def load_current_relative_gripper_map_json(path: str | Path) -> CurrentRelativeGripperMap:
    """Load :func:`current_relative_gripper_map_from_json` from disk."""

    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid gripper calibration JSON {path}: {error}") from error
    return current_relative_gripper_map_from_json(payload)


@dataclass(frozen=True)
class SafetyLimits:
    max_joint_delta: float = 0.02
    max_gripper_delta: float = 0.05
    joint_lower: tuple[float, ...] = tuple(limit[0] for limit in YAM_JOINT_LIMITS)
    joint_upper: tuple[float, ...] = tuple(limit[1] for limit in YAM_JOINT_LIMITS)

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.max_joint_delta)
            or not np.isfinite(self.max_gripper_delta)
            or self.max_joint_delta <= 0
            or self.max_gripper_delta <= 0
        ):
            raise ValueError("rate limits must be finite and positive")
        lower = np.asarray(self.joint_lower, dtype=np.float64)
        upper = np.asarray(self.joint_upper, dtype=np.float64)
        if lower.ndim != 1 or upper.shape != lower.shape or not np.isfinite((lower, upper)).all():
            raise ValueError("joint bounds must be equally sized finite vectors")
        if np.any(lower >= upper):
            raise ValueError("joint bounds must be increasing")

    def clamp_joints(
        self,
        target: np.ndarray,
        current: np.ndarray,
        *,
        joint_lower: np.ndarray | tuple[float, ...] | None = None,
        joint_upper: np.ndarray | tuple[float, ...] | None = None,
    ) -> np.ndarray:
        target = np.asarray(target, dtype=np.float64)
        current = np.asarray(current, dtype=np.float64)
        lower = np.asarray(self.joint_lower if joint_lower is None else joint_lower, dtype=np.float64)
        upper = np.asarray(self.joint_upper if joint_upper is None else joint_upper, dtype=np.float64)
        if target.shape != current.shape or target.shape != lower.shape or upper.shape != lower.shape:
            raise ValueError("joint vectors have the wrong shape")
        if not np.isfinite((target, current, lower, upper)).all() or np.any(lower >= upper):
            raise ValueError("joint vectors and bounds must be finite with increasing bounds")
        rate_limited = current + np.clip(target - current, -self.max_joint_delta, self.max_joint_delta)
        return np.clip(rate_limited, lower, upper)

    def clamp_gripper(self, target: float, current: float) -> float:
        target = float(np.clip(target, 0.0, 1.0))
        delta = np.clip(target - float(current), -self.max_gripper_delta, self.max_gripper_delta)
        return float(np.clip(float(current) + delta, 0.0, 1.0))


@dataclass(frozen=True)
class TrackingPolicy:
    mode: Literal["strict", "diagnostic"] = "strict"
    diagnostic_orientation_weight: float = 0.02
    roll_tolerance: float = 0.0
    roll_samples: int = 1

    def __post_init__(self) -> None:
        if self.roll_tolerance < 0 or self.roll_samples < 1:
            raise ValueError("roll_tolerance must be non-negative and roll_samples positive")
        if self.roll_tolerance > 0 and self.roll_samples < 2:
            raise ValueError("roll_samples must be at least 2 when roll tolerance is enabled")


@dataclass(frozen=True)
class ArmCommandDiagnostics:
    ik: IkResult
    requested_tcp: np.ndarray
    commanded_tcp: np.ndarray
    commanded_position_error: float
    commanded_orientation_error: float
    rate_limited: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "requested_tcp", np.asarray(self.requested_tcp).copy())
        object.__setattr__(self, "commanded_tcp", np.asarray(self.commanded_tcp).copy())


@dataclass
class YamUmiEeAdapter:
    """Convert measured BiYAM joints and UMI pose-policy representations."""

    left: YamArmKinematics
    right: YamArmKinematics
    frames: EpisodeFrames = field(default_factory=EpisodeFrames)
    gripper: GripperMap = field(default_factory=GripperMap)
    current_relative_gripper: CurrentRelativeGripperMap | None = None
    limits: SafetyLimits = field(default_factory=SafetyLimits)
    tracking: TrackingPolicy = field(default_factory=TrackingPolicy)
    action_semantics: UmiActionSemantics = EPISODE_START_ABSOLUTE
    last_diagnostics: dict[str, ArmCommandDiagnostics] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.action_semantics not in _UMI_ACTION_SEMANTICS:
            raise ValueError(
                f"unsupported UMI action semantics {self.action_semantics!r}; "
                f"expected one of {sorted(_UMI_ACTION_SEMANTICS)}"
            )

    def assert_hardware_ready(self) -> None:
        if _is_current_relative_r6d_semantics(self.action_semantics):
            # This representation already contains jaw-TCP motions. Independent
            # per-arm IK needs only the trusted URDF TCP and fixed jaw-axis basis.
            _as_transform(YAM_FLANGE_TO_TCP, name="YAM flange-to-TCP")
            _as_transform(YAM_TCP_FROM_UMI_TCP, name="YAM TCP from UMI TCP")
            if self.current_relative_gripper is None:
                raise RuntimeError(
                    "hardware execution is blocked until the two UMI closed/open tag-width "
                    "endpoints are supplied"
                )
            self.current_relative_gripper.assert_hardware_ready()
            return
        self.frames.assert_hardware_ready()

    @property
    def last_residual(self) -> dict[str, tuple[float, float]]:
        return {
            arm: (diagnostic.commanded_position_error, diagnostic.commanded_orientation_error)
            for arm, diagnostic in self.last_diagnostics.items()
        }

    @staticmethod
    def joints_from_observation(
        observation: dict[str, float],
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        missing = [key for key in YAM_SCALAR_KEYS if key not in observation]
        if missing:
            raise KeyError(f"observation is missing YAM scalars: {missing}")
        left = np.asarray([float(observation[key]) for key in LEFT_JOINT_KEYS], dtype=np.float64)
        right = np.asarray([float(observation[key]) for key in RIGHT_JOINT_KEYS], dtype=np.float64)
        left_gripper = float(observation[LEFT_GRIPPER_KEY])
        right_gripper = float(observation[RIGHT_GRIPPER_KEY])
        if not np.isfinite((*left, left_gripper, *right, right_gripper)).all():
            raise ValueError("measured YAM joint/gripper state must be finite")
        return left, right, left_gripper, right_gripper

    def capture_episode_start(self, observation: dict[str, float]) -> None:
        left, right, _, _ = self.joints_from_observation(observation)
        self.frames.reset()
        if not _is_current_relative_r6d_semantics(self.action_semantics):
            self.frames.capture("left", self.left.fk(left))
            self.frames.capture("right", self.right.fk(right))
        self.last_diagnostics.clear()

    def measured_yam_tcp(self, observation: dict[str, float]) -> dict[str, np.ndarray]:
        """Return fresh base-to-jaw-TCP FK for both measured arm states."""

        left, right, _, _ = self.joints_from_observation(observation)
        return {
            "left": self.left.fk(left, YAM_FLANGE_TO_TCP),
            "right": self.right.fk(right, YAM_FLANGE_TO_TCP),
        }

    def observation_to_policy_state(
        self,
        observation: dict[str, float],
        previous_observation: dict[str, float] | None = None,
    ) -> np.ndarray:
        left, right, left_gripper, right_gripper = self.joints_from_observation(observation)
        if _is_current_relative_r6d_semantics(self.action_semantics):
            previous_left, previous_right = left, right
            if previous_observation is not None:
                previous_left, previous_right, _, _ = self.joints_from_observation(previous_observation)

            state = np.empty(CURRENT_RELATIVE_R6D_DIM, dtype=np.float64)
            layouts = (
                (
                    "left",
                    self.left,
                    left,
                    previous_left,
                    left_gripper,
                    LEFT_R6D_POSE_SLICE,
                    LEFT_R6D_GRIPPER_INDEX,
                ),
                (
                    "right",
                    self.right,
                    right,
                    previous_right,
                    right_gripper,
                    RIGHT_R6D_POSE_SLICE,
                    RIGHT_R6D_GRIPPER_INDEX,
                ),
            )
            for _arm, kinematics, current_joints, previous_joints, gripper, pose_slice, grip_index in layouts:
                current_tcp = kinematics.fk(current_joints, YAM_FLANGE_TO_TCP)
                previous_tcp = kinematics.fk(previous_joints, YAM_FLANGE_TO_TCP)
                history_yam = invert_pose(current_tcp) @ previous_tcp
                state[pose_slice] = matrix_to_r6d_pose(yam_delta_to_umi_delta(history_yam))
                state[grip_index] = self._current_relative_gripper_map().yam_to_umi(
                    gripper,
                    arm=_arm,
                )
            return state.astype(np.float32)

        state = np.empty(14, dtype=np.float64)
        state[LEFT_POSE_SLICE] = self.frames.to_policy("left", self.left.fk(left))
        state[LEFT_GRIPPER_INDEX] = self.gripper.yam_to_umi(left_gripper, arm="left")
        state[RIGHT_POSE_SLICE] = self.frames.to_policy("right", self.right.fk(right))
        state[RIGHT_GRIPPER_INDEX] = self.gripper.yam_to_umi(right_gripper, arm="right")
        return state.astype(np.float32)

    def resolve_action_chunk(
        self,
        chunk: np.ndarray,
        observation_at_inference: dict[str, float],
    ) -> np.ndarray:
        """Freeze one policy chunk into episode-start absolute targets for scheduling.

        The existing UMI MolmoAct2 checkpoint takes the default path and its
        episode-start absolute outputs pass through unchanged.  For an explicitly
        retrained ``current_relative_se3`` checkpoint, the measured state is read
        once from the observation that produced the chunk.  The caller must queue
        the returned rows and must not re-resolve them against later observations.
        """

        if self.action_semantics == CURRENT_RELATIVE_R6D_SE3_JAW_DELTA:
            left, right, left_gripper, right_gripper = self.joints_from_observation(observation_at_inference)
            measured_yam_tcp = {
                "left": self.left.fk(left, YAM_FLANGE_TO_TCP),
                "right": self.right.fk(right, YAM_FLANGE_TO_TCP),
            }
            mapping = self._current_relative_gripper_map()
            measured_umi_gripper = {
                "left": mapping.yam_to_umi(left_gripper, arm="left"),
                "right": mapping.yam_to_umi(right_gripper, arm="right"),
            }
            return resolve_current_relative_r6d_jaw_delta_action_chunk(
                chunk,
                measured_yam_tcp=measured_yam_tcp,
                measured_umi_gripper=measured_umi_gripper,
            )
        if self.action_semantics == CURRENT_RELATIVE_R6D_SE3:
            return resolve_current_relative_r6d_action_chunk(
                chunk,
                measured_yam_tcp=self.measured_yam_tcp(observation_at_inference),
            )

        measured_policy_state = None
        if self.action_semantics == CURRENT_RELATIVE_SE3:
            measured_policy_state = self.observation_to_policy_state(observation_at_inference)
        return resolve_umi_action_chunk(
            chunk,
            semantics=self.action_semantics,
            measured_policy_state=measured_policy_state,
        )

    def action_row_to_joint_command(self, row: np.ndarray, observation: dict[str, float]) -> dict[str, float]:
        row = np.asarray(row, dtype=np.float64)
        if row.shape != (14,) or not np.isfinite(row).all():
            raise ValueError("policy action row must be finite with shape (14,)")
        left, right, left_gripper, right_gripper = self.joints_from_observation(observation)

        targets_are_base_tcp = _is_current_relative_r6d_semantics(self.action_semantics)
        left_command, left_diag = self._solve_arm(
            "left", row[LEFT_POSE_SLICE], left, target_is_base_tcp=targets_are_base_tcp
        )
        right_command, right_diag = self._solve_arm(
            "right", row[RIGHT_POSE_SLICE], right, target_is_base_tcp=targets_are_base_tcp
        )
        self.last_diagnostics = {"left": left_diag, "right": right_diag}

        if targets_are_base_tcp:
            left_gripper_target = self._current_relative_gripper_map().umi_to_yam(
                row[LEFT_GRIPPER_INDEX], arm="left"
            )
            right_gripper_target = self._current_relative_gripper_map().umi_to_yam(
                row[RIGHT_GRIPPER_INDEX], arm="right"
            )
        else:
            left_gripper_target = self.gripper.umi_to_yam(row[LEFT_GRIPPER_INDEX], arm="left")
            right_gripper_target = self.gripper.umi_to_yam(row[RIGHT_GRIPPER_INDEX], arm="right")
        left_gripper_command = self.limits.clamp_gripper(left_gripper_target, left_gripper)
        right_gripper_command = self.limits.clamp_gripper(right_gripper_target, right_gripper)
        command = {key: float(value) for key, value in zip(LEFT_JOINT_KEYS, left_command, strict=True)}
        command[LEFT_GRIPPER_KEY] = left_gripper_command
        command.update(
            {key: float(value) for key, value in zip(RIGHT_JOINT_KEYS, right_command, strict=True)}
        )
        command[RIGHT_GRIPPER_KEY] = right_gripper_command
        return command

    def current_relative_gripper_targets(self, row: np.ndarray) -> dict[str, float]:
        """Return unclipped BiYAM endpoint targets for one resolved v1 row."""

        if not _is_current_relative_r6d_semantics(self.action_semantics):
            raise RuntimeError("current-relative gripper targets require current-relative R6D semantics")
        row = np.asarray(row, dtype=np.float64)
        if row.shape != (14,) or not np.isfinite(row).all():
            raise ValueError("resolved action row must be finite with shape (14,)")
        mapping = self._current_relative_gripper_map()
        return {
            "left": mapping.umi_to_yam(row[LEFT_GRIPPER_INDEX], arm="left"),
            "right": mapping.umi_to_yam(row[RIGHT_GRIPPER_INDEX], arm="right"),
        }

    def measured_target_residuals(
        self,
        observation: dict[str, float],
    ) -> dict[str, tuple[float, float]]:
        """Compare fresh measured FK with the targets from the last solved row.

        This is the execution-progress signal for current-relative waypoint
        retiming. It deliberately uses measured joints, never the requested or
        driver-acknowledged command.
        """

        if not _is_current_relative_r6d_semantics(self.action_semantics):
            raise RuntimeError("measured progress gating is defined for current-relative R6D only")
        if set(self.last_diagnostics) != {"left", "right"}:
            raise RuntimeError("solve an action row before checking measured target residuals")
        left, right, _, _ = self.joints_from_observation(observation)
        return {
            "left": self.left.residual(
                left,
                self.last_diagnostics["left"].requested_tcp,
                flange_to_target=YAM_FLANGE_TO_TCP,
            ),
            "right": self.right.residual(
                right,
                self.last_diagnostics["right"].requested_tcp,
                flange_to_target=YAM_FLANGE_TO_TCP,
            ),
        }

    def _current_relative_gripper_map(self) -> CurrentRelativeGripperMap:
        if self.current_relative_gripper is None:
            raise RuntimeError(
                "current-relative gripper conversion requires verified UMI closed/open tag-width endpoints"
            )
        return self.current_relative_gripper

    def _solve_arm(
        self,
        arm: str,
        policy_pose: np.ndarray,
        current_joints: np.ndarray,
        *,
        target_is_base_tcp: bool = False,
    ) -> tuple[np.ndarray, ArmCommandDiagnostics]:
        kinematics = self.left if arm == "left" else self.right
        calibration = self.frames.calibrations[arm]
        requested_tcp = (
            vec_to_pose(policy_pose) if target_is_base_tcp else self.frames.target_tcp(arm, policy_pose)
        )
        flange_to_tcp = YAM_FLANGE_TO_TCP if target_is_base_tcp else calibration.flange_to_tcp.transform
        orientation_weight = (
            kinematics.orientation_weight
            if self.tracking.mode == "strict"
            else self.tracking.diagnostic_orientation_weight
        )

        candidates = [requested_tcp]
        if self.tracking.roll_tolerance > 0:
            candidates = []
            for angle in np.linspace(
                -self.tracking.roll_tolerance,
                self.tracking.roll_tolerance,
                self.tracking.roll_samples,
            ):
                roll = np.eye(4)
                roll[:3, :3] = Rotation.from_rotvec((0.0, 0.0, angle)).as_matrix()
                candidates.append(requested_tcp @ roll)

        solved = [
            (
                target,
                kinematics.solve(
                    target,
                    current_joints,
                    flange_to_target=flange_to_tcp,
                    orientation_weight=orientation_weight,
                ),
            )
            for target in candidates
        ]
        converged = [candidate for candidate in solved if candidate[1].converged]
        target_tcp, result = min(
            converged or solved,
            key=lambda candidate: (
                not candidate[1].converged,
                candidate[1].max_joint_change,
                candidate[1].position_error,
                candidate[1].orientation_error,
            ),
        )
        if self.tracking.mode == "strict" and not result.converged:
            raise IkResidualError(
                f"[{arm}] IK residual {result.position_error * 1e3:.2f} mm / "
                f"{np.rad2deg(result.orientation_error):.2f} deg"
            )

        # Use the same per-arm operational bounds that constrained IK. This keeps
        # adapter-side rate limiting from silently reverting to model defaults
        # when a physical BiYAM config declares tighter or asymmetric ranges.
        commanded = self.limits.clamp_joints(
            result.joints,
            current_joints,
            joint_lower=kinematics.lower,
            joint_upper=kinematics.upper,
        )
        commanded_tcp = kinematics.fk(commanded, flange_to_tcp)
        position_error, orientation_error = kinematics.residual(
            commanded,
            target_tcp,
            flange_to_target=flange_to_tcp,
        )
        diagnostics = ArmCommandDiagnostics(
            ik=result,
            requested_tcp=target_tcp,
            commanded_tcp=commanded_tcp,
            commanded_position_error=position_error,
            commanded_orientation_error=orientation_error,
            rate_limited=not np.allclose(commanded, result.joints, atol=1e-12),
        )
        return commanded, diagnostics


@dataclass(frozen=True)
class UmiEpisode:
    """One raw UMI episode plus the exact 14-D pose-policy representation."""

    episode_index: int
    fps: float
    frame_index: np.ndarray
    timestamp: np.ndarray
    raw_state: np.ndarray
    raw_action: np.ndarray
    policy_state: np.ndarray
    policy_action: np.ndarray

    def __post_init__(self) -> None:
        frame_count = len(self.frame_index)
        expected = {
            "timestamp": (frame_count,),
            "raw_state": (frame_count, 12),
            "raw_action": (frame_count, 12),
            "policy_state": (frame_count, 14),
            "policy_action": (frame_count, 14),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
            object.__setattr__(self, name, value.copy())
        object.__setattr__(self, "frame_index", np.asarray(self.frame_index, dtype=np.int64).copy())
        if self.fps <= 0:
            raise ValueError("fps must be positive")


def _clean_gripper_width(width_mm: np.ndarray) -> np.ndarray:
    width_mm = np.asarray(width_mm, dtype=np.float64).copy()
    if width_mm.ndim != 1 or len(width_mm) == 0 or not np.isfinite(width_mm).all():
        raise ValueError("gripper width must be a finite, non-empty vector")
    invalid = width_mm > 126.0
    if invalid.any():
        valid_indices = np.flatnonzero(~invalid)
        if len(valid_indices) < 2:
            raise ValueError("not enough valid gripper samples to repair outliers")
        invalid_indices = np.flatnonzero(invalid)
        width_mm[invalid_indices] = np.interp(invalid_indices, valid_indices, width_mm[valid_indices])
    return np.clip(width_mm / 125.0, 0.0, 1.0)


def load_raw_umi_episode(
    dataset_root: str | Path,
    episode_index: int,
    *,
    verify_delta_law: bool = True,
) -> UmiEpisode:
    """Load one episode directly from the raw LeRobot v3 parquet files.

    This intentionally does not instantiate :class:`LeRobotDataset`, because that
    path decodes all videos when only scalar tracks are needed.  Episodes are
    selected by their ``episode_index`` column rather than by file name.
    """

    import json

    import pyarrow.parquet as parquet

    dataset_root = Path(dataset_root)
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"missing dataset metadata: {info_path}")
    info = json.loads(info_path.read_text())
    fps = float(info["fps"])
    columns = (
        "observation.state",
        "action",
        "observation.gripper_width.umi1",
        "observation.gripper_width.umi2",
        "timestamp",
        "frame_index",
        "episode_index",
    )
    chunks: list[dict[str, np.ndarray]] = []
    for data_file in sorted((dataset_root / "data").glob("**/*.parquet")):
        table = parquet.read_table(data_file, columns=list(columns))
        episodes = table["episode_index"].to_numpy()
        mask = episodes == episode_index
        if not mask.any():
            continue
        chunks.append(
            {
                "raw_state": np.stack(table["observation.state"].to_numpy(zero_copy_only=False))[mask].astype(
                    np.float32
                ),
                "raw_action": np.stack(table["action"].to_numpy(zero_copy_only=False))[mask].astype(
                    np.float32
                ),
                "gripper1": table["observation.gripper_width.umi1"].to_numpy()[mask].astype(np.float64),
                "gripper2": table["observation.gripper_width.umi2"].to_numpy()[mask].astype(np.float64),
                "timestamp": table["timestamp"].to_numpy()[mask].astype(np.float64),
                "frame_index": table["frame_index"].to_numpy()[mask].astype(np.int64),
            }
        )
    if not chunks:
        raise ValueError(f"episode {episode_index} was not found under {dataset_root}")

    values = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
    order = np.argsort(values["frame_index"], kind="stable")
    values = {key: value[order] for key, value in values.items()}
    frame_index = values["frame_index"]
    if not np.array_equal(frame_index, np.arange(len(frame_index), dtype=np.int64)):
        raise ValueError(f"episode {episode_index} frame_index is not contiguous from zero")

    raw_state = values["raw_state"]
    raw_action = values["raw_action"]
    if raw_state.shape[1:] != (12,) or raw_action.shape != raw_state.shape:
        raise ValueError("raw UMI state/action must both have shape (frames, 12)")
    if not np.isfinite(raw_state).all() or not np.isfinite(raw_action).all():
        raise ValueError("raw UMI state/action contains non-finite values")
    if verify_delta_law:
        if not np.array_equal(raw_action[:-1], raw_state[1:] - raw_state[:-1]):
            raise ValueError("raw UMI action[t] is not exactly state[t+1] - state[t]")
        if np.any(raw_action[-1]):
            raise ValueError("raw UMI terminal action must be all zeros")

    left_pose = reanchor_pose_track(raw_state[:, :6])
    right_pose = reanchor_pose_track(raw_state[:, 6:12])
    left_gripper = _clean_gripper_width(values["gripper1"])
    right_gripper = _clean_gripper_width(values["gripper2"])
    policy_state = np.concatenate(
        (
            left_pose,
            left_gripper[:, None],
            right_pose,
            right_gripper[:, None],
        ),
        axis=1,
    ).astype(np.float32)
    policy_action = np.empty_like(policy_state)
    policy_action[:-1] = policy_state[1:]
    policy_action[-1] = policy_state[-1]
    return UmiEpisode(
        episode_index=episode_index,
        fps=fps,
        frame_index=frame_index,
        timestamp=values["timestamp"],
        raw_state=raw_state,
        raw_action=raw_action,
        policy_state=policy_state,
        policy_action=policy_action,
    )


@dataclass(frozen=True)
class ArmTrajectory:
    """Desired and achieved quantities for one independently retargeted arm."""

    arm: str
    joints: np.ndarray
    requested_joints: np.ndarray
    gripper: np.ndarray
    target_tcp: np.ndarray
    achieved_tcp: np.ndarray
    ik_position_error: np.ndarray
    ik_orientation_error: np.ndarray
    position_error: np.ndarray
    orientation_error: np.ndarray
    ik_converged: np.ndarray
    rate_limited: np.ndarray
    restart_index: np.ndarray
    ik_solver: np.ndarray | None = None

    def __post_init__(self) -> None:
        frame_count = len(self.joints)
        expected = {
            "joints": (frame_count, 6),
            "requested_joints": (frame_count, 6),
            "gripper": (frame_count,),
            "target_tcp": (frame_count, 4, 4),
            "achieved_tcp": (frame_count, 4, 4),
            "ik_position_error": (frame_count,),
            "ik_orientation_error": (frame_count,),
            "position_error": (frame_count,),
            "orientation_error": (frame_count,),
            "ik_converged": (frame_count,),
            "rate_limited": (frame_count,),
            "restart_index": (frame_count,),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
            object.__setattr__(self, name, value.copy())
        solver = (
            np.full(frame_count, IK_SOLVER_PLACO, dtype="U32")
            if self.ik_solver is None
            else np.asarray(self.ik_solver, dtype="U32")
        )
        if solver.shape != (frame_count,):
            raise ValueError(f"ik_solver must have shape ({frame_count},), got {solver.shape}")
        known = {IK_SOLVER_PLACO, IK_SOLVER_BOUNDED_LEAST_SQUARES}
        if not set(solver.tolist()) <= known:
            raise ValueError(f"ik_solver contains unsupported values: {sorted(set(solver.tolist()) - known)}")
        object.__setattr__(self, "ik_solver", solver.copy())

    def summary(self) -> dict[str, float | int | str]:
        return {
            "arm": self.arm,
            "frames": len(self.joints),
            "ik_converged_fraction": float(np.mean(self.ik_converged)),
            "ik_position_error_mm_p50": float(np.percentile(self.ik_position_error, 50) * 1e3),
            "ik_position_error_mm_p90": float(np.percentile(self.ik_position_error, 90) * 1e3),
            "ik_position_error_mm_max": float(np.max(self.ik_position_error) * 1e3),
            "command_position_error_mm_p50": float(np.percentile(self.position_error, 50) * 1e3),
            "command_position_error_mm_p90": float(np.percentile(self.position_error, 90) * 1e3),
            "command_position_error_mm_max": float(np.max(self.position_error) * 1e3),
            "rate_limited_fraction": float(np.mean(self.rate_limited)),
            "max_joint_step_rad": float(np.abs(np.diff(self.joints, axis=0)).max(initial=0.0)),
            "random_restart_fraction": float(np.mean(self.restart_index > 0)),
            "bounded_least_squares_frames": int(np.sum(self.ik_solver == IK_SOLVER_BOUNDED_LEAST_SQUARES)),
            "bounded_least_squares_fraction": float(
                np.mean(self.ik_solver == IK_SOLVER_BOUNDED_LEAST_SQUARES)
            ),
        }


@dataclass(frozen=True)
class EventAwareSmoothingConfig:
    """Task-independent joint-path smoothing around gripper actuation.

    Gripper motion is used only as a contact-phase signal.  The optimizer knows
    nothing about bowls, objects, or a particular workspace: it preserves every
    solved joint waypoint during a meaningful open/close transition, smooths the
    transit waypoints, and automatically adds anchors if the smoothed TCP leaves
    the declared tube around the original FK path.
    """

    min_gripper_speed_per_s: float = 0.04
    min_gripper_excursion: float = 0.08
    bridge_gap_s: float = 0.12
    event_padding_s: float = 0.10
    tracking_weight: float = 1.0
    acceleration_weight: float = 20.0
    jerk_weight: float = 2.0
    max_position_deviation_m: float = 5e-3
    max_orientation_deviation_rad: float = np.deg2rad(5.0)
    max_refinement_iterations: int = 4

    def __post_init__(self) -> None:
        positive = {
            "min_gripper_speed_per_s": self.min_gripper_speed_per_s,
            "min_gripper_excursion": self.min_gripper_excursion,
            "tracking_weight": self.tracking_weight,
            "max_position_deviation_m": self.max_position_deviation_m,
            "max_orientation_deviation_rad": self.max_orientation_deviation_rad,
        }
        if any(not np.isfinite(value) or value <= 0 for value in positive.values()):
            raise ValueError(f"positive smoothing values required, got {positive}")
        non_negative = {
            "bridge_gap_s": self.bridge_gap_s,
            "event_padding_s": self.event_padding_s,
            "acceleration_weight": self.acceleration_weight,
            "jerk_weight": self.jerk_weight,
        }
        if any(not np.isfinite(value) or value < 0 for value in non_negative.values()):
            raise ValueError(f"non-negative smoothing values required, got {non_negative}")
        if self.max_refinement_iterations < 0:
            raise ValueError("max_refinement_iterations must be non-negative")


def detect_gripper_motion_events(
    gripper_signal: np.ndarray,
    *,
    fps: float,
    config: EventAwareSmoothingConfig | None = None,
) -> np.ndarray:
    """Return frames belonging to meaningful normalized open/close transitions.

    Small encoder noise is rejected by both a speed threshold and a minimum net
    excursion.  Short gaps are bridged before the net excursion check.  This is
    deliberately based on the source gripper signal rather than a clipped YAM
    command, because clipping a wider source aperture can erase real events.
    """

    config = config or EventAwareSmoothingConfig()
    signal = np.asarray(gripper_signal, dtype=np.float64)
    if signal.ndim != 1 or len(signal) == 0 or not np.isfinite(signal).all():
        raise ValueError("gripper_signal must be a finite, non-empty vector")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")
    if len(signal) == 1:
        return np.zeros(1, dtype=bool)

    active_edges = np.flatnonzero(np.abs(np.diff(signal)) * fps >= config.min_gripper_speed_per_s)
    event_mask = np.zeros(len(signal), dtype=bool)
    if len(active_edges) == 0:
        return event_mask

    max_gap = max(1, int(round(config.bridge_gap_s * fps)))
    edge_groups: list[tuple[int, int]] = []
    start = previous = int(active_edges[0])
    for edge in active_edges[1:]:
        edge = int(edge)
        if edge - previous > max_gap:
            edge_groups.append((start, previous + 1))
            start = edge
        previous = edge
    edge_groups.append((start, previous + 1))

    padding = int(round(config.event_padding_s * fps))
    for start, end in edge_groups:
        if abs(float(signal[end] - signal[start])) < config.min_gripper_excursion:
            continue
        padded_start = max(0, start - padding)
        padded_end = min(len(signal) - 1, end + padding)
        event_mask[padded_start : padded_end + 1] = True
    return event_mask


@dataclass(frozen=True)
class EventAwareSmoothingResult:
    """Smoothed path plus masks and deviations needed for an auditable replay."""

    trajectory: ArmTrajectory
    event_mask: np.ndarray
    anchor_mask: np.ndarray
    position_deviation: np.ndarray
    orientation_deviation: np.ndarray
    reference_joints: np.ndarray
    fps: float

    def __post_init__(self) -> None:
        frame_count = len(self.trajectory.joints)
        expected = {
            "event_mask": (frame_count,),
            "anchor_mask": (frame_count,),
            "position_deviation": (frame_count,),
            "orientation_deviation": (frame_count,),
            "reference_joints": (frame_count, 6),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
            object.__setattr__(self, name, value.copy())
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if np.any(self.event_mask & ~self.anchor_mask):
            raise ValueError("every gripper event frame must be a hard anchor")

    @staticmethod
    def _derivative_percentile(joints: np.ndarray, order: int, fps: float) -> float:
        if len(joints) <= order:
            return 0.0
        derivative = np.diff(joints, n=order, axis=0) * fps**order
        return float(np.percentile(np.linalg.norm(derivative, axis=1), 90))

    def summary(self) -> dict[str, float | int]:
        smoothed = self.trajectory.joints
        endpoint_mask = np.zeros(len(self.anchor_mask), dtype=bool)
        endpoint_mask[[0, -1]] = True
        return {
            "event_frames": int(np.sum(self.event_mask)),
            "adaptive_anchor_frames": int(np.sum(self.anchor_mask & ~self.event_mask & ~endpoint_mask)),
            "max_position_deviation_mm": float(np.max(self.position_deviation) * 1e3),
            "max_orientation_deviation_deg": float(np.rad2deg(np.max(self.orientation_deviation))),
            "reference_acceleration_norm_rad_s2_p90": self._derivative_percentile(
                self.reference_joints, 2, self.fps
            ),
            "smoothed_acceleration_norm_rad_s2_p90": self._derivative_percentile(smoothed, 2, self.fps),
            "reference_jerk_norm_rad_s3_p90": self._derivative_percentile(self.reference_joints, 3, self.fps),
            "smoothed_jerk_norm_rad_s3_p90": self._derivative_percentile(smoothed, 3, self.fps),
        }


def _smooth_joint_path_with_anchors(
    reference_joints: np.ndarray,
    anchor_mask: np.ndarray,
    config: EventAwareSmoothingConfig,
) -> np.ndarray:
    frame_count = len(reference_joints)
    if frame_count < 4 or np.all(anchor_mask):
        return reference_joints.copy()

    identity = sparse.eye(frame_count, format="csc")
    second_difference = sparse.diags(
        (np.ones(frame_count - 2), -2.0 * np.ones(frame_count - 2), np.ones(frame_count - 2)),
        (0, 1, 2),
        shape=(frame_count - 2, frame_count),
        format="csc",
    )
    third_difference = sparse.diags(
        (
            -np.ones(frame_count - 3),
            3.0 * np.ones(frame_count - 3),
            -3.0 * np.ones(frame_count - 3),
            np.ones(frame_count - 3),
        ),
        (0, 1, 2, 3),
        shape=(frame_count - 3, frame_count),
        format="csc",
    )
    hessian = (
        config.tracking_weight * identity
        + config.acceleration_weight * (second_difference.T @ second_difference)
        + config.jerk_weight * (third_difference.T @ third_difference)
    )

    anchors = np.flatnonzero(anchor_mask)
    free = np.flatnonzero(~anchor_mask)
    free_hessian = hessian[free][:, free]
    anchor_hessian = hessian[free][:, anchors]
    result = reference_joints.copy()
    for joint_index in range(reference_joints.shape[1]):
        right_hand_side = (
            config.tracking_weight * reference_joints[free, joint_index]
            - anchor_hessian @ reference_joints[anchors, joint_index]
        )
        result[free, joint_index] = spsolve(free_hessian, right_hand_side)
    return result


def _pose_track_deviation(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    position = np.linalg.norm(candidate[:, :3, 3] - reference[:, :3, 3], axis=1)
    orientation = np.asarray(
        [
            np.linalg.norm(
                Rotation.from_matrix(reference[index, :3, :3].T @ candidate[index, :3, :3]).as_rotvec()
            )
            for index in range(len(reference))
        ]
    )
    return position, orientation


def smooth_arm_trajectory_between_gripper_events(
    trajectory: ArmTrajectory,
    kinematics: YamArmKinematics,
    flange_to_tcp: np.ndarray,
    *,
    source_gripper_signal: np.ndarray,
    fps: float,
    config: EventAwareSmoothingConfig | None = None,
) -> EventAwareSmoothingResult:
    """Smooth transit joint motion while exactly retaining gripper-event IK.

    The FK tube is measured against the original solved path, not against a task
    object or simulator.  Violating frames become additional hard anchors and the
    convex joint-space problem is resolved, so path freedom is bounded explicitly.
    """

    config = config or EventAwareSmoothingConfig()
    flange_to_tcp = _as_transform(flange_to_tcp, name="flange_to_tcp")
    source_signal = np.asarray(source_gripper_signal, dtype=np.float64)
    if source_signal.shape != trajectory.gripper.shape:
        raise ValueError(
            "source_gripper_signal must match the trajectory frame count, got "
            f"{source_signal.shape} and {trajectory.gripper.shape}"
        )
    event_mask = detect_gripper_motion_events(source_signal, fps=fps, config=config)
    anchor_mask = event_mask.copy()
    anchor_mask[[0, -1]] = True
    reference_joints = trajectory.joints.copy()
    reference_tcp = trajectory.achieved_tcp.copy()

    smoothed_joints = reference_joints.copy()
    smoothed_tcp = reference_tcp.copy()
    position_deviation = np.zeros(len(reference_joints))
    orientation_deviation = np.zeros(len(reference_joints))
    for _ in range(config.max_refinement_iterations + 1):
        smoothed_joints = _smooth_joint_path_with_anchors(reference_joints, anchor_mask, config)
        smoothed_joints = np.clip(smoothed_joints, kinematics.lower, kinematics.upper)
        smoothed_tcp = np.stack([kinematics.fk(joints, flange_to_tcp) for joints in smoothed_joints])
        position_deviation, orientation_deviation = _pose_track_deviation(reference_tcp, smoothed_tcp)
        violations = (
            (position_deviation > config.max_position_deviation_m)
            | (orientation_deviation > config.max_orientation_deviation_rad)
        ) & ~anchor_mask
        if not np.any(violations):
            break
        anchor_mask |= violations

    # A finite refinement budget must never weaken the declared geometric tube.
    # If anchoring one violating region creates a new neighboring violation on
    # the final pass, restore only those samples to the already-validated IK path.
    remaining_violations = (position_deviation > config.max_position_deviation_m) | (
        orientation_deviation > config.max_orientation_deviation_rad
    )
    if np.any(remaining_violations):
        anchor_mask |= remaining_violations
        smoothed_joints[remaining_violations] = reference_joints[remaining_violations]
        smoothed_tcp[remaining_violations] = reference_tcp[remaining_violations]
        position_deviation, orientation_deviation = _pose_track_deviation(reference_tcp, smoothed_tcp)

    target_position_error = np.linalg.norm(smoothed_tcp[:, :3, 3] - trajectory.target_tcp[:, :3, 3], axis=1)
    target_orientation_error = np.asarray(
        [
            np.linalg.norm(
                Rotation.from_matrix(
                    smoothed_tcp[index, :3, :3].T @ trajectory.target_tcp[index, :3, :3]
                ).as_rotvec()
            )
            for index in range(len(smoothed_tcp))
        ]
    )
    smoothed_trajectory = ArmTrajectory(
        arm=trajectory.arm,
        joints=smoothed_joints,
        requested_joints=trajectory.requested_joints,
        gripper=trajectory.gripper,
        target_tcp=trajectory.target_tcp,
        achieved_tcp=smoothed_tcp,
        ik_position_error=trajectory.ik_position_error,
        ik_orientation_error=trajectory.ik_orientation_error,
        position_error=target_position_error,
        orientation_error=target_orientation_error,
        ik_converged=trajectory.ik_converged,
        rate_limited=trajectory.rate_limited,
        restart_index=trajectory.restart_index,
        ik_solver=trajectory.ik_solver,
    )
    return EventAwareSmoothingResult(
        trajectory=smoothed_trajectory,
        event_mask=event_mask,
        anchor_mask=anchor_mask,
        position_deviation=position_deviation,
        orientation_deviation=orientation_deviation,
        reference_joints=reference_joints,
        fps=fps,
    )


@dataclass(frozen=True)
class RetimedArmTrajectory:
    """A joint-continuous playback path with source-frame correspondence."""

    arm: str
    fps: float
    source_frame: np.ndarray
    joints: np.ndarray
    gripper: np.ndarray
    target_tcp: np.ndarray
    achieved_tcp: np.ndarray

    def __post_init__(self) -> None:
        frame_count = len(self.source_frame)
        expected = {
            "source_frame": (frame_count,),
            "joints": (frame_count, 6),
            "gripper": (frame_count,),
            "target_tcp": (frame_count, 4, 4),
            "achieved_tcp": (frame_count, 4, 4),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
            object.__setattr__(self, name, value.copy())
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if np.any(np.diff(self.source_frame) < 0):
            raise ValueError("source_frame must be monotonic")

    def summary(self) -> dict[str, float | int | str]:
        position_error = np.linalg.norm(self.target_tcp[:, :3, 3] - self.achieved_tcp[:, :3, 3], axis=1)
        return {
            "arm": self.arm,
            "frames": len(self.joints),
            "duration_s": len(self.joints) / self.fps,
            "max_joint_step_rad": float(np.abs(np.diff(self.joints, axis=0)).max(initial=0.0)),
            "max_gripper_step": float(np.abs(np.diff(self.gripper)).max(initial=0.0)),
            "position_error_mm_p90": float(np.percentile(position_error, 90) * 1e3),
            "position_error_mm_max": float(np.max(position_error) * 1e3),
        }


def _interpolate_pose(start: np.ndarray, end: np.ndarray, fraction: float) -> np.ndarray:
    start = _as_transform(start, name="start pose")
    end = _as_transform(end, name="end pose")
    fraction = float(fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("interpolation fraction must be in [0, 1]")
    interpolated = np.eye(4)
    interpolated[:3, 3] = (1.0 - fraction) * start[:3, 3] + fraction * end[:3, 3]
    relative_rotation = Rotation.from_matrix(start[:3, :3].T @ end[:3, :3]).as_rotvec()
    interpolated[:3, :3] = start[:3, :3] @ Rotation.from_rotvec(fraction * relative_rotation).as_matrix()
    return interpolated


def time_parameterize_arm_trajectory(
    trajectory: ArmTrajectory,
    kinematics: YamArmKinematics,
    flange_to_tcp: np.ndarray,
    *,
    source_fps: float,
    output_fps: float = 30.0,
    max_joint_velocity: float = 0.6,
    max_gripper_velocity: float = 1.5,
) -> RetimedArmTrajectory:
    """Slow a solved path by inserting samples instead of clipping its geometry.

    Every original IK waypoint is retained.  A segment gets enough linear joint-space
    samples to satisfy declared velocities at ``output_fps``.  This avoids the common
    failure mode where a rate-limited command falls behind while the source trajectory
    continues advancing.
    """

    if source_fps <= 0 or output_fps <= 0:
        raise ValueError("source_fps and output_fps must be positive")
    if max_joint_velocity <= 0 or max_gripper_velocity <= 0:
        raise ValueError("velocity limits must be positive")
    flange_to_tcp = _as_transform(flange_to_tcp, name="flange_to_tcp")

    source_frames = [0.0]
    joints = [trajectory.joints[0].copy()]
    gripper = [float(trajectory.gripper[0])]
    targets = [trajectory.target_tcp[0].copy()]
    for index in range(1, len(trajectory.joints)):
        joint_distance = float(
            np.abs(trajectory.joints[index] - trajectory.joints[index - 1]).max(initial=0.0)
        )
        gripper_distance = abs(float(trajectory.gripper[index] - trajectory.gripper[index - 1]))
        duration = max(
            1.0 / source_fps,
            joint_distance / max_joint_velocity,
            gripper_distance / max_gripper_velocity,
        )
        intervals = max(1, int(np.ceil(duration * output_fps)))
        for interval in range(1, intervals + 1):
            fraction = interval / intervals
            source_frames.append((index - 1) + fraction)
            joints.append(
                (1.0 - fraction) * trajectory.joints[index - 1] + fraction * trajectory.joints[index]
            )
            gripper.append(
                (1.0 - fraction) * trajectory.gripper[index - 1] + fraction * trajectory.gripper[index]
            )
            targets.append(
                _interpolate_pose(trajectory.target_tcp[index - 1], trajectory.target_tcp[index], fraction)
            )

    joint_array = np.stack(joints)
    achieved = np.stack([kinematics.fk(sample, flange_to_tcp) for sample in joint_array])
    return RetimedArmTrajectory(
        arm=trajectory.arm,
        fps=output_fps,
        source_frame=np.asarray(source_frames),
        joints=joint_array,
        gripper=np.asarray(gripper),
        target_tcp=np.stack(targets),
        achieved_tcp=achieved,
    )


def retarget_arm_trajectory(
    episode: UmiEpisode,
    arm: Literal["left", "right"],
    kinematics: YamArmKinematics,
    frames: EpisodeFrames,
    *,
    initial_joints: np.ndarray | None = None,
    initial_gripper: float = 1.0,
    gripper_map: GripperMap | None = None,
    safety_limits: SafetyLimits | None = None,
    apply_rate_limits: bool = False,
    orientation_weight: float = 1.0,
) -> ArmTrajectory:
    """Retarget one arm independently and retain both requested and achieved TCPs.

    ``apply_rate_limits`` exists only for online one-tick command diagnostics.  Offline
    replay should leave it false and call :func:`time_parameterize_arm_trajectory`,
    which preserves every waypoint instead of advancing while the arm lags.
    """

    if arm not in {"left", "right"}:
        raise ValueError(f"unknown arm {arm!r}")
    gripper_map = gripper_map or GripperMap()
    safety_limits = safety_limits or SafetyLimits()
    joints = (
        np.zeros(6, dtype=np.float64)
        if initial_joints is None
        else np.asarray(initial_joints, dtype=np.float64).copy()
    )
    if joints.shape != (6,):
        raise ValueError("initial_joints must have shape (6,)")
    current_gripper = float(initial_gripper)
    frames.reset()
    frames.capture(arm, kinematics.fk(joints))
    calibration = frames.calibrations[arm]
    pose_slice = LEFT_POSE_SLICE if arm == "left" else RIGHT_POSE_SLICE
    gripper_index = LEFT_GRIPPER_INDEX if arm == "left" else RIGHT_GRIPPER_INDEX

    requested_joints = []
    commanded_joints = []
    grippers = []
    targets = []
    achieved = []
    ik_position_errors = []
    ik_orientation_errors = []
    command_position_errors = []
    command_orientation_errors = []
    converged = []
    rate_limited = []
    restart_indices = []
    ik_solvers = []

    for row in episode.policy_state:
        target_tcp = frames.target_tcp(arm, row[pose_slice])
        result = kinematics.solve(
            target_tcp,
            joints,
            flange_to_target=calibration.flange_to_tcp.transform,
            orientation_weight=orientation_weight,
        )
        command = (
            safety_limits.clamp_joints(
                result.joints,
                joints,
                joint_lower=kinematics.lower,
                joint_upper=kinematics.upper,
            )
            if apply_rate_limits
            else result.joints.copy()
        )
        target_gripper = gripper_map.umi_to_yam(row[gripper_index], arm=arm)
        current_gripper = (
            safety_limits.clamp_gripper(target_gripper, current_gripper)
            if apply_rate_limits
            else target_gripper
        )
        achieved_tcp = kinematics.fk(command, calibration.flange_to_tcp.transform)
        command_position_error, command_orientation_error = kinematics.residual(
            command,
            target_tcp,
            flange_to_target=calibration.flange_to_tcp.transform,
        )

        requested_joints.append(result.joints)
        commanded_joints.append(command)
        grippers.append(current_gripper)
        targets.append(target_tcp)
        achieved.append(achieved_tcp)
        ik_position_errors.append(result.position_error)
        ik_orientation_errors.append(result.orientation_error)
        command_position_errors.append(command_position_error)
        command_orientation_errors.append(command_orientation_error)
        converged.append(result.converged)
        rate_limited.append(not np.allclose(command, result.joints, atol=1e-12))
        restart_indices.append(result.restart_index)
        ik_solvers.append(result.solver)
        joints = command

    return ArmTrajectory(
        arm=arm,
        joints=np.stack(commanded_joints),
        requested_joints=np.stack(requested_joints),
        gripper=np.asarray(grippers),
        target_tcp=np.stack(targets),
        achieved_tcp=np.stack(achieved),
        ik_position_error=np.asarray(ik_position_errors),
        ik_orientation_error=np.asarray(ik_orientation_errors),
        position_error=np.asarray(command_position_errors),
        orientation_error=np.asarray(command_orientation_errors),
        ik_converged=np.asarray(converged, dtype=bool),
        rate_limited=np.asarray(rate_limited, dtype=bool),
        restart_index=np.asarray(restart_indices, dtype=np.int64),
        ik_solver=np.asarray(ik_solvers),
    )
