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

"""Measured dual-YAM rig geometry and scene-independent collision checks.

Independent UMI-to-YAM IK does not need a shared world frame. Bimanual
collision checking does: both trusted YAM models must be placed in one world,
and that placement must come from the physical rig rather than a task video.

This module deliberately models only:

* the two YAMs from i2rt's authoritative MuJoCo geometry;
* the measured pose of the right base in the left-base frame; and
* the measured tabletop plane.

It never includes task objects. ``RigCollisionChecker`` can inspect unverified
calibrations for diagnostics, but its callable hardware-dispatch interface
refuses them. The checker evaluates sampled command endpoints only; it is not a
continuous swept-path collision proof.
"""

from __future__ import annotations

import hashlib
import json
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from lerobot.robots.bi_yam.config_bi_yam import YAM_JOINT_LIMITS, YAM_SCALAR_KEYS
from lerobot.simulators.bi_yam.config import STATE_SIZE, BiYAMSimulatorConfig

_SCHEMA_VERSION = 1
_LEFT_BASE = "left_base"
_RIGHT_BASE = "right_base"
_TABLE_GEOM = "calibrated_table_plane"


def _driver_action_bounds() -> tuple[np.ndarray, np.ndarray]:
    """Return bounds from the same authoritative source as the hardware driver."""

    arm_lower = tuple(limit[0] for limit in YAM_JOINT_LIMITS) + (0.0,)
    arm_upper = tuple(limit[1] for limit in YAM_JOINT_LIMITS) + (1.0,)
    return np.asarray(arm_lower + arm_lower), np.asarray(arm_upper + arm_upper)


class RigCalibrationError(ValueError):
    """The supplied rig calibration is malformed or has ambiguous semantics."""


class UnverifiedRigCalibrationError(RuntimeError):
    """A hardware check was requested with unverified physical measurements."""


class UnsafeRigCommandError(RuntimeError):
    """A proposed hardware command violates the calibrated collision model."""

    def __init__(self, report: CollisionReport) -> None:
        self.report = report
        kinds = sorted({contact.kind for contact in report.contacts})
        super().__init__(f"Unsafe dual-YAM command: {len(report.contacts)} violation(s): {kinds}")


@dataclass(frozen=True)
class CalibrationProvenance:
    """Human-auditable evidence attached to one physical measurement."""

    verified: bool
    source: str
    measurement_id: str | None = None
    evidence_path: str | None = None
    evidence_sha256: str | None = None
    evidence_size_bytes: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.verified, bool):
            raise RigCalibrationError("provenance.verified must be a boolean")
        if not isinstance(self.source, str) or not self.source.strip():
            raise RigCalibrationError("provenance.source must be a non-empty description or URI")
        if self.measurement_id is not None and (
            not isinstance(self.measurement_id, str) or not self.measurement_id.strip()
        ):
            raise RigCalibrationError("provenance.measurement_id must be a non-empty string when set")
        evidence_values = (self.evidence_path, self.evidence_sha256, self.evidence_size_bytes)
        if any(value is not None for value in evidence_values) and not all(
            value is not None for value in evidence_values
        ):
            raise RigCalibrationError(
                "provenance evidence_path, evidence_sha256, and evidence_size_bytes must be set together"
            )
        if self.evidence_path is not None and (
            not isinstance(self.evidence_path, str) or not self.evidence_path.strip()
        ):
            raise RigCalibrationError("provenance.evidence_path must be a non-empty string")
        if self.evidence_sha256 is not None and (
            not isinstance(self.evidence_sha256, str)
            or len(self.evidence_sha256) != 64
            or self.evidence_sha256 != self.evidence_sha256.lower()
            or any(character not in "0123456789abcdef" for character in self.evidence_sha256)
        ):
            raise RigCalibrationError("provenance.evidence_sha256 must be a lowercase SHA-256 digest")
        if self.evidence_size_bytes is not None and (
            isinstance(self.evidence_size_bytes, bool)
            or not isinstance(self.evidence_size_bytes, int)
            or self.evidence_size_bytes <= 0
        ):
            raise RigCalibrationError("provenance.evidence_size_bytes must be a positive integer")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CalibrationProvenance:
        return cls(
            verified=value.get("verified"),
            source=value.get("source"),
            measurement_id=value.get("measurement_id"),
            evidence_path=value.get("evidence_path"),
            evidence_sha256=value.get("evidence_sha256"),
            evidence_size_bytes=value.get("evidence_size_bytes"),
        )

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"verified": self.verified, "source": self.source}
        if self.measurement_id is not None:
            value["measurement_id"] = self.measurement_id
        if self.evidence_path is not None:
            value["evidence_path"] = self.evidence_path
            value["evidence_sha256"] = self.evidence_sha256
            value["evidence_size_bytes"] = self.evidence_size_bytes
        return value

    def require_verified_physical_evidence(self, *, label: str) -> dict[str, Any]:
        """Require one immutable local measurement artifact for v4 hardware use."""

        if not self.verified:
            raise UnverifiedRigCalibrationError(f"{label} provenance is not verified")

        def require_nonplaceholder(value: object, *, field: str) -> str:
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise UnverifiedRigCalibrationError(f"{label} {field} must be non-empty and trimmed")
            sentinels = {
                "replace",
                "placeholder",
                "unknown",
                "unset",
                "todo",
                "na",
                "tbd",
                "dummy",
                "none",
                "null",
                "synthetic",
                "test",
                "example",
            }
            folded = value.casefold()
            normalized = re.sub(r"[^a-z0-9]+", "", folded)
            tokens = set(re.findall(r"[a-z0-9]+", folded))
            embedded_sentinels = sentinels - {"na"}
            if (
                normalized in sentinels
                or tokens & sentinels
                or any(sentinel in normalized for sentinel in embedded_sentinels)
            ):
                raise UnverifiedRigCalibrationError(f"{label} {field} is a placeholder")
            return value

        source = require_nonplaceholder(self.source, field="source")
        measurement_id = require_nonplaceholder(self.measurement_id, field="measurement_id")
        if self.evidence_path is None or self.evidence_sha256 is None or self.evidence_size_bytes is None:
            raise UnverifiedRigCalibrationError(f"{label} lacks immutable physical evidence")
        path = Path(self.evidence_path)
        if not path.is_absolute() or path.as_posix() != self.evidence_path:
            raise UnverifiedRigCalibrationError(f"{label} evidence path must be absolute and normalized")
        if path.is_symlink() or not path.is_file():
            raise UnverifiedRigCalibrationError(f"{label} evidence must be a regular file, not a symlink")
        if stat.S_IMODE(path.stat().st_mode) & 0o222:
            raise UnverifiedRigCalibrationError(f"{label} evidence must be read-only")
        resolved = path.resolve(strict=True)
        if resolved != path:
            raise UnverifiedRigCalibrationError(f"{label} evidence path must not contain symlink components")
        if resolved.stat().st_size != self.evidence_size_bytes:
            raise UnverifiedRigCalibrationError(f"{label} evidence byte size changed")
        digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
        sentinel_payloads = (
            "",
            "n/a",
            "na",
            "tbd",
            "dummy",
            "none",
            "null",
            "synthetic",
            "test",
            "example",
            "placeholder",
        )
        sentinel_digests = {
            hashlib.sha256(variant.encode("utf-8")).hexdigest()
            for value in sentinel_payloads
            for variant in (value, value.upper(), value.title())
        }
        repeated = any(
            64 % period == 0 and self.evidence_sha256 == self.evidence_sha256[:period] * (64 // period)
            for period in range(1, 33)
        )
        if repeated or self.evidence_sha256 in sentinel_digests:
            raise UnverifiedRigCalibrationError(f"{label} evidence SHA-256 is a sentinel")
        if digest != self.evidence_sha256:
            raise UnverifiedRigCalibrationError(f"{label} evidence SHA-256 changed")
        return {
            "source": source,
            "measurement_id": measurement_id,
            "evidence_path": str(resolved),
            "evidence_size_bytes": self.evidence_size_bytes,
            "evidence_sha256": digest,
        }


@dataclass(frozen=True, eq=False)
class ProvenanceRigidTransform:
    """A measured ``parent_from_child`` rigid transform.

    ``matrix @ point_in_child`` produces ``point_in_parent``. For the dual-YAM
    rig schema the only accepted semantics are ``left_base_from_right_base``:
    the translation is the right-base origin expressed in left-base axes.
    """

    parent_frame: str
    child_frame: str
    matrix: np.ndarray
    provenance: CalibrationProvenance

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise RigCalibrationError("rig transform matrix must have shape (4, 4)")
        if not np.isfinite(matrix).all():
            raise RigCalibrationError("rig transform matrix must be finite")
        if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
            raise RigCalibrationError("rig transform bottom row must be [0, 0, 0, 1]")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise RigCalibrationError("rig transform rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
            raise RigCalibrationError("rig transform rotation must have determinant +1")
        if not self.parent_frame or not self.child_frame or self.parent_frame == self.child_frame:
            raise RigCalibrationError("rig transform must name two distinct frames")
        matrix = matrix.copy()
        matrix.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ProvenanceRigidTransform:
        has_matrix = "matrix" in value
        has_pose = "translation_m" in value or "quaternion_xyzw" in value
        if has_matrix == has_pose:
            raise RigCalibrationError(
                "rig transform must provide exactly one of matrix or translation_m + quaternion_xyzw"
            )
        if has_matrix:
            matrix = np.asarray(value["matrix"], dtype=np.float64)
        else:
            if "translation_m" not in value or "quaternion_xyzw" not in value:
                raise RigCalibrationError("translation_m and quaternion_xyzw must be provided together")
            matrix = _matrix_from_translation_quaternion(
                value["translation_m"],
                value["quaternion_xyzw"],
            )
        provenance = value.get("provenance")
        if not isinstance(provenance, Mapping):
            raise RigCalibrationError("rig transform requires a provenance object")
        return cls(
            parent_frame=value.get("parent_frame"),
            child_frame=value.get("child_frame"),
            matrix=matrix,
            provenance=CalibrationProvenance.from_dict(provenance),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_frame": self.parent_frame,
            "child_frame": self.child_frame,
            "matrix": self.matrix.tolist(),
            "provenance": self.provenance.to_dict(),
        }


@dataclass(frozen=True)
class TablePlaneCalibration:
    """Measured tabletop in one frame, with its normal pointing into free space."""

    frame: str
    point_m: tuple[float, float, float]
    normal_toward_workspace: tuple[float, float, float]
    provenance: CalibrationProvenance

    def __post_init__(self) -> None:
        point = np.asarray(self.point_m, dtype=np.float64)
        normal = np.asarray(self.normal_toward_workspace, dtype=np.float64)
        if point.shape != (3,) or normal.shape != (3,):
            raise RigCalibrationError("table point and normal must each have length 3")
        if not np.isfinite(point).all() or not np.isfinite(normal).all():
            raise RigCalibrationError("table point and normal must be finite")
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-12:
            raise RigCalibrationError("table normal must be nonzero")
        object.__setattr__(self, "point_m", tuple(float(value) for value in point))
        object.__setattr__(
            self,
            "normal_toward_workspace",
            tuple(float(value) for value in normal / norm),
        )
        if not isinstance(self.frame, str) or not self.frame:
            raise RigCalibrationError("table plane frame must be a non-empty string")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TablePlaneCalibration:
        provenance = value.get("provenance")
        if not isinstance(provenance, Mapping):
            raise RigCalibrationError("table plane requires a provenance object")
        return cls(
            frame=value.get("frame"),
            point_m=value.get("point_m"),
            normal_toward_workspace=value.get("normal_toward_workspace"),
            provenance=CalibrationProvenance.from_dict(provenance),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "point_m": list(self.point_m),
            "normal_toward_workspace": list(self.normal_toward_workspace),
            "provenance": self.provenance.to_dict(),
        }


@dataclass(frozen=True)
class DualYAMRigCalibration:
    """All non-URDF measurements required for shared-world safety checks."""

    left_base_from_right_base: ProvenanceRigidTransform
    table_plane_in_left_base: TablePlaneCalibration

    def __post_init__(self) -> None:
        transform = self.left_base_from_right_base
        if transform.parent_frame != _LEFT_BASE or transform.child_frame != _RIGHT_BASE:
            raise RigCalibrationError(
                "left_base_from_right_base must be the pose of right_base in left_base: "
                "parent_frame='left_base', child_frame='right_base'"
            )
        if self.table_plane_in_left_base.frame != _LEFT_BASE:
            raise RigCalibrationError("table_plane_in_left_base.frame must be 'left_base'")

    @property
    def verified(self) -> bool:
        return (
            self.left_base_from_right_base.provenance.verified
            and self.table_plane_in_left_base.provenance.verified
        )

    def require_verified(self) -> None:
        unverified: list[str] = []
        if not self.left_base_from_right_base.provenance.verified:
            unverified.append("left_base_from_right_base")
        if not self.table_plane_in_left_base.provenance.verified:
            unverified.append("table_plane_in_left_base")
        if unverified:
            raise UnverifiedRigCalibrationError(
                "Hardware collision checks require verified measurements: " + ", ".join(unverified)
            )

    def require_verified_physical_evidence(self) -> dict[str, Any]:
        """Require immutable physical support for both rig measurements."""

        self.require_verified()
        return {
            "left_base_from_right_base": (
                self.left_base_from_right_base.provenance.require_verified_physical_evidence(
                    label="left_base_from_right_base"
                )
            ),
            "table_plane_in_left_base": (
                self.table_plane_in_left_base.provenance.require_verified_physical_evidence(
                    label="table_plane_in_left_base"
                )
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DualYAMRigCalibration:
        if value.get("schema_version") != _SCHEMA_VERSION:
            raise RigCalibrationError(f"schema_version must be {_SCHEMA_VERSION}")
        transform = value.get("left_base_from_right_base")
        table = value.get("table_plane_in_left_base")
        if not isinstance(transform, Mapping) or not isinstance(table, Mapping):
            raise RigCalibrationError(
                "calibration requires left_base_from_right_base and table_plane_in_left_base objects"
            )
        return cls(
            left_base_from_right_base=ProvenanceRigidTransform.from_dict(transform),
            table_plane_in_left_base=TablePlaneCalibration.from_dict(table),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> DualYAMRigCalibration:
        with Path(path).open(encoding="utf-8") as stream:
            value = json.load(stream)
        if not isinstance(value, Mapping):
            raise RigCalibrationError("rig calibration JSON must contain an object")
        return cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "left_base_from_right_base": self.left_base_from_right_base.to_dict(),
            "table_plane_in_left_base": self.table_plane_in_left_base.to_dict(),
        }


CollisionKind = Literal["inter_arm", "arm_table", "self_collision"]


@dataclass(frozen=True)
class CollisionContact:
    """One geometric violation at a proposed joint configuration."""

    kind: CollisionKind
    body1: str
    body2: str
    geom1: str
    geom2: str
    signed_distance_m: float

    @property
    def penetrating(self) -> bool:
        return self.signed_distance_m < 0.0


@dataclass(frozen=True)
class CollisionReport:
    """Collision result for one command in left-then-right 14D ordering."""

    command: tuple[float, ...]
    minimum_clearance_m: float
    calibration_verified: bool
    contacts: tuple[CollisionContact, ...]

    @property
    def safe(self) -> bool:
        return not self.contacts


class RigCollisionChecker:
    """Kinematic collision gate backed by the trusted i2rt YAM meshes.

    The 14D command layout is ``left[q1..q6, gripper],
    right[q1..q6, gripper]`` with radians and normalized grippers. The class is
    callable so it can be inserted immediately before a hardware dispatch:

    ``checker(command)`` returns a safe report or raises. Use
    :meth:`inspect_command` only for explicitly non-hardware diagnostics.
    """

    def __init__(
        self,
        calibration: DualYAMRigCalibration,
        *,
        minimum_clearance_m: float,
    ) -> None:
        if not np.isfinite(minimum_clearance_m) or minimum_clearance_m < 0:
            raise ValueError("minimum_clearance_m must be finite and non-negative")

        from lerobot.simulators.bi_yam.simulator import mujoco_backend_available

        if not mujoco_backend_available():
            raise ModuleNotFoundError("RigCollisionChecker requires both 'mujoco' and 'i2rt'")

        import mujoco

        from lerobot.simulators.bi_yam.mujoco_backend import MujocoBiYAMBackend

        self.calibration = calibration
        self.minimum_clearance_m = float(minimum_clearance_m)
        self._mujoco = mujoco

        right_pose = calibration.left_base_from_right_base.matrix
        config = BiYAMSimulatorConfig(
            left_base_position=(0.0, 0.0, 0.0),
            left_base_quaternion=(1.0, 0.0, 0.0, 0.0),
            right_base_position=tuple(float(value) for value in right_pose[:3, 3]),
            right_base_quaternion=_rotation_matrix_to_quaternion_wxyz(right_pose[:3, :3]),
            include_workspace_objects=False,
        )
        builder = MujocoBiYAMBackend(config)
        table = calibration.table_plane_in_left_base
        xml = builder.build_collision_world_xml(
            table_point_in_left_base=table.point_m,
            table_normal_toward_workspace=table.normal_toward_workspace,
            minimum_clearance_m=self.minimum_clearance_m,
        )
        self._model = mujoco.MjModel.from_xml_string(xml)
        self._data = mujoco.MjData(self._model)
        self._joint_qpos_addresses = np.empty((2, 8), dtype=np.int32)
        self._joint_ranges = np.empty((2, 8, 2), dtype=np.float64)
        self._resolve_joint_addresses()

    def inspect_command(
        self,
        command: Mapping[str, float] | Sequence[float] | np.ndarray,
    ) -> CollisionReport:
        """Inspect geometry without authorizing an unverified hardware rig."""

        validated = self._validate_command(command)
        self._write_command(validated)
        self._mujoco.mj_forward(self._model, self._data)
        contacts: list[CollisionContact] = []
        for index in range(self._data.ncon):
            contact = self._data.contact[index]
            if float(contact.dist) >= self.minimum_clearance_m:
                continue
            classified = self._classify_contact(contact)
            if classified is not None:
                contacts.append(classified)
        contacts.sort(
            key=lambda item: (
                item.kind,
                item.body1,
                item.body2,
                item.geom1,
                item.geom2,
                item.signed_distance_m,
            )
        )
        return CollisionReport(
            command=tuple(float(value) for value in validated),
            minimum_clearance_m=self.minimum_clearance_m,
            calibration_verified=self.calibration.verified,
            contacts=tuple(contacts),
        )

    def validate_hardware_command(
        self,
        command: Mapping[str, float] | Sequence[float] | np.ndarray,
    ) -> CollisionReport:
        """Gate one endpoint for dispatch, raising on calibration or geometry failure.

        This checks the supplied configuration only, not the swept path from the
        previous command.
        """

        self.calibration.require_verified_physical_evidence()
        report = self.inspect_command(command)
        if not report.safe:
            raise UnsafeRigCommandError(report)
        return report

    def assert_hardware_ready(self) -> None:
        """Fail cheaply unless both rig measurements have immutable physical evidence."""

        self.calibration.require_verified_physical_evidence()

    def validate_hardware_trajectory(
        self,
        commands: Sequence[Sequence[float]] | np.ndarray,
    ) -> tuple[CollisionReport, ...]:
        """Gate every already-sampled waypoint before dispatching a trajectory."""

        self.calibration.require_verified_physical_evidence()
        values = np.asarray(commands, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != STATE_SIZE:
            raise ValueError(f"commands must have shape (N, {STATE_SIZE}), got {values.shape}")
        return tuple(self.validate_hardware_command(command) for command in values)

    def __call__(
        self,
        command: Mapping[str, float] | Sequence[float] | np.ndarray,
    ) -> CollisionReport:
        return self.validate_hardware_command(command)

    def _resolve_joint_addresses(self) -> None:
        for side_index, side in enumerate(
            (_LEFT_BASE.removesuffix("_base"), _RIGHT_BASE.removesuffix("_base"))
        ):
            for local_index in range(8):
                name = f"{side}_joint{local_index + 1}"
                joint_id = self._mujoco.mj_name2id(
                    self._model,
                    self._mujoco.mjtObj.mjOBJ_JOINT,
                    name,
                )
                if joint_id < 0:
                    raise RuntimeError(f"Trusted i2rt model is missing joint {name!r}")
                self._joint_qpos_addresses[side_index, local_index] = self._model.jnt_qposadr[joint_id]
                self._joint_ranges[side_index, local_index] = self._model.jnt_range[joint_id]

    def _write_command(self, command: np.ndarray) -> None:
        self._mujoco.mj_resetData(self._model, self._data)
        for side_index, command_offset in enumerate((0, 7)):
            self._data.qpos[self._joint_qpos_addresses[side_index, :6]] = command[
                command_offset : command_offset + 6
            ]
            gripper = float(command[command_offset + 6])
            for local_index in (6, 7):
                lower, upper = self._joint_ranges[side_index, local_index]
                # BiYAM commands 0=closed and 1=open. The pinned linear_4310
                # model has qpos=upper at the closed stop and qpos=lower at the
                # open stop, so its normalized direction is reversed.
                self._data.qpos[self._joint_qpos_addresses[side_index, local_index]] = upper - gripper * (
                    upper - lower
                )
        self._data.qvel[:] = 0.0

    def _classify_contact(self, contact: Any) -> CollisionContact | None:
        geom_ids = (int(contact.geom1), int(contact.geom2))
        body_ids = tuple(int(self._model.geom_bodyid[geom_id]) for geom_id in geom_ids)
        geom_names = tuple(self._object_name(self._mujoco.mjtObj.mjOBJ_GEOM, geom_id) for geom_id in geom_ids)
        body_names = tuple(self._object_name(self._mujoco.mjtObj.mjOBJ_BODY, body_id) for body_id in body_ids)

        table_index = next((index for index, name in enumerate(geom_names) if name == _TABLE_GEOM), None)
        if table_index is not None:
            robot_index = 1 - table_index
            robot_body = body_names[robot_index]
            if robot_body in {_LEFT_BASE, _RIGHT_BASE}:
                # A mounting base may intentionally meet the tabletop. Only
                # movable arm/gripper bodies are dispatch hazards.
                return None
            if _robot_side(robot_body) is None:
                return None
            kind: CollisionKind = "arm_table"
        else:
            sides = tuple(_robot_side(name) for name in body_names)
            if sides[0] is None or sides[1] is None:
                return None
            if sides[0] != sides[1]:
                kind = "inter_arm"
            elif body_ids[0] != body_ids[1]:
                # Preserve MuJoCo/i2rt self-collision behavior. The generated
                # model excludes only native gripper pairs and the independently
                # verified base-link1 false mesh overlap.
                kind = "self_collision"
            else:
                return None

        return CollisionContact(
            kind=kind,
            body1=body_names[0],
            body2=body_names[1],
            geom1=geom_names[0],
            geom2=geom_names[1],
            signed_distance_m=float(contact.dist),
        )

    def _object_name(self, object_type: Any, object_id: int) -> str:
        name = self._mujoco.mj_id2name(self._model, object_type, object_id)
        return name if name is not None else f"unnamed_{int(object_type)}_{object_id}"

    @staticmethod
    def _validate_command(
        command: Mapping[str, float] | Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        if isinstance(command, Mapping):
            expected = set(YAM_SCALAR_KEYS)
            actual = set(command)
            missing = [key for key in YAM_SCALAR_KEYS if key not in actual]
            extra = sorted(actual - expected)
            if missing or extra:
                raise ValueError(
                    f"command mapping keys do not match YAM schema; missing={missing}, extra={extra}"
                )
            values = np.asarray([command[key] for key in YAM_SCALAR_KEYS], dtype=np.float64)
        else:
            values = np.asarray(command, dtype=np.float64)
        if values.shape != (STATE_SIZE,):
            raise ValueError(f"command must have shape ({STATE_SIZE},), got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("command must contain only finite values")
        lower, upper = _driver_action_bounds()
        invalid = np.flatnonzero((values < lower) | (values > upper))
        if invalid.size:
            raise ValueError(f"command exceeds YAM joint or gripper limits at indices {invalid.tolist()}")
        return values.copy()


def _robot_side(body_name: str) -> Literal["left", "right"] | None:
    if body_name.startswith("left_"):
        return "left"
    if body_name.startswith("right_"):
        return "right"
    return None


def _matrix_from_translation_quaternion(
    translation_m: Sequence[float],
    quaternion_xyzw: Sequence[float],
) -> np.ndarray:
    translation = np.asarray(translation_m, dtype=np.float64)
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    if translation.shape != (3,) or quaternion.shape != (4,):
        raise RigCalibrationError("translation_m must have length 3 and quaternion_xyzw length 4")
    if not np.isfinite(translation).all() or not np.isfinite(quaternion).all():
        raise RigCalibrationError("translation and quaternion must be finite")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise RigCalibrationError("quaternion must be nonzero")
    x, y, z, w = quaternion / norm
    rotation = np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def _rotation_matrix_to_quaternion_wxyz(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a proper rotation matrix to MuJoCo's scalar-first quaternion."""

    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion = (
            0.25 * scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            quaternion = (
                (matrix[2, 1] - matrix[1, 2]) / scale,
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
            )
        elif index == 1:
            scale = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            quaternion = (
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
            )
        else:
            scale = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            quaternion = (
                (matrix[1, 0] - matrix[0, 1]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
            )
    normalized = np.asarray(quaternion, dtype=np.float64)
    normalized /= np.linalg.norm(normalized)
    return tuple(float(value) for value in normalized)
