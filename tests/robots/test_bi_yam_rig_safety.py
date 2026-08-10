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

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from lerobot.robots.bi_yam.config_bi_yam import YAM_JOINT_LIMITS, YAM_SCALAR_KEYS
from lerobot.robots.bi_yam.rig_safety import (
    CalibrationProvenance,
    DualYAMRigCalibration,
    ProvenanceRigidTransform,
    RigCalibrationError,
    RigCollisionChecker,
    TablePlaneCalibration,
    UnsafeRigCommandError,
    UnverifiedRigCalibrationError,
)
from lerobot.simulators.bi_yam.config import DEFAULT_INITIAL_STATE
from lerobot.simulators.bi_yam.simulator import mujoco_backend_available


def _synthetic_calibration(
    *,
    table_height_in_left_base_m: float,
    transform_verified: bool = True,
    table_verified: bool = True,
) -> DualYAMRigCalibration:
    # Established simulator fixture expressed as ^left_base T_right_base.
    # The physical rig must replace this with its measured transform.
    transform = np.asarray(
        (
            (-1.0, 0.0, 0.0, 0.0),
            (0.0, -1.0, 0.0, -0.64),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    return DualYAMRigCalibration(
        left_base_from_right_base=ProvenanceRigidTransform(
            parent_frame="left_base",
            child_frame="right_base",
            matrix=transform,
            provenance=CalibrationProvenance(
                verified=transform_verified,
                source="synthetic unit-test base fixture; not physical calibration",
                measurement_id="synthetic-base-v1",
            ),
        ),
        table_plane_in_left_base=TablePlaneCalibration(
            frame="left_base",
            point_m=(0.0, 0.0, table_height_in_left_base_m),
            normal_toward_workspace=(0.0, 0.0, 2.0),
            provenance=CalibrationProvenance(
                verified=table_verified,
                source="synthetic unit-test table fixture; not physical calibration",
                measurement_id="synthetic-table-v1",
            ),
        ),
    )


def _physical_calibration(
    *,
    tmp_path: Path,
    table_height_in_left_base_m: float,
) -> tuple[DualYAMRigCalibration, Path, Path]:
    transform_evidence = tmp_path / "laser-tracker-rig-survey.bin"
    transform_evidence.write_bytes(b"physical laser tracker survey run 2026-08-10")
    table_evidence = tmp_path / "table-plane-survey.bin"
    table_evidence.write_bytes(b"physical table plane survey run 2026-08-10")
    transform_evidence.chmod(0o444)
    table_evidence.chmod(0o444)

    def provenance(path: Path, source: str, measurement_id: str) -> CalibrationProvenance:
        return CalibrationProvenance(
            verified=True,
            source=source,
            measurement_id=measurement_id,
            evidence_path=str(path.resolve()),
            evidence_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            evidence_size_bytes=path.stat().st_size,
        )

    geometry = _synthetic_calibration(
        table_height_in_left_base_m=table_height_in_left_base_m,
    )
    calibration = DualYAMRigCalibration(
        left_base_from_right_base=ProvenanceRigidTransform(
            parent_frame="left_base",
            child_frame="right_base",
            matrix=geometry.left_base_from_right_base.matrix,
            provenance=provenance(
                transform_evidence,
                "laser tracker rig survey 2026-08-10",
                "rig-survey-20260810",
            ),
        ),
        table_plane_in_left_base=TablePlaneCalibration(
            frame="left_base",
            point_m=geometry.table_plane_in_left_base.point_m,
            normal_toward_workspace=geometry.table_plane_in_left_base.normal_toward_workspace,
            provenance=provenance(
                table_evidence,
                "laser plane fit survey 2026-08-10",
                "table-survey-20260810",
            ),
        ),
    )
    return calibration, transform_evidence, table_evidence


def test_rig_calibration_json_schema_has_explicit_frames_units_and_provenance(tmp_path) -> None:
    value = {
        "schema_version": 1,
        "left_base_from_right_base": {
            "parent_frame": "left_base",
            "child_frame": "right_base",
            "translation_m": [0.71, -0.03, 0.02],
            "quaternion_xyzw": [0.0, 0.0, 0.7071067811865475, 0.7071067811865476],
            "provenance": {
                "verified": True,
                "source": "metrology report rig-2026-08-09.pdf section 2",
                "measurement_id": "rig-survey-20260809",
            },
        },
        "table_plane_in_left_base": {
            "frame": "left_base",
            "point_m": [0.0, 0.0, -0.12],
            "normal_toward_workspace": [0.0, 0.0, 1.0],
            "provenance": {
                "verified": True,
                "source": "metrology report rig-2026-08-09.pdf section 3",
                "measurement_id": "table-survey-20260809",
            },
        },
    }
    path = tmp_path / "rig.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    calibration = DualYAMRigCalibration.from_json(path)

    assert calibration.verified
    np.testing.assert_allclose(calibration.left_base_from_right_base.matrix[:3, 3], (0.71, -0.03, 0.02))
    np.testing.assert_allclose(
        calibration.left_base_from_right_base.matrix[:3, :3],
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        atol=1e-12,
    )
    assert calibration.table_plane_in_left_base.normal_toward_workspace == (0.0, 0.0, 1.0)
    assert calibration.to_dict()["left_base_from_right_base"]["provenance"]["source"].startswith(
        "metrology report"
    )


def test_rig_calibration_rejects_ambiguous_transform_direction_and_missing_provenance() -> None:
    provenance = CalibrationProvenance(verified=True, source="synthetic")
    with pytest.raises(RigCalibrationError, match="pose of right_base in left_base"):
        DualYAMRigCalibration(
            left_base_from_right_base=ProvenanceRigidTransform(
                parent_frame="right_base",
                child_frame="left_base",
                matrix=np.eye(4),
                provenance=provenance,
            ),
            table_plane_in_left_base=TablePlaneCalibration(
                frame="left_base",
                point_m=(0.0, 0.0, 0.0),
                normal_toward_workspace=(0.0, 0.0, 1.0),
                provenance=provenance,
            ),
        )
    with pytest.raises(RigCalibrationError, match="provenance.source"):
        CalibrationProvenance(verified=True, source="")


def test_repository_synthetic_rig_never_authorizes_hardware_boundaries() -> None:
    calibration = _synthetic_calibration(table_height_in_left_base_m=-0.5)
    checker = object.__new__(RigCollisionChecker)
    checker.calibration = calibration

    with pytest.raises(UnverifiedRigCalibrationError, match="placeholder"):
        checker.assert_hardware_ready()
    with pytest.raises(UnverifiedRigCalibrationError, match="placeholder"):
        checker.validate_hardware_command(DEFAULT_INITIAL_STATE)
    with pytest.raises(UnverifiedRigCalibrationError, match="placeholder"):
        checker.validate_hardware_trajectory([DEFAULT_INITIAL_STATE])


def test_rig_physical_provenance_binds_local_read_only_evidence(tmp_path) -> None:
    calibration, transform_evidence, table_evidence = _physical_calibration(
        tmp_path=tmp_path,
        table_height_in_left_base_m=-0.12,
    )

    identity = calibration.require_verified_physical_evidence()

    assert identity["left_base_from_right_base"]["evidence_path"] == str(transform_evidence.resolve())
    assert (
        calibration.to_dict()["table_plane_in_left_base"]["provenance"]["evidence_size_bytes"]
        == table_evidence.stat().st_size
    )

    table_evidence.chmod(0o644)
    with pytest.raises(UnverifiedRigCalibrationError, match="must be read-only"):
        calibration.require_verified_physical_evidence()


def test_hardware_ready_accepts_complete_physical_evidence(tmp_path) -> None:
    calibration, _, _ = _physical_calibration(
        tmp_path=tmp_path,
        table_height_in_left_base_m=-0.12,
    )
    checker = object.__new__(RigCollisionChecker)
    checker.calibration = calibration

    checker.assert_hardware_ready()


def test_rig_endpoint_schema_uses_authoritative_driver_joint_bounds() -> None:
    arm_lower = tuple(limit[0] for limit in YAM_JOINT_LIMITS) + (0.0,)
    arm_upper = tuple(limit[1] for limit in YAM_JOINT_LIMITS) + (1.0,)
    lower = np.asarray(arm_lower + arm_lower)
    upper = np.asarray(arm_upper + arm_upper)

    np.testing.assert_array_equal(RigCollisionChecker._validate_command(lower), lower)
    np.testing.assert_array_equal(RigCollisionChecker._validate_command(upper), upper)
    outside = upper.copy()
    outside[0] += 1e-4
    with pytest.raises(ValueError, match="exceeds YAM joint or gripper limits.*0"):
        RigCollisionChecker._validate_command(outside)


@pytest.mark.skipif(not mujoco_backend_available(), reason="MuJoCo and i2rt are optional")
def test_unverified_rig_can_be_inspected_but_never_authorizes_hardware() -> None:
    calibration = _synthetic_calibration(
        table_height_in_left_base_m=-0.5,
        transform_verified=False,
    )
    checker = RigCollisionChecker(calibration, minimum_clearance_m=0.0)

    diagnostic = checker.inspect_command(DEFAULT_INITIAL_STATE)

    assert diagnostic.safe
    assert not diagnostic.calibration_verified
    with pytest.raises(UnverifiedRigCalibrationError, match="left_base_from_right_base"):
        checker.assert_hardware_ready()
    with pytest.raises(UnverifiedRigCalibrationError, match="left_base_from_right_base"):
        checker(DEFAULT_INITIAL_STATE)


@pytest.mark.skipif(not mujoco_backend_available(), reason="MuJoCo and i2rt are optional")
def test_physically_evidenced_separated_rig_is_safe_and_collision_world_has_no_task_objects(
    tmp_path,
) -> None:
    import mujoco

    calibration, _, _ = _physical_calibration(
        tmp_path=tmp_path,
        table_height_in_left_base_m=0.0,
    )
    checker = RigCollisionChecker(calibration, minimum_clearance_m=0.0)

    command_mapping = dict(zip(YAM_SCALAR_KEYS, DEFAULT_INITIAL_STATE, strict=True))
    checker.assert_hardware_ready()
    report = checker(command_mapping)

    assert report.safe
    assert report.calibration_verified
    body_names = {
        mujoco.mj_id2name(checker._model, mujoco.mjtObj.mjOBJ_BODY, index)
        for index in range(checker._model.nbody)
    }
    assert body_names.isdisjoint({"table", "red_cube", "blue_cube", "green_cylinder"})
    assert (
        mujoco.mj_name2id(
            checker._model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "calibrated_table_plane",
        )
        >= 0
    )
    for side in ("left", "right"):
        assert (
            mujoco.mj_name2id(
                checker._model,
                mujoco.mjtObj.mjOBJ_EXCLUDE,
                f"{side}_base_link1",
            )
            >= 0
        )
    with pytest.raises(ValueError, match="missing=.*left_joint_0.pos"):
        checker({key: value for key, value in command_mapping.items() if key != "left_joint_0.pos"})
    with pytest.raises(ValueError, match="extra=.*not_a_yam_joint"):
        checker(command_mapping | {"not_a_yam_joint": 0.0})


@pytest.mark.skipif(not mujoco_backend_available(), reason="MuJoCo and i2rt are optional")
def test_rig_collision_model_maps_driver_gripper_zero_closed_and_one_open() -> None:
    import mujoco

    checker = RigCollisionChecker(
        _synthetic_calibration(table_height_in_left_base_m=-0.5),
        minimum_clearance_m=0.0,
    )

    def projected_aperture_mm(gripper: float) -> float:
        command = np.asarray(DEFAULT_INITIAL_STATE, dtype=np.float64).copy()
        command[[6, 13]] = gripper
        checker._write_command(command)
        mujoco.mj_forward(checker._model, checker._data)
        left_tip = mujoco.mj_name2id(checker._model, mujoco.mjtObj.mjOBJ_BODY, "left_tip_left")
        right_tip = mujoco.mj_name2id(checker._model, mujoco.mjtObj.mjOBJ_BODY, "left_tip_right")
        jaw_joint = mujoco.mj_name2id(checker._model, mujoco.mjtObj.mjOBJ_JOINT, "left_joint7")
        separation = checker._data.xpos[left_tip] - checker._data.xpos[right_tip]
        return float(abs(np.dot(separation, checker._data.xaxis[jaw_joint])) * 1e3)

    assert projected_aperture_mm(0.0) == pytest.approx(4.88, abs=0.05)
    assert projected_aperture_mm(1.0) == pytest.approx(90.12, abs=0.2)


@pytest.mark.skipif(not mujoco_backend_available(), reason="MuJoCo and i2rt are optional")
def test_known_bimanual_pose_is_rejected_as_inter_arm_collision(tmp_path) -> None:
    calibration, _, _ = _physical_calibration(
        tmp_path=tmp_path,
        table_height_in_left_base_m=0.0,
    )
    checker = RigCollisionChecker(calibration, minimum_clearance_m=0.0)
    command = (-2.1, 2.6, 2.3, 0.0, 0.0, 0.0, 0.5, 2.2, 0.3, 2.3, 0.0, 0.0, 0.0, 0.5)

    diagnostic = checker.inspect_command(command)

    assert any(contact.kind == "inter_arm" and contact.penetrating for contact in diagnostic.contacts)
    assert {(contact.body1, contact.body2) for contact in diagnostic.contacts} == {
        ("left_link3", "right_link3")
    }
    with pytest.raises(UnsafeRigCommandError) as error:
        checker(command)
    assert any(contact.kind == "inter_arm" for contact in error.value.report.contacts)


@pytest.mark.skipif(not mujoco_backend_available(), reason="MuJoCo and i2rt are optional")
def test_verified_table_plane_collision_is_rejected_independently_of_arm_spacing(tmp_path) -> None:
    calibration, _, _ = _physical_calibration(
        tmp_path=tmp_path,
        table_height_in_left_base_m=0.0,
    )
    checker = RigCollisionChecker(calibration, minimum_clearance_m=0.0)
    command = (-1.6, 2.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.2, 1.8, 0.0, 0.0, 0.0, 0.5)

    diagnostic = checker.inspect_command(command)

    assert {contact.kind for contact in diagnostic.contacts} == {"arm_table"}
    assert {contact.body2 for contact in diagnostic.contacts} == {"left_tip_left", "left_tip_right"}
    assert all("base" not in {contact.body1, contact.body2} for contact in diagnostic.contacts)
    with pytest.raises(UnsafeRigCommandError, match="arm_table"):
        checker(command)


@pytest.mark.skipif(not mujoco_backend_available(), reason="MuJoCo and i2rt are optional")
def test_nonadjacent_self_collisions_remain_dispatch_failures(tmp_path) -> None:
    calibration, _, _ = _physical_calibration(
        tmp_path=tmp_path,
        table_height_in_left_base_m=-0.5,
    )
    checker = RigCollisionChecker(calibration, minimum_clearance_m=0.0)
    command = (
        2.5556809846734634,
        3.150695887742668,
        0.18459647127855258,
        1.2765569157523722,
        0.5764983243238893,
        2.0387403733455005,
        0.5,
        0.0,
        1.2,
        1.8,
        0.0,
        0.0,
        0.0,
        0.5,
    )

    diagnostic = checker.inspect_command(command)

    assert {contact.kind for contact in diagnostic.contacts} == {"self_collision"}
    assert {(contact.body1, contact.body2) for contact in diagnostic.contacts} == {
        ("left_base", "left_link3"),
        ("left_base", "left_link4"),
    }
    with pytest.raises(UnsafeRigCommandError, match="self_collision"):
        checker(command)


@pytest.mark.skipif(not mujoco_backend_available(), reason="MuJoCo and i2rt are optional")
def test_hardware_trajectory_checks_every_sample(tmp_path) -> None:
    safe, _, _ = _physical_calibration(
        tmp_path=tmp_path,
        table_height_in_left_base_m=0.0,
    )
    checker = RigCollisionChecker(safe, minimum_clearance_m=0.0)
    commands = np.repeat(np.asarray(DEFAULT_INITIAL_STATE)[None, :], 3, axis=0)

    reports = checker.validate_hardware_trajectory(commands)

    assert len(reports) == 3
    assert all(report.safe for report in reports)
