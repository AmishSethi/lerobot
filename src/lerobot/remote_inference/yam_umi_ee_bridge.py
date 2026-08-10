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
"""Safe runtime bridge from a bimanual UMI pose policy to a physical BiYAM.

The frame algebra, IK, gripper conversion, action semantics, and residual checks
live in :mod:`lerobot.robots.bi_yam.umi_retargeting`.  This module owns only the
control-loop boundary:

1. capture one measured YAM observation and its two wrist images;
2. submit the checkpoint-specific UMI policy state;
3. resolve the returned chunk exactly once at that same observation;
4. execute rows at the declared control rate against fresh measured joints.

No object, bowl, image landmark, or task outcome enters the conversion. The new
20-D checkpoint uses current-relative R6D SE(3) and the fixed jaw-axis basis; the
existing run8 checkpoint remains a separate 14-D episode-start semantic. A
shared-world collision gate is optional until rig/table transforms are available.
"""

from __future__ import annotations

import hashlib
import json
import logging
import queue
import re
import stat
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401 - register Draccus camera choice
from lerobot.robots.bi_yam.config_bi_yam import (
    BI_YAM_POLICY_START_POSITION,
    YAM_JOINT_LIMITS,
    BiYAMFollowerConfig,
)
from lerobot.robots.bi_yam.umi_retargeting import (
    CURRENT_RELATIVE_R6D_SE3,
    CURRENT_RELATIVE_R6D_SE3_JAW_DELTA,
    EPISODE_START_ABSOLUTE,
    GRIPPER_ACTION_QUERY_ANCHOR_DELTA,
    LEFT_GRIPPER_KEY,
    PINNED_I2RT_YAM_URDF_SHA256,
    RIGHT_GRIPPER_KEY,
    YAM_SCALAR_KEYS,
    CurrentRelativeGripperMap,
    EpisodeFrames,
    SafetyLimits,
    TrackingPolicy,
    UmiActionSemantics,
    YamArmKinematics,
    YamUmiEeAdapter,
    load_calibrations_json,
    load_current_relative_gripper_map_json,
    validate_trusted_yam_urdf,
)

logger = logging.getLogger(__name__)

_CURRENT_RELATIVE_R6D_SEMANTICS = {
    CURRENT_RELATIVE_R6D_SE3,
    CURRENT_RELATIVE_R6D_SE3_JAW_DELTA,
}
_CURRENT_RELATIVE_SERVED_ACTION_HORIZON = 24
_CURRENT_RELATIVE_ACTIVE_EXECUTION_HORIZON = 15
_ONSET_V3_GATE_SCHEMA_ID = "umi-yam-onset-v3-checkpoint-gate-plan-v1"
_ONSET_V3_DATASET_SCHEMA_ID = "dual-lidar-umi-currentrel-r6d-onset-v3"
_ONSET_V3_DATASET_REPO_ID = "brandonyang/dual-lidar-umi-currentrel-r6d-onset-v3"
_ONSET_V3_GATE_PLAN_SHA256 = "f5841191715930db0c9edd29aa96cc45971fe1f36460c70e03f40e596ccd4d36"
_ONSET_V3_HARDWARE_COMMISSIONING_SCHEMA_ID = "umi-yam-onset-v3-hardware-commissioning-evidence-v1"
_ONSET_V3_GRIPPER_TRAINING_SUPPORT = {
    "left": (0.4486217200756073, 1.0),
    "right": (0.5472135543823242, 0.9483147859573364),
}
_PI05_EXPLICIT_NOISE_BACKEND_MODE = "supervised_hardware_trial"
_PI05_EXPLICIT_NOISE_SERVICE_PREFIX = "lerobot-pi05-onset-v3-explicit-noise-seed-"
_V4_IK_RELEASE_SCHEMA_ID = "umi-yam-v4-fixed-global-start-anchor-ik-release-v1"
_V4_HARDWARE_COMMISSIONING_SCHEMA_ID = "umi-yam-v4-hardware-commissioning-evidence-v1"
_V4_ARTIFACT_SCHEMA_ID = "dual-lidar-umi-shared-left-currentrel-r6d-jawdelta-onset-v4"
_V4_IK_QUERY_COUNT = 8_362
_V4_EXECUTED_ACTION_ROWS_PER_ARM = 125_430
_V4_ARTIFACT_BINDING_DIGESTS = {
    "checksum_manifest_sha256": "2e11de3ed8dfac9c10a6afb9872a7d0ede16e629ade62dc1762469dea9f58a93",
    "artifact_manifest_json_sha256": "2302aef33b9d8fba299cd1a3f828b8e438b358945a7e4c45f2399a7ba0159076",
    "semantic_metadata_sha256": "f3cf93d79a2c87fa183fde5daf058016346f5295679fbaff68fb5f7b0e9d1cae",
    "split_manifest_sha256": "86f65399b0a7bc9626d5cf34ce460b17332991961f0b26224dbabfc7d78bac7d",
    "pose_origin_provenance_sha256": "ab53f7a5efdb5e858d1b79c31af0008a9157ed2c91f83bf1876d128ab39279cc",
}


def _onset_v3_gripper_support_identity(
    calibrated_reset_scalars: Mapping[str, float],
) -> dict[str, Any]:
    """Bind calibrated reset jaw scalars to the frozen onset-v3 train bounds."""

    if set(calibrated_reset_scalars) != {"left", "right"}:
        raise ValueError("onset-v3 calibrated reset scalars must contain left and right")
    values = {arm: float(calibrated_reset_scalars[arm]) for arm in ("left", "right")}
    if not np.isfinite(tuple(values.values())).all():
        raise ValueError("onset-v3 calibrated reset scalars must be finite")
    support = {arm: list(_ONSET_V3_GRIPPER_TRAINING_SUPPORT[arm]) for arm in ("left", "right")}
    within = {arm: bool(support[arm][0] <= values[arm] <= support[arm][1]) for arm in ("left", "right")}
    return {
        "calibrated_umi_width": values,
        "frozen_training_support": support,
        "within_frozen_training_support": within,
        "all_within_frozen_training_support": all(within.values()),
    }


def _require_lower_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _matches_exact_json_contract(value: object, expected: object) -> bool:
    """Compare JSON values without Python's bool/int or int/float coercions."""

    try:
        options = {
            "sort_keys": True,
            "separators": (",", ":"),
            "ensure_ascii": False,
            "allow_nan": False,
        }
        return json.dumps(value, **options) == json.dumps(expected, **options)
    except (TypeError, ValueError):
        return False


def _load_onset_v3_gate_identity(
    *,
    report_path: Path,
    expected_report_sha256: str,
    policy_start_position: Sequence[float],
) -> dict[str, Any]:
    """Bind the original onset-v3 trial to its exact reviewed ACTIVE15 anchor plan."""

    expected = _require_lower_sha256(
        expected_report_sha256,
        label="onset-v3 gate plan expected SHA-256",
    )
    if expected != _ONSET_V3_GATE_PLAN_SHA256:
        raise ValueError("onset-v3 deployment must use the exact reviewed gate plan SHA-256")
    path, actual = _read_only_file_identity(Path(report_path), label="onset-v3 gate plan")
    if actual != expected:
        raise ValueError(f"onset-v3 gate plan SHA-256 mismatch: expected {expected}, got {actual}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"onset-v3 gate plan is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(report, Mapping) or report.get("schema_id") != _ONSET_V3_GATE_SCHEMA_ID:
        raise ValueError("onset-v3 gate plan schema changed")
    dataset = report.get("dataset")
    if not isinstance(dataset, Mapping) or (
        dataset.get("schema_id") != _ONSET_V3_DATASET_SCHEMA_ID
        or dataset.get("repo_id") != _ONSET_V3_DATASET_REPO_ID
        or dataset.get("holdout_episodes") != [52, 53]
        or dataset.get("query_index_within_exported_episode") != 0
    ):
        raise ValueError("onset-v3 gate plan dataset binding changed")
    execution = report.get("execution")
    if not isinstance(execution, Mapping) or (
        execution.get("horizon_rows") != 15
        or execution.get("full_24_row_execution_performed") is not False
        or execution.get("strict_ik_required_for_every_arm_row") is not True
    ):
        raise ValueError("onset-v3 gate plan ACTIVE15 execution contract changed")
    anchor = report.get("yam_start_anchor")
    expected_anchor = [0.0, 0.05, 0.05, 0.0, 0.0, 0.0, 1.0] * 2
    if not isinstance(anchor, Mapping) or (
        anchor.get("absolute_driver_target") != expected_anchor
        or anchor.get("left_arm_joints_rad") != expected_anchor[:6]
        or anchor.get("right_arm_joints_rad") != expected_anchor[7:13]
        or anchor.get("grippers_normalized") != [1.0, 1.0]
        or anchor.get("hardware_start_verified") is not False
    ):
        raise ValueError("onset-v3 gate plan start-anchor contract changed")
    configured = np.asarray(policy_start_position, dtype=np.float64)
    if configured.shape != (14,) or not np.array_equal(configured, np.asarray(expected_anchor)):
        raise ValueError("onset-v3 robot policy_start_position does not match the reviewed interior anchor")
    return {
        "schema_id": _ONSET_V3_GATE_SCHEMA_ID,
        "path": str(path),
        "sha256": actual,
        "dataset": dict(dataset),
        "execution": {
            "served_action_horizon_rows": 24,
            "active_execution_rows": 15,
            "diagnostic_only_rows_1_indexed": list(range(16, 25)),
        },
        "yam_start_anchor": dict(anchor),
        "hardware_trial_status": "EXPERIMENTAL_REQUIRES_PHYSICAL_COMMISSIONING",
    }


def _load_v4_ik_release_identity(
    *,
    report_path: Path,
    expected_report_sha256: str,
    expected_dataset_root: str,
    policy_start_position: Sequence[float],
    expected_start_gripper_yam: Sequence[float],
) -> dict[str, Any]:
    """Mirror the frozen onset-v4 training IK release contract at deployment."""

    expected_report_sha256 = _require_lower_sha256(
        expected_report_sha256,
        label="v4 IK release report expected SHA-256",
    )
    path = Path(report_path).expanduser()
    if path.is_symlink() or not path.is_file():
        raise ValueError("v4 IK release report must be a regular file, not a symlink")
    if stat.S_IMODE(path.stat().st_mode) & 0o222:
        raise ValueError("v4 IK release report must be read-only")
    resolved = path.resolve(strict=True)
    actual_report_sha256 = _sha256_path(resolved)
    if actual_report_sha256 != expected_report_sha256:
        raise ValueError(
            "v4 IK release report SHA-256 mismatch: "
            f"expected {expected_report_sha256}, got {actual_report_sha256}"
        )
    try:
        report = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"v4 IK release report is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(report, Mapping):
        raise ValueError("v4 IK release report must contain a JSON object")
    top_level_keys = {
        "schema_id",
        "validation_passed",
        "kinematic_training_gate_passed",
        "training_release_ready",
        "hardware_deployment_ready",
        "artifact_binding",
        "selection_provenance",
        "fixed_global_start_anchor",
        "coverage",
        "diagnostic_coverage",
        "result",
    }
    if set(report) != top_level_keys:
        raise ValueError(
            "v4 IK release report top-level keys changed: "
            f"expected {sorted(top_level_keys)}, got {sorted(report)}"
        )
    expected_flags = {
        "schema_id": _V4_IK_RELEASE_SCHEMA_ID,
        "validation_passed": True,
        "kinematic_training_gate_passed": True,
        "training_release_ready": True,
        "hardware_deployment_ready": False,
    }
    for field_name, expected in expected_flags.items():
        if not _matches_exact_json_contract(report.get(field_name), expected):
            raise ValueError(f"v4 IK release report field {field_name!r} changed")

    dataset_root = str(expected_dataset_root).strip()
    dataset_path = Path(dataset_root)
    if not dataset_root or not dataset_path.is_absolute() or dataset_path.as_posix() != dataset_root:
        raise ValueError("expected_v4_artifact_dataset_root must be an absolute normalized path")
    expected_artifact = {
        "schema_id": _V4_ARTIFACT_SCHEMA_ID,
        "dataset_root": dataset_root,
        **_V4_ARTIFACT_BINDING_DIGESTS,
    }
    if not _matches_exact_json_contract(report.get("artifact_binding"), expected_artifact):
        raise ValueError("v4 IK release report does not bind the exact reviewed artifact")
    expected_selection = {
        "candidate_selection_episodes": list(range(14, 135)),
        "validation_and_test_used_for_selection": False,
        "final_confirmation_episodes": list(range(150)),
    }
    if not _matches_exact_json_contract(report.get("selection_provenance"), expected_selection):
        raise ValueError("v4 IK anchor selection did not preserve held-out selection provenance")

    anchor = report.get("fixed_global_start_anchor")
    anchor_keys = {
        "contract",
        "joint_units",
        "left_arm_joint_radians",
        "right_arm_joint_radians",
        "episode_overrides",
        "frame_overrides",
        "hardware_verified",
    }
    if not isinstance(anchor, Mapping) or set(anchor) != anchor_keys:
        raise ValueError("v4 fixed_global_start_anchor keys changed")
    if anchor.get("contract") != "one_fixed_anchor_for_every_episode":
        raise ValueError("v4 IK release anchor is not one fixed global anchor")
    if anchor.get("joint_units") != "radians":
        raise ValueError("v4 IK release anchor joint units must be radians")

    def arm_anchor(arm: str) -> list[float]:
        raw = anchor.get(f"{arm}_arm_joint_radians")
        if not isinstance(raw, list) or len(raw) != 6:
            raise ValueError(f"v4 {arm} IK release anchor must contain six joints")
        values = []
        for index, value in enumerate(raw):
            if isinstance(value, bool) or not isinstance(value, int | float) or not np.isfinite(value):
                raise ValueError(f"v4 {arm} IK release anchor joint {index} must be finite")
            values.append(float(value))
        return values

    left_anchor = arm_anchor("left")
    right_anchor = arm_anchor("right")
    if anchor.get("episode_overrides") != {} or anchor.get("frame_overrides") != {}:
        raise ValueError("v4 IK release anchor overrides are forbidden")
    if anchor.get("hardware_verified") is not False:
        raise ValueError("v4 IK release report must not claim anchor hardware verification")

    expected_coverage = {
        "episode_indices": list(range(150)),
        "arms": ["left", "right"],
        "query_scope": "canonical_onset_q0_then_every_15_completed_rows_carried_q",
        "query_count": _V4_IK_QUERY_COUNT,
        "row_scope": "active_rows_1_to_15_only",
        "served_action_horizon_rows": 24,
        "active_execution_rows_per_query": 15,
        "diagnostic_only_rows_1_indexed": list(range(16, 25)),
        "expected_executed_rows_per_arm": {
            "left": _V4_EXECUTED_ACTION_ROWS_PER_ARM,
            "right": _V4_EXECUTED_ACTION_ROWS_PER_ARM,
        },
        "evaluated_executed_rows_per_arm": {
            "left": _V4_EXECUTED_ACTION_ROWS_PER_ARM,
            "right": _V4_EXECUTED_ACTION_ROWS_PER_ARM,
        },
        "passed_executed_rows_per_arm": {
            "left": _V4_EXECUTED_ACTION_ROWS_PER_ARM,
            "right": _V4_EXECUTED_ACTION_ROWS_PER_ARM,
        },
        "failed_executed_rows_per_arm": {"left": 0, "right": 0},
        "padded_rows_evaluated": 0,
    }
    if not _matches_exact_json_contract(report.get("coverage"), expected_coverage):
        raise ValueError("v4 IK release report does not cover the exact carried-q ACTIVE15 deployment grid")
    expected_diagnostic_coverage = {
        "rows_1_indexed": list(range(16, 25)),
        "eligibility_basis": False,
        "evaluated": False,
        "pass_claim": False,
    }
    if not _matches_exact_json_contract(report.get("diagnostic_coverage"), expected_diagnostic_coverage):
        raise ValueError(
            "v4 IK release report must make no unevaluated pass claim for diagnostic rows 16..24"
        )
    expected_result = {
        "all_episodes_pass": True,
        "all_arms_pass": True,
        "all_executed_rows_pass": True,
        "failed_episode_indices": [],
        "failed_executed_rows": [],
    }
    if not _matches_exact_json_contract(report.get("result"), expected_result):
        raise ValueError("v4 IK release report contains an IK failure or exclusion")

    start = np.asarray(policy_start_position, dtype=np.float64)
    if start.shape != (14,) or not np.isfinite(start).all():
        raise ValueError("v4 robot.policy_start_position must contain 14 finite values")
    if not np.array_equal(start[:6], np.asarray(left_anchor)) or not np.array_equal(
        start[7:13], np.asarray(right_anchor)
    ):
        raise ValueError(
            "v4 robot.policy_start_position arm joints do not exactly match the fixed IK release anchor"
        )
    grippers = np.asarray(expected_start_gripper_yam, dtype=np.float64)
    if grippers.shape != (2,) or not np.isfinite(grippers).all() or np.any((grippers < 0) | (grippers > 1)):
        raise ValueError("expected_v4_policy_start_gripper_yam must contain two finite values in [0, 1]")
    if not np.array_equal(start[[6, 13]], grippers):
        raise ValueError("v4 policy-start grippers do not match expected_v4_policy_start_gripper_yam")
    return _jsonable(
        {
            "path": str(resolved),
            "sha256": expected_report_sha256,
            "artifact_binding": expected_artifact,
            "selection_provenance": expected_selection,
            "fixed_global_start_anchor": dict(anchor),
            "coverage": expected_coverage,
            "diagnostic_coverage": expected_diagnostic_coverage,
            "result": expected_result,
            "policy_start_gripper_yam": grippers.tolist(),
        }
    )


def _nonplaceholder_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    folded = value.casefold()
    sentinel_tokens = {
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
    normalized = re.sub(r"[^a-z0-9]+", "", folded)
    tokens = set(re.findall(r"[a-z0-9]+", folded))
    embedded_sentinels = sentinel_tokens - {"na"}
    if (
        normalized in sentinel_tokens
        or tokens & sentinel_tokens
        or any(sentinel in normalized for sentinel in embedded_sentinels)
    ):
        raise ValueError(f"{label} must not be a placeholder")
    return value


def _require_physical_evidence_sha256(value: object, *, label: str) -> str:
    digest = _require_lower_sha256(value, label=label)
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
        for payload in sentinel_payloads
        for variant in (payload, payload.upper(), payload.title())
    }
    repeated = any(64 % period == 0 and digest == digest[:period] * (64 // period) for period in range(1, 33))
    if repeated or digest in sentinel_digests:
        raise ValueError(f"{label} must not be a zero, repeated, or sentinel digest")
    return digest


def _read_only_file_identity(path: Path, *, label: str) -> tuple[Path, str]:
    path = Path(path).expanduser()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file, not a symlink")
    if stat.S_IMODE(path.stat().st_mode) & 0o222:
        raise ValueError(f"{label} must be read-only")
    resolved = path.resolve(strict=True)
    return resolved, _sha256_path(resolved)


def _local_physical_evidence_identity(
    *,
    path_value: object,
    size_value: object,
    sha256_value: object,
    label: str,
) -> dict[str, str | int]:
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{label} path must be a non-empty absolute path")
    path = Path(path_value)
    if not path.is_absolute() or path.as_posix() != path_value:
        raise ValueError(f"{label} path must be absolute and normalized")
    resolved, actual_sha256 = _read_only_file_identity(path, label=label)
    if resolved != path:
        raise ValueError(f"{label} path must not contain symlink components")
    if isinstance(size_value, bool) or not isinstance(size_value, int) or size_value <= 0:
        raise ValueError(f"{label} size must be a positive integer")
    actual_size = resolved.stat().st_size
    if size_value != actual_size:
        raise ValueError(f"{label} size mismatch: expected {size_value}, got {actual_size}")
    claimed_sha256 = _require_physical_evidence_sha256(
        sha256_value,
        label=f"{label} SHA-256",
    )
    if claimed_sha256 != actual_sha256:
        raise ValueError(f"{label} SHA-256 mismatch: expected {claimed_sha256}, got {actual_sha256}")
    return {"path": str(resolved), "size_bytes": actual_size, "sha256": actual_sha256}


def _configured_camera_stable_id(cfg: YamUmiDeploymentConfig, *, arm: str) -> tuple[str, str]:
    camera_key = cfg.left_camera_key if arm == "left" else cfg.right_camera_key
    camera = cfg.robot.cameras.get(camera_key)
    if camera is None:
        raise ValueError(f"v4 commissioning requires configured {arm} camera {camera_key!r}")
    raw_id = None
    for attribute in ("index_or_path", "serial_number", "device_id"):
        candidate = getattr(camera, attribute, None)
        if candidate is not None:
            raw_id = candidate
            break
    if raw_id is None or isinstance(raw_id, bool | int):
        raise ValueError(f"v4 {arm} camera must use a stable string device identifier")
    stable_id = _nonplaceholder_text(str(raw_id), label=f"v4 {arm} camera stable device id")
    if stable_id.startswith("/dev/video") and "/by-" not in stable_id:
        raise ValueError(f"v4 {arm} camera must not use an unstable /dev/videoN path")
    return camera_key, stable_id


def _load_v4_hardware_commissioning_identity(
    *,
    cfg: YamUmiDeploymentConfig,
    evidence_path: Path,
    expected_evidence_sha256: str,
) -> dict[str, Any]:
    """Validate immutable physical commissioning evidence before hardware construction."""

    expected_evidence_sha256 = _require_physical_evidence_sha256(
        expected_evidence_sha256,
        label="v4 hardware commissioning evidence expected SHA-256",
    )
    configured_evidence_path = Path(evidence_path).expanduser()
    if not configured_evidence_path.is_absolute() or configured_evidence_path.as_posix() != str(
        configured_evidence_path
    ):
        raise ValueError("v4 hardware commissioning evidence path must be absolute and normalized")
    evidence, evidence_sha256 = _read_only_file_identity(
        configured_evidence_path,
        label="v4 hardware commissioning evidence",
    )
    if evidence != configured_evidence_path:
        raise ValueError("v4 hardware commissioning evidence path must not contain symlink components")
    if evidence_sha256 != expected_evidence_sha256:
        raise ValueError(
            "v4 hardware commissioning evidence SHA-256 mismatch: "
            f"expected {expected_evidence_sha256}, got {evidence_sha256}"
        )
    try:
        payload = json.loads(evidence.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"v4 hardware commissioning evidence is not valid UTF-8 JSON: {exc}") from exc
    top_keys = {
        "schema_id",
        "commissioning_passed",
        "robot",
        "fixed_global_start_anchor",
        "calibrations",
        "cameras",
        "measured_reset",
    }
    if not isinstance(payload, Mapping) or set(payload) != top_keys:
        raise ValueError("v4 hardware commissioning evidence top-level keys changed")
    expected_commissioning_schema = (
        _ONSET_V3_HARDWARE_COMMISSIONING_SCHEMA_ID
        if cfg.onset_v3_gate_plan is not None
        else _V4_HARDWARE_COMMISSIONING_SCHEMA_ID
    )
    if payload.get("schema_id") != expected_commissioning_schema:
        raise ValueError("unsupported hardware commissioning evidence schema")
    if payload.get("commissioning_passed") is not True:
        raise ValueError("v4 hardware commissioning evidence is not passed")

    robot = payload.get("robot")
    robot_keys = {
        "robot_id",
        "left_arm_id",
        "right_arm_id",
        "left_adapter_serial",
        "right_adapter_serial",
    }
    if not isinstance(robot, Mapping) or set(robot) != robot_keys:
        raise ValueError("v4 commissioning robot identity keys changed")
    _nonplaceholder_text(robot.get("robot_id"), label="commissioned robot id")
    if robot.get("robot_id") != cfg.robot.id:
        raise ValueError("v4 commissioning robot_id does not match deployment config")
    left_arm_id = _nonplaceholder_text(robot.get("left_arm_id"), label="left arm id")
    right_arm_id = _nonplaceholder_text(robot.get("right_arm_id"), label="right arm id")
    if left_arm_id == right_arm_id:
        raise ValueError("v4 commissioning left and right arm IDs must differ")
    for arm in ("left", "right"):
        configured_serial = getattr(cfg.robot, f"{arm}_arm_config").adapter_serial
        configured_serial = _nonplaceholder_text(
            configured_serial,
            label=f"configured {arm} adapter serial",
        )
        if robot.get(f"{arm}_adapter_serial") != configured_serial:
            raise ValueError(f"v4 commissioning {arm} adapter serial does not match deployment config")

    anchor = payload.get("fixed_global_start_anchor")
    anchor_keys = {
        "joint_units",
        "left_arm_joint_radians",
        "left_gripper_yam",
        "right_arm_joint_radians",
        "right_gripper_yam",
    }
    if cfg.onset_v3_gate_plan is not None:
        anchor_keys.update(
            {
                "left_gripper_calibrated_umi",
                "left_gripper_training_support",
                "right_gripper_calibrated_umi",
                "right_gripper_training_support",
            }
        )
    if not isinstance(anchor, Mapping) or set(anchor) != anchor_keys:
        raise ValueError("v4 commissioning fixed-global-start anchor keys changed")
    if anchor.get("joint_units") != "radians":
        raise ValueError("v4 commissioning arm joint units must be radians")
    left_anchor = anchor.get("left_arm_joint_radians")
    right_anchor = anchor.get("right_arm_joint_radians")
    if not isinstance(left_anchor, list) or len(left_anchor) != 6:
        raise ValueError("v4 commissioning left arm anchor must contain six joints")
    if not isinstance(right_anchor, list) or len(right_anchor) != 6:
        raise ValueError("v4 commissioning right arm anchor must contain six joints")
    configured_start = np.asarray(cfg.robot.policy_start_position, dtype=np.float64)
    try:
        claimed_start = np.asarray(
            [
                *left_anchor,
                anchor.get("left_gripper_yam"),
                *right_anchor,
                anchor.get("right_gripper_yam"),
            ],
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("v4 commissioning fixed-global-start anchor must contain numeric values") from exc
    if claimed_start.shape != (14,) or not np.isfinite(claimed_start).all():
        raise ValueError("v4 commissioning fixed-global-start anchor must contain 14 finite values")
    if not np.array_equal(claimed_start, configured_start):
        raise ValueError("v4 commissioning fixed-global-start anchor does not match deployment config")
    calibrated_reset_scalars = None
    if cfg.onset_v3_gate_plan is not None:
        mapping = load_current_relative_gripper_map_json(cfg.gripper_calibration_json)
        mapping.assert_hardware_ready()
        calibrated_reset_scalars = {}
        for arm, yam_index in (("left", 6), ("right", 13)):
            expected_support = list(_ONSET_V3_GRIPPER_TRAINING_SUPPORT[arm])
            if anchor.get(f"{arm}_gripper_training_support") != expected_support:
                raise ValueError(
                    f"onset-v3 commissioning {arm} gripper support does not match frozen train-state bounds"
                )
            computed = mapping.yam_to_umi(float(configured_start[yam_index]), arm=arm)
            claimed = anchor.get(f"{arm}_gripper_calibrated_umi")
            if isinstance(claimed, bool) or not isinstance(claimed, int | float):
                raise ValueError(f"onset-v3 commissioning {arm} calibrated UMI scalar must be numeric")
            if not np.isfinite(claimed) or not np.isclose(float(claimed), computed, rtol=0.0, atol=1e-9):
                raise ValueError(
                    f"onset-v3 commissioning {arm} calibrated UMI scalar does not match endpoint calibration"
                )
            if not expected_support[0] <= computed <= expected_support[1]:
                raise ValueError(
                    f"onset-v3 commissioning {arm} calibrated UMI scalar is outside frozen training support"
                )
            calibrated_reset_scalars[arm] = computed

    if cfg.robot.calibration_dir is None or cfg.rig_calibration_json is None:
        raise ValueError("v4 hardware commissioning requires explicit robot and rig calibration paths")
    robot_calibration_path = Path(cfg.robot.calibration_dir).expanduser() / f"{cfg.robot.id}.json"
    calibration_paths = {
        "bi_yam": robot_calibration_path,
        "gripper_endpoints": cfg.gripper_calibration_json,
        "rig_table": cfg.rig_calibration_json,
    }
    calibrations = payload.get("calibrations")
    calibration_keys = {f"{name}_{suffix}" for name in calibration_paths for suffix in ("path", "sha256")}
    if not isinstance(calibrations, Mapping) or set(calibrations) != calibration_keys:
        raise ValueError("v4 commissioning calibration identity keys changed")
    resolved_calibrations: dict[str, dict[str, str]] = {}
    for name, path in calibration_paths.items():
        resolved, digest = _read_only_file_identity(path, label=f"v4 {name} calibration")
        if calibrations.get(f"{name}_path") != str(resolved):
            raise ValueError(f"v4 commissioning {name} calibration path does not match config")
        claimed_digest = _require_physical_evidence_sha256(
            calibrations.get(f"{name}_sha256"),
            label=f"v4 commissioning {name} calibration SHA-256",
        )
        if claimed_digest != digest:
            raise ValueError(f"v4 commissioning {name} calibration SHA-256 does not match file")
        resolved_calibrations[name] = {"path": str(resolved), "sha256": digest}
    if resolved_calibrations["gripper_endpoints"]["sha256"] != cfg.expected_gripper_calibration_sha256:
        raise ValueError("v4 commissioning gripper calibration does not match release binding")

    cameras = payload.get("cameras")
    if not isinstance(cameras, Mapping) or set(cameras) != {"left", "right"}:
        raise ValueError("v4 commissioning must bind exactly the left and right cameras")
    camera_identity = {}
    for arm in ("left", "right"):
        camera = cameras.get(arm)
        camera_keys = {
            "camera_key",
            "stable_device_id",
            "training_viewpoint_verified",
            "training_viewpoint_evidence_path",
            "training_viewpoint_evidence_size_bytes",
            "training_viewpoint_evidence_sha256",
        }
        if not isinstance(camera, Mapping) or set(camera) != camera_keys:
            raise ValueError(f"v4 commissioning {arm} camera evidence keys changed")
        camera_key, stable_id = _configured_camera_stable_id(cfg, arm=arm)
        if camera.get("camera_key") != camera_key or camera.get("stable_device_id") != stable_id:
            raise ValueError(f"v4 commissioning {arm} camera identity does not match config")
        if camera.get("training_viewpoint_verified") is not True:
            raise ValueError(f"v4 commissioning {arm} training viewpoint is not verified")
        viewpoint_evidence = _local_physical_evidence_identity(
            path_value=camera.get("training_viewpoint_evidence_path"),
            size_value=camera.get("training_viewpoint_evidence_size_bytes"),
            sha256_value=camera.get("training_viewpoint_evidence_sha256"),
            label=f"v4 {arm} training-viewpoint evidence",
        )
        camera_identity[arm] = {**dict(camera), "verified_artifact": viewpoint_evidence}

    reset = payload.get("measured_reset")
    reset_keys = {
        "passed",
        "trial_count",
        "joint_tolerance_radians",
        "gripper_tolerance_normalized",
        "left_max_abs_joint_error_radians",
        "right_max_abs_joint_error_radians",
        "left_gripper_abs_error_normalized",
        "right_gripper_abs_error_normalized",
        "measurement_id",
        "evidence_path",
        "evidence_size_bytes",
        "evidence_sha256",
    }
    if not isinstance(reset, Mapping) or set(reset) != reset_keys or reset.get("passed") is not True:
        raise ValueError("v4 commissioning measured-reset evidence is incomplete or not passed")
    trial_count = reset.get("trial_count")
    if isinstance(trial_count, bool) or not isinstance(trial_count, int) or trial_count <= 0:
        raise ValueError("v4 commissioning measured reset trial_count must be positive")
    raw_joint_tolerance = reset.get("joint_tolerance_radians")
    raw_gripper_tolerance = reset.get("gripper_tolerance_normalized")
    if (
        isinstance(raw_joint_tolerance, bool)
        or not isinstance(raw_joint_tolerance, int | float)
        or isinstance(raw_gripper_tolerance, bool)
        or not isinstance(raw_gripper_tolerance, int | float)
    ):
        raise ValueError("v4 commissioning reset tolerances must be numeric")
    joint_tolerance = float(raw_joint_tolerance)
    gripper_tolerance = float(raw_gripper_tolerance)
    if not np.isfinite(joint_tolerance) or not np.isfinite(gripper_tolerance):
        raise ValueError("v4 commissioning reset tolerances must be finite")
    if not np.isclose(
        joint_tolerance, cfg.robot.policy_reset_tolerance, rtol=0.0, atol=0.0
    ) or not np.isclose(
        gripper_tolerance,
        cfg.robot.policy_reset_tolerance,
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("v4 commissioning reset tolerances do not match deployment config")
    for name, tolerance in (
        ("left_max_abs_joint_error_radians", joint_tolerance),
        ("right_max_abs_joint_error_radians", joint_tolerance),
        ("left_gripper_abs_error_normalized", gripper_tolerance),
        ("right_gripper_abs_error_normalized", gripper_tolerance),
    ):
        value = reset.get(name)
        if isinstance(value, bool) or not isinstance(value, int | float) or not np.isfinite(value):
            raise ValueError(f"v4 commissioning measured reset {name} must be finite")
        if value < 0 or value > tolerance:
            raise ValueError(f"v4 commissioning measured reset {name} exceeds its tolerance")
    _nonplaceholder_text(reset.get("measurement_id"), label="v4 measured-reset measurement ID")
    reset_evidence = _local_physical_evidence_identity(
        path_value=reset.get("evidence_path"),
        size_value=reset.get("evidence_size_bytes"),
        sha256_value=reset.get("evidence_sha256"),
        label="v4 measured-reset evidence",
    )
    return _jsonable(
        {
            "path": str(evidence),
            "sha256": evidence_sha256,
            "schema_id": expected_commissioning_schema,
            "commissioning_passed": True,
            "robot": dict(robot),
            "fixed_global_start_anchor": dict(anchor),
            "calibrated_umi_reset_scalars": calibrated_reset_scalars,
            "calibrations": resolved_calibrations,
            "cameras": camera_identity,
            "measured_reset": {**dict(reset), "verified_artifact": reset_evidence},
        }
    )


class UmiPolicyClient(Protocol):
    """The subset of :class:`UmiEeRemoteClient` needed by the controller."""

    def reset(self) -> None: ...

    def predict(self, state: np.ndarray, images: dict[str, np.ndarray], tick: int) -> np.ndarray: ...


class YamRobot(Protocol):
    """The measured-state command boundary used by the controller."""

    def get_observation(self) -> dict[str, Any]: ...

    def send_action(self, action: dict[str, float]) -> Any: ...


CollisionCheck = Callable[[Mapping[str, float]], object]
TelemetrySink = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class YamUmiControllerConfig:
    """Runtime settings that do not encode scene geometry."""

    left_camera_key: str
    right_camera_key: str
    control_hz: float = 30.0
    execution_horizon: int = 15
    policy_image_width: int = 800
    policy_image_height: int = 600
    action_semantics: UmiActionSemantics = EPISODE_START_ABSOLUTE
    measured_progress_gate: bool = False
    progress_position_tolerance_m: float = 3e-3
    progress_orientation_tolerance_rad: float = np.deg2rad(2.0)
    progress_gripper_tolerance: float = 0.02
    max_progress_hold_steps: int = 90
    served_action_horizon: int | None = None
    inference_hold_enabled: bool = False
    max_inference_latency_s: float = 20.0
    max_inference_hold_joint_drift: float = 0.01
    max_inference_hold_gripper_drift: float = 0.002
    max_observation_to_first_dispatch_s: float | None = None
    arm_worker_deadman_s: float | None = None
    worker_dispatch_expiry_margin_s: float | None = None

    def __post_init__(self) -> None:
        if not self.left_camera_key.strip() or not self.right_camera_key.strip():
            raise ValueError("left_camera_key and right_camera_key must be non-empty")
        if self.left_camera_key == self.right_camera_key:
            raise ValueError("left and right policy cameras must be different")
        if self.control_hz <= 0:
            raise ValueError("control_hz must be positive")
        if self.execution_horizon <= 0:
            raise ValueError("execution_horizon must be positive")
        if self.served_action_horizon is not None:
            if self.served_action_horizon <= 0:
                raise ValueError("served_action_horizon must be positive when configured")
            if self.execution_horizon > self.served_action_horizon:
                raise ValueError("execution_horizon cannot exceed served_action_horizon")
            if self.action_semantics in _CURRENT_RELATIVE_R6D_SEMANTICS and (
                self.served_action_horizon != _CURRENT_RELATIVE_SERVED_ACTION_HORIZON
                or self.execution_horizon != _CURRENT_RELATIVE_ACTIVE_EXECUTION_HORIZON
            ):
                raise ValueError(
                    "supervised current-relative controller requires exactly 24 served rows "
                    "and execution_horizon=15"
                )
        if self.policy_image_width <= 0 or self.policy_image_height <= 0:
            raise ValueError("policy image dimensions must be positive")
        if self.measured_progress_gate and self.action_semantics not in _CURRENT_RELATIVE_R6D_SEMANTICS:
            raise ValueError("measured progress gating requires current-relative R6D semantics")
        if self.action_semantics == CURRENT_RELATIVE_R6D_SE3_JAW_DELTA and not self.measured_progress_gate:
            raise ValueError("jaw-delta current-relative R6D semantics require measured_progress_gate=true")
        if self.action_semantics == CURRENT_RELATIVE_R6D_SE3_JAW_DELTA:
            if self.served_action_horizon != _CURRENT_RELATIVE_SERVED_ACTION_HORIZON:
                raise ValueError("jaw-delta semantics require served_action_horizon=24")
            if self.execution_horizon != _CURRENT_RELATIVE_ACTIVE_EXECUTION_HORIZON:
                raise ValueError("jaw-delta semantics require execution_horizon=15")
            if not self.inference_hold_enabled:
                raise ValueError("jaw-delta semantics require inference_hold_enabled=true")
            if self.max_observation_to_first_dispatch_s is None:
                raise ValueError("jaw-delta semantics require a first-dispatch freshness deadline")
            if self.arm_worker_deadman_s is None:
                raise ValueError("jaw-delta semantics require the arm worker deadman contract")
            if self.worker_dispatch_expiry_margin_s is None:
                raise ValueError("jaw-delta semantics require a reviewed worker dispatch expiry margin")
        if (
            self.action_semantics in _CURRENT_RELATIVE_R6D_SEMANTICS
            and self.served_action_horizon is not None
        ):
            if not self.measured_progress_gate:
                raise ValueError(
                    "supervised current-relative controller requires measured_progress_gate=true"
                )
            if not self.inference_hold_enabled:
                raise ValueError(
                    "supervised current-relative controller requires inference_hold_enabled=true"
                )
            if self.max_observation_to_first_dispatch_s is None:
                raise ValueError(
                    "supervised current-relative controller requires a first-dispatch freshness deadline"
                )
            if self.arm_worker_deadman_s is None:
                raise ValueError(
                    "supervised current-relative controller requires the arm worker deadman contract"
                )
            if self.worker_dispatch_expiry_margin_s is None:
                raise ValueError(
                    "supervised current-relative controller requires a reviewed worker dispatch expiry margin"
                )
        for name in (
            "progress_position_tolerance_m",
            "progress_orientation_tolerance_rad",
            "progress_gripper_tolerance",
            "max_inference_latency_s",
            "max_inference_hold_joint_drift",
            "max_inference_hold_gripper_drift",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.progress_gripper_tolerance > 1:
            raise ValueError("progress_gripper_tolerance cannot exceed the normalized gripper range")
        if self.max_progress_hold_steps <= 0:
            raise ValueError("max_progress_hold_steps must be positive")
        if not isinstance(self.inference_hold_enabled, bool):
            raise TypeError("inference_hold_enabled must be a boolean")
        if self.max_observation_to_first_dispatch_s is not None:
            freshness_limit = float(self.max_observation_to_first_dispatch_s)
            if not np.isfinite(freshness_limit) or freshness_limit <= 0:
                raise ValueError("max_observation_to_first_dispatch_s must be finite and positive")
        if self.arm_worker_deadman_s is not None:
            deadman_s = float(self.arm_worker_deadman_s)
            if not np.isfinite(deadman_s) or deadman_s <= 0:
                raise ValueError("arm_worker_deadman_s must be finite and positive")
            if 1.0 / self.control_hz > deadman_s:
                raise ValueError("control period cannot exceed arm_worker_deadman_s")
        if self.worker_dispatch_expiry_margin_s is not None:
            margin_s = float(self.worker_dispatch_expiry_margin_s)
            if not np.isfinite(margin_s) or margin_s <= 0:
                raise ValueError("worker_dispatch_expiry_margin_s must be finite and positive")


def _to_policy_rgb(image: Any, *, width: int, height: int, name: str) -> np.ndarray:
    """Validate and resize one hardware RGB frame without modifying its content."""

    from PIL import Image

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"{name} must be an HWC RGB image with three channels")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite pixels")
    if array.dtype != np.uint8:
        scale = 255.0 if float(np.max(array, initial=0.0)) <= 1.0 else 1.0
        array = np.clip(np.rint(array * scale), 0, 255).astype(np.uint8)
    if array.shape[:2] != (height, width):
        array = np.asarray(
            Image.fromarray(array, mode="RGB").resize(
                (width, height),
                resample=Image.Resampling.BILINEAR,
            )
        )
    return np.ascontiguousarray(array)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


@dataclass
class YamUmiRemoteController:
    """Measured-state UMI→YAM chunk executor with optional safe inference HOLD.

    ``collision_check`` is an optional shared-world gate. Independent per-arm IK
    remains available before the rig/table transforms needed by that gate exist.
    """

    client: UmiPolicyClient
    adapter: YamUmiEeAdapter
    config: YamUmiControllerConfig
    collision_check: CollisionCheck | None
    remote_model_identity: Mapping[str, Any] | None = None
    remote_session_contract: Mapping[str, Any] | None = None
    deployment_release_identity: Mapping[str, Any] | None = None
    telemetry_sink: TelemetrySink | None = None
    monotonic: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    last_records: list[dict[str, Any]] = field(default_factory=list, init=False)
    last_start_record: dict[str, Any] | None = field(default=None, init=False)
    last_inference_hold_record: dict[str, Any] | None = field(default=None, init=False)
    last_freshness_record: dict[str, Any] | None = field(default=None, init=False)
    _last_command_dispatch_at_s: float | None = field(default=None, init=False)
    _remote_session_reset_prepared: bool = field(default=False, init=False)

    def reset_remote_session_while_disarmed(self) -> None:
        """Reset recurrent policy state before any robot is armed.

        Supervised current-relative execution consumes this one-shot marker in
        :meth:`run_episode`; a caller cannot silently move the reset back inside
        the armed interval.
        """

        self.client.reset()
        self._remote_session_reset_prepared = True

    def assert_hardware_ready(self) -> None:
        self.adapter.assert_hardware_ready()
        if self.adapter.action_semantics != self.config.action_semantics:
            raise RuntimeError(
                "controller and adapter action semantics differ: "
                f"{self.config.action_semantics!r} != {self.adapter.action_semantics!r}"
            )

    def policy_images(self, observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
        missing = [
            key
            for key in (self.config.left_camera_key, self.config.right_camera_key)
            if key not in observation
        ]
        if missing:
            raise KeyError(f"observation is missing wrist cameras: {missing}")
        return {
            "umi1": _to_policy_rgb(
                observation[self.config.left_camera_key],
                width=self.config.policy_image_width,
                height=self.config.policy_image_height,
                name=self.config.left_camera_key,
            ),
            "umi2": _to_policy_rgb(
                observation[self.config.right_camera_key],
                width=self.config.policy_image_width,
                height=self.config.policy_image_height,
                name=self.config.right_camera_key,
            ),
        }

    def validate_observation(self, observation: Mapping[str, Any]) -> None:
        missing = [key for key in YAM_SCALAR_KEYS if key not in observation]
        if missing:
            raise KeyError(f"observation is missing YAM joint state: {missing}")
        self.policy_images(observation)

    def run_episode(self, robot: YamRobot, *, max_steps: int) -> list[dict[str, Any]]:
        """Execute up to ``max_steps`` commands from repeated policy chunks."""

        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.assert_hardware_ready()
        guarded_dispatch = getattr(robot, "send_action_from_measured_state", None)
        supervised_current_relative = (
            self.config.action_semantics in _CURRENT_RELATIVE_R6D_SEMANTICS
            and self.config.served_action_horizon is not None
        )
        if supervised_current_relative and not callable(guarded_dispatch):
            raise RuntimeError(
                "supervised current-relative execution requires BiYAM's guarded measured-state dispatch"
            )
        if supervised_current_relative and not self._remote_session_reset_prepared:
            raise RuntimeError(
                "supervised current-relative execution requires reset_remote_session_while_disarmed()"
            )
        # This precedes the state+camera read and is therefore a conservative
        # observation-age origin even when camera capture blocks internally.
        initial_capture_started_at_s = self.monotonic()
        initial_observation = robot.get_observation()
        self.validate_observation(initial_observation)
        self.adapter.capture_episode_start(initial_observation)
        if supervised_current_relative:
            self._remote_session_reset_prepared = False
        else:
            # Preserve the legacy episode-start path exactly: its policy reset is
            # still owned by run_episode because it predates supervised hardware.
            self.client.reset()
        self.last_records.clear()
        self.last_inference_hold_record = None
        self.last_freshness_record = None
        self._last_command_dispatch_at_s = None
        self.last_start_record = self._start_record(robot, initial_observation)
        if self.telemetry_sink is not None:
            self.telemetry_sink(self.last_start_record)

        tick = 0
        chunk_sequence = 0
        inference_observation = initial_observation
        inference_observation_capture_started_at_s = initial_capture_started_at_s
        previous_tick_observation: dict[str, Any] | None = None
        while tick < max_steps:
            # State and pixels are from one hardware observation. Current-relative
            # chunks must be resolved against this exact snapshot, once.
            query_capture_started_at_s = (
                inference_observation_capture_started_at_s
                if self.config.max_observation_to_first_dispatch_s is not None
                else None
            )
            policy_state = self.adapter.observation_to_policy_state(
                inference_observation,
                previous_observation=previous_tick_observation,
            )
            policy_images = self.policy_images(inference_observation)
            if self.config.inference_hold_enabled:
                raw_chunk, inference_duration = self._predict_while_holding(
                    robot,
                    policy_state=policy_state,
                    policy_images=policy_images,
                    tick=tick,
                    query_observation=inference_observation,
                )
            else:
                inference_started = self.monotonic()
                raw_chunk = np.asarray(
                    self.client.predict(policy_state, policy_images, tick),
                    dtype=np.float32,
                )
                inference_duration = self.monotonic() - inference_started
            if (
                self.config.served_action_horizon is not None
                and len(raw_chunk) != self.config.served_action_horizon
            ):
                raise RuntimeError(
                    f"policy returned {len(raw_chunk)} rows; expected exactly "
                    f"{self.config.served_action_horizon} served rows"
                )
            resolved_chunk = self.adapter.resolve_action_chunk(raw_chunk, inference_observation)
            if resolved_chunk.ndim != 2 or resolved_chunk.shape[1] != 14:
                raise RuntimeError(f"policy returned shape {resolved_chunk.shape}; expected (horizon, 14)")
            if (
                self.config.served_action_horizon is not None
                and len(resolved_chunk) != self.config.served_action_horizon
            ):
                raise RuntimeError(
                    f"resolved policy chunk has {len(resolved_chunk)} rows; expected exactly "
                    f"{self.config.served_action_horizon} served rows"
                )
            rows_to_execute = min(
                self.config.execution_horizon,
                len(resolved_chunk),
                max_steps - tick,
            )
            if rows_to_execute <= 0:
                raise RuntimeError("policy returned an empty action chunk")

            for row_index in range(rows_to_execute):
                waypoint_start_observation = inference_observation
                measured_before = (
                    inference_observation
                    if row_index == 0 and supervised_current_relative
                    else robot.get_observation()
                    if row_index == 0
                    else inference_observation
                )
                progress_hold_index = 0
                while True:
                    control_started = self.monotonic()
                    command: dict[str, float] | None = None
                    desired_grippers: dict[str, float] | None = None
                    arms_settled_before: bool | None = None
                    freshness_record = None
                    first_active_dispatch = row_index == 0 and progress_hold_index == 0
                    query_anchor_observation = inference_observation
                    resolved_row = resolved_chunk[row_index]

                    def resolve_from_measured_state(
                        dispatch_observation: Mapping[str, Any],
                        *,
                        _first_active_dispatch: bool = first_active_dispatch,
                        _query_anchor_observation: Mapping[str, Any] = query_anchor_observation,
                        _resolved_row: np.ndarray = resolved_row,
                    ) -> dict[str, float]:
                        nonlocal command, desired_grippers, arms_settled_before, measured_before
                        measured_before = dict(dispatch_observation)
                        if _first_active_dispatch and supervised_current_relative:
                            self._validate_final_dispatch_anchor(
                                query_observation=_query_anchor_observation,
                                dispatch_observation=measured_before,
                            )
                        command = self.adapter.action_row_to_joint_command(
                            _resolved_row,
                            measured_before,
                        )
                        if self.config.measured_progress_gate:
                            desired_grippers = self.adapter.current_relative_gripper_targets(_resolved_row)
                            residuals_before = self.adapter.measured_target_residuals(measured_before)
                            arms_settled_before = self._arms_settled(residuals_before)
                            if not arms_settled_before:
                                # Keep grasp phase synchronized with Cartesian progress.
                                # The source gripper row is released only after both
                                # measured jaw TCPs reach this waypoint.
                                command[LEFT_GRIPPER_KEY] = float(measured_before[LEFT_GRIPPER_KEY])
                                command[RIGHT_GRIPPER_KEY] = float(measured_before[RIGHT_GRIPPER_KEY])
                        if self.collision_check is not None:
                            self.collision_check(command)
                        return command

                    if first_active_dispatch and supervised_current_relative:
                        assert callable(guarded_dispatch)
                        assert query_capture_started_at_s is not None
                        assert self.config.max_observation_to_first_dispatch_s is not None
                        assert self.config.worker_dispatch_expiry_margin_s is not None
                        expires_at_monotonic_ns = int(
                            (query_capture_started_at_s + self.config.max_observation_to_first_dispatch_s)
                            * 1e9
                        )
                        try:
                            applied = guarded_dispatch(
                                resolve_from_measured_state,
                                expires_at_monotonic_ns=expires_at_monotonic_ns,
                                required_expiry_margin_ns=int(
                                    self.config.worker_dispatch_expiry_margin_s * 1e9
                                ),
                            )
                        except RuntimeError as exc:
                            dispatch_timing = getattr(robot, "last_dispatch_timing", None)
                            if self._is_freshness_dispatch_failure(exc, dispatch_timing):
                                self._record_worker_dispatch_freshness(
                                    chunk_sequence=chunk_sequence,
                                    tick=tick,
                                    query_capture_started_at_s=query_capture_started_at_s,
                                    expires_at_monotonic_ns=expires_at_monotonic_ns,
                                    dispatch_timing=dispatch_timing,
                                    expected_dispatch_observation=measured_before,
                                    passed=False,
                                    failure=(
                                        "reviewed_dispatch_margin_unavailable"
                                        if "required reviewed worker dispatch margin" in str(exc)
                                        else "deadline_expired_at_worker_boundary"
                                    ),
                                )
                                age_s = self.last_freshness_record["conservative_dispatch_age_s"]
                                limit_s = self.config.max_observation_to_first_dispatch_s
                                raise RuntimeError(
                                    "stale policy chunk rejected before a safe first ACTIVE worker "
                                    f"dispatch: conservative local age {age_s:.6f}s, freshness limit "
                                    f"{limit_s:.6f}s"
                                ) from exc
                            raise
                        dispatch_timing = getattr(robot, "last_dispatch_timing", None)
                        command_dispatched_at_s, freshness_record = self._record_worker_dispatch_freshness(
                            chunk_sequence=chunk_sequence,
                            tick=tick,
                            query_capture_started_at_s=query_capture_started_at_s,
                            expires_at_monotonic_ns=expires_at_monotonic_ns,
                            dispatch_timing=dispatch_timing,
                            expected_dispatch_observation=measured_before,
                            passed=True,
                        )
                    else:
                        command = resolve_from_measured_state(measured_before)
                        if (
                            first_active_dispatch
                            and self.config.max_observation_to_first_dispatch_s is not None
                        ):
                            command_dispatched_at_s, freshness_record = (
                                self._check_first_active_dispatch_freshness(
                                    chunk_sequence=chunk_sequence,
                                    tick=tick,
                                    query_capture_started_at_s=query_capture_started_at_s,
                                )
                            )
                        else:
                            command_dispatched_at_s = self.monotonic()
                        applied = robot.send_action(command)
                    assert command is not None
                    if freshness_record is not None and self.telemetry_sink is not None:
                        self.telemetry_sink(freshness_record)
                    dispatch_interval_s = (
                        None
                        if self._last_command_dispatch_at_s is None
                        else command_dispatched_at_s - self._last_command_dispatch_at_s
                    )
                    self._last_command_dispatch_at_s = command_dispatched_at_s
                    driver_acknowledged_at_s = self.monotonic()
                    elapsed = self.monotonic() - control_started
                    self.sleep(max(0.0, 1.0 / self.config.control_hz - elapsed))
                    # send_action() only acknowledges an accepted target. This fresh
                    # read is the first post-dispatch evidence of physical motion.
                    measurement_read_started_at_s = self.monotonic()
                    subsequent_measured = robot.get_observation()
                    measurement_received_at_s = self.monotonic()

                    measured_residuals: dict[str, tuple[float, float]] | None = None
                    measured_gripper_errors: dict[str, float] | None = None
                    waypoint_advanced = True
                    if self.config.measured_progress_gate:
                        assert desired_grippers is not None
                        measured_residuals = self.adapter.measured_target_residuals(subsequent_measured)
                        measured_gripper_errors = {
                            "left": abs(
                                float(subsequent_measured[LEFT_GRIPPER_KEY]) - desired_grippers["left"]
                            ),
                            "right": abs(
                                float(subsequent_measured[RIGHT_GRIPPER_KEY]) - desired_grippers["right"]
                            ),
                        }
                        waypoint_advanced = self._arms_settled(measured_residuals) and all(
                            error <= self.config.progress_gripper_tolerance
                            for error in measured_gripper_errors.values()
                        )

                    record = self._record(
                        tick=tick,
                        chunk_sequence=chunk_sequence,
                        chunk_row=row_index,
                        inference_duration_s=(
                            inference_duration if row_index == 0 and progress_hold_index == 0 else None
                        ),
                        requested_target=command,
                        driver_applied_target=applied,
                        measured_before=measured_before,
                        subsequent_measured=subsequent_measured,
                        progress_hold_index=progress_hold_index,
                        waypoint_advanced=waypoint_advanced,
                        arms_settled_before=arms_settled_before,
                        measured_target_residuals=measured_residuals,
                        measured_gripper_errors=measured_gripper_errors,
                        command_dispatched_at_s=command_dispatched_at_s,
                        driver_acknowledged_at_s=driver_acknowledged_at_s,
                        measurement_read_started_at_s=measurement_read_started_at_s,
                        measurement_received_at_s=measurement_received_at_s,
                        dispatch_interval_s=dispatch_interval_s,
                    )
                    self.last_records.append(record)
                    if self.telemetry_sink is not None:
                        self.telemetry_sink(record)
                    if waypoint_advanced:
                        break
                    progress_hold_index += 1
                    if progress_hold_index >= self.config.max_progress_hold_steps:
                        raise RuntimeError(
                            "measured-progress timeout while holding "
                            f"chunk {chunk_sequence} row {row_index} after "
                            f"{self.config.max_progress_hold_steps} dispatches"
                        )
                    measured_before = subsequent_measured

                tick += 1
                # A waypoint may need several driver calls. Preserve the previous
                # completed waypoint for the model's one-step history, rather than
                # leaking the final intermediate hold observation into that slot.
                previous_tick_observation = waypoint_start_observation
                inference_observation = subsequent_measured
                inference_observation_capture_started_at_s = measurement_read_started_at_s
            chunk_sequence += 1
        return list(self.last_records)

    def _predict_while_holding(
        self,
        robot: YamRobot,
        *,
        policy_state: np.ndarray,
        policy_images: dict[str, np.ndarray],
        tick: int,
        query_observation: Mapping[str, Any],
    ) -> tuple[np.ndarray, float]:
        """Keep one unchanged measured target alive while the inference RPC blocks.

        The daemon thread performs only the policy RPC. Robot reads, collision
        checks, and HOLD dispatches stay serialized on this controller thread.
        A timeout, RPC error, heartbeat error, or measured drift rejects the
        result; this method never re-arms the robot.
        """

        hold_target = {key: float(query_observation[key]) for key in YAM_SCALAR_KEYS}
        if not np.isfinite(tuple(hold_target.values())).all():
            raise ValueError("inference HOLD target contains non-finite measured state")

        dispatch_times_s: list[float] = []
        dispatch_ack_times_s: list[float] = []
        conservative_heartbeat_gaps_s: list[float] = []
        previous_dispatch_started_at_s = self._last_command_dispatch_at_s

        class _HeartbeatGapError(RuntimeError):
            pass

        def record_outcome(
            *,
            passed: bool,
            duration: float,
            failure: str | None = None,
            joint_drift: float | None = None,
            gripper_drift: float | None = None,
        ) -> None:
            gaps = np.diff(dispatch_times_s)
            self.last_inference_hold_record = _jsonable(
                {
                    "event": "inference_hold",
                    "tick": tick,
                    "passed": passed,
                    "failure": failure,
                    "target": hold_target,
                    "dispatch_count": len(dispatch_times_s),
                    "dispatch_monotonic_s": dispatch_times_s,
                    "dispatch_ack_monotonic_s": dispatch_ack_times_s,
                    "max_dispatch_gap_s": None if len(gaps) == 0 else float(np.max(gaps)),
                    "conservative_heartbeat_gaps_s": conservative_heartbeat_gaps_s,
                    "max_conservative_heartbeat_gap_s": (
                        None if not conservative_heartbeat_gaps_s else max(conservative_heartbeat_gaps_s)
                    ),
                    "arm_worker_deadman_s": self.config.arm_worker_deadman_s,
                    "heartbeat_boundary_rule": "gap <= deadman passes; gap > deadman fails",
                    "inference_duration_s": duration,
                    "max_joint_drift": joint_drift,
                    "max_gripper_drift": gripper_drift,
                    "result_fresh": passed,
                    "robot_calls_on_controller_thread": True,
                    "rearm_attempted": False,
                }
            )
            if self.telemetry_sink is not None:
                self.telemetry_sink(self.last_inference_hold_record)

        def check_deadman_gap(observed_at_s: float) -> None:
            if previous_dispatch_started_at_s is None or self.config.arm_worker_deadman_s is None:
                return
            gap_s = observed_at_s - previous_dispatch_started_at_s
            conservative_heartbeat_gaps_s.append(gap_s)
            if gap_s < 0:
                raise _HeartbeatGapError("controller-local monotonic clock regressed during HOLD")
            if gap_s > self.config.arm_worker_deadman_s:
                raise _HeartbeatGapError(
                    "inference HOLD heartbeat gap exceeded the arm worker deadman: "
                    f"{gap_s:.6f}s > {self.config.arm_worker_deadman_s:.6f}s"
                )

        def dispatch_hold() -> None:
            nonlocal previous_dispatch_started_at_s
            if self.collision_check is not None:
                self.collision_check(hold_target)
            dispatched_at_s = self.monotonic()
            robot.send_action(hold_target)
            acknowledged_at_s = self.monotonic()
            check_deadman_gap(acknowledged_at_s)
            dispatch_times_s.append(dispatched_at_s)
            dispatch_ack_times_s.append(acknowledged_at_s)
            previous_dispatch_started_at_s = dispatched_at_s
            self._last_command_dispatch_at_s = dispatched_at_s

        try:
            # This is a measured no-motion heartbeat, not the first ACTIVE row.
            dispatch_hold()
        except BaseException as exc:
            failure = (
                "heartbeat_gap_exceeded_worker_deadman"
                if isinstance(exc, _HeartbeatGapError)
                else "initial_hold_dispatch_error"
            )
            record_outcome(passed=False, duration=0.0, failure=failure)
            raise RuntimeError("failed to dispatch the measured inference HOLD target") from exc

        result_queue: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

        def infer() -> None:
            try:
                result = self.client.predict(policy_state, policy_images, tick)
            except BaseException as exc:
                result_queue.put((False, exc))
            else:
                result_queue.put((True, result))

        inference_started_at_s = self.monotonic()
        threading.Thread(target=infer, name=f"umi-inference-{tick}", daemon=True).start()
        control_period_s = 1.0 / self.config.control_hz
        result: tuple[bool, object] | None = None
        while result is None:
            try:
                result = result_queue.get_nowait()
                break
            except queue.Empty:
                pass
            elapsed_s = self.monotonic() - inference_started_at_s
            if elapsed_s >= self.config.max_inference_latency_s:
                record_outcome(passed=False, duration=elapsed_s, failure="timeout")
                raise RuntimeError(
                    "policy inference exceeded the supervised HOLD timeout "
                    f"of {self.config.max_inference_latency_s:.6f}s"
                )
            self.sleep(min(control_period_s, self.config.max_inference_latency_s - elapsed_s))
            try:
                result = result_queue.get_nowait()
                continue
            except queue.Empty:
                pass
            try:
                dispatch_hold()
            except BaseException as exc:
                elapsed_s = self.monotonic() - inference_started_at_s
                failure = (
                    "heartbeat_gap_exceeded_worker_deadman"
                    if isinstance(exc, _HeartbeatGapError)
                    else "hold_dispatch_error"
                )
                record_outcome(passed=False, duration=elapsed_s, failure=failure)
                raise RuntimeError("failed to maintain the measured inference HOLD target") from exc

        inference_duration_s = self.monotonic() - inference_started_at_s
        succeeded, payload = result
        if not succeeded:
            assert isinstance(payload, BaseException)
            record_outcome(passed=False, duration=inference_duration_s, failure="rpc_error")
            raise RuntimeError("policy inference failed while maintaining measured HOLD") from payload
        if inference_duration_s > self.config.max_inference_latency_s:
            record_outcome(
                passed=False,
                duration=inference_duration_s,
                failure="stale_timeout_result",
            )
            raise RuntimeError("policy inference result became stale before HOLD validation")

        try:
            check_deadman_gap(self.monotonic())
        except _HeartbeatGapError as exc:
            record_outcome(
                passed=False,
                duration=inference_duration_s,
                failure="heartbeat_gap_exceeded_worker_deadman",
            )
            raise RuntimeError("policy result arrived after the HOLD heartbeat safety window") from exc

        fresh_observation = robot.get_observation()
        self.validate_observation(fresh_observation)
        try:
            check_deadman_gap(self.monotonic())
        except _HeartbeatGapError as exc:
            record_outcome(
                passed=False,
                duration=inference_duration_s,
                failure="heartbeat_gap_exceeded_worker_deadman",
            )
            raise RuntimeError("HOLD validation exceeded the arm worker deadman window") from exc
        joint_drift = max(
            abs(float(fresh_observation[key]) - hold_target[key])
            for key in YAM_SCALAR_KEYS
            if "gripper" not in key
        )
        gripper_drift = max(
            abs(float(fresh_observation[key]) - hold_target[key])
            for key in (LEFT_GRIPPER_KEY, RIGHT_GRIPPER_KEY)
        )
        if joint_drift > self.config.max_inference_hold_joint_drift:
            record_outcome(
                passed=False,
                duration=inference_duration_s,
                failure="joint_drift",
                joint_drift=joint_drift,
                gripper_drift=gripper_drift,
            )
            raise RuntimeError(
                "policy inference result is stale: measured joint drift during HOLD was "
                f"{joint_drift:.6f}, above {self.config.max_inference_hold_joint_drift:.6f}"
            )
        if gripper_drift > self.config.max_inference_hold_gripper_drift:
            record_outcome(
                passed=False,
                duration=inference_duration_s,
                failure="gripper_drift",
                joint_drift=joint_drift,
                gripper_drift=gripper_drift,
            )
            raise RuntimeError(
                "policy inference result is stale: measured gripper drift during HOLD was "
                f"{gripper_drift:.6f}, above {self.config.max_inference_hold_gripper_drift:.6f}"
            )

        record_outcome(
            passed=True,
            duration=inference_duration_s,
            joint_drift=joint_drift,
            gripper_drift=gripper_drift,
        )
        return np.asarray(payload, dtype=np.float32), inference_duration_s

    def _validate_final_dispatch_anchor(
        self,
        *,
        query_observation: Mapping[str, Any],
        dispatch_observation: Mapping[str, Any],
    ) -> None:
        """Reject drift in the driver's exact final measured ACTIVE state."""

        joint_drift = max(
            abs(float(dispatch_observation[key]) - float(query_observation[key]))
            for key in YAM_SCALAR_KEYS
            if "gripper" not in key
        )
        gripper_drift = max(
            abs(float(dispatch_observation[key]) - float(query_observation[key]))
            for key in (LEFT_GRIPPER_KEY, RIGHT_GRIPPER_KEY)
        )
        if joint_drift > self.config.max_inference_hold_joint_drift:
            raise RuntimeError(
                "final measured ACTIVE state drifted from its policy query anchor: joint drift "
                f"{joint_drift:.6f} exceeds {self.config.max_inference_hold_joint_drift:.6f}"
            )
        if gripper_drift > self.config.max_inference_hold_gripper_drift:
            raise RuntimeError(
                "final measured ACTIVE state drifted from its policy query anchor: gripper drift "
                f"{gripper_drift:.6f} exceeds {self.config.max_inference_hold_gripper_drift:.6f}"
            )

    @staticmethod
    def _is_freshness_dispatch_failure(exc: RuntimeError, timing: object) -> bool:
        message = str(exc)
        if "freshness deadline" in message or "command_expired_before_worker_dispatch" in message:
            return True
        if not isinstance(timing, Mapping):
            return False
        if timing.get("worker_rejected_monotonic_ns"):
            return True
        scheduled = timing.get("scheduled_execute_at_monotonic_ns")
        expires = timing.get("expires_at_monotonic_ns")
        return isinstance(scheduled, int) and isinstance(expires, int) and scheduled > expires

    def _record_worker_dispatch_freshness(
        self,
        *,
        chunk_sequence: int,
        tick: int,
        query_capture_started_at_s: float,
        expires_at_monotonic_ns: int,
        dispatch_timing: object,
        expected_dispatch_observation: Mapping[str, Any],
        passed: bool,
        failure: str | None = None,
    ) -> tuple[float, dict[str, Any]]:
        """Record the actual worker boundary and a conservative local bound."""

        if not isinstance(dispatch_timing, Mapping):
            raise RuntimeError("BiYAM omitted required guarded-dispatch timing telemetry")
        expiry_from_driver = dispatch_timing.get("expires_at_monotonic_ns")
        if expiry_from_driver != expires_at_monotonic_ns:
            raise RuntimeError("BiYAM guarded-dispatch expiry telemetry does not match the request")
        if dispatch_timing.get("deadline_enforced_in_worker") is not True:
            raise RuntimeError("BiYAM did not enforce the freshness deadline in its arm workers")
        required_margin_ns = dispatch_timing.get("required_expiry_margin_ns")
        configured_margin_s = self.config.worker_dispatch_expiry_margin_s
        if configured_margin_s is None:
            raise RuntimeError("guarded dispatch has no configured worker expiry margin")
        expected_margin_ns = int(configured_margin_s * 1e9)
        if required_margin_ns != expected_margin_ns:
            raise RuntimeError("BiYAM did not preserve the reviewed worker dispatch expiry margin")
        state_before_raw = dispatch_timing.get("state_before_dispatch")
        state_before = np.asarray(state_before_raw, dtype=np.float64)
        expected_state = np.asarray(
            [float(expected_dispatch_observation[key]) for key in YAM_SCALAR_KEYS],
            dtype=np.float64,
        )
        if (
            state_before.shape != expected_state.shape
            or not np.isfinite(state_before).all()
            or not np.array_equal(state_before, expected_state)
        ):
            raise RuntimeError(
                "BiYAM dispatch timing did not preserve the exact final measured resolver state"
            )

        def seconds(name: str) -> float | None:
            value = dispatch_timing.get(name)
            if value is None:
                return None
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RuntimeError(f"BiYAM returned invalid {name} timing telemetry")
            return value / 1e9

        controller_boundary_s = seconds("controller_dispatch_boundary_monotonic_ns")
        scheduled_s = seconds("scheduled_execute_at_monotonic_ns")
        worker_applied_raw = dispatch_timing.get("worker_dispatch_started_monotonic_ns")
        worker_acknowledged_raw = dispatch_timing.get("worker_driver_acknowledged_monotonic_ns")
        worker_rejected_raw = dispatch_timing.get("worker_rejected_monotonic_ns")
        worker_applied_s: dict[str, float] | None = None
        worker_acknowledged_s: dict[str, float] | None = None
        worker_rejected_s: dict[str, float] | None = None
        if worker_applied_raw:
            if not isinstance(worker_applied_raw, Mapping) or not set(worker_applied_raw) <= {
                "left",
                "right",
            }:
                raise RuntimeError("BiYAM returned invalid worker application timestamp sides")
            worker_applied_s = {}
            for side, value in worker_applied_raw.items():
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise RuntimeError(f"BiYAM returned invalid {side} worker application timestamp")
                worker_applied_s[str(side)] = value / 1e9
        if worker_acknowledged_raw:
            if not isinstance(worker_acknowledged_raw, Mapping) or not set(worker_acknowledged_raw) <= {
                "left",
                "right",
            }:
                raise RuntimeError("BiYAM returned invalid worker driver acknowledgement sides")
            worker_acknowledged_s = {}
            for side, value in worker_acknowledged_raw.items():
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise RuntimeError(
                        f"BiYAM returned invalid {side} worker driver acknowledgement timestamp"
                    )
                worker_acknowledged_s[str(side)] = value / 1e9
        if worker_rejected_raw is not None:
            if not isinstance(worker_rejected_raw, Mapping) or not worker_rejected_raw:
                raise RuntimeError("BiYAM returned invalid worker rejection timing telemetry")
            worker_rejected_s = {}
            for side, value in worker_rejected_raw.items():
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise RuntimeError(f"BiYAM returned invalid {side} worker rejection timestamp")
                worker_rejected_s[str(side)] = value / 1e9

        recorded_at_s = self.monotonic()
        deadline_s = expires_at_monotonic_ns / 1e9
        actual_worker_dispatch_s = None if worker_applied_s is None else max(worker_applied_s.values())
        if passed:
            if worker_applied_s is None or set(worker_applied_s) != {"left", "right"}:
                raise RuntimeError("BiYAM did not prove both arm workers applied the guarded command")
            if worker_acknowledged_s is None or set(worker_acknowledged_s) != {"left", "right"}:
                raise RuntimeError("BiYAM did not prove both arm drivers acknowledged the guarded command")
            if worker_rejected_s is not None:
                raise RuntimeError("BiYAM reported a rejection after both arm workers applied")
            if actual_worker_dispatch_s is None or actual_worker_dispatch_s > deadline_s:
                raise RuntimeError("BiYAM reported an ACTIVE worker dispatch after its local deadline")
            if any(timestamp > deadline_s for timestamp in worker_acknowledged_s.values()):
                raise RuntimeError("BiYAM reported an arm driver acknowledgement after expiry")
            if actual_worker_dispatch_s < query_capture_started_at_s:
                raise RuntimeError("worker dispatch timestamp predates the local observation capture")
            actual_or_failure_s = actual_worker_dispatch_s
        else:
            if worker_applied_s is not None and any(
                timestamp > deadline_s for timestamp in worker_applied_s.values()
            ):
                raise RuntimeError("expired guarded dispatch reported an ACTIVE application after expiry")
            candidates = [recorded_at_s]
            if controller_boundary_s is not None:
                candidates.append(controller_boundary_s)
            if scheduled_s is not None:
                candidates.append(scheduled_s)
            if worker_rejected_s is not None:
                candidates.extend(worker_rejected_s.values())
            actual_or_failure_s = max(candidates)

        actual_age_s = (
            None
            if actual_worker_dispatch_s is None
            else actual_worker_dispatch_s - query_capture_started_at_s
        )
        conservative_dispatch_s = max(recorded_at_s, actual_or_failure_s)
        conservative_age_s = conservative_dispatch_s - query_capture_started_at_s
        self.last_freshness_record = _jsonable(
            {
                "event": "observation_to_first_active_worker_dispatch_freshness",
                "chunk_sequence": chunk_sequence,
                "tick": tick,
                "passed": passed,
                "failure": failure,
                "clock": "controller/worker-local time.monotonic; shared host clock base",
                "remote_clock_compared": False,
                "query_capture_started_at_s": query_capture_started_at_s,
                "deadline_s": deadline_s,
                "controller_dispatch_boundary_s": controller_boundary_s,
                "scheduled_worker_dispatch_s": scheduled_s,
                "worker_applied_s": worker_applied_s,
                "worker_dispatch_started_s": worker_applied_s,
                "worker_driver_acknowledged_s": worker_acknowledged_s,
                "worker_rejected_s": worker_rejected_s,
                "partial_apply_detected": (
                    worker_applied_s is not None and set(worker_applied_s) != {"left", "right"}
                ),
                "any_apply_before_failure": bool(not passed and worker_applied_s),
                "two_arm_dispatch_atomic": False,
                "required_reviewed_dispatch_margin_s": expected_margin_ns / 1e9,
                "actual_worker_dispatch_s": actual_worker_dispatch_s,
                "actual_dispatch_age_s": actual_age_s,
                "conservative_dispatch_observed_at_s": conservative_dispatch_s,
                "conservative_dispatch_age_s": conservative_age_s,
                "max_observation_to_first_dispatch_s": (self.config.max_observation_to_first_dispatch_s),
                "boundary_rule": "worker application <= deadline passes; later application fails",
                "state_before_dispatch": dispatch_timing.get("state_before_dispatch"),
            }
        )
        if not passed and self.telemetry_sink is not None:
            self.telemetry_sink(self.last_freshness_record)
        return actual_or_failure_s, self.last_freshness_record

    def _check_first_active_dispatch_freshness(
        self,
        *,
        chunk_sequence: int,
        tick: int,
        query_capture_started_at_s: float | None,
    ) -> tuple[float, dict[str, Any]]:
        """Reject a stale chunk immediately before its first ACTIVE dispatch.

        Only controller-local monotonic timestamps participate. The exact
        boundary is inclusive: ``age == limit`` is accepted and only
        ``age > limit`` is rejected.
        """

        limit_s = self.config.max_observation_to_first_dispatch_s
        if limit_s is None:
            raise RuntimeError("first-dispatch freshness check called without a configured deadline")
        if query_capture_started_at_s is None:
            raise RuntimeError("freshness deadline is configured without a local query timestamp")
        # Keep this timestamp adjacent to the caller's ACTIVE send_action. Passed
        # telemetry is emitted only after that send, so telemetry I/O cannot age a
        # just-validated command past its deadline.
        first_dispatch_checked_at_s = self.monotonic()
        age_s = first_dispatch_checked_at_s - query_capture_started_at_s
        passed = age_s >= 0 and age_s <= limit_s
        failure = None if passed else ("local_clock_regression" if age_s < 0 else "deadline_expired")
        self.last_freshness_record = _jsonable(
            {
                "event": "observation_to_first_active_dispatch_freshness",
                "chunk_sequence": chunk_sequence,
                "tick": tick,
                "passed": passed,
                "failure": failure,
                "clock": "controller-local time.monotonic",
                "remote_clock_compared": False,
                "query_capture_started_at_s": query_capture_started_at_s,
                "first_active_dispatch_checked_at_s": first_dispatch_checked_at_s,
                "observed_age_s": age_s,
                "max_observation_to_first_dispatch_s": limit_s,
                "boundary_rule": "age <= limit passes; age > limit fails",
            }
        )
        if age_s < 0:
            if self.telemetry_sink is not None:
                self.telemetry_sink(self.last_freshness_record)
            raise RuntimeError(
                "controller-local monotonic clock regressed while checking first ACTIVE dispatch: "
                f"observed age {age_s:.6f}s"
            )
        if not passed:
            if self.telemetry_sink is not None:
                self.telemetry_sink(self.last_freshness_record)
            raise RuntimeError(
                "stale policy chunk before first ACTIVE dispatch: local observation age "
                f"{age_s:.6f}s exceeds max_observation_to_first_dispatch_s={limit_s:.6f}s"
            )
        return first_dispatch_checked_at_s, self.last_freshness_record

    def _arms_settled(self, residuals: Mapping[str, tuple[float, float]]) -> bool:
        if set(residuals) != {"left", "right"}:
            raise RuntimeError("measured target residuals must contain exactly left and right")
        return all(
            position <= self.config.progress_position_tolerance_m
            and orientation <= self.config.progress_orientation_tolerance_rad
            for position, orientation in residuals.values()
        )

    def _record(
        self,
        *,
        tick: int,
        chunk_sequence: int,
        chunk_row: int,
        inference_duration_s: float | None,
        requested_target: Mapping[str, float],
        driver_applied_target: Any,
        measured_before: Mapping[str, Any],
        subsequent_measured: Mapping[str, Any],
        progress_hold_index: int,
        waypoint_advanced: bool,
        arms_settled_before: bool | None,
        measured_target_residuals: Mapping[str, tuple[float, float]] | None,
        measured_gripper_errors: Mapping[str, float] | None,
        command_dispatched_at_s: float,
        driver_acknowledged_at_s: float,
        measurement_read_started_at_s: float,
        measurement_received_at_s: float,
        dispatch_interval_s: float | None,
    ) -> dict[str, Any]:
        diagnostics = {}
        for arm, value in self.adapter.last_diagnostics.items():
            diagnostics[arm] = {
                "ik_converged": value.ik.converged,
                "ik_position_error_m": value.ik.position_error,
                "ik_orientation_error_rad": value.ik.orientation_error,
                "ik_restart_index": value.ik.restart_index,
                "ik_solver": value.ik.solver,
                "commanded_position_error_m": value.commanded_position_error,
                "commanded_orientation_error_rad": value.commanded_orientation_error,
                "rate_limited": value.rate_limited,
                "requested_tcp": value.requested_tcp,
                "commanded_tcp": value.commanded_tcp,
            }
        measured_before_state = {key: measured_before[key] for key in YAM_SCALAR_KEYS}
        subsequent_measured_state = {key: subsequent_measured[key] for key in YAM_SCALAR_KEYS}
        clipped_keys: list[str] | None = None
        if isinstance(driver_applied_target, Mapping) and all(
            key in driver_applied_target for key in YAM_SCALAR_KEYS
        ):
            clipped_keys = [
                key
                for key in YAM_SCALAR_KEYS
                if not np.isclose(
                    float(requested_target[key]),
                    float(driver_applied_target[key]),
                    rtol=0.0,
                    atol=1e-9,
                )
            ]
        return _jsonable(
            {
                "tick": tick,
                "first_target": tick == 0 and progress_hold_index == 0,
                "chunk_sequence": chunk_sequence,
                "chunk_row": chunk_row,
                "progress_gate_enabled": self.config.measured_progress_gate,
                "progress_hold_index": progress_hold_index,
                "waypoint_advanced": waypoint_advanced,
                "arms_settled_before": arms_settled_before,
                "measured_target_residuals": measured_target_residuals,
                "measured_gripper_errors": measured_gripper_errors,
                "timing": {
                    "clock": "time.monotonic seconds in controller process",
                    "command_dispatched_at_s": command_dispatched_at_s,
                    "driver_acknowledged_at_s": driver_acknowledged_at_s,
                    "measurement_read_started_at_s": measurement_read_started_at_s,
                    "measurement_received_at_s": measurement_received_at_s,
                    "dispatch_interval_s": dispatch_interval_s,
                    "command_to_ack_s": driver_acknowledged_at_s - command_dispatched_at_s,
                    "measurement_read_duration_s": (
                        measurement_received_at_s - measurement_read_started_at_s
                    ),
                    "command_to_measurement_s": measurement_received_at_s - command_dispatched_at_s,
                },
                "inference_duration_s": inference_duration_s,
                "requested_target": dict(requested_target),
                "driver_applied_target": driver_applied_target,
                "measured_before": measured_before_state,
                "subsequent_measured": subsequent_measured_state,
                "driver_clipped_keys": clipped_keys,
                "driver_acknowledgement_is_achievement": False,
                "remote_model_identity": self.remote_model_identity,
                "diagnostics": diagnostics,
            }
        )

    def _start_record(
        self,
        robot: YamRobot,
        observation: Mapping[str, Any],
    ) -> dict[str, Any]:
        measured = np.asarray([float(observation[key]) for key in YAM_SCALAR_KEYS], dtype=np.float64)
        configured = getattr(getattr(robot, "config", None), "policy_start_position", None)
        configured_values = None if configured is None else np.asarray(configured, dtype=np.float64)
        max_reset_error = None
        if configured_values is not None and configured_values.shape == measured.shape:
            max_reset_error = float(np.max(np.abs(configured_values - measured)))
        return _jsonable(
            {
                "event": "episode_start",
                "action_semantics": self.config.action_semantics,
                "remote_model_identity": self.remote_model_identity,
                "remote_session_contract": self.remote_session_contract,
                "deployment_release_identity": self.deployment_release_identity,
                "action_chunk_contract": {
                    "served_rows": self.config.served_action_horizon,
                    "active_rows": self.config.execution_horizon,
                },
                "inference_hold": {
                    "enabled": self.config.inference_hold_enabled,
                    "max_latency_s": self.config.max_inference_latency_s,
                    "max_joint_drift": self.config.max_inference_hold_joint_drift,
                    "max_gripper_drift": self.config.max_inference_hold_gripper_drift,
                    "arm_worker_deadman_s": self.config.arm_worker_deadman_s,
                    "arm_worker_deadman_relaxed": False,
                },
                "first_active_dispatch_freshness": {
                    "clock": "controller-local time.monotonic",
                    "remote_clock_compared": False,
                    "max_observation_to_first_dispatch_s": (self.config.max_observation_to_first_dispatch_s),
                    "boundary_rule": "age <= limit passes; age > limit fails",
                    "reviewed_worker_dispatch_expiry_margin_s": (self.config.worker_dispatch_expiry_margin_s),
                    "two_arm_dispatch_atomic": False,
                },
                "measured_joint_gripper_state": dict(zip(YAM_SCALAR_KEYS, measured, strict=True)),
                "configured_policy_start": configured_values,
                "max_configured_start_error": max_reset_error,
                "measured_base_to_jaw_tcp": self.adapter.measured_yam_tcp(dict(observation)),
                "camera_joint_timestamp_alignment": "not guaranteed by BiYAM get_observation",
                "shared_world_collision_gate_enabled": self.collision_check is not None,
                "driver_endpoint_collision_gate_enabled": bool(
                    getattr(robot, "has_pre_dispatch_validator", False)
                ),
                "collision_gate_scope": "sampled endpoints only; not a swept-path proof",
                "adapter_rate_limits_per_call": {
                    "joint_rad": getattr(getattr(self.adapter, "limits", None), "max_joint_delta", None),
                    "gripper": getattr(getattr(self.adapter, "limits", None), "max_gripper_delta", None),
                },
                "driver_rate_limits_per_call": {
                    "joint_rad": getattr(getattr(robot, "config", None), "max_joint_delta", None),
                    "gripper": getattr(getattr(robot, "config", None), "max_gripper_delta", None),
                },
            }
        )


def build_adapter(
    urdf: str | Path,
    calibration_json: str | Path | None = None,
    *,
    action_semantics: UmiActionSemantics = EPISODE_START_ABSOLUTE,
    gripper_calibration_json: str | Path | None = None,
    max_joint_delta_per_call: float = 0.02,
    max_gripper_delta_per_call: float | None = None,
    left_joint_limits: Sequence[tuple[float, float]] = YAM_JOINT_LIMITS,
    right_joint_limits: Sequence[tuple[float, float]] = YAM_JOINT_LIMITS,
    expected_urdf_sha256: str | None = PINNED_I2RT_YAM_URDF_SHA256,
) -> YamUmiEeAdapter:
    """Construct the strict adapter with the driver's exact per-arm bounds."""

    validate_trusted_yam_urdf(
        urdf,
        operational_joint_limits={
            "left": left_joint_limits,
            "right": right_joint_limits,
        },
        expected_sha256=expected_urdf_sha256,
    )

    if calibration_json is None:
        if action_semantics not in _CURRENT_RELATIVE_R6D_SEMANTICS:
            raise ValueError("legacy episode-frame semantics require a calibration JSON")
        frames = EpisodeFrames()
    else:
        frames = EpisodeFrames(calibrations=load_calibrations_json(calibration_json))
    current_relative_gripper = (
        None
        if gripper_calibration_json is None
        else load_current_relative_gripper_map_json(gripper_calibration_json)
    )
    if max_gripper_delta_per_call is None:
        max_gripper_delta_per_call = 0.03 if action_semantics in _CURRENT_RELATIVE_R6D_SEMANTICS else 0.05
    return YamUmiEeAdapter(
        left=YamArmKinematics(str(urdf), joint_limits=tuple(left_joint_limits)),
        right=YamArmKinematics(str(urdf), joint_limits=tuple(right_joint_limits)),
        frames=frames,
        current_relative_gripper=current_relative_gripper,
        limits=SafetyLimits(
            max_joint_delta=max_joint_delta_per_call,
            max_gripper_delta=max_gripper_delta_per_call,
        ),
        tracking=TrackingPolicy(mode="strict"),
        action_semantics=action_semantics,
    )


def jsonl_sink(path: str | Path) -> TelemetrySink:
    """Return an append-only, flush-on-command telemetry sink."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def write(record: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()

    return write


@dataclass
class YamUmiDeploymentConfig:
    """Explicit, supervised configuration for the current-relative BiYAM rollout."""

    robot: BiYAMFollowerConfig
    server_address: str
    urdf: Path
    left_camera_key: str
    right_camera_key: str
    telemetry_path: Path
    gripper_calibration_json: Path
    max_steps: int
    expected_model_id: str
    expected_model_revision: str
    expected_policy_type: str
    expected_model_fingerprint: str
    expected_command_ttl_ms: int
    worker_dispatch_expiry_margin_s: float
    action_semantics: UmiActionSemantics = CURRENT_RELATIVE_R6D_SE3
    gripper_action_representation: str = ""
    expected_server_revision: str = ""
    expected_server_config_sha256: str = ""
    expected_model_artifact_tree_sha256: str = ""
    expected_model_artifact_manifest_sha256: str = ""
    expected_runtime_dependency_tree_sha256: str = ""
    expected_runtime_dependency_manifest_sha256: str = ""
    expected_server_service_name: str = ""
    expected_pi_eval_seed: int | None = None
    expected_inference_code_manifest_sha256: str = ""
    expected_inference_attestation_identity_sha256: str = ""
    onset_v3_gate_plan: Path | None = None
    expected_onset_v3_gate_plan_sha256: str = ""
    ik_release_report: Path | None = None
    expected_ik_release_report_sha256: str = ""
    expected_v4_artifact_dataset_root: str = ""
    expected_v4_policy_start_gripper_yam: tuple[float, float] | None = None
    expected_gripper_calibration_sha256: str = ""
    hardware_commissioning_evidence: Path | None = None
    expected_hardware_commissioning_evidence_sha256: str = ""
    task: str = "Put all oranges in the bowl"
    control_hz: float = 30.0
    execution_horizon: int = 15
    max_joint_delta_per_call: float = 0.02
    max_gripper_delta_per_call: float = 0.03
    progress_position_tolerance_m: float = 3e-3
    progress_orientation_tolerance_rad: float = np.deg2rad(2.0)
    progress_gripper_tolerance: float = 0.02
    max_progress_hold_steps: int = 90
    max_inference_latency_s: float = 20.0
    max_inference_hold_joint_drift: float = 0.01
    max_inference_hold_gripper_drift: float = 0.002
    rig_calibration_json: Path | None = None
    minimum_clearance_m: float | None = None
    confirm_hardware_control: bool = False
    reset_after_policy: bool = False

    def __post_init__(self) -> None:
        self.server_address = self.server_address.strip()
        self.task = self.task.strip()
        self.expected_model_id = self.expected_model_id.strip()
        self.expected_model_revision = self.expected_model_revision.strip()
        self.expected_policy_type = self.expected_policy_type.strip()
        self.expected_model_fingerprint = self.expected_model_fingerprint.strip().lower()
        self.gripper_action_representation = self.gripper_action_representation.strip()
        self.expected_server_revision = self.expected_server_revision.strip()
        self.expected_server_config_sha256 = self.expected_server_config_sha256.strip().lower()
        self.expected_model_artifact_tree_sha256 = self.expected_model_artifact_tree_sha256.strip().lower()
        self.expected_model_artifact_manifest_sha256 = (
            self.expected_model_artifact_manifest_sha256.strip().lower()
        )
        self.expected_runtime_dependency_tree_sha256 = (
            self.expected_runtime_dependency_tree_sha256.strip().lower()
        )
        self.expected_runtime_dependency_manifest_sha256 = (
            self.expected_runtime_dependency_manifest_sha256.strip().lower()
        )
        self.expected_server_service_name = self.expected_server_service_name.strip()
        self.expected_inference_code_manifest_sha256 = (
            self.expected_inference_code_manifest_sha256.strip().lower()
        )
        self.expected_inference_attestation_identity_sha256 = (
            self.expected_inference_attestation_identity_sha256.strip().lower()
        )
        self.expected_onset_v3_gate_plan_sha256 = self.expected_onset_v3_gate_plan_sha256.strip().lower()
        self.expected_ik_release_report_sha256 = self.expected_ik_release_report_sha256.strip().lower()
        self.expected_v4_artifact_dataset_root = self.expected_v4_artifact_dataset_root.strip()
        self.expected_gripper_calibration_sha256 = self.expected_gripper_calibration_sha256.strip().lower()
        self.expected_hardware_commissioning_evidence_sha256 = (
            self.expected_hardware_commissioning_evidence_sha256.strip().lower()
        )
        self.left_camera_key = self.left_camera_key.strip()
        self.right_camera_key = self.right_camera_key.strip()
        self.urdf = Path(self.urdf).expanduser()
        self.telemetry_path = Path(self.telemetry_path).expanduser()
        self.gripper_calibration_json = Path(self.gripper_calibration_json).expanduser()
        if self.onset_v3_gate_plan is not None:
            self.onset_v3_gate_plan = Path(self.onset_v3_gate_plan).expanduser()
        if self.ik_release_report is not None:
            self.ik_release_report = Path(self.ik_release_report).expanduser()
        if self.hardware_commissioning_evidence is not None:
            self.hardware_commissioning_evidence = Path(self.hardware_commissioning_evidence).expanduser()
        if self.rig_calibration_json is not None:
            self.rig_calibration_json = Path(self.rig_calibration_json).expanduser()
        if not self.server_address or not self.task:
            raise ValueError("server_address and task must be non-empty")
        if not self.expected_model_id or not self.expected_model_revision:
            raise ValueError("expected_model_id and expected_model_revision must be non-empty")
        if self.expected_policy_type not in {"molmoact2", "pi05"}:
            raise ValueError("expected_policy_type must be exactly 'molmoact2' or 'pi05'")
        if self.action_semantics not in _CURRENT_RELATIVE_R6D_SEMANTICS:
            raise ValueError(
                "current-relative BiYAM deployment action_semantics must be "
                f"one of {sorted(_CURRENT_RELATIVE_R6D_SEMANTICS)}"
            )
        if self.action_semantics == CURRENT_RELATIVE_R6D_SE3_JAW_DELTA:
            if self.onset_v3_gate_plan is not None or self.expected_onset_v3_gate_plan_sha256:
                raise ValueError("v4 jaw-delta deployment must not declare an onset-v3 gate plan")
            if self.gripper_action_representation != GRIPPER_ACTION_QUERY_ANCHOR_DELTA:
                raise ValueError(
                    "jaw-delta deployment requires gripper_action_representation="
                    f"{GRIPPER_ACTION_QUERY_ANCHOR_DELTA!r}"
                )
            if self.ik_release_report is None:
                raise ValueError("v4 jaw-delta deployment requires ik_release_report")
            if not self.expected_v4_artifact_dataset_root:
                raise ValueError("v4 jaw-delta deployment requires expected_v4_artifact_dataset_root")
            if self.expected_v4_policy_start_gripper_yam is None:
                raise ValueError("v4 jaw-delta deployment requires expected_v4_policy_start_gripper_yam")
            if not self.expected_server_revision:
                raise ValueError("v4 jaw-delta deployment requires expected_server_revision")
            for name in (
                "expected_server_config_sha256",
                "expected_model_artifact_tree_sha256",
                "expected_model_artifact_manifest_sha256",
                "expected_ik_release_report_sha256",
                "expected_gripper_calibration_sha256",
            ):
                _require_lower_sha256(getattr(self, name), label=name)
            commissioning_values = (
                self.hardware_commissioning_evidence,
                self.expected_hardware_commissioning_evidence_sha256,
            )
            if (commissioning_values[0] is None) != (not commissioning_values[1]):
                raise ValueError(
                    "hardware_commissioning_evidence and its expected SHA-256 must be set together"
                )
            if self.hardware_commissioning_evidence is not None:
                _require_lower_sha256(
                    self.expected_hardware_commissioning_evidence_sha256,
                    label="expected_hardware_commissioning_evidence_sha256",
                )
        elif self.gripper_action_representation:
            raise ValueError("legacy absolute-jaw deployment must not declare a jaw-delta representation")
        elif (self.onset_v3_gate_plan is None) != (not self.expected_onset_v3_gate_plan_sha256):
            raise ValueError("onset_v3_gate_plan and its expected SHA-256 must be set together")
        elif self.onset_v3_gate_plan is not None:
            if self.expected_onset_v3_gate_plan_sha256 != _ONSET_V3_GATE_PLAN_SHA256:
                raise ValueError("onset-v3 deployment requires the exact reviewed gate plan SHA-256")
            if not self.expected_server_revision:
                raise ValueError("onset-v3 deployment requires expected_server_revision")
            for name in (
                "expected_server_config_sha256",
                "expected_model_artifact_tree_sha256",
                "expected_model_artifact_manifest_sha256",
                "expected_runtime_dependency_tree_sha256",
                "expected_runtime_dependency_manifest_sha256",
                "expected_gripper_calibration_sha256",
            ):
                _require_lower_sha256(getattr(self, name), label=name)
            if self.expected_policy_type == "molmoact2":
                if self.expected_server_service_name != "lerobot-remote-policy":
                    raise ValueError("onset-v3 MolmoAct2 requires service_name='lerobot-remote-policy'")
                if self.expected_pi_eval_seed is not None or any(
                    (
                        self.expected_inference_code_manifest_sha256,
                        self.expected_inference_attestation_identity_sha256,
                    )
                ):
                    raise ValueError("MolmoAct2 onset-v3 must not declare Pi explicit-noise attestations")
            else:
                if isinstance(self.expected_pi_eval_seed, bool) or self.expected_pi_eval_seed not in range(5):
                    raise ValueError("Pi0.5 onset-v3 expected_pi_eval_seed must be one of 0..4")
                expected_service = f"{_PI05_EXPLICIT_NOISE_SERVICE_PREFIX}{self.expected_pi_eval_seed}"
                if self.expected_server_service_name != expected_service:
                    raise ValueError(f"Pi0.5 onset-v3 requires service_name={expected_service!r}")
                for name in (
                    "expected_inference_code_manifest_sha256",
                    "expected_inference_attestation_identity_sha256",
                ):
                    _require_lower_sha256(getattr(self, name), label=name)
            commissioning_values = (
                self.hardware_commissioning_evidence,
                self.expected_hardware_commissioning_evidence_sha256,
            )
            if (commissioning_values[0] is None) != (not commissioning_values[1]):
                raise ValueError(
                    "hardware_commissioning_evidence and its expected SHA-256 must be set together"
                )
            if self.hardware_commissioning_evidence is not None:
                _require_lower_sha256(
                    self.expected_hardware_commissioning_evidence_sha256,
                    label="expected_hardware_commissioning_evidence_sha256",
                )
        elif self.hardware_commissioning_evidence is not None or (
            self.expected_hardware_commissioning_evidence_sha256
        ):
            raise ValueError(
                "hardware commissioning evidence requires v4 jaw-delta or onset-v3 release attestation"
            )
        if len(self.expected_model_fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in self.expected_model_fingerprint
        ):
            raise ValueError("expected_model_fingerprint must contain 64 hexadecimal characters")
        if (
            isinstance(self.expected_command_ttl_ms, bool)
            or not isinstance(self.expected_command_ttl_ms, int)
            or self.expected_command_ttl_ms <= 0
        ):
            raise ValueError("expected_command_ttl_ms must be a positive integer")
        if not self.left_camera_key or not self.right_camera_key:
            raise ValueError("left_camera_key and right_camera_key must be non-empty")
        if self.left_camera_key == self.right_camera_key:
            raise ValueError("left and right camera keys must differ")
        configured_start = np.asarray(self.robot.policy_start_position, dtype=np.float64)
        if self.action_semantics == CURRENT_RELATIVE_R6D_SE3_JAW_DELTA:
            if configured_start.shape != (14,) or not np.isfinite(configured_start).all():
                raise ValueError("v4 robot.policy_start_position must contain 14 finite values")
            start_grippers = np.asarray(
                self.expected_v4_policy_start_gripper_yam,
                dtype=np.float64,
            )
            if (
                start_grippers.shape != (2,)
                or not np.isfinite(start_grippers).all()
                or np.any((start_grippers < 0) | (start_grippers > 1))
            ):
                raise ValueError(
                    "expected_v4_policy_start_gripper_yam must contain two finite values in [0, 1]"
                )
            if not np.array_equal(configured_start[[6, 13]], start_grippers):
                raise ValueError("v4 policy-start grippers do not match expected_v4_policy_start_gripper_yam")
        elif self.onset_v3_gate_plan is None:
            canonical_start = np.asarray(BI_YAM_POLICY_START_POSITION, dtype=np.float64)
            if configured_start.shape != canonical_start.shape or not np.array_equal(
                configured_start, canonical_start
            ):
                raise ValueError(
                    "current-relative UMI deployment requires canonical BiYAM policy_start_position "
                    "[0,0,0,0,0,0,1,0,0,0,0,0,0,1]"
                )
        else:
            expected_onset_start = np.asarray([0.0, 0.05, 0.05, 0.0, 0.0, 0.0, 1.0] * 2)
            if configured_start.shape != expected_onset_start.shape or not np.array_equal(
                configured_start, expected_onset_start
            ):
                raise ValueError(
                    "onset-v3 deployment requires policy_start_position [0,.05,.05,0,0,0,1] for both arms"
                )
            for arm, camera_key in (("left", self.left_camera_key), ("right", self.right_camera_key)):
                camera = self.robot.cameras.get(camera_key)
                if (
                    camera is None
                    or getattr(camera, "width", None) != 800
                    or getattr(camera, "height", None) != 600
                ):
                    raise ValueError(
                        f"onset-v3 {arm} camera must capture the native trained 800x600 resolution"
                    )
        for side in ("left", "right"):
            arm = getattr(self.robot, f"{side}_arm_config")
            if arm.arm_type != "yam" or arm.gripper_type != "linear_4310":
                raise ValueError(
                    "current-relative YAM deployment is pinned to arm_type='yam' and "
                    f"gripper_type='linear_4310' on both sides; {side} is "
                    f"{arm.arm_type!r}/{arm.gripper_type!r}"
                )
            if not np.isclose(arm.command_ttl_s, 1.0, rtol=0.0, atol=1e-9):
                raise ValueError(
                    "current-relative supervised deployment requires the pinned 1.0s arm worker "
                    f"deadman; {side}.command_ttl_s={arm.command_ttl_s!r}"
                )
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.execution_horizon != _CURRENT_RELATIVE_ACTIVE_EXECUTION_HORIZON:
            raise ValueError(
                "current-relative supervised deployment requires execution_horizon="
                f"{_CURRENT_RELATIVE_ACTIVE_EXECUTION_HORIZON}; the remaining rows in the "
                f"{_CURRENT_RELATIVE_SERVED_ACTION_HORIZON}-row served chunk are diagnostic-only"
            )
        for name in (
            "control_hz",
            "max_joint_delta_per_call",
            "max_gripper_delta_per_call",
            "progress_position_tolerance_m",
            "progress_orientation_tolerance_rad",
            "progress_gripper_tolerance",
            "max_inference_latency_s",
            "max_inference_hold_joint_drift",
            "max_inference_hold_gripper_drift",
            "worker_dispatch_expiry_margin_s",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            self.robot.command_lead_time_s + self.worker_dispatch_expiry_margin_s
            > self.expected_command_ttl_ms / 1000.0
        ):
            raise ValueError(
                "expected command TTL must cover command lead time plus the reviewed worker "
                "dispatch expiry margin"
            )
        if self.progress_gripper_tolerance > 1:
            raise ValueError("progress_gripper_tolerance cannot exceed the normalized gripper range")
        if self.max_progress_hold_steps <= 0:
            raise ValueError("max_progress_hold_steps must be positive")
        if self.max_joint_delta_per_call > self.robot.max_joint_delta:
            raise ValueError("adapter max_joint_delta_per_call cannot exceed robot.max_joint_delta")
        if self.max_gripper_delta_per_call > self.robot.max_gripper_delta:
            raise ValueError("adapter max_gripper_delta_per_call cannot exceed robot.max_gripper_delta")
        if (self.rig_calibration_json is None) != (self.minimum_clearance_m is None):
            raise ValueError(
                "rig_calibration_json and minimum_clearance_m must either both be set or both be omitted"
            )
        if self.confirm_hardware_control and (
            self.rig_calibration_json is None or self.minimum_clearance_m is None
        ):
            raise ValueError(
                "confirm_hardware_control=true requires verified rig/table calibration and clearance"
            )
        if (
            self.confirm_hardware_control
            and self.action_semantics == CURRENT_RELATIVE_R6D_SE3_JAW_DELTA
            and self.hardware_commissioning_evidence is None
        ):
            raise ValueError(
                "v4 confirm_hardware_control=true requires immutable hardware commissioning evidence"
            )
        if (
            self.confirm_hardware_control
            and self.onset_v3_gate_plan is not None
            and (self.hardware_commissioning_evidence is None)
        ):
            raise ValueError(
                "onset-v3 confirm_hardware_control=true requires immutable hardware commissioning evidence"
            )
        if self.minimum_clearance_m is not None and (
            not np.isfinite(self.minimum_clearance_m) or self.minimum_clearance_m < 0
        ):
            raise ValueError("minimum_clearance_m must be finite and non-negative")


def _validate_remote_model_manifest(
    cfg: YamUmiDeploymentConfig,
    model: object,
    *,
    state_features: tuple[str, ...],
    action_features: tuple[str, ...],
) -> dict[str, Any]:
    """Bind deployment to one exact server-advertised model manifest."""

    required_fields = (
        "model_id",
        "revision",
        "policy_type",
        "norm_tag",
        "action_horizon",
        "action_dim",
        "state_features",
        "action_features",
        "camera_keys",
        "fingerprint",
    )
    release_attested = (
        cfg.action_semantics == CURRENT_RELATIVE_R6D_SE3_JAW_DELTA or cfg.onset_v3_gate_plan is not None
    )
    if release_attested:
        required_fields = (
            *required_fields,
            "artifact_tree_sha256",
            "artifact_manifest_sha256",
        )
    if cfg.onset_v3_gate_plan is not None:
        required_fields = (
            *required_fields,
            "runtime_dependency_tree_sha256",
            "runtime_dependency_manifest_sha256",
        )
    if cfg.onset_v3_gate_plan is not None and cfg.expected_policy_type == "pi05":
        required_fields = (
            *required_fields,
            "inference_backend_mode",
            "inference_seed",
            "inference_code_manifest_sha256",
            "inference_attestation_identity_sha256",
        )
    if cfg.action_semantics == CURRENT_RELATIVE_R6D_SE3_JAW_DELTA:
        required_fields = (*required_fields, "gripper_action_representation", "ik_release_report_sha256")
    missing = [name for name in required_fields if not hasattr(model, name)]
    if missing:
        raise RuntimeError(f"served model manifest is missing required fields: {missing}")
    values = {name: getattr(model, name) for name in required_fields}
    identity = {
        "model_id": str(values["model_id"]),
        "revision": str(values["revision"]),
        "policy_type": str(values["policy_type"]),
        "norm_tag": str(values["norm_tag"]),
        "action_horizon": int(values["action_horizon"]),
        "action_dim": int(values["action_dim"]),
        "state_features": tuple(values["state_features"]),
        "action_features": tuple(values["action_features"]),
        "camera_keys": tuple(values["camera_keys"]),
        "fingerprint": str(values["fingerprint"]).lower(),
    }
    if "gripper_action_representation" in values:
        identity["gripper_action_representation"] = str(values["gripper_action_representation"])
    for digest_name in (
        "artifact_tree_sha256",
        "artifact_manifest_sha256",
        "ik_release_report_sha256",
        "runtime_dependency_tree_sha256",
        "runtime_dependency_manifest_sha256",
        "inference_code_manifest_sha256",
        "inference_attestation_identity_sha256",
    ):
        if digest_name not in values:
            continue
        digest = str(values[digest_name])
        if (
            len(digest) != 64
            or digest != digest.lower()
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise RuntimeError(f"served model {digest_name} must be a lowercase SHA-256 digest")
        identity[digest_name] = digest
    mismatches = []
    expected_values = {
        "model_id": cfg.expected_model_id,
        "revision": cfg.expected_model_revision,
        "fingerprint": cfg.expected_model_fingerprint,
        "policy_type": cfg.expected_policy_type,
        "action_horizon": 24,
        "action_dim": 20,
        "state_features": state_features,
        "action_features": action_features,
        "camera_keys": ("umi1", "umi2"),
    }
    if release_attested:
        expected_values["artifact_tree_sha256"] = cfg.expected_model_artifact_tree_sha256
        expected_values["artifact_manifest_sha256"] = cfg.expected_model_artifact_manifest_sha256
    if cfg.onset_v3_gate_plan is not None:
        expected_values.update(
            {
                "runtime_dependency_tree_sha256": cfg.expected_runtime_dependency_tree_sha256,
                "runtime_dependency_manifest_sha256": cfg.expected_runtime_dependency_manifest_sha256,
            }
        )
    if cfg.onset_v3_gate_plan is not None and cfg.expected_policy_type == "pi05":
        identity.update(
            {
                "inference_backend_mode": str(values["inference_backend_mode"]),
                "inference_seed": int(values["inference_seed"]),
            }
        )
        expected_values.update(
            {
                "inference_backend_mode": _PI05_EXPLICIT_NOISE_BACKEND_MODE,
                "inference_seed": cfg.expected_pi_eval_seed,
                "inference_code_manifest_sha256": cfg.expected_inference_code_manifest_sha256,
                "inference_attestation_identity_sha256": (cfg.expected_inference_attestation_identity_sha256),
            }
        )
    if cfg.action_semantics == CURRENT_RELATIVE_R6D_SE3_JAW_DELTA:
        expected_values["gripper_action_representation"] = cfg.gripper_action_representation
        expected_values["ik_release_report_sha256"] = cfg.expected_ik_release_report_sha256
    for name, expected in expected_values.items():
        if identity[name] != expected:
            mismatches.append(f"{name}={identity[name]!r} (expected {expected!r})")
    if mismatches:
        raise RuntimeError("served checkpoint identity/contract mismatch: " + "; ".join(mismatches))
    return _jsonable(identity)


def _validate_remote_session_contract(
    cfg: YamUmiDeploymentConfig,
    session: object,
) -> dict[str, Any]:
    """Bind freshness to one server-advertised TTL and provenance digest."""

    if not hasattr(session, "command_ttl_ms"):
        raise RuntimeError("remote policy session is missing required command_ttl_ms")
    service_name = getattr(session, "service_name", "")
    if cfg.expected_server_service_name:
        if not isinstance(service_name, str) or not service_name or service_name != service_name.strip():
            raise RuntimeError("remote policy service_name is missing or malformed")
        if service_name != cfg.expected_server_service_name:
            raise RuntimeError(
                f"remote policy service_name mismatch: served {service_name!r}, "
                f"expected {cfg.expected_server_service_name!r}"
            )
    command_ttl_ms = session.command_ttl_ms
    if isinstance(command_ttl_ms, bool) or not isinstance(command_ttl_ms, int):
        raise RuntimeError("remote policy session command_ttl_ms must be a positive integer")
    if command_ttl_ms <= 0:
        raise RuntimeError("remote policy session command_ttl_ms must be positive")
    if command_ttl_ms != cfg.expected_command_ttl_ms:
        raise RuntimeError(
            "remote policy session command_ttl_ms mismatch: "
            f"served {command_ttl_ms}, expected {cfg.expected_command_ttl_ms}"
        )
    if not hasattr(session, "server_revision"):
        raise RuntimeError("remote policy session is missing required server_revision")
    server_revision = session.server_revision
    if not isinstance(server_revision, str) or not server_revision.strip():
        raise RuntimeError("remote policy session server_revision must be non-empty")
    if server_revision != server_revision.strip():
        raise RuntimeError("remote policy session server_revision must be trimmed")
    if not hasattr(session, "server_config_sha256"):
        raise RuntimeError("remote policy session is missing required server_config_sha256")
    server_config_sha256 = session.server_config_sha256
    if (
        not isinstance(server_config_sha256, str)
        or len(server_config_sha256) != 64
        or server_config_sha256 != server_config_sha256.lower()
        or any(character not in "0123456789abcdef" for character in server_config_sha256)
    ):
        raise RuntimeError("remote policy session server_config_sha256 must be a SHA-256 digest")
    if cfg.action_semantics == CURRENT_RELATIVE_R6D_SE3_JAW_DELTA or cfg.onset_v3_gate_plan is not None:
        if server_revision != cfg.expected_server_revision:
            raise RuntimeError(
                "remote policy session server_revision mismatch: "
                f"served {server_revision!r}, expected {cfg.expected_server_revision!r}"
            )
        if server_config_sha256 != cfg.expected_server_config_sha256:
            raise RuntimeError(
                "remote policy session server_config_sha256 mismatch: "
                f"served {server_config_sha256}, "
                f"expected {cfg.expected_server_config_sha256}"
            )
    return _jsonable(
        {
            "command_ttl_ms": command_ttl_ms,
            "service_name": service_name,
            "expected_command_ttl_ms": cfg.expected_command_ttl_ms,
            "first_active_freshness_limit_s": command_ttl_ms / 1000.0,
            "server_revision": server_revision,
            "server_config_sha256": server_config_sha256,
            "control_hz": cfg.control_hz,
            "arm_worker_deadman_s": {
                "left": cfg.robot.left_arm_config.command_ttl_s,
                "right": cfg.robot.right_arm_config.command_ttl_s,
            },
            "remote_command_ttl_is_arm_worker_deadman": False,
        }
    )


def _session_freshness_limit_s(session: object) -> float:
    """Legacy helper retained for callers outside supervised deployment."""

    if not hasattr(session, "command_ttl_ms"):
        raise RuntimeError("remote policy session is missing required command_ttl_ms")
    command_ttl_ms = session.command_ttl_ms
    if isinstance(command_ttl_ms, bool) or not isinstance(command_ttl_ms, int):
        raise RuntimeError("remote policy session command_ttl_ms must be a positive integer")
    if command_ttl_ms <= 0:
        raise RuntimeError("remote policy session command_ttl_ms must be positive")
    return command_ttl_ms / 1000.0


def _preflight_record(
    cfg: YamUmiDeploymentConfig,
    *,
    remote_model_identity: Mapping[str, Any],
    gripper_mapping: CurrentRelativeGripperMap,
    collision_check: CollisionCheck | None,
    max_observation_to_first_dispatch_s: float,
    remote_session_contract: Mapping[str, Any],
    deployment_release_identity: Mapping[str, Any] | None,
    hardware_commissioning_identity: Mapping[str, Any] | None,
) -> dict[str, Any]:
    gripper_bindings = {}
    for arm in ("left", "right"):
        calibration = getattr(gripper_mapping, arm)
        gripper_bindings[arm] = {
            "dataset_device": calibration.dataset_device,
            "device_id": calibration.device_id,
            "evidence_sha256": calibration.evidence_sha256.lower(),
            "detector_config_sha256": calibration.detector_config_sha256.lower(),
            "fisheye_calibration_sha256": calibration.fisheye_calibration_sha256.lower(),
            "geometry_config_sha256": calibration.geometry_config_sha256.lower(),
        }
    configured_start = np.asarray(cfg.robot.policy_start_position, dtype=np.float64)
    start_gripper_calibrated_umi = {
        "left": gripper_mapping.yam_to_umi(float(configured_start[6]), arm="left"),
        "right": gripper_mapping.yam_to_umi(float(configured_start[13]), arm="right"),
    }
    onset_v3_gripper_identity = None
    if cfg.onset_v3_gate_plan is not None:
        onset_v3_gripper_identity = _onset_v3_gripper_support_identity(start_gripper_calibrated_umi)
    if (
        onset_v3_gripper_identity is not None
        and not onset_v3_gripper_identity["all_within_frozen_training_support"]
    ):
        commissioning_status = "NO_GO_CALIBRATED_RESET_GRIPPER_OUTSIDE_TRAINING_SUPPORT"
    elif hardware_commissioning_identity is not None:
        commissioning_status = "VERIFIED_NO_MOTION_SEPARATE_MOTION_CONFIRMATION_REQUIRED"
    elif cfg.action_semantics == CURRENT_RELATIVE_R6D_SE3_JAW_DELTA or cfg.onset_v3_gate_plan is not None:
        commissioning_status = "NO_GO_MISSING_COMMISSIONING_EVIDENCE"
    else:
        commissioning_status = "NOT_APPLICABLE_LEGACY_SEMANTIC"
    return _jsonable(
        {
            "event": "deployment_preflight",
            "passed": True,
            "hardware_control_confirmed": cfg.confirm_hardware_control,
            "hardware_constructed": False,
            "hardware_commands_sent": False,
            "action_semantics": cfg.action_semantics,
            "gripper_action_representation": cfg.gripper_action_representation,
            "policy_start_position": configured_start.tolist(),
            "policy_start_gripper_binding": {
                "yam_command": {
                    "left": float(configured_start[6]),
                    "right": float(configured_start[13]),
                },
                "calibrated_umi_width": start_gripper_calibrated_umi,
                "onset_v3_support_identity": onset_v3_gripper_identity,
                "calibration_sha256": cfg.expected_gripper_calibration_sha256 or None,
            },
            "deployment_release_identity": deployment_release_identity,
            "hardware_commissioning": {
                "status": commissioning_status,
                "decision_scope": (
                    "NO_GO applies to any physical policy motion; a successful no-motion "
                    "software handshake is not hardware commissioning"
                ),
                "hardware_motion_allowed": False,
                "identity": hardware_commissioning_identity,
            },
            "action_chunk_contract": {
                "served_rows": _CURRENT_RELATIVE_SERVED_ACTION_HORIZON,
                "active_rows": _CURRENT_RELATIVE_ACTIVE_EXECUTION_HORIZON,
                "diagnostic_only_rows": list(
                    range(
                        _CURRENT_RELATIVE_ACTIVE_EXECUTION_HORIZON,
                        _CURRENT_RELATIVE_SERVED_ACTION_HORIZON,
                    )
                ),
            },
            "inference_hold": {
                "enabled_for_hardware": True,
                "max_latency_s": cfg.max_inference_latency_s,
                "max_joint_drift": cfg.max_inference_hold_joint_drift,
                "max_gripper_drift": cfg.max_inference_hold_gripper_drift,
            },
            "first_active_dispatch_freshness": {
                "clock": "controller-local time.monotonic",
                "remote_clock_compared": False,
                "max_observation_to_first_dispatch_s": max_observation_to_first_dispatch_s,
                "boundary_rule": "age <= limit passes; age > limit fails",
                "reviewed_worker_dispatch_expiry_margin_s": (cfg.worker_dispatch_expiry_margin_s),
                "two_arm_dispatch_atomic": False,
            },
            "urdf": {
                "path": str(cfg.urdf.resolve()),
                "sha256": hashlib.sha256(cfg.urdf.read_bytes()).hexdigest(),
                "pinned_sha256": PINNED_I2RT_YAM_URDF_SHA256,
            },
            "gripper_calibration": {
                "path": str(cfg.gripper_calibration_json.resolve()),
                "sha256": hashlib.sha256(cfg.gripper_calibration_json.read_bytes()).hexdigest(),
                "bindings": gripper_bindings,
            },
            "remote_model_identity": remote_model_identity,
            "remote_session_contract": remote_session_contract,
            "shared_world_collision_gate_enabled": collision_check is not None,
        }
    )


def run_current_relative_biyam(
    cfg: YamUmiDeploymentConfig,
    *,
    robot_factory=None,
    client_factory=None,
    adapter_factory=None,
    controller_factory=None,
) -> list[dict[str, Any]]:
    """Preflight or run one current-relative episode with fail-safe cleanup."""

    if not cfg.urdf.is_file():
        raise FileNotFoundError(cfg.urdf)
    if cfg.telemetry_path.exists():
        raise FileExistsError(f"refusing to append to existing rollout telemetry: {cfg.telemetry_path}")
    if not cfg.gripper_calibration_json.is_file():
        raise FileNotFoundError(cfg.gripper_calibration_json)
    if cfg.rig_calibration_json is not None and not cfg.rig_calibration_json.is_file():
        raise FileNotFoundError(cfg.rig_calibration_json)

    deployment_release_identity = None
    hardware_commissioning_identity = None
    if cfg.action_semantics == CURRENT_RELATIVE_R6D_SE3_JAW_DELTA:
        assert cfg.ik_release_report is not None
        assert cfg.expected_v4_policy_start_gripper_yam is not None
        deployment_release_identity = _load_v4_ik_release_identity(
            report_path=cfg.ik_release_report,
            expected_report_sha256=cfg.expected_ik_release_report_sha256,
            expected_dataset_root=cfg.expected_v4_artifact_dataset_root,
            policy_start_position=cfg.robot.policy_start_position,
            expected_start_gripper_yam=cfg.expected_v4_policy_start_gripper_yam,
        )
    elif cfg.onset_v3_gate_plan is not None:
        deployment_release_identity = _load_onset_v3_gate_identity(
            report_path=cfg.onset_v3_gate_plan,
            expected_report_sha256=cfg.expected_onset_v3_gate_plan_sha256,
            policy_start_position=cfg.robot.policy_start_position,
        )

    if deployment_release_identity is not None:
        if (
            cfg.gripper_calibration_json.is_symlink()
            or not cfg.gripper_calibration_json.is_file()
            or stat.S_IMODE(cfg.gripper_calibration_json.stat().st_mode) & 0o222
        ):
            raise ValueError("release-attested gripper calibration must be a read-only regular file")
        actual_gripper_calibration_sha256 = _sha256_path(cfg.gripper_calibration_json)
        if actual_gripper_calibration_sha256 != cfg.expected_gripper_calibration_sha256:
            raise ValueError(
                "release-attested gripper calibration SHA-256 mismatch: "
                f"expected {cfg.expected_gripper_calibration_sha256}, "
                f"got {actual_gripper_calibration_sha256}"
            )
        if cfg.hardware_commissioning_evidence is not None:
            hardware_commissioning_identity = _load_v4_hardware_commissioning_identity(
                cfg=cfg,
                evidence_path=cfg.hardware_commissioning_evidence,
                expected_evidence_sha256=cfg.expected_hardware_commissioning_evidence_sha256,
            )
        deployment_release_identity = {
            **deployment_release_identity,
            "hardware_commissioning_evidence": hardware_commissioning_identity,
        }

    gripper_mapping = load_current_relative_gripper_map_json(cfg.gripper_calibration_json)
    gripper_mapping.assert_hardware_ready()

    from lerobot.remote_inference.umi_ee_client import (
        UMI_CURRENTREL_R6D_ACTION_FEATURE_NAMES,
        UMI_CURRENTREL_R6D_JAW_DELTA_ACTION_FEATURE_NAMES,
        UMI_CURRENTREL_R6D_STATE_FEATURE_NAMES,
        UmiEeClientConfig,
        UmiEeRemoteClient,
    )
    from lerobot.robots.bi_yam.bi_yam import BiYAMFollower

    robot_factory = robot_factory or BiYAMFollower
    client_factory = client_factory or UmiEeRemoteClient
    adapter_factory = adapter_factory or build_adapter
    controller_factory = controller_factory or YamUmiRemoteController
    action_features = (
        UMI_CURRENTREL_R6D_JAW_DELTA_ACTION_FEATURE_NAMES
        if cfg.action_semantics == CURRENT_RELATIVE_R6D_SE3_JAW_DELTA
        else UMI_CURRENTREL_R6D_ACTION_FEATURE_NAMES
    )

    client = client_factory(
        UmiEeClientConfig(
            server_address=cfg.server_address,
            task=cfg.task,
            camera_keys=("umi1", "umi2"),
            control_hz=cfg.control_hz,
            feature_names=UMI_CURRENTREL_R6D_STATE_FEATURE_NAMES,
            action_feature_names=action_features,
            robot_id=f"{cfg.robot.id}-umi-currentrel",
            inference_timeout_s=cfg.max_inference_latency_s,
        )
    )
    robot = None
    connected = False
    armed = False
    completed = False
    try:
        session = client.connect()
        remote_model_identity = _validate_remote_model_manifest(
            cfg,
            session.model,
            state_features=UMI_CURRENTREL_R6D_STATE_FEATURE_NAMES,
            action_features=action_features,
        )
        remote_session_contract = _validate_remote_session_contract(cfg, session)
        max_observation_to_first_dispatch_s = float(remote_session_contract["first_active_freshness_limit_s"])

        collision_check = None
        if cfg.rig_calibration_json is not None:
            from lerobot.robots.bi_yam.rig_safety import DualYAMRigCalibration, RigCollisionChecker

            calibration = DualYAMRigCalibration.from_json(cfg.rig_calibration_json)
            if hardware_commissioning_identity is not None:
                hardware_commissioning_identity["rig_physical_provenance"] = (
                    calibration.require_verified_physical_evidence()
                )
            collision_check = RigCollisionChecker(
                calibration,
                minimum_clearance_m=float(cfg.minimum_clearance_m),
            )
            collision_check.assert_hardware_ready()
        else:
            logger.warning(
                "Shared-world inter-arm/table collision gate is disabled; run only under direct supervision"
            )

        adapter = adapter_factory(
            cfg.urdf,
            action_semantics=cfg.action_semantics,
            gripper_calibration_json=cfg.gripper_calibration_json,
            max_joint_delta_per_call=cfg.max_joint_delta_per_call,
            max_gripper_delta_per_call=cfg.max_gripper_delta_per_call,
            left_joint_limits=cfg.robot.left_joint_limits,
            right_joint_limits=cfg.robot.right_joint_limits,
        )
        controller = controller_factory(
            client=client,
            adapter=adapter,
            config=YamUmiControllerConfig(
                left_camera_key=cfg.left_camera_key,
                right_camera_key=cfg.right_camera_key,
                control_hz=cfg.control_hz,
                execution_horizon=cfg.execution_horizon,
                served_action_horizon=_CURRENT_RELATIVE_SERVED_ACTION_HORIZON,
                action_semantics=cfg.action_semantics,
                measured_progress_gate=True,
                progress_position_tolerance_m=cfg.progress_position_tolerance_m,
                progress_orientation_tolerance_rad=cfg.progress_orientation_tolerance_rad,
                progress_gripper_tolerance=cfg.progress_gripper_tolerance,
                max_progress_hold_steps=cfg.max_progress_hold_steps,
                inference_hold_enabled=True,
                max_inference_latency_s=cfg.max_inference_latency_s,
                max_inference_hold_joint_drift=cfg.max_inference_hold_joint_drift,
                max_inference_hold_gripper_drift=cfg.max_inference_hold_gripper_drift,
                max_observation_to_first_dispatch_s=max_observation_to_first_dispatch_s,
                arm_worker_deadman_s=min(
                    cfg.robot.left_arm_config.command_ttl_s,
                    cfg.robot.right_arm_config.command_ttl_s,
                ),
                worker_dispatch_expiry_margin_s=cfg.worker_dispatch_expiry_margin_s,
            ),
            collision_check=collision_check,
            remote_model_identity=remote_model_identity,
            remote_session_contract=remote_session_contract,
            deployment_release_identity=deployment_release_identity,
            telemetry_sink=jsonl_sink(cfg.telemetry_path) if cfg.confirm_hardware_control else None,
        )
        controller.assert_hardware_ready()

        if not cfg.confirm_hardware_control:
            record = _preflight_record(
                cfg,
                remote_model_identity=remote_model_identity,
                gripper_mapping=gripper_mapping,
                collision_check=collision_check,
                max_observation_to_first_dispatch_s=max_observation_to_first_dispatch_s,
                remote_session_contract=remote_session_contract,
                deployment_release_identity=deployment_release_identity,
                hardware_commissioning_identity=hardware_commissioning_identity,
            )
            logger.info("No-motion deployment preflight completed: %s", json.dumps(record, sort_keys=True))
            return [record]

        # This RPC mutates recurrent server state, so it must complete before a
        # physical robot is even constructed, much less armed.
        controller.reset_remote_session_while_disarmed()
        robot = robot_factory(cfg.robot)
        if collision_check is not None:
            install_validator = getattr(robot, "set_pre_dispatch_validator", None)
            if not callable(install_validator):
                raise RuntimeError(
                    "configured rig collision checking requires a robot with a final-target "
                    "pre-dispatch validator hook"
                )
            install_validator(collision_check)
        robot.connect()
        connected = True
        robot.arm()
        armed = True
        robot.reset_for_policy()
        records = controller.run_episode(robot, max_steps=cfg.max_steps)
        completed = True
        if cfg.reset_after_policy:
            robot.reset_after_policy()
        return records
    finally:
        if completed:
            try:
                if robot is not None and armed:
                    robot.disarm()
            finally:
                try:
                    if robot is not None and connected:
                        robot.disconnect()
                finally:
                    client.close()
        else:
            if robot is not None and armed:
                with suppress(Exception):
                    robot.disarm()
            if robot is not None and connected:
                with suppress(Exception):
                    robot.disconnect()
            with suppress(Exception):
                client.close()


def main() -> None:
    import draccus

    logging.basicConfig(level=logging.INFO)
    cfg = draccus.parse(config_class=YamUmiDeploymentConfig)
    run_current_relative_biyam(cfg)


if __name__ == "__main__":
    main()
