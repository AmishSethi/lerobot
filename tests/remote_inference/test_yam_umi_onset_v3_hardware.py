from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import draccus
import pytest

from lerobot.remote_inference.yam_umi_ee_bridge import (
    YamUmiDeploymentConfig,
    _load_onset_v3_gate_identity,
    _onset_v3_gripper_support_identity,
)


def test_reset_jaw_support_is_arm_specific_and_fail_closed() -> None:
    supported = _onset_v3_gripper_support_identity({"left": 0.967795, "right": 0.935644})
    assert supported["frozen_training_support"] == {
        "left": [0.4486217200756073, 1.0],
        "right": [0.5472135543823242, 0.9483147859573364],
    }
    assert supported["all_within_frozen_training_support"] is True

    outside = _onset_v3_gripper_support_identity({"left": 1.0, "right": 0.96})
    assert outside["within_frozen_training_support"] == {"left": True, "right": False}
    assert outside["all_within_frozen_training_support"] is False
    with pytest.raises(ValueError, match="left and right"):
        _onset_v3_gripper_support_identity({"left": 0.9})


def test_onset_gate_binds_interior_anchor_and_must_be_read_only(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[2] / "examples/umi_yam/onset_v3_checkpoint_gate.json"
    gate = tmp_path / "onset-v3-gate.json"
    gate.write_bytes(source.read_bytes())
    gate.chmod(0o444)
    start = (0.0, 0.05, 0.05, 0.0, 0.0, 0.0, 1.0) * 2
    identity = _load_onset_v3_gate_identity(
        report_path=gate,
        expected_report_sha256="f5841191715930db0c9edd29aa96cc45971fe1f36460c70e03f40e596ccd4d36",
        policy_start_position=start,
    )
    assert identity["yam_start_anchor"]["hardware_start_verified"] is False
    assert identity["execution"]["active_execution_rows"] == 15

    gate.chmod(0o644)
    with pytest.raises(ValueError, match="must be read-only"):
        _load_onset_v3_gate_identity(
            report_path=gate,
            expected_report_sha256="f5841191715930db0c9edd29aa96cc45971fe1f36460c70e03f40e596ccd4d36",
            policy_start_position=start,
        )


@pytest.mark.parametrize(
    ("filename", "policy_type"),
    [
        ("biyam_onset_v3_molmoact_hardware_trial.example.yaml", "molmoact2"),
        ("biyam_onset_v3_pi05_hardware_trial.example.yaml", "pi05"),
    ],
)
def test_hardware_trial_examples_parse_as_explicit_no_motion(filename: str, policy_type: str) -> None:
    config_path = Path(__file__).resolve().parents[2] / "examples/umi_yam" / filename
    cfg = draccus.parse(YamUmiDeploymentConfig, str(config_path), args=[])

    assert cfg.expected_policy_type == policy_type
    assert cfg.confirm_hardware_control is False
    assert cfg.max_steps == 1
    assert cfg.execution_horizon == 15
    assert cfg.hardware_commissioning_evidence is None
    assert cfg.expected_hardware_commissioning_evidence_sha256 == ""
    assert cfg.expected_gripper_calibration_sha256 == "0" * 64
    assert tuple(cfg.robot.policy_start_position) == (
        0.0,
        0.05,
        0.05,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.05,
        0.05,
        0.0,
        0.0,
        0.0,
        1.0,
    )


def test_confirmed_onset_motion_requires_commissioning_evidence() -> None:
    config_path = (
        Path(__file__).resolve().parents[2]
        / "examples/umi_yam/biyam_onset_v3_molmoact_hardware_trial.example.yaml"
    )
    no_motion = draccus.parse(YamUmiDeploymentConfig, str(config_path), args=[])
    with pytest.raises(ValueError, match="requires immutable hardware commissioning evidence"):
        replace(no_motion, confirm_hardware_control=True)
