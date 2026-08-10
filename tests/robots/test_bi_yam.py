#!/usr/bin/env python

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

import importlib.util
import json
import multiprocessing as mp
import sys
import threading
import time
import types
from types import SimpleNamespace

import draccus
import numpy as np
import pytest

from lerobot.robots.bi_yam import (
    BI_YAM_POLICY_END_POSITION,
    BI_YAM_POLICY_START_POSITION,
    YAM_SCALAR_KEYS,
    BiYAMFollower,
    BiYAMFollowerConfig,
    YAMArmConfig,
)
from lerobot.robots.bi_yam.config_bi_yam import YAM_JOINT_LIMITS
from lerobot.robots.bi_yam.worker import (
    ArmCommand,
    ArmCommandError,
    ArmControl,
    ArmState,
    ArmWorkerRuntime,
    ProcessArmWorker,
    _I2RTHardwareBackend,
    make_i2rt_backend,
)
from lerobot.robots.utils import make_robot_from_config
from lerobot.scripts.lerobot_calibrate import CalibrateConfig
from lerobot.utils.can import CANInterfaceInfo


class FakeBackend:
    def __init__(self, *, gripper_limits: tuple[float, float] | None = (0.0, 1.0)) -> None:
        self.positions = np.zeros(7, dtype=np.float64)
        self.gripper_limits = gripper_limits
        self.commands: list[np.ndarray] = []
        self.idle_calls = 0
        self.close_calls = 0

    def num_dofs(self) -> int:
        return 7

    def get_joint_pos(self) -> np.ndarray:
        return self.positions.copy()

    def get_robot_info(self) -> dict:
        return {"gripper_limits": self.gripper_limits}

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        self.positions = np.asarray(joint_pos, dtype=np.float64).copy()
        self.commands.append(self.positions.copy())

    def enter_gravity_comp_idle(self) -> None:
        self.idle_calls += 1

    def close(self) -> None:
        self.close_calls += 1


def install_fake_i2rt(monkeypatch, calls, backend):
    class FakeArmType:
        @classmethod
        def from_string_name(cls, value):
            return f"arm:{value}"

    class FakeGripperType:
        @classmethod
        def from_string_name(cls, value):
            return f"gripper:{value}"

    i2rt = types.ModuleType("i2rt")
    robots = types.ModuleType("i2rt.robots")
    get_robot = types.ModuleType("i2rt.robots.get_robot")
    utils = types.ModuleType("i2rt.robots.utils")
    i2rt.__path__ = []
    robots.__path__ = []
    get_robot.get_yam_robot = lambda **kwargs: calls.append(kwargs) or backend
    utils.ArmType = FakeArmType
    utils.GripperType = FakeGripperType
    monkeypatch.setitem(sys.modules, "i2rt", i2rt)
    monkeypatch.setitem(sys.modules, "i2rt.robots", robots)
    monkeypatch.setitem(sys.modules, "i2rt.robots.get_robot", get_robot)
    monkeypatch.setitem(sys.modules, "i2rt.robots.utils", utils)


def make_fake_backend(_config: YAMArmConfig) -> FakeBackend:
    return FakeBackend()


def make_slow_backend(_config: YAMArmConfig) -> FakeBackend:
    time.sleep(1.0)
    return FakeBackend()


class FakeWorker:
    def __init__(
        self,
        side: str,
        *,
        config: YAMArmConfig,
        fail_start: bool = False,
        apply_commands: bool = True,
        gripper_limits: tuple[float, float] | None = (0.0, 1.0),
    ) -> None:
        self.side = side
        self.config = config
        self.gripper_limits = gripper_limits
        self.positions = np.zeros(7, dtype=np.float64)
        self.is_alive = False
        self.armed = False
        self.idle = True
        self.fail_start = fail_start
        self.apply_commands = apply_commands
        self.fail_send = False
        self.fault: str | None = None
        self.state_sequence = 0
        self.control_sequence = 0
        self.last_applied_sequence = -1
        self.last_applied_positions: tuple[float, ...] | None = None
        self.last_applied_monotonic_ns: int | None = None
        self.last_dispatch_started_monotonic_ns: int | None = None
        self.last_driver_acknowledged_monotonic_ns: int | None = None
        self.commands: list[ArmCommand] = []
        self.disarm_calls = 0
        self.close_calls = 0

    def _state(self) -> ArmState:
        self.state_sequence += 1
        return ArmState(
            sequence=self.state_sequence,
            timestamp_ns=time.monotonic_ns(),
            positions=tuple(float(value) for value in self.positions),
            ready=self.is_alive,
            armed=self.armed,
            idle=self.idle,
            control_sequence=self.control_sequence,
            last_applied_command_sequence=self.last_applied_sequence,
            last_applied_positions=self.last_applied_positions,
            gripper_limits=self.gripper_limits,
            fault=self.fault,
            last_applied_monotonic_ns=self.last_applied_monotonic_ns,
            last_dispatch_started_monotonic_ns=self.last_dispatch_started_monotonic_ns,
            last_driver_acknowledged_monotonic_ns=self.last_driver_acknowledged_monotonic_ns,
        )

    def start(self, timeout_s: float) -> ArmState:
        del timeout_s
        if self.fail_start:
            raise RuntimeError(f"{self.side} startup failed")
        self.is_alive = True
        return self._state()

    def arm(self, timeout_s: float) -> ArmState:
        del timeout_s
        self.control_sequence += 1
        self.armed = True
        self.idle = True
        self.fault = None
        return self._state()

    def disarm(self, timeout_s: float) -> ArmState:
        del timeout_s
        self.disarm_calls += 1
        self.control_sequence += 1
        self.armed = False
        self.idle = True
        return self._state()

    def latest_state(self, timeout_s: float) -> ArmState:
        del timeout_s
        if not self.is_alive:
            raise RuntimeError(f"{self.side} worker failed")
        return self._state()

    def send_command(self, command: ArmCommand) -> None:
        if self.fail_send:
            raise RuntimeError(f"{self.side} send failed")
        self.commands.append(command)
        if self.apply_commands:
            self.positions = np.asarray(command.positions, dtype=np.float64)
        self.last_applied_sequence = command.sequence
        self.last_applied_positions = command.positions
        self.last_applied_monotonic_ns = time.monotonic_ns()
        self.last_dispatch_started_monotonic_ns = self.last_applied_monotonic_ns
        self.last_driver_acknowledged_monotonic_ns = self.last_applied_monotonic_ns
        self.idle = False

    def wait_applied(self, sequence: int, timeout_s: float) -> ArmState:
        del timeout_s
        if self.last_applied_sequence != sequence:
            raise TimeoutError(f"{self.side} command was not applied")
        return self._state()

    def close(self, timeout_s: float) -> None:
        del timeout_s
        self.close_calls += 1
        self.armed = False
        self.idle = True
        self.is_alive = False


class FakeCamera:
    def __init__(self, height: int, width: int, *, fail_connect: bool = False) -> None:
        self.height = height
        self.width = width
        self.fail_connect = fail_connect
        self.is_connected = False
        self.disconnect_calls = 0
        self.read_latest_calls = 0

    def connect(self) -> None:
        self.is_connected = True
        if self.fail_connect:
            raise RuntimeError("camera startup failed")

    def read_latest(self) -> np.ndarray:
        self.read_latest_calls += 1
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.is_connected = False


def make_config(tmp_path, **overrides) -> BiYAMFollowerConfig:
    values = {
        "id": "test_yam",
        "calibration_dir": tmp_path,
        "left_arm_config": YAMArmConfig(channel="can0", sim=True),
        "right_arm_config": YAMArmConfig(channel="can1", sim=True),
        "command_lead_time_s": 0.0,
        "left_joint_limits": [(-1.0, 1.0), (0.0, 1.0), (0.0, 1.0), *([(-1.0, 1.0)] * 3)],
        "right_joint_limits": [(-1.0, 1.0), (0.0, 1.0), (0.0, 1.0), *([(-1.0, 1.0)] * 3)],
        "max_joint_delta": 0.2,
        "max_gripper_delta": 0.1,
    }
    values.update(overrides)
    return BiYAMFollowerConfig(**values)


def make_robot(tmp_path, *, config=None, cameras=None, worker_options=None, can_discovery=None):
    workers: dict[str, FakeWorker] = {}
    worker_options = worker_options or {}

    def worker_factory(side, _config):
        worker = FakeWorker(side, config=_config, **worker_options.get(side, {}))
        workers[side] = worker
        return worker

    cameras = cameras or {}
    robot = BiYAMFollower(
        config or make_config(tmp_path),
        worker_factory=worker_factory,
        camera_factory=lambda _configs: cameras,
        **({"can_discovery": can_discovery} if can_discovery is not None else {}),
    )
    return robot, workers


def zero_action() -> dict[str, float]:
    return dict.fromkeys(YAM_SCALAR_KEYS, 0.0)


def test_config_registration_and_exact_feature_order(tmp_path):
    camera_config = SimpleNamespace(fps=30, width=640, height=360, use_rgb=True, use_depth=False)
    config = make_config(tmp_path, cameras={"top": camera_config})
    robot, _ = make_robot(tmp_path, config=config, cameras={"top": FakeCamera(360, 640)})

    assert config.type == "bi_yam_follower"
    assert list(robot.action_features) == list(YAM_SCALAR_KEYS)
    assert list(robot.observation_features) == [*YAM_SCALAR_KEYS, "top"]
    assert robot.observation_features["top"] == (360, 640, 3)


def test_standard_robot_factory_discovers_bi_yam_without_i2rt(tmp_path):
    robot = make_robot_from_config(make_config(tmp_path))

    assert isinstance(robot, BiYAMFollower)
    assert not robot.is_connected


def test_generic_config_has_portable_tested_defaults_and_factory_support(tmp_path, monkeypatch):
    config = BiYAMFollowerConfig(
        id="lab_yam",
        calibration_dir=tmp_path,
        left_arm_config=YAMArmConfig(adapter_serial="LEFT"),
        right_arm_config=YAMArmConfig(adapter_serial="RIGHT"),
    )

    assert config.type == "bi_yam_follower"
    assert config.id == "lab_yam"
    assert config.left_arm_config.adapter_serial == "LEFT"
    assert config.right_arm_config.adapter_serial == "RIGHT"
    assert config.left_arm_config.channel is None
    assert config.right_arm_config.channel is None
    assert config.left_arm_config.command_ttl_s == 1.0
    assert config.right_arm_config.command_ttl_s == 1.0
    assert config.left_arm_config.gripper_limits_override is None
    assert config.right_arm_config.gripper_limits_override is None
    assert not config.left_arm_config.allow_gripper_calibration
    assert not config.right_arm_config.allow_gripper_calibration
    assert config.calibration_side is None
    assert config.max_joint_delta == 0.1
    assert config.max_gripper_delta == 0.03
    assert (*([0.0] * 6), 1.0, *([0.0] * 6), 1.0) == BI_YAM_POLICY_START_POSITION
    assert config.policy_start_position == BI_YAM_POLICY_START_POSITION
    assert config.policy_reset_step_size == 0.01
    assert config.policy_reset_max_steps == 900
    assert config.policy_reset_fps == 30
    assert config.policy_reset_tolerance == 0.035
    assert config.policy_reset_timeout_s == 30
    assert config.gripper_state_tolerance == 0.15
    assert config.cameras == {}

    monkeypatch.setattr("lerobot.robots.bi_yam.bi_yam.make_cameras_from_configs", lambda _configs: {})
    robot = make_robot_from_config(config)

    assert isinstance(robot, BiYAMFollower)
    assert robot.calibration_fpath == tmp_path / "lab_yam.json"
    assert not robot.is_calibrated


@pytest.mark.parametrize(
    ("side", "joint_index", "widen_lower"),
    (("left", 0, True), ("right", 5, False)),
)
def test_config_rejects_per_arm_limits_wider_than_authoritative_yam(
    tmp_path,
    side,
    joint_index,
    widen_lower,
):
    limits = list(YAM_JOINT_LIMITS)
    lower, upper = limits[joint_index]
    limits[joint_index] = (
        lower - 0.001 if widen_lower else lower,
        upper if widen_lower else upper + 0.001,
    )

    with pytest.raises(
        ValueError,
        match=rf"{side}_joint_limits\[{joint_index}\] must stay within authoritative YAM limits",
    ):
        make_config(tmp_path, **{f"{side}_joint_limits": limits})


@pytest.mark.parametrize("side", ("left", "right"))
def test_config_accepts_tightened_per_arm_yam_limits(tmp_path, side):
    tightened = [(max(lower, -0.5), min(upper, 0.5)) for lower, upper in YAM_JOINT_LIMITS]

    config = make_config(tmp_path, **{f"{side}_joint_limits": tightened})

    assert getattr(config, f"{side}_joint_limits") == tightened


def test_policy_reset_reachability_uses_effective_driver_clipped_step(tmp_path) -> None:
    with pytest.raises(ValueError, match="effective driver-clipped reset step"):
        make_config(
            tmp_path,
            policy_reset_step_size=1.0,
            max_joint_delta=0.01,
            max_gripper_delta=0.01,
            policy_reset_max_steps=10,
        )


def test_generic_rollout_requires_id_scoped_calibration_before_worker_creation(tmp_path):
    created_workers = []
    config = BiYAMFollowerConfig(
        id="lab_yam",
        calibration_dir=tmp_path,
        left_arm_config=YAMArmConfig(adapter_serial="LEFT"),
        right_arm_config=YAMArmConfig(adapter_serial="RIGHT"),
    )
    robot = BiYAMFollower(
        config,
        worker_factory=lambda side, arm_config: created_workers.append((side, arm_config)),
        camera_factory=lambda _configs: {},
    )

    with pytest.raises(RuntimeError, match=str(tmp_path / "lab_yam.json")):
        robot.connect()

    assert created_workers == []


@pytest.mark.parametrize("side", ["left", "right"])
def test_generic_calibration_cli_accepts_runtime_adapter_serial(side):
    config = draccus.parse(
        CalibrateConfig,
        args=[
            "--robot.type=bi_yam_follower",
            "--robot.id=lab_yam",
            "--robot.left_arm_config.adapter_serial=LEFT",
            "--robot.right_arm_config.adapter_serial=RIGHT",
            f"--robot.calibration_side={side}",
            f"--robot.{side}_arm_config.allow_gripper_calibration=true",
        ],
    )

    assert isinstance(config.robot, BiYAMFollowerConfig)
    assert config.robot.left_arm_config.adapter_serial == "LEFT"
    assert config.robot.right_arm_config.adapter_serial == "RIGHT"
    assert config.robot.calibration_side == side
    assert getattr(config.robot, f"{side}_arm_config").gripper_limits_override is None
    assert getattr(config.robot, f"{side}_arm_config").allow_gripper_calibration


def test_generic_config_requires_runtime_robot_id():
    with pytest.raises(ValueError, match="--robot.id is required"):
        BiYAMFollowerConfig(
            left_arm_config=YAMArmConfig(adapter_serial="LEFT"),
            right_arm_config=YAMArmConfig(adapter_serial="RIGHT"),
        )


def test_generic_config_requires_runtime_hardware_identity():
    with pytest.raises(ValueError, match="left_arm_config.adapter_serial"):
        BiYAMFollowerConfig(id="lab_yam")

    left_calibration = BiYAMFollowerConfig(
        id="lab_yam",
        calibration_side="left",
        left_arm_config=YAMArmConfig(adapter_serial="LEFT", allow_gripper_calibration=True),
    )
    assert left_calibration.right_arm_config.adapter_serial is None


def test_connect_resolves_adapter_serials_to_current_socketcan_names(tmp_path):
    gripper_limits = (1.0, 0.0)
    config = BiYAMFollowerConfig(
        id="lab_yam",
        calibration_dir=tmp_path,
        left_arm_config=YAMArmConfig(
            adapter_serial="LEFT",
            gripper_limits_override=gripper_limits,
        ),
        right_arm_config=YAMArmConfig(
            adapter_serial="RIGHT",
            gripper_limits_override=gripper_limits,
        ),
    )
    interfaces = [
        CANInterfaceInfo("can0", "RIGHT", "gs_usb", True, 1_000_000, False),
        CANInterfaceInfo("can1", "LEFT", "gs_usb", True, 1_000_000, False),
    ]
    robot, workers = make_robot(tmp_path, config=config, can_discovery=lambda: interfaces)

    robot.connect()

    assert workers["left"].config.channel == "can1"
    assert workers["right"].config.channel == "can0"
    assert workers["left"].config.adapter_serial is None
    assert workers["right"].config.adapter_serial is None
    robot.disconnect()


def test_connect_rejects_unprepared_serial_selected_can_before_worker_creation(tmp_path):
    created_workers = []
    gripper_limits = (1.0, 0.0)
    config = BiYAMFollowerConfig(
        id="lab_yam",
        calibration_dir=tmp_path,
        left_arm_config=YAMArmConfig(
            adapter_serial="LEFT",
            gripper_limits_override=gripper_limits,
        ),
        right_arm_config=YAMArmConfig(
            adapter_serial="RIGHT",
            gripper_limits_override=gripper_limits,
        ),
    )
    interfaces = [
        CANInterfaceInfo("can0", "RIGHT", "gs_usb", False, None, False),
        CANInterfaceInfo("can1", "LEFT", "gs_usb", False, None, False),
    ]
    robot = BiYAMFollower(
        config,
        worker_factory=lambda side, arm_config: created_workers.append((side, arm_config)),
        camera_factory=lambda _configs: {},
        can_discovery=lambda: interfaces,
    )

    with pytest.raises(RuntimeError, match="lerobot-setup-can") as exc_info:
        robot.connect()

    assert "--left_adapter_serial=LEFT" in str(exc_info.value)
    assert "--right_adapter_serial=RIGHT" in str(exc_info.value)
    assert created_workers == []


def test_calibration_status_reflects_fixed_gripper_limits(tmp_path):
    uncalibrated = make_config(
        tmp_path,
        left_arm_config=YAMArmConfig(channel="can0"),
        right_arm_config=YAMArmConfig(channel="can1", gripper_limits_override=(1.0, 0.0)),
    )
    calibrated = make_config(
        tmp_path,
        left_arm_config=YAMArmConfig(channel="can0", gripper_limits_override=(1.0, 0.0)),
        right_arm_config=YAMArmConfig(channel="can1", gripper_limits_override=(1.0, 0.0)),
    )

    assert not make_robot(tmp_path, config=uncalibrated)[0].is_calibrated
    assert make_robot(tmp_path, config=calibrated)[0].is_calibrated


def test_single_arm_calibration_persists_and_reloads_both_gripper_limits(tmp_path):
    camera = FakeCamera(3, 4)
    camera_config = SimpleNamespace(fps=30, width=4, height=3, use_rgb=True, use_depth=False)
    left_config = make_config(
        tmp_path,
        id="lab_yam",
        calibration_side="left",
        left_arm_config=YAMArmConfig(channel="can_left", allow_gripper_calibration=True),
        right_arm_config=YAMArmConfig(channel="can_right"),
        cameras={"top": camera_config},
    )
    left_robot, left_workers = make_robot(
        tmp_path,
        config=left_config,
        cameras={"top": camera},
        worker_options={"left": {"gripper_limits": (1.25, -0.5)}},
    )

    left_robot.connect(calibrate=False)
    assert set(left_workers) == {"left"}
    assert not camera.is_connected
    with pytest.raises(RuntimeError, match="Policy control is disabled"):
        left_robot.arm()
    left_robot.calibrate()
    left_robot.disconnect()

    calibration_path = tmp_path / "lab_yam.json"
    assert json.loads(calibration_path.read_text()) == {
        "left": {"gripper_limits": [1.25, -0.5]},
    }

    right_config = make_config(
        tmp_path,
        id="lab_yam",
        calibration_side="right",
        left_arm_config=YAMArmConfig(channel="can_left"),
        right_arm_config=YAMArmConfig(channel="can_right", allow_gripper_calibration=True),
    )
    right_robot, right_workers = make_robot(
        tmp_path,
        config=right_config,
        worker_options={"right": {"gripper_limits": (-1.0, 0.75)}},
    )

    right_robot.connect(calibrate=False)
    assert set(right_workers) == {"right"}
    right_robot.calibrate()
    right_robot.disconnect()

    assert json.loads(calibration_path.read_text()) == {
        "left": {"gripper_limits": [1.25, -0.5]},
        "right": {"gripper_limits": [-1.0, 0.75]},
    }

    rollout_config = make_config(
        tmp_path,
        id="lab_yam",
        left_arm_config=YAMArmConfig(channel="can_left"),
        right_arm_config=YAMArmConfig(channel="can_right"),
    )
    rollout_robot, rollout_workers = make_robot(tmp_path, config=rollout_config)

    assert rollout_robot.is_calibrated
    rollout_robot.connect()
    assert rollout_workers["left"].config.gripper_limits_override == (1.25, -0.5)
    assert rollout_workers["right"].config.gripper_limits_override == (-1.0, 0.75)
    assert not rollout_workers["left"].config.allow_gripper_calibration
    assert not rollout_workers["right"].config.allow_gripper_calibration
    rollout_robot.disconnect()


def test_explicit_gripper_limits_override_persisted_calibration(tmp_path):
    calibration_path = tmp_path / "lab_yam.json"
    calibration_path.write_text(
        json.dumps(
            {
                "left": {"gripper_limits": [1.25, -0.5]},
                "right": {"gripper_limits": [-1.0, 0.75]},
            }
        )
    )
    config = make_config(
        tmp_path,
        id="lab_yam",
        left_arm_config=YAMArmConfig(channel="can_left", gripper_limits_override=(2.0, -2.0)),
        right_arm_config=YAMArmConfig(channel="can_right"),
    )
    robot, workers = make_robot(tmp_path, config=config)

    robot.connect()

    assert workers["left"].config.gripper_limits_override == (2.0, -2.0)
    assert workers["right"].config.gripper_limits_override == (-1.0, 0.75)
    robot.disconnect()


def test_calibration_mode_requires_selected_arm_opt_in_before_worker_creation(tmp_path):
    created_workers = []
    config = make_config(
        tmp_path,
        calibration_side="left",
        left_arm_config=YAMArmConfig(channel="can_left"),
        right_arm_config=YAMArmConfig(channel="can_right"),
    )
    robot = BiYAMFollower(
        config,
        worker_factory=lambda side, arm_config: created_workers.append((side, arm_config)),
        camera_factory=lambda _configs: {},
    )

    with pytest.raises(RuntimeError, match="allow_gripper_calibration=true"):
        robot.connect(calibrate=False)

    assert created_workers == []


def test_normal_rollout_rejects_moving_calibration_opt_in_without_calibration_side(tmp_path):
    created_workers = []
    config = make_config(
        tmp_path,
        left_arm_config=YAMArmConfig(channel="can_left", allow_gripper_calibration=True),
        right_arm_config=YAMArmConfig(channel="can_right", gripper_limits_override=(1.0, 0.0)),
    )
    robot = BiYAMFollower(
        config,
        worker_factory=lambda side, arm_config: created_workers.append((side, arm_config)),
        camera_factory=lambda _configs: {},
    )

    with pytest.raises(RuntimeError, match="only allowed through single-arm calibration mode"):
        robot.connect()

    assert created_workers == []


def test_connect_preflights_both_arms_before_creating_workers_or_connecting_cameras(tmp_path):
    created_workers = []
    camera = FakeCamera(3, 4)
    camera_config = SimpleNamespace(fps=30, width=4, height=3, use_rgb=True, use_depth=False)
    config = make_config(
        tmp_path,
        left_arm_config=YAMArmConfig(channel="can_left", gripper_limits_override=(1.0, 0.0)),
        right_arm_config=YAMArmConfig(channel="can_right"),
        cameras={"top": camera_config},
    )
    robot = BiYAMFollower(
        config,
        worker_factory=lambda side, arm_config: created_workers.append((side, arm_config)),
        camera_factory=lambda _configs: {"top": camera},
    )

    with pytest.raises(RuntimeError, match="Unsafe right arm configuration"):
        robot.connect()

    assert created_workers == []
    assert not camera.is_connected


def test_connect_stays_disarmed_and_observation_uses_ordered_state_and_camera(tmp_path):
    camera_config = SimpleNamespace(fps=30, width=4, height=3, use_rgb=True, use_depth=False)
    camera = FakeCamera(3, 4)
    config = make_config(tmp_path, cameras={"top": camera_config})
    robot, workers = make_robot(tmp_path, config=config, cameras={"top": camera})

    robot.connect()
    workers["left"].positions = np.arange(7, dtype=np.float64) / 10
    workers["right"].positions = np.arange(7, 14, dtype=np.float64) / 10
    observation = robot.get_observation()

    assert not robot.is_armed
    assert list(observation) == [*YAM_SCALAR_KEYS, "top"]
    expected = np.arange(14, dtype=np.float64) / 10
    expected[13] = 1.0
    assert [observation[key] for key in YAM_SCALAR_KEYS] == pytest.approx(expected)
    assert observation["top"].shape == (3, 4, 3)
    assert camera.read_latest_calls == 1
    assert set(robot.state_metadata) == {"left", "right"}

    with pytest.raises(RuntimeError, match="arm it locally"):
        robot.send_action(zero_action())
    robot.disconnect()


def test_gripper_stop_overtravel_is_accepted_and_clipped_in_observations(tmp_path):
    robot, workers = make_robot(tmp_path)
    robot.connect()
    workers["left"].positions[6] = 1.1
    workers["right"].positions[6] = -0.1

    observation = robot.get_observation()
    robot.arm()

    assert observation["left_gripper.pos"] == 1.0
    assert observation["right_gripper.pos"] == 0.0
    assert robot.is_armed
    robot.disconnect()


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda action: action.pop("left_joint_0.pos"), "missing"),
        (lambda action: action.update({"unexpected.pos": 0.0}), "extra"),
        (lambda action: action.update({"left_joint_0.pos": float("nan")}), "finite"),
        (lambda action: action.update({"left_joint_0.pos": object()}), "numeric"),
    ],
)
def test_send_action_rejects_invalid_schema_and_values(tmp_path, mutate, message):
    robot, workers = make_robot(tmp_path)
    robot.connect()
    robot.arm()
    action = zero_action()
    mutate(action)

    with pytest.raises(ValueError, match=message):
        robot.send_action(action)
    assert workers["left"].commands == []
    assert workers["right"].commands == []
    assert not robot.is_armed
    robot.disconnect()


def test_send_action_clips_limits_and_delta_and_synchronizes_workers(tmp_path):
    telemetry_path = tmp_path / "control.jsonl"
    config = make_config(tmp_path, control_telemetry_path=telemetry_path)
    robot, workers = make_robot(tmp_path, config=config)
    robot.connect()
    robot.arm()
    action = dict.fromkeys(YAM_SCALAR_KEYS, 10.0)

    applied = robot.send_action(action)

    expected = [0.2] * 6 + [0.1] + [0.2] * 6 + [0.1]
    assert list(applied) == list(YAM_SCALAR_KEYS)
    assert list(applied.values()) == pytest.approx(expected)
    left_command = workers["left"].commands[-1]
    right_command = workers["right"].commands[-1]
    assert left_command.sequence == right_command.sequence
    assert left_command.execute_at_ns == right_command.execute_at_ns
    assert (*left_command.positions, *right_command.positions) == pytest.approx(expected)
    robot.disconnect()

    record = json.loads(telemetry_path.read_text().strip())
    assert record["names"] == list(YAM_SCALAR_KEYS)
    assert record["requested"] == pytest.approx([10.0] * 14)
    assert record["applied"] == pytest.approx(expected)
    assert record["state_before_dispatch"] == pytest.approx([0.0] * 14)
    assert record["state_at_acknowledgement"] == pytest.approx(expected)
    assert record["applied_vs_ack_state_error"] == pytest.approx([0.0] * 14)
    assert record["acknowledgement_confirms_achievement"] is False
    assert record["bound_clipped"] == list(YAM_SCALAR_KEYS)
    assert record["delta_clipped"] == list(YAM_SCALAR_KEYS)


def test_pre_dispatch_validator_receives_normal_send_target(tmp_path):
    validated = []
    config = make_config(tmp_path, max_joint_delta=1.0, max_gripper_delta=1.0)
    robot, workers = make_robot(tmp_path, config=config)
    robot.set_pre_dispatch_validator(lambda command: validated.append(dict(command)))
    robot.connect()
    robot.arm()
    requested = dict.fromkeys(YAM_SCALAR_KEYS, 0.05)

    applied = robot.send_action(requested)

    assert validated == [applied]
    dispatched = (*workers["left"].commands[-1].positions, *workers["right"].commands[-1].positions)
    assert tuple(validated[0].values()) == pytest.approx(dispatched)
    robot.disconnect()


def test_dispatch_timestamp_is_allocated_after_endpoint_validation(tmp_path):
    events = []
    config = make_config(tmp_path, command_lead_time_s=0.01)
    robot, workers = make_robot(tmp_path, config=config)

    def validate(_command):
        # Discard state-refresh clock reads; only ordering at the dispatch
        # boundary matters here.
        events.clear()
        events.append("validate")

    def monotonic_ns():
        # ``send_action`` also reads this clock while checking fresh measured
        # state. Keep those reads realistic, then expose the sentinel only
        # after the validator marks the dispatch boundary.
        if events == ["validate"]:
            events.append("clock")
            return 1_000
        return time.monotonic_ns()

    robot.set_pre_dispatch_validator(validate)
    robot.connect()
    robot.arm()
    # Install the fake dispatch clock only after startup freshness checks, which
    # must compare worker timestamps against the real monotonic clock.
    robot._monotonic_ns = monotonic_ns

    robot.send_action(zero_action())

    assert events[:2] == ["validate", "clock"]
    expected_execute_at_ns = 1_000 + 10_000_000
    assert workers["left"].commands[-1].execute_at_ns == expected_execute_at_ns
    assert workers["right"].commands[-1].execute_at_ns == expected_execute_at_ns
    robot.disconnect()


def test_pre_dispatch_validator_receives_driver_clipped_target(tmp_path):
    validated = []
    config = make_config(tmp_path, max_joint_delta=0.2, max_gripper_delta=0.1)
    robot, workers = make_robot(tmp_path, config=config)
    robot.set_pre_dispatch_validator(lambda command: validated.append(dict(command)))
    robot.connect()
    robot.arm()

    applied = robot.send_action(dict.fromkeys(YAM_SCALAR_KEYS, 20.0))

    expected = [0.2] * 6 + [0.1] + [0.2] * 6 + [0.1]
    assert tuple(applied.values()) == pytest.approx(expected)
    assert tuple(validated[0].values()) == pytest.approx(expected)
    dispatched = (*workers["left"].commands[-1].positions, *workers["right"].commands[-1].positions)
    assert dispatched == pytest.approx(expected)
    robot.disconnect()


def test_pre_dispatch_validator_failure_prevents_send_and_disarms(tmp_path):
    robot, workers = make_robot(tmp_path)

    def reject(_command):
        raise RuntimeError("unsafe endpoint")

    robot.set_pre_dispatch_validator(reject)
    robot.connect()
    robot.arm()

    with pytest.raises(RuntimeError, match="unsafe endpoint"):
        robot.send_action(zero_action())

    assert workers["left"].commands == []
    assert workers["right"].commands == []
    assert not robot.is_armed
    robot.disconnect()


def test_operational_limit_clipping_is_reflected_in_returned_action(tmp_path):
    config = make_config(tmp_path, max_joint_delta=10.0, max_gripper_delta=10.0)
    robot, _ = make_robot(tmp_path, config=config)
    robot.connect()
    robot.arm()

    applied = robot.send_action(dict.fromkeys(YAM_SCALAR_KEYS, 20.0))

    assert list(applied.values()) == pytest.approx([1.0] * 6 + [1.0] + [1.0] * 6 + [1.0])
    robot.disconnect()


def test_policy_reset_reaches_configured_pose_with_molmoact2_sized_steps(tmp_path):
    target = BI_YAM_POLICY_START_POSITION
    telemetry_path = tmp_path / "reset-control.jsonl"
    config = make_config(
        tmp_path,
        control_telemetry_path=telemetry_path,
        policy_start_position=target,
        policy_reset_fps=10_000,
        policy_reset_tolerance=0.001,
        policy_reset_timeout_s=0.5,
    )
    robot, workers = make_robot(tmp_path, config=config)
    robot.connect()
    initial = np.asarray([*([0.025] * 6), 0.95, *([0.025] * 6), 0.05], dtype=np.float64)
    workers["left"].positions = initial[:7].copy()
    workers["right"].positions = initial[7:].copy()
    robot.arm()

    reset_position = robot.reset_for_policy()

    assert reset_position == dict(zip(YAM_SCALAR_KEYS, target, strict=True))
    commands = np.asarray(
        [
            (*left.positions, *right.positions)
            for left, right in zip(
                workers["left"].commands,
                workers["right"].commands,
                strict=True,
            )
        ]
    )
    trajectory = np.vstack([initial, commands])
    assert np.max(np.abs(np.diff(trajectory, axis=0))) <= 0.0100001
    assert commands[-1] == pytest.approx(target)
    assert robot.is_armed
    robot.disconnect()

    records = [json.loads(line) for line in telemetry_path.read_text().splitlines()]
    assert len(records) == len(commands)
    assert {record["phase"] for record in records} <= {"reset", "reset_hold"}
    assert records[-1]["applied"] == pytest.approx(target)


def test_policy_reset_validates_every_final_endpoint_before_dispatch(tmp_path):
    validated = []
    target = BI_YAM_POLICY_START_POSITION
    config = make_config(
        tmp_path,
        policy_start_position=target,
        policy_reset_step_size=0.5,
        policy_reset_fps=10_000,
        policy_reset_tolerance=0.001,
        policy_reset_timeout_s=0.5,
    )
    robot, workers = make_robot(tmp_path, config=config)
    initial = np.asarray([*([0.5] * 6), 0.0, *([0.5] * 6), 0.0], dtype=np.float64)
    robot.set_pre_dispatch_validator(lambda command: validated.append(dict(command)))
    robot.connect()
    workers["left"].positions = initial[:7].copy()
    workers["right"].positions = initial[7:].copy()
    robot.arm()

    robot.reset_for_policy()

    dispatched = [
        (*left.positions, *right.positions)
        for left, right in zip(
            workers["left"].commands,
            workers["right"].commands,
            strict=True,
        )
    ]
    # Every dispatched endpoint is checked, followed by one check of the fresh
    # measured endpoint that proves reset completion without another command.
    assert len(validated) == len(dispatched) + 1
    assert dispatched
    for checked, command in zip(validated[:-1], dispatched, strict=True):
        assert tuple(checked.values()) == pytest.approx(command)
    # Reset requested a 0.5 step, but the exact validated/dispatched endpoint
    # reflects the driver's tighter per-call joint/gripper caps.
    assert validated[0]["left_joint_0.pos"] == pytest.approx(0.3)
    assert validated[0]["left_gripper.pos"] == pytest.approx(0.1)
    assert tuple(validated[-1].values()) == pytest.approx(target)
    robot.disconnect()


def test_policy_reset_validates_measured_endpoint_when_already_at_target(tmp_path):
    checked = []
    robot, workers = make_robot(tmp_path)

    def reject(command):
        checked.append(dict(command))
        raise RuntimeError("unsafe reset endpoint")

    robot.set_pre_dispatch_validator(reject)
    robot.connect()
    target = np.asarray(BI_YAM_POLICY_START_POSITION, dtype=np.float64)
    workers["left"].positions = target[:7].copy()
    workers["right"].positions = target[7:].copy()
    robot.arm()

    with pytest.raises(RuntimeError, match="unsafe reset endpoint"):
        robot.reset_for_policy()

    assert len(checked) == 1
    assert tuple(checked[0].values()) == pytest.approx(target)
    assert workers["left"].commands == []
    assert workers["right"].commands == []
    assert not robot.is_armed
    robot.disconnect()


def test_policy_end_reset_uses_fixed_home_pose_instead_of_configured_start(tmp_path):
    configured_start = (
        0.0,
        0.4,
        0.8,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.4,
        0.8,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    config = make_config(
        tmp_path,
        policy_start_position=configured_start,
        policy_reset_fps=10_000,
        policy_reset_tolerance=0.001,
        policy_reset_timeout_s=0.5,
    )
    robot, workers = make_robot(tmp_path, config=config)
    robot.connect()
    workers["left"].positions = np.asarray(configured_start[:7], dtype=np.float64)
    workers["right"].positions = np.asarray(configured_start[7:], dtype=np.float64)
    robot.arm()

    reset_position = robot.reset_after_policy()

    assert reset_position == dict(zip(YAM_SCALAR_KEYS, BI_YAM_POLICY_END_POSITION, strict=True))
    assert (*workers["left"].commands[-1].positions, *workers["right"].commands[-1].positions) == (
        pytest.approx(BI_YAM_POLICY_END_POSITION)
    )
    assert tuple(reset_position.values()) != configured_start
    robot.disconnect()


def test_policy_reset_timeout_disarms_both_workers(tmp_path):
    target = (0.1, *([0.0] * 13))
    config = make_config(
        tmp_path,
        policy_start_position=target,
        policy_reset_fps=1_000,
        policy_reset_timeout_s=0.01,
    )
    robot, workers = make_robot(
        tmp_path,
        config=config,
        worker_options={
            "left": {"apply_commands": False},
            "right": {"apply_commands": False},
        },
    )
    robot.connect()
    robot.arm()

    with pytest.raises(TimeoutError, match="did not reach its policy start position"):
        robot.reset_for_policy()

    commands = np.asarray(
        [
            (*left.positions, *right.positions)
            for left, right in zip(
                workers["left"].commands,
                workers["right"].commands,
                strict=True,
            )
        ]
    )
    assert len(commands) > 1
    # The fake plant never moves. Every retry must therefore remain one reset
    # step from the same fresh measured state rather than following an open-loop
    # trajectory farther away.
    assert np.max(np.abs(commands)) <= config.policy_reset_step_size + 1e-9
    assert not robot.is_armed
    assert all(worker.disarm_calls >= 1 for worker in workers.values())
    robot.disconnect()


def test_policy_reset_is_a_noop_without_configured_pose(tmp_path):
    robot, workers = make_robot(tmp_path, config=make_config(tmp_path, policy_start_position=None))
    robot.connect()
    robot.arm()

    assert robot.reset_for_policy() is None
    assert workers["left"].commands == []
    assert workers["right"].commands == []
    robot.disconnect()


def test_worker_runtime_uses_latest_command_and_stale_watchdog_enters_idle():
    backend = FakeBackend()
    application_clock_ns = 1
    runtime = ArmWorkerRuntime(
        backend,
        command_ttl_s=0.1,
        monotonic_ns=lambda: application_clock_ns,
    )
    runtime.handle_control(ArmControl(sequence=1, operation="arm"))
    runtime.offer_command(ArmCommand(sequence=1, execute_at_ns=0, positions=(0.1,) * 7))
    runtime.offer_command(ArmCommand(sequence=2, execute_at_ns=0, positions=(0.2,) * 7))

    applied = runtime.tick(now_ns=1)
    stale = runtime.tick(now_ns=100_000_002)

    assert len(backend.commands) == 1
    assert backend.commands[0] == pytest.approx(np.full(7, 0.2))
    assert applied.last_applied_command_sequence == 2
    assert applied.gripper_limits == (0.0, 1.0)
    assert stale.fault == "command_ttl_expired"
    assert not stale.armed
    assert stale.idle
    assert backend.idle_calls >= 2
    runtime.close()
    assert backend.close_calls == 1


def test_worker_expiry_boundary_checks_immediately_before_driver_call() -> None:
    deadline_ns = 50
    backend = FakeBackend()
    clock_values = iter((deadline_ns, deadline_ns))
    runtime = ArmWorkerRuntime(
        backend,
        command_ttl_s=1.0,
        monotonic_ns=lambda: next(clock_values),
    )
    runtime.handle_control(ArmControl(sequence=1, operation="arm"))
    runtime.offer_command(
        ArmCommand(
            sequence=1,
            execute_at_ns=0,
            positions=(0.1,) * 7,
            expires_at_ns=deadline_ns,
        )
    )

    at_boundary = runtime.tick(now_ns=0)

    assert len(backend.commands) == 1
    assert at_boundary.last_dispatch_started_monotonic_ns == deadline_ns
    assert at_boundary.last_driver_acknowledged_monotonic_ns == deadline_ns
    assert at_boundary.fault is None

    expired_backend = FakeBackend()
    expired_runtime = ArmWorkerRuntime(
        expired_backend,
        command_ttl_s=1.0,
        monotonic_ns=lambda: deadline_ns + 1,
    )
    expired_runtime.handle_control(ArmControl(sequence=1, operation="arm"))
    expired_runtime.offer_command(
        ArmCommand(
            sequence=1,
            execute_at_ns=0,
            positions=(0.1,) * 7,
            expires_at_ns=deadline_ns,
        )
    )

    expired = expired_runtime.tick(now_ns=0)

    assert expired_backend.commands == []
    assert expired.fault == "command_expired_before_worker_dispatch"
    assert expired.fault_monotonic_ns == deadline_ns + 1
    assert not expired.armed


def test_worker_records_late_driver_acknowledgement_and_disarms() -> None:
    deadline_ns = 50
    backend = FakeBackend()
    clock_values = iter((deadline_ns, deadline_ns + 1))
    runtime = ArmWorkerRuntime(
        backend,
        command_ttl_s=1.0,
        monotonic_ns=lambda: next(clock_values),
    )
    runtime.handle_control(ArmControl(sequence=1, operation="arm"))
    runtime.offer_command(
        ArmCommand(
            sequence=1,
            execute_at_ns=0,
            positions=(0.1,) * 7,
            expires_at_ns=deadline_ns,
        )
    )

    late_acknowledgement = runtime.tick(now_ns=0)

    # The driver call started exactly at the allowed boundary, so it did run.
    # Its return is only an acknowledgement (not proof of physical motion), and
    # returning after expiry must still fault and enter safe idle.
    assert len(backend.commands) == 1
    assert late_acknowledgement.last_dispatch_started_monotonic_ns == deadline_ns
    assert late_acknowledgement.last_driver_acknowledged_monotonic_ns == deadline_ns + 1
    assert late_acknowledgement.fault == "command_acknowledged_after_expiry"
    assert late_acknowledgement.fault_monotonic_ns == deadline_ns + 1
    assert not late_acknowledgement.armed
    assert late_acknowledgement.idle


def test_guarded_dispatch_requires_reviewed_margin_before_either_worker_queue(tmp_path) -> None:
    robot, workers = make_robot(tmp_path)
    robot.connect()
    robot.arm()
    deadline_ns = time.monotonic_ns() + 20_000_000

    with pytest.raises(RuntimeError, match="required reviewed worker dispatch margin"):
        robot.send_action_from_measured_state(
            lambda _measured: zero_action(),
            expires_at_monotonic_ns=deadline_ns,
            required_expiry_margin_ns=50_000_000,
        )

    assert workers["left"].commands == []
    assert workers["right"].commands == []
    assert not robot.is_armed
    assert robot.last_dispatch_timing["required_expiry_margin_ns"] == 50_000_000
    robot.disconnect()


def test_guarded_dispatch_uses_exact_measured_state_and_records_worker_boundaries(tmp_path) -> None:
    robot, workers = make_robot(tmp_path)
    robot.connect()
    robot.arm()
    captured = []
    deadline_ns = time.monotonic_ns() + 1_000_000_000

    applied = robot.send_action_from_measured_state(
        lambda measured: captured.append(dict(measured)) or dict(measured),
        expires_at_monotonic_ns=deadline_ns,
        required_expiry_margin_ns=50_000_000,
    )

    assert captured == [applied]
    timing = robot.last_dispatch_timing
    assert timing["state_before_dispatch"] == pytest.approx([captured[0][key] for key in YAM_SCALAR_KEYS])
    assert set(timing["worker_dispatch_started_monotonic_ns"]) == {"left", "right"}
    assert set(timing["worker_driver_acknowledged_monotonic_ns"]) == {"left", "right"}
    for side in ("left", "right"):
        assert (
            timing["worker_dispatch_started_monotonic_ns"][side]
            <= timing["worker_driver_acknowledged_monotonic_ns"][side]
            <= deadline_ns
        )
        assert workers[side].commands[-1].expires_at_ns == deadline_ns
    robot.disconnect()


def test_two_arm_expiry_asymmetry_is_nonatomic_and_idles_both(tmp_path) -> None:
    robot, workers = make_robot(tmp_path)
    robot.connect()
    robot.arm()
    deadline_ns = time.monotonic_ns() + 1_000_000_000

    def reject_right(sequence: int, timeout_s: float) -> ArmState:
        del timeout_s
        return_state = ArmState(
            sequence=99,
            timestamp_ns=deadline_ns + 1,
            positions=tuple(workers["right"].positions),
            ready=True,
            armed=False,
            idle=True,
            control_sequence=workers["right"].control_sequence,
            last_applied_command_sequence=sequence - 1,
            last_applied_positions=None,
            fault="command_expired_before_worker_dispatch",
            fault_monotonic_ns=deadline_ns + 1,
        )
        raise ArmCommandError("right", return_state)

    workers["right"].wait_applied = reject_right

    with pytest.raises(ArmCommandError, match="right arm command failed"):
        robot.send_action_from_measured_state(
            lambda _measured: zero_action(),
            expires_at_monotonic_ns=deadline_ns,
            required_expiry_margin_ns=50_000_000,
        )

    timing = robot.last_dispatch_timing
    assert set(timing["worker_dispatch_started_monotonic_ns"]) == {"left"}
    assert timing["worker_rejected_monotonic_ns"] == {"right": deadline_ns + 1}
    assert not robot.is_armed
    assert workers["left"].disarm_calls >= 1
    assert workers["right"].disarm_calls >= 1
    robot.disconnect()


def test_i2rt_hardware_backend_stops_threads_before_closing_can():
    class FakeMotorChain:
        def __init__(self) -> None:
            self.running = True

        def control_loop(self) -> None:
            while self.running:
                time.sleep(0.001)

    class FakeI2RTRobot(FakeBackend):
        def __init__(self, motor_chain: FakeMotorChain, control_thread: threading.Thread) -> None:
            super().__init__()
            self.motor_chain = motor_chain
            self._control_thread = control_thread
            self._stop_event = threading.Event()
            self._server_thread = threading.Thread(target=self._server_loop, name="robot_server")
            self._server_thread.start()

        def _server_loop(self) -> None:
            self._stop_event.wait()

        def close(self) -> None:
            assert not self._server_thread.is_alive()
            assert not self._control_thread.is_alive()
            super().close()

    motor_chain = FakeMotorChain()
    control_thread = threading.Thread(target=motor_chain.control_loop, name="motor_control")
    control_thread.start()
    backend = FakeI2RTRobot(motor_chain, control_thread)
    wrapped = _I2RTHardwareBackend(backend, motor_chain, (control_thread,), shutdown_timeout_s=0.5)

    wrapped.close()

    assert backend.close_calls == 1
    assert not backend._server_thread.is_alive()
    assert not control_thread.is_alive()


def test_process_worker_ipc_applies_command_and_reports_ttl_fault():
    worker = ProcessArmWorker(
        "left",
        YAMArmConfig(channel="unused", sim=True, command_ttl_s=0.05, worker_poll_interval_s=0.002),
        context=mp.get_context("spawn"),
        backend_factory=make_fake_backend,
    )
    try:
        # Spawn imports are noticeably slower on the Raspberry Pi client.
        assert worker.start(timeout_s=10.0).ready
        assert worker.arm(timeout_s=1.0).armed
        command = ArmCommand(sequence=7, execute_at_ns=time.monotonic_ns(), positions=(0.25,) * 7)
        worker.send_command(command)
        applied = worker.wait_applied(sequence=7, timeout_s=1.0)
        assert applied.last_applied_positions == command.positions

        deadline = time.monotonic() + 1.0
        while True:
            state = worker.latest_state(timeout_s=0.2)
            if state.fault == "command_ttl_expired":
                break
            if time.monotonic() >= deadline:
                pytest.fail("worker command TTL did not expire")
        assert not state.armed
        assert state.idle
    finally:
        worker.close(timeout_s=1.0)


def test_process_worker_startup_timeout_terminates_child():
    worker = ProcessArmWorker(
        "left",
        YAMArmConfig(channel="unused"),
        context=mp.get_context("spawn"),
        backend_factory=make_slow_backend,
    )

    with pytest.raises(TimeoutError, match="Timed out"):
        worker.start(timeout_s=0.05)
    assert not worker.is_alive


@pytest.mark.skipif(importlib.util.find_spec("i2rt") is None, reason="i2rt is optional")
def test_i2rt_sim_backend_matches_worker_contract():
    backend = make_i2rt_backend(YAMArmConfig(channel="unused", sim=True))
    runtime = ArmWorkerRuntime(backend, command_ttl_s=0.2)
    try:
        initial = runtime.snapshot(time.monotonic_ns())
        assert len(initial.positions) == 7
        runtime.handle_control(ArmControl(sequence=1, operation="arm"))
        command = ArmCommand(
            sequence=1,
            execute_at_ns=0,
            positions=initial.positions,
        )
        runtime.offer_command(command)
        applied = runtime.tick(time.monotonic_ns())
        assert applied.last_applied_command_sequence == 1
        assert applied.last_applied_positions == initial.positions
    finally:
        runtime.close()


def test_worker_failure_idles_peer_and_disconnect_still_cleans_up(tmp_path):
    robot, workers = make_robot(tmp_path)
    robot.connect()
    robot.arm()
    workers["left"].is_alive = False

    with pytest.raises(RuntimeError, match="dead arm workers"):
        robot.get_observation()

    assert robot.is_connected
    assert not robot.is_armed
    assert workers["right"].disarm_calls >= 1
    robot.disconnect()
    assert workers["left"].close_calls == 1
    assert workers["right"].close_calls == 1


def test_local_rearm_clears_recoverable_worker_fault(tmp_path):
    robot, workers = make_robot(tmp_path)
    robot.connect()
    robot.arm()
    for worker in workers.values():
        worker.armed = False
        worker.idle = True
        worker.fault = "command_ttl_expired"

    robot.arm()

    assert robot.is_armed
    assert all(worker.armed and worker.fault is None for worker in workers.values())
    robot.disconnect()


def test_connect_failure_transactionally_closes_workers_and_cameras(tmp_path):
    first_camera = FakeCamera(3, 4)
    failing_camera = FakeCamera(3, 4, fail_connect=True)
    camera_configs = {
        "top": SimpleNamespace(fps=30, width=4, height=3),
        "left": SimpleNamespace(fps=30, width=4, height=3),
    }
    robot, workers = make_robot(
        tmp_path,
        config=make_config(tmp_path, cameras=camera_configs),
        cameras={"top": first_camera, "left": failing_camera},
    )

    with pytest.raises(RuntimeError, match="camera startup failed"):
        robot.connect()

    assert not robot.is_connected
    assert first_camera.disconnect_calls == 1
    assert failing_camera.disconnect_calls == 1
    assert workers["left"].close_calls == 1
    assert workers["right"].close_calls == 1


def test_second_worker_start_failure_closes_first_worker(tmp_path):
    robot, workers = make_robot(tmp_path, worker_options={"right": {"fail_start": True}})

    with pytest.raises(RuntimeError, match="right startup failed"):
        robot.connect()

    assert workers["left"].close_calls == 1
    assert workers["right"].close_calls == 1


def test_i2rt_factory_forwards_hardware_and_sim_configuration(monkeypatch):
    calls = []
    backend = FakeBackend()
    install_fake_i2rt(monkeypatch, calls, backend)

    result = make_i2rt_backend(
        YAMArmConfig(
            channel="can7",
            arm_type="yam_pro",
            gripper_type="linear_4310",
            sim=True,
            enable_auto_recovery=True,
        )
    )

    assert result is backend
    assert calls == [
        {
            "channel": "can7",
            "arm_type": "arm:yam_pro",
            "gripper_type": "gripper:linear_4310",
            "zero_gravity_mode": True,
            "gripper_limits_override": None,
            "sim": True,
            "enable_auto_recovery": True,
        }
    ]


def test_i2rt_factory_forwards_gripper_limits_without_calibration(monkeypatch):
    calls = []
    backend = FakeBackend()
    install_fake_i2rt(monkeypatch, calls, backend)

    result = make_i2rt_backend(YAMArmConfig(channel="can7", gripper_limits_override=(1.25, -0.75)))

    assert result is backend
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0]["gripper_limits_override"], [1.25, -0.75])
    assert calls[0]["sim"] is False


def test_i2rt_factory_refuses_unsupervised_hardware_gripper_calibration(monkeypatch):
    calls = []
    install_fake_i2rt(monkeypatch, calls, FakeBackend())

    with pytest.raises(RuntimeError, match="Refusing to open can7"):
        make_i2rt_backend(YAMArmConfig(channel="can7"))

    assert calls == []


def test_i2rt_factory_requires_explicit_opt_in_for_moving_calibration(monkeypatch):
    calls = []
    backend = FakeBackend()
    install_fake_i2rt(monkeypatch, calls, backend)

    result = make_i2rt_backend(YAMArmConfig(channel="can7", allow_gripper_calibration=True))

    assert result is backend
    assert len(calls) == 1
    assert calls[0]["gripper_limits_override"] is None


@pytest.mark.parametrize(
    "limits",
    [
        (0.0, 0.0),
        (float("nan"), 1.0),
        (0.0, float("inf")),
        (0.0,),
    ],
)
def test_arm_config_rejects_invalid_gripper_limit_overrides(limits):
    with pytest.raises(ValueError, match="gripper_limits_override"):
        YAMArmConfig(channel="can7", gripper_limits_override=limits)


def test_arm_config_rejects_override_with_calibration_opt_in():
    with pytest.raises(ValueError, match="cannot be set together"):
        YAMArmConfig(
            channel="can7",
            gripper_limits_override=(1.0, 0.0),
            allow_gripper_calibration=True,
        )
