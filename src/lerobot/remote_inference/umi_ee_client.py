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
"""Reference remote-inference client for bimanual UMI end-effector-pose policies.

The stock ``lerobot-rollout`` embodiment builder only advertises robot features named
``*.pos``/``*.vel``, so it can never open a session against a policy whose manifest
uses bare end-effector feature names (``umi1_x`` ... ``umi2_gripper``). This module is
the supported path for such policies: a thin client that owns the embodiment manifest
and documents the state/action conventions the policy was trained with.

For ``dual-lidar-umi-currentrel-r6d-v1``, pass
``UMI_CURRENTREL_R6D_STATE_FEATURE_NAMES`` and
``UMI_CURRENTREL_R6D_ACTION_FEATURE_NAMES`` separately. That checkpoint uses 20-D
``xyz + Rotation6D + gripper`` rows; the runtime bridge owns their SE(3)
interpretation. The defaults below intentionally preserve this older 14-D contract:

- ``state``: float32 (14,) = ``[umi1 xyz, umi1 rotvec, umi1_gripper, umi2 xyz,
  umi2 rotvec, umi2_gripper]``. Each arm's pose is expressed in that arm's
  EPISODE-START frame (frame 0 = identity): ``p_t = R0^T (p_world_t - p_world_0)``,
  ``r_t = rotvec(R0^T R_world_t)``. Grippers are ``width_mm / gripper_scale_mm``
  clipped to [0, 1] (open is ~1.0).
- returned actions: float32 (horizon, 14) ABSOLUTE next-frame targets in the same
  frames — NOT deltas. Execute at ``control_hz``; re-query every ``execution_horizon``
  steps. The caller must rebase the episode-start frames whenever ``reset()`` is
  called (new episode = new frame anchors).
- images: HWC uint8 RGB at the trained resolution, in manifest camera order
  (umi1 = LEFT gripper first).

Run this module directly to validate a served checkpoint by replaying dataset
observations through the wire and scoring returned chunks against ground truth:

    python -m lerobot.remote_inference.umi_ee_client \
        --server 127.0.0.1:8081 \
        --data_root /path/to/converted-14d-dataset \
        --repo_id user/dual-lidar-umi \
        --episodes 3 17 41 --stride 60 \
        --task "Put all oranges in the bowl"

fps, camera order/resolution, and feature names all come from the dataset metadata,
so the harness works for any dataset whose layout matches the served policy.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from .client import RemotePolicyClient, RemotePolicyClientConfig
from .schema import CameraSpec, EmbodimentManifest, ImageEncoding, ImageFrame, PolicyObservation

UMI_EE_FEATURE_NAMES: tuple[str, ...] = (
    "umi1_x",
    "umi1_y",
    "umi1_z",
    "umi1_rx",
    "umi1_ry",
    "umi1_rz",
    "umi1_gripper",
    "umi2_x",
    "umi2_y",
    "umi2_z",
    "umi2_rx",
    "umi2_ry",
    "umi2_rz",
    "umi2_gripper",
)

_CURRENTREL_R6D_POSE_NAMES = (
    "relative_x_m",
    "relative_y_m",
    "relative_z_m",
    "relative_r6d_col0_x",
    "relative_r6d_col0_y",
    "relative_r6d_col0_z",
    "relative_r6d_col1_x",
    "relative_r6d_col1_y",
    "relative_r6d_col1_z",
)
UMI_CURRENTREL_R6D_STATE_FEATURE_NAMES: tuple[str, ...] = (
    *(f"left_previous_{name}" for name in _CURRENTREL_R6D_POSE_NAMES),
    "left_current_gripper",
    *(f"right_previous_{name}" for name in _CURRENTREL_R6D_POSE_NAMES),
    "right_current_gripper",
)
UMI_CURRENTREL_R6D_ACTION_FEATURE_NAMES: tuple[str, ...] = (
    *(f"left_future_{name}" for name in _CURRENTREL_R6D_POSE_NAMES),
    "left_future_gripper",
    *(f"right_future_{name}" for name in _CURRENTREL_R6D_POSE_NAMES),
    "right_future_gripper",
)
UMI_CURRENTREL_R6D_JAW_DELTA_ACTION_FEATURE_NAMES: tuple[str, ...] = (
    *(f"left_future_{name}" for name in _CURRENTREL_R6D_POSE_NAMES),
    "left_future_gripper_delta_from_query",
    *(f"right_future_{name}" for name in _CURRENTREL_R6D_POSE_NAMES),
    "right_future_gripper_delta_from_query",
)


@dataclass
class UmiEeClientConfig:
    server_address: str
    task: str
    camera_keys: tuple[str, ...] = ("umi1", "umi2")
    image_width: int = 800
    image_height: int = 600
    control_hz: float = 30.0
    feature_names: tuple[str, ...] = UMI_EE_FEATURE_NAMES
    action_feature_names: tuple[str, ...] | None = None
    robot_id: str = "umi-rig"
    robot_type: str = "bimanual_umi_ee"
    # a 5B-class VLA needs far more than the transport default of 2 s
    inference_timeout_s: float = 20.0
    connect_timeout_s: float = 30.0

    def __post_init__(self) -> None:
        action_names = self.action_feature_names or self.feature_names
        if not self.feature_names or not action_names:
            raise ValueError("UMI state and action feature names must be non-empty")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("UMI state feature names must be unique")
        if len(set(action_names)) != len(action_names):
            raise ValueError("UMI action feature names must be unique")
        if len(self.feature_names) != len(action_names):
            raise ValueError("UMI state and action dimensions must match")


class UmiEeRemoteClient:
    """Session wrapper for bimanual-UMI EE-pose policies served by lerobot-policy-server."""

    def __init__(self, config: UmiEeClientConfig):
        self._config = config
        self._manifest = EmbodimentManifest(
            schema_id="lerobot-remote-v1",
            robot_id=config.robot_id,
            robot_type=config.robot_type,
            control_hz=config.control_hz,
            state_features=tuple(config.feature_names),
            action_features=tuple(config.action_feature_names or config.feature_names),
            cameras=tuple(
                CameraSpec(
                    key=key,
                    width=config.image_width,
                    height=config.image_height,
                    encoding=ImageEncoding.JPEG,
                )
                for key in config.camera_keys
            ),
        ).signed()
        self._client = RemotePolicyClient(
            RemotePolicyClientConfig(
                server_address=config.server_address,
                connect_timeout_s=config.connect_timeout_s,
                inference_timeout_s=config.inference_timeout_s,
            )
        )
        self._episode_id = uuid.uuid4().hex
        self._sequence = 0

    @property
    def model_manifest(self):
        return self._session.model

    @property
    def embodiment_manifest(self) -> EmbodimentManifest:
        return self._manifest

    def connect(self):
        self._session = self._client.connect(self._manifest, task=self._config.task)
        return self._session

    def predict(self, state: np.ndarray, images: dict[str, np.ndarray], tick: int) -> np.ndarray:
        """One inference round trip. Returns the checkpoint's declared action chunk."""
        frames = tuple(
            ImageFrame(key=key, array=images[key], capture_monotonic_ns=time.monotonic_ns())
            for key in self._config.camera_keys
        )
        observation = PolicyObservation(
            episode_id=self._episode_id,
            sequence=self._sequence,
            capture_tick=tick,
            capture_monotonic_ns=time.monotonic_ns(),
            state=np.ascontiguousarray(state, dtype=np.float32),
            images=frames,
            task=self._config.task,
            last_executed_tick=max(tick - 1, 0),
            action_queue_depth=0,
        )
        chunk = self._client.infer(observation)
        self._sequence += 1
        return chunk.actions

    def reset(self) -> None:
        """New episode: the caller must rebase episode-start pose frames alongside this."""
        self._client.reset()
        self._episode_id = uuid.uuid4().hex
        self._sequence = 0

    def close(self) -> None:
        self._client.close()


# --------------------------------------------------------------------------------------
# Replay validation harness (python -m lerobot.remote_inference.umi_ee_client)
# --------------------------------------------------------------------------------------


@dataclass
class _ReplayArgs:
    server: str
    data_root: str
    repo_id: str
    task: str
    episodes: list[int] = field(default_factory=lambda: [3, 17, 41])
    stride: int = 60
    max_obs_per_episode: int = 8
    gripper_dims: tuple[int, ...] | None = None
    out: str | None = None


def _to_hwc_uint8(image) -> np.ndarray:
    """LeRobotDataset yields CHW float32 in [0, 1]; the wire protocol wants HWC uint8 RGB."""
    array = image.numpy() if hasattr(image, "numpy") else np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    return np.ascontiguousarray(array)


def _replay_gripper_dims(
    action_names: tuple[str, ...],
    configured: tuple[int, ...] | None,
) -> tuple[int, ...]:
    """Resolve gripper dimensions from the artifact schema, not a legacy layout."""

    if configured is not None:
        dimensions = tuple(int(value) for value in configured)
    else:
        dimensions = tuple(
            index
            for index, name in enumerate(action_names)
            if name.casefold().endswith(("gripper", "gripper_delta_from_query"))
        )
    if len(dimensions) != 2 or len(set(dimensions)) != len(dimensions):
        raise ValueError(
            "replay requires exactly two distinct gripper action dimensions; "
            f"names={action_names}, resolved={dimensions}"
        )
    if any(index < 0 or index >= len(action_names) for index in dimensions):
        raise ValueError(f"gripper action dimensions are out of range: {dimensions}")
    return dimensions


def _replay_gripper_representation(
    action_names: tuple[str, ...],
    declared: str | None = None,
) -> str:
    """Bind replay jaw semantics to both the artifact declaration and feature names."""

    from lerobot.datasets.umi_current_relative import (
        GRIPPER_ACTION_ABSOLUTE_FUTURE,
        GRIPPER_ACTION_QUERY_ANCHOR_DELTA,
    )

    delta_names = tuple(name for name in action_names if name.endswith("_gripper_delta_from_query"))
    absolute_names = tuple(name for name in action_names if name.endswith("_gripper"))
    if delta_names and absolute_names:
        raise ValueError("replay action schema mixes absolute and query-anchor-delta gripper features")
    if delta_names and len(delta_names) != 2:
        raise ValueError(f"jaw-delta replay requires exactly two delta gripper features, got {delta_names}")
    if absolute_names and len(absolute_names) != 2:
        raise ValueError(f"absolute replay requires exactly two gripper features, got {absolute_names}")
    inferred = GRIPPER_ACTION_QUERY_ANCHOR_DELTA if delta_names else GRIPPER_ACTION_ABSOLUTE_FUTURE
    if declared in (None, ""):
        return inferred
    if declared not in {
        GRIPPER_ACTION_ABSOLUTE_FUTURE,
        GRIPPER_ACTION_QUERY_ANCHOR_DELTA,
    }:
        raise ValueError(f"unsupported replay gripper action representation {declared!r}")
    if declared != inferred:
        raise ValueError(
            "replay gripper action representation disagrees with action feature names: "
            f"declared={declared!r}, inferred={inferred!r}"
        )
    return declared


def _replay_valid_rows(action_is_pad, *, horizon: int) -> np.ndarray:
    """Return the rows with real future targets, rejecting an unscorable sample."""

    if action_is_pad is None:
        return np.ones(horizon, dtype=bool)
    pad = (
        action_is_pad.detach().cpu().numpy()
        if hasattr(action_is_pad, "detach")
        else np.asarray(action_is_pad)
    )
    if pad.dtype != np.bool_:
        raise ValueError(f"action_is_pad must be Boolean, got {pad.dtype}")
    if pad.shape != (horizon,):
        raise ValueError(f"action_is_pad must have shape ({horizon},), got {pad.shape}")
    valid = ~pad
    if not valid.any():
        raise RuntimeError("replay sample has zero valid action rows")
    return valid


def _replay_query_gripper_anchors(
    state: np.ndarray,
    state_names: tuple[str, ...],
    action_names: tuple[str, ...],
    gripper_dims: tuple[int, ...],
) -> np.ndarray:
    """Find the query-time normalized jaw widths paired with delta action fields."""

    state = np.asarray(state, dtype=np.float32)
    if state.shape != (len(state_names),):
        raise ValueError(f"state shape {state.shape} does not match {len(state_names)} feature names")
    lookup = {name: index for index, name in enumerate(state_names)}
    anchors = []
    for dimension in gripper_dims:
        action_name = action_names[dimension]
        if not action_name.endswith("_gripper_delta_from_query") or "_future_" not in action_name:
            raise ValueError(f"cannot resolve a query jaw anchor for action feature {action_name!r}")
        arm = action_name.split("_future_", 1)[0]
        state_name = f"{arm}_current_gripper"
        if state_name not in lookup:
            raise ValueError(
                f"jaw-delta action feature {action_name!r} requires state feature {state_name!r}"
            )
        anchor = float(state[lookup[state_name]])
        if not np.isfinite(anchor) or not 0.0 <= anchor <= 1.0:
            raise ValueError(f"query jaw anchor {state_name!r} must be finite and in [0, 1], got {anchor}")
        anchors.append(anchor)
    return np.asarray(anchors, dtype=np.float32)


def _replay_resolve_grippers(
    raw_grippers: np.ndarray,
    *,
    representation: str,
    anchors: np.ndarray | None,
) -> np.ndarray:
    """Convert raw policy jaw fields into normalized physical jaw widths."""

    from lerobot.datasets.umi_current_relative import GRIPPER_ACTION_QUERY_ANCHOR_DELTA

    values = np.asarray(raw_grippers, dtype=np.float32)
    if representation != GRIPPER_ACTION_QUERY_ANCHOR_DELTA:
        # Preserve the legacy absolute-width replay exactly: it reported and
        # scored the values as served, without clipping them first.
        return values
    if anchors is None or anchors.shape != (values.shape[-1],):
        raise ValueError("jaw-delta replay requires one query anchor per gripper action dimension")
    return np.clip(values + anchors[None, :], 0.0, 1.0)


def _replay_artifact_gripper_stats(
    dataset_stats: Mapping[str, object] | None,
    *,
    action_dim: int,
    gripper_dims: tuple[int, ...],
) -> dict[str, list[float]] | None:
    """Extract JSON-safe raw jaw bounds/quantiles from dataset artifact statistics."""

    if not dataset_stats:
        return None
    action_stats = dataset_stats.get("action")
    if not isinstance(action_stats, Mapping):
        return None
    result: dict[str, list[float]] = {}
    for statistic in ("min", "max", "q01", "q99"):
        if statistic not in action_stats:
            continue
        values = action_stats[statistic]
        if hasattr(values, "detach"):
            values = values.detach().cpu().numpy()
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if array.shape != (action_dim,):
            raise ValueError(
                f"artifact action {statistic} statistic must have shape ({action_dim},), got {array.shape}"
            )
        selected = array[list(gripper_dims)]
        if not np.isfinite(selected).all():
            raise ValueError(f"artifact action {statistic} gripper statistics contain non-finite values")
        result[statistic] = [float(value) for value in selected]
    if "min" in result and "max" in result and np.any(np.asarray(result["min"]) > result["max"]):
        raise ValueError("artifact action gripper minimum exceeds maximum")
    if "q01" in result and "q99" in result and np.any(np.asarray(result["q01"]) > result["q99"]):
        raise ValueError("artifact action gripper q01 exceeds q99")
    return result or None


def _replay_raw_gripper_bounds(representation: str) -> tuple[float, float]:
    from lerobot.datasets.umi_current_relative import (
        GRIPPER_ACTION_ABSOLUTE_FUTURE,
        GRIPPER_ACTION_QUERY_ANCHOR_DELTA,
    )

    if representation == GRIPPER_ACTION_QUERY_ANCHOR_DELTA:
        return -1.0, 1.0
    if representation == GRIPPER_ACTION_ABSOLUTE_FUTURE:
        return 0.0, 1.0
    raise ValueError(f"unsupported replay gripper action representation {representation!r}")


def _replay_range_is_physical(
    minimum: float,
    maximum: float,
    bounds: tuple[float, float],
    *,
    tolerance: float,
) -> bool:
    return minimum >= bounds[0] - tolerance and maximum <= bounds[1] + tolerance


def _evaluate_replay_sample(
    *,
    state: np.ndarray,
    state_names: tuple[str, ...],
    action_names: tuple[str, ...],
    chunk: np.ndarray,
    target: np.ndarray,
    hold: np.ndarray,
    action_is_pad,
    gripper_dims: tuple[int, ...],
    gripper_representation: str,
) -> dict[str, object]:
    """Score one replay query using only rows with ground-truth future actions."""

    action_dim = len(action_names)
    arrays = {
        "chunk": np.asarray(chunk, dtype=np.float32),
        "target": np.asarray(target, dtype=np.float32),
        "hold": np.asarray(hold, dtype=np.float32),
    }
    horizon = arrays["chunk"].shape[0] if arrays["chunk"].ndim == 2 else -1
    expected_shape = (horizon, action_dim)
    for name, array in arrays.items():
        if horizon < 1 or array.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {array.shape}")
    valid = _replay_valid_rows(action_is_pad, horizon=horizon)
    pose_dims = tuple(dimension for dimension in range(action_dim) if dimension not in gripper_dims)
    if not pose_dims:
        raise ValueError("replay requires at least one non-gripper action dimension")

    valid_chunk = arrays["chunk"][valid]
    valid_target = arrays["target"][valid]
    valid_hold = arrays["hold"][valid]
    raw_chunk_grippers = valid_chunk[:, list(gripper_dims)]
    raw_target_grippers = valid_target[:, list(gripper_dims)]
    anchors = None
    from lerobot.datasets.umi_current_relative import GRIPPER_ACTION_QUERY_ANCHOR_DELTA

    if gripper_representation == GRIPPER_ACTION_QUERY_ANCHOR_DELTA:
        anchors = _replay_query_gripper_anchors(state, state_names, action_names, gripper_dims)
    chunk_grippers = _replay_resolve_grippers(
        raw_chunk_grippers,
        representation=gripper_representation,
        anchors=anchors,
    )
    target_grippers = _replay_resolve_grippers(
        raw_target_grippers,
        representation=gripper_representation,
        anchors=anchors,
    )
    raw_bounds = _replay_raw_gripper_bounds(gripper_representation)
    raw_min = float(raw_chunk_grippers.min())
    raw_max = float(raw_chunk_grippers.max())
    target_raw_min = float(raw_target_grippers.min())
    target_raw_max = float(raw_target_grippers.max())
    # Keep the historical 5% transport tolerance for absolute widths, but a
    # normalized jaw delta has a physical hard range of exactly [-1, 1].
    raw_bounds_tolerance = 1e-6 if gripper_representation == GRIPPER_ACTION_QUERY_ANCHOR_DELTA else 0.05
    return {
        "valid_rows": int(valid.sum()),
        "padded_rows": int((~valid).sum()),
        "query_gripper_anchor": None if anchors is None else [float(value) for value in anchors],
        "l1_pose": float(np.abs(valid_chunk[:, list(pose_dims)] - valid_target[:, list(pose_dims)]).mean()),
        "l1_gripper": float(np.abs(chunk_grippers - target_grippers).mean()),
        "l1_gripper_raw": float(np.abs(raw_chunk_grippers - raw_target_grippers).mean()),
        "hold_l1_pose": float(
            np.abs(valid_hold[:, list(pose_dims)] - valid_target[:, list(pose_dims)]).mean()
        ),
        "gripper_min": float(chunk_grippers.min()),
        "gripper_max": float(chunk_grippers.max()),
        "target_gripper_min": float(target_grippers.min()),
        "target_gripper_max": float(target_grippers.max()),
        "raw_gripper_min": raw_min,
        "raw_gripper_max": raw_max,
        "target_raw_gripper_min": target_raw_min,
        "target_raw_gripper_max": target_raw_max,
        "raw_gripper_bounds_ok": _replay_range_is_physical(
            raw_min, raw_max, raw_bounds, tolerance=raw_bounds_tolerance
        ),
        "target_raw_gripper_bounds_ok": _replay_range_is_physical(
            target_raw_min, target_raw_max, raw_bounds, tolerance=1e-6
        ),
        "finite": bool(all(np.isfinite(array).all() for array in arrays.values())),
    }


def _current_relative_hold_baseline(
    state: np.ndarray,
    state_names: tuple[str, ...],
    action_names: tuple[str, ...],
    horizon: int,
) -> np.ndarray:
    """Build a no-motion target for the current-relative Rotation6D schema."""

    state = np.asarray(state, dtype=np.float32)
    if state.shape != (len(state_names),):
        raise ValueError(f"state shape {state.shape} does not match {len(state_names)} feature names")
    hold_row = np.zeros(len(action_names), dtype=np.float32)
    identity_components = {
        "relative_r6d_col0_x": 1.0,
        "relative_r6d_col1_y": 1.0,
    }
    state_lookup = {name: index for index, name in enumerate(state_names)}
    for index, name in enumerate(action_names):
        for suffix, value in identity_components.items():
            if name.endswith(suffix):
                hold_row[index] = value
                break
        else:
            if name.endswith("_gripper"):
                arm = name.split("_", 1)[0]
                current_name = f"{arm}_current_gripper"
                if current_name not in state_lookup:
                    raise ValueError(
                        f"cannot map current-relative gripper {name!r} to state features {state_names}"
                    )
                hold_row[index] = state[state_lookup[current_name]]
    return np.repeat(hold_row[None, :], horizon, axis=0)


def _make_replay_dataset(
    args: _ReplayArgs,
    *,
    episode: int,
    horizon: int,
    fps: float,
    current_relative: bool,
    generic_dataset_class=None,
    current_relative_dataset_class=None,
):
    """Construct the schema-specific replay reader.

    The injectable classes keep this dispatch testable without importing the
    optional dataset/video stack on a robot-only deployment import.
    """

    if generic_dataset_class is None or current_relative_dataset_class is None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.datasets.umi_current_relative import UmiCurrentRelativeR6dDataset

        generic_dataset_class = generic_dataset_class or LeRobotDataset
        current_relative_dataset_class = current_relative_dataset_class or UmiCurrentRelativeR6dDataset
    common = {
        "repo_id": args.repo_id,
        "root": args.data_root,
        "episodes": [episode],
        "image_transforms": None,
        "video_backend": "pyav",
    }
    if current_relative:
        return current_relative_dataset_class(**common, action_horizon=horizon)
    return generic_dataset_class(
        **common,
        delta_timestamps={"action": [index / fps for index in range(horizon)]},
    )


def _replay(args: _ReplayArgs) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.datasets.umi_current_relative import (
        GRIPPER_ACTION_ABSOLUTE_FUTURE,
        GRIPPER_ACTION_QUERY_ANCHOR_DELTA,
        is_umi_current_relative_dataset,
        load_current_relative_metadata,
    )

    # dataset metadata is the source of truth for fps, camera order, and resolution
    meta = LeRobotDatasetMetadata(repo_id=args.repo_id, root=args.data_root)
    image_keys = [key for key in meta.features if key.startswith("observation.images.")]
    if not image_keys:
        raise RuntimeError(f"{args.repo_id} declares no camera features")
    camera_keys = tuple(key.removeprefix("observation.images.") for key in image_keys)
    height, width = (int(v) for v in meta.features[image_keys[0]]["shape"][:2])
    state_names = tuple(meta.features["observation.state"]["names"])
    action_names = tuple(meta.features["action"]["names"])
    current_relative = is_umi_current_relative_dataset(meta.root)
    semantic_metadata = load_current_relative_metadata(meta.root) if current_relative else None
    declared_representation = (
        semantic_metadata.get("gripper_action_representation", GRIPPER_ACTION_ABSOLUTE_FUTURE)
        if semantic_metadata is not None
        else None
    )
    gripper_representation = _replay_gripper_representation(
        action_names,
        declared_representation,
    )

    client = UmiEeRemoteClient(
        UmiEeClientConfig(
            server_address=args.server,
            task=args.task,
            camera_keys=camera_keys,
            image_width=width,
            image_height=height,
            control_hz=float(meta.fps),
            feature_names=state_names,
            action_feature_names=action_names,
        )
    )
    try:
        session = client.connect()
        model = session.model
        horizon = model.action_horizon
        action_dim = model.action_dim
        print(
            f"session open: model={model.model_id} horizon={horizon} dim={action_dim} "
            f"cameras={model.camera_keys} fps={meta.fps} image={width}x{height}"
        )
        gripper_dims = _replay_gripper_dims(action_names, args.gripper_dims)
        if action_dim != len(action_names):
            raise ValueError(
                f"served action dimension {action_dim} does not match artifact schema {len(action_names)}"
            )
        served_representation = getattr(model, "gripper_action_representation", "")
        if gripper_representation == GRIPPER_ACTION_QUERY_ANCHOR_DELTA and not served_representation:
            raise RuntimeError(
                "jaw-delta replay requires the served model manifest to declare gripper_action_representation"
            )
        served_representation = _replay_gripper_representation(
            action_names,
            served_representation or GRIPPER_ACTION_ABSOLUTE_FUTURE,
        )
        if served_representation != gripper_representation:
            raise RuntimeError(
                "served model and dataset disagree on gripper action representation: "
                f"model={served_representation!r}, dataset={gripper_representation!r}"
            )
        artifact_gripper_stats = _replay_artifact_gripper_stats(
            getattr(meta, "stats", None),
            action_dim=action_dim,
            gripper_dims=gripper_dims,
        )
    except BaseException:
        # Once connect succeeds the server permits only this active session. Any
        # local manifest/argument validation error must release it immediately.
        client.close()
        raise
    per_obs = []
    try:
        for episode in args.episodes:
            # go through the dataset rather than guessing file paths: data files are
            # size-based (an episode is not file-<episode>), may hold several episodes,
            # and concatenated videos need per-episode timestamp offsets
            dataset = _make_replay_dataset(
                args,
                episode=episode,
                horizon=horizon,
                fps=float(meta.fps),
                current_relative=current_relative,
            )
            indices = list(range(0, len(dataset), args.stride))[: args.max_obs_per_episode]
            client.reset()
            for index in indices:
                item = dataset[index]
                state = item["observation.state"].numpy().astype(np.float32)
                target = item["action"].numpy().astype(np.float32)
                images = {
                    key: _to_hwc_uint8(item[image_key])
                    for key, image_key in zip(camera_keys, image_keys, strict=True)
                }
                started = time.perf_counter()
                chunk = client.predict(state, images, index)
                latency_s = time.perf_counter() - started
                hold = (
                    _current_relative_hold_baseline(
                        state,
                        state_names,
                        action_names,
                        horizon,
                    )
                    if current_relative
                    else np.repeat(state[None, :], horizon, axis=0)
                )
                pad = item.get("action_is_pad")
                metrics = _evaluate_replay_sample(
                    state=state,
                    state_names=state_names,
                    action_names=action_names,
                    chunk=chunk,
                    target=target,
                    hold=hold,
                    action_is_pad=pad,
                    gripper_dims=gripper_dims,
                    gripper_representation=gripper_representation,
                )
                record = {
                    "episode": episode,
                    "index": index,
                    "latency_s": round(latency_s, 3),
                    **metrics,
                }
                per_obs.append(record)
                print(record)
    finally:
        # the server allows one active session; leaking it blocks retries until the
        # idle timeout expires
        client.close()

    if not per_obs:
        raise RuntimeError("replay produced no observations")
    valid_rows = sum(int(record["valid_rows"]) for record in per_obs)

    def valid_row_weighted_mean(key: str) -> float:
        return float(sum(float(record[key]) * int(record["valid_rows"]) for record in per_obs) / valid_rows)

    l1_pose = valid_row_weighted_mean("l1_pose")
    l1_grip = valid_row_weighted_mean("l1_gripper")
    l1_grip_raw = valid_row_weighted_mean("l1_gripper_raw")
    hold_pose = valid_row_weighted_mean("hold_l1_pose")
    gr_min = min(r["gripper_min"] for r in per_obs)
    gr_max = max(r["gripper_max"] for r in per_obs)
    raw_gr_min = min(r["raw_gripper_min"] for r in per_obs)
    raw_gr_max = max(r["raw_gripper_max"] for r in per_obs)
    raw_bounds = _replay_raw_gripper_bounds(gripper_representation)
    artifact_gripper_range = None
    artifact_gripper_bounds_ok = None
    if (
        artifact_gripper_stats is not None
        and "min" in artifact_gripper_stats
        and "max" in artifact_gripper_stats
    ):
        artifact_gripper_range = [
            min(artifact_gripper_stats["min"]),
            max(artifact_gripper_stats["max"]),
        ]
        artifact_gripper_bounds_ok = _replay_range_is_physical(
            artifact_gripper_range[0],
            artifact_gripper_range[1],
            raw_bounds,
            tolerance=1e-6,
        )
    structural_ok = (
        all(record["finite"] for record in per_obs)
        and all(record["raw_gripper_bounds_ok"] for record in per_obs)
        and all(record["target_raw_gripper_bounds_ok"] for record in per_obs)
        and artifact_gripper_bounds_ok is not False
    )
    summary = {
        "n_obs": len(per_obs),
        "valid_rows": valid_rows,
        "gripper_action_representation": gripper_representation,
        "l1_pose": l1_pose,
        "l1_gripper": l1_grip,
        "l1_gripper_raw": l1_grip_raw,
        "hold_baseline_l1_pose": hold_pose,
        "beats_hold_baseline": l1_pose < hold_pose,
        "gripper_range": [gr_min, gr_max],
        "raw_gripper_range": [raw_gr_min, raw_gr_max],
        "expected_raw_gripper_bounds": list(raw_bounds),
        "artifact_gripper_stats": artifact_gripper_stats,
        "artifact_gripper_range": artifact_gripper_range,
        "artifact_gripper_bounds_ok": artifact_gripper_bounds_ok,
        "structural_ok": structural_ok,
        "per_obs": per_obs,
    }
    print(
        f"\nSUMMARY n={len(per_obs)} l1_pose={l1_pose:.4f} (hold {hold_pose:.4f}) "
        f"l1_gripper={l1_grip:.4f} raw_l1_gripper={l1_grip_raw:.4f} "
        f"gripper_range=[{gr_min:.3f},{gr_max:.3f}] "
        f"raw_gripper_range=[{raw_gr_min:.3f},{raw_gr_max:.3f}] "
        f"representation={gripper_representation}"
    )
    print(f"artifact_raw_gripper_stats={artifact_gripper_stats}")
    print("WIRE_TEST_OK" if structural_ok else "WIRE_TEST_FAILED")
    return summary


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--repo_id", required=True, help="repo id of the dataset at --data_root")
    parser.add_argument("--task", required=True)
    parser.add_argument("--episodes", type=int, nargs="+", default=[3, 17, 41])
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--max_obs_per_episode", type=int, default=8)
    parser.add_argument("--out", default=None)
    ns = parser.parse_args()
    summary = _replay(
        _ReplayArgs(
            server=ns.server,
            data_root=ns.data_root,
            repo_id=ns.repo_id,
            task=ns.task,
            episodes=ns.episodes,
            stride=ns.stride,
            max_obs_per_episode=ns.max_obs_per_episode,
            out=ns.out,
        )
    )
    if ns.out:
        with open(ns.out, "w") as f:
            json.dump(summary, f, indent=2)
    raise SystemExit(0 if summary["structural_ok"] else 1)


if __name__ == "__main__":
    main()
