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

import abc
import hashlib
import json
import logging
import random
import stat
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.constants import ACTION, OBS_STATE

from .schema import (
    EmbodimentManifest,
    ImageFrame,
    ModelManifest,
    PolicyActionChunk,
    PolicyObservation,
    ProtocolValidationError,
)

logger = logging.getLogger(__name__)

MODEL_ARTIFACT_MANIFEST_SCHEMA_ID = "lerobot-model-artifact-tree-manifest-v1"
CANONICAL_JSON_SPEC = "utf8-json-sorted-compact-ensure-ascii-false-reject-nonfinite-v1"
_QUERY_ANCHOR_GRIPPER_REPRESENTATION = "query_anchor_delta_normalized_width"


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_read_only_regular_file(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file, not a symlink: {path}")
    if stat.S_IMODE(path.stat().st_mode) & 0o222:
        raise ValueError(f"{label} must be read-only: {path}")
    return path.resolve(strict=True)


def _safe_artifact_relative_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty relative POSIX path")
    relative = Path(value)
    if value != relative.as_posix() or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} is not a safe relative POSIX path: {value!r}")
    return value


def verify_model_artifact_manifest(
    *,
    artifact_root: str | Path,
    manifest_path: str | Path,
    expected_manifest_sha256: str,
) -> str:
    """Verify an immutable, complete model/config/processor file tree.

    The manifest is deliberately external to ``artifact_root`` so its own
    checksum does not create a recursive inventory entry. The advertised tree
    digest uses the onset-v4 training contract's frozen canonical JSON function
    over the exact relative-path-to-SHA-256 map; byte sizes are independently
    bound and checked for every file.
    """

    expected_manifest_sha256 = _require_sha256(
        expected_manifest_sha256,
        label="model artifact manifest expected SHA-256",
    )
    root_input = Path(artifact_root).expanduser()
    if root_input.is_symlink() or not root_input.is_dir():
        raise ValueError("model artifact root must be a real directory, not a symlink")
    root = root_input.resolve(strict=True)
    if stat.S_IMODE(root.stat().st_mode) & 0o222:
        raise ValueError("model artifact root must be read-only")

    manifest = _require_read_only_regular_file(
        Path(manifest_path).expanduser(),
        label="model artifact manifest",
    )
    if manifest.is_relative_to(root):
        raise ValueError("model artifact manifest must be outside the inventoried artifact root")
    actual_manifest_sha256 = _sha256_file(manifest)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            "model artifact manifest SHA-256 mismatch: "
            f"expected {expected_manifest_sha256}, got {actual_manifest_sha256}"
        )
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"model artifact manifest is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("model artifact manifest must contain a JSON object")
    expected_keys = {
        "schema_id",
        "canonical_json_spec",
        "file_count",
        "files_sha256",
        "files_size_bytes",
        "tree_sha256",
    }
    if set(payload) != expected_keys:
        raise ValueError(
            f"model artifact manifest keys changed: expected {sorted(expected_keys)}, got {sorted(payload)}"
        )
    if payload["schema_id"] != MODEL_ARTIFACT_MANIFEST_SCHEMA_ID:
        raise ValueError("unsupported model artifact manifest schema")
    if payload["canonical_json_spec"] != CANONICAL_JSON_SPEC:
        raise ValueError("model artifact manifest canonical JSON contract changed")
    raw_sha256 = payload["files_sha256"]
    raw_sizes = payload["files_size_bytes"]
    if not isinstance(raw_sha256, Mapping) or not isinstance(raw_sizes, Mapping) or not raw_sha256:
        raise ValueError("model artifact manifest file maps must be non-empty objects")

    files_sha256: dict[str, str] = {}
    files_size_bytes: dict[str, int] = {}
    for raw_path, raw_digest in raw_sha256.items():
        relative = _safe_artifact_relative_path(raw_path, label="model artifact file path")
        files_sha256[relative] = _require_sha256(
            raw_digest,
            label=f"model artifact file SHA-256 for {relative}",
        )
    for raw_path, raw_size in raw_sizes.items():
        relative = _safe_artifact_relative_path(raw_path, label="model artifact size path")
        if isinstance(raw_size, bool) or not isinstance(raw_size, int) or raw_size < 0:
            raise ValueError(f"model artifact byte size for {relative} must be a non-negative integer")
        files_size_bytes[relative] = raw_size
    if set(files_sha256) != set(files_size_bytes):
        raise ValueError("model artifact SHA-256 and byte-size maps contain different paths")
    file_count = payload["file_count"]
    if isinstance(file_count, bool) or not isinstance(file_count, int) or file_count != len(files_sha256):
        raise ValueError("model artifact manifest file_count does not match its file maps")
    tree_sha256 = _require_sha256(payload["tree_sha256"], label="model artifact tree SHA-256")
    if _canonical_json_sha256(files_sha256) != tree_sha256:
        raise ValueError("model artifact tree SHA-256 does not match the canonical file map")

    actual_files: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"model artifact contains a symlink: {relative}")
        if stat.S_IMODE(path.stat().st_mode) & 0o222:
            raise ValueError(f"model artifact contains a writable path: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"model artifact contains a non-regular entry: {relative}")
        actual_files.add(relative)
    if actual_files != set(files_sha256):
        missing = sorted(set(files_sha256) - actual_files)
        extra = sorted(actual_files - set(files_sha256))
        raise ValueError(f"model artifact inventory mismatch: missing={missing}, extra={extra}")

    for relative in sorted(files_sha256):
        path = root / relative
        actual_size = path.stat().st_size
        if actual_size != files_size_bytes[relative]:
            raise ValueError(
                f"model artifact byte-size mismatch for {relative}: "
                f"expected {files_size_bytes[relative]}, got {actual_size}"
            )
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != files_sha256[relative]:
            raise ValueError(
                f"model artifact SHA-256 mismatch for {relative}: "
                f"expected {files_sha256[relative]}, got {actual_sha256}"
            )
    return tree_sha256


class PolicyBackend(abc.ABC):
    """Fixed policy loaded by a remote inference server."""

    @property
    @abc.abstractmethod
    def manifest(self) -> ModelManifest:
        pass

    def warmup(self) -> None:
        """Optional model warmup performed before accepting a robot session."""

        return None

    def prepare(self, embodiment: EmbodimentManifest, task: str) -> None:
        """Optional embodiment-specific warmup performed before a session is opened."""

        return None

    @abc.abstractmethod
    def infer(self, observation: PolicyObservation) -> PolicyActionChunk:
        pass

    @abc.abstractmethod
    def reset(self) -> None:
        pass

    def close(self) -> None:
        """Optional backend cleanup."""

        return None


@dataclass(frozen=True)
class DeterministicPolicyBackendConfig:
    model_id: str = "lerobot/deterministic-test-policy"
    action_horizon: int = 30
    state_features: tuple[str, ...] = ()
    action_features: tuple[str, ...] = ()
    camera_keys: tuple[str, ...] = ()


class DeterministicPolicyBackend(PolicyBackend):
    """Small backend used for transport, timing, and fault tests."""

    def __init__(self, config: DeterministicPolicyBackendConfig):
        if not config.action_features or not config.state_features or not config.camera_keys:
            raise ValueError("deterministic backend requires explicit feature and camera names")
        fingerprint = _model_fingerprint(
            model_id=config.model_id,
            revision="test",
            policy_type="deterministic",
            norm_tag="",
            horizon=config.action_horizon,
            state_features=config.state_features,
            action_features=config.action_features,
            camera_keys=config.camera_keys,
        )
        self._manifest = ModelManifest(
            model_id=config.model_id,
            revision="test",
            policy_type="deterministic",
            norm_tag="",
            action_horizon=config.action_horizon,
            action_dim=len(config.action_features),
            state_features=config.state_features,
            action_features=config.action_features,
            camera_keys=config.camera_keys,
            fingerprint=fingerprint,
        )
        self._manifest.validate()

    @property
    def manifest(self) -> ModelManifest:
        return self._manifest

    def infer(self, observation: PolicyObservation) -> PolicyActionChunk:
        action_dim = self._manifest.action_dim
        base = np.zeros(action_dim, dtype=np.float32)
        copy_dim = min(action_dim, observation.state.size)
        base[:copy_dim] = observation.state[:copy_dim]
        actions = np.repeat(base[None, :], self._manifest.action_horizon, axis=0)
        return PolicyActionChunk(
            observation_sequence=observation.sequence,
            first_action_tick=observation.capture_tick,
            actions=actions,
            model_fingerprint=self._manifest.fingerprint,
        )

    def reset(self) -> None:
        return None


@dataclass(frozen=True)
class LeRobotPolicyBackendConfig:
    pretrained_name_or_path: str
    policy_type: str | None = None
    revision: str | None = None
    device: str = "cuda"
    model_dtype: str | None = None
    norm_tag: str | None = None
    inference_action_mode: str | None = None
    gripper_action_representation: str | None = None
    artifact_root: str | None = None
    artifact_manifest_path: str | None = None
    artifact_manifest_sha256: str | None = None
    runtime_dependency_root: str | None = None
    runtime_dependency_manifest_path: str | None = None
    runtime_dependency_manifest_sha256: str | None = None
    ik_release_report_path: str | None = None
    ik_release_report_sha256: str | None = None
    # Optional inference-only overrides.  ``None`` preserves the value saved in
    # the checkpoint, so existing policy-server invocations are unchanged.
    per_episode_seed: bool | None = None
    eval_seed: int | None = None


class LeRobotPolicyBackend(PolicyBackend):
    """Runs a fixed LeRobot policy and all model-side processors."""

    def __init__(self, config: LeRobotPolicyBackendConfig):
        self._config = config
        self._device = torch.device(config.device)
        # Warmup snapshots and restores inference RNG state while calling the
        # regular reset/infer paths.  An RLock keeps that transaction atomic
        # without duplicating the locked inference implementation.
        self._lock = threading.RLock()
        self._prepared_input: tuple[tuple[tuple[str, int, int], ...], str] | None = None
        (
            load_config,
            self._artifact_tree_sha256,
            self._artifact_manifest_sha256,
            self._runtime_dependency_root,
            self._runtime_dependency_tree_sha256,
            self._runtime_dependency_manifest_sha256,
            self._ik_release_report_sha256,
        ) = self._verified_load_config(config)
        self._policy_config = self._load_policy_config(load_config)
        self._runtime_dependency_source = self._rebind_runtime_dependency(
            self._policy_config,
            self._runtime_dependency_root,
        )
        self._dataset_stats = None
        if config.policy_type == "molmoact2":
            self._dataset_stats = self._configure_original_molmoact2(self._policy_config)
        policy_class = get_policy_class(self._policy_config.type)
        if config.policy_type == "molmoact2":
            # Original MolmoAct2 checkpoints are loaded by the wrapper during construction;
            # they are not serialized as a complete LeRobot policy directory.
            self._policy = policy_class(self._policy_config)
        else:
            self._policy = policy_class.from_pretrained(
                load_config.pretrained_name_or_path,
                config=self._policy_config,
                revision=self._policy_config.pretrained_revision,
            )
        self._policy.to(self._device).eval()
        # Saved pipelines carry the device of the machine that trained them, which would
        # otherwise win over --policy.device: loading a GPU-trained checkpoint on a CPU-only
        # host raises "Requested device 'cuda' but CUDA is not available" while the policy
        # config itself downgrades to CPU gracefully. Actions are moved to CPU in infer(),
        # so pinning both pipelines to the serving device is safe.
        preprocessor_override = {"device_processor": {"device": config.device}}
        postprocessor_override = {"device_processor": {"device": config.device}}
        if self._runtime_dependency_root and self._policy_config.type == "molmoact2":
            # The saved pipeline contains the training host's absolute base-model
            # path.  Rebind only this dependency-bearing step to the already
            # verified local snapshot; every normalization state remains loaded
            # byte-for-byte from the manifested fine-tuned artifact.
            preprocessor_override["molmoact2_pack_inputs"] = {
                "checkpoint_path": self._runtime_dependency_root
            }
        self._preprocessor, self._postprocessor = make_pre_post_processors(
            self._policy_config,
            pretrained_path=(
                None if config.policy_type == "molmoact2" else load_config.pretrained_name_or_path
            ),
            pretrained_revision=self._policy_config.pretrained_revision,
            dataset_stats=self._dataset_stats,
            preprocessor_overrides=preprocessor_override,
            postprocessor_overrides=postprocessor_override,
        )
        if self._runtime_dependency_root and self._policy_config.type == "molmoact2":
            dependency_root = Path(self._runtime_dependency_root)
            processor_paths = [
                Path(str(step.checkpoint_path)).expanduser().resolve(strict=True)
                for step in self._preprocessor.steps
                if hasattr(step, "checkpoint_path")
            ]
            if not processor_paths or any(path != dependency_root for path in processor_paths):
                raise ValueError(
                    "MolmoAct2 saved processor dependency does not equal the verified runtime dependency root"
                )
        self._manifest = self._build_manifest()

    @staticmethod
    def _rebind_runtime_dependency(
        policy_config: PreTrainedConfig,
        runtime_dependency_root: str,
    ) -> dict[str, str] | None:
        """Portably rebind one manifested external base without changing its revision.

        Fine-tuned MolmoAct2 and Pi0.5 configs record absolute paths from the
        training cluster.  Requiring the same mount path on a hardware host is
        neither portable nor an identity check.  Identity is supplied by the
        complete read-only dependency manifest, while the saved config (itself
        inside the manifested model artifact) supplies the immutable source
        revision.  This method records that source and changes only the local
        path used to construct the policy.
        """

        if not runtime_dependency_root:
            return None
        fields = {
            "molmoact2": ("checkpoint_path", "checkpoint_revision"),
            "pi05": ("pretrained_path", "pretrained_revision"),
        }.get(policy_config.type)
        if fields is None:
            raise ValueError("runtime dependency attestation is supported only for MolmoAct2 and Pi0.5")
        path_attribute, revision_attribute = fields
        saved_path = str(getattr(policy_config, path_attribute, "") or "").strip()
        if not saved_path:
            raise ValueError(f"{policy_config.type} config.{path_attribute} is missing")
        saved_revision = str(getattr(policy_config, revision_attribute, "") or "").strip()
        if (
            len(saved_revision) != 40
            or saved_revision != saved_revision.lower()
            or any(character not in "0123456789abcdef" for character in saved_revision)
        ):
            raise ValueError(
                f"{policy_config.type} config.{revision_attribute} must be an immutable 40-hex revision"
            )
        verified_root = Path(runtime_dependency_root).resolve(strict=True)
        setattr(policy_config, path_attribute, str(verified_root))
        return {
            "saved_path": saved_path,
            "saved_revision": saved_revision,
            "verified_local_root": str(verified_root),
        }

    @staticmethod
    def _verified_load_config(
        config: LeRobotPolicyBackendConfig,
    ) -> tuple[LeRobotPolicyBackendConfig, str, str, str, str, str, str]:
        artifact_values = (
            config.artifact_root,
            config.artifact_manifest_path,
            config.artifact_manifest_sha256,
        )
        any_artifact_value = any(value is not None and str(value).strip() for value in artifact_values)
        all_manifest_values = all(
            value is not None and str(value).strip()
            for value in (config.artifact_manifest_path, config.artifact_manifest_sha256)
        )
        if any_artifact_value and not all_manifest_values:
            raise ValueError(
                "model artifact attestation requires artifact_manifest_path and artifact_manifest_sha256"
            )
        dependency_values = (
            config.runtime_dependency_root,
            config.runtime_dependency_manifest_path,
            config.runtime_dependency_manifest_sha256,
        )
        any_dependency_value = any(value is not None and str(value).strip() for value in dependency_values)
        all_dependency_values = all(value is not None and str(value).strip() for value in dependency_values)
        if any_dependency_value and not all_dependency_values:
            raise ValueError(
                "runtime dependency attestation requires root, manifest path, and manifest SHA-256"
            )
        report_values = (config.ik_release_report_path, config.ik_release_report_sha256)
        if any(value is not None and str(value).strip() for value in report_values) and not all(
            value is not None and str(value).strip() for value in report_values
        ):
            raise ValueError(
                "IK release attestation requires ik_release_report_path and ik_release_report_sha256"
            )
        gripper_representation = str(config.gripper_action_representation or "").strip()
        if gripper_representation == _QUERY_ANCHOR_GRIPPER_REPRESENTATION and (
            not all_manifest_values
            or not all(value is not None and str(value).strip() for value in report_values)
        ):
            raise ValueError(
                "query-anchor jaw-delta serving requires complete model artifact and IK release attestations"
            )

        artifact_tree_sha256 = ""
        artifact_manifest_sha256 = ""
        load_path = config.pretrained_name_or_path
        if all_manifest_values:
            configured_model_path = Path(config.pretrained_name_or_path).expanduser()
            root = Path(config.artifact_root or config.pretrained_name_or_path).expanduser()
            if configured_model_path.is_dir():
                if root.resolve(strict=True) != configured_model_path.resolve(strict=True):
                    raise ValueError("a local pretrained_name_or_path must equal the verified artifact_root")
            else:
                revision = str(config.revision or "")
                if (
                    len(revision) != 40
                    or revision != revision.lower()
                    or any(character not in "0123456789abcdef" for character in revision)
                ):
                    raise ValueError(
                        "a Hub model artifact requires an explicit immutable 40-hex commit revision"
                    )
                if config.artifact_root is None:
                    raise ValueError(
                        "a Hub model artifact must be materialized as a verified local artifact_root"
                    )
            artifact_tree_sha256 = verify_model_artifact_manifest(
                artifact_root=root,
                manifest_path=str(config.artifact_manifest_path),
                expected_manifest_sha256=str(config.artifact_manifest_sha256),
            )
            artifact_manifest_sha256 = _require_sha256(
                str(config.artifact_manifest_sha256),
                label="model artifact manifest expected SHA-256",
            )
            load_path = str(root.resolve(strict=True))

        runtime_dependency_root = ""
        runtime_dependency_tree_sha256 = ""
        runtime_dependency_manifest_sha256 = ""
        if all_dependency_values:
            dependency_root = Path(str(config.runtime_dependency_root)).expanduser().resolve(strict=True)
            if all_manifest_values and dependency_root == Path(load_path).resolve(strict=True):
                raise ValueError(
                    "runtime dependency root must be distinct from the selected model artifact root"
                )
            runtime_dependency_tree_sha256 = verify_model_artifact_manifest(
                artifact_root=dependency_root,
                manifest_path=str(config.runtime_dependency_manifest_path),
                expected_manifest_sha256=str(config.runtime_dependency_manifest_sha256),
            )
            runtime_dependency_root = str(dependency_root)
            runtime_dependency_manifest_sha256 = _require_sha256(
                str(config.runtime_dependency_manifest_sha256),
                label="runtime dependency manifest expected SHA-256",
            )

        ik_release_report_sha256 = ""
        if all(value is not None and str(value).strip() for value in report_values):
            ik_release_report_sha256 = _require_sha256(
                str(config.ik_release_report_sha256),
                label="IK release report expected SHA-256",
            )
            report = _require_read_only_regular_file(
                Path(str(config.ik_release_report_path)).expanduser(),
                label="IK release report",
            )
            actual_report_sha256 = _sha256_file(report)
            if actual_report_sha256 != ik_release_report_sha256:
                raise ValueError(
                    "IK release report SHA-256 mismatch: "
                    f"expected {ik_release_report_sha256}, got {actual_report_sha256}"
                )
        return (
            replace(config, pretrained_name_or_path=load_path),
            artifact_tree_sha256,
            artifact_manifest_sha256,
            runtime_dependency_root,
            runtime_dependency_tree_sha256,
            runtime_dependency_manifest_sha256,
            (ik_release_report_sha256),
        )

    @staticmethod
    def _load_policy_config(config: LeRobotPolicyBackendConfig) -> PreTrainedConfig:
        if config.policy_type is None:
            policy_config = PreTrainedConfig.from_pretrained(
                config.pretrained_name_or_path,
                revision=config.revision,
            )
        elif config.policy_type == "molmoact2":
            from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config

            if not str(config.norm_tag or "").strip():
                raise ValueError("original MolmoAct2 checkpoints require policy.norm_tag")
            policy_config = MolmoAct2Config(
                checkpoint_path=config.pretrained_name_or_path,
                checkpoint_revision=config.revision,
                norm_tag=config.norm_tag,
                inference_action_mode=config.inference_action_mode or "continuous",
                model_dtype=config.model_dtype or "bfloat16",
            )
            policy_config.pretrained_path = config.pretrained_name_or_path
            policy_config.pretrained_revision = config.revision
        else:
            raise ValueError(
                "policy_type is only needed for original checkpoints; currently only molmoact2 is supported"
            )
        if config.model_dtype is not None and hasattr(policy_config, "model_dtype"):
            policy_config.model_dtype = config.model_dtype
        if config.norm_tag is not None and hasattr(policy_config, "norm_tag"):
            policy_config.norm_tag = config.norm_tag
        if config.inference_action_mode is not None and hasattr(policy_config, "inference_action_mode"):
            policy_config.inference_action_mode = config.inference_action_mode
        if config.per_episode_seed is not None:
            if not hasattr(policy_config, "per_episode_seed"):
                raise ValueError(
                    "policy.per_episode_seed was provided, but this policy does not support "
                    "rollout-local inference generators"
                )
            policy_config.per_episode_seed = config.per_episode_seed
        if config.eval_seed is not None:
            if not hasattr(policy_config, "eval_seed"):
                raise ValueError("policy.eval_seed was provided, but this policy does not support eval seeds")
            policy_config.eval_seed = config.eval_seed
        policy_config.device = config.device
        return policy_config

    @staticmethod
    def _configure_original_molmoact2(policy_config: PreTrainedConfig) -> dict:
        """Populate the LeRobot feature schema from the released checkpoint metadata."""
        from lerobot.policies.molmoact2.processor_molmoact2 import _load_hf_norm_stats_for_tag

        dataset_stats, metadata = _load_hf_norm_stats_for_tag(
            policy_config.checkpoint_path,
            revision=policy_config.checkpoint_revision,
            force_download=bool(policy_config.checkpoint_force_download),
            norm_tag=policy_config.norm_tag,
        )
        state_stats = metadata.get("state_stats")
        action_stats = metadata.get("action_stats")
        camera_keys = metadata.get("camera_keys")
        if not isinstance(state_stats, dict) or not isinstance(action_stats, dict):
            raise ValueError("MolmoAct2 normalization metadata is missing state or action statistics")
        state_names = tuple(str(name) for name in state_stats.get("names", ()))
        action_names = tuple(str(name) for name in action_stats.get("names", ()))
        if not state_names or not action_names:
            raise ValueError("MolmoAct2 normalization metadata is missing state or action feature names")
        if not isinstance(camera_keys, list) or not camera_keys:
            raise ValueError("MolmoAct2 normalization metadata is missing camera keys")
        image_keys = tuple(str(key) for key in camera_keys)

        policy_config.dataset_feature_names = {
            OBS_STATE: list(state_names),
            ACTION: list(action_names),
        }
        policy_config.input_features = {
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(len(state_names),)),
            **{key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)) for key in image_keys},
        }
        policy_config.output_features = {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(len(action_names),))
        }
        policy_config.image_keys = list(image_keys)
        if metadata.get("normalize_gripper") is not None:
            policy_config.normalize_gripper = bool(metadata["normalize_gripper"])
        return dataset_stats

    @property
    def manifest(self) -> ModelManifest:
        return self._manifest

    def warmup(self) -> None:
        """Run one synthetic chunk before the server starts accepting sessions."""
        camera_shapes = tuple(
            (camera_key, *self._configured_image_size(camera_key))
            for camera_key in self._manifest.camera_keys
        )
        self._run_warmup(camera_shapes, task="warm up the policy")

    def prepare(self, embodiment: EmbodimentManifest, task: str) -> None:
        """Warm input-shape-specific paths before the robot is allowed to arm."""
        camera_shapes = tuple((camera.key, camera.height, camera.width) for camera in embodiment.cameras)
        self._run_warmup(camera_shapes, task=task)

    @contextmanager
    def _preserve_inference_rng_state(self) -> Iterator[None]:
        """Make synthetic warmup RNG-neutral for global and policy-local generators."""

        python_rng_state = random.getstate()
        numpy_rng_state = np.random.get_state()
        torch_cpu_rng_state = torch.random.get_rng_state()
        torch_cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        snapshot_policy_rng = getattr(self._policy, "snapshot_inference_rng_state", None)
        restore_policy_rng = getattr(self._policy, "restore_inference_rng_state", None)
        if callable(snapshot_policy_rng) != callable(restore_policy_rng):
            raise RuntimeError(
                "policies must implement both snapshot_inference_rng_state() and "
                "restore_inference_rng_state(), or neither"
            )
        policy_rng_state = snapshot_policy_rng() if callable(snapshot_policy_rng) else None
        try:
            yield
        finally:
            try:
                if callable(restore_policy_rng):
                    restore_policy_rng(policy_rng_state)
            finally:
                random.setstate(python_rng_state)
                np.random.set_state(numpy_rng_state)
                torch.random.set_rng_state(torch_cpu_rng_state)
                if torch_cuda_rng_states is not None:
                    torch.cuda.set_rng_state_all(torch_cuda_rng_states)

    def _run_warmup(self, camera_shapes: tuple[tuple[str, int, int], ...], *, task: str) -> None:
        # Preparation can be called with the real task text.  Keep the entire
        # check/warmup/update transaction under the inference lock so the
        # synthetic request cannot consume an episode seed or race a request.
        with self._lock:
            prepared_input = (camera_shapes, task)
            if prepared_input == self._prepared_input:
                return
            state = np.zeros(len(self._manifest.state_features), dtype=np.float32)
            if self._dataset_stats is not None:
                state_stats = self._dataset_stats.get(OBS_STATE, {})
                center = state_stats.get("q50", state_stats.get("mean"))
                if center is not None and np.asarray(center).shape == state.shape:
                    state = np.asarray(center, dtype=np.float32)
            images = tuple(
                ImageFrame(
                    key=camera_key,
                    array=np.zeros((height, width, 3), dtype=np.uint8),
                    capture_monotonic_ns=0,
                )
                for camera_key, height, width in camera_shapes
            )
            observation = PolicyObservation(
                episode_id="server-warmup",
                sequence=0,
                capture_tick=0,
                capture_monotonic_ns=0,
                state=state,
                images=images,
                task=task,
                last_executed_tick=0,
                action_queue_depth=0,
            )
            started = time.perf_counter()
            with self._preserve_inference_rng_state():
                self.reset()
                try:
                    self.infer(observation)
                finally:
                    self.reset()
            self._prepared_input = prepared_input
        logger.info("Policy warmup completed in %.2fs", time.perf_counter() - started)

    def _configured_image_size(self, camera_key: str) -> tuple[int, int]:
        feature_key = f"observation.images.{camera_key}"
        for step in self._preprocessor.steps:
            rename_map = getattr(step, "rename_map", None)
            if rename_map and feature_key in rename_map:
                feature_key = rename_map[feature_key]
                break
        feature = self._policy_config.input_features.get(feature_key)
        shape = tuple(feature.shape) if feature is not None else ()
        if len(shape) == 3 and shape[0] in (1, 3):
            height, width = int(shape[1]), int(shape[2])
        elif len(shape) == 3 and shape[2] in (1, 3):
            height, width = int(shape[0]), int(shape[1])
        else:
            height = width = 224
        return height, width

    def _feature_names(self, key: str, fallback_dim: int) -> tuple[str, ...]:
        metadata = getattr(self._policy_config, "dataset_feature_names", {}) or {}
        names = metadata.get(key)
        if names:
            return tuple(names)
        configured = getattr(self._policy_config, "action_feature_names", None)
        uses_padded_state = (
            key == OBS_STATE and int(self._policy_config.input_features[OBS_STATE].shape[-1]) != fallback_dim
        )
        if configured and (key == ACTION or (uses_padded_state and len(configured) == fallback_dim)):
            return tuple(configured)
        return tuple(f"{key}.{index}" for index in range(fallback_dim))

    def _processor_feature_dim(self, key: str, fallback_dim: int) -> int:
        """Return the raw feature width represented by the saved processor state."""
        for step in self._preprocessor.steps:
            step_state = step.state_dict()
            for stat_name in ("q01", "mean", "min", "std"):
                value = step_state.get(f"{key}.{stat_name}")
                if value is not None and value.ndim == 1:
                    return int(value.shape[0])
        return fallback_dim

    def _external_camera_keys(self) -> tuple[str, ...]:
        configured_images = tuple(
            key
            for key, feature in self._policy_config.input_features.items()
            if feature.type is FeatureType.VISUAL
        )
        for step in self._preprocessor.steps:
            rename_map = getattr(step, "rename_map", None)
            if not rename_map:
                continue
            source_keys = tuple(
                source for source, target in rename_map.items() if target in configured_images
            )
            if source_keys:
                return tuple(key.removeprefix("observation.images.") for key in source_keys)
        return tuple(key.removeprefix("observation.images.") for key in configured_images)

    def _build_manifest(self) -> ModelManifest:
        state_feature = self._policy_config.input_features.get(OBS_STATE)
        action_feature = self._policy_config.output_features.get(ACTION)
        if state_feature is None or action_feature is None:
            raise ValueError("policy configuration does not declare state and action features")
        state_dim = self._processor_feature_dim(OBS_STATE, int(state_feature.shape[-1]))
        action_dim = int(action_feature.shape[-1])
        state_features = self._feature_names(OBS_STATE, state_dim)
        action_features = self._feature_names(ACTION, action_dim)
        if len(state_features) != state_dim or len(action_features) != action_dim:
            raise ValueError("policy feature metadata does not match its declared dimensions")
        camera_keys = self._external_camera_keys()
        horizon = int(
            getattr(
                self._policy_config,
                "n_action_steps",
                getattr(self._policy_config, "chunk_size", 1),
            )
        )
        revision = self._config.revision or "main"
        norm_tag = str(getattr(self._policy_config, "norm_tag", "") or "")
        gripper_action_representation = str(self._config.gripper_action_representation or "").strip()
        artifact_tree_sha256 = str(getattr(self, "_artifact_tree_sha256", "") or "")
        artifact_manifest_sha256 = str(getattr(self, "_artifact_manifest_sha256", "") or "")
        runtime_dependency_tree_sha256 = str(getattr(self, "_runtime_dependency_tree_sha256", "") or "")
        runtime_dependency_manifest_sha256 = str(
            getattr(self, "_runtime_dependency_manifest_sha256", "") or ""
        )
        ik_release_report_sha256 = str(getattr(self, "_ik_release_report_sha256", "") or "")
        fingerprint = _model_fingerprint(
            model_id=self._config.pretrained_name_or_path,
            revision=revision,
            policy_type=self._policy_config.type,
            norm_tag=norm_tag,
            horizon=horizon,
            state_features=state_features,
            action_features=action_features,
            camera_keys=camera_keys,
            gripper_action_representation=gripper_action_representation,
            artifact_tree_sha256=artifact_tree_sha256,
            artifact_manifest_sha256=artifact_manifest_sha256,
            runtime_dependency_tree_sha256=runtime_dependency_tree_sha256,
            runtime_dependency_manifest_sha256=runtime_dependency_manifest_sha256,
            ik_release_report_sha256=ik_release_report_sha256,
        )
        manifest = ModelManifest(
            model_id=self._config.pretrained_name_or_path,
            revision=revision,
            policy_type=self._policy_config.type,
            norm_tag=norm_tag,
            action_horizon=horizon,
            action_dim=action_dim,
            state_features=state_features,
            action_features=action_features,
            camera_keys=camera_keys,
            fingerprint=fingerprint,
            gripper_action_representation=gripper_action_representation,
            artifact_tree_sha256=artifact_tree_sha256,
            artifact_manifest_sha256=artifact_manifest_sha256,
            runtime_dependency_tree_sha256=runtime_dependency_tree_sha256,
            runtime_dependency_manifest_sha256=runtime_dependency_manifest_sha256,
            ik_release_report_sha256=ik_release_report_sha256,
        )
        manifest.validate()
        return manifest

    def infer(self, observation: PolicyObservation) -> PolicyActionChunk:
        raw_observation: dict[str, np.ndarray] = {OBS_STATE: observation.state.copy()}
        for frame in observation.images:
            raw_observation[f"observation.images.{frame.key}"] = frame.array.copy()
        batch = prepare_observation_for_inference(
            raw_observation,
            self._device,
            observation.task,
            "remote_robot",
        )
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self._device.type == "cuda" and getattr(self._policy_config, "model_dtype", "") == "bfloat16"
            else nullcontext()
        )
        with self._lock, torch.inference_mode(), autocast:
            batch = self._preprocessor(batch)
            chunk = self._policy.predict_action_chunk(batch)
            if chunk.ndim == 2:
                chunk = chunk.unsqueeze(0)
            processed = [self._postprocessor(chunk[:, index, :]) for index in range(chunk.shape[1])]
            actions = torch.stack(processed, dim=1).squeeze(0).detach().cpu().float().numpy()
        actions = actions[: self._manifest.action_horizon, : self._manifest.action_dim]
        if actions.shape != (self._manifest.action_horizon, self._manifest.action_dim):
            raise ProtocolValidationError(f"policy returned unexpected action shape {actions.shape}")
        return PolicyActionChunk(
            observation_sequence=observation.sequence,
            first_action_tick=observation.capture_tick,
            actions=np.asarray(actions, dtype=np.float32),
            model_fingerprint=self._manifest.fingerprint,
        )

    def reset(self) -> None:
        with self._lock:
            self._policy.reset()
            self._preprocessor.reset()
            self._postprocessor.reset()


def _model_fingerprint(
    *,
    model_id: str,
    revision: str,
    policy_type: str,
    norm_tag: str,
    horizon: int,
    state_features: tuple[str, ...],
    action_features: tuple[str, ...],
    camera_keys: tuple[str, ...],
    gripper_action_representation: str = "",
    artifact_tree_sha256: str = "",
    artifact_manifest_sha256: str = "",
    runtime_dependency_tree_sha256: str = "",
    runtime_dependency_manifest_sha256: str = "",
    inference_backend_mode: str = "",
    inference_seed: int = 0,
    inference_code_manifest_sha256: str = "",
    inference_attestation_identity_sha256: str = "",
    ik_release_report_sha256: str = "",
) -> str:
    fingerprint_fields = {
        "model_id": model_id,
        "revision": revision,
        "policy_type": policy_type,
        "norm_tag": norm_tag,
        "horizon": horizon,
        "state_features": state_features,
        "action_features": action_features,
        "camera_keys": camera_keys,
    }
    if gripper_action_representation:
        fingerprint_fields["gripper_action_representation"] = gripper_action_representation
    if artifact_tree_sha256:
        fingerprint_fields["artifact_tree_sha256"] = artifact_tree_sha256
    if artifact_manifest_sha256:
        fingerprint_fields["artifact_manifest_sha256"] = artifact_manifest_sha256
    if runtime_dependency_tree_sha256:
        fingerprint_fields["runtime_dependency_tree_sha256"] = runtime_dependency_tree_sha256
    if runtime_dependency_manifest_sha256:
        fingerprint_fields["runtime_dependency_manifest_sha256"] = runtime_dependency_manifest_sha256
    if inference_backend_mode:
        fingerprint_fields.update(
            {
                "inference_backend_mode": inference_backend_mode,
                "inference_seed": inference_seed,
                "inference_code_manifest_sha256": inference_code_manifest_sha256,
                "inference_attestation_identity_sha256": inference_attestation_identity_sha256,
            }
        )
    if ik_release_report_sha256:
        fingerprint_fields["ik_release_report_sha256"] = ik_release_report_sha256
    payload = json.dumps(
        fingerprint_fields,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()
