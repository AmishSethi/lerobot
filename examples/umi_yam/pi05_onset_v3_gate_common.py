#!/usr/bin/env python

"""Immutable Pi0.5 onset-v3 gate constants and fail-closed helpers.

The Pi0.5 adapter deliberately has its own schemas and RNG contract.  It reuses
the already-audited UMI-to-YAM resolver and IK mathematics, but a Pi report can
never be mistaken for a MolmoAct2 report merely because the tensor shapes match.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path
from typing import Any

POLICY_TYPE = "pi05"
DATASET_SCHEMA_ID = "dual-lidar-umi-currentrel-r6d-onset-v3"
DATASET_REPO_ID = "brandonyang/dual-lidar-umi-currentrel-r6d-onset-v3"
TRAINING_REPORT_SCHEMA_ID = "umi-yam-pi05-onset-v3-training-report-v1"
GATE_CONTRACT_SCHEMA_ID = "umi-yam-pi05-onset-v3-checkpoint-gate-plan-v1"
EXECUTION_PLAN_SCHEMA_ID = "umi-yam-pi05-onset-v3-gate-execution-plan-v1"
STAGE1_CELL_SCHEMA_ID = "umi-yam-pi05-onset-v3-interior-model-ik-cell-v1"
STAGE1_AGGREGATE_SCHEMA_ID = "umi-yam-pi05-onset-v3-stage1-aggregate-v1"
TEMPORAL_CELL_SCHEMA_ID = "umi-yam-pi05-onset-v3-temporal-teacher-forced-cell-v1"
TEMPORAL_AGGREGATE_SCHEMA_ID = "umi-yam-pi05-onset-v3-temporal-aggregate-v1"
DUAL_SELECTION_SCHEMA_ID = "umi-yam-pi05-onset-v3-dual-gate-selection-v1"
SERVER_ATTESTATION_SCHEMA_ID = "umi-yam-pi05-explicit-noise-server-attestation-v1"
RNG_ALGORITHM = "torch-device-generator-explicit-float32-normal-v1"
ALLOWED_TRAINING_DISTRIBUTED = (
    {
        "profile": "multinode_16gpu",
        "world_size": 16,
        "nodes": 2,
        "gpus_per_node": 8,
        "per_rank_batch_size": 4,
        "global_batch_size": 64,
    },
    {
        "profile": "single_node_8gpu",
        "world_size": 8,
        "nodes": 1,
        "gpus_per_node": 8,
        "per_rank_batch_size": 8,
        "global_batch_size": 64,
    },
    {
        "profile": "two_node_4x2",
        "world_size": 8,
        "nodes": 2,
        "gpus_per_node": 4,
        "per_rank_batch_size": 8,
        "global_batch_size": 64,
    },
)

CHECKPOINT_STEPS = tuple(range(1_000, 12_001, 1_000))
HOLDOUT_EPISODES = (52, 53)
SEEDS = (0, 1, 2, 3, 4)
ACTION_HORIZON = 24
EXECUTION_HORIZON = 15
QUERY_STRIDE = 15
NUM_FLOW_STEPS = 10
MAX_STATE_DIM = 32
MAX_ACTION_DIM = 32
ADAPTER_MAX_JOINT_DELTA_RAD = 0.02
DRIVER_MAX_JOINT_DELTA_RAD = 0.1
MAX_PROGRESS_HOLD_STEPS = 90
MINIMUM_RELATIVE_IMPROVEMENT_OVER_IDENTITY = 0.10
START_ARM_JOINTS_RAD = (0.0, 0.05, 0.05, 0.0, 0.0, 0.0)
START_GRIPPER_NORMALIZED = 1.0
START_DRIVER_TARGET = (
    *START_ARM_JOINTS_RAD,
    START_GRIPPER_NORMALIZED,
    *START_ARM_JOINTS_RAD,
    START_GRIPPER_NORMALIZED,
)
TASK = "Put all oranges in the bowl"
DATASET_MANIFEST_RELATIVE = "meta/artifact_manifest.sha256"
CODE_MANIFEST_RELATIVE = ".umi_yam_snapshot.sha256"
HF_MANIFEST_RELATIVE = ".pi05_gate_hf_manifest.sha256"
TOKENIZER_REPO_ID = "google/paligemma-3b-pt-224"
TOKENIZER_REVISION = "35e4f46485b4d07967e7e9935bc3786aad50687c"
TOKENIZER_CACHE_PREFIX = "hub/models--google--paligemma-3b-pt-224"
TOKENIZER_REQUIRED_FILES = (
    f"{TOKENIZER_CACHE_PREFIX}/refs/main",
    f"{TOKENIZER_CACHE_PREFIX}/snapshots/{TOKENIZER_REVISION}/added_tokens.json",
    f"{TOKENIZER_CACHE_PREFIX}/snapshots/{TOKENIZER_REVISION}/config.json",
    f"{TOKENIZER_CACHE_PREFIX}/snapshots/{TOKENIZER_REVISION}/preprocessor_config.json",
    f"{TOKENIZER_CACHE_PREFIX}/snapshots/{TOKENIZER_REVISION}/special_tokens_map.json",
    f"{TOKENIZER_CACHE_PREFIX}/snapshots/{TOKENIZER_REVISION}/tokenizer.json",
    f"{TOKENIZER_CACHE_PREFIX}/snapshots/{TOKENIZER_REVISION}/tokenizer.model",
    f"{TOKENIZER_CACHE_PREFIX}/snapshots/{TOKENIZER_REVISION}/tokenizer_config.json",
)

# These bytes influence a Pi cell or its independent recomputation.  The plan
# binds their hashes in addition to the complete post-training snapshot manifest.
REQUIRED_CODE_ARTIFACTS = (
    "examples/umi_yam/pi05_onset_v3_gate_common.py",
    "examples/umi_yam/pi05_onset_v3_policy_server.py",
    "examples/umi_yam/make_pi05_onset_v3_gate_plan.py",
    "examples/umi_yam/offline_pi05_currentrel_onset_v3_checkpoint_gate.py",
    "examples/umi_yam/offline_pi05_currentrel_onset_v3_temporal_teacher_forced.py",
    "examples/umi_yam/aggregate_pi05_currentrel_onset_v3_gates.py",
    "examples/umi_yam/pi05_onset_v3_stage1_array.sbatch",
    "examples/umi_yam/pi05_onset_v3_temporal_array.sbatch",
    "examples/umi_yam/offline_currentrel_model_ik_gate.py",
    "examples/umi_yam/offline_currentrel_onset_v3_checkpoint_gate.py",
    "examples/umi_yam/offline_currentrel_onset_v3_temporal_teacher_forced.py",
    "examples/umi_yam/aggregate_currentrel_onset_v3_checkpoint_gate.py",
    "examples/umi_yam/aggregate_currentrel_onset_v3_temporal_gate.py",
    "examples/umi_yam/onset_v3_temporal_gate_common.py",
    "src/lerobot/policies/pi05/configuration_pi05.py",
    "src/lerobot/policies/pi05/modeling_pi05.py",
    "src/lerobot/policies/pi05/processor_pi05.py",
    "src/lerobot/remote_inference/backend.py",
    "src/lerobot/remote_inference/client.py",
    "src/lerobot/remote_inference/codec.py",
    "src/lerobot/remote_inference/server.py",
    "src/lerobot/remote_inference/umi_ee_client.py",
    "src/lerobot/robots/bi_yam/config_bi_yam.py",
    "src/lerobot/robots/bi_yam/umi_retargeting.py",
    "src/lerobot/scripts/umi_yam_ik_replay.py",
)


class Pi05GateError(RuntimeError):
    """A structural, provenance, or immutable-contract violation."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Pi05GateError(message)


def valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def require_sha256(value: Any, *, name: str) -> str:
    require(valid_sha256(value), f"{name} must be a lowercase SHA-256 digest")
    return str(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def float32_sha256(value: Any) -> str:
    import numpy as np

    array = np.ascontiguousarray(np.asarray(value, dtype="<f4"))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Pi05GateError(f"cannot parse JSON {path}: {error}") from error
    require(isinstance(value, dict), f"expected a JSON object in {path}")
    return value


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def require_read_only(path: Path, *, name: str) -> None:
    require(path.exists(), f"{name} is missing: {path}")
    writable = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    require(not path.stat().st_mode & writable, f"{name} must be write-protected: {path}")


def checksum_manifest_entries(root: Path, relative: str, expected_sha256: str) -> dict[str, str]:
    root = root.resolve(strict=True)
    manifest = (root / relative).resolve(strict=True)
    expected = require_sha256(expected_sha256, name=f"{relative} SHA-256")
    require(manifest.is_file() and manifest.is_relative_to(root), f"invalid manifest: {manifest}")
    require(sha256_file(manifest) == expected, f"{relative} SHA-256 mismatch")
    entries: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, separator, relative_text = line.partition("  ")
        require(bool(separator) and valid_sha256(digest), f"malformed checksum line: {line!r}")
        relative_path = Path(relative_text)
        require(
            relative_text != "" and not relative_path.is_absolute() and ".." not in relative_path.parts,
            f"unsafe checksum path: {relative_text!r}",
        )
        normalized = relative_path.as_posix().removeprefix("./")
        require(normalized not in entries, f"duplicate checksum path: {normalized}")
        unresolved = root / normalized
        require(not unresolved.is_symlink(), f"checksum artifact is a symlink: {normalized}")
        artifact = unresolved.resolve(strict=True)
        require(artifact.is_file() and artifact.is_relative_to(root), f"invalid artifact: {normalized}")
        require(sha256_file(artifact) == digest, f"checksum artifact changed: {normalized}")
        entries[normalized] = digest
    require(bool(entries), f"checksum manifest is empty: {manifest}")
    return entries


def validate_inference_hf_home(root_arg: Path, manifest_sha256: str) -> dict[str, Any]:
    """Validate the minimal regular-file cache needed by Pi0.5 tokenization."""

    root = root_arg.resolve(strict=True)
    require_read_only(root, name="Pi0.5 gate HF_HOME")
    entries = checksum_manifest_entries(root, HF_MANIFEST_RELATIVE, manifest_sha256)
    require(
        set(entries) == set(TOKENIZER_REQUIRED_FILES),
        "Pi0.5 gate HF_HOME must contain exactly the pinned tokenizer cache files",
    )
    for relative in TOKENIZER_REQUIRED_FILES:
        require_read_only(root / relative, name="Pi0.5 tokenizer cache artifact")
    reference = (root / f"{TOKENIZER_CACHE_PREFIX}/refs/main").read_text(encoding="utf-8").strip()
    require(reference == TOKENIZER_REVISION, "Pi0.5 tokenizer refs/main revision changed")
    return {
        "root": str(root),
        "manifest_sha256": require_sha256(
            manifest_sha256,
            name="Pi0.5 gate HF_HOME manifest SHA-256",
        ),
        "verified_file_count": len(entries),
        "tokenizer_repo_id": TOKENIZER_REPO_ID,
        "tokenizer_revision": TOKENIZER_REVISION,
        "required_file_sha256": {relative: entries[relative] for relative in TOKENIZER_REQUIRED_FILES},
    }


def recursive_file_inventory(root: Path) -> tuple[list[dict[str, Any]], str]:
    root = root.resolve(strict=True)
    require_read_only(root, name="checkpoint root")
    inventory: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), f"checkpoint contains a symlink: {path}")
        require_read_only(path, name="checkpoint artifact")
        if path.is_dir():
            continue
        require(path.is_file() and path.is_relative_to(root), f"invalid checkpoint artifact: {path}")
        inventory.append(
            {
                "relative": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    require(bool(inventory), f"checkpoint inventory is empty: {root}")
    digest = hashlib.sha256(json.dumps(inventory, separators=(",", ":"), sort_keys=True).encode()).hexdigest()
    return inventory, digest


def validate_gate_contract(contract: dict[str, Any]) -> None:
    exact = {
        "schema_id": GATE_CONTRACT_SCHEMA_ID,
        "policy_type": POLICY_TYPE,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "holdout_episodes": list(HOLDOUT_EPISODES),
        "fixed_seeds": list(SEEDS),
        "action_representation": "query-anchored-current-relative-SE3-R6D-plus-absolute-gripper",
        "action_horizon_rows": ACTION_HORIZON,
        "execution_horizon_rows": EXECUTION_HORIZON,
        "num_flow_inference_steps": NUM_FLOW_STEPS,
        "policy_use_relative_actions": False,
        "yam_start_driver_target": list(START_DRIVER_TARGET),
        "hardware_start_verified": False,
    }
    for key, expected in exact.items():
        require(contract.get(key) == expected, f"Pi0.5 gate contract field changed: {key}")
    stage1 = contract.get("stage1", {})
    required_stage1 = {
        "fresh_process_per_checkpoint_seed_episode": True,
        "exact_effective_seed_attestation_required": True,
        "all_first_15_rows_strict_ik_required": True,
        "strict_position_tolerance_m": 0.002,
        "strict_orientation_tolerance_deg": 1.0,
        "operational_joint_limits_required": True,
        "adapter_rate_limit_rad_per_dispatch": ADAPTER_MAX_JOINT_DELTA_RAD,
        "driver_rate_limit_rad_per_dispatch": DRIVER_MAX_JOINT_DELTA_RAD,
        "maximum_ideal_feedback_dispatches_per_waypoint": MAX_PROGRESS_HOLD_STEPS,
        "pooled_physical_se3_improvement_over_identity_required": (
            MINIMUM_RELATIVE_IMPROVEMENT_OVER_IDENTITY
        ),
        "all_24_rows_are_diagnostic_only": True,
    }
    require(
        stage1 == required_stage1,
        "Pi0.5 stage-one gate contract changed",
    )
    stage2 = contract.get("stage2", {})
    required_stage2 = {
        "required_for_every_stage1_eligible_checkpoint": True,
        "teacher_forced_source_images_and_policy_state": True,
        "predicted_yam_joints_carried_across_queries": True,
        "query_stride_source_frames": QUERY_STRIDE,
        "all_critical_first_15_rows_strict_ik_required": True,
        "pooled_physical_se3_improvement_over_identity_required": (
            MINIMUM_RELATIVE_IMPROVEMENT_OVER_IDENTITY
        ),
        "tail_and_gripper_metrics_are_diagnostic_only": True,
    }
    require(
        stage2 == required_stage2,
        "Pi0.5 temporal gate contract changed",
    )
    selection = contract.get("selection", {})
    require(
        selection
        == {
            "validation_loss_ranks_only_checkpoints_passing_both_stages": True,
            "no_seed_or_episode_cherry_picking": True,
            "hardware_deployment_ready": False,
        },
        "Pi0.5 final selection contract changed",
    )


def validate_policy_config(config: dict[str, Any]) -> None:
    exact = {
        "type": POLICY_TYPE,
        "dtype": "float32",
        "chunk_size": ACTION_HORIZON,
        "n_action_steps": ACTION_HORIZON,
        "num_inference_steps": NUM_FLOW_STEPS,
        "max_state_dim": MAX_STATE_DIM,
        "max_action_dim": MAX_ACTION_DIM,
        "use_relative_actions": False,
    }
    for key, expected in exact.items():
        require(config.get(key) == expected, f"checkpoint policy.{key} changed")
    feature_names = config.get("dataset_feature_names", {})
    require(
        isinstance(feature_names.get("observation.state"), list)
        and len(feature_names["observation.state"]) == 20
        and isinstance(feature_names.get("action"), list)
        and len(feature_names["action"]) == 20,
        "checkpoint does not preserve distinct 20-D state/action names",
    )
    require(
        config.get("output_features", {}).get("action", {}).get("shape") == [20],
        "checkpoint output action is not 20-D",
    )
    visual = {
        key for key, feature in config.get("input_features", {}).items() if feature.get("type") == "VISUAL"
    }
    require(
        visual
        == {
            "observation.images.base_0_rgb",
            "observation.images.left_wrist_0_rgb",
            "observation.images.right_wrist_0_rgb",
        },
        "checkpoint Pi0.5 camera slots changed",
    )


def expected_full_cell_grid() -> list[tuple[int, int, int]]:
    return [
        (step, seed, episode) for step in CHECKPOINT_STEPS for seed in SEEDS for episode in HOLDOUT_EPISODES
    ]


def validate_execution_plan(
    plan_path: Path,
    *,
    expected_sha256: str,
    running_relative: str | None = None,
    running_file: Path | None = None,
    checkpoint_step: int | None = None,
) -> dict[str, Any]:
    """Validate immutable plan, snapshots, dataset, and checkpoint identities."""

    plan_path = plan_path.resolve(strict=True)
    expected = require_sha256(expected_sha256, name="execution-plan SHA-256")
    require(sha256_file(plan_path) == expected, "execution-plan SHA-256 mismatch")
    require_read_only(plan_path, name="Pi0.5 execution plan")
    plan = load_json(plan_path)
    exact = {
        "schema_id": EXECUTION_PLAN_SCHEMA_ID,
        "policy_type": POLICY_TYPE,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "holdout_episodes": list(HOLDOUT_EPISODES),
        "seeds": list(SEEDS),
        "action_horizon_rows": ACTION_HORIZON,
        "execution_horizon_rows": EXECUTION_HORIZON,
        "query_stride_source_frames": QUERY_STRIDE,
        "num_flow_steps": NUM_FLOW_STEPS,
        "start_driver_target": list(START_DRIVER_TARGET),
        "adapter_max_joint_delta_rad_per_call": ADAPTER_MAX_JOINT_DELTA_RAD,
        "driver_max_joint_delta_rad_per_call": DRIVER_MAX_JOINT_DELTA_RAD,
        "max_progress_hold_steps": MAX_PROGRESS_HOLD_STEPS,
        "minimum_relative_improvement_over_identity": (MINIMUM_RELATIVE_IMPROVEMENT_OVER_IDENTITY),
        "hardware_start_verified": False,
        "hardware_deployment_ready": False,
    }
    for key, value in exact.items():
        require(plan.get(key) == value, f"execution-plan field changed: {key}")
    rng = plan.get("rng_contract", {})
    require(
        rng
        == {
            "algorithm": RNG_ALGORITHM,
            "fresh_process_per_stage1_cell": True,
            "fresh_process_per_temporal_cell": True,
            "synthetic_warmups_rng_neutral": True,
            "stage1_queries_per_sampling_epoch": 1,
            "temporal_continuous_stream_without_query_resets": True,
            "explicit_noise_shape": [1, ACTION_HORIZON, MAX_ACTION_DIM],
            "explicit_noise_dtype": "torch.float32",
        },
        "execution-plan Pi0.5 RNG contract changed",
    )
    bindings = (
        ("training_report", TRAINING_REPORT_SCHEMA_ID),
        ("gate_contract", GATE_CONTRACT_SCHEMA_ID),
    )
    bound_values: dict[str, dict[str, Any]] = {}
    for name, schema in bindings:
        binding = plan.get(name, {})
        path = Path(str(binding.get("path", ""))).resolve(strict=True)
        digest = require_sha256(binding.get("sha256"), name=f"{name} digest")
        require_read_only(path, name=name)
        require(sha256_file(path) == digest, f"{name} digest mismatch")
        value = load_json(path)
        bound_values[name] = value
        require(value.get("schema_id") == schema, f"{name} schema changed")
        if name == "gate_contract":
            validate_gate_contract(value)
    training_report = bound_values["training_report"]
    require(
        training_report.get("mode") == "full"
        and training_report.get("validation_passed") is True
        and training_report.get("hardware_control_performed") is False
        and training_report.get("hardware_deployment_ready") is False,
        "execution plan is not backed by a passing offline full-training report",
    )
    require(
        training_report.get("policy", {}).get("type") == POLICY_TYPE
        and training_report.get("policy", {}).get("action_horizon") == ACTION_HORIZON
        and training_report.get("policy", {}).get("model_advertised_action_horizon") == ACTION_HORIZON
        and training_report.get("policy", {}).get("deployment_execution_horizon") == EXECUTION_HORIZON
        and training_report.get("policy", {}).get("use_relative_actions") is False,
        "training-report Pi0.5 action contract changed",
    )
    distributed = training_report.get("distributed")
    require(
        distributed in ALLOWED_TRAINING_DISTRIBUTED,
        "training-report distributed topology changed",
    )
    launch_binding = training_report.get("launch_identity", {})
    launch_path = Path(str(launch_binding.get("path", ""))).resolve(strict=True)
    require_read_only(launch_path, name="training launch identity")
    require(
        sha256_file(launch_path) == launch_binding.get("sha256"),
        "training launch-identity digest changed",
    )
    launch_identity = load_json(launch_path)
    require(launch_identity == launch_binding.get("content"), "training launch identity changed")
    run_root = Path(str(plan.get("run_root", ""))).resolve(strict=True)
    require(
        Path(str(launch_identity.get("run_root", ""))).resolve(strict=True) == run_root,
        "execution-plan run root differs from training launch identity",
    )
    require(
        launch_identity.get("world_size") == distributed["world_size"]
        and launch_identity.get("per_rank_batch_size") == distributed["per_rank_batch_size"]
        and launch_identity.get("global_batch_size") == distributed["global_batch_size"],
        "launch-identity/training topology mismatch",
    )
    training_code = plan.get("training_code_snapshot", {})
    gate_code = plan.get("gate_code_snapshot", {})
    for name, snapshot in (("training code", training_code), ("gate code", gate_code)):
        root = Path(str(snapshot.get("root", ""))).resolve(strict=True)
        require_read_only(root, name=f"{name} snapshot")
        entries = checksum_manifest_entries(
            root,
            CODE_MANIFEST_RELATIVE,
            snapshot.get("manifest_sha256"),
        )
        require(snapshot.get("verified_file_count") == len(entries), f"{name} file count changed")
        for relative in entries:
            require_read_only(root / relative, name=f"{name} artifact")
    gate_root = Path(gate_code["root"]).resolve(strict=True)
    declared = gate_code.get("required_artifact_sha256")
    require(
        isinstance(declared, dict) and set(declared) == set(REQUIRED_CODE_ARTIFACTS),
        "gate code artifact set changed",
    )
    for relative in REQUIRED_CODE_ARTIFACTS:
        path = (gate_root / relative).resolve(strict=True)
        require(path.is_file() and path.is_relative_to(gate_root), f"missing gate artifact: {relative}")
        require(sha256_file(path) == declared[relative], f"gate code artifact changed: {relative}")
    hf_cache = plan.get("inference_hf_cache", {})
    verified_hf_cache = validate_inference_hf_home(
        Path(str(hf_cache.get("root", ""))),
        hf_cache.get("manifest_sha256"),
    )
    require(hf_cache == verified_hf_cache, "execution-plan Pi0.5 tokenizer cache record changed")
    if running_relative is not None:
        require(
            (running_file or Path(__file__)).resolve(strict=True)
            == (gate_root / running_relative).resolve(strict=True),
            f"command is outside plan-bound gate snapshot: {running_relative}",
        )
    dataset = plan.get("dataset", {})
    dataset_root = Path(str(dataset.get("root", ""))).resolve(strict=True)
    require_read_only(dataset_root, name="onset-v3 dataset")
    require(dataset_root.name == DATASET_SCHEMA_ID, "dataset basename changed")
    require(
        dataset.get("schema_id") == DATASET_SCHEMA_ID and dataset.get("repo_id") == DATASET_REPO_ID,
        "dataset identity changed",
    )
    dataset_entries = checksum_manifest_entries(
        dataset_root,
        DATASET_MANIFEST_RELATIVE,
        dataset.get("manifest_sha256"),
    )
    require(dataset.get("verified_file_count") == len(dataset_entries), "dataset file count changed")
    for relative in dataset_entries:
        require_read_only(dataset_root / relative, name="dataset artifact")
    frozen_training_inputs = training_report.get("frozen_inputs", {})
    require(
        frozen_training_inputs.get("code_manifest_sha256") == training_code.get("manifest_sha256")
        and frozen_training_inputs.get("dataset_manifest_sha256") == dataset.get("manifest_sha256")
        and frozen_training_inputs.get("gate_plan_sha256") == plan["gate_contract"]["sha256"],
        "execution plan differs from the training report's frozen inputs",
    )
    require(
        launch_identity.get("code_manifest_sha256") == training_code.get("manifest_sha256")
        and launch_identity.get("dataset_manifest_sha256") == dataset.get("manifest_sha256")
        and launch_identity.get("gate_plan_sha256") == plan["gate_contract"]["sha256"],
        "execution plan differs from the immutable training launch identity",
    )
    checkpoints = plan.get("checkpoints")
    require(
        isinstance(checkpoints, list)
        and [checkpoint.get("step") for checkpoint in checkpoints] == list(CHECKPOINT_STEPS),
        "checkpoint plan grid changed",
    )
    training_checkpoint_records = training_report.get("checkpoints")
    require(
        isinstance(training_checkpoint_records, list)
        and [record.get("step") for record in training_checkpoint_records] == list(CHECKPOINT_STEPS),
        "training-report checkpoint coverage changed",
    )
    logged_losses_raw = training_report.get("log", {}).get("eval_loss_by_step", {})
    require(
        isinstance(logged_losses_raw, dict)
        and sorted(int(step) for step in logged_losses_raw) == list(CHECKPOINT_STEPS),
        "training-report held-out loss coverage changed",
    )
    for checkpoint in checkpoints:
        root = Path(str(checkpoint.get("model_id", ""))).resolve(strict=True)
        require(
            root
            == (run_root / f"checkpoints/{checkpoint['step']:06d}/pretrained_model").resolve(strict=True),
            "checkpoint path differs from the launch-bound training run",
        )
        require_read_only(root, name=f"checkpoint {checkpoint['step']}")
        require(root.is_dir(), "checkpoint model root is not a directory")
        files_to_hash = [("config.json", "config_json_sha256")]
        if checkpoint_step is None or checkpoint["step"] == checkpoint_step:
            files_to_hash.append(("model.safetensors", "model_safetensors_sha256"))
        for filename, digest_field in files_to_hash:
            digest = require_sha256(
                checkpoint.get(digest_field),
                name=f"checkpoint {checkpoint['step']} {filename} digest",
            )
            require(sha256_file(root / filename) == digest, f"checkpoint {filename} changed")
        require_sha256(checkpoint.get("inventory_sha256"), name="checkpoint inventory digest")
        require_sha256(checkpoint.get("fingerprint"), name="checkpoint fingerprint")
        require(checkpoint.get("revision") == "main", "checkpoint revision changed")
        loss = checkpoint.get("full_holdout_eval_loss")
        require(
            isinstance(loss, int | float) and not isinstance(loss, bool) and math.isfinite(loss),
            "checkpoint immutable validation loss is invalid",
        )
        require(
            float(logged_losses_raw[str(checkpoint["step"])]) == float(loss),
            "checkpoint loss differs from immutable training-report loss",
        )
    cells = plan.get("cells")
    expected_cells = expected_full_cell_grid()
    require(
        isinstance(cells, list)
        and len(cells) == len(expected_cells)
        and [cell.get("task_index") for cell in cells] == list(range(len(cells)))
        and [(cell.get("checkpoint_step"), cell.get("seed"), cell.get("episode")) for cell in cells]
        == expected_cells,
        "execution-plan cell grid changed",
    )
    return plan


def cell_task_index(step: int, seed: int, episode: int) -> int:
    key = (step, seed, episode)
    try:
        return expected_full_cell_grid().index(key)
    except ValueError as error:
        raise Pi05GateError(f"cell is outside the immutable grid: {key}") from error


def validate_server_attestation(
    attestation: dict[str, Any],
    *,
    seed: int,
    checkpoint: dict[str, Any],
    expected_capture_ticks: list[int],
) -> dict[str, Any]:
    """Validate the exact Pi noise stream after the client has closed.

    Protocol v1 performs resets on OpenSession, the producer's explicit reset,
    and CloseSession.  Only the middle reset epoch may contain policy queries.
    Synthetic server/input-shape warmups are separate preserved transactions and
    must leave the generator byte state and reset/query history unchanged.
    """

    exact = {
        "schema_id": SERVER_ATTESTATION_SCHEMA_ID,
        "policy_type": POLICY_TYPE,
        "rng_algorithm": RNG_ALGORITHM,
        "seed": seed,
        "num_flow_steps": NUM_FLOW_STEPS,
        "noise_shape": [1, ACTION_HORIZON, MAX_ACTION_DIM],
        "noise_dtype": "torch.float32",
        "checkpoint_root": checkpoint["model_id"],
        "model_safetensors_sha256": checkpoint["model_safetensors_sha256"],
        "config_json_sha256": checkpoint["config_json_sha256"],
        "hardware_control_performed": False,
        "hardware_deployment_ready": False,
    }
    for key, expected in exact.items():
        require(attestation.get(key) == expected, f"Pi0.5 server attestation changed: {key}")
    require_sha256(attestation.get("server_source_sha256"), name="server source digest")
    require_sha256(attestation.get("code_manifest_sha256"), name="server code-manifest digest")
    runtime = attestation.get("runtime", {})
    require(
        all(runtime.get(key) for key in ("python", "torch", "device", "device_name")),
        "Pi0.5 server runtime attestation is incomplete",
    )
    hf_cache = attestation.get("inference_hf_cache", {})
    verified_hf_cache = validate_inference_hf_home(
        Path(str(hf_cache.get("root", ""))),
        hf_cache.get("manifest_sha256"),
    )
    require(hf_cache == verified_hf_cache, "attested Pi0.5 tokenizer cache changed")
    warmups = attestation.get("warmup_transactions")
    require(isinstance(warmups, list) and len(warmups) == 2, "expected two RNG-neutral warmups")
    for index, warmup in enumerate(warmups):
        require(
            warmup.get("transaction_index") == index
            and warmup.get("completed") is True
            and warmup.get("generator_state_before_sha256") == warmup.get("generator_state_after_sha256")
            and warmup.get("reset_epoch_count_before") == warmup.get("reset_epoch_count_after")
            and warmup.get("query_count_before") == warmup.get("query_count_after"),
            f"warmup {index} consumed Pi0.5 RNG or reset/query state",
        )
    epochs = attestation.get("reset_epochs")
    require(isinstance(epochs, list) and len(epochs) == 3, "unexpected protocol reset count")
    require([epoch.get("reset_ordinal") for epoch in epochs] == [1, 2, 3], "reset order changed")
    require(all(epoch.get("seed") == seed for epoch in epochs), "reset epoch seed changed")
    nonempty = [epoch for epoch in epochs if epoch.get("queries")]
    require(len(nonempty) == 1 and nonempty[0]["reset_ordinal"] == 2, "queries crossed a reset")
    queries = nonempty[0]["queries"]
    require(
        [query.get("query_index") for query in queries] == list(range(len(expected_capture_ticks)))
        and [query.get("observation_sequence") for query in queries]
        == list(range(len(expected_capture_ticks)))
        and [query.get("capture_tick") for query in queries] == expected_capture_ticks,
        "Pi0.5 explicit-noise query stream changed",
    )
    require(
        all(valid_sha256(query.get("noise_float32_sha256")) for query in queries),
        "Pi0.5 noise hashes are missing",
    )
    require(attestation.get("query_count") == len(queries), "attested query count changed")
    return {
        "rng_algorithm": RNG_ALGORITHM,
        "seed": seed,
        "warmup_transaction_count": len(warmups),
        "protocol_reset_count": len(epochs),
        "policy_sampling_reset_ordinal": 2,
        "query_count": len(queries),
        "capture_ticks": expected_capture_ticks,
        "noise_float32_sha256_by_query": [query["noise_float32_sha256"] for query in queries],
        "continuous_stream_within_sampling_epoch": True,
        "fresh_process_required": True,
    }


def pooled_relative_improvement(model_errors: list[float], identity_errors: list[float]) -> float:
    require(
        len(model_errors) == len(identity_errors) and bool(model_errors),
        "pooled behavior vectors have different or zero coverage",
    )
    require(
        all(math.isfinite(value) and value >= 0.0 for value in model_errors + identity_errors),
        "pooled behavior errors must be finite and non-negative",
    )
    identity = math.fsum(identity_errors)
    require(identity > 0.0, "pooled identity behavior error is zero")
    return 1.0 - math.fsum(model_errors) / identity


def rank_dual_gate_candidates(
    stage1_eligible: list[int],
    temporal_eligible: list[int],
    immutable_losses: dict[int, float],
) -> list[int]:
    require(len(stage1_eligible) == len(set(stage1_eligible)), "duplicate stage-one candidate")
    require(len(temporal_eligible) == len(set(temporal_eligible)), "duplicate temporal candidate")
    require(
        set(temporal_eligible) <= set(stage1_eligible),
        "temporal eligibility contains a stage-one failure",
    )
    require(
        all(step in immutable_losses and math.isfinite(immutable_losses[step]) for step in stage1_eligible),
        "immutable validation-loss mapping is incomplete",
    )
    return sorted(temporal_eligible, key=lambda step: (immutable_losses[step], step))
