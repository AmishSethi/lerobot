# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy, _combine_rollout_seeds
from lerobot.processor.rename_processor import RenameObservationsProcessorStep
from lerobot.remote_inference import backend as backend_module
from lerobot.remote_inference.backend import (
    CANONICAL_JSON_SPEC,
    MODEL_ARTIFACT_MANIFEST_SCHEMA_ID,
    LeRobotPolicyBackend,
    LeRobotPolicyBackendConfig,
    verify_model_artifact_manifest,
)
from lerobot.utils.constants import ACTION, OBS_STATE


def test_original_molmoact2_schema_is_derived_from_norm_metadata(monkeypatch):
    state_names = ["left_joint_0.pos", "left_gripper.pos"]
    action_names = ["left_joint_0.pos", "left_gripper.pos"]
    metadata = {
        "state_stats": {"names": state_names},
        "action_stats": {"names": action_names},
        "camera_keys": ["observation.images.top", "observation.images.left"],
        "normalize_gripper": False,
    }
    stats = {OBS_STATE: {"q01": [0.0, 0.0]}, ACTION: {"q01": [0.0, 0.0]}}

    from lerobot.policies.molmoact2 import processor_molmoact2

    monkeypatch.setattr(
        processor_molmoact2,
        "_load_hf_norm_stats_for_tag",
        lambda *args, **kwargs: (stats, metadata),
    )
    config = MolmoAct2Config(checkpoint_path="test/model", norm_tag="test")

    resolved_stats = LeRobotPolicyBackend._configure_original_molmoact2(config)

    assert resolved_stats is stats
    assert config.dataset_feature_names == {OBS_STATE: state_names, ACTION: action_names}
    assert config.input_features[OBS_STATE].shape == (2,)
    assert config.input_features[OBS_STATE].type is FeatureType.STATE
    assert config.output_features[ACTION].shape == (2,)
    assert config.output_features[ACTION].type is FeatureType.ACTION
    assert config.image_keys == ["observation.images.top", "observation.images.left"]
    assert all(config.input_features[key].type is FeatureType.VISUAL for key in config.image_keys)


def test_saved_policy_loads_from_backend_path(monkeypatch, tmp_path):
    loaded = {}

    class FakePolicy:
        @classmethod
        def from_pretrained(cls, pretrained_name_or_path, **kwargs):
            loaded["path"] = pretrained_name_or_path
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

    policy_config = SimpleNamespace(type="fake", pretrained_path=None, pretrained_revision=None)

    monkeypatch.setattr(LeRobotPolicyBackend, "_load_policy_config", lambda self, config: policy_config)
    monkeypatch.setattr(LeRobotPolicyBackend, "_build_manifest", lambda self: object())
    monkeypatch.setattr(backend_module, "get_policy_class", lambda policy_type: FakePolicy)

    def fake_make_pre_post_processors(*args, **kwargs):
        loaded["processor_kwargs"] = kwargs
        return object(), object()

    monkeypatch.setattr(backend_module, "make_pre_post_processors", fake_make_pre_post_processors)

    checkpoint = tmp_path / "pretrained_model"
    LeRobotPolicyBackend(LeRobotPolicyBackendConfig(pretrained_name_or_path=str(checkpoint), device="cpu"))

    assert loaded["path"] == str(checkpoint)
    assert loaded["processor_kwargs"]["preprocessor_overrides"] == {"device_processor": {"device": "cpu"}}
    assert loaded["processor_kwargs"]["postprocessor_overrides"] == {"device_processor": {"device": "cpu"}}


@pytest.mark.parametrize(
    ("policy_type", "path_attribute", "revision_attribute", "revision"),
    (
        (
            "molmoact2",
            "checkpoint_path",
            "checkpoint_revision",
            "e432d85f6e039edca44afb93c262f3084ab72a9c",
        ),
        (
            "pi05",
            "pretrained_path",
            "pretrained_revision",
            "b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba",
        ),
    ),
)
def test_manifested_external_base_is_portably_rebound_without_changing_revision(
    tmp_path,
    policy_type,
    path_attribute,
    revision_attribute,
    revision,
):
    local_base = tmp_path / f"{policy_type}-base"
    local_base.mkdir()
    config = SimpleNamespace(
        type=policy_type,
        **{
            path_attribute: "/training/host/absolute/base/path",
            revision_attribute: revision,
        },
    )

    source = LeRobotPolicyBackend._rebind_runtime_dependency(config, str(local_base))

    assert source == {
        "saved_path": "/training/host/absolute/base/path",
        "saved_revision": revision,
        "verified_local_root": str(local_base.resolve()),
    }
    assert getattr(config, path_attribute) == str(local_base.resolve())
    assert getattr(config, revision_attribute) == revision


@pytest.mark.parametrize("policy_type", ("molmoact2", "pi05"))
def test_external_base_rebind_rejects_mutable_or_missing_source_revision(tmp_path, policy_type):
    local_base = tmp_path / "base"
    local_base.mkdir()
    if policy_type == "molmoact2":
        config = SimpleNamespace(
            type=policy_type,
            checkpoint_path="allenai/MolmoAct2",
            checkpoint_revision="main",
        )
    else:
        config = SimpleNamespace(
            type=policy_type,
            pretrained_path="lerobot/pi05_base",
            pretrained_revision=None,
        )

    with pytest.raises(ValueError, match="immutable 40-hex revision"):
        LeRobotPolicyBackend._rebind_runtime_dependency(config, str(local_base))


def _write_frozen_model_artifact(tmp_path):
    root = tmp_path / "pretrained_model"
    root.mkdir()
    (root / "model.safetensors").write_bytes(b"weights-v4")
    (root / "config.json").write_text('{"policy":"v4"}\n', encoding="utf-8")
    (root / "policy_preprocessor.json").write_text('{"normalize":true}\n', encoding="utf-8")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    files_sha256 = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in files
    }
    files_size_bytes = {path.relative_to(root).as_posix(): path.stat().st_size for path in files}
    tree_sha256 = backend_module._canonical_json_sha256(files_sha256)
    manifest_payload = {
        "schema_id": MODEL_ARTIFACT_MANIFEST_SCHEMA_ID,
        "canonical_json_spec": CANONICAL_JSON_SPEC,
        "file_count": len(files),
        "files_sha256": files_sha256,
        "files_size_bytes": files_size_bytes,
        "tree_sha256": tree_sha256,
    }
    manifest = tmp_path / "model-artifact-manifest.json"
    manifest.write_text(json.dumps(manifest_payload, sort_keys=True) + "\n", encoding="utf-8")
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    for path in files:
        path.chmod(0o444)
    root.chmod(0o555)
    manifest.chmod(0o444)
    return root, manifest, manifest_sha256, tree_sha256


def test_model_artifact_canonical_json_matches_v4_training_contract_vector():
    vector = {
        "schema_id": "umi-yam-v4-canonical-json-test-vector-v1",
        "fixed_global_start_anchor": {
            "left_arm_joint_radians": [0.0, 0.6, 0.6, 0.0, 0.0, 0.0],
            "right_arm_joint_radians": [0.0, 0.45, 0.75, 0.0, -0.15, 0.0],
        },
        "candidate_selection_episodes": [14, 15, 134],
        "final_confirmation_episodes": [0, 149],
        "passed": True,
        "failed_rows": [],
    }

    assert CANONICAL_JSON_SPEC == "utf8-json-sorted-compact-ensure-ascii-false-reject-nonfinite-v1"
    assert backend_module._canonical_json_sha256(vector) == (
        "5e198db0a3ec62076610fd2cda3c5fe9b07df299adf07a33064667b9a5b423a0"
    )


def test_model_artifact_manifest_binds_all_weights_config_processors_and_rejects_mutation(
    tmp_path,
):
    root, manifest, manifest_sha256, tree_sha256 = _write_frozen_model_artifact(tmp_path)

    assert (
        verify_model_artifact_manifest(
            artifact_root=root,
            manifest_path=manifest,
            expected_manifest_sha256=manifest_sha256,
        )
        == tree_sha256
    )

    weights = root / "model.safetensors"
    weights.chmod(0o644)
    weights.write_bytes(b"tampered!!")
    weights.chmod(0o444)
    with pytest.raises(ValueError, match="SHA-256 mismatch.*model.safetensors"):
        verify_model_artifact_manifest(
            artifact_root=root,
            manifest_path=manifest,
            expected_manifest_sha256=manifest_sha256,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("extra", "inventory mismatch"),
        ("symlink", "contains a symlink"),
        ("writable", "contains a writable path"),
    ),
)
def test_model_artifact_manifest_rejects_unattested_or_mutable_entries(tmp_path, mutation, message):
    root, manifest, manifest_sha256, _tree_sha256 = _write_frozen_model_artifact(tmp_path)
    if mutation == "writable":
        (root / "config.json").chmod(0o644)
    else:
        root.chmod(0o755)
        if mutation == "extra":
            extra = root / "unattested.json"
            extra.write_text("{}\n", encoding="utf-8")
            extra.chmod(0o444)
        else:
            outside = tmp_path / "outside.bin"
            outside.write_bytes(b"not part of the model")
            (root / "escape.bin").symlink_to(outside)
        root.chmod(0o555)

    with pytest.raises(ValueError, match=message):
        verify_model_artifact_manifest(
            artifact_root=root,
            manifest_path=manifest,
            expected_manifest_sha256=manifest_sha256,
        )


def test_jaw_delta_backend_verifies_artifact_before_policy_config_load(monkeypatch, tmp_path):
    root, manifest, manifest_sha256, _tree_sha256 = _write_frozen_model_artifact(tmp_path)
    report = tmp_path / "ik-release.json"
    report.write_text("{}\n", encoding="utf-8")
    report_sha256 = hashlib.sha256(report.read_bytes()).hexdigest()
    report.chmod(0o444)
    weights = root / "model.safetensors"
    weights.chmod(0o644)
    weights.write_bytes(b"mutated-before-load")
    weights.chmod(0o444)
    config_loaded = False

    def forbidden_config_load(*_args, **_kwargs):
        nonlocal config_loaded
        config_loaded = True
        raise AssertionError("policy config must not load before artifact verification")

    monkeypatch.setattr(LeRobotPolicyBackend, "_load_policy_config", forbidden_config_load)
    with pytest.raises(ValueError, match="model artifact.*mismatch"):
        LeRobotPolicyBackend(
            LeRobotPolicyBackendConfig(
                pretrained_name_or_path=str(root),
                device="cpu",
                gripper_action_representation="query_anchor_delta_normalized_width",
                artifact_manifest_path=str(manifest),
                artifact_manifest_sha256=manifest_sha256,
                ik_release_report_path=str(report),
                ik_release_report_sha256=report_sha256,
            )
        )
    assert config_loaded is False


def test_hub_artifact_requires_materialized_root_and_immutable_commit(tmp_path):
    _root, manifest, manifest_sha256, _tree_sha256 = _write_frozen_model_artifact(tmp_path)

    with pytest.raises(ValueError, match="40-hex commit revision"):
        LeRobotPolicyBackend._verified_load_config(
            LeRobotPolicyBackendConfig(
                pretrained_name_or_path="brandonyang/v4-model",
                revision="main",
                artifact_root=str(tmp_path / "pretrained_model"),
                artifact_manifest_path=str(manifest),
                artifact_manifest_sha256=manifest_sha256,
            )
        )


def test_backend_applies_optional_inference_seed_overrides(monkeypatch):
    policy_config = SimpleNamespace(
        device="cpu",
        per_episode_seed=False,
        eval_seed=None,
    )
    monkeypatch.setattr(
        backend_module.PreTrainedConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: policy_config,
    )

    loaded = LeRobotPolicyBackend._load_policy_config(
        LeRobotPolicyBackendConfig(
            pretrained_name_or_path="test/checkpoint",
            device="cpu",
            per_episode_seed=True,
            eval_seed=17,
        )
    )

    assert loaded is policy_config
    assert loaded.per_episode_seed is True
    assert loaded.eval_seed == 17


def test_backend_rejects_seed_override_for_unsupported_policy(monkeypatch):
    policy_config = SimpleNamespace(device="cpu")
    monkeypatch.setattr(
        backend_module.PreTrainedConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: policy_config,
    )

    with pytest.raises(ValueError, match="does not support rollout-local"):
        LeRobotPolicyBackend._load_policy_config(
            LeRobotPolicyBackendConfig(
                pretrained_name_or_path="test/checkpoint",
                device="cpu",
                per_episode_seed=True,
            )
        )


def test_shape_prepare_warmup_does_not_consume_real_task_rollout_seed(monkeypatch):
    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = MolmoAct2Config(per_episode_seed=True, eval_seed=41)
    policy._rollout_action_generator = None
    policy._rollout_task_key = None
    policy._rollout_index_for_task = -1

    backend = LeRobotPolicyBackend.__new__(LeRobotPolicyBackend)
    backend._lock = backend_module.threading.RLock()
    backend._policy = policy
    backend._manifest = SimpleNamespace(state_features=("state",))
    backend._dataset_stats = None
    backend._prepared_input = None
    monkeypatch.setattr(backend, "reset", policy.reset)
    monkeypatch.setattr(
        backend,
        "infer",
        lambda observation: policy._rollout_generator_for_inputs(
            {"task": [observation.task]},
            batch_size=1,
            device=torch.device("cpu"),
        ),
    )

    backend._run_warmup((), task="pick")

    assert policy._rollout_task_key is None
    assert policy._rollout_index_for_task == -1
    assert policy._rollout_action_generator is None
    first_real_generator = policy._rollout_generator_for_inputs(
        {"task": ["pick"]},
        batch_size=1,
        device=torch.device("cpu"),
    )
    expected = torch.Generator().manual_seed(_combine_rollout_seeds(first_seed=41, batch_size=1))
    torch.testing.assert_close(
        torch.rand(4, generator=first_real_generator),
        torch.rand(4, generator=expected),
    )
    assert policy._rollout_index_for_task == 0


class _SavedStateStats:
    def __init__(self, state_dim: int):
        self.state_dim = state_dim

    def state_dict(self):
        return {f"{OBS_STATE}.q01": torch.zeros(self.state_dim)}


def _pi05_backend(rename_map: dict[str, str], *, model_state_dim: int = 32) -> LeRobotPolicyBackend:
    joint_names = [
        *(f"left_joint_{index}.pos" for index in range(6)),
        "left_gripper.pos",
        *(f"right_joint_{index}.pos" for index in range(6)),
        "right_gripper.pos",
    ]
    backend = LeRobotPolicyBackend.__new__(LeRobotPolicyBackend)
    backend._config = LeRobotPolicyBackendConfig(pretrained_name_or_path="test/pi05-yam", device="cpu")
    backend._policy_config = SimpleNamespace(
        type="pi05",
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(model_state_dim,)),
            "observation.images.base_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            "observation.images.left_wrist_0_rgb": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 224, 224)
            ),
            "observation.images.right_wrist_0_rgb": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 224, 224)
            ),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(14,))},
        dataset_feature_names=None,
        action_feature_names=joint_names,
        n_action_steps=30,
        chunk_size=30,
        norm_tag=None,
    )
    backend._preprocessor = SimpleNamespace(
        steps=[RenameObservationsProcessorStep(rename_map=rename_map), _SavedStateStats(14)]
    )
    return backend


def test_pi05_manifest_uses_raw_processor_state_and_camera_contract():
    backend = _pi05_backend(
        {
            "observation.images.top": "observation.images.base_0_rgb",
            "observation.images.left": "observation.images.left_wrist_0_rgb",
            "observation.images.right": "observation.images.right_wrist_0_rgb",
        }
    )

    manifest = backend._build_manifest()

    assert len(manifest.state_features) == 14
    assert manifest.state_features == manifest.action_features
    assert manifest.action_dim == 14
    assert manifest.action_horizon == 30
    assert manifest.camera_keys == ("top", "left", "right")
    assert backend._configured_image_size("top") == (224, 224)


def test_pi05_manifest_omits_unmapped_optional_camera():
    backend = _pi05_backend(
        {
            "observation.images.left": "observation.images.left_wrist_0_rgb",
            "observation.images.right": "observation.images.right_wrist_0_rgb",
        }
    )

    manifest = backend._build_manifest()

    assert manifest.camera_keys == ("left", "right")


def test_non_padded_policy_does_not_reuse_action_names_for_state():
    backend = _pi05_backend({}, model_state_dim=14)

    manifest = backend._build_manifest()

    assert manifest.state_features == tuple(f"{OBS_STATE}.{index}" for index in range(14))
    assert manifest.action_features != manifest.state_features


def test_pi05_manifest_preserves_distinct_current_relative_state_and_action_names():
    backend = _pi05_backend(
        {
            "observation.images.umi1": "observation.images.left_wrist_0_rgb",
            "observation.images.umi2": "observation.images.right_wrist_0_rgb",
        }
    )
    state_names = tuple(f"state_previous_{index}" for index in range(14))
    action_names = tuple(f"action_future_{index}" for index in range(14))
    backend._policy_config.dataset_feature_names = {
        OBS_STATE: list(state_names),
        ACTION: list(action_names),
    }

    manifest = backend._build_manifest()

    assert manifest.state_features == state_names
    assert manifest.action_features == action_names
    assert manifest.state_features != manifest.action_features
    assert manifest.camera_keys == ("umi1", "umi2")


def test_manifest_carries_gripper_action_representation_separately_from_norm_tag():
    backend = _pi05_backend({})
    backend._config = LeRobotPolicyBackendConfig(
        pretrained_name_or_path="test/pi05-yam",
        device="cpu",
        gripper_action_representation="query_anchor_delta_normalized_width",
    )
    backend._policy_config.norm_tag = "independent-normalization-tag"
    backend._artifact_tree_sha256 = "a" * 64
    backend._artifact_manifest_sha256 = "c" * 64
    backend._ik_release_report_sha256 = "b" * 64

    manifest = backend._build_manifest()

    assert manifest.norm_tag == "independent-normalization-tag"
    assert manifest.gripper_action_representation == "query_anchor_delta_normalized_width"
    assert manifest.artifact_tree_sha256 == "a" * 64
    assert manifest.artifact_manifest_sha256 == "c" * 64
    assert manifest.ik_release_report_sha256 == "b" * 64


def test_empty_gripper_representation_preserves_legacy_fingerprint_bytes():
    fields = {
        "model_id": "test/model",
        "revision": "main",
        "policy_type": "molmoact2",
        "norm_tag": "old-stats",
        "horizon": 24,
        "state_features": ("state.0",),
        "action_features": ("action.0",),
        "camera_keys": ("umi1", "umi2"),
    }
    expected_payload = json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()

    legacy = backend_module._model_fingerprint(**fields)
    explicit_empty_attestation = backend_module._model_fingerprint(
        **fields,
        artifact_tree_sha256="",
        artifact_manifest_sha256="",
        ik_release_report_sha256="",
    )
    explicit = backend_module._model_fingerprint(
        **fields,
        gripper_action_representation="query_anchor_delta_normalized_width",
        artifact_tree_sha256="a" * 64,
        artifact_manifest_sha256="c" * 64,
        ik_release_report_sha256="b" * 64,
    )

    assert legacy == hashlib.sha256(expected_payload).hexdigest()
    assert explicit_empty_attestation == legacy
    assert explicit != legacy
