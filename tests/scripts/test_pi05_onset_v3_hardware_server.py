from __future__ import annotations

import sys
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples/umi_yam"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

import pi05_onset_v3_gate_common as common  # noqa: E402
from pi05_onset_v3_policy_server import (  # noqa: E402
    SUPERVISED_HARDWARE_SERVICE_PREFIX,
    ExplicitNoiseStream,
    Pi05GatePolicyBackend,
    Pi05GatePolicyBackendConfig,
    Pi05GatePolicyServerConfig,
    _bound_server_and_policy_configs,
    _static_attestation_identity,
)

from lerobot.remote_inference.backend import LeRobotPolicyBackend  # noqa: E402


def test_explicit_noise_reset_snapshot_and_failed_draw_are_bit_exact() -> None:
    stream = ExplicitNoiseStream(3, "cpu")
    shape = (1, common.ACTION_HORIZON, common.MAX_ACTION_DIM)
    first = stream.draw(shape)
    assert stream.draw_count == 1 and first.dtype == torch.float32
    stream.reset()
    assert torch.equal(stream.draw(shape), first)

    snapshot = stream.snapshot()
    expected_next = stream.draw(shape)
    stream.restore(snapshot)
    assert torch.equal(stream.draw(shape), expected_next)

    stream.reset()
    with pytest.raises(RuntimeError, match="synthetic inference failure"), stream.rollback_on_error():
        failed_noise = stream.draw(shape).clone()
        raise RuntimeError("synthetic inference failure")
    assert stream.draw_count == 0
    assert torch.equal(stream.draw(shape), failed_noise)


def test_backend_retries_failed_explicit_noise_call_bit_exactly(monkeypatch) -> None:
    import pi05_onset_v3_policy_server as server_module

    class FailOncePolicy:
        def __init__(self):
            self.noises: list[torch.Tensor] = []

        def predict_action_chunk(self, _batch, *, noise):
            self.noises.append(noise.clone())
            if len(self.noises) == 1:
                raise RuntimeError("synthetic model failure")
            return noise[:, :, :20]

    backend = Pi05GatePolicyBackend.__new__(Pi05GatePolicyBackend)
    backend._lock = threading.RLock()
    backend._device = torch.device("cpu")
    backend._noise = ExplicitNoiseStream(2, "cpu")
    backend._reset_epochs = [{"reset_ordinal": 1, "seed": 2, "queries": []}]
    backend._policy_config = SimpleNamespace(model_dtype="float32")
    backend._manifest = SimpleNamespace(action_horizon=24, action_dim=20, fingerprint="f" * 64)
    backend._preprocessor = lambda batch: batch
    backend._postprocessor = lambda action: action
    backend._policy = FailOncePolicy()
    backend._write_attestation = lambda: None
    monkeypatch.setattr(server_module, "prepare_observation_for_inference", lambda *args: args[0])
    observation = SimpleNamespace(
        state=np.zeros(20, dtype=np.float32),
        images=(),
        task=common.TASK,
        sequence=0,
        capture_tick=0,
    )

    with pytest.raises(RuntimeError, match="synthetic model failure"):
        backend.infer(observation)
    assert backend._noise.draw_count == 0
    result = backend.infer(observation)
    assert torch.equal(backend._policy.noises[0], backend._policy.noises[1])
    assert result.actions.shape == (24, 20)
    assert backend._noise.draw_count == 1
    assert len(backend._reset_epochs[0]["queries"]) == 1


def _hardware_policy(seed: int = 3) -> Pi05GatePolicyBackendConfig:
    return Pi05GatePolicyBackendConfig(
        pretrained_name_or_path="/read-only/checkpoint",
        eval_seed=seed,
        artifact_root="/read-only/checkpoint",
        artifact_manifest_path="/read-only/manifests/checkpoint.json",
        artifact_manifest_sha256="a" * 64,
        runtime_dependency_root="/read-only/pi05-base",
        runtime_dependency_manifest_path="/read-only/manifests/pi05-base.json",
        runtime_dependency_manifest_sha256="c" * 64,
    )


def test_supervised_server_contract_binds_seed_service_and_artifacts() -> None:
    policy = _hardware_policy()
    cfg = Pi05GatePolicyServerConfig(
        policy=policy,
        command_ttl_ms=1000,
        session_idle_timeout_s=600.0,
        server_revision="b" * 40,
        supervised_hardware_trial=True,
    )
    server_config, bound_policy = _bound_server_and_policy_configs(cfg)
    assert server_config.service_name == f"{SUPERVISED_HARDWARE_SERVICE_PREFIX}3"
    assert bound_policy.server_mode == "supervised_hardware_trial"
    assert bound_policy.server_config_sha256 == server_config.runtime_contract_sha256()

    missing_base = replace(
        cfg,
        policy=replace(
            policy,
            runtime_dependency_root=None,
            runtime_dependency_manifest_path=None,
            runtime_dependency_manifest_sha256=None,
        ),
    )
    with pytest.raises(common.Pi05GateError, match="model and saved-base"):
        _bound_server_and_policy_configs(missing_base)

    different_server, _ = _bound_server_and_policy_configs(replace(cfg, policy=replace(policy, eval_seed=4)))
    assert different_server.runtime_contract_sha256() != server_config.runtime_contract_sha256()


def test_static_identity_binds_noise_service_seed_and_artifact_manifests() -> None:
    config = replace(
        _hardware_policy(seed=4),
        expected_model_safetensors_sha256="a" * 64,
        expected_config_json_sha256="b" * 64,
        code_manifest_sha256="c" * 64,
        server_mode="supervised_hardware_trial",
        server_revision="f" * 40,
        server_config_sha256="1" * 64,
    )
    identity = _static_attestation_identity(
        config,
        hf_cache_record={"manifest_sha256": "2" * 64},
        server_source_sha256="3" * 64,
    )

    assert identity["server_mode"] == "supervised_hardware_trial"
    assert identity["seed"] == 4
    assert identity["service_name"] == f"{SUPERVISED_HARDWARE_SERVICE_PREFIX}4"
    assert identity["server_config_sha256"] == "1" * 64
    assert identity["model_artifact_manifest_sha256"] == "a" * 64
    assert identity["runtime_dependency_manifest_sha256"] == "c" * 64
    assert identity["explicit_noise_argument_required"] is True
    assert identity["implicit_global_rng_fallback_allowed"] is False


def test_eval_seed_is_not_forwarded_to_pi_policy_config(monkeypatch) -> None:
    captured = {}

    def fake_base_loader(config):
        captured["eval_seed"] = config.eval_seed
        captured["per_episode_seed"] = config.per_episode_seed
        return SimpleNamespace(type=common.POLICY_TYPE, num_inference_steps=None)

    monkeypatch.setattr(LeRobotPolicyBackend, "_load_policy_config", fake_base_loader)
    monkeypatch.setattr(common, "load_json", lambda _path: {})
    monkeypatch.setattr(common, "validate_policy_config", lambda _config: None)
    policy_config = Pi05GatePolicyBackend._load_policy_config(
        Pi05GatePolicyBackendConfig(pretrained_name_or_path="/synthetic/checkpoint", eval_seed=3)
    )
    assert captured == {"eval_seed": None, "per_episode_seed": None}
    assert policy_config.num_inference_steps == common.NUM_FLOW_STEPS
