#!/usr/bin/env python

"""Serve Pi0.5 for onset-v3 gates with an explicit, attested noise stream.

Pi0.5 samples its flow-matching x_1 noise from PyTorch's global RNG by default.
That is unsuitable for fixed-cell gates because server warmup would silently
consume samples.  This adapter supplies the noise tensor explicitly from one
device-local generator, snapshots it across both synthetic warmups, and records
the exact noise hash for every real query.  It never constructs robot hardware.
"""

import copy
import hashlib
import json
import logging
import os
import platform
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import draccus
import numpy as np
import torch

try:
    from . import pi05_onset_v3_gate_common as common
except ImportError:  # Direct execution from examples/umi_yam.
    import pi05_onset_v3_gate_common as common

from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.remote_inference import RemotePolicyServerConfig, serve
from lerobot.remote_inference.backend import (
    LeRobotPolicyBackend,
    LeRobotPolicyBackendConfig,
    _model_fingerprint,
)
from lerobot.remote_inference.schema import (
    ModelManifest,
    PolicyActionChunk,
    PolicyObservation,
    ProtocolValidationError,
)
from lerobot.utils.constants import OBS_STATE

logger = logging.getLogger(__name__)

OFFLINE_GATE_SERVER_MODE = "offline_gate"
SUPERVISED_HARDWARE_TRIAL_SERVER_MODE = "supervised_hardware_trial"
SUPERVISED_HARDWARE_SERVICE_PREFIX = "lerobot-pi05-onset-v3-explicit-noise-seed-"


class ExplicitNoiseStream:
    """One resettable generator whose draws are explicit model inputs."""

    def __init__(self, seed: int, device: torch.device | str):
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise ValueError("Pi0.5 gate seed must be an integer in [0, 2**63)")
        self.seed = seed
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device)
        self.draw_count = 0
        self.reset()

    def reset(self) -> None:
        self.generator.manual_seed(self.seed)
        self.draw_count = 0

    def draw(self, shape: tuple[int, ...]) -> torch.Tensor:
        result = torch.randn(
            shape,
            dtype=torch.float32,
            device=self.device,
            generator=self.generator,
        )
        self.draw_count += 1
        return result

    def snapshot(self) -> tuple[torch.Tensor, int]:
        return self.generator.get_state().clone(), self.draw_count

    def restore(self, snapshot: tuple[torch.Tensor, int]) -> None:
        state, draw_count = snapshot
        self.generator.set_state(state)
        self.draw_count = int(draw_count)

    def state_sha256(self) -> str:
        state = self.generator.get_state().detach().cpu().contiguous().numpy()
        return hashlib.sha256(state.tobytes(order="C")).hexdigest()

    @contextmanager
    def rollback_on_error(self) -> Iterator[None]:
        """Do not advance the explicit stream when an inference fails."""

        snapshot = self.snapshot()
        try:
            yield
        except BaseException:
            self.restore(snapshot)
            raise


@dataclass(frozen=True)
class Pi05GatePolicyBackendConfig(LeRobotPolicyBackendConfig):
    revision: str | None = "main"
    eval_seed: int | None = None
    num_inference_steps: int = common.NUM_FLOW_STEPS
    expected_model_safetensors_sha256: str = ""
    expected_config_json_sha256: str = ""
    code_snapshot_root: str = ""
    code_manifest_sha256: str = ""
    inference_hf_home: str = ""
    inference_hf_manifest_sha256: str = ""
    attestation_path: str = ""
    server_mode: str = OFFLINE_GATE_SERVER_MODE
    server_revision: str = "development"
    server_config_sha256: str = ""


def _static_attestation_identity(
    config: Pi05GatePolicyBackendConfig,
    *,
    hf_cache_record: dict[str, Any],
    server_source_sha256: str,
) -> dict[str, Any]:
    """Return the exact Pi sampling/service identity advertised by the protocol."""

    service_name = (
        f"{SUPERVISED_HARDWARE_SERVICE_PREFIX}{config.eval_seed}"
        if config.server_mode == SUPERVISED_HARDWARE_TRIAL_SERVER_MODE
        else "lerobot-remote-policy"
    )
    return {
        "schema_id": "umi-yam-pi05-explicit-noise-static-identity-v1",
        "policy_type": common.POLICY_TYPE,
        "server_mode": config.server_mode,
        "server_revision": config.server_revision,
        "server_config_sha256": config.server_config_sha256,
        "service_name": service_name,
        "rng_algorithm": common.RNG_ALGORITHM,
        "seed": int(config.eval_seed),
        "num_flow_steps": common.NUM_FLOW_STEPS,
        "noise_shape": [1, common.ACTION_HORIZON, common.MAX_ACTION_DIM],
        "noise_dtype": "torch.float32",
        "model_safetensors_sha256": config.expected_model_safetensors_sha256,
        "config_json_sha256": config.expected_config_json_sha256,
        "code_manifest_sha256": config.code_manifest_sha256,
        "server_source_sha256": server_source_sha256,
        "inference_hf_manifest_sha256": hf_cache_record["manifest_sha256"],
        "model_artifact_manifest_sha256": str(config.artifact_manifest_sha256 or ""),
        "runtime_dependency_manifest_sha256": str(config.runtime_dependency_manifest_sha256 or ""),
        "explicit_noise_argument_required": True,
        "implicit_global_rng_fallback_allowed": False,
    }


class Pi05GatePolicyBackend(LeRobotPolicyBackend):
    """LeRobot backend with an explicit Pi0.5 x_1 generator."""

    def __init__(self, config: Pi05GatePolicyBackendConfig):
        common.require(config.eval_seed in common.SEEDS, "server seed is outside the frozen five-seed grid")
        common.require(config.num_inference_steps == common.NUM_FLOW_STEPS, "flow-step count changed")
        common.require(config.revision == "main", "local checkpoint revision must be main")
        common.require(
            config.server_mode in {OFFLINE_GATE_SERVER_MODE, SUPERVISED_HARDWARE_TRIAL_SERVER_MODE},
            "unsupported Pi0.5 server mode",
        )
        if config.server_mode == SUPERVISED_HARDWARE_TRIAL_SERVER_MODE:
            common.require(
                re.fullmatch(r"[0-9a-f]{40}", config.server_revision) is not None,
                "supervised hardware trial requires an immutable 40-hex server revision",
            )
            common.require_sha256(config.server_config_sha256, name="server runtime-contract digest")
            common.require(
                all(
                    value is not None and str(value).strip()
                    for value in (
                        config.artifact_root,
                        config.artifact_manifest_path,
                        config.artifact_manifest_sha256,
                        config.runtime_dependency_root,
                        config.runtime_dependency_manifest_path,
                        config.runtime_dependency_manifest_sha256,
                    )
                ),
                "supervised hardware trial requires complete model and saved-base attestations",
            )
        checkpoint = Path(config.pretrained_name_or_path).resolve(strict=True)
        common.require(checkpoint.is_dir(), "checkpoint root is not a directory")
        model_path = checkpoint / "model.safetensors"
        config_path = checkpoint / "config.json"
        for path, digest, name in (
            (model_path, config.expected_model_safetensors_sha256, "model.safetensors"),
            (config_path, config.expected_config_json_sha256, "config.json"),
        ):
            common.require(path.is_file(), f"checkpoint {name} is missing")
            common.require_sha256(digest, name=f"expected {name} digest")
            common.require(common.sha256_file(path) == digest, f"checkpoint {name} digest mismatch")

        code_root = Path(config.code_snapshot_root).resolve(strict=True)
        common.require_read_only(code_root, name="post-training code snapshot")
        entries = common.checksum_manifest_entries(
            code_root,
            common.CODE_MANIFEST_RELATIVE,
            config.code_manifest_sha256,
        )
        relative_source = "examples/umi_yam/pi05_onset_v3_policy_server.py"
        common.require(relative_source in entries, "code manifest omits Pi0.5 policy server")
        common.require(
            Path(__file__).resolve(strict=True) == (code_root / relative_source).resolve(strict=True),
            "Pi0.5 server is not running from the plan-bound code snapshot",
        )
        hf_cache_record = common.validate_inference_hf_home(
            Path(config.inference_hf_home),
            config.inference_hf_manifest_sha256,
        )
        common.require(
            Path(os.environ.get("HF_HOME", "")).resolve(strict=True) == Path(hf_cache_record["root"]),
            "HF_HOME differs from the plan-bound Pi0.5 tokenizer cache",
        )
        attestation_path = Path(config.attestation_path).resolve()
        common.require(not attestation_path.exists(), "refusing to overwrite server attestation")

        self._gate_config = config
        self._checkpoint_root = checkpoint
        self._code_root = code_root
        self._attestation_path = attestation_path
        self._hf_cache_record = hf_cache_record
        self._attestation_identity = _static_attestation_identity(
            config,
            hf_cache_record=hf_cache_record,
            server_source_sha256=common.sha256_file(Path(__file__).resolve()),
        )
        self._attestation_identity_sha256 = hashlib.sha256(
            json.dumps(
                self._attestation_identity,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        self._reset_epochs: list[dict[str, Any]] = []
        self._warmup_transactions: list[dict[str, Any]] = []
        super().__init__(config)
        common.require(self._policy_config.type == common.POLICY_TYPE, "served policy is not Pi0.5")
        common.require(self._manifest.action_horizon == common.ACTION_HORIZON, "manifest horizon changed")
        self._noise = ExplicitNoiseStream(int(config.eval_seed), self._device)
        self._write_attestation()

    def _build_manifest(self) -> ModelManifest:
        manifest = super()._build_manifest()
        fingerprint = _model_fingerprint(
            model_id=manifest.model_id,
            revision=manifest.revision,
            policy_type=manifest.policy_type,
            norm_tag=manifest.norm_tag,
            horizon=manifest.action_horizon,
            state_features=manifest.state_features,
            action_features=manifest.action_features,
            camera_keys=manifest.camera_keys,
            gripper_action_representation=manifest.gripper_action_representation,
            artifact_tree_sha256=manifest.artifact_tree_sha256,
            artifact_manifest_sha256=manifest.artifact_manifest_sha256,
            runtime_dependency_tree_sha256=manifest.runtime_dependency_tree_sha256,
            runtime_dependency_manifest_sha256=manifest.runtime_dependency_manifest_sha256,
            inference_backend_mode=self._gate_config.server_mode,
            inference_seed=int(self._gate_config.eval_seed),
            inference_code_manifest_sha256=self._gate_config.code_manifest_sha256,
            inference_attestation_identity_sha256=self._attestation_identity_sha256,
            ik_release_report_sha256=manifest.ik_release_report_sha256,
        )
        result = replace(
            manifest,
            fingerprint=fingerprint,
            inference_backend_mode=self._gate_config.server_mode,
            inference_seed=int(self._gate_config.eval_seed),
            inference_code_manifest_sha256=self._gate_config.code_manifest_sha256,
            inference_attestation_identity_sha256=self._attestation_identity_sha256,
        )
        result.validate()
        return result

    @staticmethod
    def _load_policy_config(config: Pi05GatePolicyBackendConfig):
        # The generic backend interprets eval_seed as a request for a policy-
        # local generator and correctly rejects Pi0.5 (which has no such config
        # field).  This adapter owns the seed and supplies x_1 explicitly, so do
        # not forward that adapter-only seed into the generic policy loader.
        base_config = replace(config, eval_seed=None, per_episode_seed=None)
        policy_config = LeRobotPolicyBackend._load_policy_config(base_config)
        common.require(policy_config.type == common.POLICY_TYPE, "checkpoint policy type is not pi05")
        policy_config.num_inference_steps = config.num_inference_steps
        serialized = common.load_json(Path(config.pretrained_name_or_path) / "config.json")
        common.validate_policy_config(serialized)
        return policy_config

    def _total_query_count(self) -> int:
        return sum(len(epoch["queries"]) for epoch in self._reset_epochs)

    def _runtime(self) -> dict[str, Any]:
        device_name = (
            torch.cuda.get_device_name(self._device) if self._device.type == "cuda" else self._device.type
        )
        return {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_build": str(torch.version.cuda),
            "device": str(self._device),
            "device_name": device_name,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        }

    def _attestation(self) -> dict[str, Any]:
        return {
            "schema_id": common.SERVER_ATTESTATION_SCHEMA_ID,
            "policy_type": common.POLICY_TYPE,
            "rng_algorithm": common.RNG_ALGORITHM,
            "seed": self._noise.seed,
            "num_flow_steps": common.NUM_FLOW_STEPS,
            "noise_shape": [1, common.ACTION_HORIZON, common.MAX_ACTION_DIM],
            "noise_dtype": "torch.float32",
            "checkpoint_root": str(self._checkpoint_root),
            "model_safetensors_sha256": self._gate_config.expected_model_safetensors_sha256,
            "config_json_sha256": self._gate_config.expected_config_json_sha256,
            "code_snapshot_root": str(self._code_root),
            "code_manifest_sha256": self._gate_config.code_manifest_sha256,
            "server_source_sha256": common.sha256_file(Path(__file__).resolve()),
            "inference_hf_cache": self._hf_cache_record,
            "runtime": self._runtime(),
            "warmup_transactions": self._warmup_transactions,
            "reset_epochs": self._reset_epochs,
            "query_count": self._total_query_count(),
            "generator_state_sha256": self._noise.state_sha256(),
            "server_mode": self._gate_config.server_mode,
            "server_revision": self._gate_config.server_revision,
            "server_config_sha256": self._gate_config.server_config_sha256,
            "model_artifact_tree_sha256": self._artifact_tree_sha256,
            "model_artifact_manifest_sha256": self._artifact_manifest_sha256,
            "runtime_dependency_tree_sha256": self._runtime_dependency_tree_sha256,
            "runtime_dependency_manifest_sha256": self._runtime_dependency_manifest_sha256,
            "static_identity": self._attestation_identity,
            "static_identity_sha256": self._attestation_identity_sha256,
            "explicit_noise_argument_required": True,
            "implicit_global_rng_fallback_allowed": False,
            "hardware_control_performed": False,
            "hardware_deployment_ready": False,
        }

    def _write_attestation(self) -> None:
        common.write_json_atomic(self._attestation_path, self._attestation())

    @contextmanager
    def _preserve_inference_rng_state(self) -> Iterator[None]:
        """Extend generic warmup preservation to the explicit Pi generator."""

        noise_snapshot = self._noise.snapshot()
        epochs_snapshot = copy.deepcopy(self._reset_epochs)
        before_state = self._noise.state_sha256()
        before_epoch_count = len(self._reset_epochs)
        before_query_count = self._total_query_count()
        completed = False
        try:
            with super()._preserve_inference_rng_state():
                yield
            completed = True
        finally:
            self._noise.restore(noise_snapshot)
            self._reset_epochs = epochs_snapshot
            after_state = self._noise.state_sha256()
            record = {
                "transaction_index": len(self._warmup_transactions),
                "completed": completed,
                "generator_state_before_sha256": before_state,
                "generator_state_after_sha256": after_state,
                "reset_epoch_count_before": before_epoch_count,
                "reset_epoch_count_after": len(self._reset_epochs),
                "query_count_before": before_query_count,
                "query_count_after": self._total_query_count(),
            }
            self._warmup_transactions.append(record)
            self._write_attestation()
            common.require(completed, "Pi0.5 warmup did not complete")
            common.require(before_state == after_state, "Pi0.5 warmup consumed explicit RNG")

    def reset(self) -> None:
        with self._lock:
            super().reset()
            self._noise.reset()
            self._reset_epochs.append(
                {
                    "reset_ordinal": len(self._reset_epochs) + 1,
                    "seed": self._noise.seed,
                    "generator_state_after_reset_sha256": self._noise.state_sha256(),
                    "queries": [],
                }
            )
            self._write_attestation()

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
        with self._lock:
            common.require(bool(self._reset_epochs), "Pi0.5 inference occurred before a reset")
            with self._noise.rollback_on_error(), torch.inference_mode(), autocast:
                batch = self._preprocessor(batch)
                noise = self._noise.draw((1, common.ACTION_HORIZON, common.MAX_ACTION_DIM))
                noise_sha256 = common.float32_sha256(noise.detach().cpu().numpy())
                # Supplying ``noise`` is mandatory: omitting it would enter Pi0.5's
                # implicit global-RNG fallback and violate the evaluated backend.
                chunk = self._policy.predict_action_chunk(batch, noise=noise)
                common.require(
                    common.float32_sha256(noise.detach().cpu().numpy()) == noise_sha256,
                    "Pi0.5 mutated the explicit input noise tensor",
                )
                if chunk.ndim == 2:
                    chunk = chunk.unsqueeze(0)
                processed = [self._postprocessor(chunk[:, index, :]) for index in range(chunk.shape[1])]
                actions = torch.stack(processed, dim=1).squeeze(0).detach().cpu().float().numpy()
                actions = actions[: self._manifest.action_horizon, : self._manifest.action_dim]
                if actions.shape != (common.ACTION_HORIZON, self._manifest.action_dim):
                    raise ProtocolValidationError(f"policy returned unexpected action shape {actions.shape}")
                epoch = self._reset_epochs[-1]
                epoch["queries"].append(
                    {
                        "query_index": len(epoch["queries"]),
                        "observation_sequence": observation.sequence,
                        "capture_tick": observation.capture_tick,
                        "noise_float32_sha256": noise_sha256,
                    }
                )
                try:
                    self._write_attestation()
                except BaseException:
                    epoch["queries"].pop()
                    raise
        return PolicyActionChunk(
            observation_sequence=observation.sequence,
            first_action_tick=observation.capture_tick,
            actions=np.asarray(actions, dtype=np.float32),
            model_fingerprint=self._manifest.fingerprint,
        )


@dataclass
class Pi05GatePolicyServerConfig:
    policy: Pi05GatePolicyBackendConfig = field(
        default_factory=lambda: Pi05GatePolicyBackendConfig(pretrained_name_or_path="")
    )
    host: str = "127.0.0.1"
    port: int = 8081
    max_message_bytes: int = 16 * 1024 * 1024
    max_image_bytes: int = 8 * 1024 * 1024
    max_decoded_image_bytes: int = 64 * 1024 * 1024
    max_task_chars: int = 4096
    command_ttl_ms: int = 200
    session_idle_timeout_s: float = 600.0
    server_revision: str = "development"
    supervised_hardware_trial: bool = False


def _bound_server_and_policy_configs(
    cfg: Pi05GatePolicyServerConfig,
) -> tuple[RemotePolicyServerConfig, Pi05GatePolicyBackendConfig]:
    """Bind the explicit seed into a hardware trial's session contract."""

    common.require(cfg.host in {"127.0.0.1", "localhost"}, "Pi0.5 gate server must be loopback")
    common.require(cfg.policy.eval_seed in common.SEEDS, "server seed is outside the frozen five-seed grid")
    server_mode = (
        SUPERVISED_HARDWARE_TRIAL_SERVER_MODE if cfg.supervised_hardware_trial else OFFLINE_GATE_SERVER_MODE
    )
    service_name = "lerobot-remote-policy"
    if cfg.supervised_hardware_trial:
        common.require(
            re.fullmatch(r"[0-9a-f]{40}", cfg.server_revision) is not None,
            "supervised hardware trial requires --server_revision=<immutable 40-hex code revision>",
        )
        common.require(cfg.command_ttl_ms == 1000, "supervised hardware trial requires 1000 ms command TTL")
        common.require(
            cfg.session_idle_timeout_s == 600.0,
            "supervised hardware trial requires the reviewed 600 s idle timeout",
        )
        common.require(
            all(
                value is not None and str(value).strip()
                for value in (
                    cfg.policy.artifact_root,
                    cfg.policy.artifact_manifest_path,
                    cfg.policy.artifact_manifest_sha256,
                    cfg.policy.runtime_dependency_root,
                    cfg.policy.runtime_dependency_manifest_path,
                    cfg.policy.runtime_dependency_manifest_sha256,
                )
            ),
            "supervised hardware trial requires complete model and saved-base attestations",
        )
        service_name = f"{SUPERVISED_HARDWARE_SERVICE_PREFIX}{cfg.policy.eval_seed}"
    server_config = RemotePolicyServerConfig(
        host=cfg.host,
        port=cfg.port,
        service_name=service_name,
        server_revision=cfg.server_revision,
        max_message_bytes=cfg.max_message_bytes,
        max_image_bytes=cfg.max_image_bytes,
        max_decoded_image_bytes=cfg.max_decoded_image_bytes,
        max_task_chars=cfg.max_task_chars,
        command_ttl_ms=cfg.command_ttl_ms,
        session_idle_timeout_s=cfg.session_idle_timeout_s,
    )
    server_config.validate()
    policy_config = replace(
        cfg.policy,
        server_mode=server_mode,
        server_revision=cfg.server_revision,
        server_config_sha256=server_config.runtime_contract_sha256(),
    )
    return server_config, policy_config


@draccus.wrap()
def run_server(cfg: Pi05GatePolicyServerConfig) -> None:
    common.require(bool(cfg.policy.pretrained_name_or_path), "checkpoint path is required")
    common.require(bool(cfg.policy.attestation_path), "server attestation path is required")
    common.require(
        os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8",
        "CUBLAS_WORKSPACE_CONFIG must be :4096:8 before CUDA initialization",
    )
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    server_config, policy_config = _bound_server_and_policy_configs(cfg)
    backend = Pi05GatePolicyBackend(policy_config)
    started = time.perf_counter()
    logger.info("Pi0.5 explicit-noise gate backend loaded in %.2fs", time.perf_counter() - started)
    serve(server_config, backend)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    run_server()


if __name__ == "__main__":
    main()
