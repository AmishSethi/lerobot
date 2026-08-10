# UMI onset-v3 experimental hardware trial

This guide covers the original 54-episode onset-v3 dataset and its two 12,000-step
checkpoints:

- MolmoAct2: `molmoact2-umi-currentrel-r6d-onset-v3-12k`
- Pi0.5: `pi05-umi-currentrel-r6d-onset-v3-12k-2n4g`

These are **experimental hardware trials**, not deployment releases. Software
evidence shows that both checkpoints and the bridge agree on their tensor and
action contracts. It does not prove autonomous task success. A trained operator,
physical E-stop, empty guarded workspace, and immediate abort authority are
mandatory. Never run either model unattended.

## Install this exact software snapshot

Check out the immutable commit reviewed in this PR and verify `git status
--short` is empty. Use Python 3.12 and the checked-in lock file; do not install
the hardware machine from a moving branch:

```bash
uv python install 3.12

# GPU policy server: choose exactly one policy extra.
uv sync --locked --python 3.12 --extra remote --extra molmoact2
uv sync --locked --python 3.12 --extra remote --extra pi

# Robot computer. umi_yam_rig includes the MuJoCo geometry checker used at the
# final driver boundary.
uv sync --locked --python 3.12 --extra core_scripts --extra umi_yam_rig
```

The resulting commit identifies the software source, but it is not by itself a
Pi code-snapshot attestation. For Pi, generate and review the required
`.umi_yam_snapshot.sha256` inside a separate read-only copy, pass that copy to
`--policy.code_snapshot_root`, and bind its manifest digest in both server and
client configuration. A clean install/no-motion handshake may be software
`GO`; physical motion remains `NO_GO` until every later physical gate passes.

## What the software evidence does and does not say

Both policies consume two wrist images in `umi1` (left), `umi2` (right) order,
plus a 20-D state. They return exactly 24 rows of 20-D actions. The supervised
bridge executes rows 1–15, waits for measured progress after each row, and then
requeries. Rows 16–24 are never dispatched.

Per arm the state/action order is:

| indices | meaning |
| --- | --- |
| 0–2 | translation, metres |
| 3–5 | first column of active rotation matrix |
| 6–8 | second column of active rotation matrix |
| 9 | jaw width divided by 125 mm |

The left arm occupies indices 0–9 and the right arm 10–19. Pose actions are
`inverse(T_query) @ T_future(k)` for `k=1..24`. Jaw actions are absolute future
widths, not deltas and not row-to-row integrals. MolmoAct2 intentionally leaves
the two jaw dimensions unnormalized; Pi0.5 has relative-action processing
disabled. Do not use the jaw-delta v4 server/config with these checkpoints.

The MolmoAct2 12k checkpoint is structurally intact and teacher-forced temporal
replay shows coherent later task phases, strong lag-zero correlation, near-unit
amplitude in the long carried-q examples, and a large improvement over holding.
Its 15-row onset diagnostic also showed right-arm under-amplitude and seed
variation. That prefix is only about 0.5 seconds; the left target motion is small,
so its ratio to a small hold-error denominator is fragile. The prefix is a
start-chunk safety/consistency check, **not** proof of autonomous failure.

The Pi0.5 12k run completed and validated all training/checkpoint artifacts. Its
final validation loss is finite, but the checkpoint-selection hardware gate must
still be run with the reviewed explicit-noise server. Treat it as having a chance
to work, not as selected or proven.

## Immutable identities to review

Do not copy shortened digests into a release form. Recompute and compare the full
values from the immutable reports.

| item | required identity |
| --- | --- |
| MolmoAct2 model | `model.safetensors` SHA-256 `475db7a53a2c3d816928eaaf5e120ce12d4e0961d66dea7b233f2114c67f3646` |
| MolmoAct2 validation | recovered full report SHA-256 `7815da37822194317063279d6255d5a16c36668a80aebc4e118f50a1b34c9e31` |
| MolmoAct2 release manifest | SHA-256 `2cca024d756047f96b6e32a4f06e7edafd925c832547e726401c52b71b0de550` |
| MolmoAct2 release tree | SHA-256 `a52411f7b01c846273e37b34ffb4fde1e53b21123ebebceed6eb5b4d57488dd0` (9 regular files) |
| MolmoAct2 base manifest | SHA-256 `e57426b7a672b438bcc6f40b2c44da4cbf2fd8f2f58d8008a1dc7a702dd3bd55` |
| MolmoAct2 base tree | SHA-256 `3edc7aecc0edc0f9ed58eba15ccc4c7ae1f5e564a832b0d959092ac31696343f` (23 regular files, including `inventory.sha256`) |
| Pi0.5 model | `model.safetensors` SHA-256 `9979ac5350ffe4bb711d03446d93e7d88fedc4bacd6a6fff0ecc12ad4d306f80` |
| Pi0.5 config | `config.json` SHA-256 `0b0fd4c2d925d6a4f96ee2d7fbd1c79c45113f2c3ef0c7f453bb452a0154afcb` |
| Pi0.5 validation | report SHA-256 `c0dd4cf0647185d3c243b4a6c126baff632492d86c9ec8c127e76c5e9718213a` |
| Pi0.5 release manifest | SHA-256 `6e76648410e43f16068f0a85debfd42b78934281372a586d14150033c3830f33` |
| Pi0.5 release tree | SHA-256 `2eea239cc9a0c1e9259f27906198ce3967a5d022769aacb9bdc1b8df48e91dc3` (7 regular files) |
| Pi0.5 initialization-provenance manifest | SHA-256 `33f420d795b02da299d75a25db2be5b16e227a340231a6371099efa3988c6386` |
| Pi0.5 initialization-provenance tree | SHA-256 `c38f701f2862d89227c74fbf6d166fc07207ccc44b1b9228b9c0e77ba16d096c` (7 regular files) |
| Pi0.5 PaliGemma tokenizer-cache manifest | SHA-256 `365242a26fd695da5ccebceeebd51600b435d6a3dffa555db68119f83fbd4c10` (8 exact files at revision `35e4f46485b4d07967e7e9935bc3786aad50687c`) |
| evaluated reset plan | `examples/umi_yam/onset_v3_checkpoint_gate.json`, SHA-256 `f5841191715930db0c9edd29aa96cc45971fe1f36460c70e03f40e596ccd4d36` |

The earlier Molmo base manifest
`molmoact2-base-e432d85-20260810T174000Z.json` (SHA-256
`66f253a1d893544d58c0064335a8b942de16c8d8a9ffd82ea86f5cd28553feea`)
inventories only 22 files and is **superseded**: the immutable base
now intentionally contains `inventory.sha256`. It must not appear in a server or
hardware allowlist. Use only the full 23-file manifest above.

The evaluated arm anchor is `[0, 0.05, 0.05, 0, 0, 0]` radians for **each**
arm. All-zero is not interchangeable and the runtime rejects it when the gate
plan is configured. The report says `hardware_start_verified=false`; therefore a
physical reset/clearance commissioning record is required before motion.

## Constants the hardware team must fill

Start from exactly one model-specific template:

- `examples/umi_yam/biyam_onset_v3_molmoact_hardware_trial.example.yaml`
- `examples/umi_yam/biyam_onset_v3_pi05_hardware_trial.example.yaml`

Every `REPLACE_...`, `/path/to/...`, and all-zero digest is a hard stop. Review:

1. stable robot ID, distinct CAN-adapter serials, and read-only BiYAM calibration;
2. stable `/dev/v4l/by-id/...` wrist-camera identifiers, left/right ordering, and
   native 800x600 at 30 Hz (do not capture 416x312 and upscale);
3. trusted YAM URDF and its pinned digest;
4. read-only gripper endpoint calibration and its exact SHA-256;
5. read-only rig/table calibration with physical—not synthetic—provenance;
6. complete model artifact root and the exact revision-pinned external base
   snapshot, materialized as regular files at any local path, with separate
   external manifest digests, derived tree digests, and the resulting model
   fingerprint;
7. immutable server revision, effective non-secret server-config digest, and
   exact advertised service name;
8. command TTL and worker-expiry margin derived from measured capture-to-driver
   timing, not copied from the example;
9. fresh telemetry output path;
10. exact checkpoint/gate paths and digests.

The two UMI devices have different width endpoints. Perform an arm-specific
physical endpoint sweep and verify the inverse YAM-to-UMI mapping at the reset
position. In particular, `YAM gripper=1` must not be assumed to mean policy
state `1.0`. The original training right-open support is around 0.94. Record the
actual measured model-state scalars and abort if calibration maps outside the
frozen training support.

## Build and attest the model tree

Materialize the checkpoint at a local read-only directory. Its saved policy
configuration names a second tree outside that directory. MolmoAct2's
`config.checkpoint_path` is the absolute Kempner training path
`/n/holylabs/kempner_ydu_lab/Lab/asethi/molmoact-ft/models/MolmoAct2-e432d85`;
`config.checkpoint_revision` binds its source, `allenai/MolmoAct2` revision
`e432d85f6e039edca44afb93c262f3084ab72a9c`. Pi0.5's
`config.pretrained_path` similarly records the absolute Kempner initialization
snapshot path and `config.pretrained_revision` is
`b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba` from `lerobot/pi05_base`. These
saved absolute paths are training provenance, not required hardware-host mount
points. Materialize each exact revision as a regular-file snapshot at any
absolute local path. The runtime verifies its complete external manifest,
preserves the saved immutable revision, and rebinds only the local path. Do not
edit the saved checkpoint config or processor JSON.

For MolmoAct2, use either a clean regular-file snapshot of the exact Hub revision
above or a byte-identical reviewed copy of the Kempner base artifact. Do not use
`main`, a cache symlink forest, or a later commit. Apply the same rule
symmetrically to Pi0.5. The hardware review must approve both base tree and
manifest digests; matching a repository name or commit string alone is not
sufficient.

The dependency roles differ. MolmoAct2 consumes its base snapshot for model and
packing-processor construction at inference. The released Pi0.5 checkpoint
contains complete policy weights; its 14.5 GB initialization snapshot is checked
and advertised for training provenance but is not read by `PI05Policy` during
inference. Pi0.5's live external inference dependency is instead the separately
manifested, revision-pinned Google PaliGemma tokenizer cache in `HF_HOME`. Keep
all three attestations: Pi model, Pi initialization provenance, and Pi tokenizer
cache. Do not describe the Pi initialization tree as a live weight dependency.

Generate a separate external manifest for each tree with
`examples/umi_yam/make_model_artifact_manifest.py`, review every entry, verify it
independently, then make both roots and manifests read-only. The server must
receive both complete three-argument attestations:

```text
--policy.artifact_root=/absolute/read-only/pretrained_model
--policy.artifact_manifest_path=/absolute/read-only/model_artifact_manifest.json
--policy.artifact_manifest_sha256=<exact manifest file SHA-256>
--policy.runtime_dependency_root=<local regular-file snapshot of exact saved revision>
--policy.runtime_dependency_manifest_path=/absolute/read-only/base_artifact_manifest.json
--policy.runtime_dependency_manifest_sha256=<exact base manifest file SHA-256>
```

Startup verifies both trees byte-for-byte, requires the saved dependency revision
to be immutable 40-hex, then portably rebinds the saved training-host path to the
verified local root. For MolmoAct2 it also overrides and verifies the sole saved
processor dependency against that same root. The client must exact-match both
advertised manifest/tree digest pairs, model
fingerprint, model revision, server revision, server-config digest, and service
name. Changing any checkpoint, base, config, or processor file requires new
manifests and review.

## Start the server

MolmoAct2 uses the generic server, pinned to the exact local artifact:

```bash
uv run python -m lerobot.scripts.lerobot_policy_server \
  --policy.pretrained_name_or_path=/absolute/read-only/molmo/pretrained_model \
  --policy.revision=REPLACE_EXACT_MODEL_REVISION \
  --policy.device=cuda \
  --policy.artifact_root=/absolute/read-only/molmo/pretrained_model \
  --policy.artifact_manifest_path=/absolute/read-only/molmo-artifact-manifest.json \
  --policy.artifact_manifest_sha256=REPLACE_EXACT_MANIFEST_SHA256 \
  --policy.runtime_dependency_root=/absolute/read-only/MolmoAct2-e432d85-regular-files \
  --policy.runtime_dependency_manifest_path=/absolute/read-only/molmo-base-artifact-manifest.json \
  --policy.runtime_dependency_manifest_sha256=REPLACE_EXACT_BASE_MANIFEST_SHA256 \
  --host=127.0.0.1 --port=8081 \
  --server_revision=REPLACE_IMMUTABLE_SERVER_REVISION \
  --command_ttl_ms=REPLACE_MEASURED_REVIEWED_TTL_MS
```

Pi0.5 must use `examples/umi_yam/pi05_onset_v3_policy_server.py`. Its explicit
device-local noise generator is part of the evaluated behavior. Choose one
reviewed seed, record it in both server attestation and trial record, and never
fall back to the generic implicit-global-RNG Pi path. `reset()` must restart the
same stream; warmup must not consume it. The Pi server additionally requires the
frozen code snapshot, code-manifest digest, tokenizer-cache manifest, checkpoint
model/config digests, and a fresh attestation output.

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVIDIA_TF32_OVERRIDE=0
export HF_HOME=/absolute/read-only/pi05-gate-hf-home
uv run python examples/umi_yam/pi05_onset_v3_policy_server.py \
  --policy.pretrained_name_or_path=/absolute/read-only/pi05/pretrained_model \
  --policy.revision=main \
  --policy.device=cuda \
  --policy.eval_seed=REPLACE_VALIDATED_SEED_0_TO_4 \
  --policy.num_inference_steps=10 \
  --policy.expected_model_safetensors_sha256=REPLACE_EXACT_MODEL_SHA256 \
  --policy.expected_config_json_sha256=REPLACE_EXACT_CONFIG_SHA256 \
  --policy.code_snapshot_root=/absolute/read-only/posttrain-code-snapshot \
  --policy.code_manifest_sha256=REPLACE_EXACT_CODE_MANIFEST_SHA256 \
  --policy.inference_hf_home="$HF_HOME" \
  --policy.inference_hf_manifest_sha256=REPLACE_EXACT_HF_MANIFEST_SHA256 \
  --policy.artifact_root=/absolute/read-only/pi05/pretrained_model \
  --policy.artifact_manifest_path=/absolute/read-only/pi05-artifact-manifest.json \
  --policy.artifact_manifest_sha256=REPLACE_EXACT_ARTIFACT_MANIFEST_SHA256 \
  --policy.runtime_dependency_root=/absolute/read-only/pi05-base-b211f3d-regular-files \
  --policy.runtime_dependency_manifest_path=/absolute/read-only/pi05-base-artifact-manifest.json \
  --policy.runtime_dependency_manifest_sha256=REPLACE_EXACT_BASE_MANIFEST_SHA256 \
  --policy.attestation_path=/absolute/fresh/path/pi05-server-attestation.json \
  --supervised_hardware_trial=true \
  --server_revision=REPLACE_IMMUTABLE_40_HEX_CODE_REVISION \
  --command_ttl_ms=1000 --session_idle_timeout_s=600 \
  --host=127.0.0.1 --port=8081
```

The service identity is exactly
`lerobot-pi05-onset-v3-explicit-noise-seed-<seed>`. The server-advertised model
manifest also binds the hardware-only backend mode, seed, code-manifest digest,
static attestation-identity digest, and both artifact trees into its fingerprint.
Copy those exact advertised values into the Pi deployment template after review.
A different seed is a different reviewed server contract. Hardware-only mode
rejects a mutable/development revision, any seed outside 0–4, either missing
artifact attestation, a TTL other than 1000 ms, or a changed idle timeout.

Run the server and client on the same trusted host or authenticated isolated
network. The examples intentionally default to loopback.

On the robot computer, copy one template to a new reviewed file and replace
every placeholder. Keep `confirm_hardware_control: false` for the first run:

```bash
cp examples/umi_yam/biyam_onset_v3_molmoact_hardware_trial.example.yaml \
  /absolute/review/path/molmo-onset-v3-no-motion.yaml
# Or copy the Pi template; never combine fields from both templates.

rg -n 'REPLACE_|/path/to/|0000000000000000000000000000000000000000000000000000000000000000' \
  /absolute/review/path/molmo-onset-v3-no-motion.yaml
# The command above must print nothing before review can continue.

uv run python -m lerobot.remote_inference.yam_umi_ee_bridge \
  --config_path=/absolute/review/path/molmo-onset-v3-no-motion.yaml
```

This invocation does not construct, connect, reset, arm, or command the robot
while `confirm_hardware_control` is false. Archive the logged preflight JSON. A
successful RPC/manifest/schema check is only `GO_SOFTWARE_NO_MOTION`; the same
configuration remains `NO_GO_PHYSICAL_MOTION` until commissioning evidence is
complete and independently reviewed. Do not flip the Boolean as a shortcut.

## Required physical commissioning record

Before `confirm_hardware_control=true`, create a read-only JSON evidence asset
using schema `umi-yam-onset-v3-hardware-commissioning-evidence-v1`. Copy
`examples/umi_yam/onset_v3_hardware_commissioning_evidence.example.json`; the
example is intentionally failing until every physical measurement is supplied.
It must bind:

- robot ID, distinct arm IDs, and both adapter serials;
- the exact 14-value interior reset target;
- the arm-specific calibrated UMI scalar produced by mapping each configured
  YAM reset command through the exact endpoint calibration, plus the frozen
  training supports: left `[0.4486217200756073, 1.0]`, right
  `[0.5472135543823242, 0.9483147859573364]`; the runtime recomputes each scalar
  and requires exact agreement within `1e-9` and inclusion in that support;
- exact path/size/SHA-256 for BiYAM, gripper, and rig/table calibrations;
- stable camera IDs and read-only evidence that each viewpoint matches training;
- repeated measured reset trials, tolerances, maximum errors, measurement ID,
  and the read-only reset evidence artifact.

Synthetic/test rig provenance, placeholder IDs, writable files, symlinks, zero
digests, and self-asserted URLs are rejected. Keep the evidence digest in the
reviewed deployment YAML.

## Trial ladder

Complete each stage, archive telemetry, and obtain a second-person review before
advancing.

1. **Static inspection:** E-stop works; hard stops/fixtures/cables clear; empty
   table; soft sacrificial objects only; operator holds enable/abort control.
2. **No-motion server handshake:** `confirm_hardware_control=false`. Verify model,
   server, artifact, action names/order, 24-row shape, TTL, camera identities,
   gate plan, and all calibration digests. No robot object should be constructed.
   This is a software handshake only. Until immutable physical commissioning is
   attached, its explicit decision is `NO_GO_MISSING_COMMISSIONING_EVIDENCE`;
   successful RPC/model checks must never be described as hardware-ready. If the
   calibrated reset scalar for either arm is outside its frozen support, the
   decision is `NO_GO_CALIBRATED_RESET_GRIPPER_OUTSIDE_TRAINING_SUPPORT` even
   though the handshake itself completed.
3. **Camera check:** compare live left/right 800x600 frames with training examples:
   side, roll, crop, exposure, color, bowl/table geometry, and occlusion. Abort on
   swap, resize, mirror, unstable auto-exposure, or changed viewpoint.
4. **Gripper check:** endpoint sweep with no object; verify monotonic direction,
   measured UMI scalars, and conservative jaw delta limits.
5. **Reset check:** command the reviewed interior anchor slowly with policy still
   disabled. Record repeated measured endpoint errors and rig/table clearances.
6. **One-query dry analysis:** allow inference while holding position. Inspect all
   24 raw/resolved targets and strict IK/collision checks; dispatch nothing.
7. **One-row motion:** set `max_steps: 1`; use low joint/jaw caps. One operator on
   E-stop and one watching telemetry. Stop and reset after the row.
8. **Three rows:** only after the one-row trace is reviewed. Do not jump directly
   to 15.
9. **One ACTIVE15 query:** rows 1–15 only, measured progress after every row,
   immediate stop after requery boundary. No unattended continuation.

Do not increase limits to make a target pass. Do not execute rows 16–24. Do not
continue after any stale observation, inference timeout, worker expiry,
partial/asymmetric arm apply, IK residual, joint bound, collision/clearance,
camera, calibration, or progress-hold error.

## Abort conditions

Abort immediately for unexpected initial motion, wrong arm/camera/gripper,
motion toward the table/other arm, grasp closure without an object, target jump,
more than one row advancing without measured convergence, stale observation,
expired TTL, missing heartbeat, worker asymmetry, any nonfinite value, IK
residual above tolerance, or operator uncertainty. Disarm both workers and use a
reviewed manual recovery; never auto-resume after a safety stop.

The first physical trace is evidence, not a success claim. Archive the exact
configuration, artifact manifests, server attestation, commissioning evidence,
camera snapshots, telemetry, and operator notes before deciding whether either
checkpoint merits a larger bounded trial.
