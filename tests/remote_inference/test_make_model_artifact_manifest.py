# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

from examples.umi_yam.make_model_artifact_manifest import make_model_artifact_manifest
from lerobot.remote_inference.backend import (
    CANONICAL_JSON_SPEC,
    MODEL_ARTIFACT_MANIFEST_SCHEMA_ID,
    verify_model_artifact_manifest,
)


def _freeze_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _artifact(tmp_path: Path) -> Path:
    root = tmp_path / "pretrained_model"
    (root / "processor").mkdir(parents=True)
    (root / "model.safetensors").write_bytes(b"weights")
    (root / "config.json").write_text('{"type":"molmoact2"}\n', encoding="utf-8")
    (root / "processor/policy_preprocessor.json").write_text('{"normalize":true}\n', encoding="utf-8")
    _freeze_tree(root)
    return root


def test_generator_matches_backend_canonical_contract_and_is_deterministic(tmp_path: Path) -> None:
    root = _artifact(tmp_path)
    first = make_model_artifact_manifest(
        artifact_root=root,
        output_path=tmp_path / "manifest-1.json",
    )
    second = make_model_artifact_manifest(
        artifact_root=root,
        output_path=tmp_path / "manifest-2.json",
    )

    assert first.manifest_sha256 == second.manifest_sha256
    assert first.tree_sha256 == second.tree_sha256
    assert first.file_count == 3
    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()
    assert stat.S_IMODE(first.manifest_path.stat().st_mode) == 0o444

    payload = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert payload == {
        "canonical_json_spec": CANONICAL_JSON_SPEC,
        "file_count": 3,
        "files_sha256": {
            "config.json": hashlib.sha256(b'{"type":"molmoact2"}\n').hexdigest(),
            "model.safetensors": hashlib.sha256(b"weights").hexdigest(),
            "processor/policy_preprocessor.json": hashlib.sha256(b'{"normalize":true}\n').hexdigest(),
        },
        "files_size_bytes": {
            "config.json": 21,
            "model.safetensors": 7,
            "processor/policy_preprocessor.json": 19,
        },
        "schema_id": MODEL_ARTIFACT_MANIFEST_SCHEMA_ID,
        "tree_sha256": first.tree_sha256,
    }
    expected_manifest_bytes = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    assert first.manifest_path.read_bytes() == expected_manifest_bytes
    assert first.manifest_sha256 == hashlib.sha256(expected_manifest_bytes).hexdigest()
    expected_tree_bytes = json.dumps(
        payload["files_sha256"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    assert first.tree_sha256 == hashlib.sha256(expected_tree_bytes).hexdigest()
    assert (
        verify_model_artifact_manifest(
            artifact_root=root,
            manifest_path=first.manifest_path,
            expected_manifest_sha256=first.manifest_sha256,
        )
        == first.tree_sha256
    )


@pytest.mark.parametrize("mutable_entry", ("root", "directory", "file"))
def test_generator_rejects_every_writable_artifact_path(tmp_path: Path, mutable_entry: str) -> None:
    root = _artifact(tmp_path)
    target = {
        "root": root,
        "directory": root / "processor",
        "file": root / "config.json",
    }[mutable_entry]
    target.chmod(target.stat().st_mode | stat.S_IWUSR)

    with pytest.raises(ValueError, match="must be read-only|contains a writable path"):
        make_model_artifact_manifest(
            artifact_root=root,
            output_path=tmp_path / "manifest.json",
        )


def test_generator_rejects_symlink_and_non_regular_entries(tmp_path: Path) -> None:
    root = _artifact(tmp_path)
    root.chmod(0o755)
    (root / "escape").symlink_to(tmp_path / "outside")
    root.chmod(0o555)

    with pytest.raises(ValueError, match="contains a symlink"):
        make_model_artifact_manifest(
            artifact_root=root,
            output_path=tmp_path / "symlink-manifest.json",
        )

    root.chmod(0o755)
    (root / "escape").unlink()
    fifo = root / "fifo"
    os.mkfifo(fifo, mode=0o444)
    root.chmod(0o555)
    with pytest.raises(ValueError, match="contains a non-regular entry"):
        make_model_artifact_manifest(
            artifact_root=root,
            output_path=tmp_path / "fifo-manifest.json",
        )


def test_generator_rejects_empty_tree_internal_output_and_overwrite(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir(mode=0o555)
    with pytest.raises(ValueError, match="at least one regular file"):
        make_model_artifact_manifest(
            artifact_root=empty,
            output_path=tmp_path / "empty-manifest.json",
        )

    root = _artifact(tmp_path)
    root.chmod(0o755)
    internal_parent = root / "unattested-output"
    internal_parent.mkdir(mode=0o555)
    root.chmod(0o555)
    with pytest.raises(ValueError, match="must be outside"):
        make_model_artifact_manifest(
            artifact_root=root,
            output_path=internal_parent / "manifest.json",
        )

    existing = tmp_path / "existing.json"
    existing.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="must not already exist"):
        make_model_artifact_manifest(artifact_root=root, output_path=existing)
    assert existing.read_text(encoding="utf-8") == "keep"
