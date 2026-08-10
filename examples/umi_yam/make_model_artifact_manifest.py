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

"""Create a complete, external manifest for an immutable model artifact tree.

This utility deliberately refuses live/writable checkpoint directories. Copy a
selected checkpoint into a dedicated release directory, make the entire tree
read-only, and only then run this command. The generated manifest is also made
read-only and is verified with the deployment backend before success is
reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from lerobot.remote_inference.backend import (
    CANONICAL_JSON_SPEC,
    MODEL_ARTIFACT_MANIFEST_SCHEMA_ID,
    verify_model_artifact_manifest,
)


@dataclass(frozen=True)
class ModelArtifactManifestResult:
    manifest_path: Path
    manifest_sha256: str
    tree_sha256: str
    file_count: int


def _canonical_json_bytes(payload: object) -> bytes:
    """Serialize with the exact canonical JSON contract used by the backend."""

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_immutable_artifact_root(artifact_root: str | Path) -> Path:
    root_input = Path(artifact_root).expanduser()
    if root_input.is_symlink() or not root_input.is_dir():
        raise ValueError("model artifact root must be a real directory, not a symlink")
    root = root_input.resolve(strict=True)
    if stat.S_IMODE(root.stat().st_mode) & 0o222:
        raise ValueError("model artifact root must be read-only")
    return root


def _resolve_new_external_output(output_path: str | Path, *, artifact_root: Path) -> Path:
    output_input = Path(output_path).expanduser()
    if output_input.is_symlink() or output_input.exists():
        raise ValueError("model artifact manifest output must not already exist")
    if not output_input.name:
        raise ValueError("model artifact manifest output must name a file")

    parent_input = output_input.parent
    if parent_input.is_symlink() or not parent_input.is_dir():
        raise ValueError("model artifact manifest output parent must be a real existing directory")
    output = parent_input.resolve(strict=True) / output_input.name
    if output.is_relative_to(artifact_root):
        raise ValueError("model artifact manifest must be outside the inventoried artifact root")
    return output


def _inventory_read_only_tree(root: Path) -> tuple[dict[str, str], dict[str, int]]:
    files_sha256: dict[str, str] = {}
    files_size_bytes: dict[str, int] = {}

    paths = sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix())
    for path in paths:
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"model artifact contains a symlink: {relative}")

        before = path.stat(follow_symlinks=False)
        if stat.S_IMODE(before.st_mode) & 0o222:
            raise ValueError(f"model artifact contains a writable path: {relative}")
        if stat.S_ISDIR(before.st_mode):
            continue
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"model artifact contains a non-regular entry: {relative}")

        digest = _sha256_file(path)
        after = path.stat(follow_symlinks=False)
        stable_fields_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        stable_fields_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if stable_fields_after != stable_fields_before:
            raise ValueError(f"model artifact changed while it was inventoried: {relative}")

        files_sha256[relative] = digest
        files_size_bytes[relative] = before.st_size

    if not files_sha256:
        raise ValueError("model artifact must contain at least one regular file")
    return files_sha256, files_size_bytes


def make_model_artifact_manifest(
    *,
    artifact_root: str | Path,
    output_path: str | Path,
) -> ModelArtifactManifestResult:
    """Inventory an immutable tree and create a backend-compatible manifest.

    ``output_path`` is created exclusively: an existing file is never
    overwritten. The complete artifact is read a second time by the runtime
    verifier before this function returns.
    """

    root = _require_immutable_artifact_root(artifact_root)
    output = _resolve_new_external_output(output_path, artifact_root=root)
    files_sha256, files_size_bytes = _inventory_read_only_tree(root)
    tree_sha256 = hashlib.sha256(_canonical_json_bytes(files_sha256)).hexdigest()
    payload = {
        "schema_id": MODEL_ARTIFACT_MANIFEST_SCHEMA_ID,
        "canonical_json_spec": CANONICAL_JSON_SPEC,
        "file_count": len(files_sha256),
        "files_sha256": files_sha256,
        "files_size_bytes": files_size_bytes,
        "tree_sha256": tree_sha256,
    }
    manifest_bytes = _canonical_json_bytes(payload)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(output, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(manifest_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        os.fchmod(descriptor, 0o444)
    finally:
        os.close(descriptor)

    verified_tree_sha256 = verify_model_artifact_manifest(
        artifact_root=root,
        manifest_path=output,
        expected_manifest_sha256=manifest_sha256,
    )
    if verified_tree_sha256 != tree_sha256:
        raise RuntimeError("backend returned a different model artifact tree SHA-256")

    return ModelArtifactManifestResult(
        manifest_path=output,
        manifest_sha256=manifest_sha256,
        tree_sha256=tree_sha256,
        file_count=len(files_sha256),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        required=True,
        help="Read-only directory containing every model/config/processor artifact.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New manifest JSON path outside --artifact-root; existing paths are rejected.",
    )
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    try:
        result = make_model_artifact_manifest(
            artifact_root=args.artifact_root,
            output_path=args.output,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(
        _canonical_json_bytes(
            {
                "file_count": result.file_count,
                "manifest_path": str(result.manifest_path),
                "manifest_sha256": result.manifest_sha256,
                "tree_sha256": result.tree_sha256,
            }
        ).decode("utf-8")
    )


if __name__ == "__main__":
    main()
