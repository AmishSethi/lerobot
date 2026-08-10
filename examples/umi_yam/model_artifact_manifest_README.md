# Immutable model artifact manifest

Use this only on a dedicated release copy of a selected checkpoint. The model
server will reject a live/writable checkpoint, a partial inventory, or a
manifest stored inside the model tree.

```bash
MODEL_RELEASE=/absolute/path/to/dedicated-read-only-model-release
MODEL_MANIFEST=/absolute/path/outside-model-release/model-artifact-manifest.json

# Review the release copy first. It must contain every weight, config, and
# processor file needed by from_pretrained; do not point this at a live run.
find "$MODEL_RELEASE" -type l -print
find "$MODEL_RELEASE" -perm /222 -print

# Freeze the reviewed release copy, then inventory it. --output must not exist.
chmod -R a-w "$MODEL_RELEASE"
uv run --extra remote python examples/umi_yam/make_model_artifact_manifest.py \
  --artifact-root "$MODEL_RELEASE" \
  --output "$MODEL_MANIFEST"
```

The command prints the exact `manifest_sha256` and `tree_sha256`. Copy those
values without abbreviation into the policy-server arguments and the matching
hardware-client constants. Keep both the model release and external manifest
read-only. The generator rejects symlinks, non-regular entries, any writable
file or directory, an empty tree, an internal output path, and an existing
output path. Before reporting success it reruns the same backend verifier used
at server startup over every inventoried byte.
