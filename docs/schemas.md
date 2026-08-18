# Canonical manifest schemas

Feature 2 introduces the first stable data contracts under
`moeatlas.core`. They are Pydantic v2 models with a JSON schema version of
`1.0`. They describe model identity, component structure, capture provenance,
and capability evidence; they do not load a model or inspect a device.

## Example

```python
from moeatlas.core import (
    DType,
    ModelManifest,
    TokenizerIdentity,
    make_config_hash,
    make_model_key,
)

manifest = ModelManifest(
    model_key=make_model_key("acme/demo-moe", "main"),
    architecture="demo_moe",
    revision="main",
    config_hash=make_config_hash({"experts": 4, "top_k": 2}),
    tokenizer=TokenizerIdentity(
        identifier="acme/demo-tokenizer",
        revision="main",
    ),
    dtype=DType.BFLOAT16,
    device_map={"": "cpu"},
)

payload = manifest.to_json(indent=2)
restored = ModelManifest.from_json(payload)
assert restored == manifest
```

`to_dict()` returns a JSON-compatible dictionary. `to_json()` and
`from_json()` preserve the explicit `schema_version` and reject malformed
payloads with Pydantic validation errors.

Model keys use exactly `model:<portable-id>@<revision>`. Hugging Face-style
slashes and revision refs such as `refs/main` are valid; `@` is reserved for
the separator. `ModelManifest` verifies that its `revision` exactly matches
the revision encoded in `model_key`.

## Capability labels

| Label | Meaning |
| --- | --- |
| `FULL` | Validated router scores/top-k, expert activity, and supported interventions are captured with semantic fidelity. |
| `ROUTING` | Reliable router/top-k capture; expert internals may remain packed or fused. |
| `MODULE` | Module-level inputs/outputs are visible, but semantic decoding is incomplete. |
| `STRUCTURE` | Static module/configuration/weight structure is available without validated inference capture. |
| `EXPERIMENTAL` | Capture works but is not certified against a native or golden reference. |
| `UNSUPPORTED` | The requested internal operation cannot currently be observed on the selected backend. |

`UNSUPPORTED` is exclusive and requires a warning. `FULL` requires capture
provenance with `verified=True`. `EXPERIMENTAL` requires capture provenance
with `verified=False`; it cannot be represented by a verified capture. Labels
are evidence tiers, not semantic expert names.

## Manifest invariants

- Every manifest carries `schema_version="1.0"` and a manifest type.
- Unknown fields are rejected; scalar fields use strict Pydantic types.
- `model_key`, component keys, revisions, tokenizer identifiers, and module
  paths are logical identifiers. Absolute paths, URI schemes, traversal
  segments, whitespace, and control characters are rejected.
- `make_model_key()` produces a readable `model:<id>@<revision>` key, and
  `parse_model_key()` validates and extracts both parts. `@` is rejected in
  either input part to avoid ambiguous keys.
- `make_component_key()` produces a SHA-256 digest over the semantic identity
  tuple, independent of Python's randomized hash seed or local filesystem.
  `ComponentManifest` recomputes that digest and rejects a mismatched key.
- `make_config_hash()` hashes canonical JSON with sorted keys and returns a
  `sha256:<hex>` token.
- Device maps, tensor shapes, warnings, and provenance metadata are JSON
  serializable. Tensor dimensions must be non-negative.
- Routed and shared status cannot both be true. Individual expert indices are
  only valid for expert or shared-expert components.
- Adapter provenance is all-or-nothing: `adapter` and `adapter_version` must
  either both be present or both be absent.

The `core` package intentionally contains no PyTorch, Transformers, or
safetensors imports. Model-dependent discovery and capture are later features.
