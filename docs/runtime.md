# Runtime execution boundary

`moeatlas.runtime` is the first runtime integration slice. It validates and
records already-instantiated `InstanceSource` objects and explicitly opted-in
`CustomLoaderSource` callables. It does not import PyTorch, Transformers, or
safetensors, inspect caches, contact a model hub, or load HF/local checkpoints.
Those paths remain deferred to MV-01 and the final VM.

## Runtime artifacts and manifests

Trusted callers provide a `RuntimeArtifacts` dataclass containing the model,
tokenizer, actual JSON-compatible config (or a `to_dict()` object), observed
architecture, core `DType`, and device map. Runtime objects and cleanup
callbacks are never placed in a `LoadingPlan` ID, JSON, `ModelManifest`, or
provenance metadata. The model is only preflight-checked for a callable
`named_modules()` method; this feature does not iterate modules, run forwards,
or perform discovery.

Before returning a `LoadedModel`, the plan must contain immutable model and
tokenizer resolution evidence. The manifest uses the logical source model ID
with the resolved model revision, the separately resolved tokenizer revision,
and the observed dtype/device/architecture. It hashes a recursively validated,
finite config copy and carries deterministic plan security warnings into both
manifest warnings and provenance metadata.

## Ownership and cleanup

Instance resources are caller-owned by default: omit `cleanup`. An explicit
ownership transfer must provide both `cleanup` and `owns_cleanup=True`; the
pair is rejected in every other form. Owned cleanup is idempotent, retries
after failure, and releases strong model/tokenizer references after success.
When validation fails after an owned custom artifact is returned, cleanup is
attempted without replacing the original error. A failed attempt attaches a
`PendingRuntimeCleanup` handle to that error; `retry()` remains available for
transient or permanent cleanup failures.

`LoadedModel` is also a context manager. A cleanup failure on a body exception
is added as a note so the body exception remains the raised error. A caller can
retry the result's cleanup afterward.

## Custom loaders

`load_custom(plan)` is inert unless `execute_user_code=True` is passed. Only
the exact validated `module:function` reference from the plan is imported, and
the callable receives exactly one argument: the immutable `LoadingPlan`. It
must return exactly `RuntimeArtifacts`; tuples, mappings, and convenience
return shapes are rejected. Import and callable failures retain their original
exception as the cause. No loader registry or plugin framework is introduced.

The runtime tests use only the standard-library synthetic MoE fixture and
lightweight tokenizer/config stubs. Real checkpoint, native tensor, fused,
quantized, and GPU behavior remains deferred to MV-01/MV-04/MV-06/MV-07.
