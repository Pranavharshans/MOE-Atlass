# Runtime execution boundary

`moeatlas.runtime` has two explicit seams. `InstanceSource` and
`CustomLoaderSource` execution remains model-runtime-free. `load_huggingface()`
and `load_local()` are lazy optional
loaders: they import `torch`/`transformers` only after a matching source,
immutable model and tokenizer resolution, policy, and local-path preflight
have passed. Importing `moeatlas.runtime` itself never imports those packages.

The loader is not model certification. It uses the exact resolved revisions,
does not use requested branch names, and keeps real checkpoint/network/cache,
GPU, quantization, and adapter validation deferred to MV-01/MV-02 and the
final VM.

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

For HF sources, both resolutions must be full Git commit evidence because the
Hub `revision=` argument cannot bind a content-digest pseudo-reference. Local
sources use the exact declared directory and always set
`local_files_only=True`; their immutable evidence may be an external content
attestation. Exposed HF commit hashes are checked against the resolved commits.

The loader forwards `torch_dtype` only for an explicit dtype policy, forwards a
copied non-empty device map, and requires Accelerate for an explicit map or
`device="auto"`. `preserve` still imports and validates torch but sends no
dtype kwarg. Observed dtype, architecture, device map, and warnings come from
the returned objects, never from the requested policy.

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

HF/local calls own every acquired config, tokenizer, and model object. Objects
without a callable `close()` still participate in the ownership lifecycle so
successful close clears references; failed callbacks remain retryable and only
failed callbacks are retried.

## Resolved load-and-scan bridge

`load_and_scan(plan)` is a one-shot convenience boundary for validated
`HuggingFaceSource` and `LocalSource` plans. It passes the identical plan to
`load_huggingface()` or `load_local()`, then passes the exact loaded model and
manifest to `discovery.scan()`. It never forwards the tokenizer, runs a
forward/generation, registers hooks, reads tensor values, writes a report, or
adds a second loading policy.

The bridge closes the loaded result on successful scan and on ordinary or
control-flow scan failures. A scan error remains the primary error when close
also fails; it receives only a deterministic safe note and a retryable
`PendingRuntimeCleanup`. A cleanup failure after a successful scan is raised
instead of publishing the report, with the same retry handle. The returned
report is validated as a `DiscoveryReport`, must retain the loaded manifest
value, and contains only `[STRUCTURE]` components. `InstanceSource` and
`CustomLoaderSource` are intentionally rejected here; callers must invoke
`load_instance()`/`load_custom()` and `discovery.scan()` manually.

## Custom loaders

`load_custom(plan)` is inert unless `execute_user_code=True` is passed. Only
the exact validated `module:function` reference from the plan is imported, and
the callable receives exactly one argument: the immutable `LoadingPlan`. It
must return exactly `RuntimeArtifacts`; tuples, mappings, and convenience
return shapes are rejected. Import and callable failures retain their original
exception as the cause. No loader registry or plugin framework is introduced.

The runtime tests use fake standard-library `torch`/`transformers` modules and
lightweight tokenizer/config stubs. No socket/network/cache access or model
artifact is used. Real checkpoint, native tensor, fused, quantized, and GPU
behavior remains deferred to MV-01/MV-04/MV-06/MV-07.
