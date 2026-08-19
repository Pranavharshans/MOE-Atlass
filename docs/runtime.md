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

The CLI exposes this same boundary only through
`moeatlas scan --loading-plan PLAN.json`. The plan file is validated as one
strict `LoadingPlan` before the runtime bridge is imported or dispatched, and
the exact validated object is passed through. CLI publication reuses the
existing report serializer and atomic writer; it does not add a second loader
policy or a plan reconstruction path.

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

## Routing capture session

`RoutingCaptureSession` is the narrow bridge from one retained static
`AdapterInspection` and its canonical routing `ProbePlan` to validated
`RoutingEvent` objects. Construction re-dumps and revalidates both inputs,
rebuilds the family-neutral plan, and derives one immutable
`RoutingCaptureTarget` per router from the same-layer MoE layer and routed
experts. It rejects shared-expert or ambiguous component evidence before it
ever traverses `named_modules()`.

Use it as a normal context manager. The existing `HookManager` owns all hook
registration and reverse cleanup. The caller-supplied decoder is called once
per synchronous forward callback with the exact opaque `(module, inputs,
output)` values and must return an exact tuple of `RoutingEvent` objects. The
session performs no tensor reads, forward calls, tuple decoding, or value
inference; detaching tensors, reducing router payloads, and architecture
specific interpretation remain the caller/decoder's responsibility. Ordinary
decoder failures are wrapped in the fixed `decode` error with the original
exception as its cause; ordinary event validation failures use the fixed
`events` error. Decoder KeyboardInterrupt and SystemExit propagate unchanged;
registration and body/control-flow failures preserve the exact primary exception.
The session may retain the caller-owned model and decoder for its lifetime,
but does not retain synchronous callback payloads after invocation. A callback
held after the active body, during cleanup retry, or after publication is
inert and cannot invoke the decoder or mutate the sealed result.

Only fresh, identity-bound events are retained. `max_events` bounds retained
events, not model execution, callback invocations, or downstream storage. If
one invocation would exceed the remaining quota, the whole invocation is
dropped atomically; later invocations are skipped after the quota is full.
Events, truncation, and dropped counts are unpublished until the body exits
normally and every hook removal succeeds. Failed removals remain retryable via
`close()`; body errors and `KeyboardInterrupt`/`SystemExit` remain exact,
while ordinary decoder failures are fixed safe `decode` wrappers retaining
their cause. No failed run publishes staged events.

This is an opaque synchronous hook-payload boundary, not routing
certification. It does not claim native router equivalence, expert
specialization, or output fidelity. Hook behavior follows the official
[PyTorch forward-hook API](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook).
The reference-layout comparisons use tagged legacy [Mixtral v4.50.0
source](https://github.com/huggingface/transformers/blob/v4.50.0/src/transformers/models/mixtral/modeling_mixtral.py)
and [Qwen3-MoE v4.57.1
source](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py),
plus pinned current [Mixtral
source](https://github.com/huggingface/transformers/blob/64f30450dbfd1d02f610ad7080535cb906637fb9/src/transformers/models/mixtral/modeling_mixtral.py)
and [Qwen3-MoE
source](https://github.com/huggingface/transformers/blob/64f30450dbfd1d02f610ad7080535cb906637fb9/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py).
Routing payload equivalence, passive output fidelity, overhead, and GPU
behavior remain deferred to MV-03, MV-04, and MV-05.

## Mixtral routing decoder (experimental)

`MixtralRoutingDecoder(inspection, token_events)` is the explicit decoder for
one already validated Mixtral inspection. It is a narrow one-forward caller
boundary: each `RoutingCaptureTarget` is bound to the exact router key/path,
same-layer MoE component, contiguous ordered experts, expert count, and
routed top-k. A decoder instance accepts one successful invocation per router
path. The supplied `TokenEvent` tuple is the authoritative row order; the
decoder does not infer tokens, drop padding, call a tokenizer, run a
generation/runner, or retain hook tensors.

The inspection is freshly revalidated and must carry the exact
`huggingface-mixtral-static` descriptor (`1.0`, family `mixtral`). Router
capture provenance must agree on one layout: `legacy_indexed` or `packed`.
Legacy evidence is an exact `[tokens, experts]` router-logit tensor-like
value. Deterministic top-k identifies selected expert/rank pairs, but only the
observed selected `router_logit` is emitted; probability and weight are not
inferred. Selected/cutoff ties are rejected. Packed evidence is the exact
`(logits, scores, indices)` tuple with `[tokens, experts]`, `[tokens, top-k]`,
and `[tokens, top-k]` shapes. Native integer indices and finite weights are
checked against deterministic softmax/top-k renormalization before observed
selected logits and native weights are emitted.

Tensor-like conversion is deliberately fixed to
`detach() -> cpu() -> float() -> tolist()` for logits/scores and
`detach() -> cpu() -> tolist()` for integer indices. The decoder imports no
tensor runtime, NumPy, or optional dependency, and retains only fresh
`RoutingEvent` values. It is an experimental observation helper, not a
tokenizer, runner, generation engine, storage sink, UI, or routing
certification mechanism. Its capability boundary is `EXPERIMENTAL`.
It is not a tokenizer; it is not a runner, and it is not a generation API.
Model-dependent validation remains deferred to
MV-03, MV-04, MV-05, MV-06, MV-07, and MV-08; this feature does not certify a
checkpoint or change the ledger's deferred status.

## One-forward Mixtral execution (experimental)

`run_mixtral_routing_forward(model, inspection, plan, token_events,
model_kwargs, *, max_events)` is the single-forward prerequisite above the
passive capture seam. The caller supplies an already callable model, the
freshly validated exact Mixtral inspection and canonical routing `ProbePlan`,
an exact non-empty tuple of caller-tokenized `TokenEvent` rows, and an exact
dictionary of model keyword arguments. The tuple row order is authoritative;
the wrapper does not infer tokens, padding, prompts, or sequence boundaries,
and requires one common run and phase.

The wrapper computes the complete-event budget before registering hooks or
traversing the model: `len(token_events) * len(canonical_targets) * routed_top_k`.
An insufficient `max_events` is rejected before model execution. Keyword
arguments are shallow-copied before hooks are entered; values, including
tensor-like values, are not inspected or copied. The caller model is invoked
exactly once as `model(**copied_model_kwargs)`, then existing
`RoutingCaptureSession` cleanup semantics run for success, ordinary failure,
`KeyboardInterrupt`, `SystemExit`, and cleanup failure.

After any initial enter, body, or exit failure, the wrapper calls
`session.close()` exactly once internally. The invocation is terminal: it
re-raises the exact primary exception even when that internal retry succeeds.
If the retry also fails, no result is published; the existing
`PendingRuntimeCleanup` is attached under both `pending_cleanup` and
`pending_runtime_cleanup` so the caller can retry cleanup.

Only a complete, non-truncated, zero-dropped capture publishes a
`MixtralRoutingForwardResult`. The frozen, slots, identity-equality result
retains only the exact caller-owned output object and fresh token/routing
events; its output identity is preserved and hidden from `repr`. Every supplied token must be
represented, every route must reference a supplied token, links must be unique
by token/layer/rank, and all routes must be selected. No partial result is
published on model, decoder, event, budget, hook, or cleanup failure. The
wrapper is `EXPERIMENTAL` and is not a tokenizer, prompt builder, or generation runner.
It is not a dataset pipeline, storage sink, CLI, server, or UI. Model-dependent
equivalence, output fidelity, overhead, GPU, fused/quantized, and packaging
validation remain deferred to MV-03 through MV-08; ledger status is unchanged.
