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
`RoutingForwardResult` (with the historical `MixtralRoutingForwardResult`
identity alias). The frozen, slots, identity-equality result retains only the
exact caller-owned output object and fresh token/routing events; its output
identity is preserved and hidden from `repr`. Fresh collection copies and
cross-event link checks are owned by the runtime-independent
`moeatlas.event_validation` seam and shared unchanged with storage. The
historical private runtime validator names remain identity aliases for
compatibility. Every supplied token must be represented, every route must
reference a supplied token, links must be unique by token/layer/rank, and all
routes must be selected. No partial result is
published on model, decoder, event, budget, hook, or cleanup failure. The
wrapper is `EXPERIMENTAL` and is not a tokenizer, prompt builder, or generation runner.
It is not a dataset pipeline, storage sink, CLI, server, or UI. Model-dependent
equivalence, output fidelity, overhead, GPU, fused/quantized, and packaging
validation remain deferred to MV-03 through MV-08; ledger status is unchanged.

### Feature 24: bounded prompt prefill (EXPERIMENTAL)

`run_mixtral_prompt_prefill(loaded, inspection, plan, prompt, *, run_key,
sequence_id, add_special_tokens, max_prompt_chars, max_tokens, max_events)` performs one plain-text tokenizer call and
delegates one Feature 18 forward. Strict manifests, plans, identifiers, and
positive budgets are revalidated before hooks, model execution, or tensor
materialization. Tokenizer failures use `tokenize`; encoding failures use
`encoding`. The complete event budget is checked before detach/cpu/tolist.
`LoadedModel` is borrowed: this seam never closes, enters, mutates, evaluates,
generates, or otherwise owns the model/tokenizer. Real tokenizer/checkpoint and
GPU validation remain deferred to the final VM.

Example composition (given an already-open `loaded` handle and its validated
`inspection` and canonical `plan`; every later stage is a separate caller
action):

```python
from pathlib import Path

from moeatlas.analysis import (
    aggregate_routing_load,
    render_routing_load_heatmap,
)
from moeatlas.runtime import run_mixtral_prompt_prefill
from moeatlas.store import append_routing_shard, list_routing_runs

workspace = Path("./moeatlas-workspace")  # This directory must already exist.
run_key = "run-1"
max_matrix_cells = 4096
expected_events = (
    256 * len(plan.targets) * inspection.report.facts.routed_top_k
)
result = run_mixtral_prompt_prefill(
    loaded, inspection, plan, "A short prompt", run_key=run_key,
    sequence_id="sequence-1", add_special_tokens=False,
    max_prompt_chars=4096, max_tokens=256, max_events=expected_events,
)
append_routing_shard(workspace, result)  # Token text is redacted by default.
inventory = list_routing_runs(
    workspace,
    max_runs=32,
    max_shards=1024,
    max_event_rows=1_000_000,
    max_source_bytes=1_000_000_000,
)
assert inventory.runs[0].run_key == run_key
matrix = aggregate_routing_load(
    workspace,
    inspection,
    run_key=run_key,
    max_routing_rows=1_000_000,
    max_source_bytes=1_000_000_000,
    max_matrix_cells=max_matrix_cells,
)
html = render_routing_load_heatmap(
    matrix, metric="load_ratios", max_cells=max_matrix_cells
)
```

The historical Mixtral storage and analysis names are identity aliases of
these neutral functions and preserve the same signatures, matrix values, and
HTML bytes.

The tokenizer call is exactly `tokenizer(prompt, add_special_tokens=..., padding=False,
truncation=False, return_attention_mask=True, return_token_type_ids=False,
return_tensors="pt")`; the returned mapping must contain only `input_ids` and
`attention_mask`, each shaped `(1, N)`. Callers may then append the resulting
events to the separate Feature 19 shard workflow or inventory/heatmap workflow;
prefill itself never stores or publishes them.

The prompt, encoding mapping, tensor-like values, and raw hook payloads are
transient. Each exact string returned by `convert_ids_to_tokens` is deliberately
retained as `TokenEvent.token_text`, and every emitted token has phase `PREFILL`.
Feature 19 redacts those pieces only when the caller uses its default
`store_token_text=False`; opting in stores them. The result otherwise contains
only fresh token and routing events plus the caller-owned model output. No path,
model/tokenizer object, cache, network request, or raw tensor is written or
retained by this seam. A tokenizer/converter ordinary exception is
reported with the fixed `tokenize` or `encoding` stage while
`KeyboardInterrupt`/`SystemExit` and a Feature 18 body exception remain the
exact primary control-flow exception; cleanup follows Feature 18's retryable
pending-handle contract. The seam has no progress stream, live subscription,
server endpoint, wire/view-model schema, generation loop, or UI state.
## Qwen3.5 routing decoder (model-free)

`Qwen3_5RoutingDecoder` is family-isolated to the official Qwen3.5 static
descriptor and packed `(router_logits, router_scores, router_indices)` capture.
It validates full-logit stable-softmax/top-k semantics, excludes shared
experts, freshens `RoutingEvent` values, rejects ties and malformed payloads,
and allows one successful invocation per router. It performs no model loading
or tensor-runtime imports. Runtime/checkpoint equivalence, GPU validation, and
release-time revision review remain deferred to the final VM.
`RoutingCaptureSession` validates any shared-expert metadata present in a
static report but excludes those components from `RoutingCaptureTarget`
`expert_keys`; this is a shared model-neutral rule, not a Qwen-specific branch.
## Qwen3.5 routing forward (experimental)

`run_qwen3_5_routing_forward` is the one-forward Qwen3.5 composition boundary.
Call it with caller-owned model, inspection, canonical plan, token rows, and
an exact kwargs dictionary, for example:

```python
result = run_qwen3_5_routing_forward(
    model, inspection, plan, token_events, {"input_ids": input_ids}, max_events=budget
)
```

The wrapper fresh-validates the Qwen token sequence, plan JSON and `plan_id`,
strict top-k, kwargs, and the complete event budget before hook registration or
model traversal. The budget is `len(token_events) * len(plan.targets) *
routed_top_k`; the caller model is invoked exactly once. It registers
`Qwen3_5RoutingDecoder` through `RoutingCaptureSession` and publishes only
complete canonical layer blocks. The result retains only fresh token/routing
events and the exact caller output identity; it does not own or retain the
model, decoder, kwargs, or router payloads.

Cleanup and pending-retry semantics match Feature 18: after a started session,
one internal `session.close()` retry preserves the exact primary body,
decoder, registration, keyboard-interrupt, or system-exit exception. A
persistent callback failure is exposed as the caller-owned
`pending_runtime_cleanup`/`pending_cleanup` handle. The neutral
`RoutingForwardResult` is identity-compatible with the historical Mixtral
result alias; storage consumes the same event schema without migration. Feature
27 downstream composes through the model-neutral append, reopen, run inventory,
aggregate, and visualization boundaries. Feature 28 accepts Qwen3.5 and
future-family inspections through the complete structural routing-universe
contract; checkpoint/GPU certification remains deferred to the final VM.

## Structure-driven generic capture (model-family-agnostic)

`run_structured_routing_forward(model, report, token_events, model_kwargs, *,
max_events, config=None)` in `moeatlas.runtime.generic_capture` turns the
caller-owned hook pattern into a product seam for foreign families. It
discovers router modules purely from a static `[STRUCTURE]` `DiscoveryReport`
by binding to the structure the scan proved: trusted router candidates are
those whose parent block publishes an `EXPERT_CONTAINER` component, each binds
the `MOE_LAYER` identity published at its parent-block path, and the report's
strict `expert_count`/`routed_top_k` facts drive every count. This keeps noisy
name-token candidates on foreign families (SwiGLU `gate_proj` modules,
`...Moe...` class names) out of the hook set; strict name guards (exact
`gate` final path segment plus expert evidence, whole-word `moe` path markers)
apply only when a report publishes no such structure. Hooks attach passively
through the existing `HookManager`, and each router payload decodes
generically against the discovered expert universe — no adapter name,
module-path convention, or certified descriptor anywhere.
`build_universal_inspection` resolves the same report universe, so generic
captures and universal documents agree by construction.

Two payload forms decode today: packed `(logits, scores, indices)` tuples use
their native scores as probabilities, and flat `[tokens, experts]` logit
matrices reduce through deterministic tie-rejecting top-k. Score
normalization follows the model config where determinable (`score_function`
of softmax or sigmoid); otherwise raw logits are recorded and a capability
note travels with the result. The complete-event budget
`len(token_events) * len(targets) * routed_top_k` is checked before hooks
exist; every router must fire exactly once and every token must receive every
layer's full rank schedule, or nothing publishes. Tensor-like conversion stays
fixed to `detach() -> cpu() -> float() -> tolist()` for scores and
`detach() -> cpu() -> tolist()` for integer indices. The module imports no
model stack; real checkpoint and payload equivalence remain deferred to the
final VM phase.

## Routing decode capabilities (model-free)

`moeatlas.runtime.capabilities` is the model-neutral seam between the shared
runtime and family-specific router-output decoding. An adapter declares a
`RouterPayloadShape` (`tuple_logits`, `tuple_scores_indices`,
`dict_arrays`, `assignment_indices`) and `ScoreSemantics`
(`logits`, `probabilities`, `none`) and implements
`RoutingDecodeCapability.decode(payload, *, universe, token_key)` producing
canonical `RoutingEvent` rows. The historical Mixtral and Qwen3.5 shapes are
ordinary vocabulary values; unknown families declare their own without a
central branch.

`validate_decoded_routing()` enforces the shared postconditions on any
decoded rows: exact token identity, layers inside the published
`RoutingUniverse`, complete per-layer rank schedules `0..top_k-1` honoring
variable top-k, unique experts drawn from that layer's universe, and score
columns agreeing with the declared semantics. Assignment-only decode makes
no logit or probability claims and pins `weight` to exactly `1.0` as the
"selected, unweighted" marker required by the event contract.
`native_id_map()` resolves sparse or unordered native expert identifiers
through the universe's parallel `expert_indices`. Failures use fixed-stage
`RoutingDecodeError` values (`dependency`, `decode`, `postcondition`). The
module imports no model stack and downloads nothing; real-payload
equivalence for unknown families remains deferred to the final VM phase.

## Universal forward execution (model-neutral)

`run_routing_forward()` is the one model-neutral execution boundary shared by
every MoE family. It composes the canonical routing plan compiler, the
inspection's adapter-published `RoutingUniverse`, a caller-supplied decoder
factory, and the passive capture session without ever inspecting an adapter
name, module-path convention, or payload type identity. The decoder factory
receives the freshly revalidated inspection and token rows and must return a
conforming `RoutingHookDecoder`: an object declaring its `payload_shape` and
`score_semantics` capabilities and decoding one router hook invocation into
canonical `RoutingEvent` rows. Family knowledge lives entirely inside that
decoder; the seam itself stays family-neutral with no central branching, so
any MoE model — not only the historical families — executes through it.

The runner applies three family-blind gates around the decoder: the
published universe must pass the explicit rectangular projection (hook
capture binds one uniform capture context per router today; non-rectangular
families fail here with the named gate instead of being silently reshaped),
every captured row is verified against the decoder-declared score semantics,
and `validate_observed_routing()` proves the complete-capture postconditions
— every captured token carries every universe layer's full rank schedule
with unique universe-bound experts. `TokenSequencePolicy` selects how
strictly token identities are validated before execution: `shared_run`
(unique keys sharing one run and phase) or `canonical_sequence` (one shared
sequence in contiguous canonical position order, the shape batched
whole-sequence payloads depend on).

`run_mixtral_routing_forward()` and `run_qwen3_5_routing_forward()` are now
thin compatibility wrappers over this seam composing their historical
decoders (`tuple_logits`/`tuple_scores_indices`, logits semantics); their
signatures, error behavior, and results are unchanged. Any other family runs
through the same seam by supplying its own declared decoder — the unknown
`blocksparse_moe` fixture covers dict-array payloads with unordered native
identifiers end-to-end. Input preparation remains caller-owned at this
boundary; a provider/task input-preparation capability protocol arrives with
the run engine. Real checkpoint execution for any family stays deferred to
the final VM phase.
