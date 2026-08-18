# Probe plans and passive hooks

`moeatlas.probe` separates serializable probe intent from runtime execution.
`ProbePlan` is a strict, versioned JSON contract; `resolve_probe_plan()` binds
its explicit target paths to a deterministic `named_modules()` surface; and
`HookManager` temporarily registers caller-supplied callbacks.

The package does not import PyTorch, store tensor values, define an event
schema, or implement interventions. A callback receives the duck-typed
runtime arguments unchanged and remains responsible for any event sink or
reduction behavior. Its return value is always discarded and the registered
wrapper returns `None`, so a passive probe cannot replace inputs, outputs, or
gradients.

## Levels and hook points

The plan levels match the PRD:

| Level | Name | Passive contract |
| ---: | --- | --- |
| 0 | `STRUCTURE` | static target resolution; no hooks |
| 1 | `ROUTING` | lightweight forward/pre-forward observation |
| 2 | `EXPERT_ACTIVITY` | forward observation of expert activity |
| 3 | `FULL_ACTIVATIONS` | explicitly opted-in, budgeted full values |
| 4 | `GRADIENTS` | explicitly opted-in, budgeted gradients and `full_backward` |
| 5 | `INTERVENTION` | declared and opt-in in the schema; not executable by this passive manager |

`HookPoint` values are `forward_pre`, `forward`, and `full_backward`. The
plan validator rejects hook points that are incompatible with the selected
level, requires `full_backward` for `GRADIENTS`, and requires a forward hook
for `FULL_ACTIVATIONS`. `FULL_ACTIVATIONS` also requires true raw capture:
`CaptureMode.RAW` with `ReductionPolicy.NONE`, an explicit opt-in, and a
positive budget. `GRADIENTS` may use reduced or raw capture, but still requires
an explicit opt-in and budget.

## Targets and identity

Each module path may occur at most once in a plan because a `HookBinding` is
identified by `(module_path, hook_point)`. Optional component metadata must be
provided as a pair: `component_kind` requires `component_key`, and the key
must be the canonical `component:<64 lowercase hex>` identity emitted by
`make_component_key()`.

## Bounded capture policy

The default `CapturePolicy` is statistics-oriented: it has no raw opt-in and
no unbounded tensor budget. Reduced or statistics plans can set sampling and
reduction intent without making storage guarantees. `RAW`/`NONE` capture
requires both `raw_opt_in=True` and a positive `max_items` or `max_bytes`
budget. `FULL_ACTIVATIONS` additionally requires that raw capture pair;
`GRADIENTS` may remain reduced. `CaptureMode.RAW` and `ReductionPolicy.NONE`
are a pair and must be selected together. Sampling below
`1.0` requires an explicit `sample_seed` for reproducibility. Gradient capture
additionally requires `include_gradients=True`. These fields are policy
declarations only; this feature does not perform reduction or storage.

## Resolution and lifecycle

Resolution rejects missing or duplicate module paths, invalid target paths,
missing selectors, empty include/exclude results, missing hook registration
methods, and incompatible plan surfaces before registration begins. Include
paths select explicit targets, while exclude paths remove them; overlapping
selectors are rejected as ambiguous. A `ResolvedProbePlan` must contain the
exact source-plan target selection, including component metadata. When a
resolved plan is passed to `HookManager`, it is re-resolved against the
provided model and module object identity must match before any hook is
registered.

`HookManager` is a single-use context manager. It preflights callback coverage,
registers bindings in deterministic target/hook order, and removes every
installed handle in reverse order on normal exit, callback/body exception,
`BaseException`, and partial registration failure. Cleanup continues after a
removal failure; successful handles are forgotten while failed handles remain
tracked and can be retried with `close()`. Normal-exit failures are raised as
`HookCleanupError`; when a body or callback already failed, cleanup failures
are attached as a note so the original error remains primary. Cleanup is
idempotent after all handles succeed, and a manager cannot be re-entered.

## Static adapter bridge

`moeatlas.adapters.build_routing_probe_plan()` accepts one already validated
`AdapterInspection` and emits an inert, family-neutral `ROUTING` plan. It
freshly reconstructs the inspection from its JSON form before semantic reads,
selects every and only router component, and rejects empty or duplicate router
module paths. The plan targets retain each router's exact module path,
canonical component key, and `ROUTER` kind. It uses only the `forward` hook,
empty selectors, and reduced `TOP_K` capture with outputs enabled and no raw,
gradient, input, sampling, or intervention opt-in.

The inspection must be retained as the evidence and provenance source; the
plan is intent only. Both `max_items` and `max_bytes` are `None`, so reduced
`TOP_K` imposes no execution, event, or storage bound. Compilation does not call an adapter or model, resolve
modules, install hooks, decode values, create token/routing events, write
storage, or certify routing. Hook fidelity is deferred to the official
[PyTorch forward-hook API](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook)
and VM validation. For both families, the comparison references are the
tagged legacy [Mixtral v4.50.0 source](https://github.com/huggingface/transformers/blob/v4.50.0/src/transformers/models/mixtral/modeling_mixtral.py)
and [Qwen3-MoE v4.57.1 source](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py),
plus the pinned current
[Mixtral source at `64f30450dbfd1d02f610ad7080535cb906637fb9`](https://github.com/huggingface/transformers/blob/64f30450dbfd1d02f610ad7080535cb906637fb9/src/transformers/models/mixtral/modeling_mixtral.py)
and [Qwen3-MoE source at the same pinned commit](https://github.com/huggingface/transformers/blob/64f30450dbfd1d02f610ad7080535cb906637fb9/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py).
These are source-layout comparisons only, not a broad compatibility claim.
Tagged legacy and pinned current implementations can expose different router
forward-payload conventions, so no tensor/tuple decoder is assumed. Native
routing equivalence and capture remain deferred to MV-03.

## Runtime routing capture

`RoutingCaptureSession` consumes a retained `AdapterInspection` and the
canonical family-neutral routing plan produced from it. Preflight derives
one `RoutingCaptureTarget` for each router, binding its exact `ProbeTarget`,
same-layer MoE component, routed expert keys, and positive top-k fact. No
shared expert, allowlist, or guessed component identity is accepted.

The session delegates module resolution and registration to `HookManager`.
The decoder receives exactly the synchronous opaque `(module, inputs, output)`
hook arguments and returns an exact tuple of freshly validated `RoutingEvent`
values. Ordinary decoder failures are chained to a fixed `decode` error and
ordinary event validation failures to a fixed `events` error. The session does
not read tensor values, decode tuples, call forwards, or retain callback
payloads; detaching and reducing them is the caller's responsibility. The
caller-owned model and decoder may remain available for the session lifetime.
Decoder KeyboardInterrupt/SystemExit, body, and control-flow failures remain
the exact primary exception. A callback held outside the active body is inert,
including while cleanup is awaiting a retry and after publication.
`max_events` bounds retained events only. An over-quota
invocation is discarded atomically, and invocations after the quota is full
are skipped. Publication waits for normal body completion and successful
reverse cleanup, with `close()` retrying failed removals.

This boundary is observational intent and event validation, not native
routing certification. It follows the official
[PyTorch forward-hook API](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook)
and compares only the tagged legacy [Mixtral v4.50.0
source](https://github.com/huggingface/transformers/blob/v4.50.0/src/transformers/models/mixtral/modeling_mixtral.py)
and [Qwen3-MoE v4.57.1
source](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py)
with pinned current [Mixtral
source](https://github.com/huggingface/transformers/blob/64f30450dbfd1d02f610ad7080535cb906637fb9/src/transformers/models/mixtral/modeling_mixtral.py)
and [Qwen3-MoE
source](https://github.com/huggingface/transformers/blob/64f30450dbfd1d02f610ad7080535cb906637fb9/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py).
Routing equivalence, output fidelity, and overhead remain deferred to
MV-03/MV-04/MV-05.

## Deferred fidelity boundary

The synthetic fixture tests lifecycle behavior without a model runtime. Real
PyTorch hook signatures, output equivalence, device/dtype behavior, backward
fidelity, overhead, and fused/compiled paths remain deferred to MV-04 and the
final VM validation phase. A passing hook-fixture test is not model support.
