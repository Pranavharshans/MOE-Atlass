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

## Deferred fidelity boundary

The synthetic fixture tests lifecycle behavior without a model runtime. Real
PyTorch hook signatures, output equivalence, device/dtype behavior, backward
fidelity, overhead, and fused/compiled paths remain deferred to MV-04 and the
final VM validation phase. A passing hook-fixture test is not model support.
