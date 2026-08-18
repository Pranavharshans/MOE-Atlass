# Static semantic adapters

`moeatlas.adapters` defines the explicit boundary for an architecture-aware,
static semantic adapter. It is a protocol, not a registry: a caller creates or
selects an adapter and passes that one object to
`inspect_static_adapter(adapter, model, config, model_manifest)`.

## Contracts

`AdapterDescriptor` is a frozen, versioned JSON manifest containing only the
trimmed adapter name, version, sorted architecture families, and sorted
compatibility notes. It intentionally has no package, source, registry, or
entry-point fields. `AdapterDetection` contains a strict finite float score in
`[0, 1]`, deterministic evidence, and warnings. The score measures evidence
strength; it is neither a probability nor a certification claim. A positive
score requires evidence, while a zero score requires a warning.

`AdapterInspection` publishes the descriptor, positive detection, and one
validated `DiscoveryReport`. Its report remains static-only: every component
must have exactly `[STRUCTURE]` capability and capture provenance with
`source=STATIC_STRUCTURE`, the adapter name/version, and `verified=False`.
An empty report is allowed only when it carries a warning explaining the empty
result.

All three manifests reject unknown fields, round-trip through JSON, and are
frozen after construction. The inspection function re-dumps and revalidates
descriptor, detection, and report values, then constructs a fresh validated
inspection, so `model_copy` or subclass validation bypasses cannot publish
tampered data.

## Execution boundary

Inspection verifies only that the three protocol members exist statically and
then calls `descriptor`, `detect(model, config)`, and
`discover(model, model_manifest)` in that order. The exact model, config, and
manifest objects are passed through unchanged; arbitrary model objects are
accepted and never introspected. It does not import optional model libraries,
call `forward` or `generate`, install hooks, inspect tensor values, tokenize,
mutate, clean up, access the network, inspect caches, or write files. The model
remains caller-owned.

Ordinary adapter exceptions are wrapped in safe fixed-stage
`AdapterExecutionError` values with the original exception as `__cause__`;
their payload is not copied into the public message. Contract failures use
`AdapterContractError`. `KeyboardInterrupt` and `SystemExit` propagate
unchanged. Architecture-specific capture, runtime loading, capability
elevation, and certification remain deferred to the validation ledger.
