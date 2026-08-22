# Normalized event contracts

`moeatlas.events` defines the model-runtime-independent event boundary from
PRD section 10. `TokenEvent`, `RoutingEvent`, and `ExpertEvent` are strict,
versioned Pydantic contracts. Each event carries an `event_type` discriminator
and `schema_version` (`"1.0"`), and can be serialized with `to_json()` or
validated back with the matching `from_json()`/`TypeAdapter(Event)` path.

## Token identity

`TokenEvent` accepts a portable `run_key`, `sequence_id`, non-negative
`token_pos`, non-negative `token_id`, presentation `token_text`, and
`phase` (`prefill` or `decode`). Its `token_key` is the full SHA-256 identity
`token:<64 lowercase hex characters>`, computed from the stable coordinates.
`token_text` is intentionally not part of identity, so presentation changes do
not orphan routing or expert evidence. `make_token_key()` and
`parse_token_key()` are the canonical helper boundary.

## Routing evidence

`RoutingEvent` links a token to distinct canonical `layer_key` and
`expert_key` component identities. `rank` is zero-based: rank `0` is the
highest-ranked route. `router_logit`, `probability`, and `weight` are optional
because capabilities may expose only a subset; at least one must be supplied.
No missing value is inferred. Probabilities are bounded to `[0, 1]`, and all
provided numeric values must be finite.

## Expert evidence and metadata

`ExpertEvent` uses optional non-negative `input_norm`, `output_norm`, and
`contribution_norm` fields. `latency_ms` is wall-clock latency in
milliseconds. At least one measurement or a non-empty metadata object is
required. Metadata is limited to finite JSON values; object keys must be
strings. Python lists and tuples normalize to immutable JSON arrays, and
nested objects/arrays are defensively frozen after validation. Metadata is
evidence, not identity: changing it cannot change a previously computed
`token_key`.

## Collection and link validation

`moeatlas.event_validation` owns model-runtime-independent validation across
normalized event collections. `fresh_token_events()` and
`fresh_routing_events()` produce exact fresh schema values, while
`validate_routing_links()` checks selected routes, supplied-token references,
unique token/layer/rank links, complete token representation, and deterministic
layer/token/rank order. Runtime and storage consume this shared seam instead of
owning duplicate or family-specific link rules. It imports no runtime, storage,
DuckDB, PyTorch, Transformers, or other model package.

These contracts do not capture tensors, install hooks, write Parquet/DuckDB,
run inference, derive model values, or provide a storage engine. Those runtime
and persistence responsibilities remain later feature boundaries and the real
PyTorch/GPU validation remains deferred in the [validation ledger](model-validation-ledger.md).

## Expert event runtime capture (R3.1)

The `ExpertEvent` contract has a runtime producer: the structure-driven expert
capture in `moeatlas.runtime.generic_capture`. `structured_expert_targets()`
resolves one generic hook target per routed-expert component from a static
`[STRUCTURE]` report — the routing universe drives every count, shared experts
are excluded by kind, and no adapter name or module-path convention appears
anywhere. `run_structured_expert_forward()` attaches router decode hooks plus
passive `EXPERT_ACTIVITY` forward hooks through the existing `HookManager`
plumbing, runs exactly one caller-owned forward, and reduces each expert
invocation to L2 norms via duck-typed tensor access (`detach/cpu/float/tolist`);
no model stack is imported.

Each fired invocation records `input_norm`, `output_norm`, and — when input
and output shapes match — `contribution_norm`; width-changing FFN blocks
honestly leave `contribution_norm` null. One `ExpertEvent` is published per
selected (token, expert) pair referencing that invocation's norms, with the
invoked batch size recorded under metadata (`latency_ms` stays null because a
passive post-forward hook cannot observe per-invocation wall time). Capture is
all-or-nothing: every selected expert must fire exactly once, routers must
fire once, and the projected event count must fit the strict
`max_expert_events` budget, otherwise nothing is published. Events land on
the result value as `expert_events` alongside routing evidence and are stored
through `append_structured_shard` (see [storage](storage.md)).
