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

## Mixtral static adapter

`MixtralStaticAdapter()` is the first concrete caller-selected adapter. It is
stateless and is not registered or auto-selected. Its descriptor is
`huggingface-mixtral-static` version `1.0`, for the exact `mixtral` family.
The adapter accepts either an exact `model_type="mixtral"` or one of the
explicit Mixtral architecture names; Qwen and other look-alikes are not
treated as Mixtral.

The model-free implementation recognizes two strict module surfaces under one
common prefix: the official Transformers 4.50-style indexed layout
`layers.N.block_sparse_moe.{gate,experts}` with indexed `experts.N`, including
the registered `w1`, `w2`, `w3`, and `act_fn` children (report layout
`legacy_indexed`), and the current packed
layout `layers.N.mlp.{gate,experts}` with its registered `act_fn` child.
Packed expert parameters are the direct `gate_up_proj` and `down_proj`
parameters (without a `.weight` suffix); their logical per-expert shapes are
reported without inventing a parameter name. Configuration fields, strict
structural attributes, contiguous layer/expert indices, router dimensions, and
expert dimensions must agree exactly. Detection evidence is weighted across family identity,
architecture, strict configuration, topology, and semantic parameter shapes;
it is evidence strength, not a probability or compatibility certificate.

Discovery emits only `[STRUCTURE]` components with
`STATIC_STRUCTURE`, method `mixtral-static-structure-v1`, adapter version
`1.0`, and `verified=False`. Packed expert entries describe logical slices and
carry a warning that those slices are not independently hookable. No routing
scores, expert activity, specialization, or routing certification is claimed.
The caller must choose this adapter explicitly and supply the already-observed
model/config/manifest. Real Transformers checkpoints, fused or quantized
variants, and VM/GPU validation remain deferred in the model-validation
ledger.

## Qwen3-MoE static adapter

`Qwen3MoeStaticAdapter()` is an explicit, stateless caller choice for the
exact `qwen3_moe` family. It requires `model.config` to be the same object
as the supplied configuration and accepts only the explicit Qwen3-MoE
architecture allowlist. Class names alone, Qwen2, Qwen3.5, fuzzy markers,
and conflicting family markers produce zero detection.

The adapter validates the complete configured dense/sparse schedule:
layer `i` is sparse exactly when `i` is not in `mlp_only_layers` and
`(i + 1) % decoder_sparse_step == 0`; at least one sparse layer is required.
Dense layers must expose `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj`,
and `mlp.act_fn` with the configured dense shapes, but dense layers are not
published as MoE components.

Sparse layers use one common prefix before `layers.N` and exactly one of:

- the `legacy_indexed` reference surface from Transformers 4.51.3 and
  4.57.1: `mlp.gate`, `mlp.experts.N`, and each expert's
  `gate_proj`, `up_proj`, `down_proj`, and `act_fn` children;
- the packed reference surface from Transformers 5.0.0/current pinned
  sources: `mlp.gate`, `mlp.experts.act_fn`, and direct
  `experts.gate_up_proj`/`experts.down_proj` parameters without `.weight`.

These are source-layout references, not a claim that every released model is
compatible. The pinned upstream references are [Transformers
4.51.3](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py),
[4.57.1](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py),
[5.0.0](https://github.com/huggingface/transformers/blob/v5.0.0/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py),
and the pinned current reference
[`64f30450dbfd1d02f610ad7080535cb906637fb9`](https://github.com/huggingface/transformers/blob/64f30450dbfd1d02f610ad7080535cb906637fb9/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py).

The adapter checks all structural attributes, exact router/expert shapes,
contiguous layer and indexed-expert names, and rejects missing, extra, mixed,
fused, or transformed surfaces. Discovery publishes only sparse layers,
routers, expert containers, and routed experts with exactly `[STRUCTURE]`,
`STATIC_STRUCTURE`, method `qwen3-moe-static-structure-v1`, and
`verified=False`. Packed expert entries are logical slices and carry the
fixed warning that they are not independently hookable. No routing scores,
expert activity, specialization, or routing certification is claimed.
Qwen2-MoE and Qwen3.5 hybrid/composite layouts are separate future adapters.
Real checkpoints, runtime versions, GPU behavior, and VM evidence remain
deferred.
