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
Qwen2-MoE remains a separate future adapter. Real checkpoints, runtime
versions, GPU behavior, and VM evidence remain deferred.

## Routing probe-plan compilation

`build_routing_probe_plan(inspection)` is the family-neutral bridge from one
already validated `AdapterInspection` to an inert `ProbePlan`. It re-dumps and
freshly reconstructs the exact inspection before reading semantic fields, then
selects every and only `ROUTER` component. At least one router is required and
duplicate router module paths are rejected. The resulting targets preserve the
canonical module path, component key, and `ComponentKind.ROUTER` identity.

The compiler emits `ROUTING` with only the `forward` hook point, empty
include/exclude selectors, and a reduced `TOP_K` capture policy:
inputs and gradients are disabled, outputs are enabled, raw opt-in and all
budgets are disabled, and sampling is deterministic at `1.0`. Because
`max_items` and `max_bytes` are both `None`, this is intent only and imposes
no execution, event, or storage bound. `ProbePlan` itself canonicalizes
ordering and computes the stable plan ID. The inspection remains the
source-of-truth artifact and must be retained alongside the plan.

This translation does not call adapters or models, resolve named modules,
install hooks, decode tensors, capture events, write storage, or claim routing
certification. Later runtime execution follows the official
[PyTorch forward-hook API](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook).
For both families, the comparison references are the tagged legacy
[Mixtral v4.50.0 source](https://github.com/huggingface/transformers/blob/v4.50.0/src/transformers/models/mixtral/modeling_mixtral.py)
and [Qwen3-MoE v4.57.1 source](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py),
plus the pinned current
[Mixtral source at `64f30450dbfd1d02f610ad7080535cb906637fb9`](https://github.com/huggingface/transformers/blob/64f30450dbfd1d02f610ad7080535cb906637fb9/src/transformers/models/mixtral/modeling_mixtral.py)
and [Qwen3-MoE source at the same pinned commit](https://github.com/huggingface/transformers/blob/64f30450dbfd1d02f610ad7080535cb906637fb9/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py).
These are source-layout comparisons only, not a broad compatibility claim.
The tagged legacy and pinned current implementations can expose different
router forward-payload conventions; this compiler assumes no tensor/tuple
decoder and treats the plan as intent only. Native routing payload
equivalence and capture remain deferred to MV-03.

## Mixtral routing evidence boundary

`MixtralRoutingDecoder` is a separate, explicit runtime decoder for the exact
`huggingface-mixtral-static` descriptor. It consumes one fresh
`AdapterInspection` and an exact non-empty tuple of caller-supplied
`TokenEvent` rows. The tuple order is authoritative for the router's token
rows; no tokenizer, generation runner, padding inference, storage, or model
loading is introduced. A decoder is single-use per router path so a one-forward
caller cannot silently combine payloads from separate executions.

Router capture metadata must consistently identify either the
`legacy_indexed` or `packed` layout. Legacy `[tokens, experts]` logits produce
only observed selected logits, with no inferred probability or weight. Packed
`(logits, scores, indices)` evidence requires native integer indices and native
finite weights to agree with deterministic softmax/top-k renormalization.
Ambiguous selected/cutoff ties, shape mismatches, non-finite values, and
tampered router/layer/expert bindings are rejected. Conversion is strictly
`detach -> cpu -> float -> tolist` for logits/scores and `detach -> cpu ->
tolist` for indices, with no tensor-runtime or NumPy import and no raw tensor
retention.

This decoder publishes `RoutingEvent` values with an `EXPERIMENTAL` evidence
boundary only; it does not elevate static components to `FULL` or claim
routing certification. Native equivalence, passive output fidelity, routing
overhead, GPU behavior, fused/quantized paths, and packaging revalidation stay
deferred to MV-03 through MV-08.

## Qwen3.5-MoE static adapter (experimental)

`Qwen3_5MoeStaticAdapter` is an explicit, caller-supplied structure-only
adapter for the current Qwen3.5-MoE identity (`qwen3_5_moe` and
`qwen3_5_moe_text`), based on the official Transformers v5.14 surface. It
accepts only the packed expert layout:

- `mlp.gate.weight`: `[experts, hidden]`;
- `mlp.experts.gate_up_proj`: `[experts, 2 * moe_intermediate, hidden]`;
- `mlp.experts.down_proj`: `[experts, hidden, moe_intermediate]`;
- `mlp.shared_expert.{gate_proj,up_proj}.weight` and
  `mlp.shared_expert.down_proj.weight` for the non-routed shared expert; and
- `mlp.shared_expert_gate.weight`: `[1, hidden]` metadata only.

The layout is anchored to the official
[Transformers v5.14.0 Qwen3.5-MoE implementation](https://github.com/huggingface/transformers/blob/v5.14.0/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py),
its [modular source](https://github.com/huggingface/transformers/blob/v5.14.0/src/transformers/models/qwen3_5_moe/modular_qwen3_5_moe.py),
and the current [Qwen3.5-35B-A3B model card](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
and [configuration](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/raw/main/config.json).
The gate's native runtime tuple is `(router_logits, router_scores,
router_indices)`; decoding that tuple is Feature 26 and is intentionally not
implemented by this static seam.

Conditional models must expose `model.language_model.config` as the exact
nested `text_config` object and must expose `model.language_model.layers`;
text-only models must expose `model.layers` (with the exact bare `layers` base
surface accepted only for the official text identity). Every configured layer
must be MoE; `layer_types` is validated as the per-layer attention-kind list,
not interpreted as a sparse/dense schedule. Indexed experts, mixed roots,
fuzzy architectures, foreign descendants, and missing shared-expert modules
are rejected. Packed experts are published as logical slices and are not
independently hookable. The seam performs no model loading, tensor reads,
registry selection, or routing/model certification; it emits only `STRUCTURE`
evidence with `verified=False`.
Feature 26 routing capture/decoding, current-checkpoint loading, GPU
equivalence, and release-time immutable revision review remain deferred to the
final VM.
