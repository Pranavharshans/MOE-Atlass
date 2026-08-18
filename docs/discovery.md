# Static discovery

`moeatlas.discovery.scan(model, model_manifest)` produces a strict,
versioned `DiscoveryReport` from a PyTorch-compatible object. The scanner is
runtime-independent dry-run analysis: it uses only a callable `named_modules()` method, and
optionally `named_parameters()` plus a `config` mapping/object. It never
imports PyTorch, calls `forward`, registers hooks, reads tensor values, or
mutates the inspected object.

The runtime `load_and_scan()` bridge may supply a resolved HF/local model and
its already-built `ModelManifest` to this same scanner. That bridge is still
static-only: it does not tokenize, run inference, install hooks, or elevate
the resulting report beyond `[STRUCTURE]` capability.

## What is reported

The report embeds the validated `ModelManifest`, normalized facts, explainable
confidence-scored candidates, and matching `ComponentManifest` records. Each
static component has exactly the `[STRUCTURE]` capability (static discovery
does not imply runtime capture). Its component key is
recomputed from the canonical model key, semantic kind, module path, and
indices, so the report cannot silently refer to a different local path or
model revision.

The normalized facts currently include:

- expert count from `num_local_experts`, `num_experts`,
  `n_routed_experts`, or `num_expert`;
- routed top-k from `num_experts_per_tok`, `num_experts_per_token`,
  `num_selected_experts`, `top_k`, or `top_k_experts`;
- shared expert count from `n_shared_experts`, `num_shared_experts`, or
  `shared_expert_count`.

When configuration is unavailable, indexed expert modules and shared-expert
modules may provide a structural fallback. Counts are computed independently
per expert container or MoE layer. A single consistent count is normalized;
conflicting per-layer/container counts remain unset with a deterministic
warning rather than selecting or silently summing one layer. Every fallback
identifies its source in the report. A populated fact and its source are always
present together. A routed `top_k` above the normalized expert count is also
reported as a configuration warning.

## Signals and confidence

Scores are bounded to `[0, 1]` and are the sum of explicit, capped evidence
weights. The evidence list on every candidate explains the contribution:

| Signal | Examples |
| --- | --- |
| `config_field` | expert count, top-k, or shared-expert field |
| `path_name` | `router`, `experts`, `shared_expert`, or `moe` path marker |
| `class_name` | router, expert-container, or MoE class marker |
| `child_structure` | router plus expert container, or indexed children |
| `parameter_shape` | router dimension or packed expert axis matching configured count |
| `indexed_expert` | numeric child below an expert container |
| `shared_name` | explicit shared-expert marker |

A parameter shape receives evidence only when it is semantically consistent:
a router shape must contain a dimension equal to configured `expert_count`,
and a packed expert container must expose that count on a rank-3-or-greater
axis. Arbitrary readable shapes do not receive the same score.

Candidate confidence is validated against the rounded sum of its evidence
weights. Duplicate identical evidence entries are rejected, and any candidate
below `0.600` must carry an ambiguity warning.

A score is evidence strength, not a probability and not a model-support
certificate. Scores below `0.600` receive an `ambiguous candidate` warning for
human review. A name-only `gate` in an otherwise dense object can therefore
be surfaced as a low-confidence router candidate without being promoted to a
certified MoE structure. Ordinary dense modules with no independent MoE
signals produce no candidates.

## Architecture-specific static adapters

The first explicit adapter is `MixtralStaticAdapter()`, selected by the
caller rather than discovered through a registry. It recognizes exact Mixtral
family identity and two strict surfaces: the official Transformers 4.50
indexed `block_sparse_moe` layout and the current packed `mlp` layout. It
requires exact configuration counts, strict block/gate/expert attributes,
contiguous layers and experts, and semantic router/expert shapes. Legacy
expert `w1`, `w2`, `w3`, and `act_fn` children and packed `act_fn` children are
part of the exact accepted surface. Packed experts use direct
`gate_up_proj`/`down_proj` parameters and are reported as logical slices with
a non-hookable-slice warning. Qwen and dense gated MLPs are separate families
and are not promoted by fuzzy names or class names.

Adapter output is still static `[STRUCTURE]` evidence with unverified static
provenance. It does not observe routing or certify expert behavior; real
Transformers versions, fused/quantized surfaces, and VM validation remain
deferred.

## Boundaries and future work

Static discovery describes names, structure, configuration, and parameter
shapes only. It does not prove that a module is executed, that a router emits
the reported top-k, or that an expert is specialized. Those claims belong in
the deferred model-validation ledger (especially MV-02 and later runtime
checks).

The positional `MODEL` form of the Phase 0 CLI accepts only the explicit
`fixture:synthetic` source; it does not turn a local path or model ID into a
loader request. The separate `--loading-plan PLAN.json` form accepts one
validated, already-resolved Hugging Face or local `LoadingPlan` and passes it
through the runtime bridge. Static semantic adapters may consume this report
as an input, but must preserve its provenance and `[STRUCTURE]` capability;
they cannot claim runtime behavior or elevate a component. Real checkpoint
certification remains deferred to Phase 1 and MV-01/MV-02.
