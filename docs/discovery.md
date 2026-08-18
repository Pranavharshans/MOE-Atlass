# Static discovery

`moeatlas.discovery.scan(model, model_manifest)` produces a strict,
versioned `DiscoveryReport` from a PyTorch-compatible object. The scanner is
runtime-independent: it uses only a callable `named_modules()` method, and
optionally `named_parameters()` plus a `config` mapping/object. It never
imports PyTorch, calls `forward`, registers hooks, reads tensor values, or
mutates the inspected object.

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

## Boundaries and future work

Static discovery describes names, structure, configuration, and parameter
shapes only. It does not prove that a module is executed, that a router emits
the reported top-k, or that an expert is specialized. Those claims belong in
the deferred model-validation ledger (especially MV-02 and later runtime
checks).

The Phase 0 CLI exposes this scanner only through the explicit
`fixture:synthetic` source. It does not turn a local path or model ID into a
loader request. The next runtime features may add dry-run inference, passive
hooks, and architecture adapters. They must consume this report as an input,
preserve its provenance, and downgrade capabilities when semantic behavior is
not verified. Real model loading remains deferred to Phase 1 and MV-01/MV-02.
