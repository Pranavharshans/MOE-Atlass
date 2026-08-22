# Routing-load analysis

Feature 28 provides one bounded, read-only model-neutral aggregation seam:
`aggregate_routing_load(workspace, inspection, *, run_key, max_routing_rows,
max_source_bytes, max_matrix_cells)`. It accepts an exact, fresh
`AdapterInspection` and reads every committed Feature 19 shard for one run. It
does not infer an expert or layer universe from observations: the inspection
must publish every routed, non-shared expert in contiguous index order for
every router layer, together with an exact `legacy_indexed` or `packed` layout.
Mixtral and Qwen3.5 both use this path; a future adapter with the same complete
structural contract is accepted without an analysis-family branch.
This neutralizes and supersedes the Feature 20 Mixtral-only analysis boundary;
the Feature 20 contract remains byte-compatible through the historical aliases.

The historical `MixtralRoutingLoadMatrix` and
`aggregate_mixtral_routing_load` names are identity aliases of
`RoutingLoadMatrix` and `aggregate_routing_load`. Adapter name, version, and
architecture family are provenance, not an allowlist. Shared-expert components
must be explicitly marked `shared=True, routed=False`; they are checked for
per-layer completeness but never enter the routed expert axis or denominator.

The function validates strict positive budgets before filesystem traversal or
the lazy DuckDB import. It budgets the matrix cell universe, manifest and
Parquet bytes, and declared and actual row counts before materializing a shard.
It uses one in-memory DuckDB connection, parameterized fixed-path queries, and
closes that connection on every ordinary or control-flow exit. It publishes no
partial matrix. Existing `RoutingShardError` values retain their storage
stage; analysis failures use the fixed text
`routing load aggregation failed at <stage>` for `inspection`, `budget`,
`source`, or `query`.

The value result is `RoutingLoadMatrix` with schema version `1.0` and
exact provenance: store/event versions, run/model/adapter identity, the
fresh-inspection digest, layout, sorted shard keys, token and assignment
counts, top-k, and explicit layer/expert axes. Its matrices are rectangular
and contain every inspection expert, including zero-count experts:

* `assignment_counts[layer, expert]` is the number of selected routing rows;
* `assignment_shares[layer, expert] = count / (token_count * routed_top_k)`;
* `load_ratios[layer, expert] = share * expert_count`.

One selected row is one assignment. A valid source has every token for every
inspection layer at every rank `0..routed_top_k-1`, with distinct known
experts. Consequently each layer count row sums to
`token_count * routed_top_k`, the total assignment count is
`token_count * layer_count * routed_top_k`, each share row sums to one, and
each ratio row has mean one. Results contain no paths, connections, raw rows,
inspection object, or token text.

This is an `EXPERIMENTAL` bounded prerequisite for later research. It is not a catalog
and does not write storage, create a persistent database, migrate or compact
shards, expose raw rows, filter runs, export DataFrames, calculate
probability/entropy/specialization metrics, or provide CLI/server/UI/query
surfaces. Tokenization, prompts, padding, generation, and model execution are
caller responsibilities. ST-04 scale/query/catalog work and MV-01 through MV-08
remain deferred; no model files are downloaded by this feature.

## Cross-run routing-load comparison

Feature 29 adds one bounded, read-only, dependency-free comparison seam:
`compare_routing_load(baseline, comparison, *, max_cells)`. It accepts two
exact, fresh `RoutingLoadMatrix` values and requires one identical universe:
schema/store/event versions, model key, adapter identity, inspection digest,
layout, routed top-k, token count, and the complete layer/expert axes must
match exactly; the two run keys must differ. The function performs no I/O and
raises plain `TypeError`/`ValueError` failures like the renderer.

The value result is `RoutingLoadComparison` with schema version `1.0`. It
preserves both run keys, both sorted shard-key tuples, both assignment
counts, and the shared frozen provenance, and publishes three rectangular
delta matrices over the shared axes:

* `count_deltas[layer, expert] = comparison count − baseline count`;
* `share_deltas[layer, expert] = comparison share − baseline share`;
* `ratio_deltas[layer, expert] = comparison ratio − baseline ratio`.

Because equal token counts and top-k are required, every count-delta row sums
to zero, every share-delta row sums to zero within `1e-12`, and every
ratio-delta row has mean zero. Share deltas stay inside the unit interval and
ratio deltas inside `±expert_count`; non-finite values are rejected. The
comparison retains only scalars and tuples—no matrix objects, paths,
connections, raw rows, or token text.

This is an `EXPERIMENTAL` primitive for later compare workflows. It does not
write files, render HTML, query storage, rank experts, or claim that a delta
is specialization, causality, or improvement; interpretation remains a caller
responsibility. MV-01 through MV-08 remain deferred; no model files are
downloaded by this feature.

## Routing-load summary metrics

Feature 32 adds one bounded, read-only, dependency-free summary seam:
`summarize_routing_load(matrix, *, max_cells)`. It accepts one exact, fresh
`RoutingLoadMatrix` and computes the PRD §11.1 metrics that are derivable from
aggregate load alone, per layer and globally:

* `layer_entropies` — Shannon entropy of assignment shares in nats; zero
  shares contribute exactly zero;
* `normalized_layer_entropies` — entropy ÷ ln(expert count), so uniform load
  is exactly 1;
* `effective_expert_counts` — exp(entropy), the entropy-derived effective
  expert count;
* `normalized_diversities` — effective count ÷ expert count;
* `layer_gini_coefficients` — exact ascending-rank Gini over integer counts;
* `layer_cv_counts` — population coefficient of variation of counts;
* `top_expert_shares` — the largest per-layer share;
* `dead_expert_count` / `dead_expert_fraction` — zero-assignment cells over
  the complete layer × expert universe.

The value result is `RoutingLoadSummary` with schema version `1.0`, frozen
matrix provenance (run/model/adapter identity, inspection digest, layout,
counts, shard keys, axes), and range-checked finite floats: entropies within
`[0, ln E]`, normalized/diversity/Gini/top-share within `[0, 1]`, effective
counts within `[0, E]`, nonnegative coefficients of variation, and an exact
dead-fraction identity. Router margin and route churn are deliberately absent:
they require score distributions and adjacent-token sequences that aggregate
load shards do not carry.

This is an `EXPERIMENTAL` descriptive statistic over association evidence. It
performs no I/O, writes nothing, ranks nothing across runs, and never claims
specialization or causal effect; a low Gini or high entropy is load balance
evidence only. MV-01 through MV-08 remain deferred; no model files are
downloaded by this feature.

## Canonical artifact serialization

Feature 33 adds `to_dict()`, `to_json()`, and `from_json()` to
`RoutingLoadMatrix`, `RoutingLoadComparison`, and `RoutingLoadSummary`.
Documents are canonical: sorted keys, compact separators, `allow_nan=False`,
and an explicit `artifact_type` marker (`moeatlas.routing_load_matrix`,
`moeatlas.routing_load_comparison`, `moeatlas.routing_load_summary`) plus the
artifact schema version. Repeated exports of one value are byte-identical, and
`from_json(to_json())` returns an exactly equal value.

Import is staged and strict: JSON parse failures, non-object documents, wrong
artifact types or schema versions, missing fields, wrong JSON value types
inside arrays (bools where integers are required, integers where floats are
required), and every value-contract violation surface as safe fixed
`TypeError`/`ValueError` messages without echoing input content. Unknown
top-level keys are ignored so future additive revisions remain readable.
Serialization is pure: it writes no files and touches no storage; callers own
publication through the existing atomic writer or their own paths. This is an
`EXPERIMENTAL` portable-evidence primitive; MV-01 through MV-08 remain
deferred.

## Analysis export bundles

Feature 34 adds one bounded publication seam over the canonical documents:
`write_analysis_bundle(destination, *, matrix=None, comparison=None,
summary=None)`. It requires at least one artifact, exact types, and one shared
model/adapter/inspection/layout identity across everything provided. The
bundle is a normal directory of fixed names (`routing_load_matrix.json`,
`routing_load_comparison.json`, `routing_load_summary.json`) plus a canonical
`manifest.json` written last: sorted compact JSON with an
`moeatlas.analysis_bundle` artifact type, per-entry SHA-256 digests and exact
UTF-8 byte counts, entry count, and total bytes.

Every file is published through a same-directory temporary file, `fsync`, and
`os.replace`; any ordinary or control-flow failure unlinks everything written
and removes a directory the seam created, so no partial bundle ever survives.
The destination must be nonexistent or empty with an existing parent. The
frozen `AnalysisBundleReceipt` value carries the sorted entries and totals so
callers can verify what was published without re-reading the tree. Bundles are
byte-deterministic: equal artifacts produce identical files in any location.

This is an `EXPERIMENTAL` export primitive, not a workspace/catalog/query
subsystem; it never reads shards, aggregates, renders, or interprets. MV-01
through MV-08 remain deferred.

## Universal structure inspections

Phase R1 unblocks the analysis lane for any architecture:
`aggregate_routing_load()` accepts, alongside a certified
`AdapterInspection`, a universal structure inspection —
`moeatlas.adapters.build_universal_inspection(report)` derives a strict
`UniversalRoutingInspection` from one exact `[STRUCTURE]` discovery report
using only model-neutral structural evidence (semantic kinds, routing flags,
contiguous layer/expert indices, and the `expert_count`/`routed_top_k`
facts). The document marks its own provenance as `universal`, carries the
architecture family and scanner version, fixes the layout tag to the
`packed` equivalent (generic scans do not certify a native indexing layout),
and binds the layer/expert axes with a canonical SHA-256 `axes_digest` that
is re-verified on every import. It round-trips through canonical JSON like
every other artifact here.

Both lanes produce the identical matrix contract; universal documents report
adapter identity as the fixed `universal` marker plus the scanner version,
so downstream artifacts never present generic evidence as certified adapter
evidence. Adapter-declared universes (`declared_universe`) remain a
certified-inspection capability; passing one with a universal document fails
at the inspection stage before any source work.

## Declared routing universes

`aggregate_routing_load()` accepts an optional `declared_universe`: the
adapter-published `RoutingUniverse` manifest for the inspection (see
[adapters](adapters.md)). The declared universe must be exactly what the
inspection publishes and must pass `project_rectangular_universe()`
explicitly, so rectangularity is a checked, named gate at the API boundary
rather than a hidden invariant of the internal axis walk. A mismatching or
non-rectangular universe fails at the inspection stage — before any shard
source work — with the violated shape named. Aggregation results are
identical with or without a declared universe; the parameter pins the
expected topology and keeps analysis honest about which structure it
consumes. Adapter-declared layout tags pass publication freely but remain
subject to routing-load's decodable-layout contract (`legacy_indexed`,
`packed`) on both paths.

## Task association metrics

`moeatlas.analysis.task_association` implements the model-neutral core of
PRD §11.2 over a strict frozen contingency table:
`TaskExpertCounts(layer_keys, task_keys, expert_keys, counts)` holds
selected-route counts per (layer, task, expert) with sorted unique keys,
exact rectangularity, non-negative integers, and a positive total for every
task so all conditionals are defined. `analyze_task_association(counts, *,
max_cells)` computes, per layer and in canonical order:

- `enrichment_rows` — `P(expert | task) / P(expert)`; a cell is `null`
  exactly when the expert is unused in the layer (no denominator), and `0.0`
  is a defined answer when the task never selects a used expert.
- `pmi_rows` — `log2` of the same ratio; additionally `null` when the
  conditional probability is zero, since pointwise information would be
  negative infinity.
- `mutual_information_rows` — per-layer MI(task; expert) in bits;
  `specific_mi_rows` hold each task's contribution, and task-share-weighted
  specific MI sums to the layer's MI exactly.
- `separability_rows` — mean pairwise base-2 Jensen-Shannon divergence
  between the tasks' routing distributions, bounded in `[0, 1]`; `null` with
  fewer than two tasks.
- `exclusivity_rows` — `max_t P(task | expert)` plus the count of tasks the
  expert actually receives: 1.0 means fully exclusive, uniform means general.

The result is a frozen `TaskAssociationMatrix` with `to_dict`/`to_json`/
`from_json` round-trips under the `moeatlas.task_association` artifact type;
`null` replaces `NaN` everywhere and documents absence as evidence. Every
metric says "routes here more/less often than baseline" — association is
never specialization or causality. Per-token task-labeled evidence arrives
with task-labeled executors in later sequences, so the contract is exercised
over synthetic tables; prompt-vs-rollout agreement and cross-run stability
of association are later slices in this sequence.

## Prompt-vs-rollout agreement

`moeatlas.analysis.routing_agreement` implements PRD §11.2's prompt-level vs
rollout-level routing agreement. `PromptRolloutCounts(layer_keys,
expert_keys, prompt_counts, rollout_counts)` pairs two rectangular
selection-count tables over one shared per-layer expert universe; every layer
needs a strictly positive total in both phases so both conditional
distributions are defined, while individual unused experts are fine.
`analyze_routing_agreement(counts, *, max_cells)` derives, per layer and in
canonical order:

- `js_divergence_rows` — base-2 Jensen-Shannon divergence between the
  prompt-phase and rollout-phase routing distributions, bounded in `[0, 1]`
  (float noise is clamped so the documented bounds hold exactly).
- `agreement_rows` — `1 - JSD`, the bounded similarity complement: 1.0 when
  both phases route identically, 0.0 on disjoint supports.
- `tv_distance_rows` — half the L1 distance between the distributions, an
  order-independent disagreement lens also bounded in `[0, 1]`.

The result is a frozen `RoutingAgreement` with `to_dict`/`to_json`/
`from_json` round-trips under the `moeatlas.routing_agreement` artifact type;
positive phase totals are construction preconditions, so no cell is ever
null. Agreement compares distributions only — consistency of routing across
phases is never specialization or causality.

## Router margin

`moeatlas.analysis.router_margin` implements PRD §11.1's router margin —
the difference between a token's two top selected-route scores at one layer,
a routing-confidence lens. `RouterMarginSamples(layer_keys, token_scores)`
holds, per layer, one inner tuple per token with that token's selected-route
scores ordered best-first (logits or probabilities alike; any finite scores;
negative values are legal). `analyze_router_margin(samples, *, max_tokens)`
derives per layer:

- `mean_margin_rows` — mean top1-minus-top2 margin over tokens with at
  least two scored ranks; `null` when no token in the layer has a defined
  margin.
- `margin_token_rows` — the count of tokens that contributed a margin.
- `token_rows` — the count of all supplied tokens.

Tokens with fewer than two scored ranks contribute no margin — absence is
evidence, never inferred. The result is a frozen `RouterMarginSummary` with
`to_dict`/`to_json`/`from_json` round-trips under the
`moeatlas.router_margin` artifact type. A margin describes routing
confidence only — never specialization or causality.

## Route churn

`moeatlas.analysis.route_churn` implements PRD §11.1's route churn — how
expert selection changes across adjacent steps. Token keys are content
digests, so adjacency is a caller-supplied ordering (adjacent generated
tokens, prompt perturbations, or any routed-step sequence);
`RouteChurnSequences(layer_keys, step_experts)` holds, per layer, one
selected-expert tuple per step in that order (order inside a step is
irrelevant; empty steps are legal). `analyze_route_churn(sequences, *,
max_steps)` derives per layer:

- `churn_rate_rows` — the fraction of adjacent pairs whose selected sets
  differ; `null` with fewer than two steps.
- `mean_jaccard_rows` — mean Jaccard distance `1 - |A ∩ B| / |A ∪ B|` over
  the pairs, with the documented conventions that empty-to-empty is no
  change and empty-to-nonempty is full change.
- `pair_rows` — the count of adjacent pairs.

The result is a frozen `RouteChurnSummary` with `to_dict`/`to_json`/
`from_json` round-trips under the `moeatlas.route_churn` artifact type.
Churn describes routing stability only — never specialization or causality.

## Co-routing graphs

`moeatlas.analysis.corouting` implements the model-neutral core of PRD
§11.3's expert co-activation and conditional co-routing graphs.
`ExpertCoRoutingCounts(layer_keys, expert_keys, pair_counts)` holds, per
layer, a symmetric square matrix of co-selection counts over that layer's
experts (zero diagonal — an expert is never paired with itself).
`summarize_co_routing(counts, *, max_cells, max_pairs)` derives per layer:

- `total_pair_selections` — the layer's total co-selection mass.
- `coupled_expert_rows` — how many experts appear in at least one
  co-selection (uncoupled experts stay explicit, never inferred away).
- `top_pairs` — `(expert_a, expert_b, count, share)` tuples sorted by
  descending count then ascending keys and bounded by `max_pairs`, with
  `share` normalizing each pair by the layer's total mass.

The result is a frozen `CoRoutingGraph` with `to_dict`/`to_json`/
`from_json` round-trips under the `moeatlas.corouting` artifact type.
Co-routing is association evidence only:
it never implies specialization or causality.

## Cross-run association stability

`moeatlas.analysis.association_stability` implements PRD §11.2's cross-run
stability of expert-task association.
`analyze_association_stability(counts_a, counts_b, *, max_cells)` takes two
`TaskExpertCounts` tables over one identical (layer, task, expert) topology —
any key mismatch is a contract error, not a silent realignment — and
compares, for every (layer, task) cell, the two runs' conditional routing
distributions P(expert | task):

- `js_divergence_rows` — base-2 Jensen-Shannon divergence between the runs,
  bounded in `[0, 1]` (float noise clamped so the bounds hold exactly).
- `agreement_rows` — `1 - JSD`: 1.0 when both runs route a task identically,
  0.0 on disjoint supports.
- `mean_agreement_rows` — per-layer mean agreement across tasks.

The result is a frozen `AssociationStability` with `to_dict`/`to_json`/
`from_json` round-trips under the `moeatlas.association_stability` artifact
type; positive task totals are a construction precondition of both inputs, so
no cell is ever null. Agreement between runs is evidence of reproducible
routing behavior — never specialization or causality.

## Evidence Cards

`moeatlas.analysis.evidence_cards` implements the structured alternative to a
single specialization score (PRD §11.5). An `EvidenceCard` binds one expert
to a model fingerprint (`sha256:<64 hex>`), its layer and expert keys, and an
explicit `expert_kind` of `routed` or `shared` — shared-expert keys are legal
only on shared cards. Every evidence tier is a separate optional section, and
`null` always means "not measured", never inferred:

- `routing` — usage share, normalized load, mean rank/margin/entropy
  (tier A, routing usage).
- `task_association` — per-task enrichment/PMI/exclusivity rows aligned with
  `task_keys`, plus the `example_count` standing behind the numbers.
- `behavior` — bounded input/output/contribution summaries (tier B,
  internal behavior).
- `causality` — intervention deltas with an optional recipe fingerprint
  (tier C); the card records results and never claims causality from
  association.
- `stability` — replicated/total seeds and datasets (tier D, replication).

Cards carry str-only `limitations` and `warnings`, `capability_labels` as
`(tier, label)` pairs over the fixed `EVIDENCE_TIERS` vocabulary with labels
`full`/`partial`/`unsupported` (one label per tier), and optional provenance
(`probe_version`, `adapter_name`, `adapter_version`, `capture_source`). The
frozen value round-trips through canonical JSON as a
`moeatlas.evidence_card` artifact; contract violations raise `TypeError` for
wrong types and `ValueError` for bad values, and `EvidenceCardError` reports
fixed `contract`/`serialization` stages. Cards are exercised over synthetic
values; real-model evidence remains deferred MV work.

## Expert similarity

`moeatlas.analysis.expert_similarity` implements PRD §11.3's expert weight
and representation similarity. `ExpertVectors(layer_keys, expert_keys,
vectors)` holds, per layer, exactly one finite vector per expert key; every
vector within one layer shares its length, and zero vectors are legal input.
The contract is agnostic to provenance: callers supply expert weight
summaries or representation summaries alike, because token keys are content
digests and vector extraction is adapter territory.
`analyze_expert_similarity(vectors, *, max_cells)` derives, per layer, the
symmetric cosine-similarity matrix over that layer's experts:

- `similarity_rows[layer][i][j]` — cosine similarity between experts i and
  j, clamped to `[-1, 1]`; diagonals of nonzero vectors are exactly `1.0`.
- cells touching a zero-norm expert are `null` — undefined geometry is
  evidence, never inferred;
- `undefined_expert_rows` counts zero-norm experts per layer.

Similarity describes geometry only:
it never implies specialization or causality.
The result is a frozen `ExpertSimilarity` with `to_dict`/`to_json`/
`from_json` round-trips under the `moeatlas.expert_similarity` artifact type;
`ExpertSimilarityError` reports fixed `contract`/`budget` stages. Vectors are
exercised over synthetic values; real weight/activation capture remains
deferred MV work.

## Causal evidence summaries (PRD §11.4)

`analyze_causal_evidence` in `moeatlas.analysis.causal_evidence` reduces
caller-supplied `CausalPair` observations — one pair per metric label and
replication index holding baseline and intervened values — into a frozen
`CausalEvidence` with sorted labels and, per label:

- `mean_baseline` / `mean_intervened` — replication means;
- `absolute_effects` — intervened minus baseline;
- `relative_effects` — effect over `abs(mean_baseline)`, `null` where the
  baseline mean is zero;
- `direction_consistency` — the share of replication effects matching the
  mean-effect sign, `null` where the mean effect is exactly zero;
- `stable_labels` — true only when every replication effect shares one
  nonzero direction (strict stability: mixed or zero directions are not
  stable);
- `zero_effect_labels` — every replication effect is exactly zero.

Duplicate `(label, replication)` entries fail at the fixed `contract`
stage; oversized inputs fail at `budget`. The result round-trips under the
`moeatlas.causal_evidence` artifact type. The layer is pure — no clocks,
randomness, storage, or model knowledge. An effect summary describes paired
observations only; it never by itself proves specialization, and real-model
causal claims stay deferred to the validation ledger. These summaries are
the content Evidence Cards' causality/stability sections carry; recipes and
restoration guarantees live in [interventions](interventions.md).

## Per-layer expert-activation summaries (R3.3)

`summarize_expert_activity(workspace, *, run_key, layer_keys, expert_keys,
max_expert_rows, max_source_bytes)` produces one canonical, frozen,
round-trippable activation summary (`moeatlas.expert_activity_summary`,
schema `1.0`) over the reopened shards of one run. Reading goes exclusively
through the public storage query seam (`query_expert_activity`): every shard
is revalidated — manifest identity, checksums, row identities, token links,
conflicts, budgets — before aggregation, and no raw expert rows are retained
anywhere; only per-cell aggregates survive.

The summary reports, per layer and per discovered expert, the event count
plus mean and max `contribution_norm` over measured events. Zero activity is
accounted explicitly: universe cells whose experts never fired carry a zero
count with null statistics and are counted in `inactive_expert_cells`,
alongside `active_expert_cells` (the two always partition the layer × expert
universe) and `total_event_count`. Because expert keys are opaque component
identities, the caller supplies the layer universe (`layer_keys` plus one
expert-key row per layer) exactly as the routing-load lane does; evidence
outside that universe fails closed rather than being silently dropped. The
value serializes deterministically via `to_json()` and restores exactly via
`ExpertActivitySummary.from_json()`.
