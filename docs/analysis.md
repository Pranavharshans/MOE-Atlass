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
