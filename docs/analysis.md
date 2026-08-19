# Routing-load analysis

Feature 20 adds one bounded, read-only aggregation seam:
`aggregate_mixtral_routing_load(workspace, inspection, *, run_key,
max_routing_rows, max_source_bytes, max_matrix_cells)`. It accepts an exact,
fresh Mixtral `AdapterInspection` and reads every committed Feature 19 shard
for one run. It does not infer an expert or layer universe from observations:
the inspection must publish every routed, non-shared expert in contiguous
index order for every router layer, together with an exact `legacy_indexed` or
`packed` layout.

The function validates strict positive budgets before filesystem traversal or
the lazy DuckDB import. It budgets the matrix cell universe, manifest and
Parquet bytes, and declared and actual row counts before materializing a shard.
It uses one in-memory DuckDB connection, parameterized fixed-path queries, and
closes that connection on every ordinary or control-flow exit. It publishes no
partial matrix. Existing `RoutingShardError` values retain their storage
stage; analysis failures use the fixed text
`mixtral routing load aggregation failed at <stage>` for `inspection`,
`budget`, `source`, or `query`.

The value result is `MixtralRoutingLoadMatrix` with schema version `1.0` and
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
