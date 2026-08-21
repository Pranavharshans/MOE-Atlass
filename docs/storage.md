# Routing-shard storage

Feature 19 adds one bounded, caller-invoked persistence seam:
`append_routing_shard(workspace, result, *, store_token_text=False)`. It accepts
only a complete model-neutral `RoutingForwardResult` from the Feature 18
one-forward boundary. `append_mixtral_routing_shard` remains an exact identity
alias for backward compatibility and accepts Qwen3.5 or future-family results
without changing the persisted schema. Because each historical name is the same
function object as its canonical name (not a wrapper), introspection attributes
such as `__name__`, `__qualname__`, and `inspect.getsource` report the neutral
canonical name; this exact-identity trade-off is accepted so alias identity,
signatures, and monkeypatch targets stay stable. The result is freshly revalidated
through the runtime-independent event collection boundary before any managed
directory is created; the opaque model output is never serialized, inspected,
or retained by storage. Concrete `RoutingForwardResult` typing is retained for
this compatibility slice; a broader result protocol requires a separate gate.
Feature 27 and Feature 28 validate Qwen3.5 through append, reopen, the
read-only run inventory, neutral aggregation, and neutral visualization. The
stored schema remains unchanged; this model-free path is not checkpoint/GPU
certification.

The workspace must already be a real, non-symlink directory. One immutable
content-addressed shard is written under:

```text
routing/v1/run-<sha256(run_key)>/shard-<sha256(semantic_rows)>/
  manifest.json
  tokens.parquet
  routing.parquet
```

Only fixed relative names and hash-derived directories are persisted. The
manifest contains the exact store/event versions, run and shard identities,
redaction choice, row counts, DuckDB writer/version, and fixed Parquet file
sizes and SHA-256 digests. Token and routing rows preserve the frozen event
field order with a contiguous `event_index`; Parquet is written with ZSTD by
an in-memory DuckDB connection using parameterized inserts and relation/path
output. Token text is NULL unless explicitly opted in.

Append is sequential and idempotent for the same semantic shard. Existing
shards are fully reopened and validated before a new shard is staged; duplicate
token identities or token/layer/rank links across distinct shards are a
`conflict`. Temporary staging is hidden and ignored by listing, while owned
staging is removed after ordinary pre-publication write failure. Publication
uses a same-parent atomic rename and directory/file fsync; a post-rename
durability ambiguity may leave a complete shard for a later idempotent retry.
Only hidden, non-symlink staging directories with the exact staging-name
pattern are ignored after a crash; staging files, symlinks, or malformed names
are rejected. There is no concurrent-writer guarantee.

`list_routing_shards(workspace, *, run_key=...)` is non-mutating.
`list_mixtral_routing_shards` is its exact compatibility alias. The function
reopens every committed shard, rejects symlinks, extras, manifest/checksum/
schema/count/identity/order corruption, and returns value-only receipts sorted
by `shard_key`. `RoutingShardError` exposes only one fixed stage:
`dependency`, `workspace`, `write`, `publish`, `reopen`, or `conflict`; its
low-level cause is chained but paths, token text, SQL, and credentials never
appear in public text.

The newline-terminated manifest has only `manifest_type="routing_shard"`,
`store_schema_version`/`event_schema_version`, identities, redaction/count fields, DuckDB
`writer_name`/version, and fixed file records (`name`, byte count, and
`sha256:<64hex>`). Parquet
tokens and routing rows repeat the store version, shard key, event index,
event schema/type, and their exact frozen event fields. Storage validates event
tuples without reading the opaque Feature 18 output; receipts retain only
scalar identity/count/path values.

The physical DuckDB schemas are fixed, ordered, and nullable (`DESCRIBE` reports
`YES` for every column because the writer does not persist SQL `NOT NULL`
constraints). Semantic validation still requires every identity/version/type
field to be present and exact. `tokens.parquet` is:

```text
store_schema_version VARCHAR NULL
shard_key            VARCHAR NULL
event_index          BIGINT  NULL
schema_version       VARCHAR NULL
event_type           VARCHAR NULL
token_key            VARCHAR NULL
run_key              VARCHAR NULL
sequence_id          VARCHAR NULL
token_pos            BIGINT  NULL
token_id             BIGINT  NULL
token_text           VARCHAR NULL   # NULL unless token text storage is opted in
token_text_stored    BOOLEAN NULL
phase                VARCHAR NULL
```

`routing.parquet` is:

```text
store_schema_version VARCHAR NULL
shard_key            VARCHAR NULL
event_index          BIGINT  NULL
schema_version       VARCHAR NULL
event_type           VARCHAR NULL
token_key            VARCHAR NULL
layer_key            VARCHAR NULL
rank                 BIGINT  NULL
expert_key           VARCHAR NULL
router_logit         DOUBLE  NULL
probability          DOUBLE  NULL
weight               DOUBLE  NULL
selected             BOOLEAN NULL   # persisted routing evidence requires TRUE
```

The listed order is part of the reopen contract. Every persisted row repeats
its store/shard/event identity and contiguous `event_index`; token text is
redacted by default, while router numeric fields retain the nullable evidence
shape of the frozen event contract.

Feature 20 may read these fixed shards through its bounded routing-load
aggregation seam. It revalidates the complete committed sources and uses the
inspection-published layer/expert universe; it does not alter shard bytes or
persist analysis output.

This is a bounded shard prerequisite, not a full workspace/catalog service.
It does not synthesize run metadata, migrate, compact, query, partition by
layer, export DataFrames, add a persistent database, or provide a CLI, server,
UI, heatmap, prompt, expert metric, or performance claim. Analysis/catalog
work remains deferred; ST-04 and model-dependent MV-01 through MV-08 remain
unchanged.

## Bounded routing-run inventory

Feature 23 adds a read-only inventory primitive over the committed shard tree.
Call it as `list_routing_runs(workspace, *, max_runs, max_shards,
max_event_rows, max_source_bytes)`. `list_mixtral_routing_runs` remains an exact
identity alias with the same signature and output.
All four budgets are required positive, non-bool integers. It scans only
`routing/v1`, counts run and committed-shard candidates before DuckDB, sums the
exact three managed files per shard, bounds declared and actual event rows,
then reuses the complete Feature 19 reopen validator, so malformed or corrupt
committed sources retain safe reopen/conflict failures. An absent `routing/v1`
is a canonical empty inventory and does not import DuckDB or mutate the
workspace.

`MixtralRoutingRunInventory` has schema version `1.0`, manifest type
`mixtral_routing_run_inventory`, exact totals, and lexically ordered
`MixtralRoutingRunSummary` values. Each summary preserves canonical shard keys,
exact source bytes, and `redacted`, `stored`, or `mixed` token-text policy.
`to_json()` is compact deterministic JSON with UTF-8 characters, no NaN, and
no trailing newline. Inventory exposes only fixed `budget` and `index` errors;
existing shard dependency, reopen, conflict, and workspace errors retain their
Feature 19 fixed-stage contract.

The inventory uses one bounded in-memory DuckDB connection, closes it on every
ordinary and control-flow path, retries a failed close once, and publishes no
value until cleanup and full reopen validation succeed. It is a rebuildable
read-only primitive, not a persistent database, workspace catalog, migration,
repair, compaction, query API, or general run registry. A caller supplies the
`run_key` to Feature 20/21 heatmap aggregation; inventory does not choose a
latest run or synthesize unavailable model, adapter, layout, inspection, time
tags, timestamp, or status metadata. It remains EXPERIMENTAL; ST-04 and MV-01 through
MV-08 remain deferred.
