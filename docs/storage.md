# Routing-shard storage

Feature 19 adds one bounded, caller-invoked persistence seam:
`append_mixtral_routing_shard(workspace, result, *, store_token_text=False)`.
It accepts only a complete `MixtralRoutingForwardResult` from the Feature 18
one-forward boundary. The result is freshly revalidated before any managed
directory is created; the opaque model output is never serialized, inspected,
or retained by storage.

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

`list_mixtral_routing_shards(workspace, *, run_key=...)` is non-mutating. It
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

This is a bounded shard prerequisite, not a full workspace/catalog service.
It does not synthesize run metadata, migrate, compact, query, partition by
layer, export DataFrames, add a persistent database, or provide a CLI, server,
UI, heatmap, prompt, expert metric, or performance claim. Analysis/catalog
work remains deferred; model-dependent MV-01 through MV-08 remain unchanged.
