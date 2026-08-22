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

## Run-evidence export bundles

The open-format interchange surface over the internal Parquet shards is the
run-evidence export bundle (EXPERIMENTAL), implemented by three seams exported
from `moeatlas.store`: `export_run_bundle(workspace, destination, *,
run_key, max_event_rows, max_file_bytes)`, `verify_run_bundle(source, *,
max_event_rows, max_file_bytes)`, and `import_run_bundle(source, workspace, *,
max_event_rows, max_file_bytes)`. One bundle carries the complete committed
evidence of exactly one run:

```text
<destination>/
  manifest.json
  data/shard-<sha256>/
    tokens.jsonl
    routing.jsonl
```

The newline-terminated `manifest.json` has `manifest_type="routing_run_export"`,
the exact `1.0` bundle schema version plus store/event schema versions, the
`run_key`, fixed `writer_name="moeatlas"`/version fields, run totals, and one
entry per committed shard with its content-addressed `shard_key`, row counts,
per-shard `token_text_stored` redaction flag, and fixed file records (`name`,
byte count, `sha256:<64hex>`). Every JSONL line is one canonically encoded JSON
object (`sort_keys`, compact separators, UTF-8, no NaN) whose fields are exactly
the frozen event fields plus a contiguous zero-based `event_index`; line order
preserves shard event order. Two exports of the same committed run state produce
byte-identical bundles.

Verification recomputes every file digest from bytes, enforces canonical
encoding per line and for the manifest, revalidates each shard's events through
the neutral collection boundary and their token/layer/rank links, cross-checks
redaction consistency (`null` text exactly when the shard did not store text),
and re-derives each shard's content-addressed identity from the exported events
themselves — so forged digests over altered evidence still fail. Verification is
a pure reader: it never imports DuckDB. Import composes full verification with
the standard shard appender, so importing into the source workspace is
idempotent (existing shards return `created=False`) and importing into a fresh
workspace reproduces identical shard identities; conflicting committed
identities surface the storage `conflict` stage.

Redaction travels with the evidence: a shard stored without token text exports
`"token_text": null` lines and its manifest entry records
`token_text_stored: false`; import reconstructs the identical shard from either
form. Absence of token text is explicit bundle evidence, never silently inferred
data. Both row totals and every file are bounded by strict positive
`max_event_rows`/`max_file_bytes` budgets on export, verify, and import alike.

Publication is atomic and crash-safe: the destination must be nonexistent or an
empty real directory, members are staged in a hidden sibling directory, fsynced,
written before the manifest, and atomically renamed into place with parent
fsync; any failure or control-flow interruption removes the staging directory.
Symlinks are refused for the destination, the source, and every bundle member.
Errors are `RunBundleError` with one fixed stage: `dependency`, `workspace`,
`source`, `format`, `budget`, `write`, `publish`, or `conflict`; underlying
causes stay chained but out of public text. Receipts (`RunBundleReceipt`,
`RunBundleFileEntry`) carry only scalar identity/digest values. Bundles are a
bounded interchange format, not an analysis export, migration tool, or
compaction path; they make no model-dependent claims, and ST-04 and MV-01
through MV-08 remain unchanged.

## Routing-run assignment query

`query_routing_run_assignments(workspace, *, run_key, layer_keys, expert_keys,
routed_top_k, max_routing_rows, max_source_bytes, duckdb, connection)` is the
public reader/query seam over committed shards. For one run it discovers every
committed shard in canonical order, revalidates each completely (manifest
identity, file metadata and digests, row identities, universe membership,
token/routing links), detects cross-shard identity conflicts, and returns one
`RoutingShardAssignmentQuery` per shard: `shard_key`, `token_count`,
`routing_count`, the validated `token_keys`/`routing_links` identity sets, and
grouped `assignment_counts` sorted by layer and expert key.

The caller owns the engine handle and the bounded in-memory query connection —
including closing it exactly once — so dependency resolution stays lazy and
connection lifecycles stay explicit at every call site; the
`DuckDBRoutingShardStore.query_assignments` port method wraps that lifecycle
for protocol consumers. Budget exhaustion raises `RoutingRunInventoryError`
with stage `budget`; storage-owned failures raise `RoutingShardError`
(`workspace`, `reopen`, `conflict`); query-engine failures raise
`RoutingRunQueryError`; absent or malformed sources raise plain
`TypeError`/`ValueError`/`OSError`. Analysis consumes exactly this seam:
`aggregate_routing_load` opens its connection, delegates discovery, budgets,
validation, conflicts, and grouped reads to the seam, then folds the returned
summaries into the same matrix as before. Results are unchanged on all
previously passing inputs, multi-shard grouped counts are now provably per
shard, and no analysis code touches concrete shard internals any more.

## Tabular run exports

`export_run_tables(workspace, destination, *, run_key, formats=("csv",),
max_event_rows=1_000_000, max_file_bytes=1_000_000_000)` projects one run's
committed shard evidence into open tabular formats under a new destination
directory: `tokens.csv`/`routing.csv` carry every token and routing event in
canonical column order, and `tokens.parquet`/`routing.parquet` repeat the same
projection when `parquet` is requested. Rows are ordered by shard key and
event index; redaction travels faithfully (redacted runs export empty
`token_text` cells with `token_text_stored=false`). CSV members are
byte-deterministic and canonically encoded — every field quoted, `\n`
line endings, UTF-8 — and verification re-encodes them byte-for-byte.
Parquet members are digest-recorded but deliberately not promised
byte-deterministic (Parquet writer metadata varies across engine versions);
they are verified by size, digest, and schema readability instead.

Publication is atomic through an export-staging sibling directory with crash
cleanup, symlink-safe member reads, strict row and per-file byte budgets, and
a canonical digest-bearing `manifest.json` recording store/event schema
versions, writer identity, formats, per-shard counts, and file digests.
`verify_run_tables(source, *, max_event_rows, max_file_bytes)` revalidates a
directory without importing DuckDB: manifest shape and canonicality, member
digests and sizes, CSV canonical re-encoding, row-count agreement with the
manifest totals, and rejection of non-canonical-but-digest-matching members.

The projection is intentionally one-way. Tabular exports are for spreadsheets,
dataframes, and external tools; lossless round-trips stay the run-evidence
bundle's contract, because flat rows cannot restore event contracts, link
validation, or content-addressed shard identity on their own. Failures raise
`RunTableError` with stages `dependency`, `workspace`, `source`, `format`,
`budget`, `write`, and `publish`; receipts (`RunTableReceipt`,
`RunTableFileEntry`) are strict dataclasses carrying the manifest digest and
sorted file entries.

## Expert-event shards (store schema 2.0)

Writers now publish store schema `2.0` (`STORE_SCHEMA_VERSION`): every shard
directory additionally contains `experts.parquet`, and the manifest gains one
declared budget/count field `expert_count`. The physical table carries the
identity columns (`store_schema_version`, `shard_key`, `event_index`,
`schema_version`, `event_type`) plus `token_key`, `expert_key`,
`input_norm`, `output_norm`, `contribution_norm`, `latency_ms`, and a
canonically-encoded JSON `metadata` string (null when empty), all nullable
norm/latency measurements, ZSTD-compressed like the routing tables. Expert
events participate in everything their routing siblings do: file checksums
and sizes are recorded in the manifest and verified on reopen; row identity,
schema identity, finite-float measurement, and canonical-metadata validation
run on reopen; token links are validated (`validate_expert_links`);
`(token_key, expert_key)` pairs join shard conflict/idempotence sets so
overlapping evidence is rejected exactly like overlapping tokens or routes;
and expert rows are folded into the semantic content address, so a changed
measurement changes the shard key and cannot silently replace a committed
shard. `append_structured_shard(workspace, result, *, store_token_text=False,
max_expert_events=65536)` writes structured results carrying expert events;
the plain `append_routing_shard` lane keeps its exact signature and writes
v2 shards with declared `expert_count: 0`.

### Compatibility matrix

| On-disk shard | Reopen | List/inventory | Query seams |
| --- | --- | --- | --- |
| `1.0` (legacy: 3 files, 11-key manifest) | accepted unchanged; rows must carry the legacy version, receipts report `1.0` | counted with zero expert events | assignment queries unchanged; expert queries contribute zero-activity records |
| `2.0` (4 files, 12-key manifest with `expert_count`) | full expert-row validation and digest coverage | expert counts included in event/byte budgets | both assignment and expert queries |

Readers decide by the manifest's `store_schema_version`, never by directory
listing order; unsupported versions, mixed key sets, or version-mismatched
row stamps are reopen failures. The semantic digest is computed against the
shard's own manifest version, which is why genuine v1 shards re-digest to
their historical keys bit-for-bit.
