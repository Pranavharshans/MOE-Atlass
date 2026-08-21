# Workspace catalog and application services

Status: experimental (model-free)

The PRD requires that a workspace is "a normal directory that can be zipped,
versioned, moved, or deleted by the user." Slice 3 adds the first durable
workspace surface on top of the bounded routing-shard storage: a versioned
catalog manifest, a run registry, and a thin shared service layer that the
CLI, the Python API, and the future local server all call instead of
duplicating orchestration.

## Catalog manifest

The catalog lives at `<workspace>/.moeatlas/catalog.json` as one canonical
JSON document (`WORKSPACE_CATALOG_SCHEMA_VERSION = "1.0"`). It is a
`WorkspaceCatalog` versioned manifest with `created_at`, `updated_at`, and a
run registry of `RunRegistryEntry` values sorted ascending by `run_key`:

| Field | Meaning |
| --- | --- |
| `run_key` | Run identity, using the same stable-identifier vocabulary as shard storage. |
| `specification_fingerprint` | The `RunSpecification` key (`run:<64 lowercase hex>`) this entry was registered from, when known. |
| `state` | A `RunState` value, or `None` for shard-discovered runs with no recorded lifecycle. |
| `attempt` | Lifecycle attempt counter (starts at 1). |
| `shard_count`, `token_event_count`, `routing_event_count` | Observed storage totals, refreshed by rebuild. |
| `token_text_policy` | `redacted`, `stored`, or `mixed`, mirroring the shard inventory. |
| `registered_at`, `updated_at` | Caller-supplied timestamps; the services never read a clock. |

The registry is metadata, not authority: lifecycle truth lives in committed
`RunRecord` manifests and storage truth lives in the shards. The catalog is a
rebuildable index over both.

## Operations

- `initialize_catalog(workspace)` creates `.moeatlas/` (mode 0700,
  symlink-rejecting) and an empty catalog; initializing twice is a `conflict`
  error.
- `read_catalog(workspace)` returns the validated manifest or raises
  `WorkspaceCatalogError("reopen")` mentioning "not initialized" when the
  catalog does not exist. Unknown `schema_version` or manifest type is also a
  reopen-stage error naming the schema mismatch; forward migration is a
  separate future contract.
- `upsert_run_entry(workspace, entry)` merges one registry entry by `run_key`,
  preserving the original `registered_at`, and skips rewriting the file
  entirely when nothing changed. Inserting beyond `max_runs`
  (`CATALOG_MAX_RUNS = 10_000`) is a `conflict`.
- `rebuild_catalog(workspace)` reconciles the registry against committed
  shards through the bounded `list_routing_runs()` inventory: observed runs
  are added or refreshed (counts and token-text policy), pre-registered
  entries keep their fingerprint/state/attempt, entries without shards are
  left untouched, and nothing is ever removed. It returns the new catalog plus
  a frozen `CatalogRebuildReceipt` with sorted `added` / `updated` /
  `unchanged` / `removed` keys and the resulting `run_count`.

Every write is atomic: canonical JSON staged beside the target, fsynced, then
published with an atomic rename and a directory fsync. Failures are reported
as `WorkspaceCatalogError` with a stage from the same vocabulary as storage
(`dependency`, `workspace`, `write`, `publish`, `reopen`, `conflict`) and
leave no partial file behind.

## Storage ports

`moeatlas.store.ports` defines the model-neutral seams over shard storage so
callers depend on protocols rather than the concrete module:

- `RoutingRunReader`: bounded `list_runs(...)`, per-run `list_shards(...)`,
  and validated per-shard assignment summaries via `query_assignments(...)` —
  the same seam analysis consumes.
- `RoutingShardAppender`: `append(result, *, store_token_text=False)`.
- `DuckDBRoutingShardStore.bind(workspace)`: the current adapter implementing
  both protocols by delegating to the canonical shard functions; duckdb is
  imported lazily at call time, never at module import.
- `reader_from_workspace(workspace)`: convenience constructor typed as the
  reader protocol.

## Application services

`moeatlas.services.workspace` composes the catalog and contracts into the
shared orchestration used by every presentation surface:

| Function | Behavior |
| --- | --- |
| `initialize_workspace(workspace, *, at=None)` | Initialize the catalog. |
| `open_workspace(workspace)` | Return a frozen `WorkspaceSnapshot(path, catalog)`. |
| `register_run(workspace, specification, *, at=None)` | Register a planned run from a `RunSpecification`; idempotent per `run_key`. |
| `record_run_record(workspace, record, *, at=None)` | Apply a `RunRecord` state to the registry, preserving counts and registration metadata; unknown runs auto-register. |
| `sync_runs_from_shards(workspace, ...)` | Rebuild the registry from shard storage; returns the receipt. |
| `query_runs(workspace, *, state=None, max_results=100)` | Bounded, catalog-ordered query filtered by `RunState`. |

The layer adds no persistence of its own and no clock access; it validates
inputs, delegates to the catalog and ports, and propagates their stage errors.

## Run-evidence export bundles

Workspaces are also the boundary for relocatable run evidence. The
`moeatlas.store` bundle seams (`export_run_bundle`, `verify_run_bundle`,
`import_run_bundle`) read committed shards from one workspace and publish or
consume open-format evidence bundles elsewhere; see
[storage](storage.md) for the exact format, budgets, redaction, and atomicity
contracts. Importing a bundle into a workspace uses the standard shard
appender, so catalog state can be rebuilt afterwards with
`sync_runs_from_shards`.

## Boundaries

- The catalog is not a query engine, scheduler, or lock manager; concurrent
  writers are out of scope until the durability work below lands.
- Routing-load analysis consumes the storage query seam
  (`query_routing_run_assignments`) rather than concrete shard internals;
  richer catalog-level queries remain future work.
- No CLI or server command exposes the catalog yet; those surfaces arrive
  with the headless product slices.
- Catalog/query/reopen/repair tests use synthetic runs only. Filesystem
  durability, reopen-under-crash, and scale evidence remain `ST-01`–`ST-04`
  deferred rows in the [model-validation ledger](model-validation-ledger.md).
