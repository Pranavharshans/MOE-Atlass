# Local server

The optional `server` layer is a thin, read-only FastAPI application over
the same shared services the CLI uses. It owns no orchestration of its
own: every endpoint delegates to workspace/catalog services or the adapter
registry, carries fixed safe error details that never echo input contents,
and never loads a model, downloads anything, writes to storage, or opens a
network egress path.

FastAPI is an optional dependency (`pip install moeatlas[server]`). The
wire DTOs in `moeatlas.server.dto` stay importable without it;
`create_app()` raises the fixed
`server dependency 'fastapi' is not installed` failure when the extra is
missing.

## Wire surface

All endpoints are GET and read-only:

- `/healthz` — package name/version, Python version, and the honest
  `model_validation_status: deferred` marker from the validation ledger.
- `/api/workspace` — one snapshot of the bound workspace catalog
  (path plus registered-run count); reports the fixed
  `workspace is not initialized` failure until the catalog exists.
- `/api/runs` — bounded run-registry listing with an optional validated
  `state` filter and a `limit` clamped to the factory's `max_results`
  budget (strict positive integer, hard ceiling applied at construction).
- `/api/runs/{run_key}` — one registered run's registry entry fields plus
  its committed routing-shard listing via the bounded store reopen seam;
  unknown keys report the fixed `run is not registered` failure and shard
  storage failures the fixed `run shards are unavailable` failure.
- `/api/runs/{run_key}/summary` — typed routing-load summary response.
  Honest scope: a summary requires a caller-supplied adapter inspection
  document (the same seam as the CLI `heatmap` command), which the server
  does not own and never invents, so every registered run reports
  `status: "unavailable"` with that fixed reason rather than computed data.
- `/api/runs/{run_key}/heatmap` — serves an already-generated static
  heatmap document published at `heatmaps/<run_key>.html` under the
  workspace as `text/html`. The managed directory and candidate file must
  be real non-symlink entries whose canonical location stays inside the
  workspace; anything else reads as absent (`run heatmap is not published`,
  404), so traversal and symlink attacks never widen the served surface.
  Reads are bounded by `max_artifact_bytes` (strict positive integer,
  hard ceiling applied at construction; default 10 MB); oversized
  documents report the fixed byte-budget failure.
- `/api/adapters` — the versioned adapter plugin registry listing with
  provenance, policy status, collisions, and isolated failures.

The application is built with documentation routes disabled
(`docs_url=None`, `redoc_url=None`, `openapi_url=None`) so no unbounded
schema surface is exposed.

## Static frontend

`create_app()` mounts the packaged single-page frontend from
`moeatlas/server/static/` at `/` **after** every API route, so `/healthz`
and `/api/*` always take precedence. The assets are dependency-free vanilla
HTML/CSS/JS: no npm/node build chain, no CDN links, and no external
resources of any kind (the repo stays offline-first). Static responses are
served with `Cache-Control: no-store` so local development always observes
freshly served bytes; API responses are untouched by that header. The mount
is skipped when the static directory is absent (for example in stripped
installations), which leaves the pure-API surface unchanged.

The frontend is hash-routed (`#/workspace`, `#/runs`, `#/runs/{run_key}`,
`#/heatmap[/{run_key}]`) and consumes only the read-only endpoints above.
Every fetch failure renders an inline error banner; heatmap documents are
embedded as-is via an iframe because they are strict-CSP, no-JavaScript
static artifacts.

## Local launch

`moeatlas ui WORKSPACE` serves the bound workspace on
`http://127.0.0.1:8000` by default. Non-loopback hosts require an explicit
`--allow-remote` opt-in; the port must be a canonical positive decimal
integer up to 65535. Missing server dependencies exit 2 with the fixed
install hint. The command never reads a clock, never writes to the
workspace, and stops cleanly on Ctrl-C.

## Honest scope

The dependency-free local single-page UI described above is served by the
same `moeatlas ui WORKSPACE` command; the browser end-to-end tests from the
PRD remain recorded as deferred release-engineering evidence in
`model-validation-ledger.md`. The read-only API above is the contract the
frontend consumes. Nothing here claims UI parity beyond the shipped views.
