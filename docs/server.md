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
- `/api/adapters` — the versioned adapter plugin registry listing with
  provenance, policy status, collisions, and isolated failures.

The application is built with documentation routes disabled
(`docs_url=None`, `redoc_url=None`, `openapi_url=None`) so no unbounded
schema surface is exposed.

## Local launch

`moeatlas ui WORKSPACE` serves the bound workspace on
`http://127.0.0.1:8000` by default. Non-loopback hosts require an explicit
`--allow-remote` opt-in; the port must be a canonical positive decimal
integer up to 65535. Missing server dependencies exit 2 with the fixed
install hint. The command never reads a clock, never writes to the
workspace, and stops cleanly on Ctrl-C.

## Honest scope

The React/TypeScript single-page UI and browser end-to-end tests from the
PRD are recorded as deferred release-engineering evidence in
`model-validation-ledger.md`; the read-only API above is the contract they
will consume. Nothing here claims UI parity before that work lands.
