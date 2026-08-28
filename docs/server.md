# Local server and control plane

The optional `server` layer is a thin FastAPI control plane over the same
loading, discovery, run, storage, and analysis services the CLI uses. It keeps
model work off the request thread behind a bounded in-process job manager,
with progress polling and cooperative cancellation. It is suitable for a
local GPU process or for the same process running inside a provider VM; it
does not open SSH sessions or proxy provider terminals.

Errors exposed to the browser use fixed safe details and never echo prompts,
Hub response bodies, credentials, or local paths. Model/dataset downloads are
performed only after an explicit user request with `allow_downloads: true`.
Each submitted job also gets a bounded JSONL diagnostic record under the
workspace's private `logs/jobs/` directory. Tracebacks are formatted without
local-variable capture and redacted before persistence. The ordinary job
response exposes only a safe diagnostics reference; the
`/api/jobs/{job_id}/diagnostics` endpoint reads records only for a job known to
the running server and returns the bounded sanitized entries. Diagnostic write
failures never change the job outcome or bypass model cleanup.

Discovery and run jobs publish explicit model-download phases for configuration,
tokenizer files, and weight shards. The console polls those phases and renders
the current message plus a progress bar; cached files pass through the same
stages without forcing another download.

FastAPI and the Hugging Face dataset reader are optional dependencies
(`pip install moeatlas[server]`). The server extra includes both because
`POST /api/runs/start` can stream an `hf_datasets` input. The wire DTOs in
`moeatlas.server.dto` stay importable without FastAPI;
`create_app()` raises the fixed
`server dependency 'fastapi' is not installed` failure when the extra is
missing.

## Read and control surface

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
- `/api/runs/{run_key}/summary` — typed routing-load summary response. It
  aggregates the persisted universal or adapter inspection and returns
  unavailable when routing evidence is not complete.
- `/api/runs/{run_key}/architecture` — serves the exact persisted static
  discovery report, including facts, candidates, components, and warnings.
- `/api/runs/{run_key}/activity` — summarizes persisted expert events (active
  cells, event counts, and contribution norms). Routing evidence without
  expert events remains explicitly unavailable rather than being called an
  activation.
- `/api/runs/{run_key}/heatmap?metric=assignment_counts|assignment_shares|load_ratios`
  — renders a bounded heatmap from the persisted matrix when no pre-rendered
  document exists. Published static documents remain symlink-safe and bounded
  by `max_artifact_bytes`.
- `/api/compare/heatmap?baseline_run_key=...&comparison_run_key=...&metric=...`
  — renders count/share/ratio deltas for two runs over the same validated
  routing universe.
- `/api/runs/{run_key}/export?format=bundle|csv|parquet` — creates a bounded
  ZIP download from the immutable run bundle or tabular export. Exports are
  cached under the workspace `exports/` directory and are rejected when the
  run's persisted privacy policy disables export.
- `/api/adapters` — the versioned adapter plugin registry listing with
  provenance, policy status, collisions, and isolated failures.
- `/api/hub/search?kind=model|dataset&q=...&limit=...` — bounded public
  Hugging Face suggestions for the research console. This endpoint is called
  only after a user types a query; it accepts no arbitrary URL or browser
  token, and exact IDs remain usable when public search is unavailable. Gated
  or private revision resolution may use an `HF_TOKEN` or
  `HUGGINGFACE_HUB_TOKEN` already configured in the server process.
- `/api/jobs/{job_id}` — queued/running/completed/cancelled/failed state and
  monotonic progress for discovery and run jobs, plus a safe diagnostics
  reference.
- `/api/jobs/{job_id}/diagnostics` — bounded, structured, sanitized stage and
  exception-chain evidence for a known job. Unknown job IDs return the same
  fixed 404 as the status endpoint.
- `POST /api/discovery` — resolves an HF revision to an immutable commit,
  loads the selected model, and returns a generic static `DiscoveryReport`.
- `POST /api/runs/start` — resolves model and dataset revisions, infers a
  common string prompt column when needed, executes the bounded dataset, and
  publishes routing plus expert evidence. `dataset_seed` deterministically
  selects a capped subset from the immutable dataset revision; the same
  revision, split, config, cap, and seed produce the same row indices for a
  baseline, resume, or derived intervention run. Set `measure_capture_overhead: true`
  to add an optional native, capture-disabled forward pass before the evidence
  run; its forward-only timing report is persisted under
  `benchmarks/capture-overhead/`. Include `resume_job_id` for a cancelled job
  with a durable checkpoint; resumed runs do not repeat the optional pass.
- `POST /api/run-groups/start` — expands one model and capture contract over
  2–16 dataset/config children. Children run sequentially with the same sample
  cap and dataset seed, publish ordinary independently inspectable runs, and
  retain partial results if a later dataset fails.
- `GET /api/run-groups` — lists durable master manifests and their child run
  keys. The master index is stored at `runs/<master>/group.json`, with readable
  dataset pointers under `runs/<master>/datasets/<slug>/run.json`; immutable
  child artifacts continue to use the normal run registry.
- `POST /api/jobs/{job_id}/cancel` — requests cooperative cancellation at a
  safe batch boundary.
- `POST /api/jobs/{job_id}/skip-overhead` — skips only the optional native pass
  while it is in progress and continues the capture run; it does not cancel
  the parent study.
- `POST /api/interventions/recipes` — validates and fingerprints a causal
  intervention recipe.
- `GET /api/runs/{run_key}/intervention-targets` — lists independently
  controllable layer × expert coordinates discovered for a completed run.
- `POST /api/interventions/start` — starts an exact baseline-derived live
  expert ablation or scaling run.
- `GET /api/runs/{run_key}/intervention` — returns paired output, optional task
  score, latency, invocation, and restoration evidence for a derived run.

The application is built with documentation routes disabled
(`docs_url=None`, `redoc_url=None`, `openapi_url=None`) so no unbounded
schema surface is exposed.

## Static frontend

`create_app()` mounts the packaged single-page frontend from
`moeatlas/server/static/` at `/` **after** every API route, so `/healthz`
and `/api/*` always take precedence. Static responses are served with
`Cache-Control: no-store` so local development always observes freshly served
bytes; API responses are untouched by that header. The mount is skipped when
the static directory is absent (for example in stripped installations), which
leaves the pure-API surface unchanged.

The React/TypeScript research console source lives in `frontend/` and uses the
same relative API surface. Its Vite development server proxies `/api` and
`/healthz` to a local `moeatlas ui` process. `npm run build:static` publishes
the hashed React bundle into the package mount; the legacy `/app.js` and
`/styles.css` assets remain available for stripped installations and contract
compatibility, but are no longer loaded by the root page.

The React console keeps its research navigation in the current browser
surface and consumes the relative API endpoints above. Discovery and capture
buttons launch real jobs and poll `/api/jobs`; results expose architecture,
activation status, metric selectors, comparison links, exports, and recipe
validation. Heatmaps remain strict-CSP, no-JavaScript iframe artifacts. The
legacy compatibility bundle remains available when used directly.

## Local launch

`moeatlas ui WORKSPACE` serves the bound workspace on
`http://127.0.0.1:8000` by default. Non-loopback hosts require an explicit
`--allow-remote` opt-in; the port must be a canonical positive decimal
integer up to 65535. Missing server dependencies exit 2 with the fixed
install hint. The server writes only the explicit run/checkpoint/inspection
and export artifacts requested through the control endpoints and stops
cleanly on Ctrl-C.

## Honest scope

The local research console described above is served by the same
`moeatlas ui WORKSPACE` command. Broad browser end-to-end tests and final
VM/GPU certification remain deferred release-engineering evidence in
`model-validation-ledger.md`; this control plane does not claim universal
model-family support until those model-specific runs pass on the target VM.
