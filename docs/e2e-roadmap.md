# End-to-end product roadmap: any-model / any-dataset observability

Status: active roadmap, derived from the 2026-08-22 vast.ai VM validation run
(see `model-validation-ledger.md` for the raw evidence).

## Product goal

A user of MoEAtlas can:

1. Load **any** Mixture-of-Experts model from HuggingFace.
2. See its architecture details: model family, layer count, expert count,
   top-k routing, shared experts, router module paths.
3. Load **any** dataset from HuggingFace (for example a coding dataset).
4. Run the loaded model over dataset rows while hooks capture routing
   decisions and per-layer expert activations.
5. Inspect the results as routing-load heatmaps and per-layer activity
   reports in the local UI.

Everything below is grounded in what was actually executed on the VM with
real checkpoints (`inclusionAI/Ling-3.0-tiny`, `amd/Instella-MoE-16B-A3B-Think`)
and real data (`openai_humaneval` @ `7dce6050`).

## Current state (VM-validated 2026-08-22)

| # | Journey step | Status | Evidence |
| --- | --- | --- | --- |
| 1 | Load any HF model | WORKS* | Real Ling checkpoint loads through the local/HF Transformers path with task-aware `AutoModelForCausalLM` selection; `trust_remote_code` and immutable revisions remain explicit, and 24 GB GPUs need `device=auto`/bf16 placement for this 7.9B checkpoint |
| 2 | Auto-discover arch/experts/routing | WORKS | Generic scanner found exact expert/top-k/shared counts on 3 foreign architectures (Ling 128/8/1, Instella 64/6/2) with zero per-model code |
| 3 | HF dataset → rows → batches | WORKS* | Local snapshots remain deterministic/offline; Hub splits now stream through the optional `datasets` package only with descriptor-level `allow_downloads=true`, config, split, and revision |
| 4 | Run model through dataset rows | WORKS* | Built-in `transformers-routing` executor ran a real Ling dataset row end to end (5 tokens, 920 routing events) and publishes the shard/catalog |
| 5a | Routing capture | WORKS* | Generic structure-driven hooks capture foreign families; real Ling emits `(indices, weights, logits)` and scaled weights are retained without probability claims |
| 5b | Per-layer expert activation capture | WORKS | `run_structured_expert_forward` captures per-invoked-expert norms from real hook plumbing; shards carry `experts.parquet` under store schema 2.0 with reopen/tamper validation; analysis publishes bounded activation summaries (real-checkpoint fidelity: final VM) |
| 6 | Persist events to shards | WORKS* | Real shard/catalog persisted; the executor also stores an immutable universal inspection beside each run (large-run DuckDB write throughput remains unbenchmarked) |
| 7 | Routing-load matrix | WORKS* | `aggregate_routing_load` accepts `UniversalRoutingInspection`; Ling produced a 23-layer × 128-expert matrix with 920 assignments |
| 8 | Heatmap render | WORKS* | Static renderer and server-on-demand path both render the universal matrix; the validated Ling artifact was ~1.4 MB |
| 9 | Local UI/frontend | WORKS* | FastAPI summary and heatmap endpoints now consume persisted universal inspections; the packaged dependency-free SPA serves those views |

## Roadmap

Ordered so each phase lands independently testable value. Every feature must
ship with model-free tests matching this repository's contract-test culture
and keep `pytest -q`, `python -m unittest discover -s tests -t .`, and
`ruff check src tests` green before merge.

### Phase R1 — Universal analysis lane (steps 7–8) — landed 2026-08-22

Unblock visualization for ANY discovered architecture.

- R1.1 Introduce a universal inspection document derived from the static
  `[STRUCTURE]` discovery report (family, routers, expert universe, layout
  digest) that `aggregate_routing_load` accepts alongside certified
  `AdapterInspection` values.
- R1.2 Route the CLI `heatmap` / `compare` commands through the same seam so
  a workspace holding generic-family shards renders real heatmaps.
- Acceptance: a shard produced from a non-certified model (Ling-class events)
  aggregates and renders all three metrics without synthetic fixtures.

### Phase R2 — Real-model executor (step 4) — landed 2026-08-22

Make `moeatlas run` drive a genuine transformers model over planned batches.

- R2.1 Ship a built-in `transformers-routing` executor plugin registered in
  the existing `moeatlas.executors` entry-point group: takes a resolved
  loading plan, tokenizes planned prompt rows, runs batched forwards, and
  emits normalized Token/Routing events through the documented capture path.
- R2.2 Auto-register persisted shards into the workspace catalog after a run
  (remove the manual `rebuild_catalog` step).
- Acceptance: `moeatlas run WS --loading-plan … --dataset … --executor
  transformers-routing` ends with a queryable catalog run; fake-executor
  behavior unchanged.

### Phase R3 — Expert activation capture (step 5b)

Status: landed model-free (2026-08-22); real-checkpoint norm fidelity and
GPU performance claims remain governed by the model-validation ledger.

Turn the `ExpertEvent` contract into runtime reality.

- R3.1 Extend the hook/capture composition to record per-layer expert
  activity (input/output/contribution norms per probe levels 2–3).
- R3.2 Add an expert-event table to the shard schema with manifest/versioned
  contracts, budgets, and tamper semantics mirroring the routing tables.
- R3.3 Surface per-layer activation summaries in the analysis package.
- Acceptance: a real forward produces expert events end-to-end into storage
  with reopen/tamper validation.

### Phase R4 — Serving surface (step 9) — landed 2026-08-22

Expose artifacts over HTTP so the frontend has one API.

- R4.1 Read-only endpoints: run detail (event counts, shard listing),
  matrix summary JSON, heatmap HTML retrieval (generated artifact or
  on-demand render), expert-activation summary.
- R4.2 Static hosting of generated heatmap documents under the workspace.
- Acceptance: every artifact a run produces is fetchable from `moeatlas ui`
  without filesystem access.

### Phase R5 — Frontend application — landed 2026-08-22

The local single-page app consuming R4.

The new React/TypeScript research console is landed under `frontend/` and is
published into the `moeatlas ui` static mount by `npm run build:static`: it
opens directly on model/dataset intake, supports optional public Hub
suggestions, and treats an in-VM/provider HTTP runner as the remote path. It
deliberately has no SSH onboarding page.

- R5.1 Model view: architecture report (family, layers, experts, top-k,
  router paths) straight from the scan document.
- R5.2 Runs view: catalog runs, event counts, state filters (endpoints exist).
- R5.3 Heatmap view: embed/served heatmap HTML plus metric selector and
  comparison mode.
- R5.4 Activation view: per-layer activity charts from R3 summaries.
- Acceptance: the four views work against a real VM-produced workspace;
  packaged assets served by `moeatlas ui`.

### Phase R6 — Robust loading polish (step 1) — remaining hardening

- R6.1 Keep config observation JSON-safe: integer JSON-object keys are
  canonicalized and post-normalization collisions are rejected.
- R6.2 Keep Hub dataset downloads explicit and bounded; immutable revision
  evidence and large-dataset throughput remain follow-up validation work.
- R6.3 Shard writer performance pass for ~1M-event captures (batch inserts)
  with identical byte-level manifests or a versioned schema bump.

### Phase R7 — Generation-bound routing evidence — landed 2026-08-23

- R7.1 Capture routing from the same deterministic `generate()` call that
  produces the persisted output digest; do not run a second prompt-only
  forward and present it as generation evidence.
- R7.2 Persist prompt-processing and answer-generation token phases in one
  ordered run shard. Cached one-token decode and uncached prefix replay both
  retain only newly routed token positions.
- R7.3 Record that the terminal output token has no subsequent model forward,
  and downgrade expert-activity evidence explicitly when generation-time
  expert hooks are unavailable.
- Acceptance: a model-free generate loop proves one call sequence produces
  both the output and the phase-labelled routing events, with all temporary
  hooks removed after success.

### Phase R8 — Exact pairing and task evaluators — landed 2026-08-23

- R8.1 Publish a privacy-preserving digest of the exact tokenized input,
  reference value, run mode, generation budget, and evaluator for every row.
- R8.2 Refuse paired intervention evidence when input digests or evaluator
  identities differ; older baselines without this evidence must be rerun.
- R8.3 Provide bounded built-in evaluators for normalized exact match, token
  F1, reference containment, multiple-choice accuracy, and numeric match.
- R8.4 Expose evaluator selection in the research console and preserve the
  choice in resolved run metadata.
- Acceptance: evaluator golden tests, executor evidence tests, paired-input
  rejection tests, server contracts, and the production frontend build pass.

`WORKS*` means the model-free suite and the real Ling VM path both passed; it
does not claim every architecture/task head or every GPU placement is
certified. The VM result is a capability boundary, not a universal performance
claim.

## Explicitly out of scope here

Certified-family certification (MV rows), GPU performance claims, and
multi-GPU serving remain governed by `model-validation-ledger.md`.
