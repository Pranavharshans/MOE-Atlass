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
| 1 | Load any HF model | PARTIAL | Universal load works; Ling required upstream-code shims (transformers 4.45-era remote code on transformers 5.15); `load_instance` rejects live `PretrainedConfig` objects (accepts plain config.json mappings) |
| 2 | Auto-discover arch/experts/routing | WORKS | Generic scanner found exact expert/top-k/shared counts on 3 foreign architectures (Ling 128/8/1, Instella 64/6/2) with zero per-model code |
| 3 | HF dataset → rows → batches | WORKS | HumanEval read through `services.datasets` + `run_inputs` planning (164 rows → deterministic batches); hub download itself is caller-owned by contract |
| 4 | Run model through dataset rows | MISSING | No real-model executor exists; `moeatlas.executors` entry-point group is empty and only fake executors are wired |
| 5a | Routing capture | PARTIAL | Real forwards captured via caller-owned hooks because `RoutingCaptureSession` requires certified-family `AdapterInspection` |
| 5b | Per-layer expert activation capture | WORKS | `run_structured_expert_forward` captures per-invoked-expert norms from real hook plumbing; shards carry `experts.parquet` under store schema 2.0 with reopen/tamper validation; analysis publishes bounded activation summaries (real-checkpoint fidelity: final VM) |
| 6 | Persist events to shards | WORKS | Real shard written (29,440 routing events); slow at ~1M-event scale (DuckDB row-by-row executemany); catalog sync is a manual second call |
| 7 | Routing-load matrix | BLOCKED BY DESIGN | `aggregate_routing_load` demands an exact certified-family `AdapterInspection`; universal scan reports cannot enter analysis |
| 8 | Heatmap render | BLOCKED BY DESIGN | Same inspection gate upstream; renderer itself is proven |
| 9 | Local UI/frontend | PARTIAL | FastAPI server serves `/healthz`, `/api/workspace`, `/api/runs`, `/api/adapters`; no heatmap/artifact endpoints; React SPA does not exist in-repo |

## Roadmap

Ordered so each phase lands independently testable value. Every feature must
ship with model-free tests matching this repository's contract-test culture
and keep `pytest -q`, `python -m unittest discover -s tests -t .`, and
`ruff check src tests` green before merge.

### Phase R1 — Universal analysis lane (steps 7–8)

Unblock visualization for ANY discovered architecture.

- R1.1 Introduce a universal inspection document derived from the static
  `[STRUCTURE]` discovery report (family, routers, expert universe, layout
  digest) that `aggregate_routing_load` accepts alongside certified
  `AdapterInspection` values.
- R1.2 Route the CLI `heatmap` / `compare` commands through the same seam so
  a workspace holding generic-family shards renders real heatmaps.
- Acceptance: a shard produced from a non-certified model (Ling-class events)
  aggregates and renders all three metrics without synthetic fixtures.

### Phase R2 — Real-model executor (step 4)

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

### Phase R4 — Serving surface (step 9)

Expose artifacts over HTTP so the frontend has one API.

- R4.1 Read-only endpoints: run detail (event counts, shard listing),
  matrix summary JSON, heatmap HTML retrieval (generated artifact or
  on-demand render), expert-activation summary.
- R4.2 Static hosting of generated heatmap documents under the workspace.
- Acceptance: every artifact a run produces is fetchable from `moeatlas ui`
  without filesystem access.

### Phase R5 — Frontend application

The local single-page app consuming R4.

- R5.1 Model view: architecture report (family, layers, experts, top-k,
  router paths) straight from the scan document.
- R5.2 Runs view: catalog runs, event counts, state filters (endpoints exist).
- R5.3 Heatmap view: embed/served heatmap HTML plus metric selector and
  comparison mode.
- R5.4 Activation view: per-layer activity charts from R3 summaries.
- Acceptance: the four views work against a real VM-produced workspace;
  packaged assets served by `moeatlas ui`.

### Phase R6 — Robust loading polish (step 1)

- R6.1 Accept live `PretrainedConfig` objects in `load_instance`
  (`config.id2label object keys must be strings` bug).
- R6.2 Document (and where cheap, automate) the HF download step feeding
  dataset descriptors, keeping the descriptors themselves network-free.
- R6.3 Shard writer performance pass for ~1M-event captures (batch inserts)
  with identical byte-level manifests or a versioned schema bump.

## Explicitly out of scope here

Certified-family certification (MV rows), GPU performance claims, and
multi-GPU serving remain governed by `model-validation-ledger.md`.
