# MoEAtlas

Map. Inspect. Understand.

MoEAtlas is an open-source, local-first observability and experimentation
layer for PyTorch Mixture-of-Experts (MoE) models. It is designed to make
routers, experts, token routes, runtime traces, comparisons, and causal tests
inspectable without turning routing frequency into an unsupported claim of
expert specialization.

The repository is being implemented feature by feature against the
[MoEAtlas product requirements](docs/specification/MoEAtlas_PRD_v1.docx).
The current release contains the repository foundation, canonical manifest
contracts, static discovery/probe contracts, normalized event schemas, and
explicit runtime loading seams. Experimental bounded routing-shard storage and
read-only routing-load analysis are available; the local UI, full
catalog/query surfaces, and real checkpoint/GPU certification remain
deliberately deferred.

## Current status

- Python 3.11+ package named `moeatlas`.
- Model-free `moeatlas doctor` command.
- Pydantic v2 model/component manifests with versioned JSON contracts and
  deterministic portable identity helpers.
- Read-only, model-runtime-independent static MoE discovery with confidence-
  scored candidates, normalized expert/top-k facts, and STRUCTURE manifests.
- Strict serializable probe plans with deterministic target resolution and a
  torch-free transactional hook lifecycle manager.
- Versioned normalized token, routing, and expert event contracts with
  portable identities and capability-driven partial evidence.
- Phase 0 `moeatlas scan fixture:synthetic` command producing a complete,
  deterministic STRUCTURE discovery report, plus explicit resolved HF/local
  plan-file scanning through the runtime bridge; real checkpoint certification
  remains deferred.
- Strict model-source and loading-plan schemas covering HF/local/instance/
  custom intent, offline defaults, policy warnings, and external revision
  evidence without performing loading.
- Validated runtime execution for already-instantiated `InstanceSource` and
  explicitly opted-in `CustomLoaderSource` artifacts, including observed
  manifests and retryable cleanup ownership.
- Lazy HF/local loading through optional Transformers factories after immutable
  resolution, with observed manifests and transactional cleanup. This seam is
  covered only by fake model-runtime tests; real checkpoint/GPU certification
  remains deferred.
- Cleanup-safe `moeatlas.runtime.load_and_scan()` composition for resolved
  HF/local plans, returning the existing static `[STRUCTURE]` discovery report;
  plan-file CLI scanning uses this same bridge; real-model certification
  remains deferred.
- Explicit caller-supplied static semantic-adapter contracts with strict
  descriptor/detection/inspection manifests; adapters remain STRUCTURE-only,
  unregistered, and model-runtime-independent.
- Explicit `MixtralStaticAdapter()` support for the indexed and packed
  structural layouts (including direct packed `gate_up_proj`/`down_proj`),
  with exact family/config/attribute/shape checks and unverified
  STRUCTURE-only provenance; it makes no routing or certification claim and
  is covered only by model-free fixtures.
- Explicit `Qwen3MoeStaticAdapter()` support for the official Qwen3-MoE
  dense/sparse schedule, indexed `legacy_indexed` layouts from
  Transformers 4.51.3/4.57.1, and packed 5.0.0/current logical slices;
  it remains caller-selected, STRUCTURE-only, unverified, and separate from
  Qwen2/Qwen3.5 adapters.
- Pure compilation of retained static `AdapterInspection` evidence into a
  family-neutral reduced `ROUTING` `ProbePlan`; it performs no model calls,
  hook registration, decoding, event capture, storage, or certification.
- Cleanup-safe `RoutingCaptureSession` execution over that canonical plan;
  callers decode opaque synchronous hook payloads into retained, identity-
  bound `RoutingEvent` values while the session enforces quota and lifecycle,
  without a built-in tensor decoder, storage sink, or routing certification.
- Experimental `MixtralRoutingDecoder` support for one-forward caller binding
  of exact `legacy_indexed` logits or packed `(logits, scores, indices)`
  payloads into observed selected `RoutingEvent` values; it performs no token
  inference, tensor retention, model loading, or routing certification.
- `EXPERIMENTAL` `run_mixtral_routing_forward()` support for one caller-supplied
  model forward with caller-tokenized `TokenEvent` rows, shallow-copied kwargs,
  pre-hook complete-event budgeting, exactly-once model execution, and a
  frozen output/event result; tokenization, generation, storage, and UI remain
  outside this boundary. It performs one model forward per invocation.
- Universal `EXPERIMENTAL` `run_routing_forward()` execution: the same
  one-forward boundary composed family-neutrally over a caller-supplied
  declared hook decoder (`RoutingHookDecoder` with `RouterPayloadShape`/
  `ScoreSemantics` capabilities), a `TokenSequencePolicy` token gate, the
  inspection-published routing universe under an explicit rectangular
  projection, and shared `validate_observed_routing()` complete-capture
  postconditions. The Mixtral and Qwen3.5 runners are thin wrappers over this
  seam; unknown families execute through it with their own decoders and no
  central branching. See [runtime](docs/runtime.md).
- `EXPERIMENTAL` bounded routing-shard persistence for complete Feature 18
  results through canonical `append_routing_shard()`, `list_routing_shards()`,
  and `list_routing_runs()` APIs: content-addressed fixed-path manifests plus
  ZSTD token/routing Parquet, explicit token-text redaction, sequential
  idempotence, conflict checks, and non-mutating reopen/list validation. The
  historical `append_mixtral_routing_shard()`,
  `list_mixtral_routing_shards()`, and `list_mixtral_routing_runs()` names remain
  exact identity aliases. This is a shard prerequisite, not a full
  workspace/catalog, query, CLI, server, UI, heatmap, prompt, or
  expert-metric subsystem; see [storage](docs/storage.md).
- `EXPERIMENTAL` bounded run-evidence export bundles: `export_run_bundle()`,
  `verify_run_bundle()`, and `import_run_bundle()` move one run's complete
  committed evidence as byte-deterministic canonical JSONL under a
  digest-bearing manifest — tamper-evident (forged digests still fail
  content-addressed identity recomputation), redaction-faithful, atomic,
  symlink-safe, and relocatable across workspaces with idempotent re-import;
  see [storage](docs/storage.md).
- `EXPERIMENTAL` routing-run assignment query seam:
  `query_routing_run_assignments()` returns validated per-shard summaries —
  identity sets plus grouped assignment counts — under strict budgets with
  typed error carriers, and `aggregate_routing_load()` consumes exactly that
  seam instead of concrete shard internals; see [storage](docs/storage.md).
- `EXPERIMENTAL` bounded tabular run exports: `export_run_tables()` projects
  one run's committed evidence into canonically encoded, byte-deterministic
  CSV plus optional Parquet under a digest-bearing manifest, with strict
  budgets, atomic publication, redaction fidelity, and duckdb-free
  `verify_run_tables()`; the projection is one-way — lossless round-trips
  stay the export bundle's contract; see [storage](docs/storage.md).
- `EXPERIMENTAL` bounded dataset reading: `read_dataset_rows()` turns
  `DatasetInputSpec` descriptors into deterministic frozen rows for
  JSONL/CSV/Parquet/text, local HF-style snapshots, and explicitly opted-in
  streamed Hub splits under strict budgets, with task-role column mappings and
  SHA-256-keyed deterministic `plan_dataset_batches()` schedules; see
  [runs](docs/runs.md).
- `EXPERIMENTAL` deterministic run-engine execution core:
  `execute_row_schedule()` drives a planned batch schedule through a
  caller-supplied row executor with per-row failure evidence, budgeted
  canonical results, lifecycle-compatible progress, cooperative
  cancellation, and a strict frozen `ExecutionOutcome`; no clocks, no
  randomness, no network, no model dependencies; see [runs](docs/runs.md).
- `EXPERIMENTAL` run input preparation: `prepare_input_rows()` and
  `plan_input_batches()` turn prompt specs and dataset descriptors into the
  exact row-value mappings and deterministic schedules the execution core
  consumes — one bounded prompt row or role-projected dataset rows under
  descriptor-driven schedules; see [runs](docs/runs.md).
- `EXPERIMENTAL` headless run service: `execute_specification()` composes
  preparation, planning, batch-by-batch execution, and lifecycle projection
  with caller-supplied timestamps, per-batch progress records streamed to an
  `on_record` observer, atomic canonical JSON checkpoints after every
  completed batch, validated `load_checkpoint()`/`resume_from` continuation,
  and explicit `publish_run_report()` catalog publication; fake-runtime tests
  only; see [runs](docs/runs.md).
- `EXPERIMENTAL` task association metrics: `analyze_task_association()` turns
  a strict per-(layer, task, expert) `TaskExpertCounts` table into enrichment
  `P(expert|task)/P(expert)`, PMI/MI in bits, Jensen-Shannon task
  separability, and exclusivity/generality — deterministic, budget-bounded,
  `null`-for-undefined, canonically serializable; association is never
  specialization or causality; see [analysis](docs/analysis.md).
- `EXPERIMENTAL` Evidence Cards: `EvidenceCard()` keeps one expert's routing,
  task-association, behavior, causality, and stability evidence in separate
  optional sections (`null` means not measured) with capability labels over a
  fixed tier vocabulary and honest limitations/warnings — canonically
  serializable as `moeatlas.evidence_card` artifacts; see
  [analysis](docs/analysis.md).
- `EXPERIMENTAL` prompt-vs-rollout routing agreement:
  `analyze_routing_agreement()` compares paired prompt-phase and
  rollout-phase selection distributions per layer with base-2 Jensen-Shannon
  divergence, its bounded agreement complement, and total-variation distance
  — deterministic, budget-bounded, canonically serializable; see
  [analysis](docs/analysis.md).
- `EXPERIMENTAL` cross-run association stability:
  `analyze_association_stability()` compares two runs' P(expert | task)
  distributions over one identical (layer, task, expert) topology with
  base-2 Jensen-Shannon divergence, its bounded agreement complement, and
  per-layer means — deterministic, budget-bounded, canonically serializable;
  see [analysis](docs/analysis.md).
- `EXPERIMENTAL` router margin: `analyze_router_margin()` summarizes
  per-layer top1-minus-top2 selected-score differences over caller-supplied
  ranked score samples, with explicit defined/total token counts and `null`
  means where no token has two scored ranks; see
  [analysis](docs/analysis.md).
- `EXPERIMENTAL` route churn: `analyze_route_churn()` measures how
  selected-expert sets change across caller-ordered adjacent steps with
  churn rates, mean Jaccard distances, and explicit pair counts; see
  [analysis](docs/analysis.md).
- `EXPERIMENTAL` co-routing graphs: `summarize_co_routing()` reduces
  symmetric per-layer expert co-selection matrices into total mass,
  coupled-expert counts, and deterministically ranked top pairs with
  normalized shares; see [analysis](docs/analysis.md).
- `EXPERIMENTAL` expert similarity: `analyze_expert_similarity()` derives
  per-layer cosine-similarity matrices over caller-supplied expert vectors,
  with exact `1.0` diagonals, explicit `null` cells wherever a zero-norm
  expert is touched, and per-layer undefined-expert counts; see
  [analysis](docs/analysis.md).
- `EXPERIMENTAL` bounded `aggregate_routing_load()` analysis over one run's
  complete Feature 19 shards, using the exact inspection-published routed
  layer/expert universe for Mixtral, Qwen3.5, or a future adapter and strict
  source/row/cell budgets. Shared experts are validated as structural metadata
  and excluded from routed axes. The historical
  `aggregate_mixtral_routing_load()` name is an identity alias. It returns only
  value-owned count/share/load-ratio matrices; it does not infer axes, write
  analysis output, expose raw rows, or claim specialization. See
  [analysis](docs/analysis.md).
- `EXPERIMENTAL` dependency-free `render_routing_load_heatmap()` output
  over one accepted routing-load matrix, with exact metric/cell validation,
  complete accessible zero-inclusive HTML tables, deterministic global heat
  bins, frozen provenance, and no JavaScript, external resource, storage, or
  model boundary. The historical `render_mixtral_routing_load_heatmap()` name
  is an identity alias. See [visualization](docs/visualization.md).
- `EXPERIMENTAL` Feature 22 bounded `moeatlas heatmap WORKSPACE` CLI composition over a
  caller-supplied inspection document and stored run. It preflights output,
  enforces canonical decimal byte/row/source/cell budgets, delegates aggregate
  and render exactly once, and reuses the existing atomic writer. See
  [CLI](docs/cli.md).
- `EXPERIMENTAL` Feature 23 bounded `moeatlas routing-runs WORKSPACE` inventory
  over committed Feature 19 shards. Required budgets preserve exact run/shard
  ordering, counts, source bytes, and redacted/stored/mixed token-text policy;
  no latest-run selector, catalog, raw-row export, model, cache, or inference
  path is created. See [storage](docs/storage.md) and [CLI](docs/cli.md).
- `EXPERIMENTAL` Feature 29 bounded `compare_routing_load()` over two accepted
  routing-load matrices with one identical universe and distinct run keys. It
  returns only value-owned count/share/ratio delta matrices whose rows sum to
  zero; it performs no I/O, rendering, ranking, or specialization claim. See
  [analysis](docs/analysis.md).
- `EXPERIMENTAL` dependency-free `render_routing_load_comparison()` output over
  one accepted comparison, with exact metric/cell validation, signed
  deterministic cold/zero/heat bins, both runs' shard provenance, complete
  accessible tables, and no JavaScript, external resource, storage, or model
  boundary. See [visualization](docs/visualization.md).
- `EXPERIMENTAL` Feature 31 bounded `moeatlas compare WORKSPACE` CLI
  composition over a caller-supplied inspection document and two stored runs.
  It rejects equal run keys first, enforces canonical decimal byte/row/source/
  cell budgets, delegates aggregate/compare/render exactly once per seam, and
  reuses the existing atomic writer. See [CLI](docs/cli.md).
- `EXPERIMENTAL` Feature 32 bounded `summarize_routing_load()` descriptive
  statistics over one accepted matrix: per-layer Shannon entropy, normalized
  entropy, effective expert counts, normalized diversity, exact integer-rank
  Gini, population CV, top-expert shares, and dead-expert counts/fractions as
  a frozen range-checked value; no margin/churn/specialization claim. See
  [analysis](docs/analysis.md).
- `EXPERIMENTAL` Feature 33 canonical JSON export/import for matrices,
  comparisons, and summaries: byte-deterministic sorted documents with
  artifact-type markers, strict staged import validation, and exact round-trip
  equality. See [analysis](docs/analysis.md).
- `EXPERIMENTAL` Feature 34 `write_analysis_bundle()` publication of one
  coherent directory of canonical artifact documents plus a digest-bearing
  manifest, atomically per file with full failure cleanup and a frozen
  receipt. See [analysis](docs/analysis.md).
- Model-neutral run contracts: content-addressed `RunSpecification` manifests
  binding resolved model/tokenizer revisions, probe plans, prompt/dataset
  fingerprints, generation settings, privacy policy, and intervention lineage
  into a deterministic `run:<hex>` key, plus the frozen `RunRecord` lifecycle
  state machine (planned → provisioning → running → finalizing → completed,
  with fail/cancel/retry paths) over pure serializable transitions. These
  contracts execute nothing; the run engine and workspace catalog arrive in
  later slices. See [runs](docs/runs.md).
- `EXPERIMENTAL` versioned workspace catalog and shared application services:
  a canonical `.moeatlas/catalog.json` run registry with atomic publication,
  idempotent upserts, bounded rebuild-from-shards repair, and reopen/conflict
  stage errors; model-neutral storage ports (`RoutingRunReader`,
  `RoutingShardAppender`, `DuckDBRoutingShardStore`) over the shard
  implementation; and the `moeatlas.services.workspace` orchestration layer
  (initialize/open/register/record/sync/query) that the CLI, Python API, and
  future server share. It is not yet a query engine, lock manager, or CLI
  surface; see [workspace](docs/workspace.md).
- `EXPERIMENTAL` adapter-published `RoutingUniverse` contracts: a versioned,
  family-blind per-layer routing topology (expert universes, parallel native
  expert indices, variable top-k schedules, shared experts, adapter-declared
  layout tags) with `publish_routing_universe()` from any conforming
  inspection and `project_rectangular_universe()` as the explicit rectangular
  reduction for legacy analysis; `aggregate_routing_load()` accepts a declared
  universe that must match the publication and pass the projection before any
  shard work. Non-rectangular and unknown-family shapes are first-class.
  See [adapters](docs/adapters.md) and [analysis](docs/analysis.md).
- `EXPERIMENTAL` model-neutral routing decode capabilities: adapters declare
  a payload shape and score semantics and decode native router payloads into
  canonical events, while `validate_decoded_routing()` enforces shared
  postconditions (complete variable top-k rank schedules, universe-bound
  experts, semantics-agreeing score columns) with no central family branch.
  Fake unknown-family runtimes cover dict arrays, assignment-only 3-D
  payloads, sparse native IDs, and variable top-k.
- `EXPERIMENTAL` adapter plugin registry: `collect_adapter_registry()` lists
  shipped adapters and `moeatlas.adapters` entry-point plugins through one
  deterministic contract with provenance records, trust/enable-disable
  policy, collision handling that reports suppressed losers, and failure
  isolation with fixed reason vocabulary; `match_adapters_for_family()` is
  the capability-negotiation seam, and `moeatlas adapters list` exposes the
  listing on the CLI with `--json`, policy flags, and `--family` filtering.
  See [adapters](docs/adapters.md) and [cli](docs/cli.md).
- `EXPERIMENTAL` headless CLI run flow: `moeatlas run WORKSPACE` turns one
  validated loading plan plus exactly one input form (`--prompt TEXT` or
  `--dataset DESCRIPTOR.json`) into a content-addressed `RunSpecification`
  executed through the shared run service with an explicitly registered
  executor plugin — mandatory, never built in, never downloading a model —
  with checkpoints, resume, caller-supplied timestamps, and workspace-catalog
  publication; `moeatlas export WORKSPACE RUN_KEY` publishes the canonical
  tamper-evident run evidence bundle. See [cli](docs/cli.md).
- `EXPERIMENTAL` local read-only server: `moeatlas.server.create_app()`
  binds one workspace behind strict budgets and serves `/healthz`,
  `/api/workspace`, bounded `/api/runs`, and `/api/adapters` — a thin wire
  layer over the same shared services, with fixed safe errors; FastAPI is
  an optional extra (`pip install moeatlas[server]`), and
  `moeatlas ui WORKSPACE` launches it loopback-by-default with an explicit
  `--allow-remote` opt-in. The packaged React/TypeScript research console is
  published with `npm run build:static`; it opens directly on model/dataset
  intake and uses an optional provider HTTP/in-VM runner, with no SSH
  onboarding page. See [server](docs/server.md).
- `EXPERIMENTAL` causal intervention mechanics: `moeatlas.interventions`
  provides immutable, content-addressed recipes over the fixed
  `ablate`/`scale`/`reroute`/`alter_router` vocabulary, immutable budgets,
  and the failure-safe `run_intervention()` engine that guarantees module
  restoration on every path behind adapter-supplied capabilities.
  The local server can run real baseline-derived expert ablation/scaling for
  independently exposed expert modules and publish paired output, score,
  latency, invocation, routing, and restoration evidence. Fused or packed
  expert paths remain explicitly unsupported, and GPU/checkpoint claims still
  require validation. See [interventions](docs/interventions.md).
  See [runtime](docs/runtime.md).
- Model-free test harness for the foundation and schemas.
- Real PyTorch, Transformers, and checkpoint/GPU fidelity explicitly deferred
  to the final VM phase.
- No model files are downloaded by the repository tests or setup.

Check the [model-validation ledger](docs/model-validation-ledger.md) before
interpreting a local test result as model compatibility evidence. The final
[PRD acceptance audit](docs/prd-audit.md) traces every v1 acceptance area to
its implementation and evidence, and lists exactly which claims stay
deferred or blocked until the VM phase runs.

The experimental `MixtralRoutingDecoder` consumes an exact fresh
`AdapterInspection` and non-empty ordered `TokenEvent` tuple. It binds each
router's key/path, same-layer contiguous experts, count, and top-k, and allows
one successful invocation per router path. Legacy payloads emit observed
selected logits only; packed payloads additionally emit native weights after
strict softmax/top-k/renormalization checks. Its evidence remains
`EXPERIMENTAL`; native equivalence and runtime/GPU validation stay deferred to
MV-03 through MV-08.

The bounded `moeatlas heatmap WORKSPACE` command is the CLI publication path
for an existing inspection JSON and Feature 19 run. It reads the inspection
under a strict byte budget, applies row/source/cell budgets, delegates Feature
20 aggregation and Feature 21 rendering exactly once, and reuses atomic publication.
It requires a caller-created `inspection.to_json()` document, the optional
DuckDB `store` extra, and exact lowercase `.html` output; existing files use
`--force`, while failed atomic publication leaves no partial artifact. It does
not tokenize, load models, inspect caches, use a browser or network, or change
MV-01 through MV-08; see [CLI](docs/cli.md).

The bounded `moeatlas routing-runs WORKSPACE` command rebuilds deterministic
JSON inventory from committed shards with required run, shard, event-row, and
source-byte budgets. It is read-only, lazy with respect to the optional DuckDB
`store` extra, atomic only when an explicit `.json` output is requested, and
does not synthesize unavailable model, adapter, layout, inspection, timestamp,
or status metadata. Use the explicit `run_key` with the Feature 20/21 heatmap;
inventory is a rebuild primitive rather than a general run registry. It does
not change MV-01 through MV-08; see [storage](docs/storage.md).

## Quick start

Using `uv`:

```bash
uv sync --extra dev
uv run --locked moeatlas doctor
uv run --locked moeatlas doctor --json
```

The foundation runtime uses Pydantic v2 for strict manifest validation. The
optional `model` extra contains dependencies for the explicit lazy HF/local
runtime calls; it is intentionally not installed by the commands above.

Run the model-free tests without downloading a checkpoint:

```bash
uv run --locked pytest -q
uv run --locked ruff check src tests
```

Run the deterministic Phase 0 semantic scan:

```bash
uv run --locked moeatlas scan fixture:synthetic
uv run --locked moeatlas scan fixture:synthetic --output report.json
uv run --locked moeatlas scan --loading-plan plan.json --output report.json
```

Direct positional CLI scans support only `fixture:synthetic`. An explicit
`--loading-plan plan.json` must contain a fully validated, immutable HF/local
`LoadingPlan`; the CLI passes it unchanged to the runtime bridge and never
infers or resolves a source. Cache inspection, real checkpoint validation, and
GPU certification remain deferred to the final VM workflow.

The loading contracts are schema-only and intentionally do not load a model;
the optional runtime loader is a separate explicit call that requires the
`model` extra and immutable resolution evidence. See [loading contracts](docs/loading.md)
and [runtime loading](docs/runtime.md) before using it.

The standard-library test discovery command is also available inside the
locked environment:

```bash
uv run --locked python -m unittest discover -s tests -t . -v
```

## Planned product shape

The implementation will remain a coherent modular monolith initially. Internal
boundaries will follow the PRD—core schemas, discovery, probing, runtime,
analysis, storage, server, adapters, and CLI—before any boundary is promoted
to a separately distributable package. This keeps early changes easy to test
and lets real model behavior guide the abstractions.

The dependency-ordered [delivery roadmap](docs/roadmap.md) links the PRD,
[architecture](docs/architecture.md), and deferred validation evidence. The
intended progression is:

1. repository foundation and model-free contracts;
2. probe core with static scanner, manifests, hooks, event schema, and a
   synthetic fixture;
3. useful alpha with small real-model adapters and routing storage/UI;
4. research and causal workflows;
5. certified compatibility, plugin SDK, packaging, and benchmarks.

MoEAtlas will not ship as a Mixtral-only tool. Mixtral remains the reference
regression family, while current Qwen MoE, DeepSeek MoE, and MiniMax MoE are v1
end-to-end compatibility requirements. Unknown architectures retain a generic
static-discovery fallback. Family tensor layouts stay inside adapters and
decoders; normalized events, storage, analysis, APIs, and UI stay model-neutral.
No family is called certified until an exact immutable revision passes loading,
inspection, routing capture/decoding, persistence, and visualization on the
final GPU VM. The compatibility matrix is reviewed against official revisions
again at release time rather than relying on a moving “latest” label.

## Evidence and validation policy

MoEAtlas separates routing association, internal behavior, causal effect, and
replication evidence. A heatmap is not by itself a specialization claim.
Likewise, package tests are not model certification. Real checkpoint and GPU
validation will run only in the final explicitly provisioned VM, with model
revisions, hardware, commands, and outcomes recorded in the ledger.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and
[docs/development.md](docs/development.md) for the model-free test contract,
feature workflow, and validation boundary. The project is licensed under the
[Apache License 2.0](LICENSE); model and dataset licenses remain external.

Runtime also provides an EXPERIMENTAL bounded plain-text Mixtral prompt-prefill
wrapper. It borrows a validated `LoadedModel`, enforces caller budgets, and
delegates exactly one passive routing forward; tokenizer and checkpoint fidelity
remain deferred to the final VM.

The model-free composition is explicit and manual: call
prefill once, append its returned model-neutral `RoutingForwardResult` with
`append_routing_shard(...)`, rebuild the read-only run inventory with
`list_routing_runs(...)`, aggregate a selected run with
`aggregate_routing_load(...)`, and finally pass that matrix to
`render_routing_load_heatmap(...)`. Historical Mixtral storage and analysis
names are exact identity aliases. Prefill does not append, inventory,
aggregate, render, or expose a server/wire/progress surface on the caller's
behalf; each later action remains independently bounded and auditable.
An explicit Qwen3.5-MoE static adapter covers the current
`qwen3_5_moe`/`qwen3_5_moe_text` conditional and text-only packed structure
surfaces. Shared experts remain non-routed metadata; the native gate tuple is
`(router_logits, router_scores, router_indices)`. Feature 26 routing decoding
is complete at the model-free boundary; current-checkpoint/GPU certification
remains deferred to the final VM and release-time official-revision review.
The model-free Qwen3.5 routing decoder validates packed router logits, scores,
and indices, excludes shared experts, and emits normalized routing events. It
does not certify checkpoint or GPU runtime behavior; that remains a final-VM
and release-time revision-review task.
An EXPERIMENTAL Qwen3.5 one-forward boundary captures complete routing evidence
through the neutral runtime result and existing storage-compatible event schema.
Feature 27 downstream now composes through append, reopen, run inventory,
neutral aggregation, and neutral visualization for any inspection whose
complete routed universe passes structural validation. Feature 28 does not
change the stored schema or the Mixtral HTML bytes.
