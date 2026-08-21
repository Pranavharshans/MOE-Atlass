# PRD delivery roadmap

Status: in progress

The [MoEAtlas PRD](specification/MoEAtlas_PRD_v1.docx) is the product authority.
[Architecture](architecture.md) records implementation boundaries, and the
[model-validation ledger](model-validation-ledger.md) is the sole authority for
real-checkpoint, VM/GPU, target-filesystem, and scale evidence. This roadmap
tracks dependency order and exit criteria without redefining those documents.

## Delivery rules

- Implement one coherent feature slice at a time.
- Run its targeted tests and the complete model-free regression gate before
  committing or pushing it.
- Push each passing feature independently; a failed gate blocks that feature.
- Keep family-specific paths, payloads, and tensor semantics inside adapters and
  decoders. Events, storage, analysis, services, CLI/server APIs, and UI remain
  model-neutral.
- Ordinary development and tests do not contact model hubs or download models,
  tokenizers, datasets, or checkpoints. They do not require a GPU.
- Package tests are model-free evidence, not model certification. Infrastructure
  tests remain `VM/GPU deferred` until their required evidence is recorded in
  the validation ledger.
- The unrelated local `kickbacks-v2.vsix` is never part of a feature commit.

## Status vocabulary

| Status | Meaning |
| --- | --- |
| `planned` | Dependency-ordered scope is defined but implementation has not started. |
| `in progress` | The current feature slice is being implemented and has not passed its gate. |
| `model-free complete` | Local acceptance, regression, lint, and build gates passed and the feature was pushed. |
| `VM/GPU deferred` | Model-free behavior exists, but named ledger evidence is still unavailable. |
| `blocked` | A required non-deferred exit criterion failed or cannot be satisfied. |
| `released` | Model-free and required infrastructure evidence passed the final PRD audit. |

## Dependency-ordered slices

| Sequence | Area | Status | Exit criterion | Evidence authority |
| ---: | --- | --- | --- | --- |
| 1 | Neutral event validation and storage APIs | `model-free complete` | Runtime-independent collection validation; canonical and historical storage functions are exact aliases; persisted behavior unchanged; all local gates pass | Events, runtime, storage docs and tests |
| 2 | Run identity, provenance, and lifecycle | `model-free complete` | Versioned immutable run specs/state/progress/cancellation/lineage with deterministic transition and serialization tests | PRD §9.3 and schema tests |
| 3 | Application services and workspace/catalog | `model-free complete` | Shared CLI/server services, storage ports, migrations, bounded queries, reopen/repair, and v1 shard compatibility | PRD §10; ST-01–ST-04 remain separate |
| 4 | Universal execution capabilities | `in progress` | Family-neutral inspection/input/execution/decoder/universe contracts with unknown-family and non-rectangular synthetic fixtures | PRD §§7–8 and adapter/runtime tests |
| 5 | Raw evidence and open storage/export | `in progress` | Bounded versioned event/manifest/metric export and import with redaction, migration, tamper, and atomicity tests; analysis consumes reader/query contracts rather than storage internals | PRD §§10 and 16 |
| 6 | Prompt and dataset run engine | `planned` | Incremental deterministic prompt/dataset execution, progress, cancellation, resume, and per-row errors over fake runtimes | PRD §9 |
| 7 | Task association and Evidence Cards | `planned` | Bounded routing, association, behavior, uncertainty, and evidence-tier analyses without unsupported causal claims | PRD §11 and formula tests |
| 8 | Plugins and complete headless CLI/API | `planned` | Versioned entry-point registry plus PRD scan/run/compare/export/adapters/doctor workflows through shared services | PRD §§15–16 |
| 9 | FastAPI server and React UI | `planned` | Packaged local UI with scan/run/progress/drill-down/compare/evidence/export flows and synthetic browser tests | PRD §§12–14 |
| 10 | Intervention and causal evidence | `planned` | Bounded recipes, snapshot/restore, cleanup, lineage, cancellation, and causal/stability metrics over synthetic modules | PRD §§11.4 and 13.7 |
| 11 | Privacy, reliability, benchmarks, and release | `planned` | Retention/redaction/plugin trust, CI, clean install, governance, examples, benchmark artifacts, and release docs | PRD §§17–19 |
| 12 | Final VM/GPU certification and PRD audit | `VM/GPU deferred` | MV-01–MV-08 and ST-01–ST-04 have recorded evidence; every v1 acceptance row traces to passing implementation and validation | PRD §20 and validation ledger |

## Current slice exit criteria

Sequences 1–3 are `model-free complete`. Sequence 1 delivered the neutral
event-validation seam and canonical storage APIs (`append_routing_shard`,
`list_routing_shards`, `list_routing_runs` with exact historical aliases);
Sequence 2 delivered the content-addressed run specifications and deterministic
lifecycle contracts in `moeatlas.runs` (see [runs](runs.md)); Sequence 3
delivered the versioned workspace catalog, model-neutral storage ports, and
the shared application-service layer (see [workspace](workspace.md)).

Sequence 4 is `in progress`. Its landed slices cover the adapter-published
`RoutingUniverse` contract (see [adapters](adapters.md)) — per-layer expert
universes, parallel native expert indices, variable top-k schedules,
adapter-declared layout tags, explicit shared-expert semantics — the named
`project_rectangular_universe()` reduction now consumed by
`aggregate_routing_load(declared_universe=...)` so routing-load gates
rectangularity explicitly at its API boundary (see [analysis](analysis.md)),
the model-neutral routing decode capabilities
(`RouterPayloadShape`, `ScoreSemantics`, `RoutingDecodeCapability`,
`validate_decoded_routing`, `native_id_map`) with fake unknown-family
runtimes covering mapping-keyed arrays, assignment-only 3-D payloads, sparse
native identifiers, and variable top-k, and the universal
`run_routing_forward()` execution seam: family-neutral forward composition
over declared `RoutingHookDecoder` capabilities and
`TokenSequencePolicy` token validation, with `validate_observed_routing()`
complete-capture postconditions, an explicit rectangular projection gate at
the capture boundary, and the Mixtral/Qwen3.5 runners preserved as thin
compatibility wrappers while an unknown-family fixture executes end-to-end
through the neutral seam alone (see [runtime](runtime.md)). A provider/task
input-preparation capability protocol remains for later slices in this
sequence, landing with the run engine where it gains a real consumer.

Sequence 4 (universal execution capabilities) is complete only when:

1. Family-neutral capability contracts cover inspection, routing-universe
   publication, input preparation, invocation, decoding, and observation
   without central family branching.
2. Unknown-family and non-rectangular synthetic fixtures pass alongside the
   Mixtral/Qwen regression suites, which remain compatible.
3. Per-layer expert universes/top-k schedules and explicit
   missing/dropped/shared/fallback semantics are contract-tested; rectangular
   analysis becomes an explicit projection.
4. Loading/input preparation follows provider/task capabilities with no
   implicit network access and optional lazy model dependencies preserved.
5. Real checkpoint/native equivalence stays deferred MV evidence; capability
   tests are not certification.
6. The full local gate is green and the feature commit is pushed before
   Sequence 5 starts.

A public structural result protocol, neutral inventory-class rename, or stored
schema migration each changes observable compatibility and requires its own
acceptance gate.

Sequence 5 is `complete`. Its first landed slice is the run-evidence export
bundle (see [storage](storage.md)): `export_run_bundle` /
`verify_run_bundle` / `import_run_bundle` publish one run's complete committed
evidence as canonical JSONL under a digest-bearing manifest with per-shard
redaction fidelity, strict row/byte budgets, byte-deterministic output,
tamper/forged-digest rejection, atomic crash-safe publication, symlink safety,
idempotent source re-import, and duckdb-free verification. Its second landed
slice routes analysis through storage: `aggregate_routing_load` now consumes
the public `query_routing_run_assignments` seam (and the
`RoutingRunReader.query_assignments` port) for source discovery, budgets,
validation, conflict detection, and grouped reads, reaching into no concrete
shard internals; analysis results are unchanged on all previously passing
inputs and multi-shard grouped counts are provably per shard. Its third landed
slice adds the tabular surfaces: `export_run_tables` projects one run into
canonically encoded, byte-deterministic CSV plus optional Parquet under a
digest-bearing canonical manifest with strict budgets, atomic crash-safe
staging, redaction fidelity, and duckdb-free `verify_run_tables`; the
projection is deliberately one-way, keeping lossless round-trips the bundle's
contract.

Sequence 5 (raw evidence and open storage/export) is complete only when:

1. Export/import round-trips preserve content-addressed identities and
   aggregates across workspaces with explicit redaction evidence.
2. Analysis reads runs through the storage ports rather than concrete shard
   internals without changing analysis results.
3. JSON/JSONL/CSV/Parquet surfaces carry canonical schemas, budgets, hashes,
   provenance, relocatability, and schema-version negotiation.
4. Corruption, truncation-as-evidence, limits, crash cleanup, and old-schema
   compatibility are contract-tested on synthetic data.
5. The full local gate is green and the feature commit is pushed.

Known limitation, deferred rather than expanded here: documentation files
(including this roadmap) are repository artifacts and are not packaged into the
wheel/sdist, matching the existing distribution pattern for all `docs/` content.
Shipping docs with distributions is release-engineering work in Sequence 11.

## Validation lanes

- **Model-free package lane:** deterministic contracts, fake runtimes, synthetic
  events, temporary workspaces, CLI/API/server/UI composition, corruption and
  failure injection. This is the default per-feature gate.
- **VM/GPU/model lane:** pinned checkpoints/tokenizers, native routing and passive
  output equivalence, CUDA/device maps, quantized/fused paths, overhead, memory,
  and packaged reruns. These are MV-01–MV-08.
- **Target storage/scale lane:** filesystem durability, reopen/recovery,
  catalog/query integration, and declared scale targets. These are ST-01–ST-04.
- **Release review lane:** official immutable revisions, licenses, compatibility
  tiers, artifacts, examples, and complete PRD traceability are reviewed again
  immediately before release.
