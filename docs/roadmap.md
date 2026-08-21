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
| 5 | Raw evidence and open storage/export | `model-free complete` | Bounded versioned event/manifest/metric export and import with redaction, migration, tamper, and atomicity tests; analysis consumes reader/query contracts rather than storage internals | PRD §§10 and 16 |
| 6 | Prompt and dataset run engine | `model-free complete` | Incremental deterministic prompt/dataset execution, progress, cancellation, resume, and per-row errors over fake runtimes | PRD §9 |
| 7 | Task association and Evidence Cards | `model-free complete` | Bounded routing, association, behavior, uncertainty, and evidence-tier analyses without unsupported causal claims | PRD §11 and formula tests; real-evidence capture stays in Sequence 12 |
| 8 | Plugins and complete headless CLI/API | `in progress` | Versioned entry-point registry plus PRD scan/run/compare/export/adapters/doctor workflows through shared services | PRD §§15–16 |
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

Sequence 7 (task association and Evidence Cards) is complete only when:

1. Routing-, behavior-, and causal-tier metrics exist with documented
   denominators, null/unsupported behavior, and deterministic ordering.
2. Evidence Cards keep routing, internal behavior, causal, replication,
   uncertainty, limitations, and capability tiers separate.
3. No output claims specialization or causality from association alone.
4. The full local gate is green and the feature commit is pushed.

Known limitation, deferred rather than expanded here: documentation files
(including this roadmap) are repository artifacts and are not packaged into the
wheel/sdist, matching the existing distribution pattern for all `docs/` content.
Shipping docs with distributions is release-engineering work in Sequence 11.

Sequence 6 is `in progress`. Its first landed slice is the bounded dataset
reader (see [runs](runs.md)): `read_dataset_rows` in
`moeatlas.services.datasets` turns `DatasetInputSpec` descriptors into
deterministic frozen `DatasetRow` tuples for JSONL/CSV/Parquet/text and
local HF-style snapshots under strict row/byte/file budgets, validates
task-role column mappings, plans SHA-256-keyed deterministic
`plan_dataset_batches` schedules (sample caps, shuffles, batches), resolves
DuckDB lazily for Parquet only, and never touches the network. Its second
landed slice is the deterministic execution core:
`execute_row_schedule` in `moeatlas.services.run_engine` drives a planned
schedule through a caller-supplied row executor with per-row failure
evidence (fixed vocabulary matching lifecycle error kinds), budgeted
canonical results, cumulative lifecycle-compatible progress, cooperative
cancellation that preserves executed work, and a strict frozen
`ExecutionOutcome` whose status suggests the terminal lifecycle state. Its
third landed slice is input preparation: `prepare_input_rows` /
`plan_input_batches` in `moeatlas.services.run_inputs` turn prompt specs and
dataset descriptors into the exact row-value mappings and deterministic
schedules the core consumes — one bounded prompt row or reader-composed
role-projected dataset rows with descriptor-driven schedules — so execution
never branches on input kind. Its fourth landed slice closes the sequence as
the headless run service (see [runs](runs.md)):
`execute_specification` in `moeatlas.services.run_service` composes
preparation, planning, batch-by-batch execution, and lifecycle projection
into one deterministic surface with caller-supplied timestamps, per-batch
`update_progress` records streamed to an `on_record` observer, terminal
transitions chosen only by the outcome's status, atomic canonical JSON
checkpoints after every completed batch, validated `load_checkpoint` /
`resume_from` continuation that never re-executes durable batches, and
explicit `publish_run_report` catalog publication. Real token/routing event
publication arrives with real executors in later sequences; every behavior
this sequence promises is exercised over fake runtimes.

Sequence 7 is `model-free complete`: every §11 metric surface constructible
without a live model exists with contract tests. Its first landed slice is the association
math of PRD §11.2 (see [analysis](analysis.md)):
`TaskExpertCounts` in `moeatlas.analysis.task_association` is a strict frozen
contingency table of selected-route counts per (layer, task, expert), and
`analyze_task_association` derives enrichment `P(expert|task)/P(expert)`,
PMI/MI in bits (with task-share-consistent specific MI), mean pairwise
Jensen-Shannon separability, and exclusivity/generality per expert — all
deterministic, budget-bounded, `null`-for-undefined, and canonically
serializable as `moeatlas.task_association` artifacts. Its second landed
slice is the Evidence Cards of PRD §11.5 (see [analysis](analysis.md)):
`EvidenceCard` in `moeatlas.analysis.evidence_cards` keeps one expert's
routing, task-association, behavior, causality, and stability evidence in
separate optional sections (`null` means not measured), with capability
labels over a fixed tier vocabulary, str-only limitations/warnings, and
canonical `moeatlas.evidence_card` round-trips — never collapsing tiers into
one score and never claiming specialization or causality from association.
Its third landed slice is prompt-vs-rollout routing agreement (see
[analysis](analysis.md)): `analyze_routing_agreement` in
`moeatlas.analysis.routing_agreement` compares paired prompt-phase and
rollout-phase selection distributions per layer with base-2 Jensen-Shannon
divergence, its bounded agreement complement, and total-variation distance,
canonically serializable as `moeatlas.routing_agreement` artifacts. Its
fourth landed slice is cross-run association stability (see
[analysis](analysis.md)): `analyze_association_stability` in
`moeatlas.analysis.association_stability` compares two runs'
P(expert | task) distributions over one identical (layer, task, expert)
topology with base-2 Jensen-Shannon divergence, its bounded agreement
complement, and per-layer means, canonically serializable as
`moeatlas.association_stability` artifacts — completing the §11.2 metric
set. Its fifth landed slice adds the router margin of PRD §11.1 (see
[analysis](analysis.md)): `analyze_router_margin` in
`moeatlas.analysis.router_margin` summarizes per-layer top1-minus-top2
selected-score differences over caller-supplied ranked score samples, with
explicit defined/total token counts and `null` means where no token has two
scored ranks, canonically serializable as `moeatlas.router_margin` artifacts.
Its sixth landed slice adds route churn (see [analysis](analysis.md)):
`analyze_route_churn` in `moeatlas.analysis.route_churn` measures how
selected-expert sets change across caller-ordered adjacent steps with churn
rates, mean Jaccard distances, and explicit pair counts — canonically
serializable as `moeatlas.route_churn` artifacts. Its seventh landed slice
opens §11.3 with co-routing graphs (see [analysis](analysis.md)):
`summarize_co_routing` in `moeatlas.analysis.corouting` reduces symmetric
per-layer co-selection matrices into total mass, coupled-expert counts, and
deterministically ranked top pairs with normalized shares, canonically
serializable as `moeatlas.corouting` artifacts. Its eighth landed slice adds
expert weight/representation similarity (see [analysis](analysis.md)):
`analyze_expert_similarity` in `moeatlas.analysis.expert_similarity`
derives per-layer cosine-similarity matrices over caller-supplied expert
vectors with exact `1.0` diagonals, explicit `null` cells wherever a
zero-norm expert is touched, and per-layer undefined-expert counts,
canonically serializable as `moeatlas.expert_similarity` artifacts. What
remains deferred to Sequence 12 is real evidence, not contracts: per-token
task-labeled routing from live runs, activation/contribution summaries and
gradient attribution over real tensors, shared-vs-routed comparison on a
real checkpoint, and intervention causality (Sequence 10 supplies the
recipe mechanics first).

Sequence 8 is `in progress`. Its first landed slice is the adapter plugin
registry (see [adapters](adapters.md)): `collect_adapter_registry()` in
`moeatlas.adapters.registry` unifies shipped adapters and
`moeatlas.adapters` entry-point plugins behind one deterministic listing
with provenance records (`AdapterPluginRecord`), trust/enable-disable
policy (`AdapterRegistryPolicy`), collision handling (built-ins first,
then lexicographic entry-point value, every suppressed loser reported),
and failure isolation with a fixed reason vocabulary — canonically
serializable as `moeatlas.adapter_registry` artifacts. Its second landed
slice surfaces the registry as the `moeatlas adapters list` command (see
[cli](cli.md)) with composable trust policy flags (`--builtin-only`,
`--enable`, `--disable`), `--family` capability filtering, `--json`
canonical output, and stderr reporting of collisions and isolated plugin
failures. The remaining PRD CLI flows (run, export, compare over shared
services) are later slices in this sequence.

Sequence 8 (plugins and complete headless CLI/API) is complete only when:

1. A versioned entry-point registry lists built-ins and plugins through
   one contract with provenance, trust policy, collision handling, and
   failure isolation.
2. Every PRD CLI flow (scan, run, compare, export, adapters list, doctor)
   works headlessly through the shared application services.
3. Parser contracts, isolated fake plugins in clean subprocesses,
   resumable synthetic runs, complete export/import round-trips, fixed
   safe errors, and backward compatibility are contract-tested.
4. No command downloads models or touches the network implicitly.
5. The full local gate is green and the feature commit is pushed.

Sequence 6 (prompt and dataset run engine) is complete only when:

1. Prompt/chat and dataset inputs prepare deterministically under budgets
   without branching on input kind.
2. Execution is incremental and deterministic over fake runtimes, with
   progress, cooperative cancellation, and per-row failure evidence.
3. Checkpoints are atomic and resume skips durable batches; reopen of a
   checkpoint is fully validated.
4. The composed headless surface projects runs onto lifecycle records with
   caller-supplied timestamps and explicit catalog publication.
5. The full local gate is green and the feature commit is pushed.

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
