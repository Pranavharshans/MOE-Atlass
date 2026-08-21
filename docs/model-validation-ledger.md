# Model and GPU validation ledger

**Status: deferred** — this ledger is intentionally opened during the
foundation feature and will be completed in the final VM phase.

No model files are downloaded by the repository setup or model-free test
commands. A package being importable is not model validation. A passing CPU
unit test is not GPU compatibility evidence.

Feature 9 adds lazy HF/local execution with fake optional modules in the
model-free suite. Those tests verify call arguments, observation, and rollback
only; they do not change MV-01/MV-02 status and do not inspect caches or fetch
checkpoints.

Feature 10 adds `runtime.load_and_scan()` as a cleanup-safe composition of the
resolved loader and static discovery. Its model-free tests verify dispatch,
identity binding, report validation, and retryable cleanup only; it does not
certify a real checkpoint or change the deferred MV-01/MV-02 status.

Feature 11 adds the plan-file CLI entrypoint. Its model-free tests verify
strict plan parsing, source/resolution preflight, unchanged delegation, and
publication safety only; it does not resolve or certify a real checkpoint and
does not change the deferred MV-01/MV-02 status.

Feature 12 adds the caller-supplied static semantic-adapter protocol. Its
model-free tests verify strict manifests, identity binding, STRUCTURE-only
provenance, safe error boundaries, and no-runtime-action behavior only; it does
not certify any architecture or change the deferred MV-01/MV-02 status.

Feature 13 adds the explicitly caller-selected `MixtralStaticAdapter()` for
two model-free structural surfaces: official Transformers 4.50 indexed
experts and current packed direct `gate_up_proj`/`down_proj` tensors. Its tests use standard-library
fixtures only and verify exact family/config/topology/shape evidence,
STRUCTURE-only unverified provenance, and safe rejection of Qwen, dense,
fused, and malformed surfaces. It does not infer a Transformers version,
observe routing, certify Mixtral behavior, or change the deferred MV-01/MV-02
status; real checkpoints and VM/GPU evidence remain required.

Feature 14 adds the explicitly caller-selected `Qwen3MoeStaticAdapter()`
for the official Qwen3-MoE dense/sparse schedule and the indexed
`legacy_indexed` and packed reference surfaces from the documented
Transformers versions. Its model-free fixtures verify exact family markers,
schedule, structural attributes, topology, shapes, safe failures, and
unverified `[STRUCTURE]` provenance only. Qwen2/Qwen3.5 layouts, routing,
real checkpoints, and VM/GPU evidence remain deferred; this feature does
not change MV-01/MV-02 status.

Feature 15 adds the pure adapter-inspection-to-routing-plan compiler. Its
model-free tests verify fresh inspection revalidation, exact router targets,
deterministic reduced `ROUTING` intent, safe errors, and no runtime/model
actions. It does not execute hooks, decode routing, create events, write
storage, or impose an execution/event/storage bound; native routing payload
equivalence and capture remain deferred to MV-03. It does not certify routing
or change the deferred MV-01/MV-02 status.

Feature 16 adds the cleanup-safe `RoutingCaptureSession`. Its model-free
tests verify canonical inspection/plan preflight, exact router contexts,
caller-owned opaque hook decoding, identity-bound `RoutingEvent` validation,
retained-event quota, and retryable hook cleanup. The session has no built-in
tensor/tuple decoder, storage sink, or certification claim; detaching and
reducing payloads remain caller responsibilities. Native routing equivalence,
passive output fidelity, and overhead remain deferred to MV-03, MV-04, and
MV-05; this feature does not change MV-01/MV-02 status.

Feature 17 adds the experimental `MixtralRoutingDecoder` as an explicit
one-forward caller decoder over fresh `AdapterInspection` and `TokenEvent`
values. Model-free tests cover exact Mixtral descriptor/layout binding,
legacy-indexed logits, packed logits/scores/indices, strict tensor-like
conversion order, deterministic top-k/tie rejection, score
softmax/top-k/renormalization cross-checks, context identity, single-use
router invocation, and integration through `RoutingCaptureSession` for both
layouts. The decoder does not load a model, download/cache artifacts, infer
tokens, invoke a tokenizer/runner/generation path, retain tensors, write
storage, or claim routing certification. No model files are downloaded; MV-03
through MV-08 remain `deferred`, and this feature does not change MV-01/MV-02
status.

Feature 18 adds the experimental `run_mixtral_routing_forward()` one-forward
wrapper and frozen model-neutral `RoutingForwardResult` (historical Mixtral
identity alias). Model-free tests verify
canonical preflight, caller-tokenized row binding, shallow keyword copying,
pre-hook complete-event budgeting, exactly one model call, legacy/packed
capture, fresh result events, output identity/ownership, no partial
publication, and authoritative hook cleanup for ordinary and control-flow
failures. It is not tokenization, prompt/generation, dataset, storage, CLI,
server, or UI infrastructure; no model files are downloaded and MV-03 through
MV-08 remain `deferred`. This feature does not change MV-01/MV-02 status.

Feature 19 adds the experimental bounded routing-shard prerequisite. Its
model-free DuckDB tests verify exact preflight, content-addressed fixed-path
layout, redaction, ordered nullable ZSTD Parquet schemas, exact row identities
and values, manifest/checksum and byte-tamper rejection, symlink/mode safety,
reopen/list validation, idempotence, conflict rejection, and safe staged
failure behavior. It is not a full workspace/catalog, persistent database,
metadata synthesizer, migration, compaction, query, layer partition, export,
CLI, server, UI, heatmap, prompt, expert metric, or performance subsystem.
ST-01 through ST-04 remain deferred; this feature does not change MV-01/MV-08.

Feature 20 adds the experimental, read-only bounded routing-load matrix over
complete Feature 19 shards. Model-free tests verify exact inspection-derived
axes, fresh inspection digest/provenance, strict source/row/cell budgets,
legacy and packed aggregation, complete token-layer-rank coverage, explicit
zero-count experts, count/share/load-ratio formulas, safe source/query
failures, connection cleanup, no raw-row/token-text/result-retention path, and
no network/cache/model imports. It does not write analysis output, infer an
expert universe, create a catalog or persistent database, expose raw rows,
calculate specialization/probability/entropy metrics, or change MV-01 through
MV-08. ST-04 remains deferred.

Feature 21 adds a model-free, dependency-free static HTML heatmap over an
accepted Feature 20 matrix. Tests cover exact metric and cell-budget
validation, fresh matrix reconstruction, both layouts, complete zero-inclusive
tables, provenance/count/shard preservation, deterministic global heat-0..8
bins using zero-only heat-0 and `1 + min(7, int((v / m) * 8))` for positive
values, accessible headers, escaped canonical keys and values, strict CSP,
offline operation, and absence of JavaScript, external resources, storage, or
model execution. The exact visible warning is `Routing load only. Selection
frequency is association evidence, not expert specialization or causal
effect.` The returned string is the permanent portable static HTML export path;
the renderer does not save or open it, and a future React UI is separate. It
remains EXPERIMENTAL and does not change MV-01 through MV-08.

Feature 22 adds the model-free bounded `moeatlas heatmap WORKSPACE` command.
Tests cover required canonical decimal budgets, output preflight before input
or optional dependencies, bounded non-symlink inspection reads, strict
inspection parsing, exactly-once aggregate/render delegation for legacy and
packed shards and all three metrics, atomic publication, race/cleanup safety,
offline operation, and unchanged KeyboardInterrupt/SystemExit semantics. It
does not load a model, inspect a cache, use a browser/network, or change
MV-01 through MV-08; the command remains EXPERIMENTAL. The CLI requires a
caller-created `inspection.to_json()` document, the optional DuckDB `store`
extra, exact lowercase `.html` output, and existing atomic `--force`/failure
cleanup semantics; it does not replace the Feature 20/21 value contracts.

Feature 23 adds model-free bounded routing-run inventory. Tests cover exact
required decimal budgets, absent-tree emptiness without DuckDB, run/shard
candidate and staging validation, hash-derived ordering, exact three-file
source-byte accounting, declared/actual event budgets, complete Feature 19
reopen and conflict validation, redaction policy, deterministic serialization,
in-memory connection cleanup, and atomic JSON CLI publication. The inventory is
not a catalog, latest-run selector, metadata synthesizer, raw-row export,
model/inference path, or persistent database. It remains EXPERIMENTAL and does
not change MV-01 through MV-08; ST-04 remains deferred.

Feature 29 adds the model-free bounded cross-run routing-load comparison.
Tests cover exact delta formulas for all three metrics, zero-delta identity
across distinct runs, determinism and value freezing, strict type/budget
rejection, universe-identity mismatch rejection (model key, adapter identity,
inspection digest, layout, top-k, token count, layer/expert axes), distinct
run-key enforcement, cell budgets, tamper-proof value invariants (zero-sum
rows, unit-interval shares, finite floats, ratio ranges, assignment-count
formulas, sorted shard keys), scalar/tuple-only retention, AST forbidden-import
guards, and package export reachability. The comparison performs no I/O and
does not change MV-01 through MV-08 or ST-01 through ST-04.

Feature 30 adds the model-free static HTML comparison renderer over an
accepted Feature 29 value. Tests cover complete documents for all three
metrics, exact signed bin math (`cold-8..delta-zero..heat-8`), byte-identical
determinism, full provenance including both shard `<details>` universes,
strict validation order (metric, budget, exact type, fresh reconstruction,
cells), tamper rejection at render time, HTMLParser structural checks with an
allowed class set, quote-aware escaping of hostile run keys, legend coverage
of the full signed scale, AST forbidden-import guards, and package export
reachability. The renderer performs no I/O and does not change MV-01 through
MV-08 or ST-01 through ST-04.

Feature 31 adds the model-free bounded `moeatlas compare WORKSPACE` command.
Tests cover real two-run workspaces for all three metrics with byte-identical
deterministic publication, equal-run-key rejection before budgets/preflight,
canonical decimal budget enforcement, exact `.html`/`--force` output
semantics, bounded non-symlink inspection reads with fixed failure messages,
missing/uncommitted run-key source failures, incomparable-universe generic
failures without detail leakage, exactly-once delegation with identity and
argument checks, writer reuse, control-flow exception propagation at every
phase, missing-DuckDB dependency-stage behavior, absence of network/cache/
browser paths, laziness of unrelated commands, and AST boundary guards on the
CLI source. It does not change MV-01 through MV-08 or ST-01 through ST-04.

Feature 32 adds the model-free bounded routing-load summary. Tests cover the
default-fixture happy path field by field, closed-form uniform and
concentrated distributions (entropy ln E / 0, normalized 1 / 0, effective E /
1, Gini 0 / (E-1)/E, population CV 0 / √(E-1)·mean-normalized), exact Gini
anchors, dead-expert universe accounting, determinism and freezing, strict
type/budget/validation-order rejection, eleven tamper rejections including
range checks and the dead-fraction identity, scalar/tuple-only retention, AST
forbidden-import guards, and package export reachability. The summary performs
no I/O and does not change MV-01 through MV-08 or ST-01 through ST-04.

Feature 33 adds canonical JSON export/import for the three analysis values.
Tests cover round-trip equality and byte determinism for all artifacts,
primitive-only dictionaries, canonical compact sorted form, artifact-type
markers, all six cross-type mispairings, malformed payloads, wrong schema
versions, missing fields, tolerated unknown extras, array type violations
(bool/int/float confusion), non-finite import rejection, str/bytes/bytearray
input acceptance, and AST purity guards on all three modules. Serialization
performs no I/O and does not change MV-01 through MV-08 or ST-01 through
ST-04.

Feature 34 adds the model-free bounded analysis-bundle writer. Tests cover the
four-file happy path with recomputed SHA-256 digests and byte counts, manifest
canonicality, from_json round-trips of every bundled document, byte-identical
determinism across directories, subset bundles, empty-bundle and identity-
coherence rejections, destination validation (missing parent, occupied/file/
empty destinations, str/Path/int/None), injected os.replace failures and
KeyboardInterrupt propagation with complete cleanup and no partial artifacts,
receipt tamper rejections, exact-type/subclass enforcement, and AST guards.
The writer does not change MV-01 through MV-08 or ST-01 through ST-04.

## Deferred checks

### Required v1 family compatibility matrix

The final release matrix must contain separate rows for current
Mistral/Mixtral, current Qwen MoE, DeepSeek MoE, and MiniMax MoE revisions. Each
row must record immutable model and tokenizer revisions and evidence for load,
static inspection, routing decode/capture, normalized events, persistence,
analysis, and visualization. The generic scanner remains the static fallback
for unknown MoE architectures but is not routing certification. All family rows
are currently `deferred`; existing Mixtral and Qwen3 model-free fixtures are
reference evidence only. A release-time official-source review is mandatory
because upstream “latest” revisions and tensor layouts can change.

| ID | Check | Current status | Required evidence before completion |
| --- | --- | --- | --- |
| MV-01 | Load a small, pinned real MoE checkpoint through the first certified adapter | deferred | model ID and immutable revision, license, download source, loader config, output |
| MV-02 | Run static discovery and inspect the semantic manifest | deferred | scan JSON, module paths, expert count/top-k, warnings, capability tier |
| MV-03 | Capture routing against native output or a golden reference | deferred | exact command, fixture/prompts, tolerances, comparison result |
| MV-04 | Verify passive output equivalence with hooks enabled and disabled | deferred | baseline/probed outputs, dtype/device, tolerance, result |
| MV-05 | Measure routing-only overhead and memory behavior | deferred | hardware, software versions, baseline/probed timings, peak memory |
| MV-06 | Run CUDA validation on the provisioned VM | deferred | GPU model/driver/CUDA, command, logs, artifact path |
| MV-07 | Validate a fused/quantized or otherwise limited execution path | deferred | backend/quantization settings, capability downgrade, trace evidence |
| MV-08 | Re-run the complete model-dependent suite after packaging | deferred | installed wheel/version, test report, model cache location, result |

## Storage checks

| ID | Check | Current status | Required evidence before completion |
| --- | --- | --- | --- |
| ST-01 | Validate one complete legacy and packed routing shard on the target VM | deferred | exact command, DuckDB version, manifest and Parquet artifacts |
| ST-02 | Validate reopen/list and idempotent/conflict behavior on the target filesystem | deferred | commands, corruption/recovery cases, artifact paths |
| ST-03 | Validate durability, permissions, and crash/rename-fsync recovery | deferred | filesystem, failure injection, modes, fsync/recovery logs |
| ST-04 | Validate scale, workspace/catalog integration, and any query surface | deferred | separately approved scope, benchmark, schema, and compatibility evidence |

## Final VM execution record

Fill this section only when the VM is provisioned. Do not replace the
`deferred` status with an assumption.

```text
VM/provider:
Date (UTC):
OS / architecture:
Python:
MoEAtlas commit/tag:
PyTorch / Transformers / safetensors:
GPU / driver / CUDA:
Model ID and immutable revision:
Tokenizer ID and revision:
Commands:
Artifacts and logs:
Results:
Known limitations:
```

## Completion rule

Each row becomes `passed`, `failed`, or `blocked` only with the evidence named
in its final column. If a model cannot expose a semantic signal, record the
capability tier and limitation instead of treating missing data as a test pass.

## Feature 24 prompt prefill

Status: deferred. The model-free Feature 24 evidence suite currently collects
115 focused cases and covers strict preflight side-effect gates, one tokenizer
call, exact mapping/shape/materialization/converter contracts, token-event
construction, legacy and packed Feature 18 composition, exact control-flow and
retryable cleanup behavior, non-retention, and AST hardkills. Prompts, paths,
raw tensors, and model/tokenizer objects are not persisted by the seam. Real
tokenizer behavior, checkpoint compatibility, GPU routing fidelity, and
performance still require the final provisioned VM; no model files are
downloaded here.
Feature 25 Qwen3.5-MoE structure inspection: Status: model-free structure
complete, runtime certification deferred. Fixtures cover the conditional
`model.language_model.layers` and text-only `model.layers`/bare-base surfaces,
strict current-family identity, exact nested `model.language_model.config`,
official v5.14 descendant allowlists (`experts.act_fn` and shared-expert
gate/up/down/act_fn), packed shapes, shared-expert metadata, and rejection of
indexed, foreign, or mixed layouts. Official checkpoint loading, routing
capture/decoding, GPU equivalence, and immutable revision pinning remain final
VM/release work. The structural basis is the official
[Transformers v5.14.0 modeling source](https://github.com/huggingface/transformers/blob/v5.14.0/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py),
[modular source](https://github.com/huggingface/transformers/blob/v5.14.0/src/transformers/models/qwen3_5_moe/modular_qwen3_5_moe.py),
[Qwen3.5-35B-A3B card](https://huggingface.co/Qwen/Qwen3.5-35B-A3B), and
[current config](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/raw/main/config.json).
The native gate tuple is `(router_logits, router_scores, router_indices)`;
Feature 26 decoding and final release revision review remain deferred. No model
files are downloaded.
Feature 26 Qwen3.5 routing decoding: Status: model-free decoder complete; runtime
certification deferred. Qwen-owned conditional/text hook fixtures cover the
official packed `(router_logits, router_scores, router_indices)` tuple,
deterministic stable-softmax top-k, tie rejection, score renormalization,
shared-expert exclusion, fresh model-neutral events, single-use router
bindings, and cleanup-safe capture integration. Official runtime/checkpoint
equivalence, GPU validation, and immutable revision pinning remain final-VM and
release-review work. No model files are downloaded.
Feature 27 Qwen3.5 forward: Status: implemented at the model-free boundary;
the decoder/session composition, neutral result, cleanup evidence, and
append/reopen/inventory storage compatibility, neutral aggregate, and neutral
visualization are covered by Feature 28. The analysis contract validates
complete routed structural universes and excludes shared experts from routed
denominators.
Official checkpoint/runtime equivalence and GPU certification status remain
deferred to the final VM and release-time revision review. No model files are
downloaded.
Run-evidence export bundles: Status: implemented at the model-free boundary.
Round-trip, byte-determinism, tamper/forged-digest rejection, canonicality
enforcement, redaction fidelity (null text exactly when not stored), row/byte
budgets, crash-safe publication cleanup, symlink refusal, idempotent
re-import, and duckdb-free verification are covered by synthetic local tests.
Bundles carry no model-dependent claims; this feature does not change MV-01/MV-08
or ST-01 through ST-04 status.
Routing-run assignment query seam: Status: implemented at the model-free
boundary. `aggregate_routing_load` now reads runs exclusively through the
public `query_routing_run_assignments` seam; equivalence with prior analysis
results, per-shard multi-shard grouped counts, budgets, conflicts,
corruption-as-reopen, and typed error carriers are covered by synthetic local
tests. The refactor is behavior-preserving for every previously passing input;
no real-model or filesystem-scale claim changes, so MV-01 through MV-08 and
ST-01 through ST-04 keep their deferred status.

Tabular run exports (CSV/Parquet): Status: implemented at the model-free
boundary. `export_run_tables` / `verify_run_tables` round-trips, canonical CSV
encoding and byte determinism, redaction fidelity, multi-shard ordering,
budgets, crash cleanup, tamper rejection, manifest-shape negotiation, and
duckdb-free verification are covered by synthetic local tests over temporary
workspaces. Parquet members are digest-recorded without a byte-determinism
promise because writer metadata varies across engine versions; no real-model,
GPU, or filesystem-scale claim is made, so MV-01 through MV-08 and ST-01
through ST-04 keep their deferred status.

Bounded dataset reading service: Status: implemented at the model-free
boundary. `moeatlas.services.datasets` reading of JSONL/CSV/Parquet/text and
local HF-style snapshots, budgets, column-mapping validation/projection,
deterministic SHA-256-keyed batch planning, error stages, and lazy DuckDB
resolution are covered by synthetic local tests over temporary files.
Descriptors never fetch data and no test downloads a dataset; real
tokenizer/dataset ingestion equivalence and large-file scale remain deferred
MV/ST evidence, so MV-01 through MV-08 and ST-01 through ST-04 keep their
deferred status.

Deterministic run-engine execution core: Status: implemented at the
model-free boundary. `execute_row_schedule` schedule driving, per-row
failure classification, result budgeting, progress monotonicity,
cancellation semantics, outcome invariants, and determinism are covered by
synthetic local tests over fake executors. No real model executes anywhere;
real forward/generation equivalence and performance remain deferred MV
evidence, so MV-01 through MV-08 and ST-01 through ST-04 keep their deferred
status.

Run input preparation service: Status: implemented at the model-free
boundary. `prepare_input_rows` / `plan_input_batches` prompt and dataset
preparation, role projection, descriptor-driven schedules, budget
propagation, and end-to-end composition with the execution core are covered
by synthetic local tests. No tokenizer or model participates; real
chat-template/tokenization equivalence remains deferred MV evidence, so
MV-01 through MV-08 and ST-01 through ST-04 keep their deferred status.
