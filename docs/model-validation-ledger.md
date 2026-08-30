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
VM/provider: vast.ai instance 48352897 (machine 17422), RTX 3090 container image cuda-12.4.1-auto
Date (UTC): 2026-08-22
OS / architecture: Ubuntu 22.04.5 LTS, x86_64 (unprivileged Docker container)
Python: 3.11.16 (uv-managed venv)
MoEAtlas commit/tag: working tree synced @ local commit 87b508cb5d92d61f051c7dfaa411e9a1eee599d2
PyTorch / Transformers / safetensors: torch 2.6.0+cu124 / transformers 5.15.0 / safetensors 0.8.0
  (accelerate 1.14.0, bitsandbytes 0.50.1, duckdb 1.4.5, pydantic 2.13.4)
GPU / driver / CUDA: NVIDIA GeForce RTX 3090 24GB (capability 8,6) / driver 550.144.03 / CUDA 12.4 (nvcc V12.4.131)
Model ID and immutable revision:
  - unsloth/Qwen3-30B-A3B-bnb-4bit @ e1b99b50cfb1b2381489585b53ad96858a09853b (bnb NF4 — BROKEN as-shipped, see below)
  - inclusionAI/Ling-3.0-tiny @ b61f4338de3e68ffc9c0bc1ed5e902981a4a929e (bf16, BailingMoeV3ForCausalLM / bailing_hybrid)
  - amd/Instella-MoE-16B-A3B-Think @ fe339bc1a946… (bf16, native deepseek_v3 class)
Tokenizer ID and revision: each model's snapshot-pinned tokenizer at the revisions above
Commands: see vm-evidence/scripts/ on the VM and per-phase logs listed below
Artifacts and logs: /root/moeatlas/vm-evidence/ (lint.log, unittest.log, build.log, env.log,
  st-checks.log, mv06-cuda.log, provenance.log, model-download.log, model-verify.log,
  ling-mv01..mv07 logs/json, instella-*.log/json, mv08-wheel-tests.log, mv07-qwen-bnb-broken.log)
Results:
  - CI-equivalent gates on Linux VM: pytest 2478 passed; ruff clean; unittest discovery OK;
    wheel+sdist built.
  - ST-01..ST-03: 21/21 sub-checks PASS on overlay filesystem (legacy+packed shards, reopen/list,
    idempotence/conflict, tamper detection, crash-injection cleanup, modes 0700/0600).
  - MV-06: CUDA validated (torch.cuda.is_available()=True, device/capability recorded).
  - MV-01/MV-02 (generic-fallback lane): Ling auto-discovered experts 128==config, top-k 8==config,
    shared 1, 23 routers, [STRUCTURE]-only tier with honest warnings; certified adapters correctly
    rejected it. Instella auto-discovered experts 64==64, top-k 6==6, shared 2==2 via the same
    universal seam with ZERO per-model code — generic scanner universality confirmed on a third
    foreign architecture.
  - MV-03..MV-05 (caller-owned lane, no certified adapter): Ling routing capture 2392/2392 complete
    events, ids∈[0,128), golden sigmoid+bias+group-limited top-k recompute matched; passive hook
    equivalence bitwise identical (max_abs_diff 0.0); overhead +0.545% mean latency, peak VRAM 14.73GiB.
  - MV-07: two paths recorded — (a) Qwen3-bnb NF4 quantized path BROKEN as-shipped under
    transformers 5.15 (packed fused expert uint8 Parameters receive no quant_state; 93,360 quant
    metadata keys reported UNEXPECTED); no reconstruction attempted per decision; (b) native bf16
    path fully functional including fused Triton kernels (fla-core 0.5.2) for hybrid attention.
  - MV-08: wheel sha256 93744dba532ac58122055e5fb5ee0e146a3f8ee5ec3099f423755e4ea4a05452;
    isolated venv install → 2403 passed (+45 subtests); 145 expected out-of-tree failures are
    source-introspection guards requiring the checkout tree; no functional/artifact failures.
Known limitations:
  - No certified adapter family was exercised against a real checkpoint in this run; all real-model
    discovery used the generic static fallback and all capture was explicitly caller-owned. MV-01
    through MV-05 remain `deferred` for the certified-family matrix (Mixtral/Qwen3/DeepSeek/MiniMax).
  - The unsloth bnb-4bit export of Qwen3-30B-A3B cannot execute under transformers 5.15; a non-broken
    quantized checkpoint is required before any MV-07 pass can be claimed for that lane.
  - load_instance rejects a live transformers PretrainedConfig object
    (`config.id2label object keys must be strings`) while accepting the plain config.json mapping —
    product seam fix pending.
```

### 2026-08-22 status annotations

- MV-01..MV-05: stay `deferred` (generic-fallback and caller-owned evidence above does not satisfy
  the certified-adapter requirements of these rows).
- MV-06: CUDA environment validation evidence collected; row completion still requires the
  certified-family command/log set.
- MV-07: quantized-path lane has an explicit FAIL record (Qwen3-bnb broken export) and a working
  native-bf16 record (Ling fused Triton kernels); row stays `deferred` until a valid quantized
  checkpoint passes end-to-end.
- MV-08: packaged-wheel re-run executed successfully for the model-free suite (2403 passed);
  row stays `deferred` until repeated after certified-family model-dependent tests exist.
- ST-01..ST-03: target-VM evidence complete (21/21); rows may move to `passed` at release review
  once the exact commands are transcribed into release docs. ST-04 remains `deferred`.

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
Feature 29 Qwen3.8 Flash Next (`qwen4_exp`) structure inspection: Status: model-free static complete;
VM FP8 certification deferred. The adapter accepts only the pinned outer
`qwen4_exp`/`Qwen4ExpForConditionalGeneration` identity with nested
`qwen4_exp_text`, validates the exact `model.language_model.layers` packed
topology and shapes, and excludes the shared expert from routed logical
experts. It emits only unverified `STRUCTURE` evidence and retains no model or
tensor values. The structural basis is the pinned
[Transformers Qwen4-Exp modeling source](https://github.com/huggingface/transformers/blob/42ca97014c85d71a88ad60d55f08cb9fb4d26e2c/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py)
and the official
[Qwen3.8-Flash-Next-FP8 configuration](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/blob/main/config.json).
Official FP8 checkpoint loading, routing/runtime equivalence, GPU behavior,
and release-time immutable revision review remain deferred to the final VM.
No model files are downloaded.
Feature 2 Qwen3.8 generic routing candidate: Status: model-free candidate
support complete; runtime and FP8 certification deferred. Generic discovery
normalizes one shared expert per layer, binds exactly one routed gate per layer,
and decodes the packed `(router_logits, router_scores, router_indices)` tuple
into routed events. Packed experts are logical slices rather than independent
hook targets, so capture support is truthfully `routing_candidate` only. Real
FP8 loading, CPU/RAM-offload behavior, native equivalence, and upstream open
FP8/CPU issues require a pinned VM run; this feature does not patch upstream
Transformers.
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

Headless run-engine service surface: Status: implemented at the model-free
boundary. `execute_specification` lifecycle projection, per-batch
`update_progress` records, cooperative cancellation, atomic canonical JSON
checkpoints with validated `load_checkpoint`/`resume_from` continuation, and
`publish_run_report` catalog publication are covered by synthetic fake-runtime
tests over temporary fixtures. No tokenizer or model participates; real
generation equivalence, checkpoint behavior under GPU execution, and
filesystem durability at scale remain deferred evidence, so MV-01 through
MV-08 and ST-01 through ST-04 keep their deferred status.

Task association metrics: Status: implemented at the model-free boundary.
`TaskExpertCounts` validation, enrichment, PMI/MI, Jensen-Shannon
separability, and exclusivity/generality are covered by synthetic-table
tests with exact expected values. No model or tokenizer participates;
association over real per-token task-labeled routing evidence remains
deferred until task-labeled executors land, so MV-01 through MV-08 and
ST-01 through ST-04 keep their deferred status.

Evidence Cards: Status: implemented at the model-free boundary. Identity
strictness, optional tiered sections (routing, task association, behavior,
causality, stability), capability-label vocabulary, limitations/warnings,
and canonical `moeatlas.evidence_card` round-trips are covered by synthetic
contract tests. No model or tokenizer participates; populating cards with
real routing, behavior, causal, and replication evidence remains deferred to
the VM/GPU lane, so MV-01 through MV-08 and ST-01 through ST-04 keep their
deferred status.

Prompt-vs-rollout routing agreement: Status: implemented at the model-free
boundary. Paired-count validation, Jensen-Shannon agreement with exact
expected values, total-variation distance, scale invariance, budget bounds,
and canonical `moeatlas.routing_agreement` round-trips are covered by
synthetic contract tests. No model or tokenizer participates; agreement over
real prompt/rollout phases of a real checkpoint remains deferred VM/GPU
evidence, so MV-01 through MV-08 and ST-01 through ST-04 keep their deferred
status.

Cross-run association stability: Status: implemented at the model-free
boundary. Topology-mismatch rejection, exact Jensen-Shannon agreement values,
scale invariance, budget bounds, and canonical
`moeatlas.association_stability` round-trips are covered by synthetic
contract tests. No model or tokenizer participates; stability across real
runs of a real checkpoint remains deferred VM/GPU evidence, so MV-01 through
MV-08 and ST-01 through ST-04 keep their deferred status.

Router margin: Status: implemented at the model-free boundary. Sample
strictness (including non-finite score rejection), exact margin values,
undefined-token accounting, budget bounds, and canonical
`moeatlas.router_margin` round-trips are covered by synthetic contract
tests. No model or tokenizer participates; margins over real router scores
of a real checkpoint remain deferred VM/GPU evidence, so MV-01 through MV-08
and ST-01 through ST-04 keep their deferred status.

Route churn: Status: implemented at the model-free boundary. Sequence
strictness (including duplicate-expert rejection), exact churn/Jaccard
values, empty-step conventions, budget bounds, and canonical
`moeatlas.route_churn` round-trips are covered by synthetic contract tests.
No model or tokenizer participates; churn across real generated tokens of a
real checkpoint remains deferred VM/GPU evidence, so MV-01 through MV-08 and
ST-01 through ST-04 keep their deferred status.

Co-routing graphs: Status: implemented at the model-free boundary. Matrix
strictness (square, symmetric, zero-diagonal), exact ranking/share values,
deterministic top-pair bounding, budget bounds, and canonical
`moeatlas.corouting` round-trips are covered by synthetic contract tests. No
model or tokenizer participates; co-routing over real activations of a real
checkpoint remains deferred VM/GPU evidence, so MV-01 through MV-08 and
ST-01 through ST-04 keep their deferred status.

Expert similarity: Status: implemented at the model-free boundary. Vector
strictness (one finite per-expert vector per layer, shared within-layer
length), exact cosine values including identical/orthogonal/opposite
directions, symmetric matrices with exact `1.0` diagonals, explicit `null`
cells touching zero-norm experts, budget bounds, and canonical
`moeatlas.expert_similarity` round-trips are covered by synthetic contract
tests. No model or tokenizer participates; similarity over real expert
weights or activations of a real checkpoint remains deferred VM/GPU
evidence, so MV-01 through MV-08 and ST-01 through ST-04 keep their deferred
status.

Adapter plugin registry: Status: implemented at the model-free boundary.
Record/policy strictness, builtin provenance, entry-point discovery with
fixed failure-reason vocabulary, deterministic collision resolution,
policy-driven enable/disable statuses, family negotiation, and canonical
`moeatlas.adapter_registry` round-trips are covered by synthetic contract
tests with injected fake entry points. No model, tokenizer, network, or
third-party distribution participates; registry behavior against real
published plugin distributions remains deferred release-engineering
evidence, so MV-01 through MV-08 and ST-01 through ST-04 keep their
deferred status.

Headless CLI run/export flows: Status: implemented at the model-free
boundary. Parser contracts, exactly-one input enforcement, executor-plugin
resolution with fixed rejection messages, loading-plan provenance
projection, checkpoint publication, workspace-catalog recording, canonical
budget parsing, bundle export with manifest digest reporting, and fixed
safe errors are covered by synthetic contract tests over fake executors and
local fixtures. No model, tokenizer, network, or GPU participates; real
model execution through a registered adapter plugin remains deferred VM/GPU
evidence (MV-01 through MV-08), and ST-01 through ST-04 keep their deferred
status.

Local server and UI launch: Status: implemented at the model-free boundary.
Wire DTO strictness, construction budgets, health identity, workspace/run
endpoints over initialized and uninitialized catalogs, bounded state
filtering, adapter-registry exposure, fixed dependency errors, and the
loopback-by-default `moeatlas ui` policy are covered by synthetic contract
tests using FastAPI's TestClient over temporary workspaces. No model,
tokenizer, download, or GPU participates. The React/TypeScript single-page UI
is now packaged and covered by static-bundle contracts plus a local browser
smoke; broad synthetic browser end-to-end coverage and all model-dependent
flows remain deferred release-engineering evidence. MV-01 through MV-08 and
ST-01 through ST-04 keep their deferred status.

Intervention mechanics: Status: implemented at the model-free boundary.
Recipe contracts (fixed operation vocabulary, per-operation parameter
exclusivity, sorted unique targets, canonical serialization, content
fingerprints binding `InterventionLineage`), budget bounds and failures,
and the failure-safe engine (`run_intervention()` capture → apply →
observe → restore with restoration guaranteed on apply failure, execution
failure, cancellation, and restore-stage reporting) are covered by
synthetic-module contract tests. No model, tokenizer, download, or GPU
participates. Real-model causal effect, regret, stability, and replication
evidence — including explicit unsupported/fused/quantized limitations —
remains deferred VM/GPU evidence under MV-01 through MV-08; ST-01 through
ST-04 keep their deferred status.

Causal evidence summaries: Status: implemented at the model-free boundary.
Paired-effect reduction (`analyze_causal_evidence` over `CausalPair`
observations), per-label means, absolute/relative effects, direction
consistency, strict stability markers, duplicate-replication rejection,
budget failures, and canonical round-trips are covered by synthetic
contract tests. No model, tokenizer, download, or GPU participates.
Real-model causal effect/regret/stability/replication measurements remain
deferred VM/GPU evidence under MV-01 through MV-08; ST-01 through ST-04
keep their deferred status.

Retention evaluation: Status: implemented at the model-free boundary.
Policy bounds and contract failures, deterministic classification rules,
untimestamped-entry semantics, combined age/count bounds, tie-breaking by
run key, empty registries, input type contracts, and canonical report
round-trips are covered by synthetic contract tests over temporary
registries. No model, tokenizer, download, or GPU participates. Real
workspace-scale retention enforcement and filesystem durability remain
deferred evidence under ST-01 through ST-04.

Release-engineering surfaces: Status: implemented at the model-free
boundary. Governance files (security policy with a private reporting path,
code of conduct, Keep-a-Changelog process), issue/pull-request templates,
the three-Python-version CI workflow mirroring the local serialized gate,
and the synthetic example workspace (registered runs, registry query,
retention evaluation) are verified by repository-anchor tests and a clean
subprocess example test. No model, tokenizer, download, or GPU
participates. Clean wheel/sdist installation into pristine environments,
NOTICE/screenshot/demo assets, optional Docker packaging, and release
publishing remain deferred release-engineering evidence.

Benchmark artifacts: Status: implemented at the model-free boundary.
Plan/result contracts (canonical-JSON workloads, content-addressed plan
fingerprints, caller-supplied environment/timestamp provenance,
one-result-per-case collection, pinned `release_evidence: false`) are
covered by synthetic contract tests. The server also exposes an opt-in,
cancellable native-versus-captured forward-timing lane and persists its
provenance-bound report, but no model, tokenizer, download, or GPU
participates in local certification. Real performance measurements on
provisioned infrastructure remain deferred VM/GPU evidence; developer-machine
timing is never promoted to release evidence by this API.

Final model-free PRD audit: Status: complete. `docs/prd-audit.md` traces
every PRD v1 acceptance area to its implementation surface, local
synthetic-test evidence, and honest status, and enumerates MV-01 through
MV-08 and ST-01 through ST-04 as the complete list of infrastructure-bound
claims this repository does not make. No model, tokenizer, download, or GPU
participated in the audit. The VM/GPU half of Sequence 12 — executing those
rows on provisioned hardware and the release-time review of official
revisions — remains blocked until access exists; it is recorded as blocked,
never passed.
