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
wrapper and frozen `MixtralRoutingForwardResult`. Model-free tests verify
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

## Deferred checks

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
