# MoEAtlas architecture foundation

This document records the implementation boundary for the first feature. The
[PRD](specification/MoEAtlas_PRD_v1.docx) remains the product authority; this
file explains how the repository starts without pretending that model support
already exists.

## Initial shape: a modular monolith

The first releases use one installable Python distribution and one import
namespace, `moeatlas`. The package is intentionally split into internal
modules as features arrive, but those modules are not separate wheels yet.
This gives us stable seams without locking immature APIs into a multi-package
release process.

Planned internal areas:

| Area | Responsibility | First implementation phase |
| --- | --- | --- |
| `core` | typed identities, manifests, capabilities, and shared contracts | Probe core |
| `discovery` | static traversal and confidence-scored MoE candidates | Probe core |
| `probe` | serializable plans, passive hooks, bounded policy, cleanup | Probe core |
| `events` | versioned token, routing, and expert evidence contracts | Probe core |
| `loading` | source requests, load policy, plan identity, resolution evidence | Probe core |
| `runtime` | validated instance/custom execution plus lazy HF/local loading | Useful alpha / causal beta |
| `analysis` | bounded routing-load aggregation and later association/behavior/causal metrics | Research beta |
| `store` | bounded content-addressed routing shards (Feature 19) | Experimental prerequisite |
| `server` | local FastAPI API and run progress | Useful alpha |
| `adapters` | explicit static semantic protocol, Mixtral/Qwen3-MoE structure adapters, and STRUCTURE-only inspection | Useful alpha |
| `cli` | headless commands and diagnostics | Foundation / all phases |

The foundation `cli` and diagnostics modules, the `core` capability/identity/
manifest contracts, and the model-runtime-independent `discovery` scanner
exist today. Empty future areas are not created merely to make the tree look
complete; each will arrive with a testable feature and a documented contract.

## Dependency boundary

The foundation requires only Pydantic v2 for strict, JSON-compatible schema
validation and does not import model libraries. PyTorch, Transformers,
Accelerate, and safetensors are optional dependencies in the `model` extra.
They remain lazy imports behind explicit resolved loading calls. This
keeps package import, docs, and tests usable on a CPU-only machine and makes it
impossible for the baseline test command to silently download a checkpoint.

The eventual runtime will prefer native PyTorch module boundaries and native
router outputs, then use adapters for packed/fused or architecture-specific
semantics. It will not rewrite model forwards in the core.

## Provenance and evidence boundary

Every model-dependent result must retain enough provenance to reproduce it:
model ID/path, immutable revision when available, tokenizer revision, runtime
versions, device/dtype, probe plan, dataset fingerprint, and tool version.
The product will show routing association separately from internal behavior,
causal contribution, and replication evidence. No foundation diagnostic is a
compatibility or specialization claim.

## Execution boundary for this phase

The foundation supports:

- installing/importing the package;
- displaying package and Python metadata;
- checking optional runtime package presence without importing it;
- running a JSON or human-readable `doctor` report;
- validating versioned model/component manifests and deterministic identities;
- statically traversing a duck-typed module tree into a STRUCTURE-only
  discovery report with confidence and warnings;
- emitting that report through the explicit model-free
  `scan fixture:synthetic` CLI source;
- resolving strict probe plans and managing torch-free synthetic hook
  lifecycles with transactional cleanup;
- validating normalized, capability-aware token/routing/expert events without
  capturing tensors or writing storage;
- validating model-source/loading intent without loading or resolving a model;
- validating already-instantiated instance/custom runtime artifacts into a
  manifest-ready result with explicit cleanup ownership;
- lazily loading resolved HF/local sources through the three audited
  Transformers factories, with staged observation and retryable rollback;
- composing one resolved HF/local load with the existing read-only scanner via
  `runtime.load_and_scan()`, returning only a validated STRUCTURE report after
  cleanup succeeds;
- scanning an explicitly supplied, already-resolved HF/local `LoadingPlan`
  JSON file through that same runtime bridge without reconstructing or mutating
  the plan;
- inspecting one caller-supplied static semantic adapter with strict,
  versioned descriptor/detection/report contracts, without a registry or
  optional model imports;
- selecting `MixtralStaticAdapter()` explicitly for the exact indexed or
  packed Mixtral structural layouts, without routing certification or runtime
  adapter behavior;
- selecting `Qwen3MoeStaticAdapter()` explicitly for the official Qwen3-MoE
  dense/sparse schedule and either indexed or packed sparse layout, without
  routing certification or runtime adapter behavior;
- compiling a validated adapter inspection into a family-neutral, reduced
  `ROUTING` probe plan without calling models, adapters, or hooks;
- capturing only caller-decoded, identity-bound `RoutingEvent` values through
  a cleanup-safe `RoutingCaptureSession`, with retained-event quota and no
  built-in tensor decoder or storage;
- executing model-free tests.

The optional loader and static adapter seam do not certify a checkpoint, perform inference, trace
tokens, store tensors/events, intervene, or claim GPU compatibility. It is a
small execution seam for a later VM: real network/cache behavior, native
PyTorch fidelity, CUDA/MPS, quantization, runtime adapter semantics, and architecture certification remain
explicit deferred validation items.

The experimental Mixtral routing decoder remains inside this same runtime
boundary. `MixtralRoutingDecoder` binds one caller-provided token-row sequence
to exact router paths and same-layer contiguous expert identities, then emits
only validated `RoutingEvent` evidence. It is one-forward, one-successful-
invocation-per-router, and has no tokenizer, generation runner, storage sink,
server, UI, auto-selection registry, or model download path. Legacy and packed
payloads are distinct evidence contracts; neither changes static component
capabilities or creates a routing certification claim. Native equivalence,
passive output fidelity, overhead, fused/quantized behavior, GPU execution,
and packaging checks remain deferred to MV-03 through MV-08.

Feature 18 keeps one-forward execution as a caller-owned runtime boundary.
`run_mixtral_routing_forward()` receives caller-tokenized row tokens and
caller model kwargs, shallow-copies only the kwargs mapping, and invokes the
callable model exactly once under the canonical passive `RoutingCaptureSession`.
It computes the complete-event budget before hooks/model traversal and
publishes only a frozen model-neutral `RoutingForwardResult` (the historical
`MixtralRoutingForwardResult` name remains an identity alias) after cleanup and event
postconditions succeed. The result owns no model or decoder and retains the
exact caller output identity plus fresh event copies. Tokenization, prompts,
padding, generation, datasets, storage, CLI, server, and UI remain outside
this seam. After any initial enter, body, or exit failure it makes exactly one
internal `session.close()` retry, re-raises the exact primary exception, and
leaves persistent cleanup as a caller-owned `PendingRuntimeCleanup` handle;
capability remains `EXPERIMENTAL` and MV-03 through MV-08 stay deferred.

Feature 19 adds only a bounded routing-shard prerequisite above that result:
DuckDB writes fixed ZSTD Parquet rows and a strict manifest into one
content-addressed shard, with sequential idempotence and reopen validation.
It is not a full workspace/catalog, persistent database, migration, compaction,
query, layer partition, DataFrame/export, CLI, server, UI, heatmap, prompt,
expert-metric, or performance subsystem. Storage capability remains
experimental and ST-01 through ST-04 are deferred; MV-01 through MV-08 remain
unchanged.

Feature 20 adds the experimental, read-only routing-load matrix above those
shards. The caller supplies an exact Mixtral inspection and strict row, source
byte, and matrix-cell budgets; analysis uses the inspection-published full
contiguous
layer/expert universe and never discovers axes from observed rows. It reads
one run across all committed shards, validates complete token-layer-rank
coverage, computes count/share/load-ratio matrices, closes its one in-memory
connection before constructing the value result, and retains no path,
connection, raw row, inspection, or token text. This is not a catalog,
persistent analysis database, query API, metric/specialization claim, or
model/tokenizer/generation path; ST-04 and MV-01 through MV-08 remain
deferred.

Feature 21 adds only a dependency-free static HTML heatmap over one accepted
Feature 20 matrix. It preserves caller-owned axes, provenance, counts, shard
keys, zero-inclusive cells, and deterministic global heat bins; it creates no
UI, server, catalog, storage, model, or specialization claim. The artifact is
EXPERIMENTAL and MV-01 through MV-08 remain deferred.

Feature 22 adds only the bounded `moeatlas heatmap WORKSPACE` CLI composition:
it preflights output, reads one bounded non-symlink inspection document,
delegates aggregation and rendering exactly once, and reuses the existing
atomic writer with exact `.html`/`--force` publication behavior. It creates no
model, tokenizer, browser, network, cache, generation, or alternate storage
path; status, publication, and control-flow semantics remain CLI-only. The
optional DuckDB `store` extra is the only required runtime dependency. The
command is EXPERIMENTAL and MV-01 through MV-08 remain deferred.

Feature 23 adds only a bounded routing-run inventory over the immutable
Feature 19 tree. It rebuilds summaries on demand with required budgets and
full reopen validation; a caller chooses the `run_key` later when invoking
Feature 20/21, and no latest-run or catalog state is owned here. The CLI
publishes JSON through the existing atomic writer only after the inventory is
complete, and inventory cleanup remains internal with no partial value.

Feature 29 adds only a bounded, read-only, in-memory comparison of two
accepted `RoutingLoadMatrix` values over one identical universe.
`compare_routing_load` requires exact schema/model/adapter/inspection/layout/
top-k/token-count/axis equality and distinct run keys, then publishes frozen
count/share/ratio delta matrices whose rows provably sum to zero. It creates
no I/O, rendering, ranking, catalog, or specialization claim; deltas are
association evidence only.

The EXPERIMENTAL prompt-prefill seam accepts plain text, borrows the caller's
validated model/tokenizer, and composes exactly one bounded Feature 18 forward;
it is not a generation, storage, or CLI subsystem. Feature 24 intentionally
does not introduce a server endpoint, websocket/progress channel, React view,
or JSON wire/view-model contract: the Python result is the existing Feature 18
result and callers explicitly choose the later Feature 19 append, Feature 23
inventory, Feature 20 analysis, and Feature 21 export actions. A future server
may define a separate versioned wire contract after these local seams are
validated; keeping that boundary out now prevents prompt/runtime objects from
leaking into persistence or UI state.

## Multi-family release constitution

MoEAtlas is model-neutral by design; Mixtral is the reference and regression
implementation, not the product boundary. Family-specific module paths, router
payloads, tensor layouts, and decoding rules belong only in isolated static
adapters and runtime decoders. Normalized events, immutable storage, analysis,
Python/CLI/server APIs, and every visualization must consume shared contracts
without branching on a model family.

Version 1 release acceptance requires end-to-end evidence for current
Mistral/Mixtral, current Qwen MoE, DeepSeek MoE, and MiniMax MoE families, plus
the existing generic static-discovery fallback for unknown architectures. For
each certified family, the compatibility matrix must name exact immutable model
and tokenizer revisions and prove loading, structural inspection, routing
capture/decoding, normalized events, persistence, analysis, and visualization.
Code-only or fake-model evidence remains EXPERIMENTAL. Certification requires
real checkpoint and GPU runs on the final VM, followed by a release-time review
of official upstream revisions; no moving `latest` reference is evidence.
Qwen3.5-MoE support is isolated to an explicit packed-only static adapter
for the `qwen3_5_moe`/`qwen3_5_moe_text` family; it does not alter the generic schema, registry, decoder, loading, or
server surfaces. The conditional wrapper and text-only surface are handled by
the same adapter, while family-specific tensor layouts stay inside that seam.
Feature 25 records the v5.14 structure contract and the native gate tuple
`(router_logits, router_scores, router_indices)`. Feature 26 adds the separate
passive Qwen3.5 decoder seam: it validates the packed tuple, excludes shared
experts, and emits fresh model-neutral routing events without modifying the
Mixtral decoder, generic event schema, adapter registry, or model server path.
The shared routing-capture context validates shared-expert metadata and keeps
it out of routed `expert_keys` for every family; this remains model-neutral.
Checkpoint/GPU validation and the release-time review of the official revision
remain deferred to the final VM.
Feature 27 composes Qwen3.5 capture with the existing session lifecycle while
keeping model-family decoding isolated. It adds no server, UI, CLI, loader, or
schema surface. Its downstream evidence boundary ends at append, reopen, and
run inventory; aggregate and visualization remain Mixtral-specific pending
Feature 28 neutralizes the analysis and visualization seams. The shared
`RoutingLoadMatrix`, `aggregate_routing_load`, and
`render_routing_load_heatmap` surfaces consume only the inspection-published
structural routing universe. Historical Mixtral names remain identity aliases;
there is no family allowlist or family-specific adapter import in analysis. Shared experts are
validated and excluded from routed axes and denominators. Mixtral output bytes
remain unchanged.
