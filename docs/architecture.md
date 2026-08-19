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
| `analysis` | routing, load, association, behavior, and causal metrics | Research beta |
| `store` | run metadata, Parquet events, and DuckDB queries | Useful alpha |
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
publishes only a frozen `MixtralRoutingForwardResult` after cleanup and event
postconditions succeed. The result owns no model or decoder and retains the
exact caller output identity plus fresh event copies. Tokenization, prompts,
padding, generation, datasets, storage, CLI, server, and UI remain outside
this seam. After any initial enter, body, or exit failure it makes exactly one
internal `session.close()` retry, re-raises the exact primary exception, and
leaves persistent cleanup as a caller-owned `PendingRuntimeCleanup` handle;
capability remains `EXPERIMENTAL` and MV-03 through MV-08 stay deferred.
