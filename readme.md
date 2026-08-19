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
explicit runtime loading seams. Storage, analysis, and the local UI remain
planned phases; real checkpoint and GPU certification are deliberately deferred.

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
- Model-free test harness for the foundation and schemas.
- Real PyTorch, Transformers, and checkpoint/GPU fidelity explicitly deferred
  to the final VM phase.
- No model files are downloaded by the repository tests or setup.

Check the [model-validation ledger](docs/model-validation-ledger.md) before
interpreting a local test result as model compatibility evidence.

The experimental `MixtralRoutingDecoder` consumes an exact fresh
`AdapterInspection` and non-empty ordered `TokenEvent` tuple. It binds each
router's key/path, same-layer contiguous experts, count, and top-k, and allows
one successful invocation per router path. Legacy payloads emit observed
selected logits only; packed payloads additionally emit native weights after
strict softmax/top-k/renormalization checks. Its evidence remains
`EXPERIMENTAL`; native equivalence and runtime/GPU validation stay deferred to
MV-03 through MV-08.

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
uv run --locked python -m unittest discover -s tests -v
```

## Planned product shape

The implementation will remain a coherent modular monolith initially. Internal
boundaries will follow the PRD—core schemas, discovery, probing, runtime,
analysis, storage, server, adapters, and CLI—before any boundary is promoted
to a separately distributable package. This keeps early changes easy to test
and lets real model behavior guide the abstractions.

The roadmap is documented in [architecture](docs/architecture.md) and the
PRD. The intended progression is:

1. repository foundation and model-free contracts;
2. probe core with static scanner, manifests, hooks, event schema, and a
   synthetic fixture;
3. useful alpha with small real-model adapters and routing storage/UI;
4. research and causal workflows;
5. certified compatibility, plugin SDK, packaging, and benchmarks.

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
