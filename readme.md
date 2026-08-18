# MoEAtlas

Map. Inspect. Understand.

MoEAtlas is an open-source, local-first observability and experimentation
layer for PyTorch Mixture-of-Experts (MoE) models. It is designed to make
routers, experts, token routes, runtime traces, comparisons, and causal tests
inspectable without turning routing frequency into an unsupported claim of
expert specialization.

The repository is being implemented feature by feature against the
[MoEAtlas product requirements](docs/specification/MoEAtlas_PRD_v1.docx).
The current release contains the repository foundation and canonical manifest
contracts. Model loading, discovery, instrumentation, storage, analysis, and
the local UI are planned phases rather than advertised as complete features.

## Current status

- Python 3.11+ package named `moeatlas`.
- Model-free `moeatlas doctor` command.
- Pydantic v2 model/component manifests with versioned JSON contracts and
  deterministic portable identity helpers.
- Read-only, model-runtime-independent static MoE discovery with confidence-
  scored candidates, normalized expert/top-k facts, and STRUCTURE manifests.
- Strict serializable probe plans with deterministic target resolution and a
  torch-free transactional hook lifecycle manager.
- Model-free test harness for the foundation and schemas.
- Real PyTorch, Transformers, and checkpoint/GPU fidelity explicitly deferred
  to the final VM phase.
- No model files are downloaded by the repository tests or setup.

Check the [model-validation ledger](docs/model-validation-ledger.md) before
interpreting a local test result as model compatibility evidence.

## Quick start

Using `uv`:

```bash
uv sync --extra dev
uv run --locked moeatlas doctor
uv run --locked moeatlas doctor --json
```

The foundation runtime uses Pydantic v2 for strict manifest validation. The
optional `model` extra describes dependencies that later model-runtime
features may use; it is intentionally not installed by the commands above.

Run the model-free tests without downloading a checkpoint:

```bash
uv run --locked pytest -q
uv run --locked ruff check src tests
```

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
