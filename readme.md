# MoEAtlas

Map. Inspect. Understand.

MoEAtlas is an open-source, local-first observability and experimentation
layer for PyTorch Mixture-of-Experts (MoE) models. It is designed to make
routers, experts, token routes, runtime traces, comparisons, and causal tests
inspectable without turning routing frequency into an unsupported claim of
expert specialization.

The repository is being implemented feature by feature against the
[MoEAtlas product requirements](docs/specification/MoEAtlas_PRD_v1.docx).
The current release is the repository foundation only. Model loading,
discovery, instrumentation, storage, analysis, and the local UI are planned
phases rather than advertised as complete features.

## Current status

- Python 3.11+ package named `moeatlas`.
- Model-free `moeatlas doctor` command.
- Standard-library test harness for the foundation.
- PyTorch, Transformers, and checkpoint/GPU validation explicitly deferred to
  the final VM phase.
- No model files are downloaded by the repository tests or setup.

Check the [model-validation ledger](docs/model-validation-ledger.md) before
interpreting a local test result as model compatibility evidence.

## Quick start

Using `uv`:

```bash
uv venv
uv pip install -e '.[dev]'
moeatlas doctor
moeatlas doctor --json
```

The package has no required runtime dependencies in this foundation slice.
The optional `model` extra describes dependencies that later model-runtime
features may use; it is intentionally not installed by the commands above.

Run the model-free tests without downloading a checkpoint:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

When `pytest` is already available, the equivalent command is:

```bash
PYTHONPATH=src python -m pytest
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
2. probe core with scanner, manifests, hooks, event schema, and a synthetic
   fixture;
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
