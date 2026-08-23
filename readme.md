# MoEAtlas

Map, inspect, and test how Mixture-of-Experts models route work.

MoEAtlas is a local-first research application for Hugging Face and PyTorch
MoE models. It loads a model and dataset, discovers routed-expert structure,
captures routing from real forwards, renders layer × expert heatmaps, and runs
controlled expert interventions against the same rows.

MoEAtlas is not a universal inference engine. Compatibility depends on the
model loading successfully, exposing a router signal that can be decoded, and
fitting the available hardware. The UI reports unavailable evidence instead of
guessing.

## What works

- Hugging Face model and dataset intake with immutable revision evidence.
- Generic MoE topology discovery without a central model-family allowlist.
- Expert count, top-k, shared-expert, router, weight-layout, and execution-backend evidence.
- Real prompt or dataset execution through Transformers.
- Routing capture, expert activity summaries, persistent run shards, and exports.
- Complete layer × expert heatmaps that scale to large matrices.
- Cross-run routing comparisons and similarity summaries.
- Baseline-derived expert zeroing and scaling when experts are independent modules.
- Output, task-score, latency, routing, target-invocation, and restoration comparisons.
- Replicated intervention studies with optional negative controls.
- Hugging Face expert-backend discovery and a reversible pass-through runtime handshake.
- Per-operation capability reports for capture, zeroing, scaling, rerouting,
  renormalization, and compute skipping.
- A local React research console served by the Python package.

The operation report is intentionally precise:

| Operation | Current support |
| --- | --- |
| Capture routing | Available after a real forward validates the router payload |
| Zero expert contribution | Available for independently exposed expert modules |
| Scale expert contribution | Available for independently exposed expert modules |
| Exclude and renormalize | Detected but not implemented |
| Reroute to next-best expert | Detected but not implemented |
| Skip expert compute | Detected but not implemented |

Packed weights and fused execution are separate facts. Detecting either one
does not imply that MoEAtlas can safely intervene inside that implementation.

## Compatibility boundary

A successful end-to-end run requires all of these stages:

1. The model and tokenizer load in the active runtime.
2. The dataset exposes a usable text column.
3. Static discovery finds an MoE topology and candidate router targets.
4. A real forward proves the router payload.
5. Captured assignments pass shape, range, count, and identity validation.
6. The run publishes its immutable evidence before the UI renders it.

Custom remote code, missing optional dependencies, unsupported Transformers
versions, opaque router outputs, quantized checkpoint defects, insufficient
RAM or accelerator memory, and process termination can stop the journey before
capture. See the [validation ledger](docs/model-validation-ledger.md) for
recorded evidence and limitations.

## Install

Python 3.11 or newer is required.

Using uv for development:

```bash
uv sync --extra dev --extra model --extra server --extra store
uv run moeatlas doctor
```

For Hugging Face datasets, install the optional dataset runtime in the same
environment:

```bash
uv pip install datasets
```

Using pip:

```bash
python -m pip install -e '.[model,server,store]'
python -m pip install datasets
```

## Start the research console

```bash
mkdir -p /workspace/moeatlas-workspace
moeatlas ui /workspace/moeatlas-workspace \
  --host 0.0.0.0 \
  --port 8000 \
  --allow-remote
```

The UI opens directly on model and dataset selection. Running it inside a GPU
VM does not require a separate SSH workflow; expose the selected port using the
provider's normal port or proxy controls.

## Development

The ordinary test suite does not download checkpoints:

```bash
uv run pytest -q
uv run ruff check src tests
cd frontend
npm run check
npm run build
```

Build the packaged UI assets with:

```bash
cd frontend
npm run build:static
```

Real checkpoint compatibility must be tested separately on the target model
revision, Transformers version, PyTorch version, device, and precision.

## Repository map

- `src/moeatlas/discovery/`: generic structure discovery.
- `src/moeatlas/runtime/`: loading, compatibility, capture, and runtime evidence.
- `src/moeatlas/executors/`: real model execution.
- `src/moeatlas/store/`: immutable routing and expert evidence.
- `src/moeatlas/analysis/`: heatmaps, comparisons, association, and stability analyses.
- `src/moeatlas/interventions/`: reversible manipulation and causal evidence.
- `src/moeatlas/server/`: local API, jobs, diagnostics, and packaged UI.
- `frontend/`: React and TypeScript research console.
- `tests/`: model-free contracts and regression tests.
- `docs/`: focused technical references and validation evidence.

## Documentation

- [Current roadmap](docs/roadmap.md)
- [Architecture](docs/architecture.md)
- [Loading and compatibility](docs/loading.md)
- [Runtime and capture](docs/runtime.md)
- [Interventions](docs/interventions.md)
- [Storage](docs/storage.md)
- [Analysis](docs/analysis.md)
- [Server and UI](docs/server.md)
- [Validation ledger](docs/model-validation-ledger.md)
- [Historical PRD v1 audit](docs/prd-audit.md)

## Evidence policy

Routing frequency is association evidence, not proof of specialization.
A changed output from one intervention is not enough either. Stronger claims
require matched inputs, a task evaluator, repeated runs, negative controls,
target-exercise evidence, and successful restoration.

MoEAtlas records missing or unsupported evidence as missing or unsupported. It
does not fill gaps with visual or semantic guesses.
