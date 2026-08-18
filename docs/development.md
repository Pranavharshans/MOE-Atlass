# Development workflow

MoEAtlas is built feature by feature. The normal loop is:

1. define one small contract against the PRD;
2. write model-free tests first where practical;
3. implement the smallest coherent change;
4. run the relevant tests and static checks;
5. document anything that remains model- or GPU-dependent;
6. hand the verified file set to the Git worker for its own commit and push.

## Foundation commands

From the repository root:

```bash
uv sync --extra dev
uv run --locked pytest -q
uv run --locked ruff check src tests
uv run --locked moeatlas doctor
uv run --locked moeatlas doctor --json
```

The manifest test suite uses the required Pydantic v2 runtime. It does not
load PyTorch, Transformers, a tokenizer, or a model checkpoint. The diagnostic
command only checks whether optional packages can be found; it does not import
them or inspect a GPU.

The static discovery suite uses a small standard-library MoE-shaped fixture.
It tests deterministic structure reports, confidence evidence, strict JSON
round-trips, and the scanner's read-only boundary. It does not execute a
model, inspect tensor values, or certify a real architecture.

The implementation keeps three internal boundaries legible: structural
surface collection, heuristic candidate/fact scoring, and public report
assembly. These are internal modules, not plugin or distribution boundaries.

## Model-free test rules

Tests that can run on every contributor machine belong in the normal suite.
They should use small in-memory values, deterministic fixtures, and temporary
directories. They must not reach the Hugging Face Hub, download checkpoints,
require CUDA, or infer model support from package presence.

Model-dependent tests are not skipped silently. Add them to the
[validation ledger](model-validation-ledger.md) with a status of `deferred`,
then run them only in the final VM phase when the user provisions that
environment.

The discovery scanner is intentionally a dry-run-free boundary. Do not add
hooks, forward calls, or a checkpoint loader to it. Static semantic adapters
may consume its report through the separate `moeatlas.adapters` protocol, but
they must preserve `[STRUCTURE]` capability and cannot perform runtime model
actions. Runtime capture and architecture certification require separate
validation evidence.

Probe plans and the passive hook manager are shipped boundaries. Keep plans
JSON-serializable and callbacks outside the plan object. Their model-free
tests use the standard-library fixture for lifecycle behavior; real PyTorch
signature, equivalence, and GPU checks remain deferred to MV-04 and the final
VM. Future runtime work must preserve the same explicit boundaries.

## Packaging notes

`pyproject.toml` uses a `src/` layout and exposes the `moeatlas` console
script. Pydantic is the only required runtime dependency in the foundation.
Optional developer and model extras are intentionally explicit so a future
environment can opt into them without changing the default model-free test
contract.
