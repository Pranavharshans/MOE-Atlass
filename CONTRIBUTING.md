# Contributing to MoEAtlas

Thanks for helping build an evidence-first MoE observability tool.

## Development contract

MoEAtlas is implemented one small feature at a time. Each feature should have
model-free tests where possible, documentation of its validation boundary, and
an isolated commit. The Git worker pushes each verified feature separately.

Do not download model checkpoints during ordinary repository development.
Model-dependent checks are intentionally deferred to the final VM validation
phase and must be recorded in
[the model-validation ledger](docs/model-validation-ledger.md).

## Local setup

Python 3.11 or newer and `uv` are the supported foundation tools:

```bash
uv venv
uv pip install -e '.[dev]'
```

The model extra is not part of the normal setup. Installing it does not fetch
checkpoints, but it is not needed for the current foundation feature.

## Tests and checks

The baseline test command has no third-party test dependency:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

If the optional developer tools are installed, also run:

```bash
PYTHONPATH=src python -m pytest
ruff check src tests
```

Do not describe a passing package test as evidence that a model adapter works.
For model/GPU work, capture the model ID and immutable revision, Python and
runtime versions, hardware, exact command, output location, result, and any
limitations in the ledger before marking the check complete.

## Scope discipline

Keep the early implementation as one installable `moeatlas` package with
clear internal modules. Add a separately distributable package only when a
stable public boundary and a concrete user need justify it. Preserve passive
behavior by default, record provenance, and avoid adding a model-specific
conditional to the core when an adapter boundary is the right home.
