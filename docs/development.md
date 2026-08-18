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
uv venv
uv pip install -e '.[dev]'
PYTHONPATH=src python -m unittest discover -s tests -v
moeatlas doctor
moeatlas doctor --json
```

The baseline test command uses only the Python standard library. It does not
load PyTorch, Transformers, a tokenizer, or a model checkpoint. The diagnostic
command only checks whether optional packages can be found; it does not import
them or inspect a GPU.

## Model-free test rules

Tests that can run on every contributor machine belong in the normal suite.
They should use small in-memory values, deterministic fixtures, and temporary
directories. They must not reach the Hugging Face Hub, download checkpoints,
require CUDA, or infer model support from package presence.

Model-dependent tests are not skipped silently. Add them to the
[validation ledger](model-validation-ledger.md) with a status of `deferred`,
then run them only in the final VM phase when the user provisions that
environment.

## Packaging notes

`pyproject.toml` uses a `src/` layout and exposes the `moeatlas` console
script. The runtime dependency list is empty in the foundation. Optional
developer and model extras are intentionally explicit so a future environment
can opt into them without changing the default install contract.
