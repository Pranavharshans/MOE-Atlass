# Command-line scan

The Phase 0 CLI exposes one truthful, model-free scan source:

```bash
uv run --locked moeatlas scan fixture:synthetic
```

`fixture:synthetic` is a deterministic standard-library MoE surface with a
fixed model manifest (`model:fixture/synthetic-moe@v1`), configuration hash,
tokenizer identity, CPU device map, and explicit non-certified fixture
provenance. The command writes the complete `DiscoveryReport` as compact JSON
to stdout and writes nothing else to stdout on success.

To save the same JSON bytes to a file:

```bash
uv run --locked moeatlas scan fixture:synthetic --output report.json
uv run --locked moeatlas scan fixture:synthetic --output report.json --force
```

The parent directory must already exist. Existing files are refused unless
`--force` is supplied. Reports are first written to a same-directory temporary
file and then atomically replaced; failed writes clean up the temporary file.
With `--output`, only a concise confirmation is written to stderr and the JSON
is not duplicated on stdout.

Any other `MODEL` value fails with a nonzero status and a concise message. Phase
0 does not inspect local paths or caches, contact Hugging Face, load
checkpoints, or substitute the fixture. Real Hugging Face/local loading is a
Phase 1 task and remains deferred in MV-01/MV-02. `doctor` and `--version`
remain model-free diagnostics.
