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

## Loading-plan scan

Resolved Hugging Face or local runtime plans can be supplied explicitly:

```bash
moeatlas scan --loading-plan plan.json
moeatlas scan --loading-plan plan.json --output report.json --force
```

`MODEL` and `--loading-plan` are mutually exclusive, and exactly one is
required. The file must be one strict `LoadingPlan` JSON document with a
`HuggingFaceSource` or `LocalSource`, immutable model/tokenizer resolution
evidence, and the plan's canonical `plan_id`. The CLI does not independently
infer a model ID, resolve or read a local path, resolve a branch, alter
offline/download policy, or reconstruct the plan. Canonical schema validation
may lexically normalize `LocalSource.path`, but performs no filesystem
resolution. The CLI preflights the document and then passes that validated
plan unchanged to `moeatlas.runtime.load_and_scan()`.

Derived plan security warnings are printed to stderr before runtime dispatch;
they are never copied from raw input. Runtime loading and static discovery
finish, including cleanup, before a report is published. Failures return
status 2 with concise stderr and leave an existing output and temporary files
untouched. KeyboardInterrupt/SystemExit remain control-flow exceptions.

Direct non-fixture `MODEL` values remain rejected. Phase 0 does not inspect local paths
or caches. The plan-file path enables the explicit runtime seam, but real
checkpoint fidelity, network/cache behavior, and GPU certification remain
deferred in MV-01/MV-02 and the final VM. `doctor` and `--version` remain
model-free diagnostics.

## Bounded routing heatmap

Feature 22 adds one explicit publication command over an existing Feature 19
workspace and a caller-supplied Feature 20 inspection document:

```bash
moeatlas heatmap WORKSPACE \
  --inspection inspection.json \
  --run-key run-1 \
  --metric load_ratios \
  --max-inspection-bytes 1000000 \
  --max-routing-rows 1000000 \
  --max-source-bytes 100000000 \
  --max-matrix-cells 100000 \
  --output routing-load.html
```

All four budgets are required canonical positive decimal integers: ASCII
digits only, no sign, whitespace, separators, leading zero, or zero value.
The output path must end in the exact lowercase `.html` suffix; an existing
destination is refused unless `--force` is supplied.
The output destination is preflighted before the inspection file, workspace,
or optional store dependency is touched. The inspection must be a regular
non-symlink file and is read at most `--max-inspection-bytes` bytes before
strict `AdapterInspection` validation. The command calls bounded aggregation
and rendering exactly once, then reuses the existing atomic report writer;
that writer is the existing `write_report_atomic()` path;
`--force` is passed only to that writer. Success writes no stdout and emits
the fixed stderr prefix `saved routing heatmap to ` after atomic publication.
Ordinary failures return status 2 without a partial artifact; failed atomic
writes clean their temporary file and preserve a prior destination. Missing
DuckDB is reported as the fixed dependency-stage error. `KeyboardInterrupt`
and `SystemExit` remain control-flow exceptions. Install the optional `store`
extra for DuckDB; no model, tokenizer, browser, network, cache, or generation
path is involved.
