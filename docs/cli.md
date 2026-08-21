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
workspace and a caller-supplied inspection document. Feature 28 makes the
aggregation and rendering delegates model-neutral: complete Mixtral, Qwen3.5,
and future-family structural inspections use the same command, while the
historical Mixtral Python names remain identity aliases.

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

## Bounded routing-run inventory

Feature 23 exposes a deterministic read-only inventory command:

```bash
moeatlas routing-runs WORKSPACE \
  --max-runs 100 --max-shards 1000 \
  --max-event-rows 1000000 --max-source-bytes 100000000

moeatlas routing-runs WORKSPACE \
  --max-runs 100 --max-shards 1000 --max-event-rows 1000000 \
  --max-source-bytes 100000000 --output inventory.json
```

Every budget is required and must be a canonical positive decimal integer.
Without `--output`, compact inventory JSON plus one newline is written only to
stdout. With `--output`, the exact lowercase `.json` suffix is required, the
parent must already exist, and publication delegates once to the existing
`write_report_atomic()` path; existing files require `--force`. Success emits
only `saved routing run inventory to <path>` on stderr in file mode. `--force`
without an output is rejected before workspace traversal or DuckDB import.

The command inventories committed Feature 19 shards, including exact run and
shard ordering, token/routing totals, source-byte totals, and redaction policy.
It does not create a catalog, infer a latest run, inspect model/cache files,
load a model, read token text, use a browser/network, or write the workspace.
Install the optional DuckDB `store` extra for non-empty inventories. Missing or
corrupt sources retain fixed Feature 19 stage messages; malformed index and
budget failures use `routing run inventory failed at index|budget`; unexpected
failures use the fixed `moeatlas routing-runs: routing run inventory failed`.
KeyboardInterrupt and SystemExit remain control-flow exceptions. The command
is EXPERIMENTAL and does not alter MV-01 through MV-08.
