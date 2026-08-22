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
strict validation. Both certified `AdapterInspection` documents and universal
structure inspections (the `universal_routing_inspection` manifest type
derived from a `[STRUCTURE]` discovery report) are accepted through the same
flag; see [analysis](analysis.md). The command calls bounded aggregation
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

## Bounded routing-run comparison

Feature 31 adds one explicit comparison publication command over an existing
Feature 19 workspace and a caller-supplied inspection document:

```bash
moeatlas compare WORKSPACE \
  --inspection inspection.json \
  --baseline-run-key run-1 \
  --comparison-run-key run-2 \
  --metric ratio_deltas \
  --max-inspection-bytes 1000000 \
  --max-routing-rows 1000000 \
  --max-source-bytes 100000000 \
  --max-matrix-cells 100000 \
  --output routing-comparison.html
```

The command composes the existing bounded seams exactly once each: two
Feature 28 aggregations (one per run key), one Feature 29 comparison, and one
Feature 30 rendering, then reuses the atomic report writer. The two run keys
must differ and are checked before budget parsing or output preflight. All
four budgets are required canonical positive decimal integers; the output must
use the exact lowercase `.html` suffix with `--force` semantics identical to
the heatmap command. The inspection read is the same bounded non-symlink path
and accepts both certified `AdapterInspection` documents and universal
structure inspections. Success emits only
`saved routing comparison to <path>` on stderr.

Failures reuse the fixed stage messages: missing or uncommitted runs report
`routing load aggregation failed at source`; incomparable universes (for
example different token counts) and all other unexpected failures use the
fixed generic `moeatlas compare: routing comparison failed` without echoing
input details. Missing DuckDB is reported as the fixed dependency-stage error.
KeyboardInterrupt and SystemExit remain control-flow exceptions. The command
is EXPERIMENTAL: it creates no model, tokenizer, browser, network, cache,
catalog, ranking, or specialization surface, and does not alter MV-01 through
MV-08.

## Adapter plugin listing

`moeatlas adapters list` prints the versioned adapter plugin registry (see
[adapters](adapters.md)): every built-in adapter and third-party
`moeatlas.adapters` entry-point plugin, one deterministic line per record
carrying name, version, policy status (`enabled`/`disabled`), source
(`builtin`/`entry_point`), publishing distribution, and declared
architecture families. Discovery is metadata-only — a plugin module is
imported just enough to read its descriptor; no model is loaded, nothing is
downloaded, and no network path exists.

Policy flags compose: `--builtin-only` treats entry-point plugins as
untrusted (still listed, marked disabled), `--enable NAME` allow-lists
specific plugins (repeatable; unlisted plugins become disabled),
`--disable NAME` force-disables names (repeatable), and `--family FAMILY`
keeps only enabled records whose families serve that architecture family.
A name passed to both `--enable` and `--disable` exits 2 with the fixed
message `moeatlas adapters list: invalid adapter policy or family filter`.
Suppressed collisions and isolated plugin failures print fixed lines on
stderr while the listing itself still succeeds. `--json` emits the exact
canonical `moeatlas.adapter_registry` document instead of text.

## Headless run execution

`moeatlas run WORKSPACE` builds one content-addressed `RunSpecification`
from a validated loading plan plus exactly one input form and executes it
through the shared run service (see [runs](runs.md)). The loading plan is
the same strict document accepted by the scan command and must carry
immutable model and tokenizer resolution evidence; the model provenance of
the specification is projected from that resolution, never inferred.

The input form is exactly one of `--prompt TEXT` or `--dataset
DESCRIPTOR.json`; passing both or neither exits 2 with the fixed message
`moeatlas run: exactly one of --prompt or --dataset is required`. Dataset
descriptors are validated `DatasetInputSpec` documents whose locations
resolve relative to the workspace directory. The executor is mandatory and
never built in: `--executor NAME` resolves a callable plugin from the
`moeatlas.executors` entry-point group, so real model execution belongs to
explicitly installed adapter plugins and no command downloads a model
implicitly. Unregistered executors, unloadable plugins, and non-callable
objects exit 2 with fixed messages.

Optional flags: `--at TIMESTAMP` supplies every lifecycle timestamp
(the CLI never reads a clock), `--created-by LABEL`,
`--checkpoint-directory DIR` for batch-granular crash-safe checkpoints,
`--resume-from CHECKPOINT.json`, and canonical positive decimal budgets
(`--max-input-bytes`, `--max-rows`, `--max-row-bytes`,
`--max-result-bytes`). Success publishes the terminal lifecycle record to
the workspace catalog and prints the run key, final state, unit counts,
resumed batch, and checkpoint path when one exists. Failures collapse to
fixed safe errors: known service stages print
`run service failed at {checkpoint|lifecycle}` while anything else prints
the generic `moeatlas run: run execution failed`.

## Run evidence export

`moeatlas export WORKSPACE RUN_KEY --output DEST_DIR` exports every
committed shard of one routing run as the canonical tamper-evident evidence
bundle from the storage layer (see [storage](storage.md)): staged sibling
temporary directory, fsynced members, manifest written last, atomic rename.
The destination must be nonexistent or an empty real directory; two exports
of the same committed run state are byte-identical. `--format bundle` is
the only format today because it is the only canonical evidence shape the
storage layer publishes. Optional canonical positive decimal budgets are
`--max-event-rows` and `--max-file-bytes`. Success prints the exported
counts and the manifest sha256; failures reuse the fixed-stage vocabulary
(`run bundle failed at {stage}`) or collapse to the generic
`moeatlas export: run export failed` without echoing input details.
