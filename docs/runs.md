# Run identity, provenance, and lifecycle

The `moeatlas.runs` package holds the model-neutral contracts that describe a
run: what it intends to execute and which lifecycle state it is in. The
contracts are strict Pydantic v2 manifests. They never load a model, read a
dataset, contact the network, execute a forward pass, or touch storage; the
prompt/dataset run engine (a later slice) drives them, and the workspace
catalog (also later) persists them.

## Run specification

`RunSpecification` is the immutable, content-addressed intent for exactly one
run (`manifest_type="run_specification"`, `schema_version="1.0"`). Its
canonical `run_key` has the `run:<64 lowercase hex>` form and is derived from
the identity-bearing groups only:

| Group | Contract | PRD §9.3 group | In identity |
| --- | --- | --- | --- |
| `model` | `ModelProvenance`: `loading_plan_id` (`loadplan:<hex>`), resolved `model_id`/`model_revision`, optional `config_hash`/`tokenizer_revision`, `quantization` | Model | yes |
| `data` | `DataProvenance`: prompt or dataset input descriptor, row count, task labels, preprocessing; `.fingerprint` property returns `data:<hex>` | Data | yes |
| `generation` | `GenerationConfig`: seed, sampling bounds, stop sequences | Generation | yes |
| `probe` | `ProbeProvenance`: `probe_plan_id` (`plan:<hex>`), capture level 0–5, intervention opt-in | Probe plan | yes |
| `adapter` | `AdapterProvenance`: paired name/version plus optional inspection fingerprint | — | yes |
| `privacy` | `PrivacyPolicy`: token-text policy (default redacted), raw-payload retention, export allowance | — | yes |
| `intervention` | `InterventionLineage`: baseline run key, recipe fingerprint, operation, targets | Intervention | yes |
| `replication` | integer replica index (default 0) so replicas get distinct keys deterministically | Identity | yes |
| `workspace`, `tags`, `created_at`, `created_by` | caller metadata | Identity | no |
| `execution` | `ExecutionEnvironment`: observed versions/device metadata recorded by the engine | Execution | no |

Metadata is deliberately excluded from the key: tags, workspace labels,
timestamps, and observed environments describe where or when a specification
was used, not what it is. Digests are computed with the shared canonical-JSON
machinery, so keys are stable across processes and hash seeds.

Input descriptors are identity only:

Descriptors never fetch data by themselves; the later read step may access a
Hub repository only when `allow_downloads=true` is explicitly set.

- `PromptInputSpec` is either chat messages (`ChatMessage(role, content)`) or
  raw text, never both.
- `DatasetInputSpec` describes JSONL/CSV/Parquet/text/iterable/HF-style data by
  location label, optional revision, optional caller-supplied content digest,
  column mapping, row cap/batch/shuffle/seed, and generation vs teacher-forced
  mode. It never reads the data; content digests are supplied by the caller.

Cross-group rules: an `InterventionLineage` requires a probe plan with
`intervention_opt_in=True`; adapter name/version must be paired; digests use
the exact `sha256:<64 hex>` shape. Supplying a non-derived `run_key` is a
validation error naming the expected key.

## Lifecycle state machine

`RunRecord` (`manifest_type="run_record"`) is a frozen snapshot binding a run
key to its lifecycle state, attempt counter, progress, failure, cancellation,
and caller-supplied `updated_at`. Every accepted action produces a new fully
revalidated record via `apply(record, action, ...)`; illegal actions raise
`RunLifecycleError` with the fixed message
`illegal run transition: <state> cannot <action>`.

States: `planned`, `provisioning`, `running`, `finalizing`, `completed`,
`failed`, `cancelling`, `cancelled`.

| Action | From → To |
| --- | --- |
| `start` | planned → provisioning |
| `begin_execution` | provisioning → running |
| `update_progress` | provisioning/running/finalizing self-loop |
| `finalize` | running → finalizing |
| `complete` | finalizing → completed |
| `fail` | provisioning/running/finalizing → failed |
| `request_cancellation` | planned/provisioning/running/finalizing → cancelling |
| `cancel` | cancelling → cancelled |
| `retry` | failed/cancelled → planned (attempt + 1) |

Payload rules: `fail` requires a `RunFailure` (fixed error-kind vocabulary:
dependency, validation, execution, storage, interruption, unknown); the
cancellation actions require a `RunCancellation`; `update_progress` requires
`RunProgress` and enforces monotonic completed-unit counts within one stage
through `advance_progress` (stage changes reset freely). Terminal states accept
only `retry`; `completed` accepts nothing.

Record invariants mirror the transitions: failed/cancelled/cancelling records
carry exactly the matching record type, completed records carry neither, and
planned runs have no progress. Serializable domain state is fully separated
from process-local model/tokenizer/runtime handles, which live in the runtime
layer.

## Bounded dataset reading

`moeatlas.services.datasets` is the first engine step over those descriptors:
`read_dataset_rows(descriptor, *, base_directory, max_rows, max_row_bytes,
max_file_bytes, duckdb)` turns a `DatasetInputSpec` into a deterministic
tuple of frozen `DatasetRow(index, values)` records for the file-backed
formats — JSONL, CSV, Parquet, and text (one `{"text": line}` row per line) —
plus `hf_datasets`, which means either an existing local snapshot directory
read in sorted filename order with a single data format or an explicit Hub
repository/split streamed through the optional `datasets` package. Hub access
requires `allow_downloads=true`; relative locations still require an explicit
local `base_directory`, and local formats never reach the network. DuckDB is
imported lazily and only for Parquet members; the Hub reader is imported lazily
and is bounded by the same row and canonical-row-byte budgets.

Reading is bounded and deterministic: strict row, per-row canonical-JSON
byte, and per-file byte budgets (exceeding any raises `DatasetReadError` at
stage `budget`); rows keep file order with assigned indices. Failures use
the fixed stage vocabulary `descriptor`, `dependency`, `format`, `budget`,
`read` with the message `dataset read failed at <stage>`. Column mappings
are validated against the fixed v1 role vocabulary
`("domain", "prompt", "reference", "task")` and their target columns must
exist in every row; `project_dataset_rows` applies the mapping onto those
roles. `plan_dataset_batches(total_rows, *, batch_size, sample_cap, shuffle,
seed)` derives sample selection, shuffles, and batch splits from SHA-256
ordering keys over `(purpose, seed, index)`, so identical arguments produce
identical schedules on every platform and process; shuffling requires a
seed, and an unseeded sample cap keeps the deterministic first-cap prefix.
`iterable` descriptors carry no file location and are rejected at the reader
boundary; the run engine consumes iterables directly.

## Input preparation

`moeatlas.services.run_inputs` closes the input-preparation half of the
engine: `prepare_input_rows(spec, ...)` turns either input descriptor into
the exact `{row_index: values}` mapping `execute_row_schedule` consumes, and
`plan_input_batches(spec, total_rows)` derives the schedule the descriptor
implies — so the execution core never branches on input kind. Prompt specs
become exactly one row: raw text as `{"prompt": text}`, chat messages as
`{"messages": [{"role", "content"}, ...]}` in declared order, canonically
encodable within `max_input_bytes` (`RunInputError`, stages `spec`,
`format`, `budget`). Dataset descriptors compose the bounded reader with
task-role projection when a column mapping is declared, preserving
read-order indices; their schedules apply the descriptor's own
sample/batch/shuffle/seed settings through the SHA-256-keyed planner.
Preparation is deterministic and touches no clock, no randomness, no
network, and no model dependency.

## Deterministic execution core

`moeatlas.services.run_engine` is the model-neutral execution heart:
`execute_row_schedule(schedule, *, executor, row_values, should_cancel,
max_result_bytes)` drives one planned batch schedule through a
caller-supplied row executor. Each row is invoked as
`executor(row_index=..., batch_index=..., values=...)`; real adapters plug in
as executors later, and local tests use fake runtimes.

A row failure is evidence, not a run death. Controlled failures are declared
by raising `RowFailure(kind, message)` with a kind from the same fixed
vocabulary as `RunFailure.error_kind`; any other `Exception` becomes an
`execution` failure carrying only the exception class name;
`KeyboardInterrupt` and `SystemExit` always propagate. Results must be
string-keyed mappings canonically encodable within `max_result_bytes`;
violations become per-row `validation` failures. Progress snapshots use the
lifecycle's `RunProgress` with the single engine stage `executing`, emitted
cumulatively after every batch and checked through `advance_progress`.
Cancellation is cooperative: `should_cancel` is consulted before each row,
and observing it freezes a cancelled outcome that preserves everything
already executed (`cancelled_before_row` names the first unexecuted row).
The returned `ExecutionOutcome` is a strict frozen record — results,
failures, progress trail, counts — whose `status` property suggests the
terminal lifecycle state: `cancelled` wins, then `failed` when no row
succeeded, else `completed`. Schedules are validated exactly (non-empty
batches, non-negative integer indices, no duplicate rows) and every input
violation raises `RunEngineError` at stage `contract`. The core performs no
clock reads, no randomness, no network access, and no publication; the
headless service below is the composition and publication layer.

## Headless run service

`moeatlas.services.run_service` composes the whole model-neutral pipeline:
`execute_specification(specification, *, executor, ...)` prepares rows with
`prepare_input_rows`, plans the schedule with `plan_input_batches`, executes
batch by batch through the core (via its additive `batch_offset`, so evidence
keeps plan-level batch indices), and projects the outcome onto deterministic
lifecycle records with `apply()` — planned → start → begin_execution → one
`update_progress` per batch → a terminal transition chosen only by
`ExecutionOutcome.status`: cancelled runs request cancellation then cancel,
failed runs (no successful row) fail with `derive_run_failure`, everything
else finalizes and completes. A run that finishes with mixed results and
failures still completes; the per-row evidence stays in the outcome.
Timestamps come solely from the caller's `at` string, so the service
never reads a clock, and every produced record is emitted in order to an
optional `on_record` callback, which is the hook CLI and server surfaces
subscribe to.

Checkpoints make durability batch-granular. With `checkpoint_directory` set,
each completed batch atomically rewrites `<digest>.checkpoint.json` — a
canonical JSON document (`manifest_type: run_checkpoint`) carrying the run
key, `next_batch_index`, total batches, and all results and failures so far —
written through staging plus rename so crashes never leave partial files.
`load_checkpoint(path)` fully validates a document into a frozen
`RunCheckpoint`; `resume_from=path` continues an interrupted run of the same
run key without re-executing durable batches (work inside an incomplete batch
is re-executed, because durability is batch-granular). Foreign or drifted
checkpoints, malformed documents, and unwritable destinations raise
`RunServiceError` at stages `checkpoint` or `lifecycle`. The merged report is
a frozen `RunExecutionReport` carrying the combined outcome, the full record
trail (`final_record` is terminal), the checkpoint path, and the resume
cursor. Publication stays explicit: `publish_run_report(workspace, report)`
records the terminal state into the workspace catalog, auto-registering
unknown run keys. Like the layers beneath it, the service never branches on
input kind or model family, and package tests exercise it exclusively over
fake executors and temporary fixtures.

## Real-model executor

`moeatlas run` can drive a genuine model over planned batches through the
built-in `transformers-routing` executor (`moeatlas.executors`). The CLI
resolves `--executor` names from the built-in registry first and falls back to
the `moeatlas.executors` entry-point group, so third-party executors publish
exactly like adapter plugins while fake test executors keep working unchanged.

The executor binds one validated `LoadingPlan` and resolves it exclusively
through existing seams: HF/local loading goes through `moeatlas.runtime`
(`load_huggingface`/`load_local`, still lazily importing torch/transformers
only at execution time), structure evidence comes from the static scanner over
the loaded model, and each planned prompt row is tokenized through the loaded
tokenizer and executed by the generic structure-driven capture composition
(see `docs/runtime.md`) — emitting `TokenEvent`/`RoutingEvent` rows validated
by `moeatlas.events`. Row failures stay evidence: missing tokenizers,
unsupported plan sources, and load failures are declared dependency/validation
`RowFailure`s with fixed messages, while capture mismatches surface as
execution failures without partial publication.

Publication is automatic. After the run service publishes the terminal
lifecycle record, the executor appends its accumulated events as one immutable
routing shard through `append_routing_shard` (unchanged) and reconciles the
workspace catalog via `rebuild_catalog`, so `/api/runs` sees the completed run
with shard counts without any manual step. Token text stays redacted unless
the caller opts in.

## Boundaries and deferred evidence

These contracts bind artifacts by identifier; they do not verify that a
loading plan resolves, that a dataset exists, or that a probe plan matches an
inspection. Those checks belong to the application services and engine slices.
Package tests prove contract behavior only — real tokenizer/model/generation
equivalence and performance remain deferred MV evidence in the
[model-validation ledger](model-validation-ledger.md).
