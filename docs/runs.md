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
plus `hf_datasets`, which means an existing local snapshot directory read in
sorted filename order with a single data format. Descriptors never fetch
data: relative locations require an explicit local `base_directory`,
absolute paths pass through, and nothing ever reaches the network. DuckDB is
imported lazily and only for Parquet members; every other format reads
without it.

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

## Boundaries and deferred evidence

These contracts bind artifacts by identifier; they do not verify that a
loading plan resolves, that a dataset exists, or that a probe plan matches an
inspection. Those checks belong to the application services and engine slices.
Package tests prove contract behavior only — real tokenizer/model/generation
equivalence and performance remain deferred MV evidence in the
[model-validation ledger](model-validation-ledger.md).
