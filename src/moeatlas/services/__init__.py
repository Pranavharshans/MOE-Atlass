"""Shared application services used by the CLI, the Python API, and the future local server."""

from __future__ import annotations

from .datasets import (
    DATASET_COLUMN_ROLES,
    DATASET_READER_SCHEMA_VERSION,
    DatasetReadError,
    DatasetRow,
    plan_dataset_batches,
    project_dataset_rows,
    read_dataset_rows,
    resolve_dataset_location,
    validate_column_mapping,
)
from .run_engine import (
    EXECUTION_PROGRESS_STAGE,
    ROW_FAILURE_KINDS,
    RUN_ENGINE_SCHEMA_VERSION,
    ExecutionOutcome,
    RowFailure,
    RowRecord,
    RowResult,
    RunEngineError,
    execute_row_schedule,
)
from .run_inputs import (
    RUN_INPUTS_SCHEMA_VERSION,
    RunInputError,
    plan_input_batches,
    prepare_input_rows,
    prepare_prompt_rows,
)
from .workspace import (
    WorkspaceSnapshot,
    initialize_workspace,
    open_workspace,
    query_runs,
    record_run_record,
    register_run,
    sync_runs_from_shards,
)

__all__ = [
    "DATASET_COLUMN_ROLES",
    "DATASET_READER_SCHEMA_VERSION",
    "EXECUTION_PROGRESS_STAGE",
    "ROW_FAILURE_KINDS",
    "RUN_ENGINE_SCHEMA_VERSION",
    "RUN_INPUTS_SCHEMA_VERSION",
    "DatasetReadError",
    "DatasetRow",
    "ExecutionOutcome",
    "RowFailure",
    "RowRecord",
    "RowResult",
    "RunEngineError",
    "RunInputError",
    "WorkspaceSnapshot",
    "execute_row_schedule",
    "initialize_workspace",
    "open_workspace",
    "plan_dataset_batches",
    "plan_input_batches",
    "prepare_input_rows",
    "prepare_prompt_rows",
    "project_dataset_rows",
    "query_runs",
    "read_dataset_rows",
    "record_run_record",
    "register_run",
    "resolve_dataset_location",
    "sync_runs_from_shards",
    "validate_column_mapping",
]
