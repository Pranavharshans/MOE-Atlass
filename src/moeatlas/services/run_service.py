"""Headless run-engine service surface over specs, preparation, and lifecycle.

This module composes the model-neutral run pipeline end to end: a
:class:`~moeatlas.runs.specs.RunSpecification` is prepared through
``prepare_input_rows``, scheduled through ``plan_input_batches``, executed one
batch at a time through ``execute_row_schedule`` (so durability boundaries are
batch-granular), and projected onto deterministic
:class:`~moeatlas.runs.lifecycle.RunRecord` transitions. Checkpoints are
canonical JSON files written atomically after every completed batch, and
``resume_from`` continues an interrupted run without re-executing durable
batches. Nothing here reads a clock (timestamps are caller-supplied), touches
the network, or branches on a model family; real adapters plug in as row
executors later, exactly as with the execution core.

Failure philosophy carries over unchanged: per-row failures stay evidence in
the outcome while the run itself fails only when no row succeeded. A run that
finishes with mixed results and failures still completes — the evidence
records carry the failures. Cancellation preserves executed work in both the
outcome and the durable checkpoint.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..runs.lifecycle import (
    RunAction,
    RunCancellation,
    RunFailure,
    RunLifecycleError,
    RunProgress,
    RunRecord,
    RunState,
    apply,
)
from ..runs.specs import RunSpecification, parse_run_key
from ..store.catalog import RunRegistryEntry
from .run_engine import (
    EXECUTION_PROGRESS_STAGE,
    RUN_ENGINE_SCHEMA_VERSION,
    ExecutionOutcome,
    RowRecord,
    RowResult,
    execute_row_schedule,
    sanitize_failure_message,
    serialize_row_failures,
)
from .run_inputs import plan_input_batches, prepare_input_rows
from .workspace import record_run_record

RUN_SERVICE_SCHEMA_VERSION = "1.0"
"""Schema version of the headless run-service contracts."""

CHECKPOINT_SCHEMA_VERSION = "1.0"
"""Schema version of the canonical run-checkpoint documents."""

CHECKPOINT_MANIFEST_TYPE = "run_checkpoint"
"""Manifest type marker inside checkpoint documents."""

_MAX_FAILURE_MESSAGE = 500

_STAGES = frozenset({"checkpoint", "lifecycle"})


class RunServiceError(RuntimeError):
    """Safe fixed-stage failure for the headless run service."""

    def __init__(
        self, stage: str, message: str | None = None, *, cause: BaseException | None = None
    ) -> None:
        if stage not in _STAGES:
            raise ValueError("run service error stage is not supported")
        self.stage = stage
        text = f"run service failed at {stage}"
        if message:
            text = f"{text}: {message}"
        super().__init__(text)
        if cause is not None:
            self.__cause__ = cause


@dataclass(frozen=True, slots=True)
class RunCheckpoint:
    """Durable batch-granular execution state for one run key."""

    schema_version: str
    run_key: str
    next_batch_index: int
    total_batches: int
    results: tuple[RowResult, ...]
    failures: tuple[RowRecord, ...]

    def __post_init__(self) -> None:
        if type(self.next_batch_index) is not int or isinstance(
            self.next_batch_index, bool
        ) or type(self.total_batches) is not int or isinstance(self.total_batches, bool):
            raise TypeError("checkpoint cursors must be integers")
        if not isinstance(self.results, tuple) or not isinstance(self.failures, tuple):
            raise TypeError("checkpoint entries must be tuples")
        for result in self.results:
            if not isinstance(result, RowResult):
                raise TypeError("checkpoint results must be RowResult values")
        for failure in self.failures:
            if not isinstance(failure, RowRecord):
                raise TypeError("checkpoint failures must be RowRecord values")
        parse_run_key(self.run_key)
        if self.next_batch_index < 0 or self.total_batches < 0:
            raise ValueError("checkpoint cursors must be non-negative")
        if self.next_batch_index > self.total_batches:
            raise ValueError("next_batch_index cannot exceed total_batches")
        seen: set[int] = set()
        for entry in (*self.results, *self.failures):
            if entry.batch_index >= self.next_batch_index:
                raise ValueError("checkpoint entries must belong to completed batches")
            if entry.row_index in seen:
                raise ValueError(f"checkpoint repeats row {entry.row_index}")
            seen.add(entry.row_index)


@dataclass(frozen=True, slots=True)
class RunExecutionReport:
    """Complete deterministic result of one headless specification run."""

    schema_version: str
    run_key: str
    outcome: ExecutionOutcome
    records: tuple[RunRecord, ...]
    checkpoint_path: str | None
    resumed_from_batch: int

    @property
    def final_record(self) -> RunRecord:
        return self.records[-1]

    @property
    def failure_evidence(self) -> tuple[dict[str, int | str], ...]:
        """Return bounded, sanitized row failures for a server/job surface.

        Checkpoints and the in-memory outcome retain their canonical row
        records.  This separate view is intentionally capped and redacted so
        callers can expose useful row/batch/kind/message evidence without
        leaking prompts, credentials, or local paths.
        """

        return serialize_row_failures(self.outcome.failures)

    @property
    def failure_summary(self) -> dict[str, int | str] | None:
        """Return one bounded summary for a failed execution, if any."""

        if not self.outcome.failures:
            return None
        first = self.outcome.failures[0]
        return {
            "kind": first.kind,
            "stage": EXECUTION_PROGRESS_STAGE,
            "count": len(self.outcome.failures),
            "message": sanitize_failure_message(first.message),
        }

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, ExecutionOutcome):
            raise TypeError("outcome must be an ExecutionOutcome")
        if not isinstance(self.records, tuple) or not self.records:
            raise ValueError("records must be a non-empty tuple of RunRecord values")
        for record in self.records:
            if not isinstance(record, RunRecord):
                raise TypeError("records must contain only RunRecord values")
        if type(self.resumed_from_batch) is not int or isinstance(
            self.resumed_from_batch, bool
        ) or self.resumed_from_batch < 0:
            raise ValueError("resumed_from_batch must be a non-negative integer")
        if self.checkpoint_path is not None and not isinstance(self.checkpoint_path, str):
            raise TypeError("checkpoint_path must be a string or None")


def build_initial_record(specification: RunSpecification) -> RunRecord:
    """Return the planned lifecycle record a specification starts from."""

    if not isinstance(specification, RunSpecification):
        raise TypeError(
            f"specification must be a RunSpecification, got {type(specification).__name__}"
        )
    return RunRecord(
        run_key=specification.run_key, specification_fingerprint=specification.run_key
    )


def derive_run_failure(failures: tuple[RowRecord, ...]) -> RunFailure:
    """Project ordered row-failure evidence onto one bounded run failure.

    The first recorded failure supplies the error kind. A single failure keeps
    its message verbatim; several failures collapse into a deterministic
    summary prefixed by their count so the record stays within its budget.
    """

    if type(failures) is not tuple or any(not isinstance(f, RowRecord) for f in failures):
        raise TypeError("failures must be a tuple of RowRecord values")
    if not failures:
        return RunFailure(
            stage=RunState.RUNNING,
            error_kind="unknown",
            message="run failed with no recorded row failures",
        )
    first = failures[0]
    if len(failures) == 1:
        message = first.message
    else:
        message = f"{len(failures)} row failures; first: {first.message}"
    return RunFailure(
        stage=RunState.RUNNING,
        error_kind=first.kind,
        message=message[:_MAX_FAILURE_MESSAGE],
    )


def _canonical_checkpoint_bytes(checkpoint: RunCheckpoint) -> bytes:
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "manifest_type": CHECKPOINT_MANIFEST_TYPE,
        "run_key": checkpoint.run_key,
        "next_batch_index": checkpoint.next_batch_index,
        "total_batches": checkpoint.total_batches,
        "results": [
            {
                "row_index": result.row_index,
                "batch_index": result.batch_index,
                "result": dict(result.result),
            }
            for result in checkpoint.results
        ],
        "failures": [
            {
                "row_index": failure.row_index,
                "batch_index": failure.batch_index,
                "kind": failure.kind,
                "message": failure.message,
            }
            for failure in checkpoint.failures
        ],
    }
    return json.dumps(
        payload, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    handle, staged_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name, suffix=".staging"
    )
    staged = Path(staged_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
        os.replace(staged, path)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise


def _write_checkpoint(
    directory: str | Path,
    *,
    run_key: str,
    next_batch_index: int,
    total_batches: int,
    results: list[RowResult],
    failures: list[RowRecord],
) -> Path:
    checkpoint = RunCheckpoint(
        schema_version=CHECKPOINT_SCHEMA_VERSION,
        run_key=run_key,
        next_batch_index=next_batch_index,
        total_batches=total_batches,
        results=tuple(results),
        failures=tuple(failures),
    )
    try:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        path = target / f"{parse_run_key(run_key)}.checkpoint.json"
        _atomic_write_bytes(path, _canonical_checkpoint_bytes(checkpoint))
    except OSError as exc:
        raise RunServiceError("checkpoint", cause=exc) from exc
    return path


def _require_entry(entry: object, field: str) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise RunServiceError("checkpoint", f"checkpoint {field} entries must be objects")
    return entry


def load_checkpoint(path: str | Path) -> RunCheckpoint:
    """Load and fully validate one canonical checkpoint document."""

    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise RunServiceError("checkpoint", cause=exc) from exc
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise RunServiceError("checkpoint", cause=exc) from exc
    if not isinstance(payload, dict):
        raise RunServiceError("checkpoint", "checkpoint documents must be objects")
    if payload.get("manifest_type") != CHECKPOINT_MANIFEST_TYPE:
        raise RunServiceError("checkpoint", "checkpoint manifest_type is unsupported")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise RunServiceError("checkpoint", "checkpoint schema_version is unsupported")

    def strict_int(value: object, field: str) -> int:
        if type(value) is not int or isinstance(value, bool):
            raise RunServiceError("checkpoint", f"checkpoint {field} must be an integer")
        return value

    run_key = payload.get("run_key")
    if not isinstance(run_key, str):
        raise RunServiceError("checkpoint", "checkpoint run_key must be a string")
    try:
        parse_run_key(run_key)
        next_batch_index = strict_int(payload.get("next_batch_index"), "next_batch_index")
        total_batches = strict_int(payload.get("total_batches"), "total_batches")
        if next_batch_index < 0 or total_batches < 0:
            raise ValueError("cursors must be non-negative")
        if next_batch_index > total_batches:
            raise ValueError("next_batch_index cannot exceed total_batches")
        raw_results = payload.get("results", [])
        raw_failures = payload.get("failures", [])
        if not isinstance(raw_results, list) or not isinstance(raw_failures, list):
            raise ValueError("results and failures must be lists")
        results = tuple(
            RowResult(
                row_index=_require_entry(entry, "results")["row_index"],
                batch_index=entry["batch_index"],
                result=entry["result"],
            )
            for entry in raw_results
        )
        failures = tuple(
            RowRecord(
                row_index=_require_entry(entry, "failures")["row_index"],
                batch_index=entry["batch_index"],
                kind=entry["kind"],
                message=entry["message"],
            )
            for entry in raw_failures
        )
        checkpoint = RunCheckpoint(
            schema_version=CHECKPOINT_SCHEMA_VERSION,
            run_key=run_key,
            next_batch_index=next_batch_index,
            total_batches=total_batches,
            results=results,
            failures=failures,
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise RunServiceError("checkpoint", cause=exc) from exc
    return checkpoint


def _transition(
    record: RunRecord,
    action: RunAction,
    *,
    at: str | None,
    on_record: Callable[[RunRecord], None] | None,
    **kwargs: Any,
) -> RunRecord:
    try:
        nxt = apply(record, action, at=at, **kwargs)
    except RunLifecycleError as exc:
        raise RunServiceError("lifecycle", cause=exc) from exc
    if on_record is not None:
        on_record(nxt)
    return nxt


def execute_specification(
    specification: RunSpecification,
    *,
    executor: Callable[..., Mapping[str, Any]],
    base_directory: str | Path | None = None,
    should_cancel: Callable[[], bool] | None = None,
    duckdb: Any = None,
    max_input_bytes: int = 65_536,
    max_rows: int = 10_000,
    max_row_bytes: int = 65_536,
    max_file_bytes: int = 100_000_000,
    max_result_bytes: int = 65_536,
    at: str | None = None,
    requested_by: str | None = None,
    cancellation_reason: str | None = None,
    on_record: Callable[[RunRecord], None] | None = None,
    checkpoint_directory: str | Path | None = None,
    resume_from: str | Path | None = None,
) -> RunExecutionReport:
    """Execute one specification headlessly and project it onto the lifecycle.

    The pipeline is fixed: prepare rows, plan the deterministic schedule, then
    execute batch by batch so each completed batch lands atomically in the
    checkpoint file when ``checkpoint_directory`` is given. Every lifecycle
    transition is emitted to ``on_record`` (when supplied) in order, starting
    from the planned record. ``resume_from`` loads a prior checkpoint of the
    same run key and skips its durable batches; work inside an incomplete
    batch is re-executed on resume because durability is batch-granular.
    Timestamps come solely from ``at`` — this function never reads a clock.
    """

    if not isinstance(specification, RunSpecification):
        raise TypeError(
            f"specification must be a RunSpecification, got {type(specification).__name__}"
        )
    if not callable(executor):
        raise TypeError("executor must be callable")
    if on_record is not None and not callable(on_record):
        raise TypeError("on_record must be callable or None")
    if should_cancel is not None and not callable(should_cancel):
        raise TypeError("should_cancel must be callable or None")

    rows = prepare_input_rows(
        specification.data.input,
        base_directory=base_directory,
        max_input_bytes=max_input_bytes,
        max_rows=max_rows,
        max_row_bytes=max_row_bytes,
        max_file_bytes=max_file_bytes,
        duckdb=duckdb,
    )
    schedule = plan_input_batches(specification.data.input, len(rows))
    total_batches = len(schedule)
    total_rows = sum(len(batch) for batch in schedule)

    cursor = 0
    results: list[RowResult] = []
    failures: list[RowRecord] = []
    if resume_from is not None:
        checkpoint = load_checkpoint(resume_from)
        if checkpoint.run_key != specification.run_key:
            raise RunServiceError("checkpoint", "checkpoint belongs to a different run key")
        if checkpoint.total_batches != total_batches:
            raise RunServiceError(
                "checkpoint", "checkpoint batch count does not match the planned schedule"
            )
        cursor = checkpoint.next_batch_index
        results.extend(checkpoint.results)
        failures.extend(checkpoint.failures)

    records: list[RunRecord] = [build_initial_record(specification)]
    if on_record is not None:
        on_record(records[0])
    record = _transition(records[0], RunAction.START, at=at, on_record=on_record)
    records.append(record)
    record = _transition(record, RunAction.BEGIN_EXECUTION, at=at, on_record=on_record)
    records.append(record)

    progress_snapshots: list[RunProgress] = []
    written_checkpoint: Path | None = None
    cancelled = False
    cancelled_before_row: int | None = None
    for batch_index, batch in enumerate(schedule):
        if batch_index < cursor:
            continue
        outcome = execute_row_schedule(
            (batch,),
            executor=executor,
            row_values=rows,
            should_cancel=should_cancel,
            max_result_bytes=max_result_bytes,
            batch_offset=batch_index,
        )
        results.extend(outcome.results)
        failures.extend(outcome.failures)
        if outcome.cancelled:
            cancelled = True
            cancelled_before_row = outcome.cancelled_before_row
            break
        snapshot = RunProgress(
            stage=EXECUTION_PROGRESS_STAGE,
            completed_units=len(results) + len(failures),
            total_units=total_rows,
        )
        record = _transition(
            record, RunAction.UPDATE_PROGRESS, at=at, on_record=on_record, progress=snapshot
        )
        records.append(record)
        progress_snapshots.append(record.progress)
        if checkpoint_directory is not None:
            written_checkpoint = _write_checkpoint(
                checkpoint_directory,
                run_key=specification.run_key,
                next_batch_index=batch_index + 1,
                total_batches=total_batches,
                results=results,
                failures=failures,
            )

    merged = ExecutionOutcome(
        schema_version=RUN_ENGINE_SCHEMA_VERSION,
        total_rows=total_rows,
        executed_rows=len(results) + len(failures),
        results=tuple(results),
        failures=tuple(failures),
        cancelled=cancelled,
        cancelled_before_row=cancelled_before_row,
        progress=tuple(progress_snapshots),
    )

    if merged.status == "cancelled":
        cancellation = RunCancellation(reason=cancellation_reason, requested_by=requested_by)
        record = _transition(
            record,
            RunAction.REQUEST_CANCELLATION,
            at=at,
            on_record=on_record,
            cancellation=cancellation,
        )
        records.append(record)
        record = _transition(
            record, RunAction.CANCEL, at=at, on_record=on_record, cancellation=cancellation
        )
        records.append(record)
    elif merged.status == "failed":
        record = _transition(
            record,
            RunAction.FAIL,
            at=at,
            on_record=on_record,
            failure=derive_run_failure(tuple(failures)),
        )
        records.append(record)
    else:
        record = _transition(record, RunAction.FINALIZE, at=at, on_record=on_record)
        records.append(record)
        record = _transition(record, RunAction.COMPLETE, at=at, on_record=on_record)
        records.append(record)

    return RunExecutionReport(
        schema_version=RUN_SERVICE_SCHEMA_VERSION,
        run_key=specification.run_key,
        outcome=merged,
        records=tuple(records),
        checkpoint_path=(
            str(written_checkpoint) if written_checkpoint is not None else None
        ),
        resumed_from_batch=cursor,
    )


def publish_run_report(
    workspace: str | Path,
    report: RunExecutionReport,
    *,
    at: str | None = None,
) -> RunRegistryEntry:
    """Record a report's terminal lifecycle state into the workspace catalog.

    Unknown run keys are auto-registered by the catalog layer, so publication
    works both for runs registered up front and for headless runs that were
    never registered.
    """

    if not isinstance(report, RunExecutionReport):
        raise TypeError(f"report must be a RunExecutionReport, got {type(report).__name__}")
    return record_run_record(workspace, report.final_record, at=at)
