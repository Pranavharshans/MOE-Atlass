"""Deterministic run-engine execution core over injected row executors.

This module is the model-neutral heart of PRD §9 execution: it drives one
planned batch schedule (see ``plan_dataset_batches``) through a caller-supplied
row executor, recording per-row results and failures, bounded progress
snapshots compatible with ``moeatlas.runs.lifecycle``, and cooperative
cancellation — without clocks, randomness, network access, or any model
dependency. Real adapters plug in as executors later; local tests use fake
runtimes.

Failure philosophy: a row failure is evidence, not a run death. Controlled
failures are declared by raising ``RowFailure`` with a kind from the same
fixed vocabulary as ``RunFailure.error_kind``; any other ``Exception``
becomes an ``execution`` failure carrying only the exception class name.
``KeyboardInterrupt`` and ``SystemExit`` always propagate. Cancellation is
cooperative: ``should_cancel`` is consulted before each row, and observing
it ends the run with a cancelled outcome that preserves everything already
executed. No publication happens here — persistence stays a separate slice.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ..runs.lifecycle import RunProgress, advance_progress

RUN_ENGINE_SCHEMA_VERSION = "1.0"
"""Schema version of the run-engine execution contracts."""

ROW_FAILURE_KINDS = (
    "dependency",
    "validation",
    "execution",
    "storage",
    "interruption",
    "unknown",
)
"""Fixed row-failure vocabulary; mirrors ``RunFailure.error_kind``."""

EXECUTION_PROGRESS_STAGE = "executing"
"""The single engine-defined progress stage used while rows execute."""

_MAX_FAILURE_MESSAGE = 500
_MAX_FAILURE_EVIDENCE = 64

# Row failures are allowed to carry adapter-provided context, but that context
# must never turn into an accidental prompt, credential, or host-path export.
# Keep this deliberately small and key-oriented: the execution core remains
# model-neutral and does not inspect model-specific exception objects.
_SECRET_VALUE = re.compile(
    r"(?is)\b(authorization|bearer|api[_-]?key|access[_-]?token|refresh[_-]?token|"
    r"hf[_-]?token|password|passwd|secret)\b\s*[:=]\s*(?:bearer\s+)?[^\s,;]+"
)
_PROMPT_VALUE = re.compile(
    r"(?is)\b(prompt|input|query|content|completion|token_text)\b"
    r"\s*[:=]\s*(\"[^\"]*\"|'[^']*'|[^\n,;}]+)"
)
_PATH_VALUE = re.compile(
    r"(?<![a-z0-9_])(?:/(?:users|home|workspace|private|tmp|var|opt|root|mnt|srv)"
    r"/[^\s'\"`;,)]*|[a-z]:[\\/][^\s'\"`;,)]*)",
    re.IGNORECASE,
)

_STAGES = frozenset({"contract", "executor", "budget"})


class RunEngineError(RuntimeError):
    """Safe fixed-stage failure for deterministic run execution."""

    def __init__(
        self, stage: str, message: str | None = None, *, cause: BaseException | None = None
    ) -> None:
        if stage not in _STAGES:
            raise ValueError("run engine error stage is not supported")
        self.stage = stage
        if message is None:
            super().__init__(f"run engine failed at {stage}")
        else:
            super().__init__(f"run engine failed at {stage}: {message}")
        if cause is not None:
            self.__cause__ = cause


class RowFailure(Exception):
    """Controlled per-row failure declared by an executor."""

    def __init__(self, kind: str, message: str) -> None:
        if kind not in ROW_FAILURE_KINDS:
            raise ValueError("row failure kind is not supported")
        if type(message) is not str or not message or len(message) > _MAX_FAILURE_MESSAGE:
            raise ValueError(
                f"row failure message length must be between 1 and {_MAX_FAILURE_MESSAGE}"
            )
        super().__init__(f"row failure [{kind}]: {message}")
        self.kind = kind
        self.message = message


def sanitize_failure_message(value: object) -> str:
    """Return bounded row-failure text safe to expose as run evidence.

    Adapters may provide useful validation context in a ``RowFailure`` message,
    but row evidence is persisted and returned by the server.  Redact common
    prompt/credential/path-shaped values while retaining the exception's
    structural wording.  No traceback or local variables are ever inspected.
    """

    if isinstance(value, str):
        text = value
    else:
        try:
            text = str(value)
        except Exception:
            text = "<unavailable failure detail>"
    text = _SECRET_VALUE.sub(lambda match: f"{match.group(1)}=<redacted>", text)
    text = _PROMPT_VALUE.sub(lambda match: f"{match.group(1)}=<redacted>", text)
    text = _PATH_VALUE.sub("<path>", text)
    # Diagnostics are line-oriented; control characters would make the record
    # ambiguous and can otherwise smuggle text into adjacent fields.
    text = " ".join(text.split())
    return text[:_MAX_FAILURE_MESSAGE] or "<empty failure detail>"


def serialize_row_failures(
    failures: tuple[RowRecord, ...], *, max_entries: int = 32
) -> tuple[dict[str, int | str], ...]:
    """Serialize bounded per-row failure evidence for a wire/checkpoint view.

    The canonical checkpoint remains lossless for its existing contract.  This
    view is intentionally capped and sanitized for job responses/diagnostics,
    so a large dataset or malformed adapter cannot expand the server surface.
    """

    if type(failures) is not tuple or any(not isinstance(f, RowRecord) for f in failures):
        raise TypeError("failures must be a tuple of RowRecord values")
    if (
        type(max_entries) is not int
        or isinstance(max_entries, bool)
        or not 1 <= max_entries <= _MAX_FAILURE_EVIDENCE
    ):
        raise ValueError(
            f"max_entries must be between 1 and {_MAX_FAILURE_EVIDENCE}"
        )
    return tuple(
        {
            "row_index": failure.row_index,
            "batch_index": failure.batch_index,
            "kind": failure.kind,
            "message": sanitize_failure_message(failure.message),
        }
        for failure in failures[:max_entries]
    )


@dataclass(frozen=True, slots=True)
class RowResult:
    """One successful row execution bound to its schedule position."""

    row_index: int
    batch_index: int
    result: Mapping[str, Any]

    def __post_init__(self) -> None:
        _strict_index(self.row_index, "row_index")
        _strict_index(self.batch_index, "batch_index")
        if not isinstance(self.result, Mapping) or not all(
            type(key) is str for key in self.result
        ):
            raise TypeError("result must be a mapping with string keys")
        object.__setattr__(self, "result", _freeze(self.result))


@dataclass(frozen=True, slots=True)
class RowRecord:
    """One failed row execution with its fixed-vocabulary kind."""

    row_index: int
    batch_index: int
    kind: str
    message: str

    def __post_init__(self) -> None:
        _strict_index(self.row_index, "row_index")
        _strict_index(self.batch_index, "batch_index")
        if self.kind not in ROW_FAILURE_KINDS:
            raise ValueError("row failure kind is not supported")
        if type(self.message) is not str or not self.message:
            raise ValueError("message must be a non-empty string")


@dataclass(frozen=True, slots=True)
class ExecutionOutcome:
    """Complete deterministic record of one schedule execution."""

    schema_version: str
    total_rows: int
    executed_rows: int
    results: tuple[RowResult, ...]
    failures: tuple[RowRecord, ...]
    cancelled: bool
    cancelled_before_row: int | None
    progress: tuple[RunProgress, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != RUN_ENGINE_SCHEMA_VERSION:
            raise ValueError("schema_version must be the exact run engine schema version")
        for name in ("total_rows", "executed_rows"):
            _strict_index(getattr(self, name), name)
        if self.executed_rows > self.total_rows:
            raise ValueError("executed_rows must not exceed total_rows")
        if len(self.results) + len(self.failures) != self.executed_rows:
            raise ValueError("results plus failures must equal executed_rows")
        if type(self.cancelled) is not bool:
            raise TypeError("cancelled must be a boolean")
        if self.cancelled:
            if self.executed_rows >= self.total_rows and self.total_rows > 0:
                raise ValueError("a cancelled outcome must leave rows unexecuted")
            if type(self.cancelled_before_row) is not int or isinstance(
                self.cancelled_before_row, bool
            ):
                raise TypeError("cancelled_before_row must be an integer when cancelled")
        elif self.cancelled_before_row is not None:
            raise ValueError("cancelled_before_row must be None unless cancelled")
        for entry in self.progress:
            if not isinstance(entry, RunProgress):
                raise TypeError("progress entries must be RunProgress values")

    @property
    def status(self) -> str:
        """Deterministic terminal suggestion: cancelled, failed, or completed."""

        if self.cancelled:
            return "cancelled"
        if self.total_rows > 0 and not self.results:
            return "failed"
        return "completed"


def _strict_index(value: object, name: str) -> None:
    if type(value) is not int or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _freeze(values: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(values))


def _canonical_result_bytes(values: Mapping[str, Any]) -> bytes:
    try:
        payload = json.dumps(
            dict(values),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise RunEngineError("budget", "row result is not canonically encodable", cause=exc)
    return payload.encode("utf-8")


def _validated_schedule(schedule: tuple[tuple[int, ...], ...]) -> int:
    if type(schedule) is not tuple:
        raise TypeError("schedule must be a tuple of index tuples")
    seen: set[int] = set()
    total = 0
    for batch_index, batch in enumerate(schedule):
        if type(batch) is not tuple or not batch:
            raise RunEngineError("contract", f"batch {batch_index} must be a non-empty tuple")
        for row_index in batch:
            if type(row_index) is not int or isinstance(row_index, bool) or row_index < 0:
                raise TypeError("schedule indices must be non-negative integers")
            if row_index in seen:
                raise RunEngineError(
                    "contract", f"row index {row_index} appears more than once in the schedule"
                )
            seen.add(row_index)
            total += 1
    return total


def execute_row_schedule(
    schedule: tuple[tuple[int, ...], ...],
    *,
    executor: Callable[..., Mapping[str, Any]],
    row_values: Mapping[int, Mapping[str, Any]],
    should_cancel: Callable[[], bool] | None = None,
    max_result_bytes: int = 65_536,
    batch_offset: int = 0,
) -> ExecutionOutcome:
    """Drive one planned schedule through ``executor`` deterministically.

    Rows execute in schedule order; each call is
    ``executor(row_index=..., batch_index=..., values=...)``. Progress snaps
    cumulatively after every batch through ``advance_progress`` so unit
    counts stay monotonic within the ``executing`` stage. Cancellation is
    checked before each row and freezes the outcome with everything already
    executed. ``batch_offset`` shifts every recorded batch index by a fixed
    amount so a driver may run one batch at a time (batch-per-call with
    ``batch_offset=batch_index``) while evidence keeps its plan-level
    numbering; it never changes execution order.
    """

    if not callable(executor):
        raise TypeError("executor must be callable")
    if should_cancel is not None and not callable(should_cancel):
        raise TypeError("should_cancel must be callable or None")
    if (
        type(max_result_bytes) is not int
        or isinstance(max_result_bytes, bool)
        or max_result_bytes <= 0
    ):
        raise TypeError("max_result_bytes must be a strict positive integer")
    if type(batch_offset) is not int or isinstance(batch_offset, bool) or batch_offset < 0:
        raise TypeError("batch_offset must be a non-negative integer")
    if not isinstance(row_values, Mapping):
        raise TypeError("row_values must be a mapping keyed by row index")
    total_rows = _validated_schedule(schedule)

    results: list[RowResult] = []
    failures: list[RowRecord] = []
    progress: list[RunProgress] = []
    previous_progress: RunProgress | None = None
    executed = 0
    cancelled = False
    cancelled_before_row: int | None = None

    for position, batch in enumerate(schedule):
        batch_index = batch_offset + position
        broke = False
        for row_index in batch:
            if should_cancel is not None and should_cancel():
                cancelled = True
                cancelled_before_row = row_index
                broke = True
                break
            if row_index not in row_values:
                raise RunEngineError("contract", f"row_values has no entry for row {row_index}")
            values = row_values[row_index]
            if not isinstance(values, Mapping):
                raise TypeError(f"row_values[{row_index}] must be a mapping")
            try:
                result = executor(row_index=row_index, batch_index=batch_index, values=values)
            except RowFailure as exc:
                failures.append(
                    RowRecord(
                        row_index=row_index, batch_index=batch_index, kind=exc.kind,
                        message=exc.message,
                    )
                )
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                failures.append(
                    RowRecord(
                        row_index=row_index,
                        batch_index=batch_index,
                        kind="execution",
                        message=type(exc).__name__,
                    )
                )
            else:
                if not isinstance(result, Mapping) or not all(type(key) is str for key in result):
                    failures.append(
                        RowRecord(
                            row_index=row_index,
                            batch_index=batch_index,
                            kind="validation",
                            message="executor returned a non-mapping result",
                        )
                    )
                else:
                    try:
                        encoded = _canonical_result_bytes(result)
                    except RunEngineError:
                        failures.append(
                            RowRecord(
                                row_index=row_index,
                                batch_index=batch_index,
                                kind="validation",
                                message="row result is not canonically encodable",
                            )
                        )
                    else:
                        if len(encoded) > max_result_bytes:
                            failures.append(
                                RowRecord(
                                    row_index=row_index,
                                    batch_index=batch_index,
                                    kind="validation",
                                    message=(
                                        f"row result exceeds the {max_result_bytes} byte budget"
                                    ),
                                )
                            )
                        else:
                            results.append(
                                RowResult(
                                    row_index=row_index, batch_index=batch_index, result=result
                                )
                            )
            executed += 1
        if broke:
            break
        snapshot = RunProgress(
            stage=EXECUTION_PROGRESS_STAGE, completed_units=executed, total_units=total_rows
        )
        previous_progress = advance_progress(previous_progress, snapshot)
        progress.append(snapshot)

    return ExecutionOutcome(
        schema_version=RUN_ENGINE_SCHEMA_VERSION,
        total_rows=total_rows,
        executed_rows=executed,
        results=tuple(results),
        failures=tuple(failures),
        cancelled=cancelled,
        cancelled_before_row=cancelled_before_row,
        progress=tuple(progress),
    )
