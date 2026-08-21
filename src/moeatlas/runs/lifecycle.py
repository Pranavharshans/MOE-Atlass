"""Deterministic run lifecycle states, transitions, and progress contracts.

The lifecycle is a pure, serializable domain state machine. It never touches a
model, tokenizer, thread, or clock: timestamps are caller-supplied strings and
every transition is a total function from ``(state, action)`` to a new frozen
record or a fixed :class:`RunLifecycleError`. Process-local handles (loaded
models, hook managers) live in the runtime layer and are never part of these
contracts.
"""

from __future__ import annotations

from enum import Enum
from typing import ClassVar, Literal, Self

from pydantic import Field, StrictInt, StrictStr, field_validator, model_validator

from ..core import StrictManifestModel, VersionedManifest, validate_stable_identifier
from .specs import parse_run_key

RUN_LIFECYCLE_SCHEMA_VERSION = "1.0"

_MAX_MESSAGE = 500
_MAX_ACTOR = 200
_MAX_STAGE = 100
_MAX_TIMESTAMP = 64


class RunLifecycleError(RuntimeError):
    """Raised when a run action is illegal for the current state."""


class RunState(str, Enum):
    """Serializable run lifecycle state."""

    PLANNED = "planned"
    PROVISIONING = "provisioning"
    RUNNING = "running"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"

    def __str__(self) -> str:
        return self.value


class RunAction(str, Enum):
    """Explicit lifecycle actions; every transition names one action."""

    START = "start"
    BEGIN_EXECUTION = "begin_execution"
    UPDATE_PROGRESS = "update_progress"
    FINALIZE = "finalize"
    COMPLETE = "complete"
    FAIL = "fail"
    REQUEST_CANCELLATION = "request_cancellation"
    CANCEL = "cancel"
    RETRY = "retry"

    def __str__(self) -> str:
        return self.value


_TERMINAL_STATES = frozenset({RunState.COMPLETED, RunState.FAILED, RunState.CANCELLED})
_RETRYABLE_STATES = frozenset({RunState.FAILED, RunState.CANCELLED})
_ACTIVE_STATES = frozenset(
    {RunState.PROVISIONING, RunState.RUNNING, RunState.FINALIZING}
)

_TRANSITIONS: dict[tuple[RunState, RunAction], RunState] = {
    (RunState.PLANNED, RunAction.START): RunState.PROVISIONING,
    (RunState.PLANNED, RunAction.REQUEST_CANCELLATION): RunState.CANCELLING,
    (RunState.PROVISIONING, RunAction.BEGIN_EXECUTION): RunState.RUNNING,
    (RunState.PROVISIONING, RunAction.UPDATE_PROGRESS): RunState.PROVISIONING,
    (RunState.PROVISIONING, RunAction.FAIL): RunState.FAILED,
    (RunState.PROVISIONING, RunAction.REQUEST_CANCELLATION): RunState.CANCELLING,
    (RunState.RUNNING, RunAction.UPDATE_PROGRESS): RunState.RUNNING,
    (RunState.RUNNING, RunAction.FINALIZE): RunState.FINALIZING,
    (RunState.RUNNING, RunAction.FAIL): RunState.FAILED,
    (RunState.RUNNING, RunAction.REQUEST_CANCELLATION): RunState.CANCELLING,
    (RunState.FINALIZING, RunAction.COMPLETE): RunState.COMPLETED,
    (RunState.FINALIZING, RunAction.UPDATE_PROGRESS): RunState.FINALIZING,
    (RunState.FINALIZING, RunAction.FAIL): RunState.FAILED,
    (RunState.FINALIZING, RunAction.REQUEST_CANCELLATION): RunState.CANCELLING,
    (RunState.CANCELLING, RunAction.CANCEL): RunState.CANCELLED,
    (RunState.FAILED, RunAction.RETRY): RunState.PLANNED,
    (RunState.CANCELLED, RunAction.RETRY): RunState.PLANNED,
}


def can_transition(state: RunState, action: RunAction) -> bool:
    """Return whether ``action`` is legal in ``state``."""

    return (state, action) in _TRANSITIONS


def transition(state: RunState, action: RunAction) -> RunState:
    """Return the deterministic next state or raise :class:`RunLifecycleError`."""

    if not isinstance(state, RunState):
        raise TypeError(f"state must be a RunState, got {type(state).__name__}")
    if not isinstance(action, RunAction):
        raise TypeError(f"action must be a RunAction, got {type(action).__name__}")
    nxt = _TRANSITIONS.get((state, action))
    if nxt is None:
        raise RunLifecycleError(
            f"illegal run transition: {state.value} cannot {action.value}"
        )
    return nxt


class RunProgress(StrictManifestModel):
    """Bounded progress snapshot for one engine-defined stage."""

    stage: StrictStr = Field(max_length=_MAX_STAGE)
    completed_units: StrictInt = Field(default=0, ge=0)
    total_units: StrictInt | None = Field(default=None, ge=0)

    @field_validator("stage")
    @classmethod
    def _bounded_stage(cls, value: str) -> str:
        if not value:
            raise ValueError(f"stage length must be between 1 and {_MAX_STAGE}")
        if any(ord(character) < 32 for character in value):
            raise ValueError("stage must not contain control characters")
        return value

    @model_validator(mode="after")
    def _total_bounds(self) -> Self:
        if self.total_units is not None and self.total_units < self.completed_units:
            raise ValueError("total_units must not be less than completed_units")
        return self


def advance_progress(previous: RunProgress | None, nxt: RunProgress) -> RunProgress:
    """Enforce monotonic unit counts within one stage.

    A stage change resets the comparison freely; repeating a stage with fewer
    completed units than before is a deterministic error.
    """

    if not isinstance(nxt, RunProgress):
        raise TypeError(f"progress must be a RunProgress, got {type(nxt).__name__}")
    if (
        previous is not None
        and previous.stage == nxt.stage
        and nxt.completed_units < previous.completed_units
    ):
        raise RunLifecycleError(
            f"run progress must not regress within stage {nxt.stage!r}: "
            f"{previous.completed_units} followed by {nxt.completed_units}"
        )
    return nxt


class RunFailure(StrictManifestModel):
    """Fixed-vocabulary failure record attached when a run enters ``failed``."""

    stage: RunState
    error_kind: Literal[
        "dependency",
        "validation",
        "execution",
        "storage",
        "interruption",
        "unknown",
    ]
    message: StrictStr = Field(max_length=_MAX_MESSAGE)

    @field_validator("message")
    @classmethod
    def _nonempty_message(cls, value: str) -> str:
        if not value:
            raise ValueError("failure message length must be between 1 and 500")
        return value


class RunCancellation(StrictManifestModel):
    """Caller-recorded cancellation request details."""

    reason: StrictStr | None = Field(default=None, max_length=_MAX_MESSAGE)
    requested_by: StrictStr | None = Field(default=None, max_length=_MAX_ACTOR)


class RunRecord(VersionedManifest):
    """Frozen lifecycle snapshot for one run key.

    Records are immutable value objects; every accepted action produces a new
    fully revalidated record. Process-local handles are never stored here.
    """

    manifest_type: ClassVar[str] = "run_record"

    run_key: StrictStr
    specification_fingerprint: StrictStr | None = None
    state: RunState = RunState.PLANNED
    attempt: StrictInt = Field(default=1, ge=1)
    progress: RunProgress | None = None
    failure: RunFailure | None = None
    cancellation: RunCancellation | None = None
    updated_at: StrictStr | None = Field(default=None, max_length=_MAX_TIMESTAMP)

    @field_validator("run_key")
    @classmethod
    def _stable_run_key(cls, value: str) -> str:
        return validate_stable_identifier(value, field_name="run_key")

    @field_validator("specification_fingerprint")
    @classmethod
    def _fingerprint_shape(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parse_run_key(value)
        return value

    @model_validator(mode="after")
    def _state_invariants(self) -> Self:
        if self.state is RunState.PLANNED and self.progress is not None:
            raise ValueError("a planned run has no progress")
        if self.state is RunState.COMPLETED:
            if self.failure is not None or self.cancellation is not None:
                raise ValueError("a completed run has no failure or cancellation record")
        elif self.state is RunState.FAILED:
            if self.failure is None:
                raise ValueError("a failed run requires a failure record")
            if self.cancellation is not None:
                raise ValueError("a failed run has no cancellation record")
        elif self.state is RunState.CANCELLING:
            if self.cancellation is None:
                raise ValueError("a cancelling run requires a cancellation record")
            if self.failure is not None:
                raise ValueError("a cancelling run has no failure record")
        elif self.state is RunState.CANCELLED:
            if self.cancellation is None:
                raise ValueError("a cancelled run requires a cancellation record")
            if self.failure is not None:
                raise ValueError("a cancelled run has no failure record")
        else:
            if self.failure is not None or self.cancellation is not None:
                raise ValueError(
                    f"a {self.state.value} run has no failure or cancellation record"
                )
        return self


def apply(
    record: RunRecord,
    action: RunAction,
    *,
    at: str | None = None,
    failure: RunFailure | None = None,
    cancellation: RunCancellation | None = None,
    progress: RunProgress | None = None,
) -> RunRecord:
    """Apply one action to a record, returning a fresh revalidated record."""

    if not isinstance(record, RunRecord):
        raise TypeError(f"record must be a RunRecord, got {type(record).__name__}")
    nxt_state = transition(record.state, action)

    if action is RunAction.FAIL and failure is None:
        raise RunLifecycleError("the fail action requires a failure record")
    if action is not RunAction.FAIL and failure is not None:
        raise RunLifecycleError("a failure record is only valid with the fail action")
    if action in (RunAction.REQUEST_CANCELLATION, RunAction.CANCEL) and cancellation is None:
        raise RunLifecycleError(f"the {action.value} action requires a cancellation record")
    if action not in (RunAction.REQUEST_CANCELLATION, RunAction.CANCEL) and (
        cancellation is not None
    ):
        raise RunLifecycleError(
            "a cancellation record is only valid with request_cancellation or cancel"
        )
    if action is RunAction.UPDATE_PROGRESS:
        if progress is None:
            raise RunLifecycleError("the update_progress action requires progress")
        progress = advance_progress(record.progress, progress)
    elif progress is not None:
        raise RunLifecycleError("progress is only valid with the update_progress action")

    attempt = record.attempt + 1 if action is RunAction.RETRY else record.attempt
    values: dict[str, object] = {"state": nxt_state, "attempt": attempt, "updated_at": at}
    if action is RunAction.FAIL:
        values["failure"] = failure
    if action in (RunAction.REQUEST_CANCELLATION, RunAction.CANCEL):
        values["cancellation"] = cancellation
    if action is RunAction.UPDATE_PROGRESS:
        values["progress"] = progress
    if action is RunAction.RETRY:
        values["failure"] = None
        values["cancellation"] = None
        values["progress"] = None

    payload = record.to_dict()
    payload.update(values)
    return RunRecord.model_validate(payload)
