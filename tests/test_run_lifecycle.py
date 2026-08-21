"""Model-free tests for the deterministic run lifecycle state machine."""

from __future__ import annotations

import sys

import pytest
from pydantic import ValidationError

from moeatlas.runs import (
    RUN_LIFECYCLE_SCHEMA_VERSION,
    RunAction,
    RunCancellation,
    RunFailure,
    RunLifecycleError,
    RunProgress,
    RunRecord,
    RunState,
    advance_progress,
    apply,
    can_transition,
    transition,
)

_RUN_KEY = "run:" + "a" * 64


def failure(**overrides: object) -> RunFailure:
    values: dict[str, object] = {
        "stage": RunState.RUNNING,
        "error_kind": "execution",
        "message": "forward pass failed",
    }
    values.update(overrides)
    return RunFailure(**values)  # type: ignore[arg-type]


def cancellation(**overrides: object) -> RunCancellation:
    values: dict[str, object] = {"reason": "user requested", "requested_by": "tester"}
    values.update(overrides)
    return RunCancellation(**values)  # type: ignore[arg-type]


def record(**overrides: object) -> RunRecord:
    values: dict[str, object] = {
        "run_key": _RUN_KEY,
        "specification_fingerprint": _RUN_KEY,
    }
    values.update(overrides)
    return RunRecord(**values)  # type: ignore[arg-type]


def progress(completed: int, *, stage: str = "rows", total: int | None = 10) -> RunProgress:
    return RunProgress(stage=stage, completed_units=completed, total_units=total)


def test_happy_path_reaches_completed() -> None:
    current = record()
    assert current.state is RunState.PLANNED
    assert current.attempt == 1
    steps = (
        RunAction.START,
        RunAction.BEGIN_EXECUTION,
        RunAction.UPDATE_PROGRESS,
        RunAction.UPDATE_PROGRESS,
        RunAction.FINALIZE,
        RunAction.COMPLETE,
    )
    for index, action in enumerate(steps):
        kwargs: dict[str, object] = {"at": f"t{index}"}
        if action is RunAction.UPDATE_PROGRESS:
            kwargs["progress"] = progress(5 if index == 2 else 9)
        current = apply(current, action, **kwargs)  # type: ignore[arg-type]
    assert current.state is RunState.COMPLETED
    assert current.updated_at == "t5"
    assert current.progress == progress(9)
    assert current.failure is None and current.cancellation is None
    revived = RunRecord.from_json(current.to_json())
    assert revived == current
    assert RunRecord.manifest_type == "run_record"
    assert revived.to_dict()["schema_version"] == RUN_LIFECYCLE_SCHEMA_VERSION


def test_transition_table_is_total_and_deterministic() -> None:
    states = tuple(RunState)
    actions = tuple(RunAction)
    legal = 0
    for state in states:
        for action in actions:
            if can_transition(state, action):
                legal += 1
                nxt = transition(state, action)
                assert isinstance(nxt, RunState)
            else:
                with pytest.raises(
                    RunLifecycleError,
                    match=f"illegal run transition: {state.value} cannot {action.value}",
                ):
                    transition(state, action)
    assert legal == 17  # the documented table size


def test_only_retry_escapes_terminal_states() -> None:
    for state in (RunState.COMPLETED, RunState.FAILED, RunState.CANCELLED):
        for action in RunAction:
            if action is RunAction.RETRY and state is not RunState.COMPLETED:
                assert can_transition(state, action)
            else:
                assert not can_transition(state, action), (state, action)
    retried = apply(record(state=RunState.FAILED, failure=failure()), RunAction.RETRY)
    assert retried.state is RunState.PLANNED
    assert retried.attempt == 2
    assert retried.failure is None and retried.progress is None
    cancelled_retry = apply(
        record(state=RunState.CANCELLED, cancellation=cancellation()), RunAction.RETRY
    )
    assert cancelled_retry.attempt == 2 and cancelled_retry.cancellation is None


def test_action_payload_requirements() -> None:
    running = record(state=RunState.RUNNING)
    with pytest.raises(RunLifecycleError, match="fail action requires a failure record"):
        apply(running, RunAction.FAIL)
    with pytest.raises(RunLifecycleError, match="requires a cancellation record"):
        apply(running, RunAction.REQUEST_CANCELLATION)
    with pytest.raises(RunLifecycleError, match="update_progress action requires progress"):
        apply(running, RunAction.UPDATE_PROGRESS)
    with pytest.raises(RunLifecycleError, match="only valid with the fail action"):
        apply(running, RunAction.FINALIZE, failure=failure())
    with pytest.raises(RunLifecycleError, match="only valid with request_cancellation"):
        apply(running, RunAction.FINALIZE, cancellation=cancellation())
    with pytest.raises(RunLifecycleError, match="progress is only valid"):
        apply(running, RunAction.FINALIZE, progress=progress(1))


def test_progress_monotonic_within_stage_and_resets_across_stages() -> None:
    running = record(state=RunState.RUNNING)
    advanced = apply(running, RunAction.UPDATE_PROGRESS, progress=progress(5))
    with pytest.raises(RunLifecycleError, match="must not regress within stage 'rows'"):
        apply(advanced, RunAction.UPDATE_PROGRESS, progress=progress(4))
    across = apply(advanced, RunAction.UPDATE_PROGRESS, progress=progress(0, stage="export"))
    assert across.progress == progress(0, stage="export")
    with pytest.raises(TypeError, match="must be a RunProgress"):
        advance_progress(None, "not progress")  # type: ignore[arg-type]


def test_progress_validation_bounds() -> None:
    with pytest.raises(ValidationError, match="total_units"):
        progress(6, total=5)
    with pytest.raises(ValidationError, match="completed_units"):
        RunProgress(stage="rows", completed_units=-1)
    with pytest.raises(ValidationError, match="stage length"):
        RunProgress(stage="")
    with pytest.raises(ValidationError, match="control characters"):
        RunProgress(stage="bad\x00stage")


def test_update_progress_is_a_self_loop_in_active_states_only() -> None:
    for state in (RunState.PROVISIONING, RunState.RUNNING, RunState.FINALIZING):
        updated = apply(record(state=state), RunAction.UPDATE_PROGRESS, progress=progress(1))
        assert updated.state is state
        assert updated.progress == progress(1)
    for state in (RunState.PLANNED, RunState.COMPLETED, RunState.FAILED, RunState.CANCELLED):
        assert not can_transition(state, RunAction.UPDATE_PROGRESS)


def test_cancellation_flow_and_records() -> None:
    cancelling = apply(record(state=RunState.RUNNING), RunAction.REQUEST_CANCELLATION,
                       cancellation=cancellation())
    assert cancelling.state is RunState.CANCELLING
    assert cancelling.cancellation == cancellation()
    cancelled = apply(cancelling, RunAction.CANCEL, cancellation=cancellation())
    assert cancelled.state is RunState.CANCELLED
    planned_cancel = apply(record(), RunAction.REQUEST_CANCELLATION,
                           cancellation=cancellation())
    assert planned_cancel.state is RunState.CANCELLING
    assert planned_cancel.progress is None


def test_fail_from_active_states_requires_stage_consistency_payload() -> None:
    for state in (RunState.PROVISIONING, RunState.RUNNING, RunState.FINALIZING):
        failed = apply(record(state=state), RunAction.FAIL, failure=failure(stage=state))
        assert failed.state is RunState.FAILED
        assert failed.failure is not None
        assert failed.failure.stage is state


def test_record_invariants() -> None:
    with pytest.raises(ValidationError, match="requires a failure record"):
        record(state=RunState.FAILED)
    with pytest.raises(ValidationError, match="no failure or cancellation record"):
        record(state=RunState.COMPLETED, failure=failure())
    with pytest.raises(ValidationError, match="requires a cancellation record"):
        record(state=RunState.CANCELLING)
    with pytest.raises(ValidationError, match="requires a cancellation record"):
        record(state=RunState.CANCELLED)
    with pytest.raises(ValidationError, match="no progress"):
        record(progress=progress(1))
    with pytest.raises(ValidationError, match="no failure or cancellation record"):
        record(state=RunState.RUNNING, cancellation=cancellation())
    with pytest.raises(ValidationError, match="run:<64 lowercase hex>"):
        record(specification_fingerprint="not-a-run-key")


def test_record_rejects_non_canonical_run_key() -> None:
    with pytest.raises(ValidationError, match="must not contain whitespace"):
        record(run_key="Run Key With Spaces")


def test_type_rejections() -> None:
    with pytest.raises(TypeError, match="state must be a RunState"):
        transition("planned", RunAction.START)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="action must be a RunAction"):
        transition(RunState.PLANNED, "start")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="record must be a RunRecord"):
        apply("not a record", RunAction.START)  # type: ignore[arg-type]


def test_no_model_runtime_imported_by_lifecycle() -> None:
    assert not any(
        name in sys.modules for name in ("torch", "transformers", "safetensors")
    )
