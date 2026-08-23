"""Contract tests for the deterministic run-engine execution core."""

from __future__ import annotations

from dataclasses import fields

import pytest

from moeatlas.runs.lifecycle import advance_progress
from moeatlas.services import (
    EXECUTION_PROGRESS_STAGE,
    ROW_FAILURE_KINDS,
    RUN_ENGINE_SCHEMA_VERSION,
    ExecutionOutcome,
    RowFailure,
    RowRecord,
    RowResult,
    RunEngineError,
    execute_row_schedule,
    plan_dataset_batches,
    sanitize_failure_message,
    serialize_row_failures,
)


def _values(count: int) -> dict[int, dict[str, str]]:
    return {index: {"prompt": f"p{index}"} for index in range(count)}


def _echo_executor(*, row_index: int, batch_index: int, values):
    return {"echo": values["prompt"], "batch": batch_index}


# ---------------------------------------------------------------------------
# Public surface


def test_surface_is_pinned() -> None:
    assert RUN_ENGINE_SCHEMA_VERSION == "1.0"
    assert EXECUTION_PROGRESS_STAGE == "executing"
    assert ROW_FAILURE_KINDS == (
        "dependency",
        "validation",
        "execution",
        "storage",
        "interruption",
        "unknown",
    )
    assert tuple(field.name for field in fields(RowResult)) == (
        "row_index",
        "batch_index",
        "result",
    )
    assert tuple(field.name for field in fields(RowRecord)) == (
        "row_index",
        "batch_index",
        "kind",
        "message",
    )
    assert tuple(field.name for field in fields(ExecutionOutcome)) == (
        "schema_version",
        "total_rows",
        "executed_rows",
        "results",
        "failures",
        "cancelled",
        "cancelled_before_row",
        "progress",
    )
    assert str(RunEngineError("contract")) == "run engine failed at contract"
    with pytest.raises(ValueError):
        RunEngineError("cancelled")


def test_row_failure_is_strict() -> None:
    failure = RowFailure("validation", "missing reference")
    assert failure.kind == "validation"
    assert failure.message == "missing reference"
    with pytest.raises(ValueError):
        RowFailure("catastrophe", "nope")
    with pytest.raises(ValueError):
        RowFailure("validation", "")
    with pytest.raises(ValueError):
        RowFailure("validation", "x" * 501)
    with pytest.raises(ValueError):
        RowFailure("validation", 3)


def test_row_result_and_record_are_strict() -> None:
    result = RowResult(row_index=0, batch_index=0, result={"a": 1})
    assert result.result == {"a": 1}
    with pytest.raises(ValueError):
        RowResult(row_index=-1, batch_index=0, result={})
    with pytest.raises(ValueError):
        RowResult(row_index=0, batch_index=-1, result={})
    with pytest.raises(TypeError):
        RowResult(row_index=0, batch_index=0, result=[("a", 1)])
    with pytest.raises(TypeError):
        RowResult(row_index=0, batch_index=0, result={"a": 1, 2: "b"})
    record = RowRecord(row_index=1, batch_index=0, kind="execution", message="ValueError")
    assert record == RowRecord(row_index=1, batch_index=0, kind="execution", message="ValueError")
    with pytest.raises(ValueError):
        RowRecord(row_index=1, batch_index=0, kind="mystery", message="x")
    with pytest.raises(ValueError):
        RowRecord(row_index=1, batch_index=0, kind="execution", message="")


def test_row_failure_wire_view_is_bounded_and_redacted() -> None:
    failure = RowRecord(
        row_index=7,
        batch_index=2,
        kind="execution",
        message=(
            "prompt='private prompt' api_key=sk-12345678901234567890 "
            "path=/workspace/private/checkpoint.json"
        ),
    )
    assert sanitize_failure_message(failure.message) == (
        "prompt=<redacted> api_key=<redacted> path=<path>"
    )
    assert serialize_row_failures((failure,)) == (
        {
            "row_index": 7,
            "batch_index": 2,
            "kind": "execution",
            "message": "prompt=<redacted> api_key=<redacted> path=<path>",
        },
    )
    with pytest.raises(ValueError):
        serialize_row_failures((failure,), max_entries=65)


def test_execution_outcome_is_strict() -> None:
    def outcome(**overrides):
        base = dict(
            schema_version=RUN_ENGINE_SCHEMA_VERSION,
            total_rows=2,
            executed_rows=2,
            results=(RowResult(row_index=0, batch_index=0, result={"a": 1}),),
            failures=(RowRecord(row_index=1, batch_index=0, kind="execution", message="E"),),
            cancelled=False,
            cancelled_before_row=None,
            progress=(),
        )
        base.update(overrides)
        return ExecutionOutcome(**base)

    assert outcome().status == "completed"
    assert outcome(
        results=(),
        failures=(
            RowRecord(row_index=0, batch_index=0, kind="execution", message="E"),
            RowRecord(row_index=1, batch_index=0, kind="execution", message="E"),
        ),
    ).status == "failed"
    cancelled = outcome(
        executed_rows=1,
        results=(),
        failures=(RowRecord(row_index=0, batch_index=0, kind="execution", message="E"),),
        cancelled=True,
        cancelled_before_row=1,
    )
    assert cancelled.status == "cancelled"
    with pytest.raises(ValueError):
        outcome(schema_version="9.9")
    with pytest.raises(ValueError):
        outcome(executed_rows=3)
    with pytest.raises(ValueError):
        outcome(executed_rows=1)
    with pytest.raises(TypeError):
        outcome(cancelled="yes")
    with pytest.raises(ValueError):
        outcome(cancelled=True)
    with pytest.raises(TypeError):
        outcome(
            executed_rows=1,
            results=(),
            failures=(RowRecord(row_index=0, batch_index=0, kind="execution", message="E"),),
            cancelled=True,
            cancelled_before_row=1,
            progress=("not-progress",),
        )


# ---------------------------------------------------------------------------
# Execution semantics


def test_execution_drives_schedule_order_and_records_results() -> None:
    calls = []

    def executor(*, row_index, batch_index, values):
        calls.append((row_index, batch_index, dict(values)))
        return {"echo": values["prompt"]}

    schedule = ((4, 2), (0,))
    outcome = execute_row_schedule(schedule, executor=executor, row_values=_values(5))
    assert calls == [(4, 0, {"prompt": "p4"}), (2, 0, {"prompt": "p2"}), (0, 1, {"prompt": "p0"})]
    assert outcome.status == "completed"
    assert outcome.total_rows == 3
    assert outcome.executed_rows == 3
    assert [result.row_index for result in outcome.results] == [4, 2, 0]
    assert [result.batch_index for result in outcome.results] == [0, 0, 1]
    assert outcome.failures == ()
    assert not outcome.cancelled
    assert outcome.cancelled_before_row is None


def test_progress_snapshots_are_cumulative_and_monotonic() -> None:
    schedule = ((0, 1), (2, 3), (4,))
    outcome = execute_row_schedule(
        schedule, executor=_echo_executor, row_values=_values(5)
    )
    assert [(p.stage, p.completed_units, p.total_units) for p in outcome.progress] == [
        (EXECUTION_PROGRESS_STAGE, 2, 5),
        (EXECUTION_PROGRESS_STAGE, 4, 5),
        (EXECUTION_PROGRESS_STAGE, 5, 5),
    ]
    # The trail is valid under the lifecycle's monotonicity contract.
    previous = None
    for snapshot in outcome.progress:
        previous = advance_progress(previous, snapshot)


def test_controlled_row_failures_carry_declared_kind() -> None:
    def executor(*, row_index, batch_index, values):
        if row_index == 1:
            raise RowFailure("dependency", "tokenizer missing")
        return {"ok": True}

    outcome = execute_row_schedule(((0, 1, 2),), executor=executor, row_values=_values(3))
    assert outcome.status == "completed"
    assert [(f.row_index, f.kind, f.message) for f in outcome.failures] == [
        (1, "dependency", "tokenizer missing")
    ]
    assert [r.row_index for r in outcome.results] == [0, 2]


def test_unexpected_exceptions_become_execution_failures() -> None:
    def executor(*, row_index, batch_index, values):
        raise RuntimeError("secret internal detail")

    outcome = execute_row_schedule(((7,),), executor=executor, row_values={7: {"a": "b"}})
    assert outcome.status == "failed"
    assert outcome.failures[0].kind == "execution"
    assert outcome.failures[0].message == "RuntimeError"
    assert "secret" not in outcome.failures[0].message


def test_keyboard_interrupt_and_system_exit_propagate() -> None:
    for exc_type in (KeyboardInterrupt, SystemExit):

        def executor(*, row_index, batch_index, values):
            raise exc_type()

        with pytest.raises(exc_type):
            execute_row_schedule(((0,),), executor=executor, row_values=_values(1))


def test_result_contract_violations_become_validation_failures() -> None:
    def non_mapping(*, row_index, batch_index, values):
        return ["not", "a", "mapping"]

    outcome = execute_row_schedule(((0,),), executor=non_mapping, row_values=_values(1))
    assert outcome.failures[0].kind == "validation"
    assert outcome.results == ()

    def bad_keys(*, row_index, batch_index, values):
        return {1: "x"}

    outcome = execute_row_schedule(((0,),), executor=bad_keys, row_values=_values(1))
    assert outcome.failures[0].kind == "validation"

    def non_encodable(*, row_index, batch_index, values):
        return {"payload": {1, 2}}

    outcome = execute_row_schedule(((0,),), executor=non_encodable, row_values=_values(1))
    assert outcome.failures[0].kind == "validation"


def test_result_byte_budget_becomes_validation_failure() -> None:
    def executor(*, row_index, batch_index, values):
        return {"text": "x" * 200}

    outcome = execute_row_schedule(
        ((0,),), executor=executor, row_values=_values(1), max_result_bytes=32
    )
    assert outcome.failures[0].kind == "validation"
    assert "32 byte budget" in outcome.failures[0].message


# ---------------------------------------------------------------------------
# Cancellation


def test_cancellation_before_first_row_executes_nothing() -> None:
    outcome = execute_row_schedule(
        ((0, 1),),
        executor=_echo_executor,
        row_values=_values(2),
        should_cancel=lambda: True,
    )
    assert outcome.status == "cancelled"
    assert outcome.cancelled is True
    assert outcome.cancelled_before_row == 0
    assert outcome.executed_rows == 0
    assert outcome.results == () and outcome.failures == ()
    assert outcome.progress == ()


def test_cancellation_mid_run_preserves_executed_rows() -> None:
    state = {"count": 0}

    def executor(*, row_index, batch_index, values):
        state["count"] += 1
        return {"n": state["count"]}

    def cancel_after_two():
        return state["count"] >= 2

    schedule = ((0, 1), (2, 3))
    outcome = execute_row_schedule(
        schedule, executor=executor, row_values=_values(4), should_cancel=cancel_after_two
    )
    assert outcome.status == "cancelled"
    assert outcome.cancelled_before_row == 2
    assert outcome.executed_rows == 2
    assert [r.row_index for r in outcome.results] == [0, 1]
    assert [(p.completed_units, p.total_units) for p in outcome.progress] == [(2, 4)]


# ---------------------------------------------------------------------------
# Contract validation and argument strictness


def test_schedule_contract_is_exact() -> None:
    with pytest.raises(RunEngineError) as empty_batch:
        execute_row_schedule(((0,), ()), executor=_echo_executor, row_values=_values(1))
    assert empty_batch.value.stage == "contract"
    with pytest.raises(RunEngineError) as duplicate:
        execute_row_schedule(((0,), (0,)), executor=_echo_executor, row_values=_values(1))
    assert duplicate.value.stage == "contract"
    with pytest.raises(TypeError):
        execute_row_schedule(((0,),), executor="not-callable", row_values=_values(1))
    with pytest.raises(TypeError):
        execute_row_schedule(((0,),), executor=_echo_executor, row_values=_values(1),
                             should_cancel="yes")
    with pytest.raises(TypeError):
        execute_row_schedule(((0,),), executor=_echo_executor, row_values=_values(1),
                             max_result_bytes=0)
    with pytest.raises(TypeError):
        execute_row_schedule(((0,),), executor=_echo_executor, row_values={0: "not-a-mapping"})
    with pytest.raises(TypeError):
        execute_row_schedule([([0],)], executor=_echo_executor, row_values=_values(1))
    with pytest.raises(TypeError):
        execute_row_schedule((("0",),), executor=_echo_executor, row_values=_values(1))


def test_missing_row_values_fail_at_contract() -> None:
    with pytest.raises(RunEngineError) as caught:
        execute_row_schedule(((5,),), executor=_echo_executor, row_values=_values(1))
    assert caught.value.stage == "contract"
    with pytest.raises(TypeError):
        execute_row_schedule(((0,),), executor=_echo_executor, row_values=[("a", "b")])


def test_empty_schedule_is_vacuously_complete() -> None:
    outcome = execute_row_schedule((), executor=_echo_executor, row_values={})
    assert outcome.status == "completed"
    assert outcome.total_rows == 0
    assert outcome.executed_rows == 0
    assert outcome.progress == ()


# ---------------------------------------------------------------------------
# Determinism and composition with batch planning


def test_batch_offset_shifts_evidence_without_changing_order() -> None:
    outcome = execute_row_schedule(
        ((0, 1),),
        executor=_echo_executor,
        row_values=_values(2),
        batch_offset=4,
    )
    assert [r.batch_index for r in outcome.results] == [4, 4]
    assert [r.row_index for r in outcome.results] == [0, 1]
    assert outcome.total_rows == 2
    assert outcome.progress[-1].completed_units == 2
    with pytest.raises(TypeError):
        execute_row_schedule(((0,),), executor=_echo_executor, row_values=_values(1),
                             batch_offset=-1)
    with pytest.raises(TypeError):
        execute_row_schedule(((0,),), executor=_echo_executor, row_values=_values(1),
                             batch_offset=True)


def test_execution_is_deterministic_across_runs() -> None:
    schedule = plan_dataset_batches(20, batch_size=6, sample_cap=9, shuffle=True, seed=4)
    first = execute_row_schedule(schedule, executor=_echo_executor, row_values=_values(20))
    second = execute_row_schedule(schedule, executor=_echo_executor, row_values=_values(20))
    assert first == second


def test_engine_composes_with_dataset_batch_planning() -> None:
    schedule = plan_dataset_batches(7, batch_size=3)
    outcome = execute_row_schedule(schedule, executor=_echo_executor, row_values=_values(7))
    assert outcome.total_rows == 7
    assert [len(batch) for batch in schedule] == [3, 3, 1]
    assert outcome.status == "completed"
    flat_results = [r.row_index for r in outcome.results]
    assert flat_results == [i for batch in schedule for i in batch]
