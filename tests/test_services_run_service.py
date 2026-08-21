"""Contract tests for the headless run-engine service surface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from moeatlas.runs.lifecycle import (
    RunLifecycleError,
    RunRecord,
    RunState,
)
from moeatlas.runs.specs import DatasetFormat, DatasetInputSpec, GenerationConfig, PromptInputSpec
from moeatlas.services import (
    CHECKPOINT_MANIFEST_TYPE,
    CHECKPOINT_SCHEMA_VERSION,
    RUN_SERVICE_SCHEMA_VERSION,
    RowFailure,
    RowRecord,
    RunExecutionReport,
    RunServiceError,
    build_initial_record,
    derive_run_failure,
    execute_specification,
    initialize_workspace,
    load_checkpoint,
    publish_run_report,
)
from moeatlas.services.run_engine import RUN_ENGINE_SCHEMA_VERSION

from .test_run_contracts import data_provenance, run_specification

# ---------------------------------------------------------------------------
# Fixtures


def _jsonl_dataset(tmp_path: Path, count: int) -> DatasetInputSpec:
    folder = tmp_path / f"ds-{count}"
    folder.mkdir(parents=True, exist_ok=True)
    location = folder / "rows.jsonl"
    location.write_text(
        "".join(json.dumps({"prompt": f"p{index}"}) + "\n" for index in range(count)),
        encoding="utf-8",
    )
    return DatasetInputSpec(
        format=DatasetFormat.JSONL, location=str(location), row_count=count, batch_size=2
    )


def _prompt_spec(**overrides: object):
    return run_specification(
        data=data_provenance(input=PromptInputSpec(text="hello")), **overrides
    )


def _echo_executor(*, row_index: int, batch_index: int, values):
    return {"echo": values["prompt"], "batch": batch_index}


# ---------------------------------------------------------------------------
# Public surface


def test_surface_is_pinned() -> None:
    assert RUN_SERVICE_SCHEMA_VERSION == "1.0"
    assert CHECKPOINT_SCHEMA_VERSION == "1.0"
    assert CHECKPOINT_MANIFEST_TYPE == "run_checkpoint"
    assert str(RunServiceError("checkpoint")) == "run service failed at checkpoint"
    assert str(RunServiceError("lifecycle")) == "run service failed at lifecycle"
    with pytest.raises(ValueError):
        RunServiceError("cancelled")


# ---------------------------------------------------------------------------
# Lifecycle projection units


def test_build_initial_record_is_planned_with_fingerprint() -> None:
    spec = run_specification()
    record = build_initial_record(spec)
    assert isinstance(record, RunRecord)
    assert record.state is RunState.PLANNED
    assert record.attempt == 1
    assert record.specification_fingerprint == spec.run_key
    with pytest.raises(TypeError):
        build_initial_record("not-a-spec")


def test_derive_run_failure_rules_are_deterministic() -> None:
    fallback = derive_run_failure(())
    assert fallback.error_kind == "unknown"
    assert fallback.stage is RunState.RUNNING
    single = derive_run_failure((RowRecord(row_index=0, batch_index=0, kind="storage",
                                           message="disk full"),))
    assert single.error_kind == "storage"
    assert single.message == "disk full"
    multi = derive_run_failure((
        RowRecord(row_index=0, batch_index=0, kind="dependency", message="first"),
        RowRecord(row_index=1, batch_index=0, kind="execution", message="second"),
    ))
    assert multi.error_kind == "dependency"
    assert multi.message.startswith("2 row failures; first: first")
    long = derive_run_failure((RowRecord(row_index=0, batch_index=0, kind="execution",
                                         message="x" * 900),))
    assert len(long.message) == 500


# ---------------------------------------------------------------------------
# Prompt execution end-to-end


def test_prompt_spec_executes_to_a_completed_report() -> None:
    spec = _prompt_spec()
    report = execute_specification(spec, executor=_echo_executor, at="2026-08-21T00:00:00Z")
    assert isinstance(report, RunExecutionReport)
    assert report.schema_version == RUN_SERVICE_SCHEMA_VERSION
    assert report.run_key == spec.run_key
    assert report.outcome.schema_version == RUN_ENGINE_SCHEMA_VERSION
    assert report.outcome.status == "completed"
    assert report.checkpoint_path is None
    assert report.resumed_from_batch == 0
    states = [record.state for record in report.records]
    assert states == [
        RunState.PLANNED,
        RunState.PROVISIONING,
        RunState.RUNNING,
        RunState.RUNNING,  # per-batch update_progress
        RunState.FINALIZING,
        RunState.COMPLETED,
    ]
    assert report.records[-1].updated_at == "2026-08-21T00:00:00Z"
    assert report.final_record is report.records[-1]
    assert report.outcome.results[0].result["echo"] == "hello"


def test_on_record_receives_every_record_in_order() -> None:
    seen: list[RunRecord] = []
    spec = _prompt_spec()
    report = execute_specification(spec, executor=_echo_executor, on_record=seen.append)
    assert seen == list(report.records)
    with pytest.raises(TypeError):
        execute_specification(spec, executor=_echo_executor, on_record="nope")


def test_executor_receives_plan_level_batch_indices(tmp_path: Path) -> None:
    spec = run_specification(data=data_provenance(input=_jsonl_dataset(tmp_path, 5)))
    observed: list[tuple[int, int]] = []

    def recorder(*, row_index: int, batch_index: int, values):
        observed.append((row_index, batch_index))
        return {"ok": True}

    report = execute_specification(spec, executor=recorder)
    assert observed == [(0, 0), (1, 0), (2, 1), (3, 1), (4, 2)]
    assert report.outcome.total_rows == 5


# ---------------------------------------------------------------------------
# Failure projection


def test_controlled_row_failure_fails_the_run() -> None:
    def failing(*, row_index: int, batch_index: int, values):
        raise RowFailure(kind="validation", message="bad row")

    spec = _prompt_spec()
    report = execute_specification(spec, executor=failing, at="t0")
    assert report.outcome.status == "failed"
    assert report.final_record.state is RunState.FAILED
    assert report.final_record.failure is not None
    assert report.final_record.failure.error_kind == "validation"
    assert report.final_record.failure.message == "bad row"
    assert report.final_record.failure.stage is RunState.RUNNING
    assert [record.state for record in report.records] == [
        RunState.PLANNED,
        RunState.PROVISIONING,
        RunState.RUNNING,
        RunState.RUNNING,  # per-batch update_progress
        RunState.FAILED,
    ]


def test_unexpected_exception_yields_safe_execution_failure() -> None:
    def broken(*, row_index: int, batch_index: int, values):
        raise RuntimeError("secret-database-url")

    spec = _prompt_spec()
    report = execute_specification(spec, executor=broken)
    assert report.final_record.failure is not None
    assert report.final_record.failure.error_kind == "execution"
    assert report.final_record.failure.message == "RuntimeError"
    assert "secret" not in report.final_record.failure.message


def test_mixed_results_and_failures_complete_with_evidence(tmp_path: Path) -> None:
    def selective(*, row_index: int, batch_index: int, values):
        if row_index == 1:
            raise RowFailure(kind="dependency", message="row unavailable")
        return {"echo": values["prompt"]}

    spec = run_specification(data=data_provenance(input=_jsonl_dataset(tmp_path, 3)))
    report = execute_specification(spec, executor=selective)
    assert report.outcome.status == "completed"
    assert len(report.outcome.results) == 2
    assert len(report.outcome.failures) == 1
    assert report.final_record.state is RunState.COMPLETED
    assert report.final_record.failure is None


def test_keyboard_interrupt_propagates() -> None:
    def interrupted(*, row_index: int, batch_index: int, values):
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        execute_specification(_prompt_spec(), executor=interrupted)


# ---------------------------------------------------------------------------
# Cancellation


def test_cooperative_cancellation_produces_a_cancelled_report(tmp_path: Path) -> None:
    calls = {"count": 0}

    def should_cancel() -> bool:
        calls["count"] += 1
        return calls["count"] > 3

    spec = run_specification(data=data_provenance(input=_jsonl_dataset(tmp_path, 4)))
    report = execute_specification(
        spec, executor=_echo_executor, should_cancel=should_cancel, at="t0"
    )
    assert report.outcome.status == "cancelled"
    assert report.outcome.cancelled_before_row is not None
    assert report.outcome.executed_rows >= 1
    states = [record.state for record in report.records]
    assert states[-2:] == [RunState.CANCELLING, RunState.CANCELLED]
    assert report.final_record.cancellation is not None
    assert report.final_record.failure is None


def test_cancellation_fields_are_caller_supplied(tmp_path: Path) -> None:
    spec = run_specification(data=data_provenance(input=_jsonl_dataset(tmp_path, 4)))
    report = execute_specification(
        spec,
        executor=_echo_executor,
        should_cancel=lambda: True,
        requested_by="tester",
        cancellation_reason="user asked",
    )
    cancellation = report.final_record.cancellation
    assert cancellation is not None
    assert cancellation.requested_by == "tester"
    assert cancellation.reason == "user asked"


# ---------------------------------------------------------------------------
# Checkpoints and resume


def test_checkpoint_round_trips_and_is_byte_deterministic(tmp_path: Path) -> None:
    spec = run_specification(data=data_provenance(input=_jsonl_dataset(tmp_path, 5)))
    first_dir = tmp_path / "ck-a"
    report = execute_specification(spec, executor=_echo_executor,
                                   checkpoint_directory=first_dir)
    path = Path(report.checkpoint_path or "")
    assert path.exists()
    checkpoint = load_checkpoint(path)
    assert checkpoint.run_key == spec.run_key
    assert checkpoint.next_batch_index == 3
    assert checkpoint.total_batches == 3
    assert len(checkpoint.results) == 5
    assert checkpoint.failures == ()
    first_bytes = path.read_bytes()

    second_dir = tmp_path / "ck-b"
    again = execute_specification(spec, executor=_echo_executor,
                                  checkpoint_directory=second_dir)
    assert Path(again.checkpoint_path or "").read_bytes() == first_bytes


def test_checkpoint_is_incremental_and_resume_completes(tmp_path: Path) -> None:
    spec = run_specification(data=data_provenance(input=_jsonl_dataset(tmp_path, 5)))
    calls = {"count": 0}

    def should_cancel() -> bool:
        calls["count"] += 1
        return calls["count"] > 3

    checkpoint_dir = tmp_path / "ck"
    partial = execute_specification(
        spec, executor=_echo_executor, should_cancel=should_cancel,
        checkpoint_directory=checkpoint_dir,
    )
    assert partial.outcome.status == "cancelled"
    path = Path(partial.checkpoint_path or "")
    survived = load_checkpoint(path)
    assert survived.next_batch_index == 1
    assert len(survived.results) == 2

    executed: list[int] = []

    def counting(*, row_index: int, batch_index: int, values):
        executed.append(row_index)
        return {"echo": values["prompt"]}

    resumed = execute_specification(spec, executor=counting, resume_from=path)
    assert resumed.resumed_from_batch == 1
    assert resumed.outcome.status == "completed"
    assert sorted(executed) == [2, 3, 4]
    assert len(resumed.outcome.results) == 5
    assert {r.row_index for r in resumed.outcome.results} == set(range(5))


def test_resume_rejects_foreign_checkpoints(tmp_path: Path) -> None:
    first = run_specification(data=data_provenance(input=_jsonl_dataset(tmp_path, 2)))
    other = run_specification(data=data_provenance(input=_jsonl_dataset(tmp_path, 3)))
    first_dir = tmp_path / "a"
    foreign = execute_specification(first, executor=_echo_executor,
                                    checkpoint_directory=first_dir)
    with pytest.raises(RunServiceError) as caught:
        execute_specification(other, executor=_echo_executor,
                              resume_from=Path(foreign.checkpoint_path or ""))
    assert caught.value.stage == "checkpoint"


def test_load_checkpoint_strictness(tmp_path: Path) -> None:
    spec = run_specification(data=data_provenance(input=_jsonl_dataset(tmp_path, 2)))
    report = execute_specification(spec, executor=_echo_executor,
                                   checkpoint_directory=tmp_path / "ck")
    good = json.loads(Path(report.checkpoint_path or "").read_text(encoding="utf-8"))

    def mutated(**changes: object) -> Path:
        payload = dict(good)
        payload.update(changes)
        bad = tmp_path / "mutated.json"
        bad.write_text(json.dumps(payload), encoding="utf-8")
        return bad

    with pytest.raises(RunServiceError):
        load_checkpoint(tmp_path / "missing.json")
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(RunServiceError):
        load_checkpoint(broken)
    for changes in (
        {"manifest_type": "other"},
        {"schema_version": "9.9"},
        {"run_key": "not-canonical"},
        {"next_batch_index": -1},
        {"next_batch_index": 99},
        {"results": [{"row_index": 0, "batch_index": 0, "result": "not-a-mapping"}]},
        {
            "failures": [
                {"row_index": 0, "batch_index": 0, "kind": "bogus", "message": "m"}
            ]
        },
        {
            "results": [
                {"row_index": 0, "batch_index": 0, "result": {}},
                {"row_index": 0, "batch_index": 1, "result": {}},
            ]
        },
    ):
        with pytest.raises(RunServiceError) as caught:
            load_checkpoint(mutated(**changes))
        assert caught.value.stage == "checkpoint"


def test_checkpoint_write_failure_raises_typed_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import moeatlas.services.run_service as run_service

    def broken_replace(src: object, dst: object) -> None:
        raise OSError("disk gone")

    monkeypatch.setattr(run_service.os, "replace", broken_replace)
    spec = run_specification(data=data_provenance(input=_jsonl_dataset(tmp_path, 2)))
    with pytest.raises(RunServiceError) as caught:
        execute_specification(spec, executor=_echo_executor,
                              checkpoint_directory=tmp_path / "ck")
    assert caught.value.stage == "checkpoint"
    assert isinstance(caught.value.__cause__, OSError)


def test_lifecycle_failures_wrap_as_service_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import moeatlas.services.run_service as run_service

    def broken_apply(*args: object, **kwargs: object) -> RunRecord:
        raise RunLifecycleError("illegal run transition: planned cannot complete")

    monkeypatch.setattr(run_service, "apply", broken_apply)
    with pytest.raises(RunServiceError) as caught:
        execute_specification(_prompt_spec(), executor=_echo_executor)
    assert caught.value.stage == "lifecycle"
    assert isinstance(caught.value.__cause__, RunLifecycleError)


# ---------------------------------------------------------------------------
# Strictness, vacuous completion, determinism


def test_execute_specification_type_strictness() -> None:
    with pytest.raises(TypeError):
        execute_specification("not-a-spec", executor=_echo_executor)
    with pytest.raises(TypeError):
        execute_specification(run_specification(), executor="not-callable")


def test_empty_dataset_completes_vacuously(tmp_path: Path) -> None:
    location = tmp_path / "empty.csv"
    location.write_text("prompt\n", encoding="utf-8")
    descriptor = DatasetInputSpec(
        format=DatasetFormat.CSV, location=str(location),
        column_mapping={"prompt": "prompt"},
    )
    spec = run_specification(data=data_provenance(input=descriptor))
    report = execute_specification(spec, executor=_echo_executor)
    assert report.outcome.total_rows == 0
    assert report.outcome.status == "completed"
    stages = [record.state for record in report.records]
    assert RunState.RUNNING in stages
    assert stages[-1] is RunState.COMPLETED


def test_same_inputs_produce_identical_records_and_checkpoints(tmp_path: Path) -> None:
    spec = run_specification(data=data_provenance(input=_jsonl_dataset(tmp_path, 4)))
    first = execute_specification(spec, executor=_echo_executor, at="t0",
                                  checkpoint_directory=tmp_path / "a")
    second = execute_specification(spec, executor=_echo_executor, at="t0",
                                   checkpoint_directory=tmp_path / "b")
    assert [r.model_dump() for r in first.records] == [
        r.model_dump() for r in second.records
    ]
    assert Path(first.checkpoint_path or "").read_bytes() == Path(
        second.checkpoint_path or ""
    ).read_bytes()


def test_run_execution_report_invariants() -> None:
    spec = _prompt_spec()
    report = execute_specification(spec, executor=_echo_executor)
    with pytest.raises(ValueError):
        RunExecutionReport(
            schema_version=RUN_SERVICE_SCHEMA_VERSION,
            run_key=spec.run_key,
            outcome=report.outcome,
            records=(),
            checkpoint_path=None,
            resumed_from_batch=0,
        )
    with pytest.raises(TypeError):
        RunExecutionReport(
            schema_version=RUN_SERVICE_SCHEMA_VERSION,
            run_key=spec.run_key,
            outcome=report.outcome,
            records=("not-a-record",),  # type: ignore[list-item]
            checkpoint_path=None,
            resumed_from_batch=0,
        )
    with pytest.raises(ValueError):
        RunExecutionReport(
            schema_version=RUN_SERVICE_SCHEMA_VERSION,
            run_key=spec.run_key,
            outcome=report.outcome,
            records=report.records,
            checkpoint_path=None,
            resumed_from_batch=-1,
        )


# ---------------------------------------------------------------------------
# Workspace publication


def test_publish_run_report_records_the_terminal_state(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    initialize_workspace(workspace)
    passing = execute_specification(_prompt_spec(), executor=_echo_executor)
    entry = publish_run_report(workspace, passing, at="t1")
    assert entry.state == "completed"
    assert entry.specification_fingerprint == passing.run_key

    def failing(*, row_index: int, batch_index: int, values):
        raise RowFailure(kind="storage", message="shard closed")

    other = _prompt_spec(generation=GenerationConfig(seed=11))
    failing_report = execute_specification(other, executor=failing)
    failed_entry = publish_run_report(workspace, failing_report)
    assert failed_entry.state == "failed"


def test_publish_run_report_type_strictness(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        publish_run_report(tmp_path, "not-a-report")
