"""Contract tests for run input preparation over specs and descriptors."""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    import duckdb
except ImportError:  # pragma: no cover - exercised only without the store extra
    duckdb = None

import moeatlas.services.datasets as datasets
from moeatlas.runs.specs import (
    ChatMessage,
    DatasetFormat,
    DatasetInputSpec,
    PromptInputSpec,
)
from moeatlas.services import (
    RUN_INPUTS_SCHEMA_VERSION,
    DatasetReadError,
    RunInputError,
    execute_row_schedule,
    plan_dataset_batches,
    plan_input_batches,
    prepare_input_rows,
    prepare_prompt_rows,
)

parquet_mark = pytest.mark.skipif(duckdb is None, reason="duckdb store extra is unavailable")


# ---------------------------------------------------------------------------
# Public surface


def test_surface_is_pinned() -> None:
    assert RUN_INPUTS_SCHEMA_VERSION == "1.0"
    assert str(RunInputError("budget")) == "run input preparation failed at budget"
    with pytest.raises(ValueError):
        RunInputError("cancelled")


# ---------------------------------------------------------------------------
# Prompt preparation


def test_text_prompts_prepare_one_bounded_row() -> None:
    prepared = prepare_prompt_rows(PromptInputSpec(text="hello world"))
    assert prepared == {0: {"prompt": "hello world"}}


def test_chat_prompts_preserve_declared_message_order() -> None:
    prepared = prepare_prompt_rows(
        PromptInputSpec(
            messages=(
                ChatMessage(role="system", content="be brief"),
                ChatMessage(role="user", content="hi"),
                ChatMessage(role="assistant", content="hello"),
            )
        )
    )
    assert prepared == {
        0: {
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        }
    }


def test_prompt_preparation_strictness_is_exact() -> None:
    with pytest.raises(TypeError):
        prepare_prompt_rows("not-a-spec")
    with pytest.raises(TypeError):
        prepare_prompt_rows(PromptInputSpec(text="x"), max_input_bytes=0)
    with pytest.raises(TypeError):
        prepare_prompt_rows(PromptInputSpec(text="x"), max_input_bytes=True)


def test_oversized_prompt_fails_at_budget() -> None:
    with pytest.raises(RunInputError) as caught:
        prepare_prompt_rows(PromptInputSpec(text="x" * 100), max_input_bytes=16)
    assert caught.value.stage == "budget"


# ---------------------------------------------------------------------------
# Schedule derivation


def test_prompt_schedules_are_single_row() -> None:
    assert plan_input_batches(PromptInputSpec(text="x"), 1) == ((0,),)
    with pytest.raises(ValueError):
        plan_input_batches(PromptInputSpec(text="x"), 2)
    with pytest.raises(ValueError):
        plan_input_batches(PromptInputSpec(text="x"), 0)


def test_dataset_schedules_apply_descriptor_settings_deterministically() -> None:
    spec = DatasetInputSpec(
        format=DatasetFormat.CSV,
        location="d.csv",
        batch_size=3,
        sample_cap=7,
        shuffle=True,
        seed=9,
    )
    planned = plan_input_batches(spec, 30)
    direct = plan_dataset_batches(30, batch_size=3, sample_cap=7, shuffle=True, seed=9)
    again = plan_input_batches(spec, 30)
    assert planned == direct == again


def test_plan_dispatch_strictness_is_exact() -> None:
    with pytest.raises(TypeError):
        plan_input_batches("not-a-spec", 1)


# ---------------------------------------------------------------------------
# Dataset preparation and dispatch


@pytest.fixture
def csv_workspace(tmp_path: Path) -> Path:
    (tmp_path / "d.csv").write_text("prompt,label\na,x\nb,y\nc,z\n", encoding="utf-8")
    return tmp_path


def test_dataset_rows_prepare_without_mapping(csv_workspace: Path) -> None:
    spec = DatasetInputSpec(format=DatasetFormat.CSV, location="d.csv")
    prepared = prepare_input_rows(spec, base_directory=csv_workspace)
    assert prepared == {
        0: {"prompt": "a", "label": "x"},
        1: {"prompt": "b", "label": "y"},
        2: {"prompt": "c", "label": "z"},
    }


def test_dataset_rows_project_mapped_roles_preserving_indices(
    csv_workspace: Path,
) -> None:
    spec = DatasetInputSpec(
        format=DatasetFormat.CSV,
        location="d.csv",
        column_mapping={"prompt": "prompt", "reference": "label"},
    )
    prepared = prepare_input_rows(spec, base_directory=csv_workspace)
    assert prepared == {
        0: {"prompt": "a", "reference": "x"},
        1: {"prompt": "b", "reference": "y"},
        2: {"prompt": "c", "reference": "z"},
    }


def test_mmlu_rows_render_choices_and_convert_numeric_answer(tmp_path: Path) -> None:
    (tmp_path / "mmlu.jsonl").write_text(
        '{"question":"Which protocol?","choices":["FTP","SSH","SMTP","DNS"],"answer":1}\n',
        encoding="utf-8",
    )
    spec = DatasetInputSpec(
        format=DatasetFormat.JSONL,
        location="mmlu.jsonl",
        column_mapping={"prompt": "question", "reference": "answer"},
        prompt_format="mmlu_multiple_choice",
        choices_column="choices",
    )

    prepared = prepare_input_rows(spec, base_directory=tmp_path)

    assert prepared == {
        0: {
            "prompt": (
                "Which protocol?\n\nA. FTP\nB. SSH\nC. SMTP\nD. DNS\n\n"
                "Answer with only A, B, C, or D."
            ),
            "reference": "B",
        }
    }


def test_mmlu_format_requires_choices_column() -> None:
    with pytest.raises(ValueError, match="choices column"):
        DatasetInputSpec(
            format=DatasetFormat.JSONL,
            location="mmlu.jsonl",
            column_mapping={"prompt": "question", "reference": "answer"},
            prompt_format="mmlu_multiple_choice",
        )


def test_dataset_budgets_propagate_from_the_reader(csv_workspace: Path) -> None:
    spec = DatasetInputSpec(format=DatasetFormat.CSV, location="d.csv")
    with pytest.raises(DatasetReadError) as row_caught:
        prepare_input_rows(spec, base_directory=csv_workspace, max_rows=2)
    assert row_caught.value.stage == "budget"
    missing = DatasetInputSpec(format=DatasetFormat.CSV, location="absent.csv")
    with pytest.raises(DatasetReadError) as read_caught:
        prepare_input_rows(missing, base_directory=csv_workspace)
    assert read_caught.value.stage == "read"


def test_relative_locations_require_base_directory() -> None:
    spec = DatasetInputSpec(format=DatasetFormat.CSV, location="d.csv")
    with pytest.raises(DatasetReadError) as caught:
        prepare_input_rows(spec)
    assert caught.value.stage == "descriptor"


@parquet_mark
def test_parquet_descriptors_prepare_through_lazy_engine(tmp_path: Path) -> None:
    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute("CREATE TABLE t (prompt VARCHAR)")
        connection.execute("INSERT INTO t VALUES ('alpha'), ('beta')")
        connection.execute(f"COPY t TO '{tmp_path / 'd.parquet'}' (FORMAT PARQUET)")
    finally:
        connection.close()
    spec = DatasetInputSpec(format=DatasetFormat.PARQUET, location="d.parquet")
    prepared = prepare_input_rows(spec, base_directory=tmp_path)
    assert prepared == {0: {"prompt": "alpha"}, 1: {"prompt": "beta"}}


def test_prepare_dispatch_strictness_is_exact(csv_workspace: Path) -> None:
    with pytest.raises(TypeError):
        prepare_input_rows("not-a-spec", base_directory=csv_workspace)


def test_csv_preparation_never_resolves_the_engine(csv_workspace: Path, monkeypatch) -> None:
    def forbidden():
        raise AssertionError("CSV preparation must not resolve DuckDB")

    monkeypatch.setattr(datasets, "_load_duckdb", forbidden)
    spec = DatasetInputSpec(format=DatasetFormat.CSV, location="d.csv")
    assert len(prepare_input_rows(spec, base_directory=csv_workspace)) == 3


# ---------------------------------------------------------------------------
# Composition with the execution core


def test_prompt_pipeline_executes_end_to_end() -> None:
    spec = PromptInputSpec(text="hello")
    row_values = prepare_input_rows(spec)
    schedule = plan_input_batches(spec, len(row_values))
    outcome = execute_row_schedule(
        schedule,
        executor=lambda *, row_index, batch_index, values: {"echo": values["prompt"]},
        row_values=row_values,
    )
    assert outcome.status == "completed"
    assert [dict(r.result) for r in outcome.results] == [{"echo": "hello"}]


def test_dataset_pipeline_executes_descriptor_schedule(csv_workspace: Path) -> None:
    spec = DatasetInputSpec(
        format=DatasetFormat.CSV,
        location="d.csv",
        column_mapping={"prompt": "prompt"},
        batch_size=2,
    )
    row_values = prepare_input_rows(spec, base_directory=csv_workspace)
    schedule = plan_input_batches(spec, len(row_values))
    assert schedule == ((0, 1), (2,))
    outcome = execute_row_schedule(
        schedule,
        executor=lambda *, row_index, batch_index, values: {"echo": values["prompt"]},
        row_values=row_values,
    )
    assert outcome.status == "completed"
    assert outcome.total_rows == 3
    echoes = {r.result["echo"] for r in outcome.results}
    assert echoes == {"a", "b", "c"}


def test_dataset_pipeline_honors_cancellation(csv_workspace: Path) -> None:
    spec = DatasetInputSpec(format=DatasetFormat.CSV, location="d.csv")
    row_values = prepare_input_rows(spec, base_directory=csv_workspace)
    schedule = plan_input_batches(spec, len(row_values))
    state = {"count": 0}

    def executor(*, row_index, batch_index, values):
        state["count"] += 1
        return {"n": state["count"]}

    outcome = execute_row_schedule(
        schedule,
        executor=executor,
        row_values=row_values,
        should_cancel=lambda: state["count"] >= 2,
    )
    assert outcome.status == "cancelled"
    assert outcome.executed_rows == 2
