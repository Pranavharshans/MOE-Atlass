"""Contract tests for bounded deterministic dataset reading services."""

from __future__ import annotations

import sys
import types
from dataclasses import fields
from pathlib import Path

import pytest

try:
    import duckdb
except ImportError:  # pragma: no cover - exercised only without the store extra
    duckdb = None

import moeatlas.services.datasets as datasets
from moeatlas.runs.specs import DatasetFormat, DatasetInputSpec, RunMode
from moeatlas.services import (
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

parquet_mark = pytest.mark.skipif(duckdb is None, reason="duckdb store extra is unavailable")


def _csv_spec(name: str = "data.csv", **kwargs) -> DatasetInputSpec:
    return DatasetInputSpec(format=DatasetFormat.CSV, location=name, **kwargs)


def _write_csv(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


# ---------------------------------------------------------------------------
# Public surface


def test_surface_is_pinned() -> None:
    assert DATASET_READER_SCHEMA_VERSION == "1.0"
    assert DATASET_COLUMN_ROLES == ("domain", "prompt", "reference", "task")
    assert tuple(field.name for field in fields(DatasetRow)) == ("index", "values")
    assert str(DatasetReadError("budget")) == "dataset read failed at budget"
    with pytest.raises(ValueError):
        DatasetReadError("conflict")


def test_dataset_row_is_strict_and_frozen() -> None:
    row = DatasetRow(index=2, values={"a": "1"})
    assert row.values == {"a": "1"}
    assert row == DatasetRow(index=2, values={"a": "1"})
    with pytest.raises(ValueError):
        DatasetRow(index=-1, values={})
    with pytest.raises(TypeError):
        DatasetRow(index=True, values={})
    with pytest.raises(TypeError):
        DatasetRow(index="0", values={})
    with pytest.raises(TypeError):
        DatasetRow(index=0, values=[("a", "1")])
    with pytest.raises(TypeError):
        DatasetRow(index=0, values={"a": "1", 2: "b"})
    with pytest.raises(Exception):
        row.index = 5


# ---------------------------------------------------------------------------
# Location resolution and column mappings


def test_location_resolution_requires_explicit_base(tmp_path: Path) -> None:
    spec = _csv_spec()
    with pytest.raises(DatasetReadError) as caught:
        resolve_dataset_location(spec)
    assert caught.value.stage == "descriptor"
    assert resolve_dataset_location(spec, base_directory=tmp_path) == tmp_path / "data.csv"
    absolute = tmp_path / "abs.csv"
    absolute_spec = DatasetInputSpec(format=DatasetFormat.CSV, location=str(absolute))
    assert resolve_dataset_location(absolute_spec) == absolute
    with pytest.raises(TypeError):
        resolve_dataset_location(spec, base_directory=42)
    with pytest.raises(TypeError):
        resolve_dataset_location("not-a-descriptor", base_directory=tmp_path)


def test_column_mapping_validation_is_exact() -> None:
    assert validate_column_mapping({"prompt": "text"}) == {"prompt": "text"}
    with pytest.raises(TypeError):
        validate_column_mapping([("prompt", "text")])
    for broken in (
        {"speed": "x"},
        {1: "x"},
        {"prompt": ""},
        {"prompt": 3},
    ):
        with pytest.raises((TypeError, ValueError)):
            validate_column_mapping(broken)


def test_project_rows_applies_roles_and_rejects_missing_columns() -> None:
    rows = (
        DatasetRow(index=0, values={"text": "hello", "label": "hi"}),
        DatasetRow(index=1, values={"text": "world", "label": "earth"}),
    )
    projected = project_dataset_rows(rows, {"prompt": "text", "reference": "label"})
    assert projected == (
        {"prompt": "hello", "reference": "hi"},
        {"prompt": "world", "reference": "earth"},
    )
    with pytest.raises(DatasetReadError) as caught:
        project_dataset_rows(rows, {"prompt": "absent"})
    assert caught.value.stage == "format"


# ---------------------------------------------------------------------------
# Format readers


def test_csv_rows_are_deterministic_with_string_values(tmp_path: Path) -> None:
    _write_csv(tmp_path / "data.csv", "prompt,label\nhello,hi\nworld,earth\n")
    rows = read_dataset_rows(_csv_spec(), base_directory=tmp_path)
    assert [row.index for row in rows] == [0, 1]
    assert dict(rows[0].values) == {"prompt": "hello", "label": "hi"}
    assert all(type(value) is str for row in rows for value in row.values.values())
    again = read_dataset_rows(_csv_spec(), base_directory=tmp_path)
    assert rows == again


@pytest.mark.parametrize(
    "body",
    [
        "",  # no header
        "prompt,prompt\na,b\n",  # duplicate names
        "prompt,\na,b\n",  # empty header name
        "prompt,label\nonly-one\n",  # narrow row
        "prompt,label\na,b,c\n",  # wide row
        'prompt,label\n"a,b\n',  # malformed quoting
    ],
)
def test_csv_format_violations_are_rejected(tmp_path: Path, body: str) -> None:
    _write_csv(tmp_path / "data.csv", body)
    with pytest.raises(DatasetReadError) as caught:
        read_dataset_rows(_csv_spec(), base_directory=tmp_path)
    assert caught.value.stage == "format"


def test_csv_non_utf8_member_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "data.csv").write_bytes(b"prompt\n\xff\xfe\n")
    with pytest.raises(DatasetReadError) as caught:
        read_dataset_rows(_csv_spec(), base_directory=tmp_path)
    assert caught.value.stage == "format"


def test_jsonl_rows_carry_nested_json_values(tmp_path: Path) -> None:
    (tmp_path / "data.jsonl").write_text(
        '{"prompt": "a", "n": 1}\n{"prompt": "b", "nested": {"x": [1, 2]}}\n',
        encoding="utf-8",
    )
    spec = DatasetInputSpec(format=DatasetFormat.JSONL, location="data.jsonl")
    rows = read_dataset_rows(spec, base_directory=tmp_path)
    assert len(rows) == 2
    assert rows[1].values["nested"] == {"x": [1, 2]}


@pytest.mark.parametrize(
    "body",
    [
        '{"prompt": "a"}\n\n{"prompt": "b"}\n',  # blank line
        '{"prompt": "a"}\n{broken\n',  # invalid JSON
        '[1, 2]\n',  # not an object
        '{1: "a"}\n',  # non-string key
    ],
)
def test_jsonl_format_violations_are_rejected(tmp_path: Path, body: str) -> None:
    (tmp_path / "data.jsonl").write_text(body, encoding="utf-8")
    spec = DatasetInputSpec(format=DatasetFormat.JSONL, location="data.jsonl")
    with pytest.raises(DatasetReadError) as caught:
        read_dataset_rows(spec, base_directory=tmp_path)
    assert caught.value.stage == "format"


def test_text_rows_wrap_each_line_including_blank_interiors(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("one\n\nthree\n", encoding="utf-8")
    spec = DatasetInputSpec(format=DatasetFormat.TEXT, location="notes.txt")
    rows = read_dataset_rows(spec, base_directory=tmp_path)
    assert [dict(row.values) for row in rows] == [
        {"text": "one"},
        {"text": ""},
        {"text": "three"},
    ]


@parquet_mark
def test_parquet_rows_round_trip_typed_values(tmp_path: Path) -> None:
    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute("CREATE TABLE t (prompt VARCHAR, score DOUBLE, n INTEGER)")
        connection.execute("INSERT INTO t VALUES ('alpha', 0.5, 3), ('beta', 0.25, 4)")
        connection.execute(f"COPY t TO '{tmp_path / 'data.parquet'}' (FORMAT PARQUET)")
    finally:
        connection.close()
    spec = DatasetInputSpec(format=DatasetFormat.PARQUET, location="data.parquet")
    rows = read_dataset_rows(spec, base_directory=tmp_path)
    assert [dict(row.values) for row in rows] == [
        {"prompt": "alpha", "score": 0.5, "n": 3},
        {"prompt": "beta", "score": 0.25, "n": 4},
    ]
    # An explicit engine handle is honored instead of the lazy resolver.
    explicit = read_dataset_rows(spec, base_directory=tmp_path, duckdb=duckdb)
    assert explicit == rows


@parquet_mark
def test_parquet_dependency_failure_maps_to_dependency_stage(
    tmp_path: Path, monkeypatch
) -> None:
    # Production _load_duckdb wraps a missing extra into exactly this error;
    # the seam must propagate it unchanged.
    def raise_dependency():
        raise DatasetReadError("dependency", cause=ImportError("duckdb unavailable"))

    monkeypatch.setattr(datasets, "_load_duckdb", raise_dependency)
    spec = DatasetInputSpec(format=DatasetFormat.PARQUET, location="absent.parquet")
    with pytest.raises(DatasetReadError) as caught:
        read_dataset_rows(spec, base_directory=tmp_path)
    assert caught.value.stage == "dependency"
    assert isinstance(caught.value.__cause__, ImportError)


# ---------------------------------------------------------------------------
# HF-style local snapshots


def test_hf_snapshot_reads_members_in_sorted_order(tmp_path: Path) -> None:
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    (snapshot / "train.jsonl").write_text('{"prompt": "s1"}\n', encoding="utf-8")
    (snapshot / "test.jsonl").write_text('{"prompt": "s2"}\n{"prompt": "s3"}\n', encoding="utf-8")
    (snapshot / "README.md").write_text("ignore me", encoding="utf-8")
    (snapshot / ".hidden.jsonl").write_text('{"prompt": "nope"}\n', encoding="utf-8")
    spec = DatasetInputSpec(format=DatasetFormat.HF_DATASETS, location="snap")
    rows = read_dataset_rows(spec, base_directory=tmp_path)
    assert [row.values["prompt"] for row in rows] == ["s2", "s3", "s1"]


def test_hf_snapshot_shape_violations_are_rejected(tmp_path: Path) -> None:
    mixed = tmp_path / "mixed"
    mixed.mkdir()
    (mixed / "a.jsonl").write_text('{"prompt": "a"}\n', encoding="utf-8")
    (mixed / "b.csv").write_text("prompt\nb\n", encoding="utf-8")
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "README.md").write_text("docs only", encoding="utf-8")
    plain = tmp_path / "plain.txt"
    plain.write_text("not a directory", encoding="utf-8")
    for location, stage in (
        ("mixed", "format"),
        ("empty", "format"),
        ("absent-snapshot", "read"),
        ("plain.txt", "format"),
    ):
        spec = DatasetInputSpec(format=DatasetFormat.HF_DATASETS, location=location)
        with pytest.raises(DatasetReadError) as caught:
            read_dataset_rows(spec, base_directory=tmp_path)
        assert caught.value.stage == stage


def test_hf_hub_dataset_streams_a_bounded_prefix_with_explicit_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []

    def load_dataset(**kwargs: object):
        calls.append(kwargs)
        return iter(
            [
                {"prompt": "one", "label": 1},
                {"prompt": "two", "label": 2},
            ]
        )

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    spec = DatasetInputSpec(
        format=DatasetFormat.HF_DATASETS,
        location="org/benchmark",
        config_name="default",
        split="test",
        revision="a" * 40,
        allow_downloads=True,
    )
    rows = read_dataset_rows(spec, base_directory=tmp_path, max_rows=2)
    assert [row.values["prompt"] for row in rows] == ["one", "two"]
    assert calls == [
        {
            "path": "org/benchmark",
            "name": "default",
            "split": "test",
            "revision": "a" * 40,
            "streaming": True,
        }
    ]


def test_hf_hub_dataset_preview_does_not_reject_a_larger_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def load_dataset(**_kwargs: object):
        return iter(
            [
                {"text": "first"},
                {"text": "second"},
                {"text": "third"},
            ]
        )

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    spec = DatasetInputSpec(
        format=DatasetFormat.HF_DATASETS,
        location="org/benchmark",
        revision="a" * 40,
        allow_downloads=True,
    )

    rows = read_dataset_rows(spec, base_directory=tmp_path, max_rows=1)

    assert [row.values["text"] for row in rows] == ["first"]


def test_hf_hub_requires_explicit_download_opt_in(tmp_path: Path) -> None:
    spec = DatasetInputSpec(format=DatasetFormat.HF_DATASETS, location="org/benchmark")
    with pytest.raises(DatasetReadError, match="allow_downloads=True") as caught:
        read_dataset_rows(spec, base_directory=tmp_path)
    assert caught.value.stage == "read"


# ---------------------------------------------------------------------------
# Budgets and argument strictness


def test_row_byte_and_file_budgets_are_enforced(tmp_path: Path) -> None:
    _write_csv(tmp_path / "data.csv", "prompt,label\nhello,hi\nworld,earth\n")
    (tmp_path / "data.jsonl").write_text('{"a": "xxxx"}\n', encoding="utf-8")
    for kwargs, stage in (
        ({"max_rows": 1}, "budget"),
        ({"max_row_bytes": 8}, "budget"),
        ({"max_file_bytes": 2}, "budget"),
    ):
        with pytest.raises(DatasetReadError) as caught:
            read_dataset_rows(_csv_spec(), base_directory=tmp_path, **kwargs)
        assert caught.value.stage == stage
    jsonl_spec = DatasetInputSpec(format=DatasetFormat.JSONL, location="data.jsonl")
    with pytest.raises(DatasetReadError) as caught:
        read_dataset_rows(jsonl_spec, base_directory=tmp_path, max_row_bytes=4)
    assert caught.value.stage == "budget"


def test_argument_strictness_is_exact(tmp_path: Path) -> None:
    _write_csv(tmp_path / "data.csv", "prompt\nx\n")
    for kwargs in (
        {"max_rows": 0},
        {"max_rows": True},
        {"max_rows": "5"},
        {"max_row_bytes": 0},
        {"max_file_bytes": -1},
    ):
        with pytest.raises(TypeError):
            read_dataset_rows(_csv_spec(), base_directory=tmp_path, **kwargs)


def test_iterable_descriptors_have_no_file_reader(tmp_path: Path) -> None:
    spec = DatasetInputSpec(format=DatasetFormat.ITERABLE, location="unused")
    with pytest.raises(DatasetReadError) as caught:
        read_dataset_rows(spec, base_directory=tmp_path)
    assert caught.value.stage == "descriptor"


def test_directory_locations_for_file_formats_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "adir").mkdir()
    spec = DatasetInputSpec(format=DatasetFormat.CSV, location="adir")
    with pytest.raises(DatasetReadError) as caught:
        read_dataset_rows(spec, base_directory=tmp_path)
    assert caught.value.stage == "format"


def test_missing_files_fail_at_read_stage(tmp_path: Path) -> None:
    with pytest.raises(DatasetReadError) as caught:
        read_dataset_rows(_csv_spec("missing.csv"), base_directory=tmp_path)
    assert caught.value.stage == "read"


def test_csv_reading_never_resolves_the_engine(tmp_path: Path, monkeypatch) -> None:
    _write_csv(tmp_path / "data.csv", "prompt\nx\n")

    def forbidden():
        raise AssertionError("CSV reading must not resolve DuckDB")

    monkeypatch.setattr(datasets, "_load_duckdb", forbidden)
    rows = read_dataset_rows(_csv_spec(), base_directory=tmp_path)
    assert len(rows) == 1


def test_mapped_columns_are_validated_at_read_time(tmp_path: Path) -> None:
    _write_csv(tmp_path / "data.csv", "prompt,label\nhello,hi\n")
    good = _csv_spec(column_mapping={"prompt": "prompt", "reference": "label"})
    rows = read_dataset_rows(good, base_directory=tmp_path)
    assert project_dataset_rows(rows, good.column_mapping) == (
        {"prompt": "hello", "reference": "hi"},
    )
    bad = _csv_spec(column_mapping={"prompt": "absent"})
    with pytest.raises(DatasetReadError) as caught:
        read_dataset_rows(bad, base_directory=tmp_path)
    assert caught.value.stage == "format"

    (tmp_path / "notes.txt").write_text("one\ntwo\n", encoding="utf-8")
    text_spec = DatasetInputSpec(
        format=DatasetFormat.TEXT,
        location="notes.txt",
        column_mapping={"prompt": "text"},
        mode=RunMode.TEACHER_FORCED,
    )
    text_rows = read_dataset_rows(text_spec, base_directory=tmp_path)
    assert project_dataset_rows(text_rows, text_spec.column_mapping) == (
        {"prompt": "one"},
        {"prompt": "two"},
    )


# ---------------------------------------------------------------------------
# Deterministic batch planning


def test_plan_without_options_is_one_identity_batch() -> None:
    assert plan_dataset_batches(0) == ()
    assert plan_dataset_batches(3) == ((0, 1, 2),)


def test_batch_size_splits_with_remainder_batch() -> None:
    assert plan_dataset_batches(7, batch_size=3) == ((0, 1, 2), (3, 4, 5), (6,))
    assert plan_dataset_batches(2, batch_size=5) == ((0, 1),)


def test_shuffle_requires_seed_and_stays_deterministic() -> None:
    with pytest.raises(DatasetReadError) as caught:
        plan_dataset_batches(5, shuffle=True)
    assert caught.value.stage == "descriptor"
    first = plan_dataset_batches(12, shuffle=True, seed=7)
    second = plan_dataset_batches(12, shuffle=True, seed=7)
    other = plan_dataset_batches(12, shuffle=True, seed=8)
    assert first == second
    assert first != other
    flat = [index for batch in first for index in batch]
    assert sorted(flat) == list(range(12))


def test_sample_cap_selects_deterministically() -> None:
    assert plan_dataset_batches(10, sample_cap=3) == ((0, 1, 2),)
    seeded = plan_dataset_batches(50, sample_cap=5, seed=11)
    again = plan_dataset_batches(50, sample_cap=5, seed=11)
    flat = [index for batch in seeded for index in batch]
    assert seeded == again
    assert len(flat) == 5
    assert flat == sorted(flat)
    assert set(flat) <= set(range(50))
    assert plan_dataset_batches(4, sample_cap=9) == ((0, 1, 2, 3),)


def test_combined_cap_shuffle_and_batches_compose() -> None:
    plan = plan_dataset_batches(40, batch_size=4, sample_cap=10, shuffle=True, seed=3)
    flat = [index for batch in plan for index in batch]
    assert len(plan) == 3  # 10 selected rows in batches of 4/4/2
    assert [len(batch) for batch in plan] == [4, 4, 2]
    assert sorted(flat) == sorted(plan_dataset_batches(40, sample_cap=10, seed=3)[0])


def test_plan_argument_strictness_is_exact() -> None:
    for kwargs in (
        {"batch_size": 0},
        {"batch_size": True},
        {"sample_cap": 0},
        {"shuffle": "yes"},
        {"seed": -1},
        {"seed": True},
    ):
        with pytest.raises((TypeError, DatasetReadError)):
            plan_dataset_batches(5, **kwargs)
    with pytest.raises(TypeError):
        plan_dataset_batches(-1)
    with pytest.raises(TypeError):
        plan_dataset_batches(True)
    with pytest.raises(TypeError):
        plan_dataset_batches("5")
