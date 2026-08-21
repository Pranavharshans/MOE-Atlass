"""Bounded deterministic dataset reading over run-specification descriptors.

``DatasetInputSpec`` (see ``moeatlas.runs.specs``) is identity only: it
describes JSONL/CSV/Parquet/text/HF-style data and never reads it. This
module is that later engine step's data layer. It turns a descriptor into a
deterministic ``DatasetRow`` tuple under strict row/byte/file budgets,
validates column mappings against a fixed task-role vocabulary, and plans
deterministic batch schedules (sample caps, shuffles, batches) derived from
SHA-256 ordering keys so results never depend on hash seeds, locales, or the
platform random generator.

Descriptors never fetch data: locations resolve against an explicit local
base directory (or are absolute paths supplied by the caller), and
``hf_datasets`` format means an existing local snapshot directory, never a
network download. Reading performs no clock reads, no network access, and no
model-runtime work; DuckDB is imported lazily at call time and only for
Parquet members.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ..runs.specs import DatasetFormat, DatasetInputSpec

DATASET_READER_SCHEMA_VERSION = "1.0"
"""Schema version of the bounded dataset-reader contracts."""

DATASET_COLUMN_ROLES = ("domain", "prompt", "reference", "task")
"""Fixed v1 task-role vocabulary accepted as column-mapping keys."""

_DEFAULT_MAX_ROWS = 10_000
_DEFAULT_MAX_ROW_BYTES = 65_536
_DEFAULT_MAX_FILE_BYTES = 100_000_000
_TEXT_VALUE_KEY = "text"

_STAGES = frozenset({"descriptor", "dependency", "format", "budget", "read"})
_DATA_SUFFIXES = {
    ".csv": DatasetFormat.CSV,
    ".jsonl": DatasetFormat.JSONL,
    ".parquet": DatasetFormat.PARQUET,
    ".txt": DatasetFormat.TEXT,
}


class DatasetReadError(RuntimeError):
    """Safe fixed-stage failure for bounded dataset reads."""

    def __init__(
        self, stage: str, message: str | None = None, *, cause: BaseException | None = None
    ) -> None:
        if stage not in _STAGES:
            raise ValueError("dataset read error stage is not supported")
        self.stage = stage
        if message is None:
            super().__init__(f"dataset read failed at {stage}")
        else:
            super().__init__(f"dataset read failed at {stage}: {message}")
        if cause is not None:
            self.__cause__ = cause


@dataclass(frozen=True, slots=True)
class DatasetRow:
    """One bounded dataset record at its deterministic read-order index."""

    index: int
    values: Mapping[str, Any]

    def __post_init__(self) -> None:
        if type(self.index) is not int or isinstance(self.index, bool):
            raise TypeError("index must be an integer")
        if self.index < 0:
            raise ValueError("index must be a non-negative integer")
        if not isinstance(self.values, Mapping) or not all(
            type(key) is str for key in self.values
        ):
            raise TypeError("values must be a mapping with string keys")
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))


def _load_duckdb() -> Any:
    """Resolve DuckDB lazily; Parquet reads are the only consumer."""

    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise DatasetReadError("dependency", cause=exc)
    return duckdb


def _strict_positive(value: object, name: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or isinstance(value, bool) or value <= 0:
        raise TypeError(f"{name} must be None or a strict positive integer")
    return value


def _canonical_row_bytes(values: Mapping[str, Any]) -> bytes:
    try:
        payload = json.dumps(
            dict(values),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise DatasetReadError("format", "row is not canonically encodable", cause=exc)
    return payload.encode("utf-8")


def resolve_dataset_location(
    descriptor: DatasetInputSpec, *, base_directory: str | Path | None = None
) -> Path:
    """Resolve a descriptor location label against an explicit local base."""

    if not isinstance(descriptor, DatasetInputSpec):
        raise TypeError("descriptor must be a DatasetInputSpec")
    if base_directory is not None and not isinstance(base_directory, str | Path):
        raise TypeError("base_directory must be a string or Path")
    location = Path(descriptor.location)
    if location.is_absolute():
        return location
    if base_directory is None:
        raise DatasetReadError(
            "descriptor",
            f"relative dataset location {descriptor.location!r} requires base_directory",
        )
    return Path(base_directory) / location


def validate_column_mapping(column_mapping: Mapping[str, str]) -> dict[str, str]:
    """Validate a column mapping against the fixed task-role vocabulary."""

    if not isinstance(column_mapping, Mapping):
        raise TypeError("column_mapping must be a mapping")
    mapping = dict(column_mapping)
    for role, column in mapping.items():
        if type(role) is not str or role not in DATASET_COLUMN_ROLES:
            raise ValueError(f"column-mapping role is not supported: {role!r}")
        if type(column) is not str or not column:
            raise ValueError(
                f"column-mapping target for role {role!r} must be a non-empty string"
            )
    return mapping


def project_dataset_rows(
    rows: tuple[DatasetRow, ...], column_mapping: Mapping[str, str]
) -> tuple[dict[str, Any], ...]:
    """Project validated rows onto their task roles."""

    mapping = validate_column_mapping(column_mapping)
    projected: list[dict[str, Any]] = []
    for row in rows:
        try:
            projected.append({role: row.values[column] for role, column in mapping.items()})
        except KeyError as exc:
            raise DatasetReadError(
                "format", f"mapped column is missing: {exc.args[0]}", cause=exc
            )
    return tuple(projected)


def _checked_file(path: Path, max_file_bytes: int) -> None:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise DatasetReadError("read", cause=exc)
    if size > max_file_bytes:
        raise DatasetReadError(
            "budget", f"dataset member exceeds the {max_file_bytes} byte budget"
        )


def _read_text_member(path: Path, max_file_bytes: int) -> str:
    _checked_file(path, max_file_bytes)
    try:
        return path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise DatasetReadError("read", cause=exc)
    except UnicodeDecodeError as exc:
        raise DatasetReadError("format", f"{path.name} is not valid UTF-8", cause=exc)


def _bounded_rows(
    rows: list[dict[str, Any]], max_rows: int, max_row_bytes: int
) -> list[dict[str, Any]]:
    if len(rows) > max_rows:
        raise DatasetReadError("budget", f"dataset exceeds the {max_rows} row budget")
    checked: list[dict[str, Any]] = []
    for row in rows:
        if len(_canonical_row_bytes(row)) > max_row_bytes:
            raise DatasetReadError("budget", f"row exceeds the {max_row_bytes} byte budget")
        checked.append(row)
    return checked


def _read_csv_rows(
    path: Path, max_rows: int, max_row_bytes: int, max_file_bytes: int
) -> list[dict[str, Any]]:
    text = _read_text_member(path, max_file_bytes)
    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        records = [tuple(row) for row in reader]
    except csv.Error as exc:
        raise DatasetReadError("format", "CSV member is malformed", cause=exc)
    if not records:
        raise DatasetReadError("format", "CSV member has no header row")
    header = records[0]
    if not header or any(not name for name in header):
        raise DatasetReadError("format", "CSV header must be non-empty names")
    if len(set(header)) != len(header):
        raise DatasetReadError("format", "CSV header names must be unique")
    rows: list[dict[str, Any]] = []
    for record in records[1 : max_rows + 2]:
        if len(record) != len(header):
            raise DatasetReadError("format", "CSV row width does not match the header")
        rows.append(dict(zip(header, record)))
    return _bounded_rows(rows, max_rows, max_row_bytes)


def _read_jsonl_rows(
    path: Path, max_rows: int, max_row_bytes: int, max_file_bytes: int
) -> list[dict[str, Any]]:
    lines = _read_text_member(path, max_file_bytes).split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            raise DatasetReadError("format", f"JSONL member has a blank line at {number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetReadError("format", f"JSONL line {number} is not valid JSON", cause=exc)
        if not isinstance(value, dict):
            raise DatasetReadError("format", f"JSONL line {number} is not a JSON object")
        if not all(type(key) is str for key in value):
            raise DatasetReadError("format", f"JSONL line {number} has non-string keys")
        rows.append(value)
    return _bounded_rows(rows, max_rows, max_row_bytes)


def _read_text_rows(
    path: Path, max_rows: int, max_row_bytes: int, max_file_bytes: int
) -> list[dict[str, Any]]:
    lines = _read_text_member(path, max_file_bytes).split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    rows = [{_TEXT_VALUE_KEY: line} for line in lines]
    return _bounded_rows(rows, max_rows, max_row_bytes)


def _read_parquet_rows(
    path: Path, max_rows: int, max_row_bytes: int, max_file_bytes: int, duckdb: Any
) -> list[dict[str, Any]]:
    _checked_file(path, max_file_bytes)
    connection = duckdb.connect(database=":memory:")
    try:
        cursor = connection.execute(
            "SELECT * FROM read_parquet(?) LIMIT ?", [str(path), max_rows + 1]
        )
        names = [item[0] for item in cursor.description]
        records = cursor.fetchall()
    finally:
        connection.close()
    rows = [dict(zip(names, record)) for record in records]
    return _bounded_rows(rows, max_rows, max_row_bytes)


def _read_member(
    path: Path,
    fmt: DatasetFormat,
    max_rows: int,
    max_row_bytes: int,
    max_file_bytes: int,
    duckdb: Any,
) -> list[dict[str, Any]]:
    if fmt is DatasetFormat.CSV:
        return _read_csv_rows(path, max_rows, max_row_bytes, max_file_bytes)
    if fmt is DatasetFormat.JSONL:
        return _read_jsonl_rows(path, max_rows, max_row_bytes, max_file_bytes)
    if fmt is DatasetFormat.TEXT:
        return _read_text_rows(path, max_rows, max_row_bytes, max_file_bytes)
    if fmt is DatasetFormat.PARQUET:
        engine = duckdb if duckdb is not None else _load_duckdb()
        return _read_parquet_rows(
            path, max_rows, max_row_bytes, max_file_bytes, engine
        )
    raise DatasetReadError("descriptor", f"format is not file-readable: {fmt}")


def _snapshot_members(directory: Path) -> list[Path]:
    if not directory.exists():
        raise DatasetReadError("read", f"dataset snapshot does not exist: {directory}")
    if not directory.is_dir():
        raise DatasetReadError("format", f"HF-style location is not a directory: {directory}")
    members = sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in _DATA_SUFFIXES
    )
    if not members:
        raise DatasetReadError("format", "HF-style snapshot contains no data files")
    kinds = {_DATA_SUFFIXES[path.suffix.lower()] for path in members}
    if len(kinds) != 1:
        raise DatasetReadError("format", "HF-style snapshot mixes data formats")
    return members


def read_dataset_rows(
    descriptor: DatasetInputSpec,
    *,
    base_directory: str | Path | None = None,
    max_rows: int = _DEFAULT_MAX_ROWS,
    max_row_bytes: int = _DEFAULT_MAX_ROW_BYTES,
    max_file_bytes: int = _DEFAULT_MAX_FILE_BYTES,
    duckdb: Any = None,
) -> tuple[DatasetRow, ...]:
    """Read one descriptor's local data into bounded deterministic rows."""

    if not isinstance(descriptor, DatasetInputSpec):
        raise TypeError("descriptor must be a DatasetInputSpec")
    if type(max_rows) is not int or isinstance(max_rows, bool) or max_rows <= 0:
        raise TypeError("max_rows must be a strict positive integer")
    if type(max_row_bytes) is not int or isinstance(max_row_bytes, bool) or max_row_bytes <= 0:
        raise TypeError("max_row_bytes must be a strict positive integer")
    if (
        type(max_file_bytes) is not int
        or isinstance(max_file_bytes, bool)
        or max_file_bytes <= 0
    ):
        raise TypeError("max_file_bytes must be a strict positive integer")
    mapping = validate_column_mapping(descriptor.column_mapping)

    fmt = descriptor.format
    if fmt is DatasetFormat.ITERABLE:
        raise DatasetReadError(
            "descriptor", "iterable datasets carry no file location for the reader"
        )
    location = resolve_dataset_location(descriptor, base_directory=base_directory)

    if fmt is DatasetFormat.HF_DATASETS:
        rows: list[dict[str, Any]] = []
        for member in _snapshot_members(location):
            remaining = max_rows - len(rows)
            member_format = _DATA_SUFFIXES[member.suffix.lower()]
            rows.extend(
                _read_member(
                    member, member_format, remaining, max_row_bytes, max_file_bytes, duckdb
                )
            )
            if len(rows) > max_rows:
                raise DatasetReadError("budget", f"dataset exceeds the {max_rows} row budget")
    else:
        if location.is_dir():
            raise DatasetReadError("format", f"dataset location is a directory: {location}")
        rows = _read_member(location, fmt, max_rows, max_row_bytes, max_file_bytes, duckdb)

    if mapping and rows:
        wanted = set(mapping.values())
        for row in rows:
            missing = sorted(wanted - set(row))
            if missing:
                raise DatasetReadError(
                    "format",
                    f"mapped columns are missing from {location.name}: {', '.join(missing)}",
                )

    return tuple(DatasetRow(index=index, values=row) for index, row in enumerate(rows))


def _ordering_key(purpose: str, seed: int, index: int) -> bytes:
    material = f"{purpose}\x00{seed}\x00{index}".encode()
    return hashlib.sha256(material).digest()


def plan_dataset_batches(
    total_rows: int,
    *,
    batch_size: int | None = None,
    sample_cap: int | None = None,
    shuffle: bool = False,
    seed: int | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Plan a deterministic batch schedule over ``total_rows`` indices.

    Selection and ordering derive from SHA-256 keys over ``(purpose, seed,
    index)``, so identical arguments always produce identical schedules on
    every platform and process. ``sample_cap`` without a seed keeps the
    deterministic first-cap prefix; with a seed it selects by digest order
    and restores ascending index order. ``shuffle`` requires a seed and
    permutes by digest order before batching.
    """

    if type(total_rows) is not int or isinstance(total_rows, bool) or total_rows < 0:
        raise TypeError("total_rows must be a non-negative integer")
    capped = _strict_positive(batch_size, "batch_size")
    limit = _strict_positive(sample_cap, "sample_cap")
    if type(shuffle) is not bool:
        raise TypeError("shuffle must be a boolean")
    if seed is not None and (type(seed) is not int or isinstance(seed, bool) or seed < 0):
        raise TypeError("seed must be None or a non-negative integer")
    if shuffle and seed is None:
        raise DatasetReadError("descriptor", "shuffled batch planning requires an integer seed")

    indices = list(range(total_rows))
    if limit is not None:
        if seed is None:
            indices = indices[:limit]
        else:
            chosen = sorted(indices, key=lambda i: _ordering_key("sample", seed, i))[:limit]
            indices = sorted(chosen)
    if shuffle:
        indices = sorted(indices, key=lambda i: _ordering_key("shuffle", seed, i))

    width = capped if capped is not None else max(len(indices), 1)
    return tuple(tuple(indices[start : start + width]) for start in range(0, len(indices), width))
