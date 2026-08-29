"""Bounded Arrow-to-Parquet writer for large routing shards.

This module is imported lazily by :mod:`routing_shards`.  Server/model
installations already carry PyArrow through ``datasets``; store-only
installations keep the dependency-free DuckDB fallback.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa

_BATCH_ROWS = 65_536
_ARROW_TYPES = {
    "VARCHAR": pa.string(),
    "BIGINT": pa.int64(),
    "DOUBLE": pa.float64(),
    "BOOLEAN": pa.bool_(),
}


def _schema(columns: tuple[tuple[str, str], ...]) -> pa.Schema:
    return pa.schema([(name, _ARROW_TYPES[column_type]) for name, column_type in columns])


def _batches(
    rows: tuple[tuple[object, ...], ...],
    schema: pa.Schema,
) -> Iterator[pa.RecordBatch]:
    """Yield bounded columnar batches without transposing the whole shard."""

    width = len(schema)
    for start in range(0, len(rows), _BATCH_ROWS):
        chunk = rows[start : start + _BATCH_ROWS]
        arrays = [
            pa.array((row[index] for row in chunk), type=schema[index].type)
            for index in range(width)
        ]
        yield pa.RecordBatch.from_arrays(arrays, schema=schema)


def _write_one(
    connection: Any,
    path: Path,
    rows: tuple[tuple[object, ...], ...],
    columns: tuple[tuple[str, str], ...],
) -> None:
    schema = _schema(columns)
    reader = pa.RecordBatchReader.from_batches(schema, _batches(rows, schema))
    connection.from_arrow(reader).order("event_index").write_parquet(
        str(path), compression="zstd", overwrite=False
    )


def write_parquets(
    duckdb: Any,
    stage: Path,
    token_rows: tuple[tuple[object, ...], ...],
    routing_rows: tuple[tuple[object, ...], ...],
    expert_rows: tuple[tuple[object, ...], ...],
    token_columns: tuple[tuple[str, str], ...],
    routing_columns: tuple[tuple[str, str], ...],
    expert_columns: tuple[tuple[str, str], ...],
    filenames: tuple[str, str, str],
) -> None:
    """Stream the three immutable shard tables through Arrow batches."""

    connection = duckdb.connect(database=":memory:")
    try:
        for filename, rows, columns in (
            (filenames[0], token_rows, token_columns),
            (filenames[1], routing_rows, routing_columns),
            (filenames[2], expert_rows, expert_columns),
        ):
            _write_one(connection, stage / filename, rows, columns)
    finally:
        connection.close()


__all__ = ["write_parquets"]
