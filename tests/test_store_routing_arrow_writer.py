"""Focused coverage for the vectorized routing-shard writer."""

from __future__ import annotations

from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

from moeatlas.store.routing_arrow_writer import write_parquets  # noqa: E402
from moeatlas.store.routing_shards import (  # noqa: E402
    _EXPERT_COLUMNS,
    _EXPERTS_FILE,
    _ROUTING_COLUMNS,
    _ROUTING_FILE,
    _TOKEN_COLUMNS,
    _TOKENS_FILE,
)


def test_arrow_writer_preserves_order_and_empty_expert_schema(tmp_path: Path) -> None:
    shard = "shard:" + "a" * 64
    token_rows = (
        (
            "2.0",
            shard,
            1,
            "1.0",
            "token",
            "token:b",
            "run:1",
            "row-0",
            1,
            8,
            None,
            False,
            "prefill",
        ),
        (
            "2.0",
            shard,
            0,
            "1.0",
            "token",
            "token:a",
            "run:1",
            "row-0",
            0,
            7,
            None,
            False,
            "prefill",
        ),
    )
    routing_rows = (
        (
            "2.0",
            shard,
            1,
            "1.0",
            "routing",
            "token:b",
            "component:layer",
            0,
            "component:expert",
            1.0,
            None,
            1.0,
            True,
        ),
        (
            "2.0",
            shard,
            0,
            "1.0",
            "routing",
            "token:a",
            "component:layer",
            0,
            "component:expert",
            1.0,
            None,
            1.0,
            True,
        ),
    )

    write_parquets(
        duckdb,
        tmp_path,
        token_rows,
        routing_rows,
        (),
        _TOKEN_COLUMNS,
        _ROUTING_COLUMNS,
        _EXPERT_COLUMNS,
        (_TOKENS_FILE, _ROUTING_FILE, _EXPERTS_FILE),
    )

    connection = duckdb.connect(database=":memory:")
    try:
        tokens = connection.execute(
            "SELECT event_index FROM read_parquet(?)", [str(tmp_path / _TOKENS_FILE)]
        ).fetchall()
        routing = connection.execute(
            "SELECT event_index FROM read_parquet(?)", [str(tmp_path / _ROUTING_FILE)]
        ).fetchall()
        expert_count = connection.execute(
            "SELECT count(*) FROM read_parquet(?)", [str(tmp_path / _EXPERTS_FILE)]
        ).fetchone()
    finally:
        connection.close()

    assert tokens == [(0,), (1,)]
    assert routing == [(0,), (1,)]
    assert expert_count == (0,)
