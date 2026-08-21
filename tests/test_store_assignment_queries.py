"""Contract tests for the public routing-run assignment query seam.

The seam is the model-neutral reader/query surface that analysis and services
consume instead of concrete shard internals: per-shard validated summaries
with typed error carriers, canonical ordering, budgets, and conflict
detection.  Equivalence with ``aggregate_routing_load`` is pinned on both
single- and multi-shard runs.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import fields
from pathlib import Path

import pytest

try:
    import duckdb
except ImportError:  # pragma: no cover - exercised only without the store extra
    duckdb = None

import moeatlas.store.routing_shards as storage
from moeatlas.analysis.routing_load import aggregate_routing_load
from moeatlas.store import (
    RoutingRunInventoryError,
    RoutingRunQueryError,
    RoutingRunReader,
    RoutingShardAssignmentQuery,
    RoutingShardError,
    reader_from_workspace,
)
from moeatlas.store.ports import DuckDBRoutingShardStore

from .test_store_routing_shards import _run_result, _workspace
from .test_store_run_export import _rekey_result


@pytest.fixture(autouse=True)
def _duckdb_required() -> None:
    if duckdb is None:
        pytest.skip("duckdb store extra is unavailable")


_UNSET = object()


def _query(workspace: Path, run_key: str, layer_keys, expert_keys, routed_top_k, **overrides):
    engine = overrides.pop("duckdb", _UNSET)
    connection = overrides.pop("connection", _UNSET)
    if engine is _UNSET:
        engine = storage._load_duckdb()
        connection = engine.connect(database=":memory:")
    try:
        return storage.query_routing_run_assignments(
            workspace,
            run_key=run_key,
            layer_keys=layer_keys,
            expert_keys=expert_keys,
            routed_top_k=routed_top_k,
            max_routing_rows=overrides.pop("max_routing_rows", 1_000_000),
            max_source_bytes=overrides.pop("max_source_bytes", 1_000_000_000),
            duckdb=engine,
            connection=connection,
            **overrides,
        )
    finally:
        if connection is not None and getattr(connection, "close", None) is not None:
            connection.close()


def _universe_of(inspection: object):
    from moeatlas.analysis.routing_load import _fresh_universe

    universe = _fresh_universe(inspection)
    return universe.layer_keys, universe.expert_keys, universe.routed_top_k


def _expected_counts(result: object) -> tuple[tuple[str, str, int], ...]:
    counts = Counter(
        (event.layer_key, event.expert_key) for event in result.routing_events
    )
    return tuple(sorted((layer, expert, count) for (layer, expert), count in counts.items()))


# ---------------------------------------------------------------------------
# Public surface and dataclass strictness


def test_seam_surface_is_pinned() -> None:
    assert callable(storage.query_routing_run_assignments)
    assert tuple(field.name for field in fields(RoutingShardAssignmentQuery)) == (
        "shard_key",
        "token_count",
        "routing_count",
        "token_keys",
        "routing_links",
        "assignment_counts",
    )
    assert RoutingShardAssignmentQuery.__slots__ == (
        "shard_key",
        "token_count",
        "routing_count",
        "token_keys",
        "routing_links",
        "assignment_counts",
    )
    record = RoutingShardAssignmentQuery(
        shard_key=f"shard:{'a' * 64}",
        token_count=1,
        routing_count=2,
        token_keys=frozenset({"tok"}),
        routing_links=frozenset({("layer", "expert", 0)}),
        assignment_counts=(("layer", "expert", 2),),
    )
    assert record == record
    with pytest.raises((AttributeError, TypeError)):
        record.token_count = 5  # type: ignore[misc]
    assert str(RoutingRunQueryError()) == "routing run query failed"


@pytest.mark.parametrize(
    ("kwargs"),
    [
        {"shard_key": "shard:short"},
        {"shard_key": 123},
        {"token_count": 0},
        {"token_count": True},
        {"token_count": "2"},
        {"routing_count": -1},
        {"token_keys": frozenset()},
        {"token_keys": {"tok"}},
        {"routing_links": frozenset()},
        {"assignment_counts": ()},
        {"assignment_counts": (("layer", "expert"),)},
        {"assignment_counts": (("layer", "expert", 0),)},
        {"assignment_counts": ((("layer", "expert", True)),)},
        {"assignment_counts": (("layer", "expert", "2"),)},
        {"assignment_counts": (("b", "expert", 1), ("a", "expert", 1))},
    ],
)
def test_assignment_query_dataclass_rejects_invalid_rows(kwargs: dict) -> None:
    base: dict = {
        "shard_key": f"shard:{'a' * 64}",
        "token_count": 1,
        "routing_count": 2,
        "token_keys": frozenset({"tok"}),
        "routing_links": frozenset({("layer", "expert", 0)}),
        "assignment_counts": (("layer", "expert", 2),),
    }
    base.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        RoutingShardAssignmentQuery(**base)


# ---------------------------------------------------------------------------
# Happy paths


def test_single_shard_query_returns_validated_summary(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    result, _, inspection = _run_result("legacy", token_count=2)
    receipt = storage.append_routing_shard(workspace, result)
    layer_keys, expert_keys, routed_top_k = _universe_of(inspection)

    records = _query(workspace, receipt.run_key, layer_keys, expert_keys, routed_top_k)

    assert len(records) == 1
    record = records[0]
    assert type(record) is RoutingShardAssignmentQuery
    assert record.shard_key == receipt.shard_key
    assert record.token_count == len(result.token_events)
    assert record.routing_count == len(result.routing_events)
    assert record.assignment_counts == _expected_counts(result)
    assert sum(count for _, _, count in record.assignment_counts) == record.routing_count
    assert len(record.token_keys) == len(result.token_events)
    assert len(record.routing_links) == len(result.routing_events)


def test_multi_shard_queries_are_per_shard_and_canonically_ordered(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    first, _, inspection = _run_result("legacy", token_count=2)
    first_receipt = storage.append_routing_shard(workspace, first)
    second = _rekey_result(first, run_key=first_receipt.run_key, offset=97)
    second_receipt = storage.append_routing_shard(workspace, second)
    assert second_receipt.shard_key != first_receipt.shard_key
    layer_keys, expert_keys, routed_top_k = _universe_of(inspection)

    records = _query(workspace, first_receipt.run_key, layer_keys, expert_keys, routed_top_k)

    assert [record.shard_key for record in records] == sorted(
        [first_receipt.shard_key, second_receipt.shard_key]
    )
    by_shard = {first_receipt.shard_key: first, second_receipt.shard_key: second}
    for record in records:
        source = by_shard[record.shard_key]
        # Each summary carries its own shard's grouped distribution; a stale
        # shared path would repeat the last shard's rows for every record.
        assert record.assignment_counts == _expected_counts(source)
        assert record.token_count == len(source.token_events)
        assert record.routing_count == len(source.routing_events)


def test_repeated_queries_are_deterministic(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    result, _, inspection = _run_result("packed", token_count=2)
    receipt = storage.append_routing_shard(workspace, result)
    layer_keys, expert_keys, routed_top_k = _universe_of(inspection)

    first = _query(workspace, receipt.run_key, layer_keys, expert_keys, routed_top_k)
    second = _query(workspace, receipt.run_key, layer_keys, expert_keys, routed_top_k)
    assert first == second


def test_analysis_aggregation_matches_folded_seam_records(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    first, _, inspection = _run_result("legacy", token_count=2)
    receipt = storage.append_routing_shard(workspace, first)
    second = _rekey_result(first, run_key=receipt.run_key, offset=31)
    storage.append_routing_shard(workspace, second)
    layer_keys, expert_keys, routed_top_k = _universe_of(inspection)

    records = _query(workspace, receipt.run_key, layer_keys, expert_keys, routed_top_k)
    matrix = aggregate_routing_load(
        workspace,
        inspection,
        run_key=receipt.run_key,
        max_routing_rows=1_000_000,
        max_source_bytes=1_000_000_000,
        max_matrix_cells=10_000,
    )

    counts = [[0 for _ in range(len(expert_keys[index]))] for index in range(len(layer_keys))]
    for record in records:
        for layer_key, expert_key, count in record.assignment_counts:
            position = layer_keys.index(layer_key)
            counts[position][expert_keys[position].index(expert_key)] += count
    assert matrix.shard_keys == tuple(sorted(record.shard_key for record in records))
    assert matrix.token_count == sum(record.token_count for record in records)
    assert matrix.assignment_count == sum(record.routing_count for record in records)
    assert matrix.assignment_counts == tuple(tuple(row) for row in counts)


# ---------------------------------------------------------------------------
# Argument validation and laziness


def test_unknown_run_fails_as_source_without_touching_the_engine(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    with pytest.raises(ValueError) as caught:
        _query(workspace, "run-with-no-shards", ("l",), (("e",),), 1, duckdb=None, connection=None)
    assert str(caught.value) == "run has no committed routing shards"


@pytest.mark.parametrize(
    ("kwargs"),
    [
        {"max_routing_rows": 0},
        {"max_routing_rows": -1},
        {"max_routing_rows": True},
        {"max_routing_rows": "10"},
        {"max_source_bytes": 0},
        {"routed_top_k": 0},
        {"routed_top_k": "1"},
    ],
)
def test_query_budget_arguments_are_strict(tmp_path: Path, kwargs: dict) -> None:
    workspace = _workspace(tmp_path)
    with pytest.raises((TypeError, ValueError)):
        _query(workspace, "any-run", ("l",), (("e",),), 1, **kwargs)


def test_workspace_and_universe_shapes_are_validated(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    with pytest.raises(TypeError):
        _query(123, "any-run", ("l",), (("e",),), 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        _query(workspace, "any-run", ["l"], (("e",),), 1)
    with pytest.raises(TypeError):
        _query(workspace, "any-run", ("l", "l2"), (("e",),), 1)


# ---------------------------------------------------------------------------
# Budgets, staging entries, corruption, conflicts


def test_row_and_byte_budgets_fail_at_inventory_budget_stage(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    result, _, inspection = _run_result("legacy", token_count=2)
    receipt = storage.append_routing_shard(workspace, result)
    layer_keys, expert_keys, routed_top_k = _universe_of(inspection)

    with pytest.raises(RoutingRunInventoryError) as row_caught:
        _query(
            workspace,
            receipt.run_key,
            layer_keys,
            expert_keys,
            routed_top_k,
            max_routing_rows=1,
        )
    assert row_caught.value.stage == "budget"

    with pytest.raises(RoutingRunInventoryError) as byte_caught:
        _query(
            workspace,
            receipt.run_key,
            layer_keys,
            expert_keys,
            routed_top_k,
            max_source_bytes=1,
        )
    assert byte_caught.value.stage == "budget"


def test_staging_entries_are_skipped_and_bad_children_are_reopen(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    result, _, inspection = _run_result("legacy", token_count=1)
    receipt = storage.append_routing_shard(workspace, result)
    run_parent = (workspace / receipt.relative_path).parent
    staging = run_parent / f".staging-{'b' * 16}"
    staging.mkdir()
    (staging / "partial.parquet").write_bytes(b"partial")
    layer_keys, expert_keys, routed_top_k = _universe_of(inspection)

    records = _query(workspace, receipt.run_key, layer_keys, expert_keys, routed_top_k)
    assert len(records) == 1

    malformed = run_parent / ".staging-not-canonical!"
    malformed.mkdir()
    with pytest.raises(RoutingShardError) as reopen_caught:
        _query(workspace, receipt.run_key, layer_keys, expert_keys, routed_top_k)
    assert reopen_caught.value.stage == "reopen"
    malformed.rmdir()

    stranger = run_parent / "unrelated-directory"
    stranger.mkdir()
    with pytest.raises(RoutingShardError) as stranger_caught:
        _query(workspace, receipt.run_key, layer_keys, expert_keys, routed_top_k)
    assert stranger_caught.value.stage == "reopen"


def test_tampered_shard_fails_at_reopen_before_grouped_reads(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    result, _, inspection = _run_result("legacy", token_count=1)
    receipt = storage.append_routing_shard(workspace, result)
    shard = workspace / receipt.relative_path
    (shard / "routing.parquet").write_bytes(
        (shard / "routing.parquet").read_bytes() + b"tamper"
    )
    layer_keys, expert_keys, routed_top_k = _universe_of(inspection)

    with pytest.raises(RoutingShardError) as caught:
        _query(workspace, receipt.run_key, layer_keys, expert_keys, routed_top_k)
    assert caught.value.stage == "reopen"


def test_universe_mismatch_fails_as_source(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    result, _, inspection = _run_result("legacy", token_count=1)
    receipt = storage.append_routing_shard(workspace, result)
    layer_keys, expert_keys, routed_top_k = _universe_of(inspection)

    with pytest.raises(ValueError) as caught:
        _query(
            workspace,
            receipt.run_key,
            ("layer-outside-universe",),
            (expert_keys[0],),
            routed_top_k,
        )
    assert isinstance(caught.value.__cause__, ValueError)


def test_overlapping_identities_across_shards_conflict(tmp_path: Path, monkeypatch) -> None:
    workspace = _workspace(tmp_path)
    first, _, inspection = _run_result("legacy", token_count=1)
    first_receipt = storage.append_routing_shard(workspace, first)
    second = _rekey_result(first, run_key=first_receipt.run_key, offset=11)
    storage.append_routing_shard(workspace, second)
    layer_keys, expert_keys, routed_top_k = _universe_of(inspection)

    overlapping = [
        (frozenset({"shared-token"}), frozenset({("l", "e", 0)})),
        (frozenset({"shared-token"}), frozenset({("l", "e", 1)})),
    ]

    def fake_validate(shard, run_key, duckdb, connection, layers, experts, top_k):
        return overlapping.pop(0)

    monkeypatch.setattr(storage, "_validate_routing_load_source", fake_validate)
    with pytest.raises(RoutingShardError) as caught:
        _query(workspace, first_receipt.run_key, layer_keys, expert_keys, routed_top_k)
    assert caught.value.stage == "conflict"


def test_source_validator_failures_map_to_query_and_source_carriers(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = _workspace(tmp_path)
    result, _, inspection = _run_result("legacy", token_count=1)
    receipt = storage.append_routing_shard(workspace, result)
    layer_keys, expert_keys, routed_top_k = _universe_of(inspection)

    def raise_oserror(*args, **kwargs):
        raise OSError("reader failure")

    monkeypatch.setattr(storage, "_validate_routing_load_source", raise_oserror)
    with pytest.raises(RoutingRunQueryError) as query_caught:
        _query(workspace, receipt.run_key, layer_keys, expert_keys, routed_top_k)
    assert isinstance(query_caught.value.__cause__, OSError)

    def raise_runtime_error(*args, **kwargs):
        raise RuntimeError("unexpected validator failure")

    monkeypatch.setattr(storage, "_validate_routing_load_source", raise_runtime_error)
    with pytest.raises(ValueError) as source_caught:
        _query(workspace, receipt.run_key, layer_keys, expert_keys, routed_top_k)
    assert isinstance(source_caught.value.__cause__, RuntimeError)


class _FailingGroupConnection:
    """Counts match the manifest; grouped reads fail; nothing else succeeds."""

    def __init__(self, token_count: int, routing_count: int) -> None:
        self.token_count = token_count
        self.routing_count = routing_count
        self.close_calls = 0

    def execute(self, sql: str, parameters: list[str]):
        if "GROUP BY layer_key" in sql:
            raise OSError("group read failure")
        if parameters[0].endswith("tokens.parquet"):
            return _Row((self.token_count,))
        return _Row((self.routing_count,))

    def close(self) -> None:
        self.close_calls += 1


class _Row:
    def __init__(self, values: tuple) -> None:
        self.values = values

    def fetchone(self) -> tuple:
        return self.values

    def fetchall(self) -> list:
        return []


def test_group_read_failure_carries_query_error(tmp_path: Path, monkeypatch) -> None:
    workspace = _workspace(tmp_path)
    result, _, inspection = _run_result("legacy", token_count=1)
    receipt = storage.append_routing_shard(workspace, result)
    layer_keys, expert_keys, routed_top_k = _universe_of(inspection)

    def fake_validate(shard, run_key, duckdb, connection, layers, experts, top_k):
        return frozenset({event.token_key for event in result.token_events}), frozenset(
            (event.token_key, event.layer_key, event.rank) for event in result.routing_events
        )

    monkeypatch.setattr(storage, "_validate_routing_load_source", fake_validate)
    connection = _FailingGroupConnection(
        len(result.token_events), len(result.routing_events)
    )

    with pytest.raises(RoutingRunQueryError) as caught:
        _query(
            workspace,
            receipt.run_key,
            layer_keys,
            expert_keys,
            routed_top_k,
            duckdb=duckdb,
            connection=connection,
        )
    assert isinstance(caught.value.__cause__, OSError)
    # The seam never closes the injected connection; the caller's finally
    # closes it exactly once even on failure.
    assert connection.close_calls == 1


# ---------------------------------------------------------------------------
# Ports


def test_store_adapter_satisfies_reader_protocol_and_delegates(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    result, _, inspection = _run_result("legacy", token_count=2)
    receipt = storage.append_routing_shard(workspace, result)
    layer_keys, expert_keys, routed_top_k = _universe_of(inspection)
    store = DuckDBRoutingShardStore.bind(workspace)

    assert isinstance(store, RoutingRunReader)
    assert isinstance(reader_from_workspace(workspace), RoutingRunReader)

    delegated = store.query_assignments(
        run_key=receipt.run_key,
        layer_keys=layer_keys,
        expert_keys=expert_keys,
        routed_top_k=routed_top_k,
        max_routing_rows=1_000_000,
        max_source_bytes=1_000_000_000,
    )
    direct = _query(workspace, receipt.run_key, layer_keys, expert_keys, routed_top_k)
    assert delegated == direct


def test_store_adapter_dependency_failure_propagates_raw(tmp_path: Path, monkeypatch) -> None:
    workspace = _workspace(tmp_path)
    result, _, inspection = _run_result("legacy", token_count=1)
    receipt = storage.append_routing_shard(workspace, result)
    layer_keys, expert_keys, routed_top_k = _universe_of(inspection)

    def raise_dependency():
        raise RoutingShardError("dependency")

    monkeypatch.setattr(storage, "_load_duckdb", raise_dependency)
    store = DuckDBRoutingShardStore.bind(workspace)
    with pytest.raises(RoutingShardError) as caught:
        store.query_assignments(
            run_key=receipt.run_key,
            layer_keys=layer_keys,
            expert_keys=expert_keys,
            routed_top_k=routed_top_k,
            max_routing_rows=1_000_000,
            max_source_bytes=1_000_000_000,
        )
    assert caught.value.stage == "dependency"
