"""Model-free contract tests for expert-event shards (R3.2, store schema 2.0).

Expert events ride the same immutable content-addressed shards under
``experts.parquet`` with manifest budgets/checksums/tamper semantics that
mirror the routing tables. ``1.0`` shards must keep reopening unchanged.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

try:
    import duckdb
except ImportError:  # pragma: no cover - exercised only without the store extra
    duckdb = None

import moeatlas.store.routing_shards as storage
from moeatlas.events import ExpertEvent, RoutingEvent, TokenEvent, TokenPhase
from moeatlas.runtime import StructuredRoutingForwardResult
from moeatlas.store import (
    LEGACY_STORE_SCHEMA_VERSION,
    STORE_SCHEMA_VERSION,
    RoutingShardError,
    append_routing_shard,
    append_structured_shard,
    list_routing_shards,
)

_RUN_KEY = "run-expert-1"


def _token(pos: int) -> TokenEvent:
    return TokenEvent(
        run_key=_RUN_KEY,
        sequence_id="sequence-1",
        token_pos=pos,
        token_id=100 + pos,
        token_text=str(pos),
        phase=TokenPhase.PREFILL,
    )


def _layer_key(index: int) -> str:
    return "component:" + format(index + 1, "064x")


def _expert_key(layer: int, expert: int) -> str:
    return "component:" + format(1000 + layer * 16 + expert, "064x")


def _routing(token: TokenEvent, layer: int, rank: int) -> RoutingEvent:
    return RoutingEvent(
        token_key=token.token_key,
        layer_key=_layer_key(layer),
        rank=rank,
        expert_key=_expert_key(layer, rank),
        router_logit=float(rank),
        probability=None,
        weight=None,
        selected=True,
    )


def _expert(
    token: TokenEvent,
    layer: int,
    rank: int,
    *,
    contribution: float | None = None,
    metadata: dict[str, object] | None = None,
) -> ExpertEvent:
    return ExpertEvent(
        token_key=token.token_key,
        expert_key=_expert_key(layer, rank),
        input_norm=1.5,
        output_norm=2.5,
        contribution_norm=contribution,
        latency_ms=None,
        metadata={"invocation_token_count": 2} if metadata is None else metadata,
    )


def _result(
    *,
    token_count: int = 1,
    experts: bool = True,
    run_key: str = _RUN_KEY,
) -> StructuredRoutingForwardResult:
    tokens = tuple(
        TokenEvent(
            run_key=run_key,
            sequence_id="sequence-1",
            token_pos=pos,
            token_id=100 + pos,
            token_text=str(pos),
            phase=TokenPhase.PREFILL,
        )
        for pos in range(token_count)
    )
    routing = []
    for layer in range(2):
        for token in tokens:
            for rank in range(2):
                routing.append(_routing(token, layer, rank))
    expert_events = []
    if experts:
        for token in tokens:
            for layer in range(2):
                for rank in range(2):
                    expert_events.append(
                        _expert(token, layer, rank, contribution=0.5 * (rank + 1))
                    )
    return StructuredRoutingForwardResult(
        output=object(),
        token_events=tokens,
        routing_events=tuple(routing),
        expert_events=tuple(expert_events),
    )


def _workspace(tmp_path: Path) -> Path:
    path = tmp_path / "workspace"
    path.mkdir()
    return path


def _manifest(receipt, workspace: Path) -> dict[str, object]:
    return json.loads((workspace / receipt.relative_path / "manifest.json").read_text())


def _rewrite_manifest(receipt, workspace: Path, payload: object) -> None:
    (workspace / receipt.relative_path / "manifest.json").write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    )


def _refresh_manifest_file(receipt, workspace: Path, filename: str) -> None:
    manifest = _manifest(receipt, workspace)
    path = workspace / receipt.relative_path / filename
    manifest["files"][filename] = {
        "name": filename,
        "bytes": path.stat().st_size,
        "sha256": f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}",
    }
    _rewrite_manifest(receipt, workspace, manifest)


@pytest.fixture(autouse=True)
def _store_extra_required(request: pytest.FixtureRequest) -> None:
    if duckdb is None and request.node.name != "test_dependency_is_lazy_without_duckdb":
        pytest.skip("duckdb store extra is unavailable")


def test_v2_write_read_round_trip_and_exact_manifest(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    result = _result()
    receipt = append_structured_shard(workspace, result)
    assert receipt.schema_version == STORE_SCHEMA_VERSION == "2.0"
    assert receipt.created is True
    shard = workspace / receipt.relative_path
    assert {path.name for path in shard.iterdir()} == {
        "manifest.json",
        "tokens.parquet",
        "routing.parquet",
        "experts.parquet",
    }
    if os.name == "posix":
        assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in shard.iterdir())
    manifest = _manifest(receipt, workspace)
    assert manifest["store_schema_version"] == STORE_SCHEMA_VERSION
    assert manifest["expert_count"] == len(result.expert_events)
    assert set(manifest["files"]) == {
        "tokens.parquet",
        "routing.parquet",
        "experts.parquet",
    }
    reopened = list_routing_shards(workspace, run_key=receipt.run_key)
    assert [item.shard_key for item in reopened] == [receipt.shard_key]
    assert reopened[0].created is False


def test_experts_parquet_physical_schema_and_values(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    result = _result()
    receipt = append_structured_shard(workspace, result)
    connection = duckdb.connect(database=":memory:")
    try:
        schema = connection.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)",
            [str(workspace / receipt.relative_path / "experts.parquet")],
        ).fetchall()
        rows = connection.execute(
            "SELECT * FROM read_parquet(?) ORDER BY event_index",
            [str(workspace / receipt.relative_path / "experts.parquet")],
        ).fetchall()
    finally:
        connection.close()
    assert [(row[0], row[1]) for row in schema] == [
        ("store_schema_version", "VARCHAR"),
        ("shard_key", "VARCHAR"),
        ("event_index", "BIGINT"),
        ("schema_version", "VARCHAR"),
        ("event_type", "VARCHAR"),
        ("token_key", "VARCHAR"),
        ("expert_key", "VARCHAR"),
        ("input_norm", "DOUBLE"),
        ("output_norm", "DOUBLE"),
        ("contribution_norm", "DOUBLE"),
        ("latency_ms", "DOUBLE"),
        ("metadata", "VARCHAR"),
    ]
    expected_metadata = json.dumps(
        {"invocation_token_count": 2}, ensure_ascii=False, sort_keys=True
    )
    assert tuple(rows) == tuple(
        (
            STORE_SCHEMA_VERSION,
            receipt.shard_key,
            index,
            event.schema_version,
            event.event_type,
            event.token_key,
            event.expert_key,
            event.input_norm,
            event.output_norm,
            event.contribution_norm,
            event.latency_ms,
            expected_metadata,
        )
        for index, event in enumerate(result.expert_events)
    )


def test_plain_routing_results_write_empty_expert_tables(tmp_path: Path) -> None:
    from .fixtures.helpers import plain_routing_result

    workspace = _workspace(tmp_path)
    receipt = append_routing_shard(workspace, plain_routing_result())
    manifest = _manifest(receipt, workspace)
    assert manifest["store_schema_version"] == STORE_SCHEMA_VERSION
    assert manifest["expert_count"] == 0
    assert (workspace / receipt.relative_path / "experts.parquet").is_file()


def test_idempotent_rewrite_and_conflict_rejection_mirror_routing_rows(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    result = _result()
    first = append_structured_shard(workspace, result)
    again = append_structured_shard(workspace, _result())
    assert again.created is False
    assert again.shard_key == first.shard_key
    # Same token/routing identities but different expert evidence produce a
    # different content address while overlapping every identity: conflict.
    tampered_expert = result.expert_events[0].model_copy(update={"output_norm": 9.0})
    changed = StructuredRoutingForwardResult(
        output=result.output,
        token_events=result.token_events,
        routing_events=result.routing_events,
        expert_events=(tampered_expert, *result.expert_events[1:]),
    )
    with pytest.raises(RoutingShardError) as conflict:
        append_structured_shard(workspace, changed)
    assert conflict.value.stage == "conflict"


@pytest.mark.parametrize(
    "bad",
    [
        "unknown_token",
        "duplicate_link",
        "no_evidence",
        "negative_norm",
        "not_tuple",
        "wrong_type",
    ],
)
def test_event_validation_failures_never_touch_the_workspace(
    tmp_path: Path, bad: str
) -> None:
    import pydantic

    workspace = _workspace(tmp_path)
    before = sorted(path.name for path in workspace.iterdir())
    result = _result()
    if bad == "unknown_token":
        stranger = _token(9)
        experts = (_expert(stranger, 0, 0),)
        with pytest.raises(ValueError, match="reference a supplied token"):
            StructuredRoutingForwardResult(
                output=result.output,
                token_events=result.token_events,
                routing_events=result.routing_events,
                expert_events=experts,
            )
    elif bad == "duplicate_link":
        doubled = result.expert_events + result.expert_events[:1]
        with pytest.raises(ValueError, match="unique"):
            StructuredRoutingForwardResult(
                output=result.output,
                token_events=result.token_events,
                routing_events=result.routing_events,
                expert_events=doubled,
            )
    elif bad == "no_evidence":
        with pytest.raises(pydantic.ValidationError, match="measurement or non-empty"):
            ExpertEvent(token_key=result.token_events[0].token_key, expert_key=_expert_key(0, 0))
    elif bad == "negative_norm":
        with pytest.raises(pydantic.ValidationError):
            ExpertEvent(
                token_key=result.token_events[0].token_key,
                expert_key=_expert_key(0, 0),
                input_norm=-1.0,
            )
    elif bad == "not_tuple":
        with pytest.raises(TypeError, match="exact tuple"):
            StructuredRoutingForwardResult(
                output=result.output,
                token_events=result.token_events,
                routing_events=result.routing_events,
                expert_events=list(result.expert_events),  # type: ignore[arg-type]
            )
    else:
        with pytest.raises(TypeError, match="exact ExpertEvent"):
            StructuredRoutingForwardResult(
                output=result.output,
                token_events=result.token_events,
                routing_events=result.routing_events,
                expert_events=(object(),),  # type: ignore[list-item]
            )
    assert sorted(path.name for path in workspace.iterdir()) == before


@pytest.mark.parametrize("budget", [0, -1, True, 1.5, "8"])
def test_strict_expert_budget_argument_validation(tmp_path: Path, budget: object) -> None:
    workspace = _workspace(tmp_path)
    before = sorted(path.name for path in workspace.iterdir())
    with pytest.raises((TypeError, ValueError)):
        append_structured_shard(workspace, _result(), max_expert_events=budget)  # type: ignore[arg-type]
    assert sorted(path.name for path in workspace.iterdir()) == before


def test_expert_budget_exhaustion_is_a_preflight_value_error(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    result = _result(token_count=4, experts=True)
    with pytest.raises(ValueError, match="budget"):
        append_structured_shard(workspace, result, max_expert_events=8)
    assert list(workspace.iterdir()) == []


def test_experts_parquet_tampering_is_rejected(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    receipt = append_structured_shard(workspace, _result())
    path = workspace / receipt.relative_path / "experts.parquet"
    original = path.read_bytes()
    path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(RoutingShardError) as caught:
        list_routing_shards(workspace, run_key=receipt.run_key)
    assert caught.value.stage == "reopen"


def _rewrite_parquet(path: Path, columns, rows) -> None:
    connection = duckdb.connect(database=":memory:")
    temporary = path.with_name(".rewrite.parquet")
    try:
        column_sql = ", ".join(f'"{name}" {column_type}' for name, column_type in columns)
        connection.execute(f"CREATE TABLE altered ({column_sql})")
        if rows:
            placeholders = ", ".join("?" for _ in columns)
            connection.executemany(f"INSERT INTO altered VALUES ({placeholders})", rows)
        connection.table("altered").write_parquet(
            str(temporary), compression="zstd", overwrite=False
        )
    finally:
        connection.close()
    if os.name == "posix":
        os.chmod(temporary, 0o600)
    os.replace(temporary, path)


@pytest.mark.parametrize(
    "tamper",
    [
        "wrong_column",
        "wrong_type",
        "row_count",
        "noncontiguous_index",
        "store_schema_version",
        "shard_key",
        "event_type",
        "semantic_contribution_change",
        "metadata_not_canonical",
        "nonfinite_norm_is_unrepresentable_so_semantic_change",
    ],
)
def test_valid_checksum_expert_parquet_tampering_reaches_reopen_validation(
    tmp_path: Path, tamper: str
) -> None:
    workspace = _workspace(tmp_path)
    receipt = append_structured_shard(workspace, _result())
    shard = workspace / receipt.relative_path
    connection = duckdb.connect(database=":memory:")
    try:
        rows = tuple(
            connection.execute(
                "SELECT * FROM read_parquet(?) ORDER BY event_index",
                [str(shard / "experts.parquet")],
            ).fetchall()
        )
    finally:
        connection.close()
    columns = storage._EXPERT_COLUMNS
    mutable = [list(row) for row in rows]
    if tamper == "wrong_column":
        altered = list(columns)
        altered[6] = ("expert_key_wrong", altered[6][1])
        columns = tuple(altered)
    elif tamper == "wrong_type":
        altered = list(columns)
        altered[7] = ("input_norm", "VARCHAR")
        columns = tuple(altered)
        mutable[0][7] = str(mutable[0][7])
    elif tamper == "row_count":
        mutable = []
    elif tamper == "noncontiguous_index":
        mutable[0][2] = 7
    elif tamper == "store_schema_version":
        mutable[0][0] = "3.0"
    elif tamper == "shard_key":
        mutable[0][1] = "shard:" + "0" * 64
    elif tamper == "event_type":
        mutable[0][4] = "routing"
    elif tamper == "metadata_not_canonical":
        mutable[0][11] = '{"z": 1, "a": 2}'
    else:
        mutable[0][9] = float(mutable[0][9]) + 1.0
    _rewrite_parquet(shard / "experts.parquet", columns, tuple(tuple(row) for row in mutable))
    _refresh_manifest_file(receipt, workspace, "experts.parquet")
    with pytest.raises(RoutingShardError) as caught:
        list_routing_shards(workspace, run_key=receipt.run_key)
    assert caught.value.stage == "reopen"


def test_manifest_expert_budget_tampering_is_rejected(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    receipt = append_structured_shard(workspace, _result())
    payload = _manifest(receipt, workspace)
    payload["expert_count"] = int(payload["expert_count"]) + 1
    _rewrite_manifest(receipt, workspace, payload)
    with pytest.raises(RoutingShardError) as caught:
        list_routing_shards(workspace, run_key=receipt.run_key)
    assert caught.value.stage == "reopen"


def test_crash_injection_cleans_owned_staging(tmp_path: Path, monkeypatch) -> None:
    workspace = _workspace(tmp_path)
    original = storage._write_parquets

    def fail_after_write(*args: object, **kwargs: object) -> object:
        original(*args, **kwargs)
        raise OSError("injected post-parquet failure")

    monkeypatch.setattr(storage, "_write_parquets", fail_after_write)
    with pytest.raises(RoutingShardError) as caught:
        append_structured_shard(workspace, _result())
    assert caught.value.stage == "write"
    assert not any(path.name.startswith(".staging-") for path in workspace.rglob("*"))
    assert not any(path.name.startswith("shard-") for path in workspace.rglob("*"))


def test_legacy_v1_shards_reopen_unchanged(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    result = _result()
    receipt = append_structured_shard(workspace, result)
    shard = workspace / receipt.relative_path
    (shard / "experts.parquet").unlink()
    # The legacy content address digests the legacy store version.
    _, _, legacy_semantic = storage._semantic_rows(
        result.token_events,
        result.routing_events,
        store_token_text=False,
        expert_events=(),
        store_version=LEGACY_STORE_SCHEMA_VERSION,
    )
    legacy_shard_key = storage._shard_key(legacy_semantic)
    legacy_hex = legacy_shard_key.removeprefix("shard:")
    # A genuine historical shard carries the legacy version and its own
    # content address inside every row.
    for filename, columns in (
        ("tokens.parquet", storage._TOKEN_COLUMNS),
        ("routing.parquet", storage._ROUTING_COLUMNS),
    ):
        connection = duckdb.connect(database=":memory:")
        try:
            rows = connection.execute(
                "SELECT * FROM read_parquet(?) ORDER BY event_index",
                [str(shard / filename)],
            ).fetchall()
        finally:
            connection.close()
        downgraded = tuple(
            (LEGACY_STORE_SCHEMA_VERSION, legacy_shard_key) + row[2:] for row in rows
        )
        _rewrite_parquet(shard / filename, columns, downgraded)
    payload = _manifest(receipt, workspace)
    del payload["expert_count"]
    payload["store_schema_version"] = LEGACY_STORE_SCHEMA_VERSION
    del payload["files"]["experts.parquet"]
    new_relative = receipt.relative_path.replace(
        receipt.shard_key.removeprefix("shard:"), legacy_hex
    )
    shard.rename(workspace / new_relative)
    payload["shard_key"] = legacy_shard_key
    receipt = type(receipt)(
        schema_version=receipt.schema_version,
        shard_key=legacy_shard_key,
        run_key=receipt.run_key,
        relative_path=new_relative,
        token_count=receipt.token_count,
        routing_count=receipt.routing_count,
        token_text_stored=receipt.token_text_stored,
        created=False,
    )
    _rewrite_manifest(receipt, workspace, payload)
    _refresh_manifest_file(receipt, workspace, "tokens.parquet")
    _refresh_manifest_file(receipt, workspace, "routing.parquet")
    reopened = list_routing_shards(workspace, run_key=receipt.run_key)
    assert [item.shard_key for item in reopened] == [receipt.shard_key]
    assert reopened[0].schema_version == LEGACY_STORE_SCHEMA_VERSION


def test_downgraded_v1_and_fresh_v2_runs_coexist(tmp_path: Path) -> None:
    from .fixtures.helpers import plain_routing_result

    workspace = _workspace(tmp_path)
    plain_receipt = append_routing_shard(
        workspace, plain_routing_result(run_key="run-plain")
    )
    structured = _result(run_key="run-v2")
    structured_receipt = append_structured_shard(workspace, structured)
    # The plain lane writes current-version shards with empty expert tables.
    listed_plain = list_routing_shards(workspace, run_key="run-plain")
    assert [item.schema_version for item in listed_plain] == [STORE_SCHEMA_VERSION]
    manifest = _manifest(plain_receipt, workspace)
    assert manifest["expert_count"] == 0
    listed_v2 = list_routing_shards(workspace, run_key="run-v2")
    assert [item.shard_key for item in listed_v2] == [structured_receipt.shard_key]


def test_dependency_is_lazy_without_duckdb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    import builtins

    original_import = builtins.__import__

    def blocked_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "duckdb":
            raise ModuleNotFoundError("duckdb intentionally blocked")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(RoutingShardError) as caught:
        append_structured_shard(workspace, _result())
    assert caught.value.stage == "dependency"
    assert not (workspace / "routing").exists()


def test_ast_guards_no_model_stack_or_raw_row_retention() -> None:
    source = Path("src/moeatlas/store/routing_shards.py").read_text()
    tree = ast.parse(source)
    forbidden_modules = {
        "torch",
        "transformers",
        "accelerate",
        "safetensors",
        "numpy",
        "pyarrow",
        "pandas",
        "polars",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] not in forbidden_modules for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in forbidden_modules
