from __future__ import annotations

import ast
import builtins
import json
import os
import shutil
import stat
from dataclasses import FrozenInstanceError, fields, is_dataclass
from pathlib import Path

import pytest

import moeatlas.store.routing_inventory as storage
from moeatlas.events import RoutingEvent, TokenEvent
from moeatlas.store import (
    ROUTING_RUN_INVENTORY_SCHEMA_VERSION,
    STORE_SCHEMA_VERSION,
    RoutingRunInventory,
    RoutingRunInventoryError,
    RoutingRunSummary,
    RoutingShardError,
    append_routing_shard,
    list_routing_runs,
)

from .test_runtime_routing_forward import _run


def _workspace(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "inventory workspace — quotes ' and spaces"
    path.mkdir()
    return path


def _append(tmp_path: Path, layout: str = "legacy", *, stored: bool = False):
    result, _model, inspection = _run(layout, token_count=1)
    workspace = _workspace(tmp_path)
    receipt = append_routing_shard(workspace, result, store_token_text=stored)
    return workspace, receipt, inspection


def _retag_result(result: object, *, run_key: str, token_prefix: str):
    tokens_list: list[TokenEvent] = []
    for index, event in enumerate(result.token_events):
        payload = event.model_dump(mode="json")
        payload["run_key"] = run_key
        payload["token_id"] = event.token_id + index + (ord(token_prefix) - ord("a"))
        payload.pop("token_key", None)
        tokens_list.append(TokenEvent.model_validate(payload))
    tokens = tuple(tokens_list)
    routes = tuple(
        RoutingEvent.model_validate(
            {
                **event.model_dump(mode="json"),
                "token_key": tokens[index // 4].token_key,
            }
        )
        for index, event in enumerate(result.routing_events)
    )
    from moeatlas.runtime import RoutingForwardResult

    return RoutingForwardResult(result.output, tokens, routes)


def test_public_inventory_surface_is_exact_and_serializable() -> None:
    assert is_dataclass(RoutingRunSummary)
    assert is_dataclass(RoutingRunInventory)
    assert tuple(field.name for field in fields(RoutingRunSummary)) == (
        "run_key",
        "shard_keys",
        "shard_count",
        "token_count",
        "routing_count",
        "source_bytes",
        "token_text_policy",
    )
    assert tuple(field.name for field in fields(RoutingRunInventory)) == (
        "schema_version",
        "manifest_type",
        "store_schema_version",
        "event_schema_version",
        "run_count",
        "shard_count",
        "token_count",
        "routing_count",
        "source_bytes",
        "runs",
    )
    assert getattr(RoutingRunSummary, "__slots__")
    assert getattr(RoutingRunInventory, "__slots__")
    summary = RoutingRunSummary(
        "run-1",
        ("shard:" + "a" * 64,),
        1,
        1,
        1,
        1,
        "redacted",
    )
    inventory = RoutingRunInventory(
        "1.0",
        "mixtral_routing_run_inventory",
        STORE_SCHEMA_VERSION,
        "1.0",
        1,
        1,
        1,
        1,
        1,
        (summary,),
    )
    assert inventory.schema_version == ROUTING_RUN_INVENTORY_SCHEMA_VERSION
    assert json.loads(inventory.to_json()) == inventory.to_dict()
    assert "\n" not in inventory.to_json()
    with pytest.raises((FrozenInstanceError, AttributeError)):
        summary.run_key = "run-2"  # type: ignore[misc]
    with pytest.raises((FrozenInstanceError, AttributeError)):
        inventory.runs = ()  # type: ignore[misc]


def test_inventory_constructor_invariant_matrix() -> None:
    key = "shard:" + "a" * 64
    summary = RoutingRunSummary("run-1", (key,), 1, 1, 1, 1, "redacted")
    valid = {
        "schema_version": "1.0",
        "manifest_type": "mixtral_routing_run_inventory",
        "store_schema_version": "1.0",
        "event_schema_version": "1.0",
        "run_count": 1,
        "shard_count": 1,
        "token_count": 1,
        "routing_count": 1,
        "source_bytes": 1,
        "runs": (summary,),
    }
    for field, value in (
        ("schema_version", "9.9"),
        ("manifest_type", "other"),
        ("store_schema_version", "3.0"),
        ("event_schema_version", "9.9"),
        ("run_count", True),
        ("shard_count", -1),
        ("token_count", 1.0),
        ("routing_count", 2),
        ("source_bytes", -1),
        ("runs", [summary]),
    ):
        payload = dict(valid)
        if field == "routing_count":
            payload[field] = value
            payload["routing_count"] = value
        else:
            payload[field] = value
        with pytest.raises((TypeError, ValueError)):
            RoutingRunInventory(**payload)
    with pytest.raises(ValueError):
        RoutingRunSummary("run-1", (key, key), 2, 1, 1, 1, "redacted")
    with pytest.raises(ValueError):
        RoutingRunSummary("run-1", (key,), 1, 1, 1, 1, "invalid")


@pytest.mark.parametrize("layout", ["legacy", "packed"])
def test_inventory_round_trip_layouts_and_exact_totals(tmp_path: Path, layout: str) -> None:
    workspace, receipt, _inspection = _append(tmp_path, layout)
    inventory = list_routing_runs(
        workspace,
        max_runs=1,
        max_shards=1,
        max_event_rows=100,
        max_source_bytes=10_000_000,
    )
    assert inventory.run_count == 1
    assert inventory.shard_count == 1
    assert inventory.token_count == receipt.token_count
    assert inventory.routing_count == receipt.routing_count
    assert inventory.runs[0].run_key == receipt.run_key
    assert inventory.runs[0].shard_keys == (receipt.shard_key,)
    assert inventory.runs[0].source_bytes == inventory.source_bytes
    assert inventory.runs[0].token_text_policy == "redacted"


def test_multi_run_multi_shard_order_bytes_and_mixed_redaction(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    base, _model, _inspection = _run("legacy", token_count=1)
    first = append_routing_shard(workspace, base, store_token_text=False)
    second_result = _retag_result(base, run_key=first.run_key, token_prefix="b")
    second = append_routing_shard(workspace, second_result, store_token_text=True)
    other_result = _retag_result(base, run_key="run-2", token_prefix="c")
    other = append_routing_shard(workspace, other_result, store_token_text=True)
    inventory = list_routing_runs(
        workspace, max_runs=2, max_shards=3, max_event_rows=1000, max_source_bytes=10_000_000
    )
    assert tuple(item.run_key for item in inventory.runs) == ("run-1", "run-2")
    run_one, run_two = inventory.runs
    assert run_one.shard_keys == tuple(sorted((first.shard_key, second.shard_key)))
    assert run_one.shard_count == 2
    assert run_one.token_text_policy == "mixed"
    assert run_two.shard_keys == (other.shard_key,)
    assert inventory.shard_count == 3
    assert inventory.source_bytes == sum(item.source_bytes for item in inventory.runs)
    expected_bytes = 0
    for item in (first, second, other):
        shard = workspace / item.relative_path
        expected_bytes += sum(
            (shard / name).stat().st_size
            for name in (
                "manifest.json",
                "tokens.parquet",
                "routing.parquet",
                "experts.parquet",
            )
        )
    assert inventory.source_bytes == expected_bytes


def test_absent_tree_is_canonical_empty_and_does_not_import_duckdb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    original_import = builtins.__import__

    def blocked(name: str, *args: object, **kwargs: object) -> object:
        if name == "duckdb":
            raise AssertionError("absent inventory must not import duckdb")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    before = tuple(sorted(item.relative_to(workspace).as_posix() for item in workspace.rglob("*")))
    inventory = list_routing_runs(
        workspace, max_runs=1, max_shards=1, max_event_rows=1, max_source_bytes=1
    )
    after = tuple(sorted(item.relative_to(workspace).as_posix() for item in workspace.rglob("*")))
    assert inventory.to_dict()["runs"] == []
    assert before == after == ()


@pytest.mark.parametrize("name", ["max_runs", "max_shards", "max_event_rows", "max_source_bytes"])
@pytest.mark.parametrize("value", [0, -1, True, 1.0, "1"])
def test_inventory_budgets_are_strict_and_preflight_before_workspace(
    tmp_path: Path, name: str, value: object
) -> None:
    workspace = tmp_path / "missing"
    kwargs = dict(max_runs=1, max_shards=1, max_event_rows=1, max_source_bytes=1)
    kwargs[name] = value  # type: ignore[assignment]
    with pytest.raises((TypeError, ValueError)):
        list_routing_runs(workspace, **kwargs)
    assert not workspace.exists()


def test_inventory_budgets_cover_declared_source_and_events(tmp_path: Path) -> None:
    workspace, _receipt, _inspection = _append(tmp_path)
    with pytest.raises(RoutingRunInventoryError) as caught:
        list_routing_runs(
            workspace, max_runs=1, max_shards=1, max_event_rows=1, max_source_bytes=10_000_000
        )
    assert caught.value.stage == "budget"
    with pytest.raises(RoutingRunInventoryError) as caught:
        list_routing_runs(
            workspace, max_runs=1, max_shards=1, max_event_rows=100, max_source_bytes=1
        )
    assert caught.value.stage == "budget"


def test_actual_row_budget_precedes_reconstruct_and_declared_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, _receipt, _inspection = _append(tmp_path)
    reconstruct_called = False

    def forbidden(*_args: object, **_kwargs: object) -> object:
        nonlocal reconstruct_called
        reconstruct_called = True
        raise AssertionError("deep reopen must follow actual budget")

    monkeypatch.setattr(storage, "_inventory_count_rows", lambda *_args: 10_000)
    monkeypatch.setattr(storage, "_reconstruct_shard_with_connection", forbidden)
    with pytest.raises(RoutingRunInventoryError) as caught:
        list_routing_runs(
            workspace, max_runs=1, max_shards=1, max_event_rows=1, max_source_bytes=10_000_000
        )
    assert caught.value.stage == "budget"
    assert reconstruct_called is False


def test_all_inventory_budget_boundaries_are_exact(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    first_result, _model, _inspection = _run("legacy", token_count=1)
    first = append_routing_shard(workspace, first_result)
    second_result = _retag_result(first_result, run_key="run-2", token_prefix="b")
    second = append_routing_shard(workspace, second_result)
    unconstrained = list_routing_runs(
        workspace, max_runs=10, max_shards=10, max_event_rows=1000, max_source_bytes=10_000_000
    )
    totals = {
        "max_runs": unconstrained.run_count,
        "max_shards": unconstrained.shard_count,
        "max_event_rows": unconstrained.routing_count + unconstrained.token_count,
        "max_source_bytes": unconstrained.source_bytes,
    }
    assert totals["max_runs"] == 2
    assert totals["max_shards"] == 2
    assert first.run_key != second.run_key
    exact = dict(
        max_runs=2,
        max_shards=2,
        max_event_rows=totals["max_event_rows"],
        max_source_bytes=totals["max_source_bytes"],
    )
    assert list_routing_runs(workspace, **exact).run_count == 2
    for name, value in totals.items():
        below = dict(exact)
        below[name] = value - 1
        if below[name] <= 0:
            with pytest.raises((TypeError, ValueError)):
                list_routing_runs(workspace, **below)
        else:
            with pytest.raises(RoutingRunInventoryError) as caught:
                list_routing_runs(workspace, **below)
            assert caught.value.stage == "budget"


def test_empty_and_staging_only_candidates_count_max_runs(tmp_path: Path) -> None:
    workspace, _receipt, _inspection = _append(tmp_path)
    version = workspace / "routing" / "v1"
    empty = version / ("run-" + "f" * 64)
    empty.mkdir()
    (empty / ".staging-crash").mkdir()
    with pytest.raises(RoutingRunInventoryError) as caught:
        list_routing_runs(
            workspace, max_runs=1, max_shards=10, max_event_rows=1000, max_source_bytes=10_000_000
        )
    assert caught.value.stage == "budget"
    inventory = list_routing_runs(
        workspace, max_runs=2, max_shards=10, max_event_rows=1000, max_source_bytes=10_000_000
    )
    assert inventory.run_count == 1


def test_routing_root_and_version_entries_are_exact(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    routing = workspace / "routing"
    routing.mkdir()
    (routing / "rogue").mkdir()
    with pytest.raises(RoutingRunInventoryError) as caught:
        list_routing_runs(
            workspace, max_runs=1, max_shards=1, max_event_rows=1, max_source_bytes=1
        )
    assert caught.value.stage == "index"
    (routing / "rogue").rmdir()
    (routing / "v1").mkdir()
    (routing / ".staging-version").mkdir()
    with pytest.raises(RoutingRunInventoryError) as caught:
        list_routing_runs(
            workspace, max_runs=1, max_shards=1, max_event_rows=1, max_source_bytes=1
        )
    assert caught.value.stage == "index"


def test_empty_routing_root_without_v1_is_canonical_empty_and_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    routing = workspace / "routing"
    routing.mkdir()
    before = tuple(sorted(item.relative_to(workspace).as_posix() for item in workspace.rglob("*")))
    monkeypatch.setattr(
        storage, "_load_duckdb", lambda: (_ for _ in ()).throw(AssertionError("lazy"))
    )
    inventory = list_routing_runs(
        workspace, max_runs=1, max_shards=1, max_event_rows=1, max_source_bytes=1
    )
    after = tuple(sorted(item.relative_to(workspace).as_posix() for item in workspace.rglob("*")))
    assert inventory.run_count == 0
    assert before == after == ("routing",)


def test_malformed_run_and_managed_links_are_index_errors(tmp_path: Path) -> None:
    workspace, _receipt, _inspection = _append(tmp_path)
    version = workspace / "routing" / "v1"
    (version / "not-a-run").mkdir()
    with pytest.raises(RoutingRunInventoryError) as caught:
        list_routing_runs(
            workspace, max_runs=10, max_shards=10, max_event_rows=1000, max_source_bytes=10_000_000
        )
    assert caught.value.stage == "index"

    workspace2, receipt, _inspection = _append(tmp_path / "second")
    shard = workspace2 / receipt.relative_path
    target = shard / "routing.parquet"
    backup = shard / "routing.parquet.bak"
    target.rename(backup)
    try:
        target.symlink_to(backup)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    with pytest.raises(RoutingShardError) as caught:
        list_routing_runs(
            workspace2, max_runs=1, max_shards=1, max_event_rows=1000, max_source_bytes=10_000_000
        )
    assert caught.value.stage == "reopen"


@pytest.mark.parametrize("target", ["routing", "version", "run"])
def test_managed_root_version_run_symlinks_are_rejected(tmp_path: Path, target: str) -> None:
    workspace, receipt, _inspection = _append(tmp_path)
    routing = workspace / "routing"
    version = routing / "v1"
    run_parent = workspace / receipt.relative_path.rsplit("/", 1)[0]
    selected = {"routing": routing, "version": version, "run": run_parent}[target]
    backup = selected.with_name(selected.name + ".backup")
    selected.rename(backup)
    try:
        selected.symlink_to(backup, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    with pytest.raises(RoutingRunInventoryError) as caught:
        list_routing_runs(
            workspace, max_runs=10, max_shards=10, max_event_rows=1000, max_source_bytes=10_000_000
        )
    assert caught.value.stage == "index"


@pytest.mark.parametrize("target", ["routing", "version", "run"])
def test_managed_root_version_run_nondirs_are_rejected(tmp_path: Path, target: str) -> None:
    workspace, receipt, _inspection = _append(tmp_path)
    routing = workspace / "routing"
    version = routing / "v1"
    run_parent = workspace / receipt.relative_path.rsplit("/", 1)[0]
    selected = {"routing": routing, "version": version, "run": run_parent}[target]
    backup = selected.with_name(selected.name + ".backup")
    selected.rename(backup)
    selected.write_text("not a directory")
    with pytest.raises(RoutingRunInventoryError) as caught:
        list_routing_runs(
            workspace, max_runs=10, max_shards=10, max_event_rows=1000, max_source_bytes=10_000_000
        )
    assert caught.value.stage == "index"


def test_inventory_reuses_authoritative_reopen_and_conflict_stages(tmp_path: Path) -> None:
    workspace, receipt, _inspection = _append(tmp_path)
    manifest_path = workspace / receipt.relative_path / "manifest.json"
    payload = json.loads(manifest_path.read_text())
    payload["routing_count"] += 1
    manifest_path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(RoutingShardError) as caught:
        list_routing_runs(
            workspace, max_runs=1, max_shards=1, max_event_rows=1000, max_source_bytes=10_000_000
        )
    assert caught.value.stage == "reopen"


def test_inventory_detects_cross_shard_identity_conflict_after_reopen(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    result, _model, _inspection = _run("legacy", token_count=1)
    first = append_routing_shard(workspace, result)
    from moeatlas.runtime import RoutingForwardResult

    changed_routes = tuple(
        RoutingEvent.model_validate(
            {
                **event.model_dump(mode="json"),
                "router_logit": event.router_logit + 0.5,
            }
        )
        for event in result.routing_events
    )
    changed = RoutingForwardResult(result.output, result.token_events, changed_routes)
    source_workspace = _workspace(tmp_path / "source")
    second = append_routing_shard(source_workspace, changed)
    destination = workspace / second.relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_workspace / second.relative_path, destination)
    assert first.run_key == second.run_key
    with pytest.raises(RoutingShardError) as caught:
        list_routing_runs(
            workspace, max_runs=1, max_shards=2, max_event_rows=1000, max_source_bytes=20_000_000
        )
    assert caught.value.stage == "conflict"


@pytest.mark.parametrize("tamper", ["checksum", "manifest-shape"])
def test_inventory_committed_corruption_uses_reopen_stage(tmp_path: Path, tamper: str) -> None:
    workspace, receipt, _inspection = _append(tmp_path)
    shard = workspace / receipt.relative_path
    if tamper == "checksum":
        path = shard / "routing.parquet"
        path.write_bytes(path.read_bytes() + b"tamper")
    else:
        path = shard / "manifest.json"
        payload = json.loads(path.read_text())
        payload["unexpected"] = True
        path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(RoutingShardError) as caught:
        list_routing_runs(
            workspace, max_runs=1, max_shards=1, max_event_rows=1000, max_source_bytes=10_000_000
        )
    assert caught.value.stage == "reopen"


def test_inventory_modes_and_policy_are_deterministic(tmp_path: Path) -> None:
    workspace, receipt, _inspection = _append(tmp_path, stored=True)
    inventory = list_routing_runs(
        workspace, max_runs=1, max_shards=1, max_event_rows=1000, max_source_bytes=10_000_000
    )
    assert inventory.runs[0].token_text_policy == "stored"
    if os.name == "posix":
        shard = workspace / receipt.relative_path
        assert stat.S_IMODE(shard.stat().st_mode) == 0o700
        assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in shard.iterdir())
    assert inventory.to_json() == inventory.to_json()


def test_inventory_connection_close_lifecycle_and_primary_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    duckdb = pytest.importorskip("duckdb")
    workspace, _receipt, _inspection = _append(tmp_path)

    class ConnectionProxy:
        def __init__(self, connection, *, execute_error=None, close_failures=0, close_error=None):
            self.connection = connection
            self.execute_error = execute_error
            self.close_failures = close_failures
            self.close_error = close_error or OSError("close failed")
            self.close_calls = 0

        def execute(self, *args, **kwargs):
            if self.execute_error is not None:
                raise self.execute_error
            return self.connection.execute(*args, **kwargs)

        def close(self):
            self.close_calls += 1
            if self.close_failures:
                self.close_failures -= 1
                raise self.close_error
            return self.connection.close()

    class DuckProxy:
        __version__ = duckdb.__version__

        def __init__(self, connection):
            self.connection = connection

        def connect(self, **kwargs):
            return self.connection

    base_connection = duckdb.connect(database=":memory:")
    proxy_connection = ConnectionProxy(base_connection)
    monkeypatch.setattr(storage, "_load_duckdb", lambda: DuckProxy(proxy_connection))
    inventory = list_routing_runs(
        workspace, max_runs=1, max_shards=1, max_event_rows=1000, max_source_bytes=10_000_000
    )
    assert inventory.run_count == 1
    assert proxy_connection.close_calls == 1

    for primary in (ValueError("query"), KeyboardInterrupt(), SystemExit(7)):
        base_connection = duckdb.connect(database=":memory:")
        proxy_connection = ConnectionProxy(base_connection, execute_error=primary, close_failures=2)
        monkeypatch.setattr(
            storage,
            "_load_duckdb",
            lambda proxy_connection=proxy_connection: DuckProxy(proxy_connection),
        )
        expected_type = RoutingShardError if isinstance(primary, ValueError) else type(primary)
        with pytest.raises(expected_type) as caught:
            list_routing_runs(
                workspace,
                max_runs=1,
                max_shards=1,
                max_event_rows=1000,
                max_source_bytes=10_000_000,
            )
        if isinstance(primary, ValueError):
            assert caught.value.stage == "reopen"
        else:
            assert caught.value is primary
        assert proxy_connection.close_calls == 2
        assert caught.value.__notes__ == ["routing run inventory cleanup failed"]

    base_connection = duckdb.connect(database=":memory:")
    proxy_connection = ConnectionProxy(base_connection, close_failures=1)
    monkeypatch.setattr(storage, "_load_duckdb", lambda: DuckProxy(proxy_connection))
    assert (
        list_routing_runs(
            workspace, max_runs=1, max_shards=1, max_event_rows=1000, max_source_bytes=10_000_000
        ).run_count
        == 1
    )
    assert proxy_connection.close_calls == 2

    base_connection = duckdb.connect(database=":memory:")
    proxy_connection = ConnectionProxy(base_connection, close_failures=2)
    monkeypatch.setattr(storage, "_load_duckdb", lambda: DuckProxy(proxy_connection))
    with pytest.raises(RoutingShardError) as caught:
        list_routing_runs(
            workspace, max_runs=1, max_shards=1, max_event_rows=1000, max_source_bytes=10_000_000
        )
    assert caught.value.stage == "reopen"
    assert proxy_connection.close_calls == 2


def test_inventory_ast_is_bounded_parameterized_and_model_free() -> None:
    source = Path(storage.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    function_names = {
        "_inventory_index",
        "_inventory_shards_for_run",
        "_inventory_count_rows",
        "list_routing_runs",
    }
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name in function_names
    }
    forbidden_names = {
        "torch",
        "transformers",
        "accelerate",
        "safetensors",
        "tokenizer",
        "webbrowser",
        "socket",
        "urllib",
        "requests",
        "catalog",
        "cache",
        "importlib",
        "mkstemp",
        "tempfile",
        "write_text",
        "write_bytes",
        "mkdir",
        "rename",
        "replace",
        "unlink",
        "rmtree",
    }
    for function in functions.values():
        for node in ast.walk(function):
            if isinstance(node, ast.Import):
                assert all(
                    alias.name.split(".", 1)[0] not in forbidden_names for alias in node.names
                )
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".", 1)[0] not in forbidden_names
            if isinstance(node, ast.Name):
                assert node.id not in forbidden_names
            if isinstance(node, ast.Attribute):
                assert node.attr not in forbidden_names
            if isinstance(node, ast.JoinedStr):
                assert not any(
                    isinstance(parent, ast.Call)
                    and isinstance(parent.func, ast.Attribute)
                    and parent.func.attr in {"execute", "executemany"}
                    for parent in ast.walk(function)
                    if node in ast.walk(parent)
                )
    count_function = functions["_inventory_count_rows"]
    execute_calls = [
        node
        for node in ast.walk(count_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "execute"
    ]
    assert len(execute_calls) == 1
    call = execute_calls[0]
    assert isinstance(call.args[0], ast.Constant)
    assert call.args[0].value == "SELECT COUNT(*) FROM read_parquet(?)"
    assert len(call.args) == 2
    assert isinstance(call.args[1], ast.List)
    assert len(call.args[1].elts) == 1
    assert isinstance(call.args[1].elts[0], ast.Call)
    assert isinstance(call.args[1].elts[0].func, ast.Name)
    assert call.args[1].elts[0].func.id == "str"
