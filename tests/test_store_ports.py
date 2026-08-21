"""Model-free tests for the model-neutral storage ports."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from moeatlas.store import (
    RoutingShardError,
    append_routing_shard,
    list_routing_runs,
    list_routing_shards,
)
from moeatlas.store.ports import (
    DuckDBRoutingShardStore,
    RoutingRunReader,
    RoutingShardAppender,
    reader_from_workspace,
)

try:
    import duckdb
except ImportError:  # pragma: no cover
    duckdb = None

ROOT = Path(__file__).resolve().parents[1]

_BUDGETS = {
    "max_runs": 10,
    "max_shards": 100,
    "max_event_rows": 10_000,
    "max_source_bytes": 10**9,
}


def _workspace(tmp_path: Path) -> Path:
    path = tmp_path / "workspace with spaces—演示"
    path.mkdir(parents=True)
    return path


def test_store_satisfies_both_protocols(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    store = DuckDBRoutingShardStore.bind(ws)
    assert isinstance(store, RoutingRunReader)
    assert isinstance(store, RoutingShardAppender)


def test_bind_accepts_str_and_path_rejects_other_types(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    from_path = DuckDBRoutingShardStore.bind(ws)
    from_str = DuckDBRoutingShardStore.bind(str(ws))
    assert from_path.workspace == Path(str(ws))
    assert from_str.workspace == from_path.workspace
    with pytest.raises(TypeError, match="workspace must be"):
        DuckDBRoutingShardStore.bind(123)  # type: ignore[arg-type]


def test_reader_from_workspace_returns_reader_protocol(tmp_path: Path) -> None:
    reader = reader_from_workspace(_workspace(tmp_path))
    assert isinstance(reader, RoutingRunReader)


def test_append_delegation_matches_direct_call(tmp_path: Path) -> None:
    if duckdb is None:
        pytest.skip("duckdb store extra is unavailable")
    from tests.test_store_routing_shards import _run_result

    result, _model, _inspection = _run_result()
    ws_one = _workspace(tmp_path / "one")
    ws_two = _workspace(tmp_path / "two")

    via_port = DuckDBRoutingShardStore.bind(ws_one).append(result)
    direct = append_routing_shard(ws_two, result)
    assert via_port == direct


def test_append_with_token_text_storage(tmp_path: Path) -> None:
    if duckdb is None:
        pytest.skip("duckdb store extra is unavailable")
    from tests.test_store_routing_shards import _run_result

    result, _model, _inspection = _run_result()
    receipt = DuckDBRoutingShardStore.bind(_workspace(tmp_path)).append(
        result, store_token_text=True
    )
    assert receipt.token_text_stored is True


def test_list_shards_delegation_matches_direct_call(tmp_path: Path) -> None:
    if duckdb is None:
        pytest.skip("duckdb store extra is unavailable")
    from tests.test_store_routing_shards import _run_result

    result, _model, _inspection = _run_result()
    ws = _workspace(tmp_path)
    receipt = append_routing_shard(ws, result)

    store = DuckDBRoutingShardStore.bind(ws)
    assert store.list_shards(run_key=receipt.run_key) == list_routing_shards(
        ws, run_key=receipt.run_key
    )
    assert len(store.list_shards(run_key=receipt.run_key)) == 1


def test_list_runs_delegation_matches_direct_call(tmp_path: Path) -> None:
    if duckdb is None:
        pytest.skip("duckdb store extra is unavailable")
    from tests.test_store_routing_shards import _run_result

    result, _model, _inspection = _run_result()
    ws = _workspace(tmp_path)
    append_routing_shard(ws, result)

    store = DuckDBRoutingShardStore.bind(ws)
    via_port = store.list_runs(**_BUDGETS)
    direct = list_routing_runs(ws, **_BUDGETS)
    assert via_port.run_count == direct.run_count == 1
    assert via_port.shard_count == direct.shard_count
    assert via_port.token_count == direct.token_count
    assert via_port.routing_count == direct.routing_count
    assert via_port.runs == direct.runs


def test_append_error_propagates_workspace_stage(tmp_path: Path) -> None:
    if duckdb is None:
        pytest.skip("duckdb store extra is unavailable")
    from tests.test_store_routing_shards import _run_result

    result, _model, _inspection = _run_result()
    store = DuckDBRoutingShardStore(workspace=tmp_path / "missing")
    with pytest.raises(RoutingShardError) as exc:
        store.append(result)
    assert exc.value.stage == "workspace"
    assert str(exc.value) == "routing shard failed at workspace"


def test_ports_import_without_duckdb() -> None:
    script = (
        "import sys\n"
        "sys.modules['duckdb'] = None\n"
        "import moeatlas.store.ports\n"
        "print('ports-import-ok')\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    ).rstrip(os.pathsep)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "ports-import-ok" in completed.stdout


def test_no_model_runtime_imported_by_ports() -> None:
    assert not any(
        name in sys.modules for name in ("torch", "transformers", "safetensors")
    )
