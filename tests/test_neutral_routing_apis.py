from __future__ import annotations

import ast
import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest

import moeatlas.event_validation as event_validation
import moeatlas.runtime.routing_forward as routing_forward
import moeatlas.store.routing_shards as routing_shards
from moeatlas.store import (
    append_mixtral_routing_shard,
    append_routing_shard,
    list_mixtral_routing_runs,
    list_mixtral_routing_shards,
    list_routing_runs,
    list_routing_shards,
)

try:
    import duckdb
except ImportError:  # pragma: no cover - exercised only without the store extra
    duckdb = None

ROOT = Path(__file__).resolve().parents[1]


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add("." * node.level + (node.module or ""))
    return imported


def test_event_validation_is_runtime_and_storage_independent() -> None:
    source = ROOT / "src" / "moeatlas" / "event_validation.py"
    imports = _imports(source)
    assert imports == {"__future__", ".events"}
    tree = ast.parse(source.read_text())
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"__import__", "exec", "eval"}
        for node in ast.walk(tree)
    )


def test_runtime_private_validators_are_exact_compatibility_aliases() -> None:
    assert routing_forward._fresh_token_events is event_validation.fresh_token_events
    assert routing_forward._fresh_routing_events is event_validation.fresh_routing_events
    assert routing_forward._validate_routing_links is event_validation.validate_routing_links


def test_storage_uses_neutral_validation_not_runtime_private_helpers() -> None:
    source = (ROOT / "src" / "moeatlas" / "store" / "routing_shards.py").read_text()
    tree = ast.parse(source)
    runtime_imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "runtime.routing_forward"
    ]
    assert len(runtime_imports) == 1
    assert [alias.name for alias in runtime_imports[0].names] == ["RoutingForwardResult"]
    neutral = ("fresh_token_events", "fresh_routing_events", "validate_routing_links")
    assert all(name in source for name in neutral)
    called: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            called[node.func.id] = called.get(node.func.id, 0) + 1
    # Every internal call routes through the historical private aliases; the
    # neutral functions are only imported to define those aliases.
    assert not set(neutral) & set(called)
    assert called.get("_fresh_token_events", 0) >= 1
    assert called.get("_fresh_routing_events", 0) >= 1
    assert called.get("_validate_routing_links", 0) >= 2


def test_storage_private_validators_are_exact_compatibility_aliases() -> None:
    assert routing_shards._fresh_token_events is event_validation.fresh_token_events
    assert routing_shards._fresh_routing_events is event_validation.fresh_routing_events
    assert routing_shards._validate_routing_links is (
        event_validation.validate_routing_links
    )


def test_storage_internal_calls_honor_patched_private_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if duckdb is None:
        pytest.skip("duckdb store extra is unavailable")
    from .test_store_routing_shards import _qwen_result

    calls: list[tuple[int, int]] = []
    real = routing_shards._validate_routing_links

    def recording(token_events: tuple, routing_events: tuple) -> None:
        calls.append((len(token_events), len(routing_events)))
        real(token_events, routing_events)

    monkeypatch.setattr(routing_shards, "_validate_routing_links", recording)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    result = _qwen_result()
    receipt = append_routing_shard(workspace, result)
    assert receipt.created is True
    assert len(calls) >= 1
    assert list_routing_shards(workspace, run_key=receipt.run_key)
    # The reopen path validates links through the same patchable attribute.
    assert len(calls) >= 2


def test_event_validation_imports_without_runtime_storage_or_model_stack() -> None:
    script = "\n".join(
        (
            "import sys",
            "blocked = (",
            "    'torch',",
            "    'transformers',",
            "    'duckdb',",
            "    'moeatlas.runtime',",
            "    'moeatlas.runtime.routing_forward',",
            "    'moeatlas.store',",
            "    'moeatlas.store.routing_shards',",
            ")",
            "for name in blocked:",
            "    sys.modules[name] = None",
            "import moeatlas.event_validation as event_validation",
            "assert event_validation.fresh_token_events is not None",
            "assert event_validation.fresh_routing_events is not None",
            "assert event_validation.validate_routing_links is not None",
            "print('isolated-import-ok')",
        )
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
    assert "isolated-import-ok" in completed.stdout


def test_neutral_and_historical_storage_functions_are_exact_aliases() -> None:
    pairs = (
        (append_routing_shard, append_mixtral_routing_shard),
        (list_routing_shards, list_mixtral_routing_shards),
        (list_routing_runs, list_mixtral_routing_runs),
    )
    for neutral, historical in pairs:
        assert neutral is historical
        assert inspect.signature(neutral) == inspect.signature(historical)


def test_neutral_storage_surface_round_trips_qwen_result(tmp_path: Path) -> None:
    if duckdb is None:
        pytest.skip("duckdb store extra is unavailable")
    from .test_store_routing_shards import _qwen_result

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    result = _qwen_result()
    receipt = append_routing_shard(workspace, result)
    assert receipt.created is True
    assert list_routing_shards(workspace, run_key=receipt.run_key) == (
        append_routing_shard(workspace, result),
    )
    inventory = list_routing_runs(
        workspace,
        max_runs=1,
        max_shards=1,
        max_event_rows=len(result.token_events) + len(result.routing_events),
        max_source_bytes=1_000_000,
    )
    assert inventory.run_count == 1
    assert inventory.runs[0].run_key == receipt.run_key
    assert inventory.routing_count == len(result.routing_events)
