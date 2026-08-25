"""Model-free tests for the workspace application-service layer."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from moeatlas.runs import GenerationConfig
from moeatlas.services import (
    WorkspaceSnapshot,
    initialize_workspace,
    open_workspace,
    query_runs,
    record_run_record,
    register_run,
    sync_runs_from_shards,
)
from moeatlas.store.catalog import (
    CatalogRebuildReceipt,
    WorkspaceCatalogError,
)
from tests.test_run_contracts import run_specification
from tests.test_run_lifecycle import record as _make_record

try:
    import duckdb
except ImportError:  # pragma: no cover
    duckdb = None

ROOT = Path(__file__).resolve().parents[1]


def _workspace(tmp_path: Path) -> Path:
    path = tmp_path / "workspace with spaces—演示"
    path.mkdir()
    return path


# ---------------------------------------------------------------------------
# initialize + open round-trip
# ---------------------------------------------------------------------------


def test_initialize_and_open(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    cat = initialize_workspace(ws, at="2026-01-01T00:00:00Z")
    assert cat.created_at == "2026-01-01T00:00:00Z"
    assert cat.runs == ()

    snapshot = open_workspace(ws)
    assert isinstance(snapshot, WorkspaceSnapshot)
    assert snapshot.path == ws.resolve()
    assert snapshot.catalog.created_at == cat.created_at


def test_open_uninitialized_raises(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    with pytest.raises(WorkspaceCatalogError) as exc:
        open_workspace(ws)
    assert exc.value.stage == "reopen"


# ---------------------------------------------------------------------------
# register_run + query_runs
# ---------------------------------------------------------------------------


def test_register_run_and_query(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    initialize_workspace(ws)

    spec = run_specification()
    entry = register_run(ws, spec, at="t0")
    assert entry.run_key == spec.run_key
    assert entry.state == "planned"
    assert entry.specification_fingerprint == spec.run_key
    assert entry.registered_at == "t0"

    results = query_runs(ws)
    assert len(results) == 1
    assert results[0].run_key == spec.run_key


def test_register_run_idempotent(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    initialize_workspace(ws)

    spec = run_specification()
    first = register_run(ws, spec, at="t0")
    assert first.registered_at == "t0"

    second = register_run(ws, spec, at="t1")
    assert second.run_key == first.run_key
    # registered_at preserved from original registration
    assert second.registered_at == "t0"


def test_query_runs_by_state(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    initialize_workspace(ws)

    # Tags are metadata, not identity: differ by generation seed so the two
    # specifications get distinct content-addressed run keys.
    spec_a = run_specification(generation=GenerationConfig(seed=1, temperature=0.7))
    spec_b = run_specification(generation=GenerationConfig(seed=2, temperature=0.7))
    assert spec_a.run_key != spec_b.run_key
    register_run(ws, spec_a, at="t0")
    register_run(ws, spec_b, at="t0")

    # Record spec_b as running
    rec = _make_record(
        run_key=spec_b.run_key,
        specification_fingerprint=spec_b.run_key,
        state="running",
    )
    record_run_record(ws, rec, at="t1")

    running = query_runs(ws, state="running")
    assert len(running) == 1
    assert running[0].run_key == spec_b.run_key

    planned = query_runs(ws, state="planned")
    assert len(planned) == 1
    assert planned[0].run_key == spec_a.run_key


def test_query_runs_invalid_state_raises(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    initialize_workspace(ws)
    with pytest.raises(ValueError, match="state must be one of"):
        query_runs(ws, state="nonexistent")


def test_query_runs_max_results_truncation(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    initialize_workspace(ws)
    for i in range(5):
        s = run_specification(generation=GenerationConfig(seed=i + 1, temperature=0.7))
        register_run(ws, s, at=f"t{i}")

    results = query_runs(ws, max_results=3)
    assert len(results) == 3


def test_query_runs_max_results_zero_raises(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    initialize_workspace(ws)
    with pytest.raises(ValueError, match="max_results must be a positive int"):
        query_runs(ws, max_results=0)


# ---------------------------------------------------------------------------
# record_run_record
# ---------------------------------------------------------------------------


def test_record_run_record_auto_registers_unknown_run(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    initialize_workspace(ws)

    rec = _make_record(state="provisioning")
    entry = record_run_record(ws, rec, at="t0")
    assert entry.run_key == rec.run_key
    assert entry.state == "provisioning"
    assert entry.attempt == 1

    results = query_runs(ws)
    assert len(results) == 1
    assert results[0].state == "provisioning"


def test_record_run_record_preserves_counts(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    initialize_workspace(ws)

    # First, upsert an entry with counts via catalog directly
    from moeatlas.store.catalog import RunRegistryEntry, upsert_run_entry

    spec = run_specification()
    entry_with_counts = RunRegistryEntry(
        run_key=spec.run_key,
        run_name="preserved-name",
        specification_fingerprint=spec.run_key,
        state="planned",
        attempt=1,
        shard_count=5,
        token_event_count=100,
        routing_event_count=20,
        registered_at="t0",
        updated_at="t0",
    )
    upsert_run_entry(ws, entry_with_counts)

    # Now record a lifecycle update - counts should be preserved
    rec = _make_record(run_key=spec.run_key, state="running")
    entry = record_run_record(ws, rec, at="t1")
    assert entry.shard_count == 5
    assert entry.token_event_count == 100
    assert entry.routing_event_count == 20
    assert entry.state == "running"
    assert entry.run_name == "preserved-name"


# ---------------------------------------------------------------------------
# sync_runs_from_shards
# ---------------------------------------------------------------------------


def test_sync_runs_from_shards_returns_receipt(tmp_path: Path) -> None:
    if duckdb is None:
        pytest.skip("duckdb store extra is unavailable")
    ws = _workspace(tmp_path)
    initialize_workspace(ws, at="t0")

    receipt = sync_runs_from_shards(ws)
    assert isinstance(receipt, CatalogRebuildReceipt)
    assert receipt.added == () and receipt.updated == () and receipt.run_count == 0


def test_sync_runs_from_shards_reconciles_registry(tmp_path: Path) -> None:
    if duckdb is None:
        pytest.skip("duckdb store extra is unavailable")
    from moeatlas.store import append_routing_shard
    from moeatlas.store.catalog import RunRegistryEntry, upsert_run_entry
    from tests.test_store_routing_shards import _run_result

    ws = _workspace(tmp_path)
    initialize_workspace(ws)
    # Storage keys use the broad stable-identifier vocabulary ('run-1' in the
    # synthetic fixture), so pre-register lifecycle metadata for the shard run
    # directly instead of deriving it from a strict RunSpecification key.
    result, _model, _inspection = _run_result()
    shard_receipt = append_routing_shard(ws, result)
    upsert_run_entry(
        ws,
        RunRegistryEntry(
            run_key=shard_receipt.run_key,
            state="planned",
            registered_at="t0",
        ),
    )

    receipt = sync_runs_from_shards(ws, at="t1")
    assert receipt.updated == (shard_receipt.run_key,)
    entries = query_runs(ws, state="planned")
    assert len(entries) == 1
    assert entries[0].shard_count == 1
    assert entries[0].token_event_count > 0

    resync = sync_runs_from_shards(ws, at="t2")
    assert resync.unchanged == (shard_receipt.run_key,)
    assert resync.added == () and resync.updated == ()
    assert query_runs(ws, state="planned")[0].state == "planned"


# ---------------------------------------------------------------------------
# TypeError guards
# ---------------------------------------------------------------------------


def test_register_run_rejects_non_run_specification(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    initialize_workspace(ws)
    with pytest.raises(TypeError, match="must be a RunSpecification"):
        register_run(ws, "not a spec")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be a RunSpecification"):
        register_run(ws, object())  # type: ignore[arg-type]


def test_record_run_record_rejects_non_run_record(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    initialize_workspace(ws)
    with pytest.raises(TypeError, match="must be a RunRecord"):
        record_run_record(ws, "not a record")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be a RunRecord"):
        record_run_record(ws, 42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Subprocess isolation: services module must not require duckdb/torch
# ---------------------------------------------------------------------------


def test_services_import_without_model_stack() -> None:
    # Services deliberately compose the storage package; the isolation
    # contract is that no ML/download stack is required for import.
    # moeatlas.runtime and moeatlas.store modules are themselves model-free
    # at import time (torch/duckdb are lazy), so only the ML stack is poisoned.
    script = (
        "import sys\n"
        "for name in ('torch', 'transformers', 'duckdb', 'safetensors'):\n"
        "    sys.modules[name] = None\n"
        "import moeatlas.services\n"
        "print('services-import-ok')\n"
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
    assert "services-import-ok" in completed.stdout


def test_no_model_runtime_imported_by_services() -> None:
    assert not any(
        name in sys.modules for name in ("torch", "transformers", "safetensors")
    )
