"""Model-free tests for the versioned workspace catalog and run registry."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from moeatlas.store.catalog import (
    WORKSPACE_CATALOG_SCHEMA_VERSION,
    CatalogRebuildReceipt,
    RunRegistryEntry,
    WorkspaceCatalog,
    WorkspaceCatalogError,
    catalog_path,
    initialize_catalog,
    read_catalog,
    rebuild_catalog,
    upsert_run_entry,
)

try:
    import duckdb
except ImportError:  # pragma: no cover
    duckdb = None

ROOT = Path(__file__).resolve().parents[1]

_RUN_KEY_A = "run:" + "a" * 64
_RUN_KEY_B = "run:" + "b" * 64


def entry(**overrides: object) -> RunRegistryEntry:
    values: dict[str, object] = {"run_key": _RUN_KEY_A}
    values.update(overrides)
    return RunRegistryEntry(**values)  # type: ignore[arg-type]


def _workspace(tmp_path: Path) -> Path:
    path = tmp_path / "workspace with spaces—演示"
    path.mkdir()
    return path


def _raw_write(workspace: Path, payload: dict[str, object]) -> None:
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    catalog_path(workspace).write_bytes(body.encode("utf-8") + b"\n")


def _file_bytes(workspace: Path) -> bytes:
    return catalog_path(workspace).read_bytes()


def test_initialize_and_read_round_trip(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    catalog = initialize_catalog(ws, at="2026-01-01T00:00:00Z")
    assert catalog.created_at == "2026-01-01T00:00:00Z"
    assert catalog.updated_at == "2026-01-01T00:00:00Z"
    assert catalog.runs == ()
    assert catalog.manifest_type == "workspace_catalog"
    assert catalog.schema_version == WORKSPACE_CATALOG_SCHEMA_VERSION

    reopened = read_catalog(ws)
    assert reopened == catalog


def test_initialize_twice_conflicts(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    initialize_catalog(ws)
    with pytest.raises(WorkspaceCatalogError) as exc:
        initialize_catalog(ws)
    assert exc.value.stage == "conflict"


def test_read_uninitialized_is_reopen(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    with pytest.raises(WorkspaceCatalogError, match="not initialized") as exc:
        read_catalog(ws)
    assert exc.value.stage == "reopen"
    assert str(exc.value).startswith("workspace catalog failed at reopen")


def test_workspace_validation_stages(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(WorkspaceCatalogError) as exc:
        read_catalog(missing)
    assert exc.value.stage == "workspace"

    not_dir = tmp_path / "plain-file"
    not_dir.write_text("data")
    with pytest.raises(WorkspaceCatalogError) as exc:
        read_catalog(not_dir)
    assert exc.value.stage == "workspace"

    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    os.symlink(real, link)
    with pytest.raises(WorkspaceCatalogError) as exc:
        initialize_catalog(link)
    assert exc.value.stage == "workspace"


def test_upsert_add_update_and_idempotence(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    initialize_catalog(ws, at="t0")

    catalog = upsert_run_entry(ws, entry(state="planned", registered_at="t0"), at="t0")
    assert [item.run_key for item in catalog.runs] == [_RUN_KEY_A]

    updated = upsert_run_entry(
        ws, entry(state="running", shard_count=3, updated_at="t9"), at="t1"
    )
    merged = updated.runs[0]
    assert merged.state == "running"
    assert merged.shard_count == 3
    assert merged.registered_at == "t0"
    assert merged.updated_at == "t1"

    before = _file_bytes(ws)
    stat_before = catalog_path(ws).stat().st_mtime_ns
    unchanged = upsert_run_entry(ws, merged, at=None)
    assert unchanged == updated
    assert _file_bytes(ws) == before
    assert catalog_path(ws).stat().st_mtime_ns == stat_before


def test_catalog_enforces_unique_immutable_run_names(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    initialize_catalog(ws)
    upsert_run_entry(ws, entry(run_name="cyber-baseline"))

    with pytest.raises(WorkspaceCatalogError, match="cannot be renamed"):
        upsert_run_entry(ws, entry(run_name="renamed"))
    with pytest.raises(WorkspaceCatalogError, match="already in use"):
        upsert_run_entry(
            ws,
            RunRegistryEntry(run_key=_RUN_KEY_B, run_name="CYBER-BASELINE"),
        )


def test_upsert_max_runs_conflict(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    initialize_catalog(ws)
    upsert_run_entry(ws, entry(), max_runs=1)
    with pytest.raises(WorkspaceCatalogError) as exc:
        upsert_run_entry(ws, entry(run_key=_RUN_KEY_B), max_runs=1)
    assert exc.value.stage == "conflict"


def test_rebuild_adds_observed_run(tmp_path: Path) -> None:
    if duckdb is None:
        pytest.skip("duckdb store extra is unavailable")
    from moeatlas.store import append_routing_shard
    from tests.test_store_routing_shards import _run_result, _workspace

    ws = _workspace(tmp_path)
    initialize_catalog(ws)
    result, _model, _inspection = _run_result()
    append_routing_shard(ws, result)

    catalog, receipt = rebuild_catalog(ws, at="t1")
    assert isinstance(receipt, CatalogRebuildReceipt)
    assert receipt.added == ("run-1",)
    assert receipt.updated == () and receipt.removed == ()
    assert receipt.run_count == 1
    observed = catalog.runs[0]
    assert observed.run_key == "run-1"
    assert observed.shard_count == 1
    assert observed.token_event_count > 0
    assert observed.routing_event_count > 0
    assert observed.state is None
    assert observed.token_text_policy == "redacted"
    assert observed.registered_at == "t1"


def test_rebuild_preserves_registered_lifecycle(tmp_path: Path) -> None:
    if duckdb is None:
        pytest.skip("duckdb store extra is unavailable")
    from moeatlas.store import append_routing_shard
    from tests.test_store_routing_shards import _run_result, _workspace

    ws = _workspace(tmp_path)
    initialize_catalog(ws)
    result, _model, _inspection = _run_result()
    append_routing_shard(ws, result)
    upsert_run_entry(
        ws, entry(run_key="run-1", state="planned", registered_at="t0"), at="t0"
    )

    catalog, receipt = rebuild_catalog(ws, at="t1")
    assert receipt.updated == ("run-1",)
    observed = catalog.runs[0]
    assert observed.state == "planned"
    assert observed.registered_at == "t0"
    assert observed.shard_count == 1


def test_rebuild_is_idempotent_and_keeps_unsharded_entries(tmp_path: Path) -> None:
    if duckdb is None:
        pytest.skip("duckdb store extra is unavailable")
    from moeatlas.store import append_routing_shard
    from tests.test_store_routing_shards import _run_result, _workspace

    ws = _workspace(tmp_path)
    initialize_catalog(ws)
    result, _model, _inspection = _run_result()
    append_routing_shard(ws, result)
    upsert_run_entry(ws, entry(run_key="run-planned-only", state="planned"), at="t0")

    first, first_receipt = rebuild_catalog(ws, at="t1")
    assert first_receipt.added == ("run-1",)
    assert first_receipt.unchanged == ("run-planned-only",)
    planned_only = next(item for item in first.runs if item.run_key == "run-planned-only")
    assert planned_only.shard_count == 0

    before = _file_bytes(ws)
    second, second_receipt = rebuild_catalog(ws, at=None)
    assert second_receipt.added == () and second_receipt.updated == ()
    assert set(second_receipt.unchanged) == {"run-1", "run-planned-only"}
    assert _file_bytes(ws) == before
    assert second == first


def test_read_rejects_unknown_schema_version(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    initialize_catalog(ws)
    payload = dict(json.loads(_file_bytes(ws).decode("utf-8")))
    payload["schema_version"] = "2.0"
    _raw_write(ws, payload)
    with pytest.raises(WorkspaceCatalogError, match="schema_version") as exc:
        read_catalog(ws)
    assert exc.value.stage == "reopen"


def test_read_rejects_wrong_manifest_type(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    initialize_catalog(ws)
    payload = dict(json.loads(_file_bytes(ws).decode("utf-8")))
    payload["manifest_type"] = "something_else"
    _raw_write(ws, payload)
    with pytest.raises(WorkspaceCatalogError, match="manifest") as exc:
        read_catalog(ws)
    assert exc.value.stage == "reopen"


def test_read_rejects_corrupt_payloads(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    initialize_catalog(ws)
    catalog_path(ws).write_bytes(b"{not json")
    with pytest.raises(WorkspaceCatalogError) as exc:
        read_catalog(ws)
    assert exc.value.stage == "reopen"

    catalog_path(ws).write_bytes(b"[]\n")
    with pytest.raises(WorkspaceCatalogError, match="JSON object") as exc:
        read_catalog(ws)
    assert exc.value.stage == "reopen"


def test_entry_field_validation() -> None:
    with pytest.raises(ValidationError, match="run_key"):
        entry(run_key="has spaces")
    with pytest.raises(ValidationError, match="specification_fingerprint"):
        entry(specification_fingerprint="run:XYZ")
    with pytest.raises(ValidationError, match="state must be one of"):
        entry(state="flying")
    with pytest.raises(ValidationError, match="attempt"):
        entry(attempt=0)
    with pytest.raises(ValidationError, match="shard_count"):
        entry(shard_count=-1)
    with pytest.raises(ValidationError, match="token_text_policy"):
        entry(token_text_policy="sometimes")
    with pytest.raises(ValidationError, match="control characters"):
        entry(registered_at="bad\x00ts")
    with pytest.raises(ValidationError, match="at most 64 characters"):
        entry(updated_at="t" * 65)


def test_catalog_registry_order_enforced() -> None:
    first = entry()
    second = entry(run_key=_RUN_KEY_B)
    with pytest.raises(ValidationError, match="sorted"):
        WorkspaceCatalog(runs=(second, first))
    with pytest.raises(ValidationError, match="unique"):
        WorkspaceCatalog(runs=(first, entry()))


def test_error_stage_contract() -> None:
    assert str(WorkspaceCatalogError("reopen")) == "workspace catalog failed at reopen"
    assert WorkspaceCatalogError("write", "boom").stage == "write"
    with pytest.raises(ValueError, match="unsupported workspace catalog stage"):
        WorkspaceCatalogError("bogus")


def test_catalog_path_is_pure(tmp_path: Path) -> None:
    path = catalog_path(tmp_path)
    assert path == tmp_path / ".moeatlas" / "catalog.json"
    assert not path.parent.exists()


def test_publication_is_canonical_and_leaves_no_staging(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    initialize_catalog(ws)
    upsert_run_entry(ws, entry(state="planned"), at="t0")

    staging = list((ws / ".moeatlas").glob(".staging-catalog-*"))
    assert staging == []
    raw = _file_bytes(ws)
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    text = raw.decode("utf-8")
    canonical = json.dumps(
        json.loads(text), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    assert text == canonical + "\n"


def test_catalog_import_without_duckdb() -> None:
    script = (
        "import sys\n"
        "sys.modules['duckdb'] = None\n"
        "import moeatlas.store.catalog\n"
        "print('catalog-import-ok')\n"
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
    assert "catalog-import-ok" in completed.stdout


def test_no_model_runtime_imported_by_catalog() -> None:
    assert not any(
        name in sys.modules for name in ("torch", "transformers", "safetensors")
    )
