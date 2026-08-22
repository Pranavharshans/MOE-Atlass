"""Contract tests for the local read-only server application."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from moeatlas.server import (
    SERVER_SCHEMA_VERSION,
    ServerDependencyError,
    create_app,
)
from moeatlas.services import initialize_workspace

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    path = tmp_path / "workspace"
    path.mkdir()
    initialize_workspace(path)
    return path


@pytest.fixture()
def client(workspace: Path):
    from fastapi.testclient import TestClient

    return TestClient(create_app(workspace, max_results=50))


def test_surface_is_pinned() -> None:
    assert SERVER_SCHEMA_VERSION == "1.0"
    assert str(ServerDependencyError()) == (
        "server dependency 'fastapi' is not installed"
    )


def test_create_app_validates_bounds(workspace: Path) -> None:
    with pytest.raises(TypeError):
        create_app(workspace, max_results=True)
    with pytest.raises(TypeError):
        create_app(workspace, max_results="10")
    with pytest.raises(ValueError):
        create_app(workspace, max_results=0)
    with pytest.raises(ValueError):
        create_app(workspace, max_results=100_001)
    with pytest.raises(TypeError):
        create_app(7)


def test_create_app_without_fastapi_reports_the_fixed_dependency_error(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = __import__

    def _blocked(name: str, *args: object, **kwargs: object):
        if name == "fastapi" or name.startswith("fastapi."):
            raise ImportError(f"blocked for test: {name}")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setitem(sys.modules, "fastapi", None)
    monkeypatch.setattr("builtins.__import__", _blocked)
    with pytest.raises(ServerDependencyError):
        create_app(workspace)


def test_healthz_reports_package_identity(client) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    document = response.json()
    assert document["package_name"] == "MoEAtlas"
    assert isinstance(document["package_version"], str)
    assert document["model_validation_status"] == "deferred"


def test_workspace_snapshot_on_initialized_catalog(client) -> None:
    response = client.get("/api/workspace")
    assert response.status_code == 200
    document = response.json()
    assert document["run_count"] == 0


def test_uninitialized_workspace_is_reported_with_fixed_detail(
    tmp_path: Path,
) -> None:
    from fastapi.testclient import TestClient

    bare = tmp_path / "bare"
    bare.mkdir()
    client = TestClient(create_app(bare))
    response = client.get("/api/workspace")
    assert response.status_code == 404
    assert response.json()["detail"] == "workspace is not initialized"
    runs = client.get("/api/runs")
    assert runs.status_code == 404
    assert runs.json()["detail"] == "workspace is not initialized"


def test_runs_listing_is_bounded_and_sorted(
    workspace: Path,
) -> None:
    from fastapi.testclient import TestClient

    from moeatlas.runs.specs import RunSpecification
    from moeatlas.services import register_run
    from tests.test_run_contracts import data_provenance, model_provenance

    specification = RunSpecification(
        model=model_provenance(),
        data=data_provenance(),
    )
    register_run(workspace, specification, at="2026-08-21T00:00:00Z")
    client = TestClient(create_app(workspace))
    response = client.get("/api/runs")
    assert response.status_code == 200
    document = response.json()
    assert document["count"] == 1
    assert document["entries"][0]["state"] == "planned"

    empty_state = client.get("/api/runs", params={"state": "completed"})
    assert empty_state.status_code == 200
    assert empty_state.json()["count"] == 0

    bad_state = client.get("/api/runs", params={"state": "not-a-state"})
    assert bad_state.status_code == 400
    assert bad_state.json()["detail"] == "invalid run state filter"


def test_adapters_endpoint_lists_builtin_plugins(client) -> None:
    response = client.get("/api/adapters")
    assert response.status_code == 200
    document = response.json()
    names = [entry["name"] for entry in document["entries"]]
    assert "huggingface-mixtral-static" in names
    assert all(entry["status"] == "enabled" for entry in document["entries"])
    assert document["failures"] == []


# ---------------------------------------------------------------------------
# Run artifact endpoints (R4)
# ---------------------------------------------------------------------------

try:
    import duckdb as _duckdb
except ImportError:  # pragma: no cover - exercised only without the store extra
    _duckdb = None

_STORE_REQUIRED = pytest.mark.skipif(_duckdb is None, reason="duckdb store extra unavailable")


def _append_synthetic_shard(workspace: Path):
    from moeatlas.store import append_routing_shard
    from tests.test_runtime_routing_forward import _run

    result, _, _ = _run(token_count=2)
    return append_routing_shard(workspace, result)


def test_artifact_endpoints_report_typed_404s_on_empty_workspace(client) -> None:
    detail = client.get("/api/runs/run-1")
    assert detail.status_code == 404
    assert detail.json()["detail"] == "run is not registered"

    summary = client.get("/api/runs/run-1/summary")
    assert summary.status_code == 404
    assert summary.json()["detail"] == "run is not registered"

    heatmap = client.get("/api/runs/run-1/heatmap")
    assert heatmap.status_code == 404
    assert heatmap.json()["detail"] == "run is not registered"


def test_artifact_endpoints_report_uninitialized_workspace(
    tmp_path: Path,
) -> None:
    from fastapi.testclient import TestClient

    bare = tmp_path / "bare"
    bare.mkdir()
    bare_client = TestClient(create_app(bare))
    for path in (
        "/api/runs/run-1",
        "/api/runs/run-1/summary",
        "/api/runs/run-1/heatmap",
    ):
        response = bare_client.get(path)
        assert response.status_code == 404
        assert response.json()["detail"] == "workspace is not initialized"


def test_invalid_run_keys_never_reach_the_registry_or_filesystem(client) -> None:
    # Percent-decoded to a whitespace-bearing identifier: rejected before any
    # catalog lookup or filesystem resolution, with the fixed safe detail.
    detail = client.get("/api/runs/run%201")
    assert detail.status_code == 404
    assert detail.json()["detail"] == "run is not registered"

    heatmap = client.get("/api/runs/run%201/heatmap")
    assert heatmap.status_code == 404
    assert heatmap.json()["detail"] == "run is not registered"

    summary = client.get("/api/runs/run%201/summary")
    assert summary.status_code == 404
    assert summary.json()["detail"] == "run is not registered"


@_STORE_REQUIRED
def test_run_detail_serves_registry_entry_and_shard_listing(
    workspace: Path,
) -> None:
    from fastapi.testclient import TestClient
    from pydantic import ValidationError

    from moeatlas.server import RunDetailResponse
    from moeatlas.services import record_run_record
    from tests.test_run_lifecycle import record as _make_record

    receipt = _append_synthetic_shard(workspace)
    record_run_record(
        workspace, _make_record(run_key=receipt.run_key), at="2026-08-21T00:00:00Z"
    )
    http = TestClient(create_app(workspace))
    response = http.get(f"/api/runs/{receipt.run_key}")
    assert response.status_code == 200
    document = response.json()
    assert document["run_key"] == receipt.run_key
    assert document["state"] == "planned"
    assert document["attempt"] == 1
    assert document["registered_at"] == "2026-08-21T00:00:00Z"
    assert len(document["shards"]) == 1
    shard = document["shards"][0]
    assert shard["shard_key"] == receipt.shard_key
    assert shard["relative_path"] == receipt.relative_path
    assert shard["token_count"] == receipt.token_count
    assert shard["routing_count"] == receipt.routing_count
    assert shard["token_text_stored"] is False

    # Wire round-trip strictness: the DTO revalidates the payload and forbids
    # extra fields.
    parsed = RunDetailResponse.model_validate_json(response.content)
    assert parsed.run_key == receipt.run_key
    assert len(parsed.shards) == 1
    tampered = dict(document)
    tampered["unexpected"] = True
    with pytest.raises(ValidationError):
        RunDetailResponse.model_validate(tampered)


def test_run_detail_reports_unknown_run_with_fixed_detail(workspace: Path) -> None:
    from fastapi.testclient import TestClient

    http = TestClient(create_app(workspace))
    response = http.get("/api/runs/run-missing")
    assert response.status_code == 404
    assert response.json()["detail"] == "run is not registered"


@_STORE_REQUIRED
def test_run_detail_maps_storage_failure_to_fixed_detail(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi.testclient import TestClient

    import moeatlas.store.ports as ports_module
    from moeatlas.services import record_run_record
    from tests.test_run_lifecycle import record as _make_record

    record_run_record(workspace, _make_record(), at="t0")
    monkeypatch.setattr(
        ports_module.DuckDBRoutingShardStore,
        "list_shards",
        lambda self, *, run_key: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    http = TestClient(create_app(workspace))
    response = http.get("/api/runs/" + _make_record().run_key)
    assert response.status_code == 404
    assert response.json()["detail"] == "run shards are unavailable"


def test_summary_reports_typed_unavailability(workspace: Path) -> None:
    from fastapi.testclient import TestClient
    from pydantic import ValidationError

    from moeatlas.runs.specs import RunSpecification
    from moeatlas.server import RunSummaryResponse
    from moeatlas.services import register_run
    from tests.test_run_contracts import data_provenance, model_provenance

    specification = RunSpecification(
        model=model_provenance(),
        data=data_provenance(),
    )
    register_run(workspace, specification, at="2026-08-21T00:00:00Z")
    http = TestClient(create_app(workspace))
    response = http.get(f"/api/runs/{specification.run_key}/summary")
    assert response.status_code == 200
    document = response.json()
    assert document["run_key"] == specification.run_key
    assert document["status"] == "unavailable"
    assert document["reason"] == "published routing inspection is unavailable"

    parsed = RunSummaryResponse.model_validate_json(response.content)
    assert parsed.status == "unavailable"
    tampered = dict(document)
    tampered["matrix"] = [[1.0]]
    with pytest.raises(ValidationError):
        RunSummaryResponse.model_validate(tampered)


def _register_simple_run(workspace: Path, run_key: str) -> str:
    from moeatlas.services import record_run_record
    from tests.test_run_lifecycle import record as _make_record

    record_run_record(
        workspace, _make_record(run_key=run_key, specification_fingerprint=None), at="t0"
    )
    return run_key


_HEATMAP_BODY = b"<!doctype html><html><body>routing load</body></html>"


def _publish_heatmap(workspace: Path, run_key: str, payload: bytes) -> Path:
    directory = workspace / "heatmaps"
    directory.mkdir(exist_ok=True)
    target = directory / f"{run_key}.html"
    target.write_bytes(payload)
    return target


def test_heatmap_serves_published_document(workspace: Path) -> None:
    from fastapi.testclient import TestClient

    run_key = _register_simple_run(workspace, "run-1")
    published = _publish_heatmap(workspace, run_key, _HEATMAP_BODY)
    http = TestClient(create_app(workspace))
    response = http.get(f"/api/runs/{run_key}/heatmap")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.content == _HEATMAP_BODY
    assert published.exists()


def test_heatmap_absent_document_is_typed_404(workspace: Path) -> None:
    from fastapi.testclient import TestClient

    run_key = _register_simple_run(workspace, "run-1")
    http = TestClient(create_app(workspace))
    response = http.get(f"/api/runs/{run_key}/heatmap")
    assert response.status_code == 404
    assert response.json()["detail"] == "run heatmap is not published"


def test_heatmap_symlink_document_is_rejected(workspace: Path) -> None:
    from fastapi.testclient import TestClient

    run_key = _register_simple_run(workspace, "run-1")
    outside = workspace / "outside.html"
    outside.write_bytes(_HEATMAP_BODY)
    directory = workspace / "heatmaps"
    directory.mkdir(exist_ok=True)
    (directory / f"{run_key}.html").symlink_to(outside)
    http = TestClient(create_app(workspace))
    response = http.get(f"/api/runs/{run_key}/heatmap")
    assert response.status_code == 404
    assert response.json()["detail"] == "run heatmap is not published"


def test_heatmap_symlinked_directory_is_rejected(workspace: Path) -> None:
    from fastapi.testclient import TestClient

    run_key = _register_simple_run(workspace, "run-1")
    outside = workspace / "real-heatmaps"
    outside.mkdir()
    (outside / f"{run_key}.html").write_bytes(_HEATMAP_BODY)
    (workspace / "heatmaps").symlink_to(outside)
    http = TestClient(create_app(workspace))
    response = http.get(f"/api/runs/{run_key}/heatmap")
    assert response.status_code == 404
    assert response.json()["detail"] == "run heatmap is not published"


def test_heatmap_oversized_document_exceeds_the_budget(workspace: Path) -> None:
    from fastapi.testclient import TestClient

    run_key = _register_simple_run(workspace, "run-1")
    _publish_heatmap(workspace, run_key, b"x" * 64)
    http = TestClient(create_app(workspace, max_artifact_bytes=8))
    response = http.get(f"/api/runs/{run_key}/heatmap")
    assert response.status_code == 404
    assert response.json()["detail"] == "run heatmap exceeds the serving byte budget"


def test_create_app_validates_artifact_budget(workspace: Path) -> None:
    with pytest.raises(TypeError):
        create_app(workspace, max_artifact_bytes=True)
    with pytest.raises(ValueError):
        create_app(workspace, max_artifact_bytes=0)
    with pytest.raises(ValueError):
        create_app(workspace, max_artifact_bytes=100_000_001)
