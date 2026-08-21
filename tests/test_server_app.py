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
