from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import moeatlas.analysis as analysis_module
from moeatlas.server import RoutingSimilarityResponse, create_app
from moeatlas.services import initialize_workspace
from tests.test_analysis_routing_compare import _comparison_matrix, _matrix

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
TestClient = pytest.importorskip("fastapi.testclient").TestClient


def _client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    initialize_workspace(workspace)
    from moeatlas.services import record_run_record
    from tests.test_run_lifecycle import record

    record_run_record(workspace, record(run_key="run-baseline"), at="t0")
    record_run_record(workspace, record(run_key="run-comparison"), at="t1")
    matrices = {
        "run-baseline": _matrix(),
        "run-comparison": _comparison_matrix(),
    }
    from tests.test_runtime_routing_forward import _run

    _, _, inspection = _run(token_count=2)
    inspection_directory = workspace / "inspections"
    inspection_directory.mkdir()
    for run_key in matrices:
        (inspection_directory / f"{run_key}.json").write_text(
            inspection.to_json(), encoding="utf-8"
        )
    monkeypatch.setattr(
        analysis_module,
        "aggregate_routing_load",
        lambda _workspace, _inspection, *, run_key, **_kwargs: matrices[run_key],
    )
    return TestClient(create_app(workspace))


def test_similarity_endpoint_returns_typed_normalized_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    response = client.get(
        "/api/compare/similarity",
        params={
            "baseline_run_key": "run-baseline",
            "comparison_run_key": "run-comparison",
            "top_n": 2,
        },
    )
    assert response.status_code == 200
    parsed = RoutingSimilarityResponse.model_validate_json(response.content)
    assert parsed.status == "available"
    assert parsed.report is not None
    assert parsed.report["top_n"] == 2
    assert len(parsed.report["js_divergence_rows"]) == 2


def test_similarity_endpoint_keeps_incompatible_runs_explicitly_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    monkeypatch.setattr(
        analysis_module,
        "aggregate_routing_load",
        lambda _workspace, _inspection, *, run_key, **_kwargs: (
            _matrix()
            if run_key == "run-baseline"
            else replace(_comparison_matrix(), model_key="model:acme/other@r1")
        ),
    )
    response = client.get(
        "/api/compare/similarity",
        params={
            "baseline_run_key": "run-baseline",
            "comparison_run_key": "run-comparison",
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "unavailable"
    assert "exact shared routing topology" in response.json()["reason"]


def test_similarity_endpoint_rejects_same_run_and_invalid_top_n(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    same = client.get(
        "/api/compare/similarity",
        params={
            "baseline_run_key": "run-baseline",
            "comparison_run_key": "run-baseline",
        },
    )
    assert same.status_code == 400
    invalid = client.get(
        "/api/compare/similarity",
        params={
            "baseline_run_key": "run-baseline",
            "comparison_run_key": "run-comparison",
            "top_n": 0,
        },
    )
    assert invalid.status_code == 422
