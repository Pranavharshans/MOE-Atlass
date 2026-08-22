"""Bounded public Hub search contracts used by the research console."""

from __future__ import annotations

from pathlib import Path

import pytest

import moeatlas.services.hub as hub_service
from moeatlas.services import initialize_workspace


def test_search_hub_normalizes_and_bounds_public_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[str] = []

    def fake_fetch(url: str) -> object:
        captured.append(url)
        return [
            {
                "id": "org/model",
                "author": "org",
                "downloads": 1234,
                "likes": 12,
                "pipeline_tag": "text-generation",
                "library_name": "transformers",
                "tags": ["a", "b", "a"],
                "lastModified": "2026-08-22T00:00:00Z",
            },
            {"id": "org/model"},
            {"modelId": "org/second", "downloads": -1},
            {"id": 7},
        ]

    monkeypatch.setattr(hub_service, "_fetch_json", fake_fetch)
    entries = hub_service.search_hub("model", "org/mod", limit=3)

    assert [entry.identifier for entry in entries] == ["org/model", "org/second"]
    assert entries[0].downloads == 1234
    assert entries[0].tags == ("a", "b", "a")
    assert "https://huggingface.co/api/models?" in captured[0]
    assert "limit=3" in captured[0]
    assert "search=org%2Fmod" in captured[0]


@pytest.mark.parametrize(
    ("kind", "query", "limit"),
    (("unknown", "org/model", 6), ("model", "x", 6), ("model", "org/model", 0)),
)
def test_search_hub_rejects_invalid_requests(
    kind: str, query: str, limit: int
) -> None:
    with pytest.raises((TypeError, ValueError)):
        hub_service.search_hub(kind, query, limit=limit)


def test_search_hub_failure_has_fixed_public_message(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(_: str) -> object:
        raise hub_service.HubSearchError()

    monkeypatch.setattr(hub_service, "_fetch_json", fail)
    with pytest.raises(hub_service.HubSearchError) as error:
        hub_service.search_hub("dataset", "org/data")
    assert str(error.value) == "Hugging Face search is temporarily unavailable"
    assert "org/data" not in str(error.value)


fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    path = tmp_path / "workspace"
    path.mkdir()
    initialize_workspace(path)
    return path


def test_server_hub_search_has_strict_wire_shape(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    monkeypatch.setattr(
        hub_service,
        "search_hub",
        lambda kind, query, *, limit: (
            hub_service.HubSearchEntry(
                identifier="org/model",
                kind=kind,
                downloads=42,
                tags=("text-generation",),
            ),
        ),
    )
    from moeatlas.server import create_app

    response = TestClient(create_app(workspace)).get(
        "/api/hub/search", params={"kind": "model", "q": "org/mod", "limit": 1}
    )
    assert response.status_code == 200
    document = response.json()
    assert document["schema_version"] == "1.0"
    assert document["kind"] == "model"
    assert document["query"] == "org/mod"
    assert document["entries"][0]["identifier"] == "org/model"
    assert document["entries"][0]["tags"] == ["text-generation"]


def test_server_hub_search_maps_invalid_and_unavailable_requests(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    from moeatlas.server import create_app

    client = TestClient(create_app(workspace))
    invalid = client.get("/api/hub/search", params={"kind": "model", "q": "x"})
    assert invalid.status_code == 400
    assert invalid.json()["detail"] == "invalid Hugging Face search request"

    monkeypatch.setattr(
        hub_service,
        "search_hub",
        lambda *args, **kwargs: (_ for _ in ()).throw(hub_service.HubSearchError()),
    )
    unavailable = client.get(
        "/api/hub/search", params={"kind": "dataset", "q": "org/data"}
    )
    assert unavailable.status_code == 503
    assert unavailable.json()["detail"] == "Hugging Face search is temporarily unavailable"
    assert "org/data" not in unavailable.text
