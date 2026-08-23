"""Contract tests for the packaged static frontend served by the local app."""

from __future__ import annotations

from pathlib import Path

import pytest

from moeatlas.services import initialize_workspace

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

STATIC_DIRECTORY = Path(__file__).resolve().parents[1] / "src" / "moeatlas" / "server" / "static"


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    path = tmp_path / "workspace"
    path.mkdir()
    initialize_workspace(path)
    return path


@pytest.fixture()
def client(workspace: Path):
    from fastapi.testclient import TestClient

    from moeatlas.server import create_app

    return TestClient(create_app(workspace))


def test_static_directory_is_packaged() -> None:
    assert (STATIC_DIRECTORY / "index.html").is_file()
    assert (STATIC_DIRECTORY / "styles.css").is_file()
    assert (STATIC_DIRECTORY / "app.js").is_file()
    assert (STATIC_DIRECTORY / "favicon.svg").is_file()
    assert tuple((STATIC_DIRECTORY / "assets").glob("index-*.js"))
    assert tuple((STATIC_DIRECTORY / "assets").glob("index-*.css"))


def test_root_serves_index_html(client) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    body = response.text
    assert "<!DOCTYPE html>" in body
    assert 'id="view"' in body
    assert "/app.js" in body


def test_root_serves_react_bundle(client) -> None:
    response = client.get("/")
    assert '<div id="root"></div>' in response.text
    script_names = tuple(path.name for path in (STATIC_DIRECTORY / "assets").glob("index-*.js"))
    stylesheet_names = tuple(
        path.name for path in (STATIC_DIRECTORY / "assets").glob("index-*.css")
    )
    assert script_names and stylesheet_names
    script = client.get(f"/assets/{script_names[0]}")
    stylesheet = client.get(f"/assets/{stylesheet_names[0]}")
    assert script.status_code == 200
    assert "javascript" in script.headers["content-type"]
    assert "createRoot" in script.text
    assert "Click cells to prepare an exact intervention target" in script.text
    assert "Build replicated study" in script.text
    assert stylesheet.status_code == 200
    assert stylesheet.headers["content-type"].startswith("text/css")
    assert "app-shell" in stylesheet.text


def test_static_assets_have_correct_content_types(client) -> None:
    styles = client.get("/styles.css")
    assert styles.status_code == 200
    assert styles.headers["content-type"].startswith("text/css")
    assert "--mono" in styles.text

    script = client.get("/app.js")
    assert script.status_code == 200
    assert "javascript" in script.headers["content-type"]
    assert "fetchJson" in script.text


def test_static_responses_disable_caching(client) -> None:
    for path in ("/", "/styles.css", "/app.js"):
        response = client.get(path)
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"


def test_unknown_static_path_is_not_found(client) -> None:
    response = client.get("/does-not-exist.txt")
    assert response.status_code == 404


def test_api_routes_are_unaffected_by_the_static_mount(client) -> None:
    health = client.get("/healthz")
    assert health.status_code == 200
    document = health.json()
    assert document["package_name"] == "MoEAtlas"

    workspace_snapshot = client.get("/api/workspace")
    assert workspace_snapshot.status_code == 200
    assert workspace_snapshot.json()["run_count"] == 0

    runs = client.get("/api/runs")
    assert runs.status_code == 200
    assert runs.json()["count"] == 0


def test_api_routes_do_not_gain_static_cache_headers(client) -> None:
    runs = client.get("/api/runs")
    assert runs.status_code == 200
    assert runs.headers.get("cache-control") != "no-store"


def test_index_references_no_external_resources() -> None:
    html = (STATIC_DIRECTORY / "index.html").read_text(encoding="utf-8")
    for forbidden in ("https://", "http://", "//cdn", "@import"):
        assert forbidden not in html, f"external resource marker {forbidden!r} found"


def test_app_js_references_no_external_resources() -> None:
    script = (STATIC_DIRECTORY / "app.js").read_text(encoding="utf-8")
    for forbidden in ("https://", "http://", "import(", "require("):
        assert forbidden not in script, f"external resource marker {forbidden!r} found"
