"""Contract tests for the `moeatlas ui` local server command."""

from __future__ import annotations

from pathlib import Path

import pytest

import moeatlas.cli as cli_module
from moeatlas.cli import build_parser, main
from moeatlas.services import initialize_workspace


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    path = tmp_path / "workspace"
    path.mkdir()
    initialize_workspace(path)
    return path


@pytest.fixture()
def capture_serve(monkeypatch: pytest.MonkeyPatch):
    calls: list[dict[str, object]] = []

    def _fake_run(app: object, *, host: str, port: int) -> None:
        calls.append({"host": host, "port": port})

    monkeypatch.setattr(cli_module, "_run_ui_server", _fake_run)
    return calls


def test_parser_exposes_ui_with_loopback_defaults() -> None:
    parser = build_parser()
    subparsers = next(action for action in parser._actions if action.dest == "command")
    command = subparsers.choices["ui"]
    actions = {action.dest: action for action in command._actions}
    assert actions["host"].default == "127.0.0.1"
    assert actions["port"].default == "8000"
    assert {"allow_remote", "workspace"}.issubset(set(actions))


def test_ui_serves_the_bound_workspace_on_loopback(
    tmp_path: Path,
    workspace: Path,
    capture_serve,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["ui", str(workspace), "--port", "8123"])
    captured = capsys.readouterr()
    assert code == 0
    assert capture_serve == [{"host": "127.0.0.1", "port": 8123}]
    assert "http://127.0.0.1:8123" in captured.err


def test_non_loopback_host_requires_explicit_opt_in(
    tmp_path: Path,
    workspace: Path,
    capture_serve,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["ui", str(workspace), "--host", "0.0.0.0"])
    captured = capsys.readouterr()
    assert code == 2
    assert "non-loopback hosts require --allow-remote" in captured.err
    assert capture_serve == []

    allowed = main(["ui", str(workspace), "--host", "0.0.0.0", "--allow-remote"])
    assert allowed == 0
    assert capture_serve[-1]["host"] == "0.0.0.0"


def test_noncanonical_port_is_rejected(
    tmp_path: Path,
    workspace: Path,
    capture_serve,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["ui", str(workspace), "--port", "0"])
    captured = capsys.readouterr()
    assert code == 2
    assert "canonical positive decimal" in captured.err
    assert capture_serve == []
