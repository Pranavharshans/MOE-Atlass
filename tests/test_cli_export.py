"""Contract tests for the `moeatlas export` command."""

from __future__ import annotations

from pathlib import Path

import pytest

from moeatlas.cli import build_parser, main


def _args(workspace: Path, run_key: str, output: Path) -> list[str]:
    return ["export", str(workspace), run_key, "--output", str(output)]


def test_parser_exposes_export_with_format_choices() -> None:
    parser = build_parser()
    subparsers = next(action for action in parser._actions if action.dest == "command")
    command = subparsers.choices["export"]
    dests = {action.dest for action in command._actions}
    assert {"workspace", "run_key", "format", "output"}.issubset(dests)
    format_action = next(action for action in command._actions if action.dest == "format")
    assert format_action.choices == ("bundle",)
    assert format_action.default == "bundle"


def test_export_writes_a_bundle_and_prints_the_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from .test_cli_routing_runs import _fixture

    workspace, run_key = _fixture(tmp_path)
    destination = tmp_path / "bundle"
    code = main(_args(workspace, run_key, destination))
    captured = capsys.readouterr()
    assert code == 0
    assert f"exported run {run_key}" in captured.out
    assert "manifest sha256:" in captured.out
    assert destination.is_dir()
    assert any(destination.iterdir())


def test_unknown_run_key_is_rejected_without_echoing_details(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from .test_cli_routing_runs import _fixture

    workspace, _real_key = _fixture(tmp_path)
    destination = tmp_path / "bundle"
    code = main(_args(workspace, "run:" + "f" * 64, destination))
    captured = capsys.readouterr()
    assert code == 2
    # Detail-bearing errors collapse to the fixed generic message; the
    # unknown key is never echoed back.
    assert captured.err == "moeatlas export: run export failed\n"


def test_nonempty_destination_is_rejected(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from .test_cli_routing_runs import _fixture

    workspace, run_key = _fixture(tmp_path)
    destination = tmp_path / "bundle"
    destination.mkdir()
    (destination / "occupied.txt").write_text("x", encoding="utf-8")
    code = main(_args(workspace, run_key, destination))
    captured = capsys.readouterr()
    assert code == 2
    assert "moeatlas export:" in captured.err


def test_noncanonical_budget_is_rejected(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from .test_cli_routing_runs import _fixture

    workspace, run_key = _fixture(tmp_path)
    destination = tmp_path / "bundle"
    argv = _args(workspace, run_key, destination) + ["--max-event-rows", "-3"]
    code = main(argv)
    captured = capsys.readouterr()
    assert code == 2
    assert "canonical positive decimal" in captured.err
