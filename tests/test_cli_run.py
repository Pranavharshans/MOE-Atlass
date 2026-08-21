"""Contract tests for the `moeatlas run` command."""

from __future__ import annotations

from pathlib import Path

import pytest

import moeatlas.cli as cli_module
from moeatlas.cli import build_parser, main
from moeatlas.services import initialize_workspace, query_runs

from .test_cli_scan import _loading_plan, _write_plan


def _echo_executor(*, row_index: int, batch_index: int, values):
    return {"echo": values.get("prompt"), "row": row_index}


@pytest.fixture()
def fake_executor(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cli_module, "_load_executor", lambda name: _echo_executor)


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    path = tmp_path / "workspace"
    path.mkdir()
    initialize_workspace(path)
    return path


def _plan_file(tmp_path: Path) -> Path:
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path, _loading_plan())
    return plan_path


def test_parser_exposes_run_with_required_flags() -> None:
    parser = build_parser()
    subparsers = next(action for action in parser._actions if action.dest == "command")
    command = subparsers.choices["run"]
    required = {
        action.dest
        for action in command._actions
        if getattr(action, "required", False)
    }
    assert {"loading_plan", "executor"}.issubset(required)
    dests = {action.dest for action in command._actions}
    assert {"prompt", "dataset", "at", "checkpoint_directory", "resume_from"}.issubset(dests)


def test_prompt_run_completes_and_publishes(
    tmp_path: Path,
    workspace: Path,
    fake_executor,
) -> None:
    plan_path = _plan_file(tmp_path)
    code = main(
        [
            "run",
            str(workspace),
            "--loading-plan",
            str(plan_path),
            "--prompt",
            "hello world",
            "--executor",
            "fake-echo",
            "--at",
            "2026-08-21T00:00:00Z",
        ]
    )
    assert code == 0
    entries = query_runs(workspace)
    assert len(entries) == 1
    assert entries[0].state == "completed"


def test_dataset_run_completes_over_local_rows(
    tmp_path: Path,
    workspace: Path,
    fake_executor,
) -> None:
    (workspace / "rows.jsonl").write_text(
        '{"prompt": "a"}\n{"prompt": "b"}\n', encoding="utf-8"
    )
    descriptor = tmp_path / "dataset.json"
    descriptor.write_text(
        '{"format": "jsonl", "location": "rows.jsonl", "batch_size": 1}',
        encoding="utf-8",
    )
    plan_path = _plan_file(tmp_path)
    code = main(
        [
            "run",
            str(workspace),
            "--loading-plan",
            str(plan_path),
            "--dataset",
            str(descriptor),
            "--executor",
            "fake-echo",
            "--at",
            "2026-08-21T00:00:00Z",
        ]
    )
    assert code == 0
    entries = query_runs(workspace)
    assert len(entries) == 1


def test_checkpoint_directory_is_reported(
    tmp_path: Path,
    workspace: Path,
    fake_executor,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    plan_path = _plan_file(tmp_path)
    code = main(
        [
            "run",
            str(workspace),
            "--loading-plan",
            str(plan_path),
            "--prompt",
            "hello",
            "--executor",
            "fake-echo",
            "--checkpoint-directory",
            str(checkpoints),
            "--at",
            "2026-08-21T00:00:00Z",
        ]
    )
    assert code == 0
    assert "checkpoint:" in capsys.readouterr().out
    assert list(checkpoints.iterdir())


def test_exactly_one_input_form_is_enforced(
    tmp_path: Path,
    workspace: Path,
    fake_executor,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan_path = _plan_file(tmp_path)
    both = main(
        [
            "run",
            str(workspace),
            "--loading-plan",
            str(plan_path),
            "--prompt",
            "hello",
            "--dataset",
            "descriptor.json",
            "--executor",
            "fake-echo",
        ]
    )
    assert both == 2
    assert "exactly one of --prompt or --dataset" in capsys.readouterr().err

    neither = main(
        [
            "run",
            str(workspace),
            "--loading-plan",
            str(plan_path),
            "--executor",
            "fake-echo",
        ]
    )
    assert neither == 2


def test_unregistered_executor_is_rejected(
    tmp_path: Path,
    workspace: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan_path = _plan_file(tmp_path)
    code = main(
        [
            "run",
            str(workspace),
            "--loading-plan",
            str(plan_path),
            "--prompt",
            "hello",
            "--executor",
            "no-such-executor",
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert "executor plugin is not registered" in captured.err


def test_uninitialized_workspace_is_rejected(
    tmp_path: Path,
    fake_executor,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bare = tmp_path / "bare"
    bare.mkdir()
    plan_path = _plan_file(tmp_path)
    code = main(
        [
            "run",
            str(bare),
            "--loading-plan",
            str(plan_path),
            "--prompt",
            "hello",
            "--executor",
            "fake-echo",
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert "moeatlas run:" in captured.err


def test_noncanonical_budgets_are_rejected(
    tmp_path: Path,
    workspace: Path,
    fake_executor,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan_path = _plan_file(tmp_path)
    code = main(
        [
            "run",
            str(workspace),
            "--loading-plan",
            str(plan_path),
            "--prompt",
            "hello",
            "--executor",
            "fake-echo",
            "--max-rows",
            "0",
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert "canonical positive decimal" in captured.err
