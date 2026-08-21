"""The synthetic example must run end-to-end in clean subprocesses."""

from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

EXAMPLE = ROOT / "examples" / "synthetic_workspace.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(EXAMPLE), *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_synthetic_example_builds_a_workspace_and_evaluates_retention(
    tmp_path: pathlib.Path,
) -> None:
    target = tmp_path / "workspace"
    process = _run(str(target))
    assert process.returncode == 0, process.stderr
    assert "registry holds 2 runs" in process.stdout
    assert "retained: 1 expired: 1" in process.stdout
    assert '"artifact_type":"moeatlas.retention_report"' in process.stdout
    assert target.is_dir() and any(target.iterdir())


def test_example_refuses_to_touch_an_existing_directory(
    tmp_path: pathlib.Path,
) -> None:
    target = tmp_path / "workspace"
    first = _run(str(target))
    assert first.returncode == 0, first.stderr
    second = _run(str(target))
    assert second.returncode == 2
    assert "refusing to touch an existing directory" in second.stdout


def test_example_usage_error_is_bounded() -> None:
    process = _run()
    assert process.returncode == 2
