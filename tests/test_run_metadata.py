"""Durable exact-run reconstruction metadata tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from moeatlas.services import (
    initialize_workspace,
    publish_run_metadata,
    read_run_metadata,
)

from .test_run_contracts import run_specification


def test_run_metadata_round_trips_idempotently(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    initialize_workspace(workspace)
    specification = run_specification(workspace=str(workspace))
    request = {
        "model_id": "org/model",
        "model_revision": "a" * 40,
        "dataset_id": "org/dataset",
        "dataset_revision": "b" * 40,
        "prompt_column": "text",
    }

    first = publish_run_metadata(workspace, specification, request)
    second = publish_run_metadata(workspace, specification, request)

    assert first == second
    document = read_run_metadata(workspace, specification.run_key)
    assert document["request"] == request
    assert document["specification"]["run_key"] == specification.run_key


def test_named_run_metadata_uses_the_readable_workspace_folder(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    initialize_workspace(workspace)
    specification = run_specification(
        workspace=str(workspace), run_name="v4-cybersecurity-baseline"
    )

    published = publish_run_metadata(workspace, specification, {"sample_cap": 4})

    assert published == workspace / "runs" / "v4-cybersecurity-baseline" / "run.json"
    assert read_run_metadata(workspace, specification.run_key)["request"] == {
        "sample_cap": 4
    }


def test_run_metadata_rejects_conflicting_reconstruction_request(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    initialize_workspace(workspace)
    specification = run_specification(workspace=str(workspace))
    publish_run_metadata(workspace, specification, {"sample_cap": 4})

    with pytest.raises(Exception, match="conflicts"):
        publish_run_metadata(workspace, specification, {"sample_cap": 8})
