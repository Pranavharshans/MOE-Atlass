"""Model-free contracts for sequential multi-dataset run groups."""

from __future__ import annotations

import json
from pathlib import Path

from moeatlas.server.jobs import JobOutcome
from moeatlas.server.workers import _run_group_worker
from moeatlas.services import initialize_workspace
from moeatlas.services.run_groups import (
    child_run_name,
    dataset_slug,
    list_run_groups,
    publish_run_group,
)


class _Cancel:
    def is_set(self) -> bool:
        return False


def test_dataset_slug_and_child_name_are_stable_and_safe() -> None:
    slug = dataset_slug("cais/mmlu", "computer_security", 0)

    assert slug == "01-mmlu-computer_security"
    assert child_run_name("cyber-study", slug) == "cyber-study--01-mmlu-computer_security"


def test_publish_and_list_group_with_dataset_child_pointers(tmp_path: Path) -> None:
    initialize_workspace(tmp_path)
    children = [
        {
            "slug": "01-mmlu-cyber",
            "dataset_id": "cais/mmlu",
            "dataset_config": "computer_security",
            "child_run_name": "study--01-mmlu-cyber",
            "run_key": "run:abc",
            "state": "completed",
        }
    ]

    target = publish_run_group(tmp_path, "study", children, state="completed")

    assert target == tmp_path / "runs" / "study" / "group.json"
    assert (tmp_path / "runs" / "study" / "datasets" / "01-mmlu-cyber" / "run.json").is_file()
    assert list_run_groups(tmp_path)[0]["children"] == children


def test_group_worker_runs_children_sequentially_and_persists_results(tmp_path: Path) -> None:
    initialize_workspace(tmp_path)
    calls: list[dict[str, object]] = []
    progress: list[dict[str, object]] = []

    def execute_child(payload: dict[str, object], report_progress: object) -> JobOutcome:
        calls.append(payload)
        report_progress(stage="execute", completed=1, total=1, message="row complete")
        return JobOutcome(
            {"status": "completed", "run_key": f"run:{len(calls)}"},
            "completed",
        )

    outcome = _run_group_worker(
        str(tmp_path),
        {
            "run_name": "study",
            "model_id": "org/model",
            "datasets": [
                {
                    "dataset_id": "cais/mmlu",
                    "dataset_config": "computer_security",
                },
                {
                    "dataset_id": "cais/mmlu",
                    "dataset_config": "college_computer_science",
                },
            ],
        },
        cancel=_Cancel(),
        report_progress=lambda **fields: progress.append(fields),
        execute_child=execute_child,
    )

    assert outcome.state == "completed"
    assert [call["run_name"] for call in calls] == [
        "study--01-mmlu-computer_security",
        "study--02-mmlu-college_computer_science",
    ]
    assert [child["run_key"] for child in outcome.payload["children"]] == ["run:1", "run:2"]
    document = json.loads((tmp_path / "runs" / "study" / "group.json").read_text())
    assert document["state"] == "completed"
    assert progress[-1]["completed"] == 2


def test_group_worker_preserves_child_process_failure_details(tmp_path: Path) -> None:
    initialize_workspace(tmp_path)

    class DetailedFailure(RuntimeError):
        error_type = "RoutingRunInventoryError"
        safe_message = "routing run inventory failed at budget"

    def execute_child(_payload: object, _report_progress: object) -> JobOutcome:
        raise DetailedFailure

    outcome = _run_group_worker(
        str(tmp_path),
        {
            "run_name": "study",
            "model_id": "org/model",
            "datasets": [
                {"dataset_id": "cais/mmlu", "dataset_config": "computer_security"},
                {"dataset_id": "cais/mmlu", "dataset_config": "college_computer_science"},
            ],
        },
        cancel=_Cancel(),
        report_progress=lambda **_fields: None,
        execute_child=execute_child,
    )

    assert outcome.state == "failed"
    assert outcome.payload["children"][0]["error_type"] == "RoutingRunInventoryError"
    assert (
        outcome.payload["children"][0]["error_message"]
        == "routing run inventory failed at budget"
    )
