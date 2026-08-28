"""Model-free contracts for the live server control plane."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from moeatlas.server import create_app
from moeatlas.server.jobs import JobOutcome
from moeatlas.services import initialize_workspace
from moeatlas.services.model_resolution import (
    resolve_huggingface_plan,
    resolve_huggingface_revision,
    resolve_huggingface_revision_metadata,
)


def isolated_process_crash(_payload, report):
    report(stage="load", completed=0, total=1, message="loading synthetic model")
    os._exit(44)


def isolated_process_success(payload, report):
    report(stage="execute", completed=1, total=1, message="synthetic capture")
    return {"state": "completed", "payload": {"status": "completed", **payload}}


def test_immutable_hub_revision_builds_a_loader_ready_plan() -> None:
    commit = "a" * 40
    plan = resolve_huggingface_plan("org/model", commit)
    assert plan.resolution is not None
    assert plan.resolution.resolved_model_revision == commit
    assert plan.resolution.resolved_tokenizer_revision == commit
    assert plan.source.allow_downloads is True


def test_offline_hub_plan_rejects_branch_without_contacting_the_hub(monkeypatch) -> None:
    import moeatlas.services.model_resolution as resolution

    def fail_request(*args: object, **kwargs: object):
        raise AssertionError("offline model resolution must not contact the Hub")

    monkeypatch.setattr(resolution, "urlopen", fail_request)
    with pytest.raises(resolution.ModelResolutionError, match="offline loading"):
        resolution.resolve_huggingface_plan(
            "org/model",
            "main",
            allow_downloads=False,
        )


def test_hub_branch_resolution_uses_bounded_fixed_metadata_request(monkeypatch) -> None:
    import moeatlas.services.model_resolution as resolution

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self, limit: int) -> bytes:
            assert limit > 40
            return ('{"sha":"' + "b" * 40 + '"}').encode()

    seen: dict[str, object] = {}

    def fake_urlopen(request, *, timeout):
        seen["url"] = request.full_url
        seen["timeout"] = timeout
        return Response()

    monkeypatch.setattr(resolution, "urlopen", fake_urlopen)
    commit, source = resolve_huggingface_revision("org/data", "main", kind="datasets")
    assert commit == "b" * 40
    assert source.endswith("/datasets/org/data?revision=main")
    assert seen["timeout"] == 20.0


def test_hub_resolution_retains_bounded_repository_size(monkeypatch) -> None:
    import moeatlas.services.model_resolution as resolution

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self, limit: int) -> bytes:
            del limit
            return ('{"sha":"' + "c" * 40 + '","usedStorage":123456789}').encode()

    monkeypatch.setattr(resolution, "urlopen", lambda request, timeout: Response())

    metadata = resolve_huggingface_revision_metadata("org/model", "main")
    assert metadata.resolved_revision == "c" * 40
    assert metadata.repository_size_bytes == 123456789


def test_json_job_routes_and_intervention_recipe_are_live(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    initialize_workspace(workspace)

    import moeatlas.server.app as app_module

    def fake_discovery(payload, *, cancel, report_progress):
        report_progress(stage="scan", completed=1, total=1, message="synthetic")
        return JobOutcome(
            {"status": "available", "report": {"facts": {"expert_count": 4}}},
            "completed",
        )

    monkeypatch.setattr(app_module, "_discovery_worker", fake_discovery)
    client = TestClient(create_app(workspace, isolate_model_workers=False))
    created = client.post("/api/discovery", json={"model_id": "org/model"})
    assert created.status_code == 202
    job_id = created.json()["job_id"]
    for _ in range(50):
        status = client.get(f"/api/jobs/{job_id}")
        if status.json()["state"] not in {"queued", "running"}:
            break
        time.sleep(0.01)
    assert status.json()["state"] == "completed"
    assert status.json()["result"]["report"]["facts"]["expert_count"] == 4

    recipe = client.post(
        "/api/interventions/recipes",
        json={"operation": "ablate", "targets": ["layer:0/expert:1"]},
    )
    assert recipe.status_code == 200
    assert recipe.json()["status"] == "prepared"
    assert recipe.json()["fingerprint"].startswith("sha256:")


def test_real_intervention_route_reconstructs_a_completed_baseline(
    tmp_path: Path, monkeypatch
) -> None:
    from moeatlas.runs.specs import PromptInputSpec
    from moeatlas.services import (
        execute_specification,
        publish_run_metadata,
        publish_run_report,
    )
    from tests.test_interventions_transformers import _model
    from tests.test_run_contracts import data_provenance, run_specification
    from tests.test_runtime_generic_expert_capture import scan_report

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    initialize_workspace(workspace)
    specification = run_specification(
        workspace=str(workspace),
        data=data_provenance(input=PromptInputSpec(text="baseline")),
    )
    report = execute_specification(
        specification,
        executor=lambda **_: {
            "output_digest": "sha256:baseline",
            "forward_ms": 1.0,
        },
        base_directory=workspace,
        checkpoint_directory=workspace / "checkpoints",
    )
    publish_run_report(workspace, report)
    publish_run_metadata(
        workspace,
        specification,
        {
            "model_id": "org/model",
            "model_revision": "a" * 40,
            "dataset_id": "org/data",
            "dataset_revision": "b" * 40,
            "prompt_column": "text",
        },
    )
    discovery_root = workspace / "discoveries"
    discovery_root.mkdir()
    (discovery_root / f"{specification.run_key}.json").write_text(
        scan_report(_model()).to_json(), encoding="utf-8"
    )

    import moeatlas.server.app as app_module

    def fake_run(workspace, payload, *, cancel, report_progress, **kwargs):
        del workspace, cancel, kwargs
        assert payload["run_name"] == "baseline-ablation"
        assert payload["baseline_run_key"] == specification.run_key
        assert payload["model_revision"] == "a" * 40
        assert payload["intervention"]["operation"] == "ablate"
        report_progress(stage="execute", completed=1, total=1, message="intervened")
        return JobOutcome({"status": "completed", "run_key": "run:" + "9" * 64}, "completed")

    monkeypatch.setattr(app_module, "_run_worker", fake_run)
    client = TestClient(create_app(workspace, isolate_model_workers=False))
    targets = client.get(f"/api/runs/{specification.run_key}/intervention-targets")
    assert targets.status_code == 200
    assert targets.json()["status"] == "available"
    operations = {
        item["operation"]: item for item in targets.json()["capability"]["operation_capabilities"]
    }
    assert operations["zero_contribution"]["status"] == "available"
    assert operations["exclude_and_renormalize"]["status"] == "not_implemented"
    selected = targets.json()["targets"][0]["label"]

    created = client.post(
        "/api/interventions/start",
        json={
            "run_name": "baseline-ablation",
            "baseline_run_key": specification.run_key,
            "operation": "ablate",
            "targets": [selected],
        },
    )
    assert created.status_code == 202
    job_id = created.json()["job_id"]
    for _ in range(50):
        status = client.get(f"/api/jobs/{job_id}")
        if status.json()["state"] not in {"queued", "running"}:
            break
        time.sleep(0.01)
    assert status.json()["state"] == "completed"
    assert status.json()["result"]["run_key"] == "run:" + "9" * 64


def test_optional_overhead_can_be_skipped_without_cancelling_capture(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    initialize_workspace(workspace)

    import moeatlas.server.app as app_module

    def fake_run(
        workspace,
        payload,
        *,
        cancel,
        report_progress,
        resume_from=None,
        skip_overhead=None,
    ):
        del workspace, resume_from
        assert payload["dataset_seed"] == 20260828
        assert skip_overhead is not None
        report_progress(stage="overhead", completed=0, total=1, message="synthetic overhead")
        while not skip_overhead():
            if cancel.is_set():
                return JobOutcome({"status": "cancelled"}, "cancelled")
            time.sleep(0.005)
        report_progress(stage="execute", completed=1, total=1, message="synthetic capture")
        return JobOutcome(
            {"status": "completed", "capture_overhead": {"status": "skipped"}},
            "completed",
        )

    monkeypatch.setattr(app_module, "_run_worker", fake_run)
    client = TestClient(create_app(workspace, isolate_model_workers=False))
    created = client.post(
        "/api/runs/start",
        json={
            "run_name": "overhead-test",
            "model_id": "org/model",
            "dataset_id": "org/data",
            "dataset_seed": 20260828,
            "measure_capture_overhead": True,
        },
    )
    assert created.status_code == 202
    job_id = created.json()["job_id"]
    for _ in range(50):
        status = client.get(f"/api/jobs/{job_id}")
        if status.json()["progress"]["stage"] == "overhead":
            break
        time.sleep(0.01)
    skipped = client.post(f"/api/jobs/{job_id}/skip-overhead")
    assert skipped.status_code == 200
    for _ in range(100):
        status = client.get(f"/api/jobs/{job_id}")
        if status.json()["state"] not in {"queued", "running"}:
            break
        time.sleep(0.01)
    assert status.json()["state"] == "completed"
    assert status.json()["result"]["capture_overhead"]["status"] == "skipped"


def test_isolated_model_crash_fails_one_job_and_next_job_still_runs(tmp_path: Path) -> None:
    from moeatlas.server.app import _isolated_job_worker
    from moeatlas.server.jobs import JobManager

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manager = JobManager(max_workers=1, workspace=workspace)

    def submit(entry, payload):
        return manager.submit(
            "discovery",
            lambda cancel, progress: _isolated_job_worker(
                entry,
                payload,
                cancel=cancel,
                report_progress=progress,
            ),
        )

    def terminal(job_id: str) -> dict[str, object]:
        for _ in range(300):
            snapshot = manager.snapshot(job_id)
            assert snapshot is not None
            if snapshot["state"] not in {"queued", "running"}:
                return snapshot
            time.sleep(0.01)
        raise AssertionError("isolated job did not finish")

    try:
        crashed = terminal(submit(isolated_process_crash, {}))
        assert crashed["state"] == "failed"
        assert crashed["error"] == "job failed (ChildProcessExit)"
        diagnostics = manager.diagnostics(str(crashed["job_id"]))
        assert diagnostics is not None
        assert diagnostics["entries"][-1]["stage"] == "load"
        assert diagnostics["entries"][-1]["exception_type"] == "ChildProcessExit"

        completed = terminal(submit(isolated_process_success, {"next": True}))
        assert completed["state"] == "completed"
        assert completed["result"] == {"status": "completed", "next": True}
    finally:
        manager.shutdown(wait=True)


def test_export_honors_persisted_privacy_policy(tmp_path: Path) -> None:
    from moeatlas.runs.specs import PrivacyPolicy, RunSpecification
    from moeatlas.server.workers import _publish_run_policy
    from moeatlas.services import register_run

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    initialize_workspace(workspace)
    from tests.test_run_contracts import data_provenance, model_provenance

    specification = RunSpecification(
        model=model_provenance(),
        data=data_provenance(),
        privacy=PrivacyPolicy(allow_export=False),
    )
    register_run(workspace, specification)
    _publish_run_policy(workspace, specification.run_key, specification.privacy)
    response = TestClient(create_app(workspace)).get(f"/api/runs/{specification.run_key}/export")
    assert response.status_code == 403
    assert response.json()["detail"] == "run export is disabled by privacy policy"
