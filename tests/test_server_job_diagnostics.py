"""Contracts for bounded server-job diagnostic evidence."""

from __future__ import annotations

import stat
import time
from pathlib import Path

from fastapi.testclient import TestClient

from moeatlas.server import create_app
from moeatlas.server.job_diagnostics import JobDiagnosticStore
from moeatlas.server.jobs import JobManager
from moeatlas.services import initialize_workspace


def _wait_for_terminal(client: TestClient, job_id: str) -> dict:
    for _ in range(100):
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        document = response.json()
        if document["state"] not in {"queued", "running"}:
            return document
        time.sleep(0.005)
    raise AssertionError("job did not reach a terminal state")


def test_store_persists_exception_chain_with_redaction_and_private_permissions(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = JobDiagnosticStore(workspace)
    job_id = "job:" + "a" * 32
    reference = store.start(job_id, "discovery")
    assert reference.available is True

    try:
        try:
            raise ValueError(
                "prompt='private prompt' api_key=sk-12345678901234567890 "
                "Authorization: Bearer eyJhbGciOiJIUzI1Ni.secret.signature "
                f"path={workspace / 'checkpoint.bin'}"
            )
        except ValueError as cause:
            raise RuntimeError("outer failure") from cause
    except RuntimeError as exc:
        store.record(job_id, event="failed", stage="load", exc=exc)

    path = workspace / "logs" / "jobs" / ("a" * 32 + ".jsonl")
    assert path.is_file()
    assert stat.S_IMODE(path.stat().st_mode) & 0o077 == 0
    raw = path.read_text(encoding="utf-8")
    assert "private prompt" not in raw
    assert "sk-12345678901234567890" not in raw
    assert "eyJhbGciOiJIUzI1Ni" not in raw
    assert str(workspace) not in raw
    document = store.read(job_id)
    assert document["available"] is True
    assert document["entry_count"] == 2
    failure = document["entries"][-1]
    assert failure["exception_type"] == "RuntimeError"
    assert "ValueError" in failure["traceback"]
    assert "outer failure" in failure["traceback"]


def test_store_enforces_entry_and_byte_budgets(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = JobDiagnosticStore(workspace, max_bytes=4096, max_entries=3)
    job_id = "job:" + "b" * 32
    store.start(job_id, "run")
    for index in range(20):
        store.record(job_id, event="progress", stage=f"stage-{index}")
    document = store.read(job_id)
    assert document["entry_count"] <= 3
    assert document["truncated"] is True
    assert (workspace / "logs" / "jobs" / ("b" * 32 + ".jsonl")).stat().st_size <= 4096


def test_job_manager_keeps_wire_error_safe_and_persists_diagnostics(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manager = JobManager(max_workers=1, workspace=workspace)

    def worker(_cancel, _report):
        raise RuntimeError("prompt='do not expose' hf_token=hf_secret_123456789")

    job_id = manager.submit("run", worker)
    try:
        for _ in range(100):
            snapshot = manager.snapshot(job_id)
            assert snapshot is not None
            if snapshot["state"] == "failed":
                break
            time.sleep(0.005)
        assert snapshot["state"] == "failed"
        assert snapshot["error"] == "job failed (RuntimeError)"
        assert snapshot["diagnostics"]["available"] is True
        diagnostics = manager.diagnostics(job_id)
        assert diagnostics is not None
        assert diagnostics["entries"][-1]["exception_type"] == "RuntimeError"
        serialized = str(diagnostics)
        assert "do not expose" not in serialized
        assert "hf_secret_123456789" not in serialized
    finally:
        manager.shutdown(wait=True)


def test_diagnostics_endpoint_is_server_scoped_and_typed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    initialize_workspace(workspace)

    import moeatlas.server.app as app_module

    def fake_discovery(payload, *, cancel, report_progress):
        del payload, cancel, report_progress
        raise ValueError("prompt='not persisted'")

    monkeypatch.setattr(app_module, "_discovery_worker", fake_discovery)
    client = TestClient(create_app(workspace))
    created = client.post("/api/discovery", json={"model_id": "org/model"})
    assert created.status_code == 202
    job_id = created.json()["job_id"]
    terminal = _wait_for_terminal(client, job_id)
    assert terminal["state"] == "failed"
    assert terminal["error"] == "job failed (ValueError)"
    reference = terminal["diagnostics"]
    assert reference["endpoint"] == f"/api/jobs/{job_id}/diagnostics"
    assert reference["available"] is True

    response = client.get(reference["endpoint"])
    assert response.status_code == 200
    document = response.json()
    assert document["job_id"] == job_id
    assert document["state"] == "failed"
    assert document["entries"][-1]["exception_type"] == "ValueError"
    assert "not persisted" not in response.text
    assert client.get("/api/jobs/job:" + "f" * 32 + "/diagnostics").status_code == 404
