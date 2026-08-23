"""Small in-process job control plane for the local research console.

The server is intentionally not a distributed queue.  A local process (or a
provider VM running the same process) needs a bounded way to launch one model
operation, expose progress, and request cooperative cancellation.  The worker
owns all model/storage work; this module only tracks its lifecycle and keeps
errors safe for the wire layer.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

JobWorker = Callable[[threading.Event, Callable[..., None]], Any]


@dataclass(frozen=True, slots=True)
class JobOutcome:
    """Worker return value with an explicit terminal state."""

    payload: Mapping[str, Any]
    state: str = "completed"

    def __post_init__(self) -> None:
        if not isinstance(self.payload, Mapping):
            raise TypeError("job payload must be a mapping")
        if self.state not in {"completed", "cancelled", "failed"}:
            raise ValueError("job outcome state is unsupported")
        object.__setattr__(self, "payload", dict(self.payload))


@dataclass(slots=True)
class _Job:
    job_id: str
    kind: str
    state: str = "queued"
    progress: dict[str, Any] = field(
        default_factory=lambda: {
            "stage": "queued",
            "completed": 0,
            "total": None,
            "message": "Waiting for a worker",
        }
    )
    result: dict[str, Any] | None = None
    error: str | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    optional_skip_event: threading.Event | None = None
    future: Future[Any] | None = None
    diagnostic_ref: dict[str, Any] = field(default_factory=dict)
    last_logged_progress: tuple[str, int, int | None] | None = None


class JobManager:
    """Thread-backed, bounded lifecycle state for discovery and run jobs."""

    def __init__(self, *, max_workers: int = 2, workspace: str | Path | None = None) -> None:
        if type(max_workers) is not int or isinstance(max_workers, bool) or max_workers <= 0:
            raise ValueError("max_workers must be a positive integer")
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="moeatlas")
        self._lock = threading.RLock()
        self._jobs: dict[str, _Job] = {}
        from .job_diagnostics import JobDiagnosticStore

        self._diagnostics = JobDiagnosticStore(workspace)

    def submit(
        self,
        kind: str,
        worker: JobWorker,
        *,
        optional_skip_event: threading.Event | None = None,
    ) -> str:
        if type(kind) is not str or not kind or len(kind) > 64:
            raise ValueError("job kind must be a bounded non-empty string")
        if not callable(worker):
            raise TypeError("job worker must be callable")
        if optional_skip_event is not None and not isinstance(optional_skip_event, threading.Event):
            raise TypeError("optional_skip_event must be a threading.Event or None")
        job_id = f"job:{uuid.uuid4().hex}"
        job = _Job(job_id=job_id, kind=kind, optional_skip_event=optional_skip_event)
        job.diagnostic_ref = self._diagnostics.start(job_id, kind).to_dict()
        with self._lock:
            self._jobs[job_id] = job
            job.future = self._executor.submit(self._run, job, worker)
        return job_id

    def _run(self, job: _Job, worker: JobWorker) -> None:
        with self._lock:
            job.state = "running"
            job.progress = {
                "stage": "starting",
                "completed": 0,
                "total": None,
                "message": "Worker started",
            }
        self._diagnostics.record(job.job_id, event="started", kind=job.kind, stage="starting")

        def report(
            *,
            stage: str,
            completed: int = 0,
            total: int | None = None,
            message: str = "",
        ) -> None:
            if type(stage) is not str or not stage:
                return
            if type(completed) is not int or isinstance(completed, bool) or completed < 0:
                completed = 0
            if total is not None and (
                type(total) is not int or isinstance(total, bool) or total < 0
            ):
                total = None
            with self._lock:
                if job.state in {"completed", "cancelled", "failed"}:
                    return
                job.progress = {
                    "stage": stage[:80],
                    "completed": completed,
                    "total": total,
                    "message": (message if isinstance(message, str) else "")[:240],
                }
                signature = (stage[:80], completed, total)
                # A row-level worker can report thousands of progress updates;
                # keep diagnostics bounded while preserving stage transitions
                # and periodic progress checkpoints for failed jobs.
                should_log = (
                    job.last_logged_progress is None
                    or job.last_logged_progress[0] != signature[0]
                    or completed == 0
                    or (completed > 0 and completed % 32 == 0)
                )
                if should_log:
                    job.last_logged_progress = signature
            if should_log:
                self._diagnostics.record(
                    job.job_id,
                    event="progress",
                    kind=job.kind,
                    stage=stage,
                    completed=completed,
                    total=total,
                )

        try:
            result = worker(job.cancel_event, report)
            outcome = result if isinstance(result, JobOutcome) else JobOutcome(result)
            with self._lock:
                job.result = dict(outcome.payload)
                job.state = outcome.state
                if job.state == "completed":
                    job.progress = {
                        "stage": "complete",
                        "completed": job.progress.get("total") or job.progress.get("completed", 0),
                        "total": job.progress.get("total"),
                        "message": "Worker completed",
                    }
                elif job.state == "cancelled":
                    job.progress = {
                        **job.progress,
                        "stage": "cancelled",
                        "message": "Cancellation was acknowledged",
                    }
            structured_failure = self._structured_failure(outcome)
            if structured_failure is None:
                self._diagnostics.record(
                    job.job_id,
                    event=job.state,
                    kind=job.kind,
                    stage=job.progress.get("stage"),
                    message=job.progress.get("message"),
                )
            else:
                self._diagnostics.record(
                    job.job_id,
                    event="failed",
                    kind=structured_failure["kind"],
                    stage=structured_failure["stage"],
                    message=structured_failure["message"],
                    exception_type="RowFailure",
                    exception_message=structured_failure["message"],
                )
        except BaseException as exc:  # workers must never take down the server thread
            with self._lock:
                failure_stage = job.progress.get("stage") or "failed"
                failure_error = self._safe_error(exc)
                failure_progress = {
                    **job.progress,
                    "stage": "failed",
                    "message": "Worker failed; inspect the server log for details",
                }
            try:
                from .process_worker import ProcessWorkerFailure

                isolated_failure = exc if isinstance(exc, ProcessWorkerFailure) else None
            except Exception:
                isolated_failure = None
            if isolated_failure is None:
                self._diagnostics.record(
                    job.job_id,
                    event="failed",
                    kind=job.kind,
                    stage=failure_stage,
                    exc=exc,
                )
            else:
                self._diagnostics.record(
                    job.job_id,
                    event="failed",
                    kind=job.kind,
                    stage=failure_stage,
                    exception_type=isolated_failure.error_type,
                    exception_message=isolated_failure.safe_message,
                    traceback_text=isolated_failure.traceback_text,
                )
            # Publish the failed wire state only after its diagnostic record is
            # visible. A client that observes ``state=failed`` can therefore
            # immediately read the typed failure instead of a stale
            # submitted/started entry.
            with self._lock:
                job.state = "failed"
                job.error = failure_error
                job.progress = failure_progress
        finally:
            # A failed optional-runtime load can leave reserved CUDA blocks in
            # the long-lived server process even when the worker has returned.
            # Keep this lazy so the model stack remains optional for the server.
            try:
                from ..runtime.memory import release_accelerator_memory

                release_accelerator_memory()
            except Exception:
                pass

    @staticmethod
    def _safe_error(exc: BaseException) -> str:
        # Known service errors already use fixed messages.  Runtime exception
        # text can contain local paths, prompts, or Hub payloads, so expose only
        # its type and a generic phrase on the wire.
        child_type = getattr(exc, "error_type", None)
        name = child_type if isinstance(child_type, str) and child_type else type(exc).__name__
        return f"job failed ({name})"

    @staticmethod
    def _structured_failure(outcome: JobOutcome) -> dict[str, str] | None:
        """Extract typed row-failure evidence from a returned failed outcome.

        ``execute_specification`` deliberately records row failures as normal
        evidence and returns a failed report when no row succeeds.  No Python
        exception reaches this job boundary in that case, so diagnostics must
        consume the bounded report payload explicitly.
        """

        if outcome.state != "failed":
            return None
        payload = outcome.payload
        summary = payload.get("failure_summary")
        evidence = payload.get("failure_evidence")
        if not isinstance(summary, Mapping):
            return None
        kind = summary.get("kind")
        stage = summary.get("stage")
        message = summary.get("message")
        count = summary.get("count")
        if not all(isinstance(value, str) and value for value in (kind, stage, message)):
            return None
        try:
            from ..services.run_engine import sanitize_failure_message

            safe_message = sanitize_failure_message(message)
        except Exception:
            safe_message = "row execution failed"
        first = evidence[0] if isinstance(evidence, list | tuple) and evidence else None
        if isinstance(first, Mapping):
            row_index = first.get("row_index")
            batch_index = first.get("batch_index")
            if type(row_index) is int and type(batch_index) is int:
                safe_message = f"row {row_index} (batch {batch_index}): {safe_message}"
        if type(count) is int and count > 1:
            safe_message = f"{count} row failures; first: {safe_message}"
        return {
            "kind": kind[:80],
            "stage": stage[:80],
            "message": safe_message[:500],
        }

    def snapshot(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return {
                "job_id": job.job_id,
                "kind": job.kind,
                "state": job.state,
                "progress": dict(job.progress),
                "result": dict(job.result) if job.result is not None else None,
                "error": job.error,
                "diagnostics": dict(self._diagnostics.reference(job.job_id).to_dict()),
            }

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            if job.state in {"completed", "cancelled", "failed"}:
                return False
            job.cancel_event.set()
            job.progress = {
                **job.progress,
                "stage": "cancelling",
                "message": "Cancellation requested; waiting for the current safe boundary",
            }
            self._diagnostics.record(
                job.job_id,
                event="cancel_requested",
                kind=job.kind,
                stage="cancelling",
            )
            return True

    def skip_optional(self, job_id: str) -> bool:
        """Skip a running optional phase without cancelling the parent job."""

        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.optional_skip_event is None:
                return False
            if job.state in {"completed", "cancelled", "failed"}:
                return False
            if job.progress.get("stage") != "overhead":
                return False
            job.optional_skip_event.set()
            job.progress = {
                **job.progress,
                "message": "Optional phase skip requested; waiting for a safe boundary",
            }
            self._diagnostics.record(
                job.job_id,
                event="optional_phase_skip_requested",
                kind=job.kind,
                stage="overhead",
            )
            return True

    def diagnostics(self, job_id: str) -> dict[str, Any] | None:
        """Return persisted diagnostics only for a known in-process job."""

        with self._lock:
            if job_id not in self._jobs:
                return None
            job = self._jobs[job_id]
            result = self._diagnostics.read(job_id)
            return {
                "job_id": job.job_id,
                "kind": job.kind,
                "state": job.state,
                **result,
            }

    def result(self, job_id: str) -> dict[str, Any] | None:
        snapshot = self.snapshot(job_id)
        return None if snapshot is None else snapshot["result"]

    def shutdown(self, *, wait: bool = False) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=not wait)


__all__ = ["JobManager", "JobOutcome"]
