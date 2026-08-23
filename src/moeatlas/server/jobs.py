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


class JobManager:
    """Thread-backed, bounded lifecycle state for discovery and run jobs."""

    def __init__(self, *, max_workers: int = 2) -> None:
        if type(max_workers) is not int or isinstance(max_workers, bool) or max_workers <= 0:
            raise ValueError("max_workers must be a positive integer")
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="moeatlas")
        self._lock = threading.RLock()
        self._jobs: dict[str, _Job] = {}

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
        except BaseException as exc:  # workers must never take down the server thread
            with self._lock:
                job.state = "failed"
                job.error = self._safe_error(exc)
                job.progress = {
                    **job.progress,
                    "stage": "failed",
                    "message": "Worker failed; inspect the server log for details",
                }

    @staticmethod
    def _safe_error(exc: BaseException) -> str:
        # Known service errors already use fixed messages.  Runtime exception
        # text can contain local paths, prompts, or Hub payloads, so expose only
        # its type and a generic phrase on the wire.
        name = type(exc).__name__
        return f"job failed ({name})"

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
            return True

    def result(self, job_id: str) -> dict[str, Any] | None:
        snapshot = self.snapshot(job_id)
        return None if snapshot is None else snapshot["result"]

    def shutdown(self, *, wait: bool = False) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=not wait)


__all__ = ["JobManager", "JobOutcome"]
