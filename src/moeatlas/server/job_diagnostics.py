"""Bounded, workspace-owned diagnostics for local server jobs.

The live server deliberately keeps exception details out of its normal job
response.  This module provides a second, bounded evidence channel for
debugging failed jobs.  It stores JSONL records below the bound workspace and
returns only records for a job that is already known to the in-process job
manager.  Tracebacks are formatted without local-variable capture and pass
through conservative redaction before they are persisted.
"""

from __future__ import annotations

import json
import re
import threading
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_LOG_DIRECTORY = "logs"
_JOB_LOG_DIRECTORY = "jobs"
_MAX_LOG_BYTES = 512_000
_MAX_ENTRY_BYTES = 32_768
_MAX_ENTRIES = 256
_MAX_TEXT = 8_192

# Credentials that can occur in Hub/HTTP/runtime errors.  The expression is
# intentionally key based; arbitrary exception text is retained (with paths
# and common prompt fields redacted) so the diagnostic remains useful.
_SECRET_ASSIGNMENT = re.compile(
    r"(?is)\b(authorization|bearer|api[_-]?key|access[_-]?token|refresh[_-]?token|"
    r"hf[_-]?token|huggingface[_-]?hub[_-]?token|password|passwd|secret)\b"
    r"\s*[:=]\s*(?:bearer\s+)?([^\s,;]+)"
)
_BEARER_CREDENTIAL = re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/-]+=*")
_TOKEN_LITERAL = re.compile(
    r"(?i)\b(?:hf_[a-z0-9]{8,}|sk-[a-z0-9]{16,}|xox[baprs]-[a-z0-9-]{8,})\b"
)
_PATH_LITERAL = re.compile(
    r"(?<![a-z0-9_])(?:/(?:users|home|workspace|private|tmp|var|opt|root|mnt|srv)"
    r"/[^\s'\"`;,)]*|[a-z]:[\\/][^\s'\"`;,)]*)",
    re.IGNORECASE,
)
_PROMPT_FIELD = re.compile(
    r"(?is)\b(prompt|input|query|content|completion|token_text)\b"
    r"\s*[:=]\s*(\"[^\"]*\"|'[^']*'|[^\n,;}]+)"
)


def _redact_text(value: object, *, workspace: Path | None = None) -> str:
    """Bound and redact human-readable diagnostics without importing runtime code."""

    if isinstance(value, str):
        text = value
    else:
        try:
            text = str(value)
        except Exception:
            text = "<unavailable diagnostic text>"
    if workspace is not None:
        try:
            text = text.replace(str(workspace.resolve()), "<workspace>")
        except OSError:
            text = text.replace(str(workspace), "<workspace>")
    text = _SECRET_ASSIGNMENT.sub(lambda match: f"{match.group(1)}=<redacted>", text)
    text = _BEARER_CREDENTIAL.sub("Bearer <redacted>", text)
    text = _TOKEN_LITERAL.sub("<redacted-token>", text)
    text = _PROMPT_FIELD.sub(lambda match: f"{match.group(1)}=<redacted>", text)
    text = _PATH_LITERAL.sub("<path>", text)
    return text[:_MAX_TEXT]


def _exception_document(exc: BaseException, *, workspace: Path | None) -> dict[str, str]:
    """Serialize an exception chain with locals excluded from the traceback."""

    # TracebackException(capture_locals=False) preserves chained exceptions
    # while ensuring prompt-bearing local variables never enter the record.
    try:
        formatted = "".join(
            traceback.TracebackException.from_exception(exc, capture_locals=False).format()
        )
    except Exception:
        formatted = f"{type(exc).__name__}: <traceback unavailable>"
    try:
        message = str(exc)
    except Exception:
        message = "<exception message unavailable>"
    return {
        "type": type(exc).__name__[:200],
        "message": _redact_text(message, workspace=workspace),
        "traceback": _redact_text(formatted, workspace=workspace),
    }


@dataclass(frozen=True, slots=True)
class DiagnosticReference:
    """Safe UI-facing pointer to a job's diagnostic endpoint."""

    endpoint: str
    available: bool
    entry_count: int = 0
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "available": self.available,
            "entry_count": self.entry_count,
            "truncated": self.truncated,
        }


class JobDiagnosticStore:
    """Persist bounded JSONL diagnostics under one existing workspace.

    The store is best effort: diagnostics must never turn a model/job failure
    into a server failure.  Invalid or symlinked workspace paths disable the
    store rather than widening the filesystem surface.
    """

    def __init__(
        self,
        workspace: str | Path | None,
        *,
        max_bytes: int = _MAX_LOG_BYTES,
        max_entries: int = _MAX_ENTRIES,
    ) -> None:
        if max_bytes <= 0 or max_bytes > _MAX_LOG_BYTES:
            raise ValueError("max_bytes is outside the diagnostic budget")
        if max_entries <= 0 or max_entries > _MAX_ENTRIES:
            raise ValueError("max_entries is outside the diagnostic budget")
        self._workspace = Path(workspace) if workspace is not None else None
        self._max_bytes = max_bytes
        self._max_entries = max_entries
        self._lock = threading.RLock()
        self._counts: dict[str, int] = {}
        self._truncated: set[str] = set()
        self._disabled = False

    @property
    def enabled(self) -> bool:
        return self._workspace is not None and not self._disabled

    @staticmethod
    def _filename(job_id: str) -> str:
        # JobManager validates the complete identifier.  Keep this method
        # defensive because it is also useful in direct unit tests.
        if (
            type(job_id) is not str
            or len(job_id) != 36
            or not job_id.startswith("job:")
            or not re.fullmatch(r"job:[0-9a-f]{32}", job_id)
        ):
            raise ValueError("invalid job identifier")
        return f"{job_id[4:]}.jsonl"

    def _directory(self) -> Path | None:
        workspace = self._workspace
        if workspace is None or self._disabled:
            return None
        try:
            if workspace.is_symlink() or not workspace.is_dir():
                return None
            root = workspace / _LOG_DIRECTORY
            jobs = root / _JOB_LOG_DIRECTORY
            if root.exists() and root.is_symlink():
                return None
            if jobs.exists() and jobs.is_symlink():
                return None
            root.mkdir(exist_ok=True)
            jobs.mkdir(exist_ok=True)
            # Diagnostic files contain tracebacks; keep them private on POSIX
            # without relying on the process umask.
            try:
                root.chmod(0o700)
                jobs.chmod(0o700)
            except OSError:
                pass
            return jobs
        except (OSError, RuntimeError):
            return None

    def _path(self, job_id: str) -> Path | None:
        try:
            directory = self._directory()
            if directory is None:
                return None
            candidate = directory / self._filename(job_id)
            if candidate.exists() and candidate.is_symlink():
                return None
            resolved_directory = directory.resolve()
            resolved_candidate = candidate.resolve()
            if resolved_candidate.parent != resolved_directory:
                return None
            return candidate
        except (OSError, RuntimeError, ValueError):
            return None

    def _append(self, job_id: str, record: Mapping[str, Any]) -> None:
        with self._lock:
            path = self._path(job_id)
            if path is None:
                return
            count = self._counts.get(job_id, 0)
            if count >= self._max_entries or job_id in self._truncated:
                self._truncated.add(job_id)
                return
            payload = {
                "schema_version": "1.0",
                "sequence": count + 1,
                "at": datetime.now(UTC).isoformat(timespec="milliseconds"),
                **dict(record),
            }
            # Bound every scalar string before serializing.  This keeps one
            # pathological exception from exhausting the file budget.
            for key, value in tuple(payload.items()):
                if isinstance(value, str):
                    payload[key] = value[:_MAX_TEXT]
            encoded = (
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            if len(encoded) > _MAX_ENTRY_BYTES:
                for key in ("traceback", "message", "exception_message"):
                    if isinstance(payload.get(key), str):
                        payload[key] = str(payload[key])[:1024]
                encoded = (
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
                ).encode("utf-8")
            try:
                current = path.stat().st_size if path.exists() else 0
                if current + len(encoded) > self._max_bytes:
                    self._truncated.add(job_id)
                    return
                with path.open("ab") as stream:
                    stream.write(encoded)
                try:
                    path.chmod(0o600)
                except OSError:
                    pass
                self._counts[job_id] = count + 1
            except OSError:
                self._disabled = True

    def start(self, job_id: str, kind: str) -> DiagnosticReference:
        try:
            self._append(job_id, {"event": "submitted", "kind": _redact_text(kind)})
        except Exception:
            # Diagnostics are strictly best effort and cannot affect job
            # submission or the server's primary failure path.
            pass
        return self.reference(job_id)

    def record(
        self,
        job_id: str,
        *,
        event: str,
        kind: str | None = None,
        stage: str | None = None,
        completed: int | None = None,
        total: int | None = None,
        message: str | None = None,
        exc: BaseException | None = None,
        exception_type: str | None = None,
        exception_message: str | None = None,
        traceback_text: str | None = None,
    ) -> None:
        try:
            payload: dict[str, Any] = {"event": _redact_text(event)}
            if kind is not None:
                payload["kind"] = _redact_text(kind)
            if stage is not None:
                payload["stage"] = _redact_text(stage)
            if completed is not None:
                payload["completed"] = completed
            if total is not None:
                payload["total"] = total
            if message:
                payload["message"] = _redact_text(message)
            if exc is not None:
                document = _exception_document(exc, workspace=self._workspace)
                payload.update(
                    {
                        "exception_type": document["type"],
                        "exception_message": document["message"],
                        "traceback": document["traceback"],
                    }
                )
            # A completed worker can return a structured failed outcome rather
            # than raising an outer exception.  Preserve that typed evidence
            # in the same diagnostics channel so the UI does not fall back to
            # ``unknown``.  Values use the same conservative redaction and
            # bounds as ordinary exception text.
            if exception_type is not None:
                payload["exception_type"] = _redact_text(
                    exception_type, workspace=self._workspace
                )
            if exception_message is not None:
                payload["exception_message"] = _redact_text(
                    exception_message, workspace=self._workspace
                )
            if traceback_text is not None:
                payload["traceback"] = _redact_text(
                    traceback_text, workspace=self._workspace
                )
            self._append(job_id, payload)
        except Exception:
            # A malformed exception object or a transient filesystem failure
            # must never replace the original worker outcome.
            return

    def reference(self, job_id: str) -> DiagnosticReference:
        with self._lock:
            try:
                self._filename(job_id)
            except ValueError:
                return DiagnosticReference(
                    endpoint="", available=False, entry_count=0, truncated=False
                )
            path = self._path(job_id)
            count = self._counts.get(job_id, 0)
            if path is not None and count == 0 and path.exists():
                # Recover metadata after a server-side object is recreated in
                # the same workspace without exposing or trusting file text.
                try:
                    with path.open("rb") as stream:
                        count = sum(1 for _ in stream)
                    self._counts[job_id] = min(count, self._max_entries)
                except (OSError, RuntimeError):
                    count = 0
            return DiagnosticReference(
                endpoint=f"/api/jobs/{job_id}/diagnostics",
                available=path is not None and path.exists() and count > 0,
                entry_count=min(count, self._max_entries),
                truncated=job_id in self._truncated or count >= self._max_entries,
            )

    def read(self, job_id: str) -> dict[str, Any]:
        """Read one bounded job log for the existing server's API layer."""

        with self._lock:
            reference = self.reference(job_id)
            entries: list[dict[str, Any]] = []
            path = self._path(job_id)
            if path is not None and path.is_file() and not path.is_symlink():
                try:
                    raw = path.read_bytes()
                    if len(raw) <= self._max_bytes:
                        for line in raw.splitlines()[: self._max_entries]:
                            try:
                                item = json.loads(line)
                            except (TypeError, ValueError):
                                continue
                            if isinstance(item, dict):
                                entries.append(item)
                except (OSError, RuntimeError):
                    pass
            return {
                "available": bool(entries),
                "entries": entries,
                "entry_count": len(entries),
                "truncated": reference.truncated or len(entries) >= self._max_entries,
            }


__all__ = ["DiagnosticReference", "JobDiagnosticStore"]
