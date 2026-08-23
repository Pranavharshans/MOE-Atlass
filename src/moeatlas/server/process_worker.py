"""Bounded, killable subprocess execution for model-backed server jobs.

The HTTP process must not own a model.  A model can terminate itself with an
out-of-memory signal, crash in native code, or import an incompatible remote
module.  This module provides the small process boundary used by the server
integration: the child receives a JSON payload, reports JSON progress, and
returns one JSON result or error record.  No shell is involved.

The worker callable must be importable/picklable (define it at module scope).
It receives ``(payload, report_progress)`` and must return a JSON-safe value.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import signal
import time
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pickle import PicklingError
from threading import Event
from typing import Any, Literal, TypeAlias

WorkerState: TypeAlias = Literal[
    "completed",
    "failed",
    "cancelled",
    "timed_out",
    "crashed",
    "protocol_error",
]

ProgressCallback: TypeAlias = Callable[[Mapping[str, Any]], None]
WorkerCallable: TypeAlias = Callable[[Any, ProgressCallback], Any]


class ProcessWorkerError(ValueError):
    """Raised when a worker request cannot be encoded or configured."""


@dataclass(frozen=True, slots=True)
class ProcessWorkerResult:
    """Terminal, bounded result from :func:`run_process_worker`.

    ``payload`` and ``error`` are JSON-safe dictionaries.  ``progress`` keeps
    only a bounded tail of observed progress records so a noisy model cannot
    grow the parent process without limit.
    """

    state: WorkerState
    payload: Any = None
    error: Mapping[str, Any] | None = None
    exit_code: int | None = None
    progress: tuple[Mapping[str, Any], ...] = ()
    elapsed_ms: int = 0

    def __post_init__(self) -> None:
        if self.state not in {
            "completed",
            "failed",
            "cancelled",
            "timed_out",
            "crashed",
            "protocol_error",
        }:
            raise ValueError("unsupported process-worker state")
        if type(self.elapsed_ms) is not int or isinstance(self.elapsed_ms, bool):
            raise TypeError("elapsed_ms must be an integer")
        object.__setattr__(self, "progress", tuple(dict(item) for item in self.progress))
        if self.error is not None:
            object.__setattr__(self, "error", dict(self.error))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe wire representation."""

        return {
            "state": self.state,
            "payload": self.payload,
            "error": dict(self.error) if self.error is not None else None,
            "exit_code": self.exit_code,
            "progress": [dict(item) for item in self.progress],
            "elapsed_ms": self.elapsed_ms,
        }


@dataclass(slots=True)
class _MonitorState:
    progress: list[Mapping[str, Any]] = field(default_factory=list)
    final_message_seen: bool = False
    terminal: Mapping[str, Any] | None = None


def run_process_worker(
    worker: WorkerCallable,
    payload: Any,
    *,
    cancel_event: Event | None = None,
    timeout_s: float | None = None,
    on_progress: ProgressCallback | None = None,
    max_payload_bytes: int = 1_048_576,
    max_message_bytes: int = 262_144,
    max_progress_events: int = 256,
    terminate_grace_s: float = 1.0,
    poll_interval_s: float = 0.02,
) -> ProcessWorkerResult:
    """Execute one importable worker in a fresh ``spawn`` process.

    The parent owns the cancellation and timeout policy.  A cancellation or
    timeout terminates the child, waits for it, and reports a terminal state;
    a signal/non-zero exit without a terminal message is classified as a
    crash.  All child-to-parent messages are UTF-8 JSON bytes and are bounded
    before crossing the pipe.
    """

    _validate_configuration(
        worker=worker,
        timeout_s=timeout_s,
        max_payload_bytes=max_payload_bytes,
        max_message_bytes=max_message_bytes,
        max_progress_events=max_progress_events,
        terminate_grace_s=terminate_grace_s,
        poll_interval_s=poll_interval_s,
    )
    payload_bytes = _encode_json(payload, limit=max_payload_bytes, label="payload")
    if cancel_event is not None and cancel_event.is_set():
        return ProcessWorkerResult(state="cancelled", error=_error("Cancelled", "cancel requested"))

    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_worker_entry,
        args=(worker, payload_bytes, child_connection, max_message_bytes, max_progress_events),
        name="moeatlas-model-worker",
        daemon=True,
    )
    started = time.monotonic()
    monitor = _MonitorState()
    deadline = started + timeout_s if timeout_s is not None else None
    try:
        try:
            process.start()
        except (OSError, PicklingError, RuntimeError, TypeError, ValueError) as exc:
            return _terminal_result(
                state="protocol_error",
                process=process,
                monitor=monitor,
                started=started,
                error=_error(type(exc).__name__, "worker could not be started"),
            )
        # The child owns its endpoint after start.  Closing it in the parent
        # is important: otherwise EOF is never observable after child exit.
        child_connection.close()
        while True:
            _drain_messages(
                parent_connection,
                monitor,
                on_progress=on_progress,
                max_message_bytes=max_message_bytes,
                max_progress_events=max_progress_events,
            )
            if monitor.final_message_seen:
                # Drain until process exit so the terminal exit code is known.
                if not process.is_alive():
                    break
            if cancel_event is not None and cancel_event.is_set():
                _terminate_process(process, grace_s=terminate_grace_s)
                return _terminal_result(
                    state="cancelled",
                    process=process,
                    monitor=monitor,
                    started=started,
                    error=_error("Cancelled", "cancel requested"),
                )
            if deadline is not None and time.monotonic() >= deadline:
                _terminate_process(process, grace_s=terminate_grace_s)
                return _terminal_result(
                    state="timed_out",
                    process=process,
                    monitor=monitor,
                    started=started,
                    error=_error("TimeoutError", "worker exceeded its time limit"),
                )
            if not process.is_alive():
                break
            parent_connection.poll(poll_interval_s)

        _drain_messages(
            parent_connection,
            monitor,
            on_progress=on_progress,
            max_message_bytes=max_message_bytes,
            max_progress_events=max_progress_events,
        )
        exit_code = process.exitcode
        if monitor.final_message_seen:
            terminal = monitor.terminal
            if terminal is None:
                return _terminal_result(
                    state="protocol_error",
                    process=process,
                    monitor=monitor,
                    started=started,
                    error=_error("ProtocolError", "worker terminal record was incomplete"),
                )
            return _terminal_result(
                state=terminal["state"],
                process=process,
                monitor=monitor,
                started=started,
                payload=terminal.get("payload"),
                error=terminal.get("error"),
            )
        return _terminal_result(
            state="crashed" if exit_code else "protocol_error",
            process=process,
            monitor=monitor,
            started=started,
            error=_child_exit_error(exit_code),
        )
    except (OSError, EOFError) as exc:
        _terminate_process(process, grace_s=terminate_grace_s)
        return _terminal_result(
            state="protocol_error",
            process=process,
            monitor=monitor,
            started=started,
            error=_error(type(exc).__name__, "worker communication failed"),
        )
    finally:
        _close_process_resources(process, parent_connection, child_connection)


def _worker_entry(
    worker: WorkerCallable,
    payload_bytes: bytes,
    connection: Any,
    max_message_bytes: int,
    max_progress_events: int,
) -> None:
    """Spawn target; keep every exception inside the child process."""

    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
        sent_progress = 0

        def report(progress: Mapping[str, Any] | None = None, **fields: Any) -> None:
            nonlocal sent_progress
            if sent_progress >= max_progress_events:
                return
            record: dict[str, Any] = {}
            if progress is not None:
                if not isinstance(progress, Mapping):
                    raise TypeError("progress must be a mapping")
                record.update(progress)
            record.update(fields)
            _send(connection, {"kind": "progress", "progress": record}, max_message_bytes)
            sent_progress += 1

        result = worker(payload, report)
        result_bytes = _encode_json(result, limit=max_message_bytes, label="result")
        _send(
            connection,
            {"kind": "terminal", "state": "completed", "payload": json.loads(result_bytes)},
            max_message_bytes,
        )
    except BaseException as exc:
        try:
            error = _bounded_error(_exception_error(exc), max_message_bytes)
            _send(
                connection,
                {
                    "kind": "terminal",
                    "state": "failed",
                    "error": error,
                },
                max_message_bytes,
            )
        except BaseException:
            # A broken pipe or an oversized exception cannot escape into the
            # parent; its exit code will classify this as a child crash.
            os._exit(70)
    finally:
        try:
            connection.close()
        except Exception:
            pass


def _drain_messages(
    connection: Any,
    monitor: _MonitorState,
    *,
    on_progress: ProgressCallback | None,
    max_message_bytes: int,
    max_progress_events: int,
) -> None:
    while connection.poll():
        try:
            raw = connection.recv_bytes()
        except EOFError:
            # A clean child closes its endpoint immediately after sending the
            # terminal record.  ``poll()`` can still report readable once for
            # that EOF marker, so treat it as end-of-stream rather than a
            # parent-side communication failure.
            return
        if len(raw) > max_message_bytes:
            monitor.final_message_seen = True
            monitor.terminal = {
                "state": "protocol_error",
                "error": _error("ProtocolError", "worker message exceeded its size limit"),
            }
            return
        try:
            message = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            monitor.final_message_seen = True
            monitor.terminal = {
                "state": "protocol_error",
                "error": _error("ProtocolError", "worker sent invalid JSON"),
            }
            return
        if not isinstance(message, Mapping):
            monitor.final_message_seen = True
            monitor.terminal = {
                "state": "protocol_error",
                "error": _error("ProtocolError", "worker sent a non-object message"),
            }
            return
        kind = message.get("kind")
        if kind == "progress":
            if len(monitor.progress) < max_progress_events:
                progress = message.get("progress")
                if isinstance(progress, Mapping):
                    bounded = dict(progress)
                    monitor.progress.append(bounded)
                    if on_progress is not None:
                        try:
                            on_progress(bounded)
                        except Exception:
                            pass
            continue
        if kind == "terminal" and not monitor.final_message_seen:
            state = message.get("state")
            if state not in {"completed", "failed"}:
                monitor.terminal = {
                    "state": "protocol_error",
                    "error": _error("ProtocolError", "worker sent an invalid terminal state"),
                }
            else:
                monitor.terminal = {
                    "state": state,
                    "payload": message.get("payload"),
                    "error": message.get("error"),
                }
            monitor.final_message_seen = True
            continue
        monitor.final_message_seen = True
        monitor.terminal = {
            "state": "protocol_error",
            "error": _error("ProtocolError", "worker sent an unknown message"),
        }
        return


def _terminal_result(
    *,
    state: WorkerState,
    process: Any,
    monitor: _MonitorState,
    started: float,
    payload: Any = None,
    error: Mapping[str, Any] | None = None,
) -> ProcessWorkerResult:
    return ProcessWorkerResult(
        state=state,
        payload=payload,
        error=error,
        exit_code=_process_exit_code(process),
        progress=tuple(monitor.progress),
        elapsed_ms=max(0, int((time.monotonic() - started) * 1000)),
    )


def _terminate_process(process: Any, *, grace_s: float) -> None:
    try:
        alive = process.is_alive()
    except (AssertionError, ValueError):
        return
    if not alive:
        process.join(timeout=grace_s)
        return
    process.terminate()
    process.join(timeout=grace_s)
    if process.is_alive() and hasattr(process, "kill"):
        process.kill()
        process.join(timeout=grace_s)


def _close_process_resources(process: Any, parent_connection: Any, child_connection: Any) -> None:
    for connection in (parent_connection, child_connection):
        try:
            connection.close()
        except Exception:
            pass
    try:
        if process.is_alive():
            _terminate_process(process, grace_s=1.0)
        process.join(timeout=1.0)
    except Exception:
        pass
    try:
        process.close()
    except Exception:
        pass


def _process_exit_code(process: Any) -> int | None:
    try:
        return process.exitcode
    except (AssertionError, ValueError):
        return None


def _send(connection: Any, message: Mapping[str, Any], limit: int) -> None:
    connection.send_bytes(_encode_json(message, limit=limit, label="message"))


def _encode_json(value: Any, *, limit: int, label: str) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProcessWorkerError(f"{label} must be JSON-safe") from exc
    if len(encoded) > limit:
        raise ProcessWorkerError(f"{label} exceeds the {limit}-byte limit")
    return encoded


def _exception_error(exc: BaseException) -> dict[str, str]:
    # Keep useful class identity while bounding arbitrary model/Hub text.  The
    # server diagnostics layer performs its stronger secret/path redaction.
    message = str(exc).replace("\x00", "")[:512]
    formatted = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return {
        "type": type(exc).__name__[:96],
        "message": message,
        "traceback": formatted[-8192:],
    }


def _error(error_type: str, message: str) -> dict[str, str]:
    return {"type": error_type[:96], "message": message[:512]}


def _bounded_error(error: Mapping[str, Any], limit: int) -> dict[str, str]:
    """Fit an exception record inside a terminal protocol message."""

    bounded = {
        "type": str(error.get("type", "WorkerError"))[:96],
        "message": str(error.get("message", "worker failed"))[:512],
        "traceback": str(error.get("traceback", ""))[:8192],
    }
    # Remove the least important text first, retaining type and a useful
    # message even when a caller deliberately chooses a tiny message budget.
    for key in ("traceback", "message", "type"):
        while (
            _wire_size({"kind": "terminal", "state": "failed", "error": bounded}) > limit
            and bounded[key]
        ):
            bounded[key] = bounded[key][: max(0, len(bounded[key]) // 2)]
    return bounded


def _wire_size(value: Any) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _child_exit_error(exit_code: int | None) -> dict[str, str]:
    if exit_code is None:
        return _error("ChildProcessExit", "worker exit status was unavailable")
    if exit_code < 0:
        try:
            signal_name = signal.Signals(-exit_code).name
        except ValueError:
            signal_name = f"signal {-exit_code}"
        return _error("ChildProcessExit", f"worker terminated by {signal_name}")
    return _error("ChildProcessExit", f"worker exited with status {exit_code}")


def _validate_configuration(
    *,
    worker: WorkerCallable,
    timeout_s: float | None,
    max_payload_bytes: int,
    max_message_bytes: int,
    max_progress_events: int,
    terminate_grace_s: float,
    poll_interval_s: float,
) -> None:
    if not callable(worker):
        raise ProcessWorkerError("worker must be callable")
    if timeout_s is not None and (type(timeout_s) not in {int, float} or timeout_s <= 0):
        raise ProcessWorkerError("timeout_s must be positive or None")
    for name, value in (
        ("max_payload_bytes", max_payload_bytes),
        ("max_message_bytes", max_message_bytes),
        ("max_progress_events", max_progress_events),
    ):
        if type(value) is not int or isinstance(value, bool) or value <= 0:
            raise ProcessWorkerError(f"{name} must be a positive integer")
    for name, value in (
        ("terminate_grace_s", terminate_grace_s),
        ("poll_interval_s", poll_interval_s),
    ):
        if type(value) not in {int, float} or value <= 0:
            raise ProcessWorkerError(f"{name} must be positive")


__all__ = [
    "ProcessWorkerError",
    "ProcessWorkerResult",
    "run_process_worker",
]
