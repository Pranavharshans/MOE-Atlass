"""Model-free tests for the killable process boundary."""

from __future__ import annotations

import json
import os
import threading
import time

import pytest

from moeatlas.server.process_worker import ProcessWorkerError, run_process_worker


def worker_echo(payload, report):
    report({"stage": "started", "value": payload["value"]})
    report(stage="finished")
    return {"echo": payload["value"]}


def worker_failure(_payload, _report):
    raise RuntimeError("private model path and prompt should remain bounded")


def worker_sleep(_payload, report):
    report(stage="waiting")
    time.sleep(10)
    return {"done": True}


def worker_exit(_payload, _report):
    os._exit(23)


def worker_large(_payload, _report):
    return {"value": "x" * 1000}


def test_worker_returns_json_result_and_progress() -> None:
    observed = []
    result = run_process_worker(
        worker_echo,
        {"value": 7},
        on_progress=observed.append,
    )

    assert result.state == "completed"
    assert result.payload == {"echo": 7}
    assert result.error is None
    assert observed == list(result.progress)
    json.dumps(result.to_dict(), allow_nan=False)


def test_worker_exception_is_failed_and_has_bounded_error() -> None:
    result = run_process_worker(worker_failure, {})

    assert result.state == "failed"
    assert result.error is not None
    assert result.error["type"] == "RuntimeError"
    assert len(result.error["traceback"]) <= 8192
    assert result.payload is None


def test_timeout_terminates_child_and_classifies_result() -> None:
    result = run_process_worker(worker_sleep, {}, timeout_s=0.05, terminate_grace_s=0.2)

    assert result.state == "timed_out"
    assert result.error == {"type": "TimeoutError", "message": "worker exceeded its time limit"}


def test_cancellation_terminates_child_and_classifies_result() -> None:
    cancel = threading.Event()
    timer = threading.Timer(0.05, cancel.set)
    timer.start()
    try:
        result = run_process_worker(worker_sleep, {}, cancel_event=cancel, terminate_grace_s=0.2)
    finally:
        timer.cancel()

    assert result.state == "cancelled"
    assert result.error == {"type": "Cancelled", "message": "cancel requested"}


def test_signal_exit_is_classified_as_crash() -> None:
    result = run_process_worker(worker_exit, {})

    assert result.state == "crashed"
    assert result.exit_code == 23
    assert result.error == {
        "type": "ChildProcessExit",
        "message": "worker exited with status 23",
    }


def test_payload_and_result_limits_are_enforced() -> None:
    with pytest.raises(ProcessWorkerError, match="payload exceeds"):
        run_process_worker(worker_echo, {"value": "x" * 100}, max_payload_bytes=32)

    result = run_process_worker(worker_large, {}, max_message_bytes=128)
    assert result.state == "failed"
    assert result.error is not None
    assert result.error["type"] == "ProcessWorkerError"


def test_non_json_payload_is_rejected_before_spawn() -> None:
    with pytest.raises(ProcessWorkerError, match="payload must be JSON-safe"):
        run_process_worker(worker_echo, {"value": float("nan")})
