"""Contract tests for retention evaluation over the run registry."""

from __future__ import annotations

import pytest

from moeatlas.services import (
    RETENTION_SCHEMA_VERSION,
    RetentionError,
    RetentionPolicy,
    RetentionReport,
    evaluate_retention,
)
from moeatlas.store.catalog import RunRegistryEntry


def _entry(
    run_key: str,
    *,
    registered_at: str | None = None,
    state: str | None = "completed",
) -> RunRegistryEntry:
    return RunRegistryEntry(
        run_key=run_key,
        state=state,
        registered_at=registered_at,
    )


def _keys(entries: tuple[RunRegistryEntry, ...]) -> tuple[str, ...]:
    return tuple(entry.run_key for entry in entries)


def test_surface_is_pinned() -> None:
    assert RETENTION_SCHEMA_VERSION == "1.0"
    with pytest.raises(ValueError, match="stage is not supported"):
        RetentionError("unknown")


def test_policy_requires_at_least_one_bound() -> None:
    with pytest.raises(RetentionError, match="at least one bound"):
        RetentionPolicy()
    assert RetentionPolicy(max_runs=5) == RetentionPolicy(max_runs=5)
    assert RetentionPolicy(before="2026-01-01T00:00:00Z") == RetentionPolicy(
        before="2026-01-01T00:00:00Z"
    )
    assert hash(RetentionPolicy(max_runs=5)) == hash(RetentionPolicy(max_runs=5))


def test_policy_contract_violations() -> None:
    with pytest.raises(TypeError):
        RetentionPolicy(max_runs=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        RetentionPolicy(max_runs="5")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        RetentionPolicy(before=20260101)  # type: ignore[arg-type]
    with pytest.raises(RetentionError, match="strictly positive"):
        RetentionPolicy(max_runs=0)
    with pytest.raises(RetentionError, match="canonical YYYY-MM-DDTHH:MM:SSZ form"):
        RetentionPolicy(before="2026-01-01")
    policy = RetentionPolicy(max_runs=1)
    with pytest.raises(AttributeError):
        policy.max_runs = 9  # type: ignore[misc]


def test_before_boundary_expires_only_older_entries() -> None:
    entries = (
        _entry("run:" + "a" * 64, registered_at="2026-01-02T00:00:00Z"),
        _entry("run:" + "b" * 64, registered_at="2026-01-01T00:00:00Z"),
        _entry("run:" + "c" * 64, registered_at="2026-01-03T00:00:00Z"),
    )
    report = evaluate_retention(
        entries, RetentionPolicy(before="2026-01-02T00:00:00Z")
    )
    assert report.evaluated_count == 3
    # Strictly earlier than the boundary expires; equal stays.
    assert report.expired_keys == ("run:" + "b" * 64,)
    assert _keys(tuple()) == ()
    assert report.retained_keys == (
        "run:" + "a" * 64,
        "run:" + "c" * 64,
    )


def test_untimestamped_entries_carry_no_age_evidence() -> None:
    # Absence is evidence: age-based expiry never touches entries without
    # a registration timestamp, but they still order as oldest for
    # count-based retention.
    stampless = _entry("run:" + "d" * 64, registered_at=None)
    stamped_new = _entry("run:" + "e" * 64, registered_at="2026-06-01T00:00:00Z")
    stamped_old = _entry("run:" + "f" * 64, registered_at="2020-06-01T00:00:00Z")
    report = evaluate_retention(
        (stampless, stamped_new), RetentionPolicy(before="2026-01-01T00:00:00Z")
    )
    assert report.expired_keys == ()
    assert report.retained_keys == (
        "run:" + "d" * 64,
        "run:" + "e" * 64,
    )
    report_count = evaluate_retention(
        (stamped_old, stampless, stamped_new), RetentionPolicy(max_runs=2)
    )
    assert report_count.expired_keys == ("run:" + "d" * 64,)
    assert report_count.retained_keys == (
        "run:" + "f" * 64,
        "run:" + "e" * 64,
    )


def test_max_runs_keeps_the_newest_tail() -> None:
    entries = (
        _entry("run:" + "a" * 64, registered_at="2026-03-01T00:00:00Z"),
        _entry("run:" + "b" * 64, registered_at="2026-01-01T00:00:00Z"),
        _entry("run:" + "c" * 64, registered_at="2026-02-01T00:00:00Z"),
    )
    report = evaluate_retention(entries, RetentionPolicy(max_runs=2))
    assert report.expired_keys == ("run:" + "b" * 64,)
    assert report.retained_keys == (
        "run:" + "c" * 64,
        "run:" + "a" * 64,
    )
    keep_all = evaluate_retention(entries, RetentionPolicy(max_runs=10))
    assert keep_all.expired_keys == ()


def test_timestamp_ties_break_by_run_key() -> None:
    same_stamp = "2026-05-01T00:00:00Z"
    entries = (
        _entry("run:" + "b" * 64, registered_at=same_stamp),
        _entry("run:" + "a" * 64, registered_at=same_stamp),
    )
    report = evaluate_retention(entries, RetentionPolicy(max_runs=1))
    assert report.expired_keys == ("run:" + "a" * 64,)
    assert report.retained_keys == ("run:" + "b" * 64,)


def test_combined_bounds_apply_after_expiry() -> None:
    entries = (
        _entry("run:" + "a" * 64, registered_at="2025-12-01T00:00:00Z"),
        _entry("run:" + "b" * 64, registered_at="2026-01-01T00:00:00Z"),
        _entry("run:" + "c" * 64, registered_at="2026-02-01T00:00:00Z"),
    )
    report = evaluate_retention(
        entries,
        RetentionPolicy(max_runs=1, before="2026-01-01T00:00:00Z"),
    )
    # a expires by age; among survivors only the newest tail of one is kept.
    assert report.expired_keys == ("run:" + "a" * 64, "run:" + "b" * 64)
    assert report.retained_keys == ("run:" + "c" * 64,)


def test_empty_registry_evaluates_to_empty_report() -> None:
    report = evaluate_retention((), RetentionPolicy(max_runs=3))
    assert report.evaluated_count == 0
    assert report.retained_keys == () and report.expired_keys == ()


def test_evaluation_input_contracts() -> None:
    entry = _entry("run:" + "a" * 64, registered_at="2026-01-01T00:00:00Z")
    with pytest.raises(TypeError):
        evaluate_retention([entry], RetentionPolicy(max_runs=1))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        evaluate_retention((entry,), "not-a-policy")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        evaluate_retention(("not-an-entry",), RetentionPolicy(max_runs=1))  # type: ignore[list-item]


def test_report_canonical_round_trip_and_rejection() -> None:
    entries = (
        _entry("run:" + "b" * 64, registered_at="2026-01-01T00:00:00Z"),
        _entry("run:" + "a" * 64, registered_at="2026-02-01T00:00:00Z"),
    )
    report = evaluate_retention(entries, RetentionPolicy(max_runs=1))
    document = report.to_json()
    assert '"artifact_type":"moeatlas.retention_report"' in document
    assert RetentionReport.from_json(document) == report
    assert RetentionReport.from_json(document.encode()) == report
    with pytest.raises(ValueError, match="not a retention report artifact"):
        RetentionReport.from_json('{"schema_version":"1.0"}')
    with pytest.raises(ValueError, match="not valid JSON"):
        RetentionReport.from_json("{oops")
    with pytest.raises(ValueError, match="must be a JSON object"):
        RetentionReport.from_json("[]")
