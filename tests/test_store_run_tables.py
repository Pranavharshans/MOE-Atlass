"""Contract tests for bounded tabular (CSV/Parquet) run-evidence exports."""

from __future__ import annotations

import csv
import io
import json
import os
from dataclasses import fields
from pathlib import Path

import pytest

try:
    import duckdb
except ImportError:  # pragma: no cover - exercised only without the store extra
    duckdb = None

import moeatlas.store.routing_shards as storage
from moeatlas import __version__
from moeatlas.events import EVENT_SCHEMA_VERSION
from moeatlas.store import (
    RUN_TABLES_SCHEMA_VERSION,
    STORE_SCHEMA_VERSION,
    TABLES_MANIFEST_TYPE,
    RunTableError,
    RunTableFileEntry,
    RunTableReceipt,
    append_routing_shard,
    export_run_tables,
    verify_run_tables,
)

from .test_store_routing_shards import _run_result, _workspace
from .test_store_run_export import _rekey_result


@pytest.fixture(autouse=True)
def _duckdb_required() -> None:
    if duckdb is None:
        pytest.skip("duckdb store extra is unavailable")


def _seed(tmp_path: Path, *, tokens: int = 2):
    workspace = _workspace(tmp_path)
    result, _, inspection = _run_result("legacy", token_count=tokens)
    receipt = append_routing_shard(workspace, result)
    return workspace, result, inspection, receipt


def _export(workspace: Path, destination: Path, run_key: str, **kwargs):
    return export_run_tables(workspace, destination, run_key=run_key, **kwargs)


def _manifest(destination: Path) -> dict:
    payload = (destination / "manifest.json").read_bytes()
    document = json.loads(payload[:-1].decode("utf-8"))
    assert payload.endswith(b"\n") and not payload[:-1].endswith(b"\n")
    return document


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        + b"\n"
    )


def _csv_rows(destination: Path, name: str) -> tuple[list[str], list[list[str]]]:
    text = (destination / name).read_bytes().decode("utf-8")
    rows = [tuple(row) for row in csv.reader(io.StringIO(text, newline=""))]
    return list(rows[0]), [list(row) for row in rows[1:]]


# ---------------------------------------------------------------------------
# Public surface


def test_surface_is_pinned() -> None:
    assert RUN_TABLES_SCHEMA_VERSION == "1.0"
    assert TABLES_MANIFEST_TYPE == "routing_run_tables"
    assert tuple(field.name for field in fields(RunTableReceipt)) == (
        "schema_version",
        "manifest_type",
        "run_key",
        "formats",
        "shard_count",
        "token_count",
        "routing_count",
        "manifest_sha256",
        "entries",
    )
    assert str(RunTableError("budget")) == "run table export failed at budget"
    with pytest.raises(ValueError):
        RunTableError("conflict")


def test_receipt_dataclasses_are_strict() -> None:
    entry = RunTableFileEntry(name="tokens.csv", bytes=3, sha256=f"sha256:{'a' * 64}")
    assert entry == RunTableFileEntry(name="tokens.csv", bytes=3, sha256=f"sha256:{'a' * 64}")
    with pytest.raises(ValueError):
        RunTableFileEntry(name="nested/tokens.csv", bytes=3, sha256=f"sha256:{'a' * 64}")
    with pytest.raises(ValueError):
        RunTableFileEntry(name="tokens.csv", bytes=-1, sha256=f"sha256:{'a' * 64}")
    with pytest.raises(ValueError):
        RunTableFileEntry(name="tokens.csv", bytes=3, sha256="sha256:short")
    base = dict(
        schema_version=RUN_TABLES_SCHEMA_VERSION,
        manifest_type=TABLES_MANIFEST_TYPE,
        run_key="valid-run",
        formats=("csv",),
        shard_count=1,
        token_count=1,
        routing_count=1,
        manifest_sha256=f"sha256:{'b' * 64}",
        entries=(entry,),
    )
    assert RunTableReceipt(**base).run_key == "valid-run"
    # Both requested formats construct fine; entries are not cross-checked
    # against formats here (the manifest loader owns that exactness).
    assert RunTableReceipt(**{**base, "formats": ("csv", "parquet")}).formats == (
        "csv",
        "parquet",
    )
    for key, value in (
        ("schema_version", "9.9"),
        ("manifest_type", "other"),
        ("formats", ("bogus",)),
        ("formats", ("parquet", "csv")),
        ("shard_count", 0),
        ("token_count", True),
        ("manifest_sha256", "sha256:short"),
        ("entries", ()),
    ):
        broken = dict(base)
        broken[key] = value
        with pytest.raises((TypeError, ValueError)):
            RunTableReceipt(**broken)


# ---------------------------------------------------------------------------
# Export happy paths


def test_csv_export_round_trips_and_manifest_is_canonical(tmp_path: Path) -> None:
    workspace, result, _, receipt = _seed(tmp_path)
    destination = tmp_path / "tables-out"

    exported = _export(workspace, destination, receipt.run_key)
    verified = verify_run_tables(destination)

    assert exported.schema_version == RUN_TABLES_SCHEMA_VERSION
    assert exported.manifest_type == TABLES_MANIFEST_TYPE
    assert exported.run_key == receipt.run_key
    assert exported.formats == ("csv",)
    assert exported.shard_count == 1
    assert exported.token_count == len(result.token_events)
    assert exported.routing_count == len(result.routing_events)
    assert [entry.name for entry in exported.entries] == [
        "manifest.json",
        "routing.csv",
        "tokens.csv",
    ]
    assert verified.manifest_sha256 == exported.manifest_sha256
    assert verified.token_count == exported.token_count

    manifest = _manifest(destination)
    assert manifest["manifest_type"] == TABLES_MANIFEST_TYPE
    assert manifest["store_schema_version"] == STORE_SCHEMA_VERSION
    assert manifest["event_schema_version"] == EVENT_SCHEMA_VERSION
    assert manifest["writer_name"] == "moeatlas"
    assert manifest["writer_version"] == __version__
    assert (destination / "manifest.json").read_bytes() == _canonical(manifest)


def test_csv_members_are_byte_deterministic_across_exports(tmp_path: Path) -> None:
    workspace, _, _, receipt = _seed(tmp_path)
    first = tmp_path / "tables-a"
    second = tmp_path / "tables-b"
    _export(workspace, first, receipt.run_key)
    _export(workspace, second, receipt.run_key)
    for name in ("tokens.csv", "routing.csv", "manifest.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_both_formats_publish_exact_member_sets_with_readable_parquet(
    tmp_path: Path,
) -> None:
    workspace, result, _, receipt = _seed(tmp_path)
    destination = tmp_path / "tables-both"

    exported = _export(workspace, destination, receipt.run_key, formats=("csv", "parquet"))
    assert [entry.name for entry in exported.entries] == [
        "manifest.json",
        "routing.csv",
        "routing.parquet",
        "tokens.csv",
        "tokens.parquet",
    ]
    verify_run_tables(destination)

    connection = duckdb.connect(database=":memory:")
    try:
        count = connection.execute(
            "SELECT COUNT(*) FROM read_parquet(?)", [str(destination / "routing.parquet")]
        ).fetchone()[0]
        columns = [
            row[0]
            for row in connection.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)",
                [str(destination / "routing.parquet")],
            ).fetchall()
        ]
    finally:
        connection.close()
    assert count == len(result.routing_events)
    assert columns[0] == "store_schema_version"
    assert columns[-1] == "selected"


def test_multi_shard_export_orders_rows_by_shard_then_index(tmp_path: Path) -> None:
    workspace, first, _, receipt = _seed(tmp_path)
    second = _rekey_result(first, run_key=receipt.run_key, offset=41)
    second_receipt = append_routing_shard(workspace, second)
    destination = tmp_path / "tables-multi"

    exported = _export(workspace, destination, receipt.run_key)
    assert exported.shard_count == 2
    assert exported.token_count == len(first.token_events) * 2
    verify_run_tables(destination)

    header, rows = _csv_rows(destination, "tokens.csv")
    assert header[1] == "shard_key"
    assert header[2] == "event_index"
    shard_keys = {row[1] for row in rows}
    assert shard_keys == {receipt.shard_key, second_receipt.shard_key}
    order = [(row[1], int(row[2])) for row in rows]
    assert order == sorted(order)


def test_redaction_travels_through_tabular_members(tmp_path: Path) -> None:
    redacted_workspace = _workspace(tmp_path)
    stored_workspace = tmp_path / "stored workspace"
    stored_workspace.mkdir()
    result, _, _ = _run_result("legacy", token_count=1)
    redacted_receipt = append_routing_shard(redacted_workspace, result)
    stored_receipt = append_routing_shard(stored_workspace, result, store_token_text=True)

    redacted_destination = tmp_path / "redacted-tables"
    stored_destination = tmp_path / "stored-tables"
    _export(redacted_workspace, redacted_destination, redacted_receipt.run_key)
    _export(stored_workspace, stored_destination, stored_receipt.run_key)

    _, redacted_rows = _csv_rows(redacted_destination, "tokens.csv")
    _, stored_rows = _csv_rows(stored_destination, "tokens.csv")
    text_column = _csv_rows(redacted_destination, "tokens.csv")[0].index("token_text")
    flag_column = _csv_rows(redacted_destination, "tokens.csv")[0].index("token_text_stored")
    assert all(row[text_column] == "" for row in redacted_rows)
    assert all(row[flag_column] == "false" for row in redacted_rows)
    assert all(row[flag_column] == "true" for row in stored_rows)
    assert any(row[text_column] != "" for row in stored_rows)


def test_parquet_rows_match_csv_projection(tmp_path: Path) -> None:
    workspace, _, _, receipt = _seed(tmp_path)
    destination = tmp_path / "tables-parity"
    _export(workspace, destination, receipt.run_key, formats=("csv", "parquet"))

    connection = duckdb.connect(database=":memory:")
    try:
        parquet_keys = {
            row[0]
            for row in connection.execute(
                "SELECT token_key FROM read_parquet(?) ORDER BY event_index",
                [str(destination / "tokens.parquet")],
            ).fetchall()
        }
    finally:
        connection.close()
    header, csv_rows = _csv_rows(destination, "tokens.csv")
    key_column = header.index("token_key")
    assert parquet_keys == {row[key_column] for row in csv_rows}


# ---------------------------------------------------------------------------
# Argument validation and failure stages


def test_unknown_run_fails_at_source_without_creating_destination(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    destination = tmp_path / "absent"
    with pytest.raises(RunTableError) as caught:
        _export(workspace, destination, "run-with-no-shards")
    assert caught.value.stage == "source"
    assert not destination.exists()


def test_argument_validation_is_exact(tmp_path: Path) -> None:
    workspace, _, _, receipt = _seed(tmp_path)
    with pytest.raises(TypeError):
        _export(workspace, tmp_path / "out", ["not-a-str"])
    with pytest.raises(TypeError):
        _export(workspace, tmp_path / "out", receipt.run_key, formats=["csv"])
    with pytest.raises(TypeError):
        _export(workspace, tmp_path / "out", receipt.run_key, formats=())
    with pytest.raises(ValueError):
        _export(workspace, tmp_path / "out", receipt.run_key, formats=("xlsx",))
    with pytest.raises(ValueError):
        _export(workspace, tmp_path / "out", receipt.run_key, formats=("csv", "csv"))
    with pytest.raises((TypeError, ValueError)):
        _export(workspace, tmp_path / "out", receipt.run_key, max_event_rows=0)
    with pytest.raises((TypeError, ValueError)):
        _export(workspace, tmp_path / "out", receipt.run_key, max_file_bytes="tiny")


def test_row_and_byte_budgets_leave_no_destination(tmp_path: Path) -> None:
    workspace, _, _, receipt = _seed(tmp_path)
    row_destination = tmp_path / "rows-out"
    with pytest.raises(RunTableError) as row_caught:
        _export(workspace, row_destination, receipt.run_key, max_event_rows=1)
    assert row_caught.value.stage == "budget"
    assert not row_destination.exists()

    byte_destination = tmp_path / "bytes-out"
    with pytest.raises(RunTableError) as byte_caught:
        _export(workspace, byte_destination, receipt.run_key, max_file_bytes=16)
    assert byte_caught.value.stage == "budget"
    assert not byte_destination.exists()
    assert not any(tmp_path.glob(".bytes-out.export-staging-*"))


def test_destination_validation_refuses_symlinks_and_non_empty_dirs(
    tmp_path: Path,
) -> None:
    workspace, _, _, receipt = _seed(tmp_path)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "stale.txt").write_text("stale")
    with pytest.raises(RunTableError) as occupied_caught:
        _export(workspace, occupied, receipt.run_key)
    assert occupied_caught.value.stage == "workspace"

    link = tmp_path / "linked"
    real = tmp_path / "real"
    real.mkdir()
    link.symlink_to(real)
    with pytest.raises(RunTableError) as link_caught:
        _export(workspace, link, receipt.run_key)
    assert link_caught.value.stage == "workspace"

    empty = tmp_path / "empty"
    empty.mkdir()
    exported = _export(workspace, empty, receipt.run_key)
    assert exported.token_count > 0


def test_dependency_failure_maps_to_dependency_stage(tmp_path: Path, monkeypatch) -> None:
    workspace, _, _, receipt = _seed(tmp_path)

    def raise_dependency():
        raise storage.RoutingShardError("dependency")

    monkeypatch.setattr(storage, "_load_duckdb", raise_dependency)
    with pytest.raises(RunTableError) as caught:
        _export(workspace, tmp_path / "out", receipt.run_key)
    assert caught.value.stage == "dependency"


def test_crash_during_publication_cleans_staging(tmp_path: Path, monkeypatch) -> None:
    workspace, _, _, receipt = _seed(tmp_path)
    destination = tmp_path / "crash-out"

    def crash_rename(source: object, target: object) -> None:
        raise OSError("simulated crash")

    monkeypatch.setattr(os, "rename", crash_rename)
    with pytest.raises(RunTableError) as caught:
        _export(workspace, destination, receipt.run_key)
    assert caught.value.stage == "publish"
    assert not destination.exists()
    assert not any(tmp_path.glob(".crash-out.export-staging-*"))


# ---------------------------------------------------------------------------
# Verification and tamper evidence


def test_verify_requires_no_duckdb(tmp_path: Path, monkeypatch) -> None:
    workspace, _, _, receipt = _seed(tmp_path)
    destination = tmp_path / "verify-out"
    _export(workspace, destination, receipt.run_key)

    def raise_dependency():
        raise AssertionError("verification must not resolve the storage engine")

    monkeypatch.setattr(storage, "_load_duckdb", raise_dependency)
    verified = verify_run_tables(destination)
    assert verified.token_count > 0


def test_tampered_member_fails_digest_verification(tmp_path: Path) -> None:
    workspace, _, _, receipt = _seed(tmp_path)
    destination = tmp_path / "tampered"
    _export(workspace, destination, receipt.run_key)
    target = destination / "tokens.csv"
    target.write_bytes(target.read_bytes() + b"junk")

    with pytest.raises(RunTableError) as caught:
        verify_run_tables(destination)
    assert caught.value.stage == "format"


def test_non_canonical_csv_with_matching_digest_is_rejected(tmp_path: Path) -> None:
    workspace, _, _, receipt = _seed(tmp_path)
    destination = tmp_path / "respaced"
    _export(workspace, destination, receipt.run_key)

    manifest = _manifest(destination)
    original = (destination / "tokens.csv").read_bytes().decode("utf-8")
    lines = original.split("\n")
    lines[1] = lines[1].replace(",", ", ", 1)
    respaced = "\n".join(lines).encode("utf-8")
    (destination / "tokens.csv").write_bytes(respaced)
    manifest["files"]["tokens.csv"] = {
        "bytes": len(respaced),
        "sha256": f"sha256:{__import__('hashlib').sha256(respaced).hexdigest()}",
    }
    (destination / "manifest.json").write_bytes(_canonical(manifest))

    with pytest.raises(RunTableError) as caught:
        verify_run_tables(destination)
    assert "canonically encoded" in str(caught.value)


def test_wrong_header_or_row_count_is_rejected(tmp_path: Path) -> None:
    workspace, _, _, receipt = _seed(tmp_path)
    destination = tmp_path / "headerless"
    _export(workspace, destination, receipt.run_key)

    manifest = _manifest(destination)
    payload = (destination / "tokens.csv").read_bytes()
    swapped = b"x,y,z\n" + b"\n".join(payload.split(b"\n")[1:])
    (destination / "tokens.csv").write_bytes(swapped)
    manifest["files"]["tokens.csv"] = {
        "bytes": len(swapped),
        "sha256": f"sha256:{__import__('hashlib').sha256(swapped).hexdigest()}",
    }
    (destination / "manifest.json").write_bytes(_canonical(manifest))
    with pytest.raises(RunTableError) as caught:
        verify_run_tables(destination)
    assert caught.value.stage == "format"


@pytest.mark.parametrize(
    ("mutate"),
    [
        "schema_version",
        "manifest_type",
        "missing_key",
        "extra_key",
        "bad_formats",
        "wrong_files",
        "store_schema_version",
    ],
)
def test_manifest_shape_violations_are_rejected(tmp_path: Path, mutate: str) -> None:
    workspace, _, _, receipt = _seed(tmp_path)
    destination = tmp_path / f"manifest-{mutate}"
    _export(workspace, destination, receipt.run_key)
    manifest = _manifest(destination)
    if mutate == "schema_version":
        manifest["schema_version"] = "9.9"
    elif mutate == "manifest_type":
        manifest["manifest_type"] = "other_tables"
    elif mutate == "missing_key":
        manifest.pop("writer_name")
    elif mutate == "extra_key":
        manifest["surprise"] = True
    elif mutate == "bad_formats":
        manifest["formats"] = ["xlsx"]
    elif mutate == "wrong_files":
        manifest["files"].pop("routing.csv")
    elif mutate == "store_schema_version":
        manifest["store_schema_version"] = "0.0"
    (destination / "manifest.json").write_bytes(_canonical(manifest))
    with pytest.raises(RunTableError) as caught:
        verify_run_tables(destination)
    assert caught.value.stage == "format"


def test_symlinked_member_is_refused(tmp_path: Path) -> None:
    workspace, _, _, receipt = _seed(tmp_path)
    destination = tmp_path / "linked-member"
    _export(workspace, destination, receipt.run_key)
    outside = tmp_path / "outside.csv"
    outside.write_bytes((destination / "tokens.csv").read_bytes())
    (destination / "tokens.csv").unlink()
    (destination / "tokens.csv").symlink_to(outside)
    with pytest.raises(RunTableError) as caught:
        verify_run_tables(destination)
    assert caught.value.stage == "format"
