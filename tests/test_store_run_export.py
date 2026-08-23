"""Run-evidence export bundles: bounded open-format export/verify/import."""

from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

try:
    import duckdb
except ImportError:  # pragma: no cover - exercised only without the store extra
    duckdb = None

import moeatlas.store.run_export as run_export
from moeatlas.events import RoutingEvent, TokenEvent
from moeatlas.runtime import RoutingForwardResult
from moeatlas.store import (
    BUNDLE_MANIFEST_TYPE,
    RUN_EXPORT_SCHEMA_VERSION,
    RoutingShardError,
    RunBundleError,
    RunBundleFileEntry,
    RunBundleReceipt,
    append_routing_shard,
    export_run_bundle,
    import_run_bundle,
    list_routing_shards,
    verify_run_bundle,
)


def _rekey_result(result: object, *, run_key: str, offset: int):
    """Re-key every event into ``run_key`` with disjoint token identities.

    Unlike the inventory helper, the routing re-key follows each original
    token exactly, so per-token route structure and link validity survive.
    """

    tokens = []
    for index, event in enumerate(result.token_events):
        payload = event.model_dump(mode="json")
        payload["run_key"] = run_key
        payload["token_id"] = event.token_id + offset + index
        payload.pop("token_key", None)
        tokens.append(TokenEvent.model_validate(payload))
    tokens = tuple(tokens)
    key_map = {
        original.token_key: replacement.token_key
        for original, replacement in zip(result.token_events, tokens)
    }
    routes = tuple(
        RoutingEvent.model_validate(
            {**event.model_dump(mode="json"), "token_key": key_map[event.token_key]}
        )
        for event in result.routing_events
    )
    return RoutingForwardResult(result.output, tokens, routes)


def _workspace(tmp_path: Path, name: str = "workspace") -> Path:
    path = tmp_path / name
    path.mkdir()
    return path


def _seed_run(
    tmp_path: Path,
    *,
    shards: int = 1,
    stored_flags: tuple[bool, ...] = (False,),
):
    from .test_runtime_routing_forward import _run

    result, _model, _inspection = _run("legacy", token_count=2)
    workspace = _workspace(tmp_path)
    receipts = []
    for index in range(shards):
        if index == 0:
            shard_result = result
        else:
            shard_result = _rekey_result(
                result, run_key=receipts[0].run_key, offset=100 * index
            )
        receipts.append(
            append_routing_shard(
                workspace,
                shard_result,
                store_token_text=stored_flags[index % len(stored_flags)],
            )
        )
    return workspace, tuple(receipts), receipts[0].run_key


def _export(
    tmp_path: Path,
    *,
    shards: int = 1,
    stored_flags: tuple[bool, ...] = (False,),
    name: str = "bundle",
):
    workspace, receipts, run_key = _seed_run(
        tmp_path, shards=shards, stored_flags=stored_flags
    )
    destination = tmp_path / name
    receipt = export_run_bundle(workspace, destination, run_key=run_key)
    return workspace, destination, receipt, run_key


def _manifest(destination: Path) -> dict:
    payload = (destination / "manifest.json").read_bytes()
    assert payload.endswith(b"\n") and not payload[:-1].endswith(b"\n")
    return json.loads(payload[:-1].decode("utf-8"))


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _canonical_lines(records: list[dict]) -> bytes:
    return b"".join(_canonical(record) for record in records)


def _member(destination: Path, suffix: str) -> Path:
    members = sorted((destination / "data").glob(f"*/{suffix}"))
    assert len(members) == 1
    return members[0]


def _member_records(destination: Path, suffix: str) -> list[dict]:
    return [
        json.loads(line)
        for line in _member(destination, suffix).read_text(encoding="utf-8").splitlines()
    ]


def test_public_surface_is_pinned() -> None:
    export_signature = inspect.signature(export_run_bundle)
    assert tuple(export_signature.parameters) == (
        "workspace",
        "destination",
        "run_key",
        "max_event_rows",
        "max_file_bytes",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for name, parameter in export_signature.parameters.items()
        if name not in {"workspace", "destination"}
    )
    verify_signature = inspect.signature(verify_run_bundle)
    assert tuple(verify_signature.parameters) == (
        "source",
        "max_event_rows",
        "max_file_bytes",
    )
    import_signature = inspect.signature(import_run_bundle)
    assert tuple(import_signature.parameters) == (
        "source",
        "workspace",
        "max_event_rows",
        "max_file_bytes",
    )
    assert export_signature.parameters["max_event_rows"].default == 1_000_000
    assert verify_signature.parameters["max_event_rows"].default == 1_000_000
    assert import_signature.parameters["max_file_bytes"].default == 1_000_000_000
    assert BUNDLE_MANIFEST_TYPE == "routing_run_export"
    assert RUN_EXPORT_SCHEMA_VERSION == "1.0"
    with pytest.raises(ValueError, match="unsupported run bundle stage"):
        RunBundleError("bogus")
    error = RunBundleError("format", "detail")
    assert str(error) == "run bundle failed at format: detail"
    assert error.stage == "format"


def test_receipt_dataclasses_are_strict() -> None:
    entry = RunBundleFileEntry(
        name="data/shard-" + "a" * 64 + "/tokens.jsonl",
        bytes=8,
        sha256="sha256:" + "b" * 64,
    )
    assert getattr(RunBundleFileEntry, "__slots__")
    assert getattr(RunBundleReceipt, "__slots__")
    with pytest.raises(ValueError, match="safe relative bundle path"):
        RunBundleFileEntry(name="../escape.jsonl", bytes=8, sha256=entry.sha256)
    with pytest.raises(ValueError, match="canonical sha256 digest"):
        RunBundleFileEntry(name="ok.jsonl", bytes=8, sha256="md5:deadbeef")
    with pytest.raises(ValueError, match="entry bytes must be positive"):
        RunBundleFileEntry(name="ok.jsonl", bytes=0, sha256=entry.sha256)
    base = {
        "schema_version": RUN_EXPORT_SCHEMA_VERSION,
        "manifest_type": BUNDLE_MANIFEST_TYPE,
        "run_key": "run-1",
        "shard_count": 1,
        "token_count": 1,
        "routing_count": 1,
        "manifest_sha256": "sha256:" + "c" * 64,
        "entries": (entry,),
    }
    receipt = RunBundleReceipt(**base)
    assert receipt.shard_count == 1
    with pytest.raises(ValueError, match="exact run export schema version"):
        RunBundleReceipt(**{**base, "schema_version": "9.9"})
    with pytest.raises(ValueError, match="entries must be sorted by name"):
        RunBundleReceipt(
            **{
                **base,
                "entries": (
                    RunBundleFileEntry(
                        name="z.jsonl", bytes=1, sha256="sha256:" + "d" * 64
                    ),
                    RunBundleFileEntry(
                        name="a.jsonl", bytes=1, sha256="sha256:" + "e" * 64
                    ),
                ),
            }
        )


def test_export_verify_round_trip_single_stored_shard(tmp_path: Path) -> None:
    if duckdb is None:
        pytest.skip("duckdb store extra is unavailable")
    workspace, destination, receipt, run_key = _export(tmp_path, stored_flags=(True,))
    assert isinstance(receipt, RunBundleReceipt)
    assert receipt.schema_version == RUN_EXPORT_SCHEMA_VERSION
    assert receipt.manifest_type == BUNDLE_MANIFEST_TYPE
    assert receipt.run_key == run_key
    assert receipt.shard_count == 1
    original = list_routing_shards(workspace, run_key=run_key)[0]
    assert receipt.token_count == original.token_count
    assert receipt.routing_count == original.routing_count
    hex_digest = original.shard_key.removeprefix("shard:")
    member_directory = destination / "data" / f"shard-{hex_digest}"
    assert {item.name for item in destination.iterdir()} == {"manifest.json", "data"}
    assert {item.name for item in member_directory.iterdir()} == {
        "tokens.jsonl",
        "routing.jsonl",
    }
    manifest = _manifest(destination)
    assert manifest["manifest_type"] == BUNDLE_MANIFEST_TYPE
    assert manifest["run_key"] == run_key
    assert manifest["writer_name"] == "moeatlas"
    assert manifest["shards"][0]["token_text_stored"] is True
    assert (destination / "manifest.json").read_bytes() == _canonical(manifest)
    verified = verify_run_bundle(destination)
    assert verified == receipt


def test_export_is_byte_deterministic(tmp_path: Path) -> None:
    if duckdb is None:
        pytest.skip("duckdb store extra is unavailable")
    workspace, _receipts, run_key = _seed_run(
        tmp_path, shards=2, stored_flags=(False, True)
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    one = export_run_bundle(workspace, first, run_key=run_key)
    two = export_run_bundle(workspace, second, run_key=run_key)
    assert one.manifest_sha256 == two.manifest_sha256
    names = sorted(entry.name for entry in one.entries)
    assert [entry.name for entry in one.entries] == names
    for entry in one.entries:
        assert (first / entry.name).read_bytes() == (second / entry.name).read_bytes()


def test_import_reproduces_identities_and_is_idempotent(tmp_path: Path) -> None:
    if duckdb is None:
        pytest.skip("duckdb store extra is unavailable")
    workspace, originals, run_key = _seed_run(
        tmp_path, shards=2, stored_flags=(False, True)
    )
    original_signatures = {
        (item.shard_key, item.token_count, item.routing_count, item.token_text_stored)
        for item in originals
    }
    destination = tmp_path / "bundle"
    receipt = export_run_bundle(workspace, destination, run_key=run_key)
    assert receipt.shard_count == 2

    fresh = _workspace(tmp_path, "fresh-workspace")
    imported = import_run_bundle(destination, fresh)
    assert {item.shard_key for item in imported} == {
        item.shard_key for item in originals
    }
    assert all(item.created for item in imported)
    reopened = list_routing_shards(fresh, run_key=run_key)
    assert {
        (item.shard_key, item.token_count, item.routing_count, item.token_text_stored)
        for item in reopened
    } == original_signatures

    again = import_run_bundle(destination, workspace)
    assert {item.created for item in again} == {False}
    assert {item.shard_key for item in again} == {
        item.shard_key for item in originals
    }


def test_redaction_and_text_round_trips_across_workspaces(tmp_path: Path) -> None:
    if duckdb is None:
        pytest.skip("duckdb store extra is unavailable")
    from .test_runtime_routing_forward import _run

    result, _model, _inspection = _run("legacy", token_count=2)

    redacted_workspace = _workspace(tmp_path, "redacted-ws")
    redacted_receipt = append_routing_shard(
        redacted_workspace, result, store_token_text=False
    )
    redacted_bundle = tmp_path / "redacted-bundle"
    export_run_bundle(
        redacted_workspace,
        redacted_bundle,
        run_key=redacted_receipt.run_key,
    )
    redacted_lines = _member_records(redacted_bundle, "tokens.jsonl")
    assert all(record["token_text"] is None for record in redacted_lines)
    assert _manifest(redacted_bundle)["shards"][0]["token_text_stored"] is False

    stored_workspace = _workspace(tmp_path, "stored-ws")
    stored_receipt = append_routing_shard(
        stored_workspace, result, store_token_text=True
    )
    stored_bundle = tmp_path / "stored-bundle"
    export_run_bundle(stored_workspace, stored_bundle, run_key=stored_receipt.run_key)
    stored_lines = _member_records(stored_bundle, "tokens.jsonl")
    assert all(isinstance(record["token_text"], str) for record in stored_lines)
    assert [record["token_text"] for record in stored_lines] == [
        event.token_text for event in result.token_events
    ]

    imported_workspace = _workspace(tmp_path, "imported-ws")
    import_run_bundle(stored_bundle, imported_workspace)
    reexported = tmp_path / "reexported"
    reexport_receipt = export_run_bundle(
        imported_workspace,
        reexported,
        run_key=stored_receipt.run_key,
    )
    verified = verify_run_bundle(stored_bundle)
    assert reexport_receipt.manifest_sha256 == verified.manifest_sha256
    assert (reexported / "manifest.json").read_bytes() == (
        stored_bundle / "manifest.json"
    ).read_bytes()


def test_tampered_member_bytes_fail_verification(tmp_path: Path) -> None:
    if duckdb is None:
        pytest.skip("duckdb store extra is unavailable")
    _source_ws, destination, _receipt, _run_key = _export(tmp_path, stored_flags=(True,))
    member = _member(destination, "routing.jsonl")
    payload = bytearray(member.read_bytes())
    position = payload.index(b'"rank":') + len(b'"rank":')
    payload[position] = ord("9") if payload[position] != ord("9") else ord("8")
    member.write_bytes(bytes(payload))
    with pytest.raises(RunBundleError, match=r"failed at format.*digest mismatch"):
        verify_run_bundle(destination)


def test_forged_digests_still_fail_identity_recomputation(tmp_path: Path) -> None:
    if duckdb is None:
        pytest.skip("duckdb store extra is unavailable")
    _source_ws, destination, _receipt, _run_key = _export(tmp_path, stored_flags=(False,))
    member = _member(destination, "routing.jsonl")
    records = [
        json.loads(line) for line in member.read_text(encoding="utf-8").splitlines()
    ]
    target = next(record for record in records if record["router_logit"] is not None)
    target["router_logit"] = target["router_logit"] + 1.5
    tampered = _canonical_lines(records)
    member.write_bytes(tampered)
    manifest = _manifest(destination)
    info = manifest["shards"][0]["files"]["routing.jsonl"]
    info["bytes"] = len(tampered)
    info["sha256"] = "sha256:" + hashlib.sha256(tampered).hexdigest()
    (destination / "manifest.json").write_bytes(_canonical(manifest))
    with pytest.raises(
        RunBundleError,
        match=r"failed at format.*does not match its exported events",
    ):
        verify_run_bundle(destination)


def test_non_canonical_encoding_is_rejected_even_with_matching_digest(
    tmp_path: Path,
) -> None:
    if duckdb is None:
        pytest.skip("duckdb store extra is unavailable")
    _source_ws, destination, _receipt, _run_key = _export(tmp_path, stored_flags=(True,))
    member = _member(destination, "tokens.jsonl")
    records = [
        json.loads(line) for line in member.read_text(encoding="utf-8").splitlines()
    ]
    spaced = "".join(
        json.dumps(record, ensure_ascii=False, separators=(", ", ": ")) + "\n"
        for record in records
    ).encode("utf-8")
    member.write_bytes(spaced)
    manifest = _manifest(destination)
    info = manifest["shards"][0]["files"]["tokens.jsonl"]
    info["bytes"] = len(spaced)
    info["sha256"] = "sha256:" + hashlib.sha256(spaced).hexdigest()
    (destination / "manifest.json").write_bytes(_canonical(manifest))
    with pytest.raises(RunBundleError, match=r"failed at format.*not canonically encoded"):
        verify_run_bundle(destination)


def test_budgets_are_enforced_on_export_and_verify(tmp_path: Path) -> None:
    if duckdb is None:
        pytest.skip("duckdb store extra is unavailable")
    workspace, receipts, run_key = _seed_run(tmp_path, shards=1)
    total_rows = receipts[0].token_count + receipts[0].routing_count
    cramped = tmp_path / "cramped"
    with pytest.raises(RunBundleError, match=r"failed at budget"):
        export_run_bundle(
            workspace, cramped, run_key=run_key, max_event_rows=total_rows - 1
        )
    assert not cramped.exists()
    assert [item.name for item in tmp_path.iterdir() if "staging" in item.name] == []

    destination = tmp_path / "bundle"
    export_run_bundle(workspace, destination, run_key=run_key)
    with pytest.raises(RunBundleError, match=r"failed at budget"):
        verify_run_bundle(destination, max_file_bytes=1)


def test_crash_during_publication_cleans_the_stage(tmp_path: Path, monkeypatch) -> None:
    if duckdb is None:
        pytest.skip("duckdb store extra is unavailable")
    workspace, _receipts, run_key = _seed_run(tmp_path, shards=1)

    def _boom(src: object, dst: object) -> None:
        raise OSError("simulated crash")

    monkeypatch.setattr(run_export.os, "rename", _boom)
    destination = tmp_path / "bundle"
    with pytest.raises(RunBundleError, match=r"failed at publish"):
        export_run_bundle(workspace, destination, run_key=run_key)
    assert not destination.exists()
    assert [item.name for item in tmp_path.iterdir() if "staging" in item.name] == []


def test_destination_arguments_are_validated(tmp_path: Path) -> None:
    if duckdb is None:
        pytest.skip("duckdb store extra is unavailable")
    workspace, _receipts, run_key = _seed_run(tmp_path, shards=1)
    with pytest.raises(TypeError, match="destination must be a string or pathlib.Path"):
        export_run_bundle(workspace, 42, run_key=run_key)

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(RunBundleError, match=r"failed at workspace.*not empty"):
        export_run_bundle(workspace, occupied, run_key=run_key)
    assert (occupied / "keep.txt").read_text(encoding="utf-8") == "keep"

    real = tmp_path / "real-dir"
    real.mkdir()
    link = tmp_path / "symlinked"
    link.symlink_to(real)
    with pytest.raises(RunBundleError, match=r"failed at workspace.*not be a symlink"):
        export_run_bundle(workspace, link, run_key=run_key)

    orphan = tmp_path / "missing-parent" / "bundle"
    with pytest.raises(
        RunBundleError,
        match=r"failed at workspace.*parent is not an existing directory",
    ):
        export_run_bundle(workspace, orphan, run_key=run_key)

    fresh = _workspace(tmp_path, "fresh-target")
    receipt = export_run_bundle(workspace, fresh, run_key=run_key)
    assert receipt.shard_count == 1


def test_unknown_runs_and_empty_workspaces_fail_at_source(tmp_path: Path) -> None:
    if duckdb is None:
        pytest.skip("duckdb store extra is unavailable")
    workspace, _receipts, run_key = _seed_run(tmp_path, shards=1)
    del run_key
    empty = _workspace(tmp_path, "empty")
    with pytest.raises(RunBundleError, match=r"failed at source.*no committed shards"):
        export_run_bundle(empty, tmp_path / "nope", run_key="run-a")
    with pytest.raises(RunBundleError, match=r"failed at source.*no committed shards"):
        export_run_bundle(workspace, tmp_path / "nope", run_key="run-zzz")


def test_symlinked_members_are_refused(tmp_path: Path) -> None:
    if duckdb is None:
        pytest.skip("duckdb store extra is unavailable")
    _source_ws, destination, _receipt, _run_key = _export(tmp_path, stored_flags=(True,))
    member = _member(destination, "tokens.jsonl")
    outside = tmp_path / "outside-tokens.jsonl"
    outside.write_bytes(member.read_bytes())
    member.unlink()
    member.symlink_to(outside)
    with pytest.raises(RunBundleError, match=r"failed at format.*not a regular file"):
        verify_run_bundle(destination)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda m: m.pop("writer_name"), "fields are not exact"),
        (lambda m: m.update(extra=1), "fields are not exact"),
        (lambda m: m.update(manifest_type="analysis_bundle"), "type is unsupported"),
        (lambda m: m.update(schema_version="9.9"), "schema version is unsupported"),
        (lambda m: m.update(run_key="bad key!"), "run identity is invalid"),
        (lambda m: m["shards"][0].update(token_count=0), "count 'token_count' is invalid"),
        (lambda m: m["shards"][0]["files"].pop("tokens.jsonl"), "file set is not exact"),
    ],
)
def test_manifest_shape_violations_are_rejected(
    tmp_path: Path, mutate, message: str
) -> None:
    if duckdb is None:
        pytest.skip("duckdb store extra is unavailable")
    _source_ws, destination, _receipt, _run_key = _export(tmp_path, stored_flags=(False,))
    manifest = _manifest(destination)
    mutate(manifest)
    (destination / "manifest.json").write_bytes(_canonical(manifest))
    with pytest.raises(RunBundleError, match=f"failed at format.*{message}"):
        verify_run_bundle(destination)


def test_tampered_import_never_touches_the_target(tmp_path: Path) -> None:
    if duckdb is None:
        pytest.skip("duckdb store extra is unavailable")
    _source_ws, destination, _receipt, _run_key = _export(tmp_path, stored_flags=(False,))
    manifest = _manifest(destination)
    manifest["token_count"] = manifest["token_count"] + 5
    (destination / "manifest.json").write_bytes(_canonical(manifest))
    target = _workspace(tmp_path, "target")
    with pytest.raises(RunBundleError, match=r"failed at format.*totals do not match"):
        import_run_bundle(destination, target)
    assert not (target / "routing").exists()


def test_verify_does_not_require_duckdb_but_import_does(
    tmp_path: Path, monkeypatch
) -> None:
    if duckdb is None:
        pytest.skip("duckdb store extra is unavailable")
    _source_ws, destination, _receipt, _run_key = _export(tmp_path, stored_flags=(False,))

    def _absent() -> object:
        raise RoutingShardError("dependency")

    monkeypatch.setattr(run_export, "_load_duckdb", _absent)
    verified = verify_run_bundle(destination)
    assert verified.shard_count == 1

    target = _workspace(tmp_path, "target")
    with pytest.raises(RunBundleError, match=r"failed at dependency"):
        import_run_bundle(destination, target)


def test_module_imports_without_model_or_store_engines() -> None:
    script = "\n".join(
        [
            "import sys",
            "for name in ('torch', 'transformers', 'safetensors', 'duckdb'):",
            "    sys.modules[name] = None",
            "import moeatlas.store.run_export",
            "print('run-export-import-ok')",
        ]
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert completed.returncode == 0, completed.stderr
    assert "run-export-import-ok" in completed.stdout
