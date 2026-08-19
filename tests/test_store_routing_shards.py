from __future__ import annotations

import ast
import builtins
import gc
import hashlib
import inspect
import json
import os
import shutil
import socket
import stat
import urllib.request
import weakref
from pathlib import Path

import pytest

try:
    import duckdb
except ImportError:  # pragma: no cover - exercised only without the store extra
    duckdb = None

import moeatlas.store.routing_shards as storage
from moeatlas.runtime import MixtralRoutingForwardResult
from moeatlas.store import (
    STORE_SCHEMA_VERSION,
    RoutingShardError,
    RoutingShardReceipt,
    append_mixtral_routing_shard,
    list_mixtral_routing_shards,
)


def _run_result(layout: str = "legacy", *, token_count: int = 2):
    from .test_runtime_routing_forward import _run

    result, model, inspection = _run(layout, token_count=token_count)
    return result, model, inspection


def _workspace(tmp_path: Path) -> Path:
    path = tmp_path / "workspace with spaces—演示"
    path.mkdir()
    return path


def _tree_snapshot(path: Path) -> tuple[str, ...]:
    return tuple(sorted(item.relative_to(path).as_posix() for item in path.rglob("*")))


def _manifest(receipt: RoutingShardReceipt, workspace: Path) -> dict[str, object]:
    return json.loads((workspace / receipt.relative_path / "manifest.json").read_text())


@pytest.fixture(autouse=True)
def _store_extra_required(request: pytest.FixtureRequest) -> None:
    if duckdb is None and request.node.name != "test_dependency_is_lazy_and_safe_without_duckdb":
        pytest.skip("duckdb store extra is unavailable")


def _rewrite_manifest(receipt: RoutingShardReceipt, workspace: Path, payload: object) -> None:
    (workspace / receipt.relative_path / "manifest.json").write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _refresh_manifest_file(receipt: RoutingShardReceipt, workspace: Path, filename: str) -> None:
    manifest = _manifest(receipt, workspace)
    path = workspace / receipt.relative_path / filename
    manifest["files"][filename] = {
        "name": filename,
        "bytes": path.stat().st_size,
        "sha256": f"sha256:{_sha256(path)}",
    }
    _rewrite_manifest(receipt, workspace, manifest)


def _rewrite_parquet(
    path: Path,
    columns: tuple[tuple[str, str], ...],
    rows: tuple[tuple[object, ...], ...],
) -> None:
    connection = duckdb.connect(database=":memory:")
    temporary = path.with_name(".rewrite.parquet")
    try:
        column_sql = ", ".join(f'"{name}" {column_type}' for name, column_type in columns)
        connection.execute(f"CREATE TABLE altered ({column_sql})")
        if rows:
            placeholders = ", ".join("?" for _ in columns)
            connection.executemany(f"INSERT INTO altered VALUES ({placeholders})", rows)
        connection.table("altered").write_parquet(
            str(temporary), compression="zstd", overwrite=False
        )
    finally:
        connection.close()
    if os.name == "posix":
        os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def test_dependency_is_lazy_and_safe_without_duckdb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    result, _, _ = _run_result("legacy", token_count=1)
    original_import = builtins.__import__

    def blocked_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "duckdb":
            raise ModuleNotFoundError("duckdb intentionally blocked")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(RoutingShardError) as caught:
        append_mixtral_routing_shard(workspace, result)
    assert caught.value.stage == "dependency"
    assert not (workspace / "routing").exists()


@pytest.mark.parametrize("layout", ["legacy", "packed"])
def test_public_append_list_reopen_and_idempotence(layout: str, tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    result, _, _ = _run_result(layout)
    receipt = append_mixtral_routing_shard(workspace, result)
    assert isinstance(receipt, RoutingShardReceipt)
    assert receipt.schema_version == STORE_SCHEMA_VERSION
    assert receipt.created is True
    assert receipt.relative_path.startswith("routing/v1/run-")
    shard = workspace / receipt.relative_path
    assert {path.name for path in shard.iterdir()} == {
        "manifest.json",
        "tokens.parquet",
        "routing.parquet",
    }
    if os.name == "posix":
        managed_directories = (
            workspace / "routing",
            workspace / "routing" / "v1",
            shard.parent,
            shard,
        )
        assert all(stat.S_IMODE(path.stat().st_mode) == 0o700 for path in managed_directories)
        assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in shard.iterdir())
    again = append_mixtral_routing_shard(workspace, result)
    assert again.created is False
    assert again.shard_key == receipt.shard_key
    listed = list_mixtral_routing_shards(workspace, run_key=receipt.run_key)
    assert listed == (again,)


def test_quote_unicode_space_workspace_round_trip(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace 'quoted' — 演示 with spaces"
    workspace.mkdir()
    result, _, _ = _run_result("packed", token_count=1)
    receipt = append_mixtral_routing_shard(workspace, result)
    assert list_mixtral_routing_shards(workspace, run_key=receipt.run_key)[0].shard_key == (
        receipt.shard_key
    )


def test_exact_manifest_and_physical_schemas_and_values(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    result, _, _ = _run_result("legacy", token_count=1)
    receipt = append_mixtral_routing_shard(workspace, result, store_token_text=True)
    manifest = _manifest(receipt, workspace)
    assert (workspace / receipt.relative_path / "manifest.json").read_bytes().endswith(b"\n")
    assert set(manifest) == {
        "manifest_type",
        "store_schema_version",
        "event_schema_version",
        "shard_key",
        "run_key",
        "token_text_stored",
        "token_count",
        "routing_count",
        "writer_name",
        "writer_version",
        "files",
    }
    assert manifest["manifest_type"] == "routing_shard"
    assert manifest["store_schema_version"] == STORE_SCHEMA_VERSION
    assert manifest["writer_name"] == "duckdb"
    assert manifest["token_text_stored"] is True
    assert set(manifest["files"]) == {"tokens.parquet", "routing.parquet"}
    assert all(
        set(info) == {"name", "bytes", "sha256"}
        and info["name"] == name
        and type(info["bytes"]) is int
        and info["sha256"].startswith("sha256:")
        for name, info in manifest["files"].items()
    )
    connection = duckdb.connect(database=":memory:")
    try:
        token_path = str(workspace / receipt.relative_path / "tokens.parquet")
        routing_path = str(workspace / receipt.relative_path / "routing.parquet")
        token_schema = connection.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)", [token_path]
        ).fetchall()
        routing_schema = connection.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)", [routing_path]
        ).fetchall()
        assert [(row[0], row[1]) for row in token_schema] == [
            ("store_schema_version", "VARCHAR"),
            ("shard_key", "VARCHAR"),
            ("event_index", "BIGINT"),
            ("schema_version", "VARCHAR"),
            ("event_type", "VARCHAR"),
            ("token_key", "VARCHAR"),
            ("run_key", "VARCHAR"),
            ("sequence_id", "VARCHAR"),
            ("token_pos", "BIGINT"),
            ("token_id", "BIGINT"),
            ("token_text", "VARCHAR"),
            ("token_text_stored", "BOOLEAN"),
            ("phase", "VARCHAR"),
        ]
        assert [(row[0], row[1]) for row in routing_schema] == [
            ("store_schema_version", "VARCHAR"),
            ("shard_key", "VARCHAR"),
            ("event_index", "BIGINT"),
            ("schema_version", "VARCHAR"),
            ("event_type", "VARCHAR"),
            ("token_key", "VARCHAR"),
            ("layer_key", "VARCHAR"),
            ("rank", "BIGINT"),
            ("expert_key", "VARCHAR"),
            ("router_logit", "DOUBLE"),
            ("probability", "DOUBLE"),
            ("weight", "DOUBLE"),
            ("selected", "BOOLEAN"),
        ]
        assert all(row[2] == "YES" for row in token_schema)
        assert all(row[2] == "YES" for row in routing_schema)
        token_rows = connection.execute(
            "SELECT * FROM read_parquet(?) ORDER BY event_index", [token_path]
        ).fetchall()
        routing_rows = connection.execute(
            "SELECT * FROM read_parquet(?) ORDER BY event_index", [routing_path]
        ).fetchall()
    finally:
        connection.close()
    assert token_rows[0][0] == STORE_SCHEMA_VERSION
    assert token_rows[0][1] == receipt.shard_key
    assert token_rows[0][2] == 0
    assert token_rows[0][5] == result.token_events[0].token_key
    assert token_rows[0][10] == result.token_events[0].token_text
    assert token_rows[0][11] is True
    assert routing_rows[0][0] == STORE_SCHEMA_VERSION
    assert routing_rows[0][1] == receipt.shard_key
    assert routing_rows[0][2] == 0
    assert routing_rows[0][5] == result.routing_events[0].token_key
    assert tuple(token_rows) == tuple(
        (
            STORE_SCHEMA_VERSION,
            receipt.shard_key,
            index,
            event.schema_version,
            event.event_type,
            event.token_key,
            event.run_key,
            event.sequence_id,
            event.token_pos,
            event.token_id,
            event.token_text,
            True,
            event.phase.value,
        )
        for index, event in enumerate(result.token_events)
    )
    assert tuple(routing_rows) == tuple(
        (
            STORE_SCHEMA_VERSION,
            receipt.shard_key,
            index,
            event.schema_version,
            event.event_type,
            event.token_key,
            event.layer_key,
            event.rank,
            event.expert_key,
            event.router_logit,
            event.probability,
            event.weight,
            event.selected,
        )
        for index, event in enumerate(result.routing_events)
    )


def test_redaction_is_explicit_and_content_addressed(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    result, _, _ = _run_result("legacy", token_count=1)
    redacted = append_mixtral_routing_shard(workspace, result)
    stored_parent = tmp_path / "stored"
    stored_parent.mkdir()
    stored_workspace = _workspace(stored_parent)
    stored = append_mixtral_routing_shard(stored_workspace, result, store_token_text=True)
    assert redacted.shard_key != stored.shard_key
    assert _manifest(redacted, workspace)["token_text_stored"] is False
    connection = duckdb.connect(database=":memory:")
    try:
        path = str(workspace / redacted.relative_path / "tokens.parquet")
        assert connection.execute("SELECT token_text FROM read_parquet(?)", [path]).fetchone() == (
            None,
        )
    finally:
        connection.close()


def test_conflict_duplicate_identity_and_multiple_runs(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    result, _, _ = _run_result("legacy", token_count=1)
    first = append_mixtral_routing_shard(workspace, result)
    changed_route = result.routing_events[0].model_copy(update={"router_logit": 99.0})
    changed = MixtralRoutingForwardResult(
        result.output,
        result.token_events,
        (changed_route, *result.routing_events[1:]),
    )
    with pytest.raises(RoutingShardError) as conflict:
        append_mixtral_routing_shard(workspace, changed)
    assert conflict.value.stage == "conflict"
    other_result, _, _ = _run_result("legacy", token_count=1)
    from moeatlas.events import TokenEvent

    other_tokens = tuple(
        TokenEvent(
            run_key="run-2",
            sequence_id="sequence-2",
            token_pos=event.token_pos,
            token_id=event.token_id,
            token_text=event.token_text,
            phase=event.phase,
        )
        for event in other_result.token_events
    )
    other_routes = tuple(
        route.model_copy(update={"token_key": other_tokens[0].token_key})
        for route in other_result.routing_events
    )
    other = MixtralRoutingForwardResult(other_result.output, other_tokens, other_routes)
    second = append_mixtral_routing_shard(workspace, other)
    assert second.created is True
    assert second.run_key == "run-2"
    assert tuple(
        item.shard_key for item in list_mixtral_routing_shards(workspace, run_key=first.run_key)
    ) == (first.shard_key,)
    assert tuple(
        item.shard_key for item in list_mixtral_routing_shards(workspace, run_key=second.run_key)
    ) == (second.shard_key,)


def test_two_unique_same_run_shards_are_sorted(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    first_result, _, _ = _run_result("legacy", token_count=1)
    first = append_mixtral_routing_shard(workspace, first_result)
    from moeatlas.events import TokenEvent

    token = TokenEvent(
        run_key=first_result.token_events[0].run_key,
        sequence_id="sequence-2",
        token_pos=0,
        token_id=99,
        token_text="unique",
        phase=first_result.token_events[0].phase,
    )
    routes = tuple(
        route.model_copy(update={"token_key": token.token_key})
        for route in first_result.routing_events
    )
    second_result = MixtralRoutingForwardResult(first_result.output, (token,), routes)
    second = append_mixtral_routing_shard(workspace, second_result)
    assert second.shard_key != first.shard_key
    listed = list_mixtral_routing_shards(workspace, run_key=first.run_key)
    assert tuple(item.shard_key for item in listed) == tuple(
        sorted((first.shard_key, second.shard_key))
    )


def test_corruption_and_managed_extras_are_reopen_failures(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    result, _, _ = _run_result()
    receipt = append_mixtral_routing_shard(workspace, result)
    shard = workspace / receipt.relative_path
    (shard / "extra").write_text("bad")
    with pytest.raises(RoutingShardError) as caught:
        list_mixtral_routing_shards(workspace, run_key=receipt.run_key)
    assert caught.value.stage == "reopen"


@pytest.mark.parametrize(
    "tamper",
    [
        "extra",
        "missing",
        "manifest_type",
        "store_schema_version",
        "event_schema_version",
        "shard_key",
        "run_key",
        "token_text_stored",
        "token_count",
        "routing_count",
        "writer_name",
        "writer_version",
        "manifest_newline",
        "files_extra",
        "file_name",
        "file_bytes",
        "file_hash",
    ],
)
def test_manifest_shape_identity_and_checksums_are_strict(tmp_path: Path, tamper: str) -> None:
    workspace = _workspace(tmp_path)
    result, _, _ = _run_result("legacy", token_count=1)
    receipt = append_mixtral_routing_shard(workspace, result)
    payload = _manifest(receipt, workspace)
    if tamper == "extra":
        payload["unexpected"] = True
    elif tamper == "missing":
        del payload["writer_version"]
    elif tamper == "manifest_type":
        payload["manifest_type"] = "other"
    elif tamper == "store_schema_version":
        payload["store_schema_version"] = "2.0"
    elif tamper == "event_schema_version":
        payload["event_schema_version"] = "2.0"
    elif tamper == "shard_key":
        payload["shard_key"] = "shard:" + "0" * 64
    elif tamper == "run_key":
        payload["run_key"] = "run-other"
    elif tamper == "token_text_stored":
        payload["token_text_stored"] = 1
    elif tamper == "token_count":
        payload["token_count"] = 0
    elif tamper == "routing_count":
        payload["routing_count"] = 0
    elif tamper == "writer_name":
        payload["writer_name"] = "other"
    elif tamper == "writer_version":
        payload["writer_version"] = "9.9.9"
    elif tamper == "files_extra":
        payload["files"]["extra.parquet"] = {
            "name": "extra.parquet",
            "bytes": 1,
            "sha256": "sha256:" + "0" * 64,
        }
    elif tamper == "file_name":
        payload["files"]["tokens.parquet"]["name"] = "other.parquet"
    elif tamper == "file_bytes":
        payload["files"]["tokens.parquet"]["bytes"] = 0
    elif tamper == "file_hash":
        payload["files"]["tokens.parquet"]["sha256"] = "sha256:bad"
    if tamper == "manifest_newline":
        manifest_path = workspace / receipt.relative_path / "manifest.json"
        manifest_path.write_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))
    else:
        _rewrite_manifest(receipt, workspace, payload)
    with pytest.raises(RoutingShardError) as caught:
        list_mixtral_routing_shards(workspace, run_key=receipt.run_key)
    assert caught.value.stage == "reopen"


@pytest.mark.parametrize(
    "tamper",
    [
        "wrong_token_column",
        "wrong_token_type",
        "wrong_token_order",
        "row_count",
        "noncontiguous_event_index",
        "token_store_schema_version",
        "token_shard_key",
        "token_schema_version",
        "token_event_type",
        "token_text_redaction",
        "token_text_stored_redaction",
        "invalid_token_value",
        "selected_false",
        "routing_store_schema_version",
        "routing_shard_key",
        "routing_schema_version",
        "routing_event_type",
        "semantic_token_change",
        "semantic_routing_change",
    ],
)
def test_valid_checksum_parquet_tampering_reaches_exact_reopen_validation(
    tmp_path: Path, tamper: str
) -> None:
    workspace = _workspace(tmp_path)
    store_text = tamper == "semantic_token_change"
    result, _, _ = _run_result("legacy", token_count=1)
    receipt = append_mixtral_routing_shard(workspace, result, store_token_text=store_text)
    shard = workspace / receipt.relative_path
    connection = duckdb.connect(database=":memory:")
    try:
        token_rows = tuple(
            connection.execute(
                "SELECT * FROM read_parquet(?) ORDER BY event_index",
                [str(shard / "tokens.parquet")],
            ).fetchall()
        )
        routing_rows = tuple(
            connection.execute(
                "SELECT * FROM read_parquet(?) ORDER BY event_index",
                [str(shard / "routing.parquet")],
            ).fetchall()
        )
    finally:
        connection.close()

    target = "routing.parquet"
    columns = storage._ROUTING_COLUMNS
    rows = routing_rows
    if tamper.startswith("token_") or tamper in {
        "wrong_token_column",
        "wrong_token_type",
        "wrong_token_order",
        "row_count",
        "noncontiguous_event_index",
        "invalid_token_value",
        "semantic_token_change",
    }:
        target = "tokens.parquet"
        columns = storage._TOKEN_COLUMNS
        rows = token_rows

    if target == "tokens.parquet":
        mutable = [list(row) for row in rows]
        if tamper == "wrong_token_column":
            altered_columns = list(columns)
            altered_columns[5] = ("token_key_wrong", altered_columns[5][1])
            columns = tuple(altered_columns)
        elif tamper == "wrong_token_type":
            altered_columns = list(columns)
            altered_columns[9] = ("token_id", "VARCHAR")
            columns = tuple(altered_columns)
            mutable[0][9] = str(mutable[0][9])
        elif tamper == "wrong_token_order":
            altered_columns = list(columns)
            altered_columns[3], altered_columns[4] = altered_columns[4], altered_columns[3]
            columns = tuple(altered_columns)
            mutable = [row[:3] + [row[4], row[3]] + row[5:] for row in mutable]
        elif tamper == "row_count":
            mutable = []
        elif tamper == "noncontiguous_event_index":
            mutable[0][2] = 7
        elif tamper == "token_store_schema_version":
            mutable[0][0] = "2.0"
        elif tamper == "token_shard_key":
            mutable[0][1] = "shard:" + "0" * 64
        elif tamper == "token_schema_version":
            mutable[0][3] = "2.0"
        elif tamper == "token_event_type":
            mutable[0][4] = "routing"
        elif tamper == "token_text_redaction":
            mutable[0][10] = "secret"
        elif tamper == "token_text_stored_redaction":
            mutable[0][11] = True
        elif tamper == "invalid_token_value":
            mutable[0][9] = -1
        elif tamper == "semantic_token_change":
            mutable[0][10] = "semantically changed"
        rows = tuple(tuple(row) for row in mutable)
    else:
        mutable = [list(row) for row in rows]
        if tamper == "selected_false":
            mutable[0][12] = False
        elif tamper == "routing_store_schema_version":
            mutable[0][0] = "2.0"
        elif tamper == "routing_shard_key":
            mutable[0][1] = "shard:" + "0" * 64
        elif tamper == "routing_schema_version":
            mutable[0][3] = "2.0"
        elif tamper == "routing_event_type":
            mutable[0][4] = "token"
        elif tamper == "semantic_routing_change":
            mutable[0][9] = float(mutable[0][9]) + 1.0
        rows = tuple(tuple(row) for row in mutable)

    _rewrite_parquet(shard / target, columns, rows)
    _refresh_manifest_file(receipt, workspace, target)
    with pytest.raises(RoutingShardError) as caught:
        list_mixtral_routing_shards(workspace, run_key=receipt.run_key)
    assert caught.value.stage == "reopen"


@pytest.mark.parametrize("tamper", ["byte_content", "valid_positive_size", "valid_format_digest"])
def test_file_metadata_tampering_reaches_checksum_validation(tmp_path: Path, tamper: str) -> None:
    workspace = _workspace(tmp_path)
    result, _, _ = _run_result("legacy", token_count=1)
    receipt = append_mixtral_routing_shard(workspace, result)
    shard = workspace / receipt.relative_path
    path = shard / "tokens.parquet"
    manifest = _manifest(receipt, workspace)
    original = path.read_bytes()
    if tamper == "byte_content":
        path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    elif tamper == "valid_positive_size":
        manifest["files"]["tokens.parquet"]["bytes"] = path.stat().st_size + 1
    else:
        manifest["files"]["tokens.parquet"]["sha256"] = "sha256:" + "f" * 64
    _rewrite_manifest(receipt, workspace, manifest)
    with pytest.raises(RoutingShardError) as caught:
        list_mixtral_routing_shards(workspace, run_key=receipt.run_key)
    assert caught.value.stage == "reopen"


@pytest.mark.parametrize("bad_kind", ["file", "symlink", "wrong-name"])
def test_hidden_staging_shape_is_validated(tmp_path: Path, bad_kind: str) -> None:
    workspace = _workspace(tmp_path)
    result, _, _ = _run_result("legacy", token_count=1)
    receipt = append_mixtral_routing_shard(workspace, result)
    run_parent = workspace / "/".join(receipt.relative_path.split("/")[:-1])
    if bad_kind == "file":
        (run_parent / ".staging-crash").write_text("partial")
    elif bad_kind == "symlink":
        (run_parent / ".staging-crash").symlink_to(workspace)
    else:
        (run_parent / ".staging-unsafe.name").mkdir()
    if bad_kind == "wrong-name":
        with pytest.raises(RoutingShardError) as caught:
            list_mixtral_routing_shards(workspace, run_key=receipt.run_key)
        assert caught.value.stage == "reopen"
    else:
        with pytest.raises(RoutingShardError) as caught:
            list_mixtral_routing_shards(workspace, run_key=receipt.run_key)
        assert caught.value.stage == "reopen"


def test_hidden_crash_staging_directory_is_ignored(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    result, _, _ = _run_result("legacy", token_count=1)
    receipt = append_mixtral_routing_shard(workspace, result)
    run_parent = workspace / "/".join(receipt.relative_path.split("/")[:-1])
    stage = run_parent / ".staging-crash"
    stage.mkdir()
    (stage / "partial.parquet").write_bytes(b"partial")
    listed = list_mixtral_routing_shards(workspace, run_key=receipt.run_key)
    assert tuple(item.shard_key for item in listed) == (receipt.shard_key,)


def test_managed_root_and_shard_symlink_attacks_are_rejected(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    result, _, _ = _run_result("legacy", token_count=1)
    receipt = append_mixtral_routing_shard(workspace, result)
    shard = workspace / receipt.relative_path
    token_path = shard / "tokens.parquet"
    token_backup = shard / "tokens.backup"
    token_path.rename(token_backup)
    token_path.symlink_to(token_backup)
    with pytest.raises(RoutingShardError) as caught:
        list_mixtral_routing_shards(workspace, run_key=receipt.run_key)
    assert caught.value.stage == "reopen"

    other_parent = tmp_path / "other"
    other_parent.mkdir()
    other_workspace = _workspace(other_parent)
    other_result, _, _ = _run_result("legacy", token_count=1)
    append_mixtral_routing_shard(other_workspace, other_result)
    routing_root = other_workspace / "routing"
    moved_root = other_workspace / "routing.backup"
    routing_root.rename(moved_root)
    routing_root.symlink_to(moved_root, target_is_directory=True)
    with pytest.raises(RoutingShardError) as caught:
        append_mixtral_routing_shard(other_workspace, other_result)
    assert caught.value.stage == "workspace"


def test_managed_children_and_final_nondirectory_collisions_are_rejected(tmp_path: Path) -> None:
    result, _, _ = _run_result("legacy", token_count=1)
    workspace = _workspace(tmp_path)
    (workspace / "routing").mkdir()
    (workspace / "routing" / "v1").write_text("not a directory")
    with pytest.raises(RoutingShardError) as caught:
        append_mixtral_routing_shard(workspace, result)
    assert caught.value.stage == "workspace"

    second_parent = tmp_path / "second"
    second_parent.mkdir()
    second_workspace = _workspace(second_parent)
    second_result, _, _ = _run_result("legacy", token_count=1)
    receipt = append_mixtral_routing_shard(second_workspace, second_result)
    shard = second_workspace / receipt.relative_path
    shutil.rmtree(shard)
    shard.write_text("not a shard directory")
    with pytest.raises(RoutingShardError) as caught:
        append_mixtral_routing_shard(second_workspace, second_result)
    assert caught.value.stage == "workspace"


def test_absent_subtree_list_is_empty_and_nonmutating(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    before = _tree_snapshot(workspace)
    assert list_mixtral_routing_shards(workspace, run_key="run-absent") == ()
    assert _tree_snapshot(workspace) == before


@pytest.mark.parametrize("attack", ["root", "version", "run", "shard", "manifest", "routing"])
def test_each_managed_directory_and_file_symlink_is_rejected(tmp_path: Path, attack: str) -> None:
    parent = tmp_path / attack
    parent.mkdir()
    workspace = _workspace(parent)
    result, _, _ = _run_result("legacy", token_count=1)
    receipt = append_mixtral_routing_shard(workspace, result)
    shard = workspace / receipt.relative_path
    run_parent = shard.parent
    version = run_parent.parent
    root = version.parent
    if attack == "root":
        backup = workspace / "routing-backup"
        root.rename(backup)
        root.symlink_to(backup, target_is_directory=True)
    elif attack == "version":
        backup = root / "v1-backup"
        version.rename(backup)
        version.symlink_to(backup, target_is_directory=True)
    elif attack == "run":
        backup = version / "run-backup"
        run_parent.rename(backup)
        run_parent.symlink_to(backup, target_is_directory=True)
    elif attack == "shard":
        backup = run_parent / "shard-backup"
        shard.rename(backup)
        shard.symlink_to(backup, target_is_directory=True)
    else:
        name = "manifest.json" if attack == "manifest" else "routing.parquet"
        path = shard / name
        backup = shard / f"{name}.backup"
        path.rename(backup)
        path.symlink_to(backup)
    with pytest.raises(RoutingShardError) as caught:
        list_mixtral_routing_shards(workspace, run_key=receipt.run_key)
    assert caught.value.stage in {"workspace", "reopen"}


@pytest.mark.parametrize("fsync_index", [1, 2, 3, 4, 5])
def test_prepublication_fsync_failures_clean_owned_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fsync_index: int
) -> None:
    workspace = _workspace(tmp_path)
    result, _, _ = _run_result("legacy", token_count=1)
    calls = 0
    original_fsync = storage.os.fsync

    def fail_at(fd: int) -> None:
        nonlocal calls
        calls += 1
        if calls == fsync_index:
            raise OSError("injected fsync failure")
        original_fsync(fd)

    monkeypatch.setattr(storage.os, "fsync", fail_at)
    with pytest.raises(RoutingShardError) as caught:
        append_mixtral_routing_shard(workspace, result)
    assert caught.value.stage == "write"
    assert not (workspace / "routing").exists() or not any(
        path.name.startswith("shard-") for path in workspace.rglob("shard-*")
    )
    assert not any(path.is_dir() for path in workspace.rglob(".staging-*"))


def test_manifest_write_and_parquet_write_failures_clean_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    result, _, _ = _run_result("legacy", token_count=1)
    original_write_parquets = storage._write_parquets

    def fail_write(*args: object, **kwargs: object) -> object:
        raise OSError("injected parquet failure")

    monkeypatch.setattr(storage, "_write_parquets", fail_write)
    with pytest.raises(RoutingShardError) as caught:
        append_mixtral_routing_shard(workspace, result)
    assert caught.value.stage == "write"
    assert not any(path.name.startswith(".staging-") for path in workspace.rglob("*"))

    after_parent = tmp_path / "after-parquet"
    after_parent.mkdir()
    workspace = _workspace(after_parent)

    def fail_after_write(*args: object, **kwargs: object) -> object:
        original_write_parquets(*args, **kwargs)
        raise OSError("injected post-parquet failure")

    monkeypatch.setattr(storage, "_write_parquets", fail_after_write)
    with pytest.raises(RoutingShardError) as caught:
        append_mixtral_routing_shard(workspace, result)
    assert caught.value.stage == "write"
    assert not any(path.name.startswith(".staging-") for path in workspace.rglob("*"))

    manifest_parent = tmp_path / "manifest"
    manifest_parent.mkdir()
    workspace = _workspace(manifest_parent)
    monkeypatch.setattr(storage, "_write_parquets", original_write_parquets)

    def fail_manifest(*args: object, **kwargs: object) -> object:
        raise OSError("injected manifest failure")

    monkeypatch.setattr(storage, "_write_manifest", fail_manifest)
    with pytest.raises(RoutingShardError) as caught:
        append_mixtral_routing_shard(workspace, result)
    assert caught.value.stage == "write"
    assert not any(path.name.startswith(".staging-") for path in workspace.rglob("*"))


@pytest.mark.parametrize("write_index", [1, 2])
@pytest.mark.parametrize("when", ["before", "after"])
def test_each_parquet_write_failure_cleans_owned_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    write_index: int,
    when: str,
) -> None:
    workspace = _workspace(tmp_path)
    result, _, _ = _run_result("legacy", token_count=1)
    original_connect = duckdb.connect
    calls = 0

    class RelationProxy:
        def __init__(self, relation: object) -> None:
            self._relation = relation

        def order(self, expression: str) -> RelationProxy:
            return RelationProxy(self._relation.order(expression))  # type: ignore[attr-defined]

        def write_parquet(self, path: str, **kwargs: object) -> None:
            nonlocal calls
            calls += 1
            if calls == write_index and when == "before":
                raise OSError("injected parquet write failure")
            self._relation.write_parquet(path, **kwargs)  # type: ignore[attr-defined]
            if calls == write_index and when == "after":
                raise OSError("injected parquet write failure")

    class ConnectionProxy:
        def __init__(self, connection: object) -> None:
            self._connection = connection

        def __getattr__(self, name: str) -> object:
            return getattr(self._connection, name)

        def table(self, name: str) -> RelationProxy:
            return RelationProxy(self._connection.table(name))  # type: ignore[attr-defined]

    class DuckDBProxy:
        __version__ = duckdb.__version__

        @staticmethod
        def connect(*args: object, **kwargs: object) -> ConnectionProxy:
            return ConnectionProxy(original_connect(*args, **kwargs))

    monkeypatch.setattr(storage, "_load_duckdb", lambda: DuckDBProxy)
    with pytest.raises(RoutingShardError) as caught:
        append_mixtral_routing_shard(workspace, result)
    assert caught.value.stage == "write"
    assert not any(path.name.startswith(".staging-") for path in workspace.rglob("*"))
    assert not any(path.name.startswith("shard-") for path in workspace.rglob("*"))


def test_rename_failure_is_publish_and_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    result, _, _ = _run_result("legacy", token_count=1)

    def fail_rename(*args: object, **kwargs: object) -> object:
        raise OSError("injected rename failure")

    monkeypatch.setattr(storage.os, "rename", fail_rename)
    with pytest.raises(RoutingShardError) as caught:
        append_mixtral_routing_shard(workspace, result)
    assert caught.value.stage == "publish"
    assert not any(path.name.startswith(".staging-") for path in workspace.rglob("*"))


def test_post_rename_parent_fsync_failure_recovers_by_idempotent_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    result, _, _ = _run_result("legacy", token_count=1)
    calls = 0
    original_fsync = storage.os.fsync

    def fail_parent_only(fd: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 6:
            raise OSError("injected parent fsync failure")
        original_fsync(fd)

    monkeypatch.setattr(storage.os, "fsync", fail_parent_only)
    with pytest.raises(RoutingShardError) as caught:
        append_mixtral_routing_shard(workspace, result)
    assert caught.value.stage == "publish"
    monkeypatch.setattr(storage.os, "fsync", original_fsync)
    recovered = append_mixtral_routing_shard(workspace, result)
    assert recovered.created is False


def test_duckdb_connections_close_on_write_and_reopen_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    result, _, _ = _run_result("legacy", token_count=1)
    original_connect = duckdb.connect

    class ClosingProxy:
        def __init__(self, connection: object) -> None:
            self._connection = connection

        def __getattr__(self, name: str) -> object:
            return getattr(self._connection, name)

        def close(self) -> None:
            self._connection.close()
            raise OSError("injected close failure")

    class DuckDBProxy:
        __version__ = duckdb.__version__

        @staticmethod
        def connect(*args: object, **kwargs: object) -> ClosingProxy:
            return ClosingProxy(original_connect(*args, **kwargs))

    monkeypatch.setattr(storage, "_load_duckdb", lambda: DuckDBProxy)
    with pytest.raises(RoutingShardError) as caught:
        append_mixtral_routing_shard(workspace, result)
    assert caught.value.stage == "write"
    assert not any(path.name.startswith(".staging-") for path in workspace.rglob("*"))
    monkeypatch.setattr(storage, "_load_duckdb", lambda: duckdb)
    receipt = append_mixtral_routing_shard(workspace, result)
    monkeypatch.setattr(storage, "_load_duckdb", lambda: DuckDBProxy)
    with pytest.raises(RoutingShardError) as caught:
        list_mixtral_routing_shards(workspace, run_key=receipt.run_key)
    assert caught.value.stage == "reopen"


def test_workspace_validation_and_no_write_on_preflight(tmp_path: Path) -> None:
    result, _, _ = _run_result()
    missing = tmp_path / "missing"
    with pytest.raises(RoutingShardError) as caught:
        append_mixtral_routing_shard(missing, result)
    assert caught.value.stage == "workspace"
    assert not missing.exists()
    before = _tree_snapshot(tmp_path)
    with pytest.raises(TypeError):
        append_mixtral_routing_shard(tmp_path, result, store_token_text=1)  # type: ignore[arg-type]
    assert _tree_snapshot(tmp_path) == before


def test_strict_preflight_argument_types_leave_workspace_unchanged(tmp_path: Path) -> None:
    result, _, _ = _run_result("legacy", token_count=1)
    for bad_workspace in (None, 1, object()):
        workspace_parent = tmp_path / f"workspace-{len(_tree_snapshot(tmp_path))}"
        workspace_parent.mkdir()
        workspace = _workspace(workspace_parent)
        before = _tree_snapshot(workspace)
        with pytest.raises(TypeError):
            append_mixtral_routing_shard(bad_workspace, result)  # type: ignore[arg-type]
        assert _tree_snapshot(workspace) == before

    class ResultSubclass(MixtralRoutingForwardResult):
        pass

    subclass = ResultSubclass(result.output, result.token_events, result.routing_events)
    subclass_parent = tmp_path / "subclass"
    subclass_parent.mkdir()
    workspace = _workspace(subclass_parent)
    before = _tree_snapshot(workspace)
    with pytest.raises(TypeError):
        append_mixtral_routing_shard(workspace, subclass)
    assert _tree_snapshot(workspace) == before

    not_result_parent = tmp_path / "not-result"
    not_result_parent.mkdir()
    workspace = _workspace(not_result_parent)
    before = _tree_snapshot(workspace)
    with pytest.raises(TypeError):
        append_mixtral_routing_shard(workspace, object())  # type: ignore[arg-type]
    assert _tree_snapshot(workspace) == before

    invalid_run_parent = tmp_path / "invalid-run"
    invalid_run_parent.mkdir()
    workspace = _workspace(invalid_run_parent)
    before = _tree_snapshot(workspace)
    for invalid_run_key in (None, 1, "", "  run"):
        with pytest.raises((TypeError, ValueError)):
            list_mixtral_routing_shards(workspace, run_key=invalid_run_key)  # type: ignore[arg-type]
        assert _tree_snapshot(workspace) == before


def test_fixed_error_text_and_canonical_receipt_constructor() -> None:
    for stage in ("dependency", "workspace", "write", "publish", "reopen", "conflict"):
        assert str(RoutingShardError(stage)) == f"routing shard failed at {stage}"
    from moeatlas.store.routing_shards import _relative_path

    shard_key = "shard:" + "0" * 64
    kwargs = {
        "schema_version": STORE_SCHEMA_VERSION,
        "shard_key": shard_key,
        "run_key": "run-1",
        "relative_path": _relative_path("run-1", shard_key),
        "token_count": 1,
        "routing_count": 1,
        "token_text_stored": False,
        "created": True,
    }
    assert kwargs["relative_path"] == _relative_path(kwargs["run_key"], kwargs["shard_key"])
    RoutingShardReceipt(**kwargs)
    with pytest.raises(ValueError, match="canonical"):
        RoutingShardReceipt(**(kwargs | {"relative_path": "routing/v1/not-canonical"}))


def test_receipt_is_value_only_and_payload_is_not_retained(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    class Marker:
        def __getattribute__(self, name: str) -> object:
            raise AssertionError(f"opaque output was inspected: {name}")

    from .test_runtime_routing_forward import _ForwardModel, _run

    marker = Marker()
    result, model, _ = _run("legacy", token_count=1, model=_ForwardModel("legacy", output=marker))
    output_ref = weakref.ref(marker)
    token_event_ref = weakref.ref(result.token_events[0])
    routing_event_ref = weakref.ref(result.routing_events[0])
    receipt = append_mixtral_routing_shard(workspace, result)
    assert receipt.__slots__ == (
        "schema_version",
        "shard_key",
        "run_key",
        "relative_path",
        "token_count",
        "routing_count",
        "token_text_stored",
        "created",
    )
    del result
    del model
    del marker
    gc.collect()
    assert output_ref() is None
    assert token_event_ref() is None
    assert routing_event_ref() is None
    assert receipt.created is True


def test_offline_cache_network_and_ast_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    result, _, _ = _run_result()
    cache_root = tmp_path / "empty-cache"
    temp_root = tmp_path / "empty-system-tmp"
    cache_root.mkdir()
    temp_root.mkdir()
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_root))
    monkeypatch.setenv("TMPDIR", str(temp_root))
    cache_before = tuple(path.relative_to(cache_root).as_posix() for path in cache_root.rglob("*"))
    temp_before = tuple(path.relative_to(temp_root).as_posix() for path in temp_root.rglob("*"))

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("network access forbidden")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    before = tuple(path.relative_to(workspace).as_posix() for path in workspace.rglob("*"))
    append_mixtral_routing_shard(workspace, result)
    after = tuple(path.relative_to(workspace).as_posix() for path in workspace.rglob("*"))
    assert before != after
    assert tuple(path.relative_to(cache_root).as_posix() for path in cache_root.rglob("*")) == (
        cache_before
    )
    assert tuple(path.relative_to(temp_root).as_posix() for path in temp_root.rglob("*")) == (
        temp_before
    )
    source = ast.parse(Path("src/moeatlas/store/routing_shards.py").read_text())
    assert ".output" not in Path("src/moeatlas/store/routing_shards.py").read_text()
    top_level_imports = [
        node for node in source.body if isinstance(node, ast.Import | ast.ImportFrom)
    ]
    assert all(
        not (isinstance(node, ast.Import) and any(alias.name == "duckdb" for alias in node.names))
        for node in top_level_imports
    )
    forbidden_modules = {
        "torch",
        "transformers",
        "accelerate",
        "safetensors",
        "numpy",
        "pyarrow",
        "pandas",
        "polars",
        "importlib",
    }
    for node in ast.walk(source):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] not in forbidden_modules for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in forbidden_modules
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "import_module"
        ):
            raise AssertionError("dynamic import calls are forbidden")


def test_sql_is_literal_parameterized_and_parquet_uses_relation_path_api() -> None:
    source = ast.parse(Path("src/moeatlas/store/routing_shards.py").read_text())
    for node in ast.walk(source):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr in {"execute", "executemany"}:
            assert node.args and isinstance(node.args[0], ast.Constant)
            assert isinstance(node.args[0].value, str)
            sql = node.args[0].value
            assert not isinstance(node.args[0], ast.JoinedStr)
            if "read_parquet" in sql or "INSERT INTO" in sql:
                assert "?" in sql
                assert len(node.args) >= 2
        if isinstance(node.func, ast.Attribute) and node.func.attr == "write_parquet":
            assert node.args
            path_argument = node.args[0]
            assert not isinstance(path_argument, ast.JoinedStr)
            assert isinstance(path_argument, ast.Call)
            assert isinstance(path_argument.func, ast.Name)
            assert path_argument.func.id == "str"
        if isinstance(node.func, ast.Attribute) and node.func.attr == "format":
            raise AssertionError("SQL/path formatting is forbidden")


def test_public_signatures_and_no_output_serialization() -> None:
    assert STORE_SCHEMA_VERSION == "1.0"
    assert tuple(inspect.signature(append_mixtral_routing_shard).parameters) == (
        "workspace",
        "result",
        "store_token_text",
    )
    assert (
        inspect.signature(append_mixtral_routing_shard).parameters["store_token_text"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    assert tuple(inspect.signature(list_mixtral_routing_shards).parameters) == (
        "workspace",
        "run_key",
    )
