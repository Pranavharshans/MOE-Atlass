"""Bounded read-only inventory over committed routing shards."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ..core import validate_stable_identifier
from ..events import EVENT_SCHEMA_VERSION
from .routing_shards import (
    _EXPERTS_FILE,
    _FILE_DIGEST,
    _FILE_INFO_KEYS,
    _MANIFEST_FILE,
    _MANIFEST_KEYS,
    _MANIFEST_KEYS_V2,
    _ROUTING_FILE,
    _ROUTING_ROOT,
    _ROUTING_VERSION,
    _RUN_PREFIX,
    _SHARD_KEY,
    _SHARD_PREFIX,
    _STAGING_NAME,
    _TOKENS_FILE,
    LEGACY_STORE_SCHEMA_VERSION,
    ROUTING_RUN_INVENTORY_SCHEMA_VERSION,
    STORE_SCHEMA_VERSION,
    RoutingRunInventory,
    RoutingRunInventoryError,
    RoutingRunSummary,
    RoutingShardError,
    _error,
    _expected_final_names,
    _load_duckdb,
    _reconstruct_shard_with_connection,
    _run_digest,
    _ShardData,
    _validate_workspace,
)


@dataclass(frozen=True, slots=True)
class _InventoryShard:
    run_key: str
    shard_key: str
    shard: Path
    manifest: dict[str, object]
    source_bytes: int


def _inventory_error(
    stage: Literal["budget", "index"], cause: BaseException
) -> RoutingRunInventoryError:
    error = RoutingRunInventoryError(stage)
    error.__cause__ = cause
    return error


def _validate_inventory_budget(value: object, name: str) -> int:
    if type(value) is not int or isinstance(value, bool):
        raise TypeError(f"{name} must be a strict positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a strict positive integer")
    return value


def _inventory_manifest_bytes(path: Path, max_source_bytes: int) -> tuple[bytes, int]:
    try:
        if path.is_symlink() or not path.is_file():
            raise ValueError("managed shard manifest is not a regular file")
        size = path.stat().st_size
        if size > max_source_bytes:
            raise RoutingRunInventoryError("budget")
        with path.open("rb") as stream:
            payload = stream.read(max_source_bytes + 1)
        if len(payload) > max_source_bytes:
            raise RoutingRunInventoryError("budget")
        return payload, size
    except RoutingRunInventoryError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        raise _error("reopen", exc)


def _inventory_basic_manifest(
    payload: bytes,
    run_key: str | None,
    shard_key: str,
) -> dict[str, object]:
    if not payload.endswith(b"\n") or payload[:-1].endswith(b"\n"):
        raise ValueError("manifest newline is not exact")
    parsed = json.loads(payload[:-1].decode("utf-8"))
    if type(parsed) is not dict:
        raise ValueError("manifest shape is not exact")
    version = parsed.get("store_schema_version")
    if version == STORE_SCHEMA_VERSION:
        expected_keys = _MANIFEST_KEYS_V2
    elif version == LEGACY_STORE_SCHEMA_VERSION:
        expected_keys = _MANIFEST_KEYS
    else:
        raise ValueError("store schema version is unsupported")
    if set(parsed) != expected_keys:
        raise ValueError("manifest shape is not exact")
    if parsed["manifest_type"] != "routing_shard":
        raise ValueError("manifest type is unsupported")
    if parsed["event_schema_version"] != EVENT_SCHEMA_VERSION:
        raise ValueError("event schema version is unsupported")
    if type(parsed["run_key"]) is not str:
        raise ValueError("manifest run identity is invalid")
    validate_stable_identifier(parsed["run_key"], field_name="run_key")
    if run_key is not None and parsed["run_key"] != run_key:
        raise ValueError("run identity is inconsistent")
    if parsed["shard_key"] != shard_key or _SHARD_KEY.fullmatch(shard_key) is None:
        raise ValueError("manifest shard identity mismatch")
    count_names = ("token_count", "routing_count")
    if version == STORE_SCHEMA_VERSION:
        count_names = (*count_names, "expert_count")
    for name in count_names:
        value = parsed[name]
        if type(value) is not int or isinstance(value, bool) or value < 0:
            raise ValueError("manifest event count is invalid")
    if parsed["token_count"] <= 0 or parsed["routing_count"] <= 0:
        raise ValueError("manifest event count is invalid")
    if type(parsed["token_text_stored"]) is not bool:
        raise ValueError("manifest redaction value is invalid")
    files = parsed["files"]
    expected_files = {_TOKENS_FILE, _ROUTING_FILE}
    if version == STORE_SCHEMA_VERSION:
        expected_files.add(_EXPERTS_FILE)
    if type(files) is not dict or set(files) != expected_files:
        raise ValueError("manifest file set is not exact")
    for name in sorted(expected_files):
        info = files[name]
        if type(info) is not dict or set(info) != _FILE_INFO_KEYS:
            raise ValueError("manifest file metadata is not exact")
        if info["name"] != name:
            raise ValueError("manifest file name is invalid")
        if type(info["bytes"]) is not int or isinstance(info["bytes"], bool) or info["bytes"] <= 0:
            raise ValueError("manifest file size is invalid")
        if type(info["sha256"]) is not str or _FILE_DIGEST.fullmatch(info["sha256"]) is None:
            raise ValueError("manifest file digest is invalid")
    return parsed


def _inventory_shards_for_run(
    run_parent: Path,
    run_digest: str,
    max_source_bytes: int,
    max_shards_remaining: int,
) -> tuple[_InventoryShard, ...]:
    try:
        children = tuple(run_parent.iterdir())
    except Exception as exc:
        raise _inventory_error("index", exc)
    committed: list[tuple[Path, str]] = []
    for child in children:
        if _STAGING_NAME.fullmatch(child.name):
            try:
                valid = not child.is_symlink() and child.is_dir()
            except Exception as exc:
                raise _inventory_error("index", exc)
            if not valid:
                raise _inventory_error("index", ValueError("managed staging entry is invalid"))
            continue
        if child.name.startswith(".staging-"):
            raise _inventory_error("index", ValueError("managed staging entry name is invalid"))
        if not child.name.startswith(_SHARD_PREFIX):
            raise _inventory_error(
                "index", ValueError("managed run directory contains an extra entry")
            )
        shard_key = f"shard:{child.name.removeprefix(_SHARD_PREFIX)}"
        if _SHARD_KEY.fullmatch(shard_key) is None:
            raise _inventory_error("index", ValueError("committed shard key is invalid"))
        try:
            if child.is_symlink() or not child.is_dir():
                raise ValueError("committed shard is not a directory")
            raw_manifest = (child / _MANIFEST_FILE).read_bytes()
            if not raw_manifest.endswith(b"\n"):
                raise ValueError("manifest newline is not exact")
            manifest_payload = json.loads(raw_manifest[:-1].decode("utf-8"))
            if {item.name for item in child.iterdir()} != _expected_final_names(manifest_payload):
                raise ValueError("committed shard contains unsupported entries")
        except Exception as exc:
            raise _error("reopen", exc)
        committed.append((child, shard_key))

    if len(committed) > max_shards_remaining:
        raise RoutingRunInventoryError("budget")

    result: list[_InventoryShard] = []
    known_run_key: str | None = None
    for shard, shard_key in sorted(committed, key=lambda item: item[1]):
        manifest_path = shard / _MANIFEST_FILE
        token_path = shard / _TOKENS_FILE
        routing_path = shard / _ROUTING_FILE
        payload, manifest_bytes = _inventory_manifest_bytes(manifest_path, max_source_bytes)
        try:
            manifest = _inventory_basic_manifest(payload, known_run_key, shard_key)
        except (KeyboardInterrupt, SystemExit):
            raise
        except RoutingRunInventoryError:
            raise
        except Exception as exc:
            raise _error("reopen", exc)
        known_run_key = str(manifest["run_key"])
        if _run_digest(known_run_key) != run_digest:
            raise _inventory_error("index", ValueError("run directory digest mismatch"))
        try:
            data_paths = [token_path, routing_path]
            if manifest["store_schema_version"] == STORE_SCHEMA_VERSION:
                data_paths.append(shard / _EXPERTS_FILE)
            for path in data_paths:
                if path.is_symlink() or not path.is_file():
                    raise ValueError("managed shard file is not a regular file")
            source_bytes = (
                manifest_bytes + sum(path.stat().st_size for path in data_paths)
            )
        except Exception as exc:
            raise _error("reopen", exc)
        result.append(
            _InventoryShard(
                run_key=known_run_key,
                shard_key=shard_key,
                shard=shard,
                manifest=manifest,
                source_bytes=source_bytes,
            )
        )
    return tuple(result)


def _inventory_index(
    workspace: Path,
    *,
    max_runs: int,
    max_shards: int,
    max_event_rows: int,
    max_source_bytes: int,
) -> tuple[tuple[_InventoryShard, ...], ...]:
    root = workspace / _ROUTING_ROOT
    version = root / _ROUTING_VERSION
    try:
        if root.is_symlink():
            raise ValueError("managed routing root is a symlink")
        if not root.exists():
            return ()
        if not root.is_dir():
            raise ValueError("managed routing root is not a directory")
        root_entries = tuple(root.iterdir())
        if not root_entries:
            return ()
        if len(root_entries) != 1 or root_entries[0].name != _ROUTING_VERSION:
            raise ValueError("routing root contains unsupported entries")
        if version.is_symlink():
            raise ValueError("managed routing version is a symlink")
        if not version.exists():
            return ()
        if not version.is_dir():
            raise ValueError("managed routing version is not a directory")
        children = tuple(version.iterdir())
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        raise _inventory_error("index", exc)

    run_dirs: list[tuple[Path, str]] = []
    for child in children:
        if not child.name.startswith(_RUN_PREFIX):
            raise _inventory_error("index", ValueError("routing version contains an extra entry"))
        digest = child.name.removeprefix(_RUN_PREFIX)
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise _inventory_error("index", ValueError("run directory name is invalid"))
        try:
            if child.is_symlink() or not child.is_dir():
                raise ValueError("run directory is not a regular directory")
        except Exception as exc:
            raise _inventory_error("index", exc)
        run_dirs.append((child, digest))
    if len(run_dirs) > max_runs:
        raise RoutingRunInventoryError("budget")

    all_runs: list[tuple[_InventoryShard, ...]] = []
    shard_total = 0
    source_total = 0
    event_total = 0
    for run_parent, digest in sorted(run_dirs, key=lambda item: item[1]):
        shards = _inventory_shards_for_run(
            run_parent,
            digest,
            max_source_bytes,
            max_shards - shard_total,
        )
        shard_total += len(shards)
        if shard_total > max_shards:
            raise RoutingRunInventoryError("budget")
        for shard in shards:
            source_total += shard.source_bytes
            event_total += (
                int(shard.manifest["token_count"])
                + int(shard.manifest["routing_count"])
                + (
                    int(shard.manifest["expert_count"])
                    if shard.manifest["store_schema_version"] == STORE_SCHEMA_VERSION
                    else 0
                )
            )
        if source_total > max_source_bytes or event_total > max_event_rows:
            raise RoutingRunInventoryError("budget")
        if shards:
            all_runs.append(shards)
    return tuple(all_runs)


def _inventory_count_rows(connection: Any, path: Path) -> int:
    try:
        row = connection.execute("SELECT COUNT(*) FROM read_parquet(?)", [str(path)]).fetchone()
        if type(row) is not tuple or len(row) != 1 or type(row[0]) is not int:
            raise ValueError("parquet count is not an integer")
        return row[0]
    except (KeyboardInterrupt, SystemExit):
        raise
    except RoutingShardError:
        raise
    except Exception as exc:
        raise _error("reopen", exc)


def _close_inventory_connection(
    connection: Any, primary: BaseException | None
) -> BaseException | None:
    close_failure: BaseException | None = None
    try:
        connection.close()
    except BaseException as first:
        try:
            connection.close()
        except BaseException as second:
            close_failure = first if isinstance(first, KeyboardInterrupt | SystemExit) else second
            if not isinstance(close_failure, KeyboardInterrupt | SystemExit):
                close_failure = _error("reopen", first)
    if close_failure is not None:
        if primary is None:
            return close_failure
        primary.add_note("routing run inventory cleanup failed")
    return primary


def list_routing_runs(
    workspace: str | Path,
    *,
    max_runs: int,
    max_shards: int,
    max_event_rows: int,
    max_source_bytes: int,
) -> RoutingRunInventory:
    """Return a bounded, read-only inventory of all committed routing runs."""

    max_runs = _validate_inventory_budget(max_runs, "max_runs")
    max_shards = _validate_inventory_budget(max_shards, "max_shards")
    max_event_rows = _validate_inventory_budget(max_event_rows, "max_event_rows")
    max_source_bytes = _validate_inventory_budget(max_source_bytes, "max_source_bytes")
    path = _validate_workspace(workspace)
    indexed = _inventory_index(
        path,
        max_runs=max_runs,
        max_shards=max_shards,
        max_event_rows=max_event_rows,
        max_source_bytes=max_source_bytes,
    )
    if not indexed:
        return RoutingRunInventory(
            schema_version=ROUTING_RUN_INVENTORY_SCHEMA_VERSION,
            manifest_type="mixtral_routing_run_inventory",
            store_schema_version=STORE_SCHEMA_VERSION,
            event_schema_version=EVENT_SCHEMA_VERSION,
            run_count=0,
            shard_count=0,
            token_count=0,
            routing_count=0,
            source_bytes=0,
            runs=(),
        )

    duckdb = _load_duckdb()
    connection: Any | None = None
    primary: BaseException | None = None
    summaries: list[RoutingRunSummary] = []
    try:
        connection = duckdb.connect(database=":memory:")
        actual_events = 0
        for run_shards in indexed:
            for source in run_shards:
                token_rows = _inventory_count_rows(connection, source.shard / _TOKENS_FILE)
                routing_rows = _inventory_count_rows(connection, source.shard / _ROUTING_FILE)
                actual_events += token_rows + routing_rows
                if source.manifest["store_schema_version"] == STORE_SCHEMA_VERSION:
                    actual_events += _inventory_count_rows(
                        connection, source.shard / _EXPERTS_FILE
                    )
                if actual_events > max_event_rows:
                    raise RoutingRunInventoryError("budget")
                if (
                    token_rows != source.manifest["token_count"]
                    or routing_rows != source.manifest["routing_count"]
                ):
                    raise _error("reopen", ValueError("parquet row counts do not match manifest"))
        for run_shards in indexed:
            data: list[_ShardData] = []
            seen_tokens: set[str] = set()
            seen_links: set[tuple[str, str, int]] = set()
            seen_expert_links: set[tuple[str, str]] = set()
            for source in run_shards:
                actual = _reconstruct_shard_with_connection(
                    source.shard, source.run_key, duckdb, connection
                )
                if actual.receipt.shard_key != source.shard_key:
                    raise _error("reopen", ValueError("shard identity changed during reopen"))
                if (
                    seen_tokens.intersection(actual.token_keys)
                    or seen_links.intersection(actual.routing_links)
                    or seen_expert_links.intersection(actual.expert_links)
                ):
                    raise _error("conflict", ValueError("committed shards overlap identities"))
                seen_tokens.update(actual.token_keys)
                seen_links.update(actual.routing_links)
                seen_expert_links.update(actual.expert_links)
                data.append(actual)
            if not data:
                continue
            run_key = data[0].receipt.run_key
            if any(item.receipt.run_key != run_key for item in data):
                raise _error("reopen", ValueError("run identity changed during reopen"))
            receipts = tuple(
                sorted((item.receipt for item in data), key=lambda item: item.shard_key)
            )
            policies = {"stored" if item.token_text_stored else "redacted" for item in receipts}
            policy = policies.pop() if len(policies) == 1 else "mixed"
            summaries.append(
                RoutingRunSummary(
                    run_key=run_key,
                    shard_keys=tuple(item.shard_key for item in receipts),
                    shard_count=len(receipts),
                    token_count=sum(item.token_count for item in receipts),
                    routing_count=sum(item.routing_count for item in receipts),
                    source_bytes=sum(source.source_bytes for source in run_shards),
                    token_text_policy=policy,
                )
            )
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt | SystemExit | RoutingShardError):
            primary = exc
        else:
            primary = _error("reopen", exc)
    finally:
        if connection is not None:
            primary = _close_inventory_connection(connection, primary)
    if primary is not None:
        raise primary
    ordered = tuple(sorted(summaries, key=lambda item: item.run_key))
    return RoutingRunInventory(
        schema_version=ROUTING_RUN_INVENTORY_SCHEMA_VERSION,
        manifest_type="mixtral_routing_run_inventory",
        store_schema_version=STORE_SCHEMA_VERSION,
        event_schema_version=EVENT_SCHEMA_VERSION,
        run_count=len(ordered),
        shard_count=sum(item.shard_count for item in ordered),
        token_count=sum(item.token_count for item in ordered),
        routing_count=sum(item.routing_count for item in ordered),
        source_bytes=sum(item.source_bytes for item in ordered),
        runs=ordered,
    )


__all__ = ["list_routing_runs"]
