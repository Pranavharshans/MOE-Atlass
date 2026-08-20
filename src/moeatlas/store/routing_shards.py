"""Bounded, immutable-on-publish DuckDB routing-shard storage."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ..core import stable_digest, validate_stable_identifier
from ..events import EVENT_SCHEMA_VERSION, RoutingEvent, TokenEvent
from ..runtime.routing_forward import (
    RoutingForwardResult,
    _fresh_routing_events,
    _fresh_token_events,
    _validate_routing_links,
)

STORE_SCHEMA_VERSION = "1.0"
ROUTING_RUN_INVENTORY_SCHEMA_VERSION = "1.0"

_DUCKDB_MIN = (1, 4, 5)
_DUCKDB_MAX = (1, 5, 0)
_SHARD_KEY = re.compile(r"^shard:[0-9a-f]{64}$")
_RUN_PREFIX = "run-"
_SHARD_PREFIX = "shard-"
_ROUTING_ROOT = "routing"
_ROUTING_VERSION = "v1"
_TOKENS_FILE = "tokens.parquet"
_ROUTING_FILE = "routing.parquet"
_MANIFEST_FILE = "manifest.json"
_FINAL_NAMES = frozenset({_MANIFEST_FILE, _TOKENS_FILE, _ROUTING_FILE})
_STAGES = frozenset({"dependency", "workspace", "write", "publish", "reopen", "conflict"})
_STAGING_NAME = re.compile(r"^\.staging-[A-Za-z0-9]+$")
_FILE_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")

_TOKEN_COLUMNS = (
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
)
_ROUTING_COLUMNS = (
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
)
_MANIFEST_KEYS = frozenset(
    {
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
)
_FILE_INFO_KEYS = frozenset({"name", "bytes", "sha256"})


class RoutingShardError(RuntimeError):
    """Safe fixed-stage failure for bounded routing-shard persistence."""

    def __init__(self, stage: str) -> None:
        if stage not in _STAGES:
            raise ValueError("routing shard error stage is not supported")
        self.stage = stage
        super().__init__(f"routing shard failed at {stage}")


class RoutingRunInventoryError(RuntimeError):
    """Safe fixed-stage failure for the bounded run inventory."""

    def __init__(self, stage: Literal["budget", "index"]) -> None:
        if stage not in {"budget", "index"}:
            raise ValueError("routing run inventory error stage is not supported")
        self.stage = stage
        super().__init__(f"routing run inventory failed at {stage}")


@dataclass(frozen=True, slots=True)
class MixtralRoutingRunSummary:
    """Immutable summary of the committed shards belonging to one run."""

    run_key: str
    shard_keys: tuple[str, ...]
    shard_count: int
    token_count: int
    routing_count: int
    source_bytes: int
    token_text_policy: str

    def __post_init__(self) -> None:
        if type(self.run_key) is not str:
            raise TypeError("run_key must be an exact string")
        validate_stable_identifier(self.run_key, field_name="run_key")
        if type(self.shard_keys) is not tuple or not self.shard_keys:
            raise TypeError("shard_keys must be a non-empty tuple")
        if any(
            type(key) is not str or _SHARD_KEY.fullmatch(key) is None for key in self.shard_keys
        ):
            raise ValueError("shard_keys must be canonical shard digests")
        if tuple(sorted(set(self.shard_keys))) != self.shard_keys:
            raise ValueError("shard_keys must be sorted and unique")
        for name in ("shard_count", "token_count", "routing_count", "source_bytes"):
            value = getattr(self, name)
            if type(value) is not int or isinstance(value, bool):
                raise TypeError(f"{name} must be a strict integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.shard_count != len(self.shard_keys):
            raise ValueError("shard_count must match shard_keys")
        if type(self.token_text_policy) is not str:
            raise TypeError("token_text_policy must be an exact string")
        if self.token_text_policy not in {"redacted", "stored", "mixed"}:
            raise ValueError("token_text_policy is unsupported")

    def to_dict(self) -> dict[str, object]:
        return {
            "run_key": self.run_key,
            "shard_keys": list(self.shard_keys),
            "shard_count": self.shard_count,
            "token_count": self.token_count,
            "routing_count": self.routing_count,
            "source_bytes": self.source_bytes,
            "token_text_policy": self.token_text_policy,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass(frozen=True, slots=True)
class MixtralRoutingRunInventory:
    """Immutable bounded inventory of all committed routing shards."""

    schema_version: str
    manifest_type: str
    store_schema_version: str
    event_schema_version: str
    run_count: int
    shard_count: int
    token_count: int
    routing_count: int
    source_bytes: int
    runs: tuple[MixtralRoutingRunSummary, ...]

    def __post_init__(self) -> None:
        for name in (
            "schema_version",
            "manifest_type",
            "store_schema_version",
            "event_schema_version",
        ):
            if type(getattr(self, name)) is not str:
                raise TypeError(f"{name} must be an exact string")
        if self.schema_version != ROUTING_RUN_INVENTORY_SCHEMA_VERSION:
            raise ValueError("schema_version is unsupported")
        if self.manifest_type != "mixtral_routing_run_inventory":
            raise ValueError("manifest_type is unsupported")
        if self.store_schema_version != STORE_SCHEMA_VERSION:
            raise ValueError("store_schema_version is unsupported")
        if self.event_schema_version != EVENT_SCHEMA_VERSION:
            raise ValueError("event_schema_version is unsupported")
        if type(self.runs) is not tuple or any(
            type(run) is not MixtralRoutingRunSummary for run in self.runs
        ):
            raise TypeError("runs must be a tuple of exact run summaries")
        for name in ("run_count", "shard_count", "token_count", "routing_count", "source_bytes"):
            value = getattr(self, name)
            if type(value) is not int or isinstance(value, bool):
                raise TypeError(f"{name} must be a strict integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.run_count != len(self.runs):
            raise ValueError("run_count must match runs")
        run_keys = tuple(run.run_key for run in self.runs)
        if len(set(run_keys)) != len(run_keys) or run_keys != tuple(sorted(run_keys)):
            raise ValueError("runs must be sorted by run_key")
        if self.shard_count != sum(run.shard_count for run in self.runs):
            raise ValueError("shard_count must match runs")
        if self.token_count != sum(run.token_count for run in self.runs):
            raise ValueError("token_count must match runs")
        if self.routing_count != sum(run.routing_count for run in self.runs):
            raise ValueError("routing_count must match runs")
        if self.source_bytes != sum(run.source_bytes for run in self.runs):
            raise ValueError("source_bytes must match runs")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "manifest_type": self.manifest_type,
            "store_schema_version": self.store_schema_version,
            "event_schema_version": self.event_schema_version,
            "run_count": self.run_count,
            "shard_count": self.shard_count,
            "token_count": self.token_count,
            "routing_count": self.routing_count,
            "source_bytes": self.source_bytes,
            "runs": [run.to_dict() for run in self.runs],
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass(frozen=True, slots=True)
class RoutingShardReceipt:
    """Portable receipt for one immutable routing shard."""

    schema_version: str
    shard_key: str
    run_key: str
    relative_path: str
    token_count: int
    routing_count: int
    token_text_stored: bool
    created: bool

    def __post_init__(self) -> None:
        if type(self.schema_version) is not str or self.schema_version != STORE_SCHEMA_VERSION:
            raise ValueError("schema_version must be the exact store schema version")
        if type(self.shard_key) is not str or _SHARD_KEY.fullmatch(self.shard_key) is None:
            raise ValueError("shard_key must be a canonical shard digest")
        if type(self.run_key) is not str:
            raise TypeError("run_key must be a string")
        validate_stable_identifier(self.run_key, field_name="run_key")
        if type(self.relative_path) is not str or not self.relative_path:
            raise ValueError("relative_path must be non-empty")
        path_parts = self.relative_path.split("/")
        if (
            "\\" in self.relative_path
            or any(part in {"", ".", ".."} for part in path_parts)
            or self.relative_path.startswith("/")
        ):
            raise ValueError("relative_path must be a safe relative shard path")
        if self.relative_path != _relative_path(self.run_key, self.shard_key):
            raise ValueError("relative_path must be the canonical shard path")
        if type(self.token_count) is not int or isinstance(self.token_count, bool):
            raise TypeError("token_count must be a strict integer")
        if type(self.routing_count) is not int or isinstance(self.routing_count, bool):
            raise TypeError("routing_count must be a strict integer")
        if self.token_count <= 0 or self.routing_count <= 0:
            raise ValueError("receipt counts must be positive")
        if type(self.token_text_stored) is not bool or type(self.created) is not bool:
            raise TypeError("receipt booleans must be exact bool values")


@dataclass(frozen=True, slots=True)
class _ShardData:
    receipt: RoutingShardReceipt
    token_rows: tuple[tuple[object, ...], ...]
    routing_rows: tuple[tuple[object, ...], ...]
    token_keys: frozenset[str]
    routing_links: frozenset[tuple[str, str, int]]


def _error(stage: str, cause: BaseException) -> RoutingShardError:
    error = RoutingShardError(stage)
    error.__cause__ = cause
    return error


def _validate_workspace(value: str | Path) -> Path:
    if not isinstance(value, str | Path):
        raise TypeError("workspace must be a string or pathlib.Path")
    try:
        path = Path(value)
    except (TypeError, ValueError):
        raise
    except Exception as exc:
        raise _error("workspace", exc)
    try:
        invalid = path.is_symlink() or not path.exists() or not path.is_dir()
    except Exception as exc:
        raise _error("workspace", exc)
    if invalid:
        raise _error("workspace", ValueError("workspace is not an existing directory"))
    return path


def _validate_run_key(value: object) -> str:
    if type(value) is not str:
        raise TypeError("run_key must be an exact string")
    return validate_stable_identifier(value, field_name="run_key")


def _validate_store_token_text(value: object) -> bool:
    if type(value) is not bool:
        raise TypeError("store_token_text must be an exact bool")
    return value


def _fresh_events(
    value: object,
) -> tuple[tuple[TokenEvent, ...], tuple[RoutingEvent, ...]]:
    """Freshly validate only event payloads; never touch the opaque output."""

    if type(value) is not RoutingForwardResult:
        raise TypeError("result must be an exact RoutingForwardResult")
    token_events = _fresh_token_events(value.token_events)
    routing_events = _fresh_routing_events(value.routing_events)
    _validate_routing_links(token_events, routing_events)
    return token_events, routing_events


def _run_digest(run_key: str) -> str:
    return hashlib.sha256(run_key.encode("utf-8")).hexdigest()


def _relative_path(run_key: str, shard_key: str) -> str:
    return (
        f"{_ROUTING_ROOT}/{_ROUTING_VERSION}/{_RUN_PREFIX}{_run_digest(run_key)}/"
        f"{_SHARD_PREFIX}{shard_key.removeprefix('shard:')}"
    )


def _semantic_rows(
    token_events: tuple[TokenEvent, ...],
    routing_events: tuple[RoutingEvent, ...],
    *,
    store_token_text: bool,
) -> tuple[
    tuple[tuple[object, ...], ...],
    tuple[tuple[object, ...], ...],
    dict[str, object],
]:
    token_rows = tuple(
        (
            index,
            event.schema_version,
            event.event_type,
            event.token_key,
            event.run_key,
            event.sequence_id,
            event.token_pos,
            event.token_id,
            event.token_text if store_token_text else None,
            store_token_text,
            event.phase.value,
        )
        for index, event in enumerate(token_events)
    )
    routing_rows = tuple(
        (
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
        for index, event in enumerate(routing_events)
    )
    semantic = {
        "store_schema_version": STORE_SCHEMA_VERSION,
        "event_schema_version": EVENT_SCHEMA_VERSION,
        "redaction": {"token_text_stored": store_token_text},
        "run_key": token_events[0].run_key,
        "tokens": [
            {
                "event_index": row[0],
                "schema_version": row[1],
                "event_type": row[2],
                "token_key": row[3],
                "run_key": row[4],
                "sequence_id": row[5],
                "token_pos": row[6],
                "token_id": row[7],
                "token_text": row[8],
                "token_text_stored": row[9],
                "phase": row[10],
            }
            for row in token_rows
        ],
        "routing": [
            {
                "event_index": row[0],
                "schema_version": row[1],
                "event_type": row[2],
                "token_key": row[3],
                "layer_key": row[4],
                "rank": row[5],
                "expert_key": row[6],
                "router_logit": row[7],
                "probability": row[8],
                "weight": row[9],
                "selected": row[10],
            }
            for row in routing_rows
        ],
    }
    return token_rows, routing_rows, semantic


def _shard_key(semantic: dict[str, object]) -> str:
    return f"shard:{stable_digest(semantic)}"


def _persisted_rows(
    token_events: tuple[TokenEvent, ...],
    routing_events: tuple[RoutingEvent, ...],
    *,
    store_token_text: bool,
    shard_key: str,
) -> tuple[tuple[tuple[object, ...], ...], tuple[tuple[object, ...], ...]]:
    token_rows = tuple(
        (
            STORE_SCHEMA_VERSION,
            shard_key,
            index,
            event.schema_version,
            event.event_type,
            event.token_key,
            event.run_key,
            event.sequence_id,
            event.token_pos,
            event.token_id,
            event.token_text if store_token_text else None,
            store_token_text,
            event.phase.value,
        )
        for index, event in enumerate(token_events)
    )
    routing_rows = tuple(
        (
            STORE_SCHEMA_VERSION,
            shard_key,
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
        for index, event in enumerate(routing_events)
    )
    return token_rows, routing_rows


def _ensure_directory(path: Path) -> None:
    try:
        if path.is_symlink():
            raise ValueError("managed directory symlink rejected")
        if path.exists():
            if not path.is_dir():
                raise ValueError("managed path is not a directory")
        else:
            path.mkdir(mode=0o700)
        if os.name == "posix":
            os.chmod(path, 0o700)
    except Exception as exc:
        raise _error("workspace", exc)


def _ensure_run_parent(workspace: Path, run_key: str) -> Path:
    root = workspace / _ROUTING_ROOT
    version = root / _ROUTING_VERSION
    run_parent = version / f"{_RUN_PREFIX}{_run_digest(run_key)}"
    _ensure_directory(root)
    _ensure_directory(version)
    _ensure_directory(run_parent)
    return run_parent


def _existing_run_parent(workspace: Path, run_key: str) -> Path | None:
    root = workspace / _ROUTING_ROOT
    version = root / _ROUTING_VERSION
    run_parent = version / f"{_RUN_PREFIX}{_run_digest(run_key)}"
    for path in (root, version, run_parent):
        try:
            if path.is_symlink():
                raise ValueError("managed path symlink rejected")
            if not path.exists():
                return None
            if not path.is_dir():
                raise ValueError("managed path is not a directory")
        except Exception as exc:
            raise _error("workspace", exc)
    return run_parent


def _load_duckdb() -> Any:
    try:
        import duckdb
    except Exception as exc:
        raise _error("dependency", exc)
    try:
        parts = tuple(int(part) for part in duckdb.__version__.split(".")[:3])
        if len(parts) != 3 or not (_DUCKDB_MIN <= parts < _DUCKDB_MAX):
            raise ValueError("unsupported duckdb version")
    except Exception as exc:
        raise _error("dependency", exc)
    return duckdb


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_parquets(
    duckdb: Any,
    stage: Path,
    token_rows: tuple[tuple[object, ...], ...],
    routing_rows: tuple[tuple[object, ...], ...],
) -> None:
    connection: Any | None = None
    try:
        connection = duckdb.connect(database=":memory:")
        connection.execute(
            "CREATE TEMP TABLE tokens (store_schema_version VARCHAR, shard_key VARCHAR, "
            "event_index BIGINT, schema_version VARCHAR, event_type VARCHAR, "
            "token_key VARCHAR, run_key VARCHAR, sequence_id VARCHAR, token_pos BIGINT, "
            "token_id BIGINT, token_text VARCHAR, token_text_stored BOOLEAN, phase VARCHAR)"
        )
        connection.execute(
            "CREATE TEMP TABLE routing (store_schema_version VARCHAR, shard_key VARCHAR, "
            "event_index BIGINT, schema_version VARCHAR, event_type VARCHAR, token_key VARCHAR, "
            "layer_key VARCHAR, rank BIGINT, expert_key VARCHAR, router_logit DOUBLE, "
            "probability DOUBLE, weight DOUBLE, selected BOOLEAN)"
        )
        connection.executemany(
            "INSERT INTO tokens VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", token_rows
        )
        connection.executemany(
            "INSERT INTO routing VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", routing_rows
        )
        connection.table("tokens").order("event_index").write_parquet(
            str(stage / _TOKENS_FILE), compression="zstd", overwrite=False
        )
        connection.table("routing").order("event_index").write_parquet(
            str(stage / _ROUTING_FILE), compression="zstd", overwrite=False
        )
    except Exception as exc:
        raise _error("write", exc)
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception as exc:
                raise _error("write", exc)


def _write_manifest(stage: Path, manifest: dict[str, object]) -> None:
    payload = _manifest_bytes(manifest)
    try:
        path = stage / _MANIFEST_FILE
        with path.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            if os.name == "posix":
                os.chmod(path, 0o600)
            os.fsync(stream.fileno())
    except Exception as exc:
        raise _error("write", exc)


def _manifest_bytes(manifest: dict[str, object]) -> bytes:
    return (
        json.dumps(
            manifest,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _manifest_for(
    run_key: str,
    shard_key: str,
    token_rows: tuple[tuple[object, ...], ...],
    routing_rows: tuple[tuple[object, ...], ...],
    stage: Path,
    duckdb: Any,
    token_text_stored: bool,
) -> dict[str, object]:
    files = {
        _TOKENS_FILE: {
            "name": _TOKENS_FILE,
            "bytes": (stage / _TOKENS_FILE).stat().st_size,
            "sha256": f"sha256:{_sha256_file(stage / _TOKENS_FILE)}",
        },
        _ROUTING_FILE: {
            "name": _ROUTING_FILE,
            "bytes": (stage / _ROUTING_FILE).stat().st_size,
            "sha256": f"sha256:{_sha256_file(stage / _ROUTING_FILE)}",
        },
    }
    return {
        "manifest_type": "routing_shard",
        "store_schema_version": STORE_SCHEMA_VERSION,
        "event_schema_version": EVENT_SCHEMA_VERSION,
        "shard_key": shard_key,
        "run_key": run_key,
        "token_text_stored": token_text_stored,
        "token_count": len(token_rows),
        "routing_count": len(routing_rows),
        "writer_name": "duckdb",
        "writer_version": duckdb.__version__,
        "files": files,
    }


def _cleanup_stage(stage: Path) -> None:
    try:
        if stage.exists() and stage.is_dir() and not stage.is_symlink():
            shutil.rmtree(stage)
    except Exception:
        return


def _strict_manifest(
    manifest: object,
    run_key: str,
    shard_key: str,
    duckdb: Any,
) -> dict[str, object]:
    if type(manifest) is not dict or set(manifest) != _MANIFEST_KEYS:
        raise ValueError("manifest shape is not exact")
    if manifest["manifest_type"] != "routing_shard":
        raise ValueError("manifest type is unsupported")
    if manifest["store_schema_version"] != STORE_SCHEMA_VERSION:
        raise ValueError("store schema version is unsupported")
    if manifest["event_schema_version"] != EVENT_SCHEMA_VERSION:
        raise ValueError("event schema version is unsupported")
    if manifest["shard_key"] != shard_key or not isinstance(shard_key, str):
        raise ValueError("manifest shard identity mismatch")
    if manifest["run_key"] != run_key:
        raise ValueError("manifest run identity mismatch")
    if type(manifest["token_count"]) is not int or isinstance(manifest["token_count"], bool):
        raise ValueError("manifest token count is invalid")
    if type(manifest["routing_count"]) is not int or isinstance(manifest["routing_count"], bool):
        raise ValueError("manifest routing count is invalid")
    if manifest["token_count"] <= 0 or manifest["routing_count"] <= 0:
        raise ValueError("manifest counts must be positive")
    if type(manifest["token_text_stored"]) is not bool:
        raise ValueError("manifest redaction value is invalid")
    if manifest["writer_name"] != "duckdb" or type(manifest["writer_version"]) is not str:
        raise ValueError("manifest writer is invalid")
    if manifest["writer_version"] != duckdb.__version__:
        parts = tuple(int(part) for part in manifest["writer_version"].split(".")[:3])
        if len(parts) != 3 or not (_DUCKDB_MIN <= parts < _DUCKDB_MAX):
            raise ValueError("manifest writer version is unsupported")
    files = manifest["files"]
    if type(files) is not dict or set(files) != {_TOKENS_FILE, _ROUTING_FILE}:
        raise ValueError("manifest file set is not exact")
    for name in (_TOKENS_FILE, _ROUTING_FILE):
        info = files[name]
        if type(info) is not dict or set(info) != _FILE_INFO_KEYS:
            raise ValueError("manifest file metadata is not exact")
        if info["name"] != name:
            raise ValueError("manifest file name is invalid")
        if type(info["bytes"]) is not int or isinstance(info["bytes"], bool) or info["bytes"] <= 0:
            raise ValueError("manifest file size is invalid")
        if type(info["sha256"]) is not str or _FILE_DIGEST.fullmatch(info["sha256"]) is None:
            raise ValueError("manifest file digest is invalid")
    return manifest


def _read_rows(
    duckdb: Any, path: Path, columns: tuple[tuple[str, str], ...]
) -> tuple[tuple[Any, ...], ...]:
    connection: Any | None = None
    try:
        connection = duckdb.connect(database=":memory:")
        description = connection.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]
        ).fetchall()
        actual = tuple((row[0], row[1]) for row in description)
        if actual != columns:
            raise ValueError("parquet schema is not exact")
        rows = connection.execute(
            "SELECT * FROM read_parquet(?) ORDER BY event_index", [str(path)]
        ).fetchall()
        return tuple(rows)
    except Exception as exc:
        raise _error("reopen", exc)
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception as exc:
                raise _error("reopen", exc)


def _read_rows_with_connection(
    connection: Any, path: Path, columns: tuple[tuple[str, str], ...]
) -> tuple[tuple[Any, ...], ...]:
    """Read and schema-check one parquet file on a caller-owned connection."""

    description = connection.execute(
        "DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]
    ).fetchall()
    actual = tuple((row[0], row[1]) for row in description)
    if actual != columns:
        raise ValueError("parquet schema is not exact")
    rows = connection.execute(
        "SELECT * FROM read_parquet(?) ORDER BY event_index", [str(path)]
    ).fetchall()
    return tuple(rows)


def _read_shard_manifest(
    shard: Path,
    run_key: str,
    duckdb: Any,
    *,
    validate_files: bool = True,
) -> tuple[dict[str, object], str]:
    """Validate fixed shard identity/manifest/files without opening parquet data."""

    try:
        if shard.is_symlink() or not shard.is_dir():
            raise ValueError("committed shard is not a directory")
        if {item.name for item in shard.iterdir()} != _FINAL_NAMES:
            raise ValueError("committed shard contains unsupported entries")
        manifest_path = shard / _MANIFEST_FILE
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError("managed shard manifest is not a regular file")
        manifest_bytes = manifest_path.read_bytes()
        if not manifest_bytes.endswith(b"\n") or manifest_bytes[:-1].endswith(b"\n"):
            raise ValueError("manifest newline is not exact")
        manifest_payload = json.loads(manifest_bytes[:-1].decode("utf-8"))
        name = shard.name
        if not name.startswith(_SHARD_PREFIX):
            raise ValueError("committed shard name is invalid")
        shard_key = f"shard:{name.removeprefix(_SHARD_PREFIX)}"
        if _SHARD_KEY.fullmatch(shard_key) is None:
            raise ValueError("committed shard key is invalid")
        manifest = _strict_manifest(manifest_payload, run_key, shard_key, duckdb)
        if manifest_bytes != _manifest_bytes(manifest):
            raise ValueError("manifest bytes are not canonical")
        if validate_files:
            _validate_file_metadata(shard, manifest)
        return manifest, shard_key
    except RoutingShardError:
        raise
    except Exception as exc:
        raise _error("reopen", exc)


def _reconstruct_shard_with_connection(
    shard: Path,
    run_key: str,
    duckdb: Any,
    connection: Any,
) -> _ShardData:
    """Reopen one shard using a caller-owned connection."""

    manifest, shard_key = _read_shard_manifest(shard, run_key, duckdb)
    try:
        token_rows = _read_rows_with_connection(connection, shard / _TOKENS_FILE, _TOKEN_COLUMNS)
        routing_rows = _read_rows_with_connection(
            connection, shard / _ROUTING_FILE, _ROUTING_COLUMNS
        )
        if (
            len(token_rows) != manifest["token_count"]
            or len(routing_rows) != manifest["routing_count"]
        ):
            raise ValueError("parquet row counts do not match manifest")
        token_events: list[TokenEvent] = []
        for index, row in enumerate(token_rows):
            _validate_token_row(row, index, shard_key, manifest["token_text_stored"])
            token_text = row[10] if manifest["token_text_stored"] else ""
            token_events.append(
                TokenEvent(
                    schema_version=row[3],
                    event_type=row[4],
                    token_key=row[5],
                    run_key=row[6],
                    sequence_id=row[7],
                    token_pos=row[8],
                    token_id=row[9],
                    token_text=token_text,
                    phase=row[12],
                )
            )
        routing_events: list[RoutingEvent] = []
        for index, row in enumerate(routing_rows):
            _validate_routing_row(row, index, shard_key)
            routing_events.append(
                RoutingEvent(
                    schema_version=row[3],
                    event_type=row[4],
                    token_key=row[5],
                    layer_key=row[6],
                    rank=row[7],
                    expert_key=row[8],
                    router_logit=row[9],
                    probability=row[10],
                    weight=row[11],
                    selected=row[12],
                )
            )
        token_events_tuple = tuple(token_events)
        routing_events_tuple = tuple(routing_events)
        _validate_routing_links(token_events_tuple, routing_events_tuple)
        _, _, semantic = _semantic_rows(
            token_events_tuple,
            routing_events_tuple,
            store_token_text=manifest["token_text_stored"],
        )
        if _shard_key(semantic) != manifest["shard_key"]:
            raise ValueError("committed shard semantic digest mismatch")
    except RoutingShardError:
        raise
    except Exception as exc:
        raise _error("reopen", exc)

    receipt = RoutingShardReceipt(
        schema_version=STORE_SCHEMA_VERSION,
        shard_key=manifest["shard_key"],
        run_key=run_key,
        relative_path=_relative_path(run_key, shard_key),
        token_count=manifest["token_count"],
        routing_count=manifest["routing_count"],
        token_text_stored=manifest["token_text_stored"],
        created=False,
    )
    return _ShardData(
        receipt=receipt,
        token_rows=token_rows,
        routing_rows=routing_rows,
        token_keys=frozenset(row[5] for row in token_rows),
        routing_links=frozenset((row[5], row[6], row[7]) for row in routing_rows),
    )


def _validate_routing_universe(
    shard: _ShardData,
    layer_keys: tuple[str, ...],
    expert_keys: tuple[tuple[str, ...], ...],
    routed_top_k: int,
) -> None:
    """Validate token-complete layer/rank coverage without exposing raw rows."""

    token_keys = tuple(row[5] for row in shard.token_rows)
    if len(token_keys) != len(set(token_keys)):
        raise ValueError("shard contains duplicate token identities")
    expected_layers = set(layer_keys)
    coverage: dict[str, dict[str, set[int]]] = {
        token_key: {layer_key: set() for layer_key in layer_keys} for token_key in token_keys
    }
    expert_coverage: dict[str, dict[str, set[str]]] = {
        token_key: {layer_key: set() for layer_key in layer_keys} for token_key in token_keys
    }
    expected_ranks = set(range(routed_top_k))
    for row in shard.routing_rows:
        token_key, layer_key, rank, expert_key = row[5], row[6], row[7], row[8]
        if token_key not in coverage or layer_key not in expected_layers:
            raise ValueError("shard routing references an unknown token or layer")
        layer_position = layer_keys.index(layer_key)
        if expert_key not in expert_keys[layer_position]:
            raise ValueError("shard routing references an unknown expert")
        if type(rank) is not int or isinstance(rank, bool) or rank not in expected_ranks:
            raise ValueError("shard routing rank is not in the exact top-k range")
        coverage[token_key][layer_key].add(rank)
        expert_coverage[token_key][layer_key].add(expert_key)
    expected_assignments = len(token_keys) * len(layer_keys) * routed_top_k
    if len(shard.routing_rows) != expected_assignments:
        raise ValueError("shard routing assignment count is incomplete")
    if any(ranks != expected_ranks for layers in coverage.values() for ranks in layers.values()):
        raise ValueError("shard routing does not cover every token-layer rank")
    if any(
        len(experts) != routed_top_k
        for layers in expert_coverage.values()
        for experts in layers.values()
    ):
        raise ValueError("shard routing does not cover distinct experts per token-layer")


def _validate_routing_load_source(
    shard: Path,
    run_key: str,
    duckdb: Any,
    connection: Any,
    layer_keys: tuple[str, ...],
    expert_keys: tuple[tuple[str, ...], ...],
    routed_top_k: int,
) -> tuple[frozenset[str], frozenset[tuple[str, str, int]]]:
    """Reopen and validate one source, returning only identity summaries."""

    data = _reconstruct_shard_with_connection(shard, run_key, duckdb, connection)
    _validate_routing_universe(data, layer_keys, expert_keys, routed_top_k)
    return data.token_keys, data.routing_links


def _validate_file_metadata(shard: Path, manifest: dict[str, object]) -> None:
    files = manifest["files"]
    if type(files) is not dict:
        raise ValueError("manifest file metadata is not an object")
    for name in (_TOKENS_FILE, _ROUTING_FILE):
        path = shard / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("managed shard file is not a regular file")
        info = files[name]
        if type(info) is not dict:
            raise ValueError("manifest file metadata is not an object")
        if path.stat().st_size != info["bytes"] or f"sha256:{_sha256_file(path)}" != info["sha256"]:
            raise ValueError("managed shard file checksum mismatch")


def _validate_token_row(
    row: tuple[Any, ...], index: int, shard_key: str, token_text_stored: bool
) -> None:
    if len(row) != len(_TOKEN_COLUMNS):
        raise ValueError("token row shape is not exact")
    if row[0] != STORE_SCHEMA_VERSION or row[1] != shard_key or row[2] != index:
        raise ValueError("token row identity is invalid")
    if row[3] != EVENT_SCHEMA_VERSION or row[4] != "token":
        raise ValueError("token row schema identity is invalid")
    if type(row[11]) is not bool or row[11] is not token_text_stored:
        raise ValueError("token row redaction identity is invalid")
    if token_text_stored:
        if type(row[10]) is not str:
            raise ValueError("stored token text is invalid")
    elif row[10] is not None:
        raise ValueError("redacted token text is not null")


def _validate_routing_row(row: tuple[Any, ...], index: int, shard_key: str) -> None:
    if len(row) != len(_ROUTING_COLUMNS):
        raise ValueError("routing row shape is not exact")
    if row[0] != STORE_SCHEMA_VERSION or row[1] != shard_key or row[2] != index:
        raise ValueError("routing row identity is invalid")
    if row[3] != EVENT_SCHEMA_VERSION or row[4] != "routing":
        raise ValueError("routing row schema identity is invalid")
    if type(row[12]) is not bool or row[12] is not True:
        raise ValueError("routing row selection is invalid")


def _reconstruct_shard(
    shard: Path,
    run_key: str,
    duckdb: Any,
) -> _ShardData:
    try:
        if shard.is_symlink() or not shard.is_dir():
            raise ValueError("committed shard is not a directory")
        if {item.name for item in shard.iterdir()} != _FINAL_NAMES:
            raise ValueError("committed shard contains unsupported entries")
        manifest_path = shard / _MANIFEST_FILE
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError("managed shard manifest is not a regular file")
        manifest_bytes = manifest_path.read_bytes()
        if not manifest_bytes.endswith(b"\n") or manifest_bytes[:-1].endswith(b"\n"):
            raise ValueError("manifest newline is not exact")
        manifest_payload = json.loads(manifest_bytes[:-1].decode("utf-8"))
        name = shard.name
        if not name.startswith(_SHARD_PREFIX):
            raise ValueError("committed shard name is invalid")
        shard_key = f"shard:{name.removeprefix(_SHARD_PREFIX)}"
        if _SHARD_KEY.fullmatch(shard_key) is None:
            raise ValueError("committed shard key is invalid")
        manifest = _strict_manifest(manifest_payload, run_key, shard_key, duckdb)
        if manifest_bytes != _manifest_bytes(manifest):
            raise ValueError("manifest bytes are not canonical")
        _validate_file_metadata(shard, manifest)
    except RoutingShardError:
        raise
    except Exception as exc:
        raise _error("reopen", exc)

    token_rows = _read_rows(duckdb, shard / _TOKENS_FILE, _TOKEN_COLUMNS)
    routing_rows = _read_rows(duckdb, shard / _ROUTING_FILE, _ROUTING_COLUMNS)
    try:
        if (
            len(token_rows) != manifest["token_count"]
            or len(routing_rows) != manifest["routing_count"]
        ):
            raise ValueError("parquet row counts do not match manifest")
        token_events: list[TokenEvent] = []
        for index, row in enumerate(token_rows):
            _validate_token_row(row, index, shard_key, manifest["token_text_stored"])
            token_text = row[10] if manifest["token_text_stored"] else ""
            token_events.append(
                TokenEvent(
                    schema_version=row[3],
                    event_type=row[4],
                    token_key=row[5],
                    run_key=row[6],
                    sequence_id=row[7],
                    token_pos=row[8],
                    token_id=row[9],
                    token_text=token_text,
                    phase=row[12],
                )
            )
        routing_events: list[RoutingEvent] = []
        for index, row in enumerate(routing_rows):
            _validate_routing_row(row, index, shard_key)
            routing_events.append(
                RoutingEvent(
                    schema_version=row[3],
                    event_type=row[4],
                    token_key=row[5],
                    layer_key=row[6],
                    rank=row[7],
                    expert_key=row[8],
                    router_logit=row[9],
                    probability=row[10],
                    weight=row[11],
                    selected=row[12],
                )
            )
        token_events_tuple = tuple(token_events)
        routing_events_tuple = tuple(routing_events)
        _validate_routing_links(token_events_tuple, routing_events_tuple)
        _, _, semantic = _semantic_rows(
            token_events_tuple,
            routing_events_tuple,
            store_token_text=manifest["token_text_stored"],
        )
        if _shard_key(semantic) != manifest["shard_key"]:
            raise ValueError("committed shard semantic digest mismatch")
    except RoutingShardError:
        raise
    except Exception as exc:
        raise _error("reopen", exc)

    receipt = RoutingShardReceipt(
        schema_version=STORE_SCHEMA_VERSION,
        shard_key=manifest["shard_key"],
        run_key=run_key,
        relative_path=_relative_path(run_key, shard_key),
        token_count=manifest["token_count"],
        routing_count=manifest["routing_count"],
        token_text_stored=manifest["token_text_stored"],
        created=False,
    )
    return _ShardData(
        receipt=receipt,
        token_rows=token_rows,
        routing_rows=routing_rows,
        token_keys=frozenset(row[5] for row in token_rows),
        routing_links=frozenset((row[5], row[6], row[7]) for row in routing_rows),
    )


def _existing_shards(run_parent: Path, run_key: str, duckdb: Any) -> tuple[_ShardData, ...]:
    try:
        children = tuple(run_parent.iterdir())
    except Exception as exc:
        raise _error("reopen", exc)
    shards: list[_ShardData] = []
    for child in children:
        if _STAGING_NAME.fullmatch(child.name):
            try:
                valid_crash_stage = not child.is_symlink() and child.is_dir()
            except Exception as exc:
                raise _error("reopen", exc)
            if not valid_crash_stage:
                raise _error("reopen", ValueError("managed staging entry is invalid"))
            continue
        if child.name.startswith(".staging-"):
            raise _error("reopen", ValueError("managed staging entry name is invalid"))
        if not child.name.startswith(_SHARD_PREFIX):
            raise _error("reopen", ValueError("managed run directory contains an extra entry"))
        shards.append(_reconstruct_shard(child, run_key, duckdb))
    shards.sort(key=lambda item: item.receipt.shard_key)
    seen_tokens: set[str] = set()
    seen_links: set[tuple[str, str, int]] = set()
    for shard in shards:
        if seen_tokens.intersection(shard.token_keys) or seen_links.intersection(
            shard.routing_links
        ):
            raise _error("conflict", ValueError("committed shards overlap identities"))
        seen_tokens.update(shard.token_keys)
        seen_links.update(shard.routing_links)
    return tuple(shards)


def _append_internal(
    workspace: Path,
    token_events: tuple[TokenEvent, ...],
    routing_events: tuple[RoutingEvent, ...],
    *,
    store_token_text: bool,
    duckdb: Any,
) -> RoutingShardReceipt:
    run_key = token_events[0].run_key
    token_rows, routing_rows, semantic = _semantic_rows(
        token_events,
        routing_events,
        store_token_text=store_token_text,
    )
    shard_key = _shard_key(semantic)
    token_rows, routing_rows = _persisted_rows(
        token_events,
        routing_events,
        store_token_text=store_token_text,
        shard_key=shard_key,
    )
    run_parent = _ensure_run_parent(workspace, run_key)
    final = workspace / _relative_path(run_key, shard_key)
    try:
        final_exists = final.exists()
        final_symlink = final.is_symlink()
    except Exception as exc:
        raise _error("workspace", exc)
    if final_exists or final_symlink:
        try:
            final_is_dir = final.is_dir()
        except Exception as exc:
            raise _error("workspace", exc)
        if final_symlink or not final_is_dir:
            raise _error("workspace", ValueError("final shard path collision"))
    existing = _existing_shards(run_parent, run_key, duckdb)
    for shard in existing:
        if shard.receipt.shard_key == shard_key:
            return shard.receipt
        if set(row[5] for row in token_rows).intersection(shard.token_keys) or set(
            (row[5], row[6], row[7]) for row in routing_rows
        ).intersection(shard.routing_links):
            raise _error("conflict", ValueError("new shard overlaps a committed identity"))

    if final_exists or final_symlink:
        if final_symlink or not final_is_dir:
            raise _error("workspace", ValueError("final shard path collision"))
        return _reconstruct_shard(final, run_key, duckdb).receipt

    try:
        stage = Path(tempfile.mkdtemp(prefix=".staging-", dir=str(run_parent)))
        if os.name == "posix":
            os.chmod(stage, 0o700)
    except Exception as exc:
        raise _error("write", exc)

    published = False
    try:
        _write_parquets(duckdb, stage, token_rows, routing_rows)
        for name in (_TOKENS_FILE, _ROUTING_FILE):
            path = stage / name
            if os.name == "posix":
                os.chmod(path, 0o600)
            _fsync_file(path)
        _fsync_directory(stage)
        manifest = _manifest_for(
            run_key,
            shard_key,
            token_rows,
            routing_rows,
            stage,
            duckdb,
            store_token_text,
        )
        _write_manifest(stage, manifest)
        _fsync_directory(stage)
    except RoutingShardError:
        _cleanup_stage(stage)
        raise
    except Exception as exc:
        _cleanup_stage(stage)
        raise _error("write", exc)

    try:
        os.rename(stage, final)
        published = True
    except Exception as exc:
        _cleanup_stage(stage)
        raise _error("publish", exc)
    if published:
        try:
            _fsync_directory(run_parent)
        except Exception as exc:
            raise _error("publish", exc)
    return RoutingShardReceipt(
        schema_version=STORE_SCHEMA_VERSION,
        shard_key=shard_key,
        run_key=run_key,
        relative_path=_relative_path(run_key, shard_key),
        token_count=len(token_rows),
        routing_count=len(routing_rows),
        token_text_stored=store_token_text,
        created=True,
    )


def append_mixtral_routing_shard(
    workspace: str | Path,
    result: RoutingForwardResult,
    *,
    store_token_text: bool = False,
) -> RoutingShardReceipt:
    """Append one complete routing result as an immutable content-addressed shard."""

    path = _validate_workspace(workspace)
    token_events, routing_events = _fresh_events(result)
    store_text = _validate_store_token_text(store_token_text)
    duckdb = _load_duckdb()
    try:
        return _append_internal(
            path,
            token_events,
            routing_events,
            store_token_text=store_text,
            duckdb=duckdb,
        )
    except RoutingShardError:
        raise
    except Exception as exc:
        raise _error("write", exc)


def list_mixtral_routing_shards(
    workspace: str | Path,
    *,
    run_key: str,
) -> tuple[RoutingShardReceipt, ...]:
    """List all validated immutable routing shards for one caller run."""

    path = _validate_workspace(workspace)
    stable_run_key = _validate_run_key(run_key)
    duckdb = _load_duckdb()
    run_parent = _existing_run_parent(path, stable_run_key)
    if run_parent is None:
        return ()
    return tuple(shard.receipt for shard in _existing_shards(run_parent, stable_run_key, duckdb))


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
    if type(parsed) is not dict or set(parsed) != _MANIFEST_KEYS:
        raise ValueError("manifest shape is not exact")
    if parsed["manifest_type"] != "routing_shard":
        raise ValueError("manifest type is unsupported")
    if parsed["store_schema_version"] != STORE_SCHEMA_VERSION:
        raise ValueError("store schema version is unsupported")
    if parsed["event_schema_version"] != EVENT_SCHEMA_VERSION:
        raise ValueError("event schema version is unsupported")
    if type(parsed["run_key"]) is not str:
        raise ValueError("manifest run identity is invalid")
    validate_stable_identifier(parsed["run_key"], field_name="run_key")
    if run_key is not None and parsed["run_key"] != run_key:
        raise ValueError("run identity is inconsistent")
    if parsed["shard_key"] != shard_key or _SHARD_KEY.fullmatch(shard_key) is None:
        raise ValueError("manifest shard identity mismatch")
    for name in ("token_count", "routing_count"):
        value = parsed[name]
        if type(value) is not int or isinstance(value, bool) or value <= 0:
            raise ValueError("manifest event count is invalid")
    if type(parsed["token_text_stored"]) is not bool:
        raise ValueError("manifest redaction value is invalid")
    files = parsed["files"]
    if type(files) is not dict or set(files) != {_TOKENS_FILE, _ROUTING_FILE}:
        raise ValueError("manifest file set is not exact")
    for name in (_TOKENS_FILE, _ROUTING_FILE):
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
            if {item.name for item in child.iterdir()} != _FINAL_NAMES:
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
            for path in (token_path, routing_path):
                if path.is_symlink() or not path.is_file():
                    raise ValueError("managed shard file is not a regular file")
            source_bytes = manifest_bytes + token_path.stat().st_size + routing_path.stat().st_size
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
            event_total += int(shard.manifest["token_count"]) + int(shard.manifest["routing_count"])
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


def list_mixtral_routing_runs(
    workspace: str | Path,
    *,
    max_runs: int,
    max_shards: int,
    max_event_rows: int,
    max_source_bytes: int,
) -> MixtralRoutingRunInventory:
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
        return MixtralRoutingRunInventory(
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
    summaries: list[MixtralRoutingRunSummary] = []
    try:
        connection = duckdb.connect(database=":memory:")
        actual_events = 0
        for run_shards in indexed:
            for source in run_shards:
                token_rows = _inventory_count_rows(connection, source.shard / _TOKENS_FILE)
                routing_rows = _inventory_count_rows(connection, source.shard / _ROUTING_FILE)
                actual_events += token_rows + routing_rows
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
            for source in run_shards:
                actual = _reconstruct_shard_with_connection(
                    source.shard, source.run_key, duckdb, connection
                )
                if actual.receipt.shard_key != source.shard_key:
                    raise _error("reopen", ValueError("shard identity changed during reopen"))
                if seen_tokens.intersection(actual.token_keys) or seen_links.intersection(
                    actual.routing_links
                ):
                    raise _error("conflict", ValueError("committed shards overlap identities"))
                seen_tokens.update(actual.token_keys)
                seen_links.update(actual.routing_links)
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
                MixtralRoutingRunSummary(
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
    return MixtralRoutingRunInventory(
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


__all__ = [
    "STORE_SCHEMA_VERSION",
    "ROUTING_RUN_INVENTORY_SCHEMA_VERSION",
    "RoutingShardError",
    "RoutingShardReceipt",
    "RoutingRunInventoryError",
    "MixtralRoutingRunSummary",
    "MixtralRoutingRunInventory",
    "append_mixtral_routing_shard",
    "list_mixtral_routing_shards",
    "list_mixtral_routing_runs",
]
