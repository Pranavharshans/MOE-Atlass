"""Bounded, immutable-on-publish DuckDB routing-shard storage."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ..core import stable_digest, validate_stable_identifier
from ..event_validation import (
    fresh_expert_events,
    fresh_routing_events,
    fresh_token_events,
    validate_expert_links,
    validate_routing_links,
)
from ..events import EVENT_SCHEMA_VERSION, ExpertEvent, RoutingEvent, TokenEvent
from ..runtime.generic_capture import StructuredRoutingForwardResult
from ..runtime.routing_forward import RoutingForwardResult

# Historical private validation names. The neutral event_validation module is
# the implementation source; internal calls route through these module
# attributes so downstream monkeypatching keeps working exactly as before.
_fresh_token_events = fresh_token_events
_fresh_routing_events = fresh_routing_events
_fresh_expert_events = fresh_expert_events
_validate_routing_links = validate_routing_links
_validate_expert_links = validate_expert_links

STORE_SCHEMA_VERSION = "2.0"
LEGACY_STORE_SCHEMA_VERSION = "1.0"
_KNOWN_STORE_SCHEMA_VERSIONS = frozenset(
    {STORE_SCHEMA_VERSION, LEGACY_STORE_SCHEMA_VERSION}
)
ROUTING_RUN_INVENTORY_SCHEMA_VERSION = "1.0"

_DEFAULT_MAX_EXPERT_EVENTS = 65536

_DUCKDB_MIN = (1, 4, 5)
_DUCKDB_MAX = (1, 5, 0)
_SHARD_KEY = re.compile(r"^shard:[0-9a-f]{64}$")
_RUN_PREFIX = "run-"
_SHARD_PREFIX = "shard-"
_ROUTING_ROOT = "routing"
_ROUTING_VERSION = "v1"
_TOKENS_FILE = "tokens.parquet"
_ROUTING_FILE = "routing.parquet"
_EXPERTS_FILE = "experts.parquet"
_MANIFEST_FILE = "manifest.json"
_V1_FINAL_NAMES = frozenset({_MANIFEST_FILE, _TOKENS_FILE, _ROUTING_FILE})
_V2_FINAL_NAMES = _V1_FINAL_NAMES | {_EXPERTS_FILE}
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
_EXPERT_COLUMNS = (
    ("store_schema_version", "VARCHAR"),
    ("shard_key", "VARCHAR"),
    ("event_index", "BIGINT"),
    ("schema_version", "VARCHAR"),
    ("event_type", "VARCHAR"),
    ("token_key", "VARCHAR"),
    ("expert_key", "VARCHAR"),
    ("input_norm", "DOUBLE"),
    ("output_norm", "DOUBLE"),
    ("contribution_norm", "DOUBLE"),
    ("latency_ms", "DOUBLE"),
    ("metadata", "VARCHAR"),
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
_MANIFEST_KEYS_V2 = _MANIFEST_KEYS | {"expert_count"}
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
        if (
            type(self.schema_version) is not str
            or self.schema_version not in _KNOWN_STORE_SCHEMA_VERSIONS
        ):
            raise ValueError("schema_version must be a supported store schema version")
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
    store_version: str
    token_rows: tuple[tuple[object, ...], ...]
    routing_rows: tuple[tuple[object, ...], ...]
    expert_rows: tuple[tuple[object, ...], ...]
    token_keys: frozenset[str]
    routing_links: frozenset[tuple[str, str, int]]
    expert_links: frozenset[tuple[str, str]]


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
    expert_events: tuple[ExpertEvent, ...] = (),
    store_version: str = STORE_SCHEMA_VERSION,
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
        "store_schema_version": store_version,
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
    if expert_events:
        semantic["experts"] = [
            {
                "event_index": index,
                "schema_version": payload["schema_version"],
                "event_type": payload["event_type"],
                "token_key": payload["token_key"],
                "expert_key": payload["expert_key"],
                "input_norm": payload["input_norm"],
                "output_norm": payload["output_norm"],
                "contribution_norm": payload["contribution_norm"],
                "latency_ms": payload["latency_ms"],
                "metadata": payload["metadata"],
            }
            for index, payload in enumerate(
                event.model_dump(mode="json") for event in expert_events
            )
        ]
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


def _persisted_expert_rows(
    expert_events: tuple[ExpertEvent, ...],
    *,
    shard_key: str,
) -> tuple[tuple[object, ...], ...]:
    rows: list[tuple[object, ...]] = []
    for index, event in enumerate(expert_events):
        # Field values come from one JSON dump so the opaque payload boundary
        # stays symmetric with reopen-time reconstruction.
        dumped = event.model_dump(mode="json")
        metadata = dumped["metadata"]
        rows.append(
            (
                STORE_SCHEMA_VERSION,
                shard_key,
                index,
                dumped["schema_version"],
                dumped["event_type"],
                dumped["token_key"],
                dumped["expert_key"],
                dumped["input_norm"],
                dumped["output_norm"],
                dumped["contribution_norm"],
                dumped["latency_ms"],
                json.dumps(metadata, ensure_ascii=False, allow_nan=False, sort_keys=True)
                if metadata
                else None,
            )
        )
    return tuple(rows)


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
    expert_rows: tuple[tuple[object, ...], ...] = (),
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
        connection.execute(
            "CREATE TEMP TABLE experts (store_schema_version VARCHAR, shard_key VARCHAR, "
            "event_index BIGINT, schema_version VARCHAR, event_type VARCHAR, token_key VARCHAR, "
            "expert_key VARCHAR, input_norm DOUBLE, output_norm DOUBLE, "
            "contribution_norm DOUBLE, latency_ms DOUBLE, metadata VARCHAR)"
        )
        connection.executemany(
            "INSERT INTO tokens VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", token_rows
        )
        connection.executemany(
            "INSERT INTO routing VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", routing_rows
        )
        if expert_rows:
            connection.executemany(
                "INSERT INTO experts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", expert_rows
            )
        connection.table("tokens").order("event_index").write_parquet(
            str(stage / _TOKENS_FILE), compression="zstd", overwrite=False
        )
        connection.table("routing").order("event_index").write_parquet(
            str(stage / _ROUTING_FILE), compression="zstd", overwrite=False
        )
        connection.table("experts").order("event_index").write_parquet(
            str(stage / _EXPERTS_FILE), compression="zstd", overwrite=False
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
    expert_count: int = 0,
) -> dict[str, object]:
    files = {
        name: {
            "name": name,
            "bytes": (stage / name).stat().st_size,
            "sha256": f"sha256:{_sha256_file(stage / name)}",
        }
        for name in sorted(_V2_FINAL_NAMES - {_MANIFEST_FILE})
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
        "expert_count": expert_count,
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
    if type(manifest) is not dict:
        raise ValueError("manifest shape is not exact")
    version = manifest.get("store_schema_version")
    if version == STORE_SCHEMA_VERSION:
        expected_keys = _MANIFEST_KEYS_V2
    elif version == LEGACY_STORE_SCHEMA_VERSION:
        expected_keys = _MANIFEST_KEYS
    else:
        raise ValueError("store schema version is unsupported")
    if set(manifest) != expected_keys:
        raise ValueError("manifest shape is not exact")
    if manifest["manifest_type"] != "routing_shard":
        raise ValueError("manifest type is unsupported")
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
    if version == STORE_SCHEMA_VERSION:
        value = manifest["expert_count"]
        if type(value) is not int or isinstance(value, bool) or value < 0:
            raise ValueError("manifest expert count is invalid")
    if type(manifest["token_text_stored"]) is not bool:
        raise ValueError("manifest redaction value is invalid")
    if manifest["writer_name"] != "duckdb" or type(manifest["writer_version"]) is not str:
        raise ValueError("manifest writer is invalid")
    if manifest["writer_version"] != duckdb.__version__:
        parts = tuple(int(part) for part in manifest["writer_version"].split(".")[:3])
        if len(parts) != 3 or not (_DUCKDB_MIN <= parts < _DUCKDB_MAX):
            raise ValueError("manifest writer version is unsupported")
    files = manifest["files"]
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


def _expected_final_names(payload: object) -> frozenset[str]:
    """Return the exact committed entry set for a manifest's store version."""

    if type(payload) is dict and payload.get("store_schema_version") == (
        LEGACY_STORE_SCHEMA_VERSION
    ):
        return _V1_FINAL_NAMES
    return _V2_FINAL_NAMES


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
        manifest_path = shard / _MANIFEST_FILE
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError("managed shard manifest is not a regular file")
        manifest_bytes = manifest_path.read_bytes()
        if not manifest_bytes.endswith(b"\n") or manifest_bytes[:-1].endswith(b"\n"):
            raise ValueError("manifest newline is not exact")
        manifest_payload = json.loads(manifest_bytes[:-1].decode("utf-8"))
        if {item.name for item in shard.iterdir()} != _expected_final_names(manifest_payload):
            raise ValueError("committed shard contains unsupported entries")
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
    store_version = str(manifest["store_schema_version"])
    try:
        token_rows = _read_rows_with_connection(connection, shard / _TOKENS_FILE, _TOKEN_COLUMNS)
        routing_rows = _read_rows_with_connection(
            connection, shard / _ROUTING_FILE, _ROUTING_COLUMNS
        )
        expert_rows: tuple[tuple[object, ...], ...] = ()
        if store_version == STORE_SCHEMA_VERSION:
            expert_rows = _read_rows_with_connection(
                connection, shard / _EXPERTS_FILE, _EXPERT_COLUMNS
            )
            if len(expert_rows) != manifest["expert_count"]:
                raise ValueError("parquet row counts do not match manifest")
        if (
            len(token_rows) != manifest["token_count"]
            or len(routing_rows) != manifest["routing_count"]
        ):
            raise ValueError("parquet row counts do not match manifest")
        token_events: list[TokenEvent] = []
        for index, row in enumerate(token_rows):
            _validate_token_row(row, index, shard_key, manifest["token_text_stored"], store_version)
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
            _validate_routing_row(row, index, shard_key, store_version)
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
        expert_events: list[ExpertEvent] = []
        for index, row in enumerate(expert_rows):
            _validate_expert_row(row, index, shard_key)
            metadata = json.loads(row[11]) if row[11] is not None else {}
            expert_events.append(
                ExpertEvent(
                    schema_version=row[3],
                    event_type=row[4],
                    token_key=row[5],
                    expert_key=row[6],
                    input_norm=row[7],
                    output_norm=row[8],
                    contribution_norm=row[9],
                    latency_ms=row[10],
                    metadata=metadata,
                )
            )
        token_events_tuple = tuple(token_events)
        routing_events_tuple = tuple(routing_events)
        expert_events_tuple = tuple(expert_events)
        _validate_routing_links(token_events_tuple, routing_events_tuple)
        _validate_expert_links(token_events_tuple, expert_events_tuple)
        _, _, semantic = _semantic_rows(
            token_events_tuple,
            routing_events_tuple,
            store_token_text=manifest["token_text_stored"],
            expert_events=expert_events_tuple,
            store_version=store_version,
        )
        if _shard_key(semantic) != manifest["shard_key"]:
            raise ValueError("committed shard semantic digest mismatch")
    except RoutingShardError:
        raise
    except Exception as exc:
        raise _error("reopen", exc)

    receipt = RoutingShardReceipt(
        schema_version=store_version,
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
        store_version=store_version,
        token_rows=token_rows,
        routing_rows=routing_rows,
        expert_rows=expert_rows,
        token_keys=frozenset(row[5] for row in token_rows),
        routing_links=frozenset((row[5], row[6], row[7]) for row in routing_rows),
        expert_links=frozenset((row[5], row[6]) for row in expert_rows),
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
    for name in sorted(files):
        path = shard / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("managed shard file is not a regular file")
        info = files[name]
        if type(info) is not dict:
            raise ValueError("manifest file metadata is not an object")
        if path.stat().st_size != info["bytes"] or f"sha256:{_sha256_file(path)}" != info["sha256"]:
            raise ValueError("managed shard file checksum mismatch")


def _validate_token_row(
    row: tuple[Any, ...],
    index: int,
    shard_key: str,
    token_text_stored: bool,
    store_version: str,
) -> None:
    if len(row) != len(_TOKEN_COLUMNS):
        raise ValueError("token row shape is not exact")
    if row[0] != store_version or row[1] != shard_key or row[2] != index:
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


def _validate_routing_row(
    row: tuple[Any, ...], index: int, shard_key: str, store_version: str
) -> None:
    if len(row) != len(_ROUTING_COLUMNS):
        raise ValueError("routing row shape is not exact")
    if row[0] != store_version or row[1] != shard_key or row[2] != index:
        raise ValueError("routing row identity is invalid")
    if row[3] != EVENT_SCHEMA_VERSION or row[4] != "routing":
        raise ValueError("routing row schema identity is invalid")
    if type(row[12]) is not bool or row[12] is not True:
        raise ValueError("routing row selection is invalid")


def _validate_expert_row(row: tuple[Any, ...], index: int, shard_key: str) -> None:
    if len(row) != len(_EXPERT_COLUMNS):
        raise ValueError("expert row shape is not exact")
    if row[0] != STORE_SCHEMA_VERSION or row[1] != shard_key or row[2] != index:
        raise ValueError("expert row identity is invalid")
    if row[3] != EVENT_SCHEMA_VERSION or row[4] != "expert":
        raise ValueError("expert row schema identity is invalid")
    for value in row[7:11]:
        if value is not None and (type(value) is not float or not math.isfinite(value)):
            raise ValueError("expert row measurements must be finite floats or null")
    metadata = row[11]
    if metadata is not None:
        if type(metadata) is not str:
            raise ValueError("expert row metadata must be a canonical JSON string or null")
        try:
            parsed = json.loads(metadata)
        except (TypeError, ValueError) as exc:
            raise ValueError("expert row metadata must be valid JSON") from exc
        canonical = json.dumps(parsed, ensure_ascii=False, allow_nan=False, sort_keys=True)
        if canonical.encode("utf-8") != metadata.encode("utf-8"):
            raise ValueError("expert row metadata is not canonically encoded")


def _reconstruct_shard(
    shard: Path,
    run_key: str,
    duckdb: Any,
) -> _ShardData:
    try:
        if shard.is_symlink() or not shard.is_dir():
            raise ValueError("committed shard is not a directory")
        manifest_path = shard / _MANIFEST_FILE
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError("managed shard manifest is not a regular file")
        manifest_bytes = manifest_path.read_bytes()
        if not manifest_bytes.endswith(b"\n") or manifest_bytes[:-1].endswith(b"\n"):
            raise ValueError("manifest newline is not exact")
        manifest_payload = json.loads(manifest_bytes[:-1].decode("utf-8"))
        if {item.name for item in shard.iterdir()} != _expected_final_names(manifest_payload):
            raise ValueError("committed shard contains unsupported entries")
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

    store_version = str(manifest["store_schema_version"])
    token_rows = _read_rows(duckdb, shard / _TOKENS_FILE, _TOKEN_COLUMNS)
    routing_rows = _read_rows(duckdb, shard / _ROUTING_FILE, _ROUTING_COLUMNS)
    expert_rows: tuple[tuple[object, ...], ...] = ()
    if store_version == STORE_SCHEMA_VERSION:
        expert_rows = _read_rows(duckdb, shard / _EXPERTS_FILE, _EXPERT_COLUMNS)
    try:
        if (
            len(token_rows) != manifest["token_count"]
            or len(routing_rows) != manifest["routing_count"]
        ):
            raise ValueError("parquet row counts do not match manifest")
        if expert_rows and len(expert_rows) != manifest["expert_count"]:
            raise ValueError("parquet row counts do not match manifest")
        token_events: list[TokenEvent] = []
        for index, row in enumerate(token_rows):
            _validate_token_row(row, index, shard_key, manifest["token_text_stored"], store_version)
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
            _validate_routing_row(row, index, shard_key, store_version)
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
        expert_events: list[ExpertEvent] = []
        for index, row in enumerate(expert_rows):
            _validate_expert_row(row, index, shard_key)
            metadata = json.loads(row[11]) if row[11] is not None else {}
            expert_events.append(
                ExpertEvent(
                    schema_version=row[3],
                    event_type=row[4],
                    token_key=row[5],
                    expert_key=row[6],
                    input_norm=row[7],
                    output_norm=row[8],
                    contribution_norm=row[9],
                    latency_ms=row[10],
                    metadata=metadata,
                )
            )
        token_events_tuple = tuple(token_events)
        routing_events_tuple = tuple(routing_events)
        expert_events_tuple = tuple(expert_events)
        _validate_routing_links(token_events_tuple, routing_events_tuple)
        _validate_expert_links(token_events_tuple, expert_events_tuple)
        _, _, semantic = _semantic_rows(
            token_events_tuple,
            routing_events_tuple,
            store_token_text=manifest["token_text_stored"],
            expert_events=expert_events_tuple,
            store_version=store_version,
        )
        if _shard_key(semantic) != manifest["shard_key"]:
            raise ValueError("committed shard semantic digest mismatch")
    except RoutingShardError:
        raise
    except Exception as exc:
        raise _error("reopen", exc)

    receipt = RoutingShardReceipt(
        schema_version=store_version,
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
        store_version=store_version,
        token_rows=token_rows,
        routing_rows=routing_rows,
        expert_rows=expert_rows,
        token_keys=frozenset(row[5] for row in token_rows),
        routing_links=frozenset((row[5], row[6], row[7]) for row in routing_rows),
        expert_links=frozenset((row[5], row[6]) for row in expert_rows),
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
    seen_expert_links: set[tuple[str, str]] = set()
    for shard in shards:
        if (
            seen_tokens.intersection(shard.token_keys)
            or seen_links.intersection(shard.routing_links)
            or seen_expert_links.intersection(shard.expert_links)
        ):
            raise _error("conflict", ValueError("committed shards overlap identities"))
        seen_tokens.update(shard.token_keys)
        seen_links.update(shard.routing_links)
        seen_expert_links.update(shard.expert_links)
    return tuple(shards)


def _append_internal(
    workspace: Path,
    token_events: tuple[TokenEvent, ...],
    routing_events: tuple[RoutingEvent, ...],
    *,
    store_token_text: bool,
    duckdb: Any,
    expert_events: tuple[ExpertEvent, ...] = (),
) -> RoutingShardReceipt:
    run_key = token_events[0].run_key
    token_rows, routing_rows, semantic = _semantic_rows(
        token_events,
        routing_events,
        store_token_text=store_token_text,
        expert_events=expert_events,
    )
    shard_key = _shard_key(semantic)
    token_rows, routing_rows = _persisted_rows(
        token_events,
        routing_events,
        store_token_text=store_token_text,
        shard_key=shard_key,
    )
    expert_rows = _persisted_expert_rows(expert_events, shard_key=shard_key)
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
        if (
            set(row[5] for row in token_rows).intersection(shard.token_keys)
            or set((row[5], row[6], row[7]) for row in routing_rows).intersection(
                shard.routing_links
            )
            or set((row[5], row[6]) for row in expert_rows).intersection(shard.expert_links)
        ):
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
        _write_parquets(duckdb, stage, token_rows, routing_rows, expert_rows)
        for name in sorted(_V2_FINAL_NAMES - {_MANIFEST_FILE}):
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
            expert_count=len(expert_rows),
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


def append_routing_shard(
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


def _fresh_structured_experts(
    value: object,
) -> tuple[tuple[TokenEvent, ...], tuple[RoutingEvent, ...], tuple[ExpertEvent, ...]]:
    """Freshly validate a structured result including its expert events."""

    if type(value) is not StructuredRoutingForwardResult:
        raise TypeError("result must be an exact StructuredRoutingForwardResult")
    token_events = _fresh_token_events(value.token_events)
    routing_events = _fresh_routing_events(value.routing_events)
    expert_events = _fresh_expert_events(value.expert_events)
    _validate_routing_links(token_events, routing_events)
    _validate_expert_links(token_events, expert_events)
    return token_events, routing_events, expert_events


def append_structured_shard(
    workspace: str | Path,
    result: StructuredRoutingForwardResult,
    *,
    store_token_text: bool = False,
    max_expert_events: int = _DEFAULT_MAX_EXPERT_EVENTS,
) -> RoutingShardReceipt:
    """Append one structured capture result (routing plus expert events).

    Expert events ride the same immutable content-addressed shard under
    ``experts.parquet``; identity overlap against committed shards mirrors the
    routing-row conflict semantics. The strict per-shard expert budget is
    checked before any workspace mutation.
    """

    path = _validate_workspace(workspace)
    token_events, routing_events, expert_events = _fresh_structured_experts(result)
    store_text = _validate_store_token_text(store_token_text)
    if type(max_expert_events) is not int or isinstance(max_expert_events, bool):
        raise TypeError("max_expert_events must be a strict positive integer")
    if max_expert_events <= 0:
        raise ValueError("max_expert_events must be a strict positive integer")
    if len(expert_events) > max_expert_events:
        raise ValueError("expert events exceed the per-shard expert-event budget")
    duckdb = _load_duckdb()
    try:
        return _append_internal(
            path,
            token_events,
            routing_events,
            store_token_text=store_text,
            duckdb=duckdb,
            expert_events=expert_events,
        )
    except RoutingShardError:
        raise
    except Exception as exc:
        raise _error("write", exc)


def list_routing_shards(
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


class RoutingRunQueryError(RuntimeError):
    """Safe fixed failure for bounded routing-run assignment queries."""

    def __init__(self) -> None:
        super().__init__("routing run query failed")


@dataclass(frozen=True, slots=True)
class RoutingShardAssignmentQuery:
    """Validated per-shard assignment summary from one bounded run query."""

    shard_key: str
    token_count: int
    routing_count: int
    token_keys: frozenset[str]
    routing_links: frozenset[tuple[str, str, int]]
    assignment_counts: tuple[tuple[str, str, int], ...]

    def __post_init__(self) -> None:
        if type(self.shard_key) is not str or _SHARD_KEY.fullmatch(self.shard_key) is None:
            raise ValueError("shard_key must be a canonical shard digest")
        for name in ("token_count", "routing_count"):
            value = getattr(self, name)
            if type(value) is not int or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a strict positive integer")
        if type(self.token_keys) is not frozenset or not self.token_keys:
            raise ValueError("token_keys must be a non-empty frozenset")
        if type(self.routing_links) is not frozenset or not self.routing_links:
            raise ValueError("routing_links must be a non-empty frozenset")
        if type(self.assignment_counts) is not tuple or not self.assignment_counts:
            raise ValueError("assignment_counts must be a non-empty tuple")
        for entry in self.assignment_counts:
            if type(entry) is not tuple or len(entry) != 3:
                raise ValueError("assignment_counts entries must be (layer, expert, count)")
            layer_key, expert_key, count = entry
            if type(layer_key) is not str or type(expert_key) is not str:
                raise TypeError("assignment keys must be exact strings")
            if type(count) is not int or isinstance(count, bool) or count <= 0:
                raise ValueError("assignment counts must be strict positive integers")
        keys = [entry[:2] for entry in self.assignment_counts]
        if keys != sorted(keys):
            raise ValueError("assignment_counts must be sorted by layer and expert key")


def _strict_positive_query_budget(value: object, name: str) -> int:
    if type(value) is not int or isinstance(value, bool):
        raise TypeError(f"{name} must be a strict positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a strict positive integer")
    return value


def query_routing_run_assignments(
    workspace: str | Path,
    *,
    run_key: str,
    layer_keys: tuple[str, ...],
    expert_keys: tuple[tuple[str, ...], ...],
    routed_top_k: int,
    max_routing_rows: int,
    max_source_bytes: int,
    duckdb: Any,
    connection: Any,
) -> tuple[RoutingShardAssignmentQuery, ...]:
    """Return validated per-shard assignment summaries for one committed run.

    This is the public reader/query seam over committed shards: every shard is
    reopened and fully validated (manifest identity, file metadata and digests,
    row identities, universe membership, links) before its grouped assignment
    counts are returned in canonical shard order.  The caller owns the lazily
    imported store engine and the bounded in-memory query connection, including
    closing it exactly once.  Failures are carried by typed errors:
    :class:`RoutingShardError` for storage-owned stages (``workspace``,
    ``reopen``, ``conflict``), :class:`RoutingRunInventoryError` with stage
    ``budget`` for exhausted row or byte budgets,
    :class:`RoutingRunQueryError` for query-engine failures, and plain
    ``ValueError``/``TypeError``/``OSError`` for absent or malformed sources.
    """

    for value, name in (
        (max_routing_rows, "max_routing_rows"),
        (max_source_bytes, "max_source_bytes"),
    ):
        _strict_positive_query_budget(value, name)
    _strict_positive_query_budget(routed_top_k, "routed_top_k")
    if type(layer_keys) is not tuple or not layer_keys:
        raise TypeError("layer_keys must be a non-empty tuple of strings")
    if type(expert_keys) is not tuple or len(expert_keys) != len(layer_keys):
        raise TypeError("expert_keys must match layer_keys exactly")

    # Validate the managed workspace/run boundary before touching any source,
    # so absent or malformed source is deterministic without reading shards.
    path = _validate_workspace(workspace)
    stable_run_key = _validate_run_key(run_key)
    run_parent = _existing_run_parent(path, stable_run_key)
    if run_parent is None:
        raise ValueError("run has no committed routing shards")
    try:
        children = tuple(run_parent.iterdir())
    except Exception as exc:
        raise ValueError("managed run directory is unreadable") from exc
    shard_paths: list[Path] = []
    for child in children:
        if _STAGING_NAME.fullmatch(child.name):
            if child.is_symlink() or not child.is_dir():
                raise RoutingShardError("reopen")
            continue
        if child.name.startswith(".staging-"):
            raise RoutingShardError("reopen")
        if not child.name.startswith(_SHARD_PREFIX):
            raise RoutingShardError("reopen")
        shard_paths.append(child)
    if not shard_paths:
        raise ValueError("run has no committed routing shards")
    sources: list[tuple[Path, dict[str, object], str]] = []
    source_bytes = 0
    declared_routing_rows = 0
    for shard in sorted(shard_paths, key=lambda item: item.name):
        manifest, shard_key = _read_shard_manifest(
            shard, stable_run_key, duckdb, validate_files=False
        )
        declared_routing_rows += manifest["routing_count"]
        if declared_routing_rows > max_routing_rows:
            raise RoutingRunInventoryError("budget") from ValueError(
                "routing rows exceed the source budget"
            )
        try:
            sizes = [
                (shard / _MANIFEST_FILE).stat().st_size,
                (shard / _TOKENS_FILE).stat().st_size,
                (shard / _ROUTING_FILE).stat().st_size,
            ]
            if manifest["store_schema_version"] == STORE_SCHEMA_VERSION:
                sizes.append((shard / _EXPERTS_FILE).stat().st_size)
        except Exception as exc:
            raise ValueError("managed shard metadata is unreadable") from exc
        source_bytes += sum(sizes)
        if source_bytes > max_source_bytes:
            raise RoutingRunInventoryError("budget") from ValueError(
                "source bytes exceed the source budget"
            )
        sources.append((shard, manifest, shard_key))

    records: list[RoutingShardAssignmentQuery] = []
    seen_tokens: set[str] = set()
    seen_links: set[tuple[str, str, int]] = set()
    actual_counts: dict[Path, tuple[int, int]] = {}
    actual_routing_rows = 0
    try:
        for shard, manifest, shard_key in sources:
            try:
                _validate_file_metadata(shard, manifest)
            except RoutingShardError:
                raise
            except Exception as exc:
                raise RoutingShardError("reopen") from exc
            token_path = shard / _TOKENS_FILE
            routing_path = shard / _ROUTING_FILE
            try:
                actual_token_count = connection.execute(
                    "SELECT COUNT(*) FROM read_parquet(?)", [str(token_path)]
                ).fetchone()[0]
                actual_routing_count = connection.execute(
                    "SELECT COUNT(*) FROM read_parquet(?)", [str(routing_path)]
                ).fetchone()[0]
            except RoutingShardError:
                raise
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as exc:
                raise RoutingShardError("reopen") from exc
            actual_routing_rows += int(actual_routing_count)
            if actual_routing_rows > max_routing_rows:
                raise RoutingRunInventoryError("budget") from ValueError(
                    "actual routing rows exceed the row budget"
                )
            if (
                actual_token_count != manifest["token_count"]
                or actual_routing_count != manifest["routing_count"]
            ):
                raise RoutingShardError("reopen") from ValueError(
                    "parquet row counts do not match manifest"
                )
            actual_counts[shard] = (int(actual_token_count), int(actual_routing_count))
        for shard, manifest, shard_key in sources:
            actual_token_count, actual_routing_count = actual_counts[shard]
            try:
                token_keys, routing_links = _validate_routing_load_source(
                    shard,
                    stable_run_key,
                    duckdb,
                    connection,
                    layer_keys,
                    expert_keys,
                    routed_top_k,
                )
            except RoutingShardError:
                raise
            except OSError as exc:
                raise RoutingRunQueryError() from exc
            except Exception as exc:
                raise ValueError("shard source validation failed") from exc
            if seen_tokens.intersection(token_keys) or seen_links.intersection(routing_links):
                raise RoutingShardError("conflict")
            seen_tokens.update(token_keys)
            seen_links.update(routing_links)
            rows = connection.execute(
                "SELECT layer_key, expert_key, COUNT(*) AS assignment_count "
                "FROM read_parquet(?) GROUP BY layer_key, expert_key "
                "ORDER BY layer_key, expert_key",
                [str(shard / _ROUTING_FILE)],
            ).fetchall()
            records.append(
                RoutingShardAssignmentQuery(
                    shard_key=shard_key,
                    token_count=actual_token_count,
                    routing_count=actual_routing_count,
                    token_keys=frozenset(token_keys),
                    routing_links=frozenset(routing_links),
                    assignment_counts=tuple(
                        (str(layer), str(expert), int(count)) for layer, expert, count in rows
                    ),
                )
            )
    except (KeyboardInterrupt, SystemExit):
        raise
    except (
        RoutingShardError,
        RoutingRunInventoryError,
        RoutingRunQueryError,
        ValueError,
    ):
        raise
    except BaseException as exc:
        raise RoutingRunQueryError() from exc
    return tuple(records)


@dataclass(frozen=True, slots=True)
class RoutingShardExpertActivityQuery:
    """Validated per-shard expert-activity summary from one bounded query."""

    shard_key: str
    expert_event_count: int
    activity_cells: tuple[tuple[str, str, int, int, float, float], ...]

    def __post_init__(self) -> None:
        if type(self.shard_key) is not str or _SHARD_KEY.fullmatch(self.shard_key) is None:
            raise ValueError("shard_key must be a canonical shard digest")
        if type(self.expert_event_count) is not int or isinstance(
            self.expert_event_count, bool
        ) or self.expert_event_count < 0:
            raise ValueError("expert_event_count must be a non-negative integer")
        if type(self.activity_cells) is not tuple:
            raise TypeError("activity_cells must be an exact tuple")
        keys: list[tuple[str, str]] = []
        for entry in self.activity_cells:
            if type(entry) is not tuple or len(entry) != 6:
                raise ValueError(
                    "activity_cells entries must be (layer, expert, events, measured, sum, max)"
                )
            layer_key, expert_key, count, measured, total, peak = entry
            if type(layer_key) is not str or type(expert_key) is not str:
                raise TypeError("activity cell keys must be exact strings")
            for value in (count, measured):
                if type(value) is not int or isinstance(value, bool) or value <= 0:
                    raise ValueError("activity cell counts must be strict positive integers")
            if measured > count:
                raise ValueError("measured contributions cannot exceed the event count")
            for value in (total, peak):
                if type(value) is not float or not math.isfinite(value) or value < 0.0:
                    raise ValueError("activity cell sums/maxima must be finite nonnegative floats")
            keys.append((layer_key, expert_key))
        if keys != sorted(keys):
            raise ValueError("activity_cells must be sorted by layer and expert key")


def query_expert_activity(
    workspace: str | Path,
    *,
    run_key: str,
    layer_keys: tuple[str, ...],
    expert_keys: tuple[tuple[str, ...], ...],
    max_expert_rows: int,
    max_source_bytes: int,
    duckdb: Any,
    connection: Any,
) -> tuple[RoutingShardExpertActivityQuery, ...]:
    """Return validated per-shard expert-activity summaries for one run.

    Mirrors :func:`query_routing_run_assignments`: every shard is reopened and
    fully validated before its grouped per-expert contribution aggregates are
    returned in canonical shard order. ``layer_keys``/``expert_keys`` supply
    the layer mapping because expert rows carry opaque component identities.
    Shards without expert evidence (legacy ``1.0`` shards or empty tables)
    contribute zero-activity records. Raw rows are never retained.
    """

    for value, name in (
        (max_expert_rows, "max_expert_rows"),
        (max_source_bytes, "max_source_bytes"),
    ):
        _strict_positive_query_budget(value, name)
    if type(layer_keys) is not tuple or not layer_keys:
        raise TypeError("layer_keys must be a non-empty tuple of strings")
    if type(expert_keys) is not tuple or len(expert_keys) != len(layer_keys):
        raise TypeError("expert_keys must match layer_keys exactly")
    layer_of_expert: dict[str, str] = {}
    for layer_key, row in zip(layer_keys, expert_keys):
        if type(layer_key) is not str or type(row) is not tuple or not row:
            raise TypeError("universe rows must map string layers to non-empty key tuples")
        for expert_key in row:
            if type(expert_key) is not str:
                raise TypeError("expert universe keys must be exact strings")
            if expert_key in layer_of_expert:
                raise ValueError("expert universe keys must be globally unique")
            layer_of_expert[expert_key] = layer_key

    path = _validate_workspace(workspace)
    stable_run_key = _validate_run_key(run_key)
    run_parent = _existing_run_parent(path, stable_run_key)
    if run_parent is None:
        raise ValueError("run has no committed routing shards")
    try:
        children = tuple(run_parent.iterdir())
    except Exception as exc:
        raise ValueError("managed run directory is unreadable") from exc
    shard_paths: list[Path] = []
    for child in children:
        if _STAGING_NAME.fullmatch(child.name):
            if child.is_symlink() or not child.is_dir():
                raise RoutingShardError("reopen")
            continue
        if child.name.startswith(".staging-"):
            raise RoutingShardError("reopen")
        if not child.name.startswith(_SHARD_PREFIX):
            raise RoutingShardError("reopen")
        shard_paths.append(child)
    if not shard_paths:
        raise ValueError("run has no committed routing shards")

    sources: list[tuple[Path, dict[str, object], str]] = []
    source_bytes = 0
    declared_expert_rows = 0
    for shard in sorted(shard_paths, key=lambda item: item.name):
        manifest, shard_key = _read_shard_manifest(shard, stable_run_key, duckdb)
        if manifest["store_schema_version"] == STORE_SCHEMA_VERSION:
            declared_expert_rows += int(manifest["expert_count"])
            if declared_expert_rows > max_expert_rows:
                raise RoutingRunInventoryError("budget") from ValueError(
                    "expert rows exceed the source budget"
                )
        try:
            sizes = [
                (shard / _MANIFEST_FILE).stat().st_size,
                (shard / _TOKENS_FILE).stat().st_size,
                (shard / _ROUTING_FILE).stat().st_size,
            ]
            if manifest["store_schema_version"] == STORE_SCHEMA_VERSION:
                sizes.append((shard / _EXPERTS_FILE).stat().st_size)
        except Exception as exc:
            raise ValueError("managed shard metadata is unreadable") from exc
        source_bytes += sum(sizes)
        if source_bytes > max_source_bytes:
            raise RoutingRunInventoryError("budget") from ValueError(
                "source bytes exceed the source budget"
            )
        sources.append((shard, manifest, shard_key))

    records: list[RoutingShardExpertActivityQuery] = []
    seen_tokens: set[str] = set()
    seen_links: set[tuple[str, str, int]] = set()
    seen_expert_links: set[tuple[str, str]] = set()
    actual_expert_rows = 0
    try:
        for shard, manifest, shard_key in sources:
            try:
                actual = _reconstruct_shard_with_connection(
                    shard, stable_run_key, duckdb, connection
                )
                if actual.receipt.shard_key != shard_key:
                    raise RoutingShardError("reopen") from ValueError(
                        "shard identity changed during reopen"
                    )
            except RoutingShardError:
                raise
            except OSError as exc:
                raise RoutingRunQueryError() from exc
            except Exception as exc:
                raise ValueError("shard source validation failed") from exc
            if (
                seen_tokens.intersection(actual.token_keys)
                or seen_links.intersection(actual.routing_links)
                or seen_expert_links.intersection(actual.expert_links)
            ):
                raise RoutingShardError("conflict")
            seen_tokens.update(actual.token_keys)
            seen_links.update(actual.routing_links)
            seen_expert_links.update(actual.expert_links)
            actual_expert_rows += len(actual.expert_rows)
            if actual_expert_rows > max_expert_rows:
                raise RoutingRunInventoryError("budget") from ValueError(
                    "actual expert rows exceed the row budget"
                )
            grouped: dict[tuple[str, str], list[object]] = {}
            for row in actual.expert_rows:
                token_key, expert_key, contribution = row[5], row[6], row[9]
                del token_key
                layer_key = layer_of_expert.get(expert_key)
                if layer_key is None:
                    raise ValueError("shard expert references an unknown expert")
                cell = grouped.get((layer_key, expert_key))
                if cell is None:
                    cell = [0, []]
                    grouped[(layer_key, expert_key)] = cell
                cell[0] += 1
                if contribution is not None:
                    if (
                        type(contribution) is not float
                        or not math.isfinite(contribution)
                        or contribution < 0.0
                    ):
                        raise ValueError("expert contribution norms must be finite nonnegative")
                    cell[1].append(contribution)
            cells: list[tuple[str, str, int, int, float, float]] = []
            for (cell_layer, cell_expert), (event_count, contributions) in sorted(
                grouped.items()
            ):
                cells.append(
                    (
                        cell_layer,
                        cell_expert,
                        event_count,
                        len(contributions),
                        math.fsum(contributions),
                        max(contributions),
                    )
                )
            records.append(
                RoutingShardExpertActivityQuery(
                    shard_key=shard_key,
                    expert_event_count=len(actual.expert_rows),
                    activity_cells=tuple(cells),
                )
            )
    except (KeyboardInterrupt, SystemExit):
        raise
    except (
        RoutingShardError,
        RoutingRunInventoryError,
        RoutingRunQueryError,
        ValueError,
    ):
        raise
    except BaseException as exc:
        raise RoutingRunQueryError() from exc
    return tuple(records)


append_mixtral_routing_shard = append_routing_shard
list_mixtral_routing_shards = list_routing_shards
list_mixtral_routing_runs = list_routing_runs


__all__ = [
    "STORE_SCHEMA_VERSION",
    "LEGACY_STORE_SCHEMA_VERSION",
    "ROUTING_RUN_INVENTORY_SCHEMA_VERSION",
    "RoutingShardError",
    "RoutingShardReceipt",
    "RoutingRunInventoryError",
    "RoutingRunQueryError",
    "RoutingShardAssignmentQuery",
    "RoutingShardExpertActivityQuery",
    "MixtralRoutingRunSummary",
    "MixtralRoutingRunInventory",
    "append_routing_shard",
    "append_structured_shard",
    "list_routing_shards",
    "list_routing_runs",
    "query_routing_run_assignments",
    "query_expert_activity",
    "append_mixtral_routing_shard",
    "list_mixtral_routing_shards",
    "list_mixtral_routing_runs",
]
