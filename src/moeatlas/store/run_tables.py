"""Bounded tabular (CSV/Parquet) evidence exports for one committed run.

Where the run-evidence bundle is the lossless interchange format, this module
publishes one run's committed token and routing rows as open tabular surfaces
for external tools: fixed column schemas mirroring the persisted shard
layout, deterministic row ordering (``shard_key`` then ``event_index``),
strict row/byte budgets, digest-bearing provenance manifest, atomic crash-safe
publication, and symlink safety.  CSV members are byte-deterministic and
canonically re-encoded during verification; Parquet members are verified by
digest and size because the DuckDB writer may embed its own metadata.

Tabular exports are one-way projections: they carry every persisted column
(including ``token_text_stored`` redaction evidence) but are not an import
path — lossless round-trips stay the run-evidence bundle's contract.  Like the
rest of the persistence layer this module imports duckdb lazily at call time
through routing_shards; verification needs no duckdb at all.  It performs no
network access, no clock reads, and no model-runtime work.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import __version__
from ..core import validate_stable_identifier
from ..events import EVENT_SCHEMA_VERSION
from .routing_shards import (
    _ROUTING_COLUMNS,
    _TOKEN_COLUMNS,
    STORE_SCHEMA_VERSION,
    RoutingShardError,
    _load_duckdb,
    _validate_workspace,
    list_routing_shards,
)
from .run_export import (
    _canonical_json,
    _check_parent,
    _digest_bytes,
    _publish_bundle,
    _read_member,
    _resolve_directory,
    _validate_budget,
)

RUN_TABLES_SCHEMA_VERSION = "1.0"
"""Schema version of the tabular export format."""

TABLES_MANIFEST_TYPE = "routing_run_tables"
"""Manifest type marker carried by every tabular export."""

_WRITER_NAME = "moeatlas"
_TOKENS_CSV = "tokens.csv"
_ROUTING_CSV = "routing.csv"
_TOKENS_PARQUET = "tokens.parquet"
_ROUTING_PARQUET = "routing.parquet"
_MANIFEST_FILE = "manifest.json"

_TABLE_FORMATS = ("csv", "parquet")
_FORMAT_NAMES = {
    "csv": (_TOKENS_CSV, _ROUTING_CSV),
    "parquet": (_TOKENS_PARQUET, _ROUTING_PARQUET),
}
_FILE_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_STAGES = frozenset({"dependency", "workspace", "source", "format", "budget", "write", "publish"})
_SHARD_STAGE_MAP = {
    "dependency": "dependency",
    "workspace": "workspace",
    "reopen": "source",
}

_DEFAULT_MAX_EVENT_ROWS = 1_000_000
_DEFAULT_MAX_FILE_BYTES = 1_000_000_000

_SHARD_ENTRY_KEYS = frozenset(
    {"shard_key", "token_count", "routing_count", "token_text_stored"}
)
_FILE_INFO_KEYS = frozenset({"bytes", "sha256"})
_MANIFEST_KEYS = frozenset(
    {
        "manifest_type",
        "schema_version",
        "store_schema_version",
        "event_schema_version",
        "run_key",
        "writer_name",
        "writer_version",
        "formats",
        "token_count",
        "routing_count",
        "shards",
        "files",
    }
)


class RunTableError(RuntimeError):
    """Safe fixed-stage failure for bounded tabular run exports."""

    def __init__(
        self, stage: str, message: str | None = None, *, cause: BaseException | None = None
    ) -> None:
        if stage not in _STAGES:
            raise ValueError("run table error stage is not supported")
        self.stage = stage
        if message is None:
            super().__init__(f"run table export failed at {stage}")
        else:
            super().__init__(f"run table export failed at {stage}: {message}")
        if cause is not None:
            self.__cause__ = cause


@dataclass(frozen=True, slots=True)
class RunTableFileEntry:
    """Digest-bearing identity of one published export member."""

    name: str
    bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise ValueError("name must be a non-empty string")
        if "\\" in self.name or self.name.startswith("/") or "/" in self.name:
            raise ValueError("name must be a flat safe file name")
        if type(self.bytes) is not int or isinstance(self.bytes, bool) or self.bytes < 0:
            raise ValueError("bytes must be a non-negative integer")
        if type(self.sha256) is not str or _FILE_DIGEST.fullmatch(self.sha256) is None:
            raise ValueError("sha256 must be a digest string")


@dataclass(frozen=True, slots=True)
class RunTableReceipt:
    """Portable receipt describing one published tabular export."""

    schema_version: str
    manifest_type: str
    run_key: str
    formats: tuple[str, ...]
    shard_count: int
    token_count: int
    routing_count: int
    manifest_sha256: str
    entries: tuple[RunTableFileEntry, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not str or self.schema_version != RUN_TABLES_SCHEMA_VERSION:
            raise ValueError("schema_version must be the exact run tables schema version")
        if type(self.manifest_type) is not str or self.manifest_type != TABLES_MANIFEST_TYPE:
            raise ValueError("manifest_type must be the exact run tables manifest type")
        if type(self.run_key) is not str:
            raise TypeError("run_key must be a string")
        validate_stable_identifier(self.run_key, field_name="run_key")
        if (
            type(self.formats) is not tuple
            or not self.formats
            or any(type(item) is not str or item not in _TABLE_FORMATS for item in self.formats)
            or sorted(set(self.formats)) != list(self.formats)
        ):
            raise ValueError("formats must be a sorted duplicate-free supported tuple")
        for name in ("shard_count", "token_count", "routing_count"):
            value = getattr(self, name)
            if type(value) is not int or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a strict positive integer")
        if (
            type(self.manifest_sha256) is not str
            or _FILE_DIGEST.fullmatch(self.manifest_sha256) is None
        ):
            raise ValueError("manifest_sha256 must be a digest string")
        if type(self.entries) is not tuple or not self.entries:
            raise ValueError("entries must be a non-empty tuple")
        names = [entry.name for entry in self.entries]
        if names != sorted(names) or len(set(names)) != len(names):
            raise ValueError("entries must be sorted by unique name")
        for entry in self.entries:
            if type(entry) is not RunTableFileEntry:
                raise TypeError("entries must be RunTableFileEntry values")


def _token_header() -> tuple[str, ...]:
    return tuple(name for name, _ in _TOKEN_COLUMNS)


def _routing_header() -> tuple[str, ...]:
    return tuple(name for name, _ in _ROUTING_COLUMNS)


def _csv_cell(value: object) -> str:
    if value is None:
        return ""
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float is not canonically encodable")
        return repr(value)
    if isinstance(value, str):
        return value
    raise ValueError(f"unsupported CSV cell type: {type(value).__name__}")


def _csv_bytes(header: tuple[str, ...], rows: list[tuple[Any, ...]]) -> bytes:
    # QUOTE_ALL keeps the canonical encoding unique: a bare csv.writer would
    # reproduce space-after-delimiter variants byte-for-byte, so re-encoding
    # could not distinguish them from the canonical spelling.
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n", quoting=csv.QUOTE_ALL)
    writer.writerow(header)
    for row in rows:
        writer.writerow([_csv_cell(value) for value in row])
    return buffer.getvalue().encode("utf-8")


def _canonical_csv(header: tuple[str, ...], payload: bytes) -> int:
    """Re-encode parsed CSV bytes and require byte equality; return data rows."""

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RunTableError("format", "CSV member is not valid UTF-8", cause=exc)
    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        rows = [tuple(row) for row in reader]
    except csv.Error as exc:
        raise RunTableError("format", "CSV member is malformed", cause=exc)
    if not rows or tuple(rows[0]) != header:
        raise RunTableError("format", "CSV header does not match the canonical schema")
    for row in rows[1:]:
        if len(row) != len(header):
            raise RunTableError("format", "CSV row width does not match the header")
    if _csv_bytes(header, rows[1:]) != payload:
        raise RunTableError("format", "CSV member is not canonically encoded")
    return len(rows) - 1


def _wrap_storage(exc: RoutingShardError) -> RunTableError:
    stage = _SHARD_STAGE_MAP.get(exc.stage, "source")
    return RunTableError(stage, "storage layer rejected the operation", cause=exc)


def _validated_formats(formats: object) -> tuple[str, ...]:
    if type(formats) is not tuple or not formats:
        raise TypeError("formats must be a non-empty tuple of supported format names")
    for item in formats:
        if type(item) is not str:
            raise TypeError("formats must be a non-empty tuple of supported format names")
    unknown = sorted(set(formats) - set(_TABLE_FORMATS))
    if unknown:
        raise ValueError("formats contains unsupported names")
    if len(set(formats)) != len(formats):
        raise ValueError("formats must not contain duplicates")
    return tuple(sorted(set(formats)))


def _member_names(formats: tuple[str, ...]) -> tuple[str, ...]:
    """Data-member names implied by ``formats`` (the manifest is separate)."""

    names: list[str] = []
    for item in formats:
        names.extend(_FORMAT_NAMES[item])
    return tuple(sorted(names))


def _sql_columns(header: tuple[str, ...]) -> str:
    return ", ".join(header)


def _read_ordered_rows(
    connection: Any,
    files: list[str],
    header: tuple[str, ...],
    max_event_rows: int,
    label: str,
) -> list[tuple[Any, ...]]:
    # read_parquet accepts one bound path or a bound list of paths.
    parameter: object = files[0] if len(files) == 1 else files
    try:
        rows = connection.execute(
            f"SELECT {_sql_columns(header)} FROM read_parquet(?) "
            "ORDER BY shard_key, event_index",
            [parameter],
        ).fetchall()
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        raise RunTableError("source", f"committed {label} rows are unreadable", cause=exc)
    if len(rows) > max_event_rows:
        raise RunTableError("budget", f"{label} rows exceed the row budget")
    return rows


def _parquet_bytes(
    duckdb: Any,
    header: tuple[str, ...],
    types: tuple[str, ...],
    rows: list[tuple[Any, ...]],
    table_name: str,
    order: str,
) -> bytes:
    columns = ", ".join(f"{name} {kind}" for name, kind in zip(header, types))
    with tempfile.TemporaryDirectory(prefix="moeatlas-tables-") as tmp:
        target = Path(tmp) / f"{table_name}.parquet"
        connection: Any | None = None
        try:
            connection = duckdb.connect(database=":memory:")
            connection.execute(f"CREATE TEMP TABLE {table_name} ({columns})")
            placeholders = ", ".join("?" for _ in header)
            connection.executemany(
                f"INSERT INTO {table_name} VALUES ({placeholders})", rows
            )
            connection.table(table_name).order(order).write_parquet(
                str(target), compression="zstd", overwrite=False
            )
            payload = target.read_bytes()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            raise RunTableError("write", "parquet member encoding failed", cause=exc)
        finally:
            if connection is not None:
                try:
                    connection.close()
                except Exception as exc:
                    raise RunTableError("write", "parquet connection close failed", cause=exc)
    return payload


def _enforce_byte_budget(payload: bytes, name: str, max_file_bytes: int) -> None:
    if len(payload) > max_file_bytes:
        raise RunTableError("budget", f"table member exceeds the byte budget: {name}")


def _receipt_from(manifest: dict[str, object], manifest_bytes: bytes) -> RunTableReceipt:
    files = manifest["files"]
    if type(files) is not dict:
        raise RunTableError("format", "manifest files must be an object")
    entries = []
    for name, info in files.items():
        if type(info) is not dict:
            raise RunTableError("format", "manifest file info shape is not exact")
        entries.append(
            RunTableFileEntry(
                name=str(name), bytes=int(info["bytes"]), sha256=str(info["sha256"])
            )
        )
    entries.append(
        RunTableFileEntry(
            name=_MANIFEST_FILE,
            bytes=len(manifest_bytes),
            sha256=_digest_bytes(manifest_bytes),
        )
    )
    shards = manifest["shards"]
    if type(shards) is not list:
        raise RunTableError("format", "manifest shards must be a list")
    return RunTableReceipt(
        schema_version=str(manifest["schema_version"]),
        manifest_type=str(manifest["manifest_type"]),
        run_key=str(manifest["run_key"]),
        formats=tuple(str(item) for item in manifest["formats"]),  # type: ignore[arg-type]
        shard_count=len(shards),
        token_count=int(manifest["token_count"]),  # type: ignore[arg-type]
        routing_count=int(manifest["routing_count"]),  # type: ignore[arg-type]
        manifest_sha256=_digest_bytes(manifest_bytes),
        entries=tuple(sorted(entries, key=lambda item: item.name)),
    )


def export_run_tables(
    workspace: str | Path,
    destination: str | Path,
    *,
    run_key: str,
    formats: tuple[str, ...] = ("csv",),
    max_event_rows: int = _DEFAULT_MAX_EVENT_ROWS,
    max_file_bytes: int = _DEFAULT_MAX_FILE_BYTES,
) -> RunTableReceipt:
    """Export one run's committed rows as bounded CSV and/or Parquet tables.

    The source is validated through :func:`list_routing_shards` before any
    byte is read.  Members are built in memory, byte-budgeted, and published
    atomically as ``manifest.json`` plus one token/routing pair per requested
    format.  CSV members are byte-deterministic; Parquet members are
    digest-recorded without a byte-determinism promise.
    """

    if type(run_key) is not str:
        raise TypeError("run_key must be an exact string")
    validate_stable_identifier(run_key, field_name="run_key")
    ordered_formats = _validated_formats(formats)
    _validate_budget(max_event_rows, "max_event_rows")
    _validate_budget(max_file_bytes, "max_file_bytes")
    destination_path = _resolve_directory(
        destination, param="destination", stage="workspace", require_empty=True,
        error=RunTableError,
    )
    _check_parent(destination_path, param="destination", error=RunTableError)

    try:
        path = _validate_workspace(workspace)
    except RoutingShardError as exc:
        raise _wrap_storage(exc)
    try:
        receipts = list_routing_shards(path, run_key=run_key)
    except RoutingShardError as exc:
        raise _wrap_storage(exc)
    if not receipts:
        raise RunTableError("source", "run has no committed routing shards")

    token_files = [str(path / receipt.relative_path / "tokens.parquet") for receipt in receipts]
    routing_files = [str(path / receipt.relative_path / "routing.parquet") for receipt in receipts]

    try:
        duckdb = _load_duckdb()
    except RoutingShardError as exc:
        raise RunTableError(
            "dependency", "storage engine unavailable for table export", cause=exc
        )

    token_rows: list[tuple[Any, ...]] = []
    routing_rows: list[tuple[Any, ...]] = []
    connection: Any | None = None
    primary: BaseException | None = None
    try:
        connection = duckdb.connect(database=":memory:")
        token_header = _token_header()
        routing_header = _routing_header()
        token_rows = _read_ordered_rows(
            connection, token_files, token_header, max_event_rows, "token"
        )
        routing_rows = _read_ordered_rows(
            connection, routing_files, routing_header, max_event_rows, "routing"
        )
        payloads: dict[str, bytes] = {}
        if "csv" in ordered_formats:
            payloads[_TOKENS_CSV] = _csv_bytes(token_header, token_rows)
            payloads[_ROUTING_CSV] = _csv_bytes(routing_header, routing_rows)
        if "parquet" in ordered_formats:
            token_types = tuple(kind for _, kind in _TOKEN_COLUMNS)
            routing_types = tuple(kind for _, kind in _ROUTING_COLUMNS)
            payloads[_TOKENS_PARQUET] = _parquet_bytes(
                duckdb, token_header, token_types, token_rows, "tokens", "event_index"
            )
            payloads[_ROUTING_PARQUET] = _parquet_bytes(
                duckdb, routing_header, routing_types, routing_rows, "routing", "event_index"
            )
    except (KeyboardInterrupt, SystemExit) as exc:
        primary = exc
    except RunTableError as exc:
        primary = exc
    except BaseException as exc:
        primary = RunTableError("source", "table read failed", cause=exc)
    finally:
        if connection is not None:
            try:
                connection.close()
            except BaseException as close_error:
                if primary is None:
                    primary = RunTableError("write", "query connection close failed",
                                            cause=close_error)
    if primary is not None:
        raise primary

    for name in sorted(payloads):
        _enforce_byte_budget(payloads[name], name, max_file_bytes)

    manifest: dict[str, object] = {
        "manifest_type": TABLES_MANIFEST_TYPE,
        "schema_version": RUN_TABLES_SCHEMA_VERSION,
        "store_schema_version": STORE_SCHEMA_VERSION,
        "event_schema_version": EVENT_SCHEMA_VERSION,
        "run_key": run_key,
        "writer_name": _WRITER_NAME,
        "writer_version": __version__,
        "formats": list(ordered_formats),
        "token_count": len(token_rows),
        "routing_count": len(routing_rows),
        "shards": [
            {
                "shard_key": receipt.shard_key,
                "token_count": receipt.token_count,
                "routing_count": receipt.routing_count,
                "token_text_stored": receipt.token_text_stored,
            }
            for receipt in receipts
        ],
        "files": {
            name: {"bytes": len(payload), "sha256": _digest_bytes(payload)}
            for name, payload in sorted(payloads.items())
        },
    }
    manifest_bytes = _canonical_json(manifest)
    try:
        _publish_bundle(
            destination_path,
            payloads,
            manifest_bytes,
            error=RunTableError,
            noun="table",
        )
    except RunTableError:
        raise
    except BaseException as exc:
        raise RunTableError("publish", "table publication failed", cause=exc)
    return _receipt_from(manifest, manifest_bytes)


def _load_manifest(
    payload: bytes, *, max_event_rows: int, max_file_bytes: int
) -> dict[str, object]:
    if not payload.endswith(b"\n") or payload[:-1].endswith(b"\n"):
        raise RunTableError("format", "manifest newline is not exact")
    try:
        document = json.loads(payload[:-1].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunTableError("format", "manifest is not valid JSON", cause=exc)
    if _canonical_json(document) != payload:
        raise RunTableError("format", "manifest is not canonically encoded")
    if type(document) is not dict or set(document) != _MANIFEST_KEYS:
        raise RunTableError("format", "manifest shape is not exact")
    if document["manifest_type"] != TABLES_MANIFEST_TYPE:
        raise RunTableError("format", "manifest type is unsupported")
    if document["schema_version"] != RUN_TABLES_SCHEMA_VERSION:
        raise RunTableError("format", "run tables schema version is unsupported")
    if document["store_schema_version"] != STORE_SCHEMA_VERSION:
        raise RunTableError("format", "store schema version is unsupported")
    if document["event_schema_version"] != EVENT_SCHEMA_VERSION:
        raise RunTableError("format", "event schema version is unsupported")
    if type(document["run_key"]) is not str:
        raise RunTableError("format", "manifest run key must be a string")
    if document["writer_name"] != _WRITER_NAME or type(document["writer_version"]) is not str:
        raise RunTableError("format", "manifest writer identity is not exact")
    try:
        _validated_formats(tuple(document["formats"]))  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise RunTableError("format", "manifest formats are not supported", cause=exc)
    try:
        validate_stable_identifier(document["run_key"], field_name="run_key")
    except ValueError as exc:
        raise RunTableError("format", "manifest run key is not a stable identifier", cause=exc)
    for name in ("token_count", "routing_count"):
        value = document[name]
        if type(value) is not int or isinstance(value, bool) or value <= 0:
            raise RunTableError("format", f"manifest {name} is not a strict positive integer")
        if value > max_event_rows:
            raise RunTableError("budget", f"manifest {name} exceeds the row budget")
    shards = document["shards"]
    if type(shards) is not list or not shards:
        raise RunTableError("format", "manifest shards must be a non-empty list")
    for section in shards:
        if type(section) is not dict or set(section) != _SHARD_ENTRY_KEYS:
            raise RunTableError("format", "manifest shard section shape is not exact")
        if type(section["token_text_stored"]) is not bool:
            raise RunTableError("format", "manifest shard redaction flag is not exact")
        for name in ("token_count", "routing_count"):
            value = section[name]
            if type(value) is not int or isinstance(value, bool) or value <= 0:
                raise RunTableError("format", f"manifest shard {name} is not positive")
    files = document["files"]
    if type(files) is not dict or not files:
        raise RunTableError("format", "manifest files must be a non-empty object")
    expected = set(_member_names(tuple(document["formats"])))  # type: ignore[arg-type]
    if set(files) != expected:
        raise RunTableError("format", "manifest files do not match the requested formats")
    for name, info in files.items():
        if type(info) is not dict or set(info) != _FILE_INFO_KEYS:
            raise RunTableError("format", "manifest file info shape is not exact")
        size = info["bytes"]
        digest = info["sha256"]
        if type(size) is not int or isinstance(size, bool) or size < 0:
            raise RunTableError("format", "manifest file size is not a byte count")
        if size > max_file_bytes:
            raise RunTableError("budget", f"table member exceeds the byte budget: {name}")
        if type(digest) is not str or _FILE_DIGEST.fullmatch(digest) is None:
            raise RunTableError("format", "manifest file digest is not exact")
    return document


def verify_run_tables(
    source: str | Path,
    *,
    max_event_rows: int = _DEFAULT_MAX_EVENT_ROWS,
    max_file_bytes: int = _DEFAULT_MAX_FILE_BYTES,
) -> RunTableReceipt:
    """Verify one published tabular export without importing duckdb.

    Checks manifest canonicality and schema, exact member sets, every digest
    and size, canonical CSV encoding with the exact header and row counts, and
    all budgets.  Parquet members are verified by digest and size only.
    """

    _validate_budget(max_event_rows, "max_event_rows")
    _validate_budget(max_file_bytes, "max_file_bytes")
    source_path = _resolve_directory(
        source, param="source", stage="workspace", require_empty=False, error=RunTableError
    )
    manifest_payload = _read_member(
        source_path / _MANIFEST_FILE,
        label=_MANIFEST_FILE,
        max_file_bytes=max_file_bytes,
        error=RunTableError,
        noun="table",
    )
    manifest = _load_manifest(
        manifest_payload, max_event_rows=max_event_rows, max_file_bytes=max_file_bytes
    )

    files = manifest["files"]
    if type(files) is not dict:
        raise RunTableError("format", "manifest files must be an object")
    for name in sorted(files):
        info = files[name]
        if type(info) is not dict:
            raise RunTableError("format", "manifest file info shape is not exact")
        payload = _read_member(
            source_path / name,
            label=name,
            max_file_bytes=max_file_bytes,
            error=RunTableError,
            noun="table",
        )
        if len(payload) != info["bytes"] or _digest_bytes(payload) != info["sha256"]:
            raise RunTableError("format", f"table member digest mismatch: {name}")
        if name == _TOKENS_CSV:
            rows = _canonical_csv(_token_header(), payload)
            if rows != manifest["token_count"]:
                raise RunTableError("format", "tokens.csv row count does not match the manifest")
        if name == _ROUTING_CSV:
            rows = _canonical_csv(_routing_header(), payload)
            if rows != manifest["routing_count"]:
                raise RunTableError("format", "routing.csv row count does not match the manifest")
    return _receipt_from(manifest, manifest_payload)
