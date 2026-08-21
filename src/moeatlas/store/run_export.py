"""Bounded, relocatable run-evidence export bundles.

One bundle carries the complete committed evidence of exactly one routing run
in open formats: a canonical digest-bearing ``manifest.json`` plus one
canonical JSONL file per shard for token and routing events.  Bundles are the
open-format interchange surface over the internal DuckDB/Parquet shards: they
are byte-deterministic, tamper-evident, atomic on publication, symlink-safe,
and relocatable across workspaces because shard identity is content-addressed
over the exported events themselves.

Redaction is honored per shard exactly as stored: a shard that did not store
token text exports ``null`` text fields and its manifest entry records
``token_text_stored: false``; import reconstructs the identical content-
addressed shard from either form.  Absence of token text is therefore explicit
bundle evidence, never silently inferred data.

All sizes are bounded by explicit budgets.  Like the rest of the persistence
layer this module imports duckdb lazily at call time through routing_shards;
it performs no network access, no clock reads, and no model-runtime work.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import __version__
from ..core import validate_stable_identifier
from ..event_validation import (
    fresh_routing_events,
    fresh_token_events,
    validate_routing_links,
)
from ..events import EVENT_SCHEMA_VERSION, RoutingEvent, TokenEvent
from .routing_shards import (
    STORE_SCHEMA_VERSION,
    RoutingShardError,
    RoutingShardReceipt,
    _append_internal,
    _existing_run_parent,
    _existing_shards,
    _load_duckdb,
    _semantic_rows,
    _shard_key,
    _ShardData,
    _validate_workspace,
)

RUN_EXPORT_SCHEMA_VERSION = "1.0"
"""Schema version of the run-evidence export bundle format."""

BUNDLE_MANIFEST_TYPE = "routing_run_export"
"""Manifest type marker carried by every run-evidence export bundle."""

_WRITER_NAME = "moeatlas"
_TOKENS_FILE = "tokens.jsonl"
_ROUTING_FILE = "routing.jsonl"
_MANIFEST_FILE = "manifest.json"
_DATA_DIRECTORY = "data"
_SHARD_DIRECTORY_PREFIX = "shard-"

_SHARD_KEY = re.compile(r"^shard:[0-9a-f]{64}$")
_FILE_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_STAGES = frozenset(
    {"dependency", "workspace", "source", "format", "budget", "write", "publish", "conflict"}
)
_SHARD_STAGE_MAP = {
    "dependency": "dependency",
    "workspace": "workspace",
    "write": "write",
    "publish": "publish",
    "reopen": "source",
    "conflict": "conflict",
}

_DEFAULT_MAX_EVENT_ROWS = 1_000_000
_DEFAULT_MAX_FILE_BYTES = 1_000_000_000

_TOKEN_RECORD_KEYS = frozenset(
    {
        "event_index",
        "schema_version",
        "type",
        "token_key",
        "run_key",
        "sequence_id",
        "token_pos",
        "token_id",
        "token_text",
        "phase",
    }
)
_ROUTING_RECORD_KEYS = frozenset(
    {
        "event_index",
        "schema_version",
        "type",
        "token_key",
        "layer_key",
        "rank",
        "expert_key",
        "router_logit",
        "probability",
        "weight",
        "selected",
    }
)
_SHARD_ENTRY_KEYS = frozenset(
    {"shard_key", "token_count", "routing_count", "token_text_stored", "files"}
)
_BUNDLE_MEMBER_NAMES = frozenset({_TOKENS_FILE, _ROUTING_FILE})
_FILE_INFO_KEYS = frozenset({"name", "bytes", "sha256"})
_MANIFEST_KEYS = frozenset(
    {
        "manifest_type",
        "schema_version",
        "store_schema_version",
        "event_schema_version",
        "run_key",
        "writer_name",
        "writer_version",
        "token_count",
        "routing_count",
        "shards",
    }
)


class RunBundleError(RuntimeError):
    """Safe fixed-stage failure for bounded run-evidence export bundles."""

    def __init__(
        self,
        stage: str,
        message: str | None = None,
        *,
        cause: BaseException | None = None,
    ) -> None:
        if type(stage) is not str or stage not in _STAGES:
            raise ValueError(f"unsupported run bundle stage: {stage!r}")
        self.stage = stage
        text = f"run bundle failed at {stage}"
        if message is not None:
            text = f"{text}: {message}"
        super().__init__(text)
        if cause is not None:
            self.__cause__ = cause


@dataclass(frozen=True, slots=True)
class RunBundleFileEntry:
    """Digest receipt for one canonical file inside an export bundle."""

    name: str
    bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise TypeError("entry name must be a non-empty string")
        parts = self.name.split("/")
        if (
            "\\" in self.name
            or any(part in {"", ".", ".."} for part in parts)
            or self.name.startswith("/")
            or self.name.endswith("/")
        ):
            raise ValueError("entry name must be a safe relative bundle path")
        if type(self.bytes) is not int or isinstance(self.bytes, bool):
            raise TypeError("entry bytes must be a strict integer")
        if self.bytes <= 0:
            raise ValueError("entry bytes must be positive")
        if type(self.sha256) is not str or _FILE_DIGEST.fullmatch(self.sha256) is None:
            raise ValueError("entry sha256 must be a canonical sha256 digest")


@dataclass(frozen=True, slots=True)
class RunBundleReceipt:
    """Immutable receipt describing one verified run-evidence export bundle."""

    schema_version: str
    manifest_type: str
    run_key: str
    shard_count: int
    token_count: int
    routing_count: int
    manifest_sha256: str
    entries: tuple[RunBundleFileEntry, ...]

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not str
            or self.schema_version != RUN_EXPORT_SCHEMA_VERSION
        ):
            raise ValueError("schema_version must be the exact run export schema version")
        if type(self.manifest_type) is not str or self.manifest_type != BUNDLE_MANIFEST_TYPE:
            raise ValueError("manifest_type must be the exact bundle manifest type")
        if type(self.run_key) is not str:
            raise TypeError("run_key must be a string")
        validate_stable_identifier(self.run_key, field_name="run_key")
        for name in ("shard_count", "token_count", "routing_count"):
            value = getattr(self, name)
            if type(value) is not int or isinstance(value, bool):
                raise TypeError(f"{name} must be a strict integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if (
            type(self.manifest_sha256) is not str
            or _FILE_DIGEST.fullmatch(self.manifest_sha256) is None
        ):
            raise ValueError("manifest_sha256 must be a canonical sha256 digest")
        if type(self.entries) is not tuple or any(
            type(entry) is not RunBundleFileEntry for entry in self.entries
        ):
            raise TypeError("entries must be a tuple of exact RunBundleFileEntry values")
        names = [entry.name for entry in self.entries]
        if names != sorted(names):
            raise ValueError("entries must be sorted by name")
        if len(set(names)) != len(names):
            raise ValueError("entries must have unique names")


@dataclass(frozen=True, slots=True)
class _BundleShard:
    """One fully validated shard section parsed back out of a bundle."""

    shard_key: str
    token_events: tuple[TokenEvent, ...]
    routing_events: tuple[RoutingEvent, ...]
    token_text_stored: bool


@dataclass(frozen=True, slots=True)
class _LoadedBundle:
    """A complete validated bundle held in memory, ready to verify or append."""

    run_key: str
    shards: tuple[_BundleShard, ...]
    manifest_bytes: bytes
    entries: tuple[RunBundleFileEntry, ...]


def _validate_budget(value: object, name: str) -> int:
    if type(value) is not int or isinstance(value, bool):
        raise TypeError(f"{name} must be a strict positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a strict positive integer")
    return value


def _canonical_json(value: object) -> bytes:
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


def _digest_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _wrap_storage(exc: RoutingShardError) -> RunBundleError:
    stage = _SHARD_STAGE_MAP.get(exc.stage, "source")
    return RunBundleError(stage, "storage layer rejected the operation", cause=exc)


def _resolve_directory(
    value: object, *, param: str, stage: str, require_empty: bool
) -> Path:
    if not isinstance(value, str | Path):
        raise TypeError(f"{param} must be a string or pathlib.Path")
    path = Path(value)
    try:
        symlinked = path.is_symlink()
        present = path.exists()
        directory = path.is_dir() if present else False
        empty = directory and not any(path.iterdir())
    except RunBundleError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        raise RunBundleError(stage, f"{param} state is unreadable", cause=exc) from exc
    if symlinked:
        raise RunBundleError(stage, f"{param} must not be a symlink")
    if require_empty:
        if present and not directory:
            raise RunBundleError(stage, f"{param} is not a directory")
        if directory and not empty:
            raise RunBundleError(stage, f"{param} is not empty")
        return path
    if not directory:
        raise RunBundleError(stage, f"{param} is not an existing directory")
    return path


def _check_parent(path: Path, *, param: str) -> None:
    parent = path.parent
    try:
        invalid = parent.is_symlink() or not parent.is_dir()
    except Exception as exc:
        raise RunBundleError("workspace", f"{param} parent is unreadable", cause=exc) from exc
    if invalid:
        raise RunBundleError("workspace", f"{param} parent is not an existing directory")


def _read_member(path: Path, *, label: str, max_file_bytes: int) -> bytes:
    try:
        if path.is_symlink() or not path.is_file():
            raise RunBundleError("format", f"bundle member is not a regular file: {label}")
        size = path.stat().st_size
        if size > max_file_bytes:
            raise RunBundleError("budget", f"bundle member exceeds the byte budget: {label}")
        payload = path.read_bytes()
    except RunBundleError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        raise RunBundleError("format", f"bundle member is unreadable: {label}", cause=exc)
    if len(payload) > max_file_bytes:
        raise RunBundleError("budget", f"bundle member exceeds the byte budget: {label}")
    return payload


def _parse_jsonl(payload: bytes, *, label: str) -> tuple[dict[str, Any], ...]:
    if not payload.endswith(b"\n") or payload[:-1].endswith(b"\n"):
        raise RunBundleError("format", f"JSONL newlines are not exact: {label}")
    body = payload[:-1]
    records: list[dict[str, Any]] = []
    for number, line in enumerate(body.split(b"\n")):
        try:
            parsed = json.loads(line.decode("utf-8"))
        except Exception as exc:
            raise RunBundleError(
                "format", f"{label} line {number} is not valid JSON", cause=exc
            ) from exc
        if type(parsed) is not dict:
            raise RunBundleError("format", f"{label} line {number} is not a JSON object")
        if _canonical_json(parsed) != line + b"\n":
            raise RunBundleError(
                "format", f"{label} line {number} is not canonically encoded"
            )
        records.append(parsed)
    if not records:
        raise RunBundleError("format", f"JSONL file is empty: {label}")
    return tuple(records)


def _parse_token_records(
    payload: bytes,
    *,
    expected_count: int,
    store_text: bool,
    run_key: str,
    label: str,
) -> tuple[TokenEvent, ...]:
    records = _parse_jsonl(payload, label=label)
    if len(records) != expected_count:
        raise RunBundleError(
            "format", f"{label} row count does not match the manifest count"
        )
    events: list[TokenEvent] = []
    for index, record in enumerate(records):
        where = f"{label} row {index}"
        if set(record) != _TOKEN_RECORD_KEYS:
            raise RunBundleError("format", f"{where} fields are not exact")
        if record["event_index"] != index:
            raise RunBundleError("format", f"{where} event_index is not contiguous")
        if record["schema_version"] != EVENT_SCHEMA_VERSION or record["type"] != "token":
            raise RunBundleError("format", f"{where} event identity is unsupported")
        text = record["token_text"]
        if store_text:
            if type(text) is not str:
                raise RunBundleError(
                    "format", f"{where} must carry token text when the shard stores it"
                )
        elif text is not None:
            raise RunBundleError(
                "format", f"{where} must not carry token text under redaction"
            )
        try:
            event = TokenEvent.model_validate(
                {
                    "schema_version": record["schema_version"],
                    "event_type": "token",
                    "token_key": record["token_key"],
                    "run_key": record["run_key"],
                    "sequence_id": record["sequence_id"],
                    "token_pos": record["token_pos"],
                    "token_id": record["token_id"],
                    "token_text": text if store_text else "",
                    "phase": record["phase"],
                }
            )
        except Exception as exc:
            raise RunBundleError(
                "format", f"{where} failed event revalidation", cause=exc
            ) from exc
        if event.run_key != run_key:
            raise RunBundleError("format", f"{where} references a foreign run identity")
        events.append(event)
    try:
        return fresh_token_events(tuple(events))
    except (TypeError, ValueError) as exc:
        raise RunBundleError(
            "format", f"{label} token events failed collection validation", cause=exc
        ) from exc


def _parse_routing_records(
    payload: bytes,
    *,
    expected_count: int,
    label: str,
) -> tuple[RoutingEvent, ...]:
    records = _parse_jsonl(payload, label=label)
    if len(records) != expected_count:
        raise RunBundleError(
            "format", f"{label} row count does not match the manifest count"
        )
    events: list[RoutingEvent] = []
    for index, record in enumerate(records):
        where = f"{label} row {index}"
        if set(record) != _ROUTING_RECORD_KEYS:
            raise RunBundleError("format", f"{where} fields are not exact")
        if record["event_index"] != index:
            raise RunBundleError("format", f"{where} event_index is not contiguous")
        if record["schema_version"] != EVENT_SCHEMA_VERSION or record["type"] != "routing":
            raise RunBundleError("format", f"{where} event identity is unsupported")
        if record["selected"] is not True:
            raise RunBundleError("format", f"{where} must be a selected routing row")
        try:
            event = RoutingEvent.model_validate(
                {
                    "schema_version": record["schema_version"],
                    "event_type": "routing",
                    "token_key": record["token_key"],
                    "layer_key": record["layer_key"],
                    "rank": record["rank"],
                    "expert_key": record["expert_key"],
                    "router_logit": record["router_logit"],
                    "probability": record["probability"],
                    "weight": record["weight"],
                    "selected": True,
                }
            )
        except Exception as exc:
            raise RunBundleError(
                "format", f"{where} failed event revalidation", cause=exc
            ) from exc
        events.append(event)
    try:
        return fresh_routing_events(tuple(events))
    except (TypeError, ValueError) as exc:
        raise RunBundleError(
            "format", f"{label} routing events failed collection validation", cause=exc
        ) from exc


def _load_manifest(
    source: Path, *, max_file_bytes: int
) -> tuple[dict[str, Any], bytes]:
    payload = _read_member(
        source / _MANIFEST_FILE, label=_MANIFEST_FILE, max_file_bytes=max_file_bytes
    )
    if not payload.endswith(b"\n") or payload[:-1].endswith(b"\n"):
        raise RunBundleError("format", "manifest newlines are not exact")
    try:
        parsed = json.loads(payload[:-1].decode("utf-8"))
    except Exception as exc:
        raise RunBundleError("format", "manifest is not valid JSON", cause=exc) from exc
    if type(parsed) is not dict or set(parsed) != _MANIFEST_KEYS:
        raise RunBundleError("format", "manifest fields are not exact")
    if parsed["manifest_type"] != BUNDLE_MANIFEST_TYPE:
        raise RunBundleError("format", "manifest type is unsupported")
    if parsed["schema_version"] != RUN_EXPORT_SCHEMA_VERSION:
        raise RunBundleError("format", "bundle schema version is unsupported")
    if parsed["store_schema_version"] != STORE_SCHEMA_VERSION:
        raise RunBundleError("format", "store schema version is unsupported")
    if parsed["event_schema_version"] != EVENT_SCHEMA_VERSION:
        raise RunBundleError("format", "event schema version is unsupported")
    if type(parsed["run_key"]) is not str:
        raise RunBundleError("format", "manifest run identity is invalid")
    try:
        validate_stable_identifier(parsed["run_key"], field_name="run_key")
    except (TypeError, ValueError) as exc:
        raise RunBundleError("format", "manifest run identity is invalid", cause=exc) from exc
    for key in ("writer_name", "writer_version"):
        value = parsed[key]
        if type(value) is not str or not value:
            raise RunBundleError("format", f"manifest writer field {key!r} is invalid")
    for key in ("token_count", "routing_count"):
        value = parsed[key]
        if type(value) is not int or isinstance(value, bool) or value <= 0:
            raise RunBundleError("format", f"manifest count {key!r} is invalid")
    shards = parsed["shards"]
    if type(shards) is not list or not shards:
        raise RunBundleError("format", "manifest shard list is invalid")
    seen_keys: list[str] = []
    total_tokens = 0
    total_routing = 0
    for position, entry in enumerate(shards):
        where = f"manifest shard {position}"
        if type(entry) is not dict or set(entry) != _SHARD_ENTRY_KEYS:
            raise RunBundleError("format", f"{where} fields are not exact")
        shard_key = entry["shard_key"]
        if type(shard_key) is not str or _SHARD_KEY.fullmatch(shard_key) is None:
            raise RunBundleError("format", f"{where} shard identity is invalid")
        seen_keys.append(shard_key)
        for key in ("token_count", "routing_count"):
            value = entry[key]
            if type(value) is not int or isinstance(value, bool) or value <= 0:
                raise RunBundleError("format", f"{where} count {key!r} is invalid")
        if type(entry["token_text_stored"]) is not bool:
            raise RunBundleError("format", f"{where} redaction flag is invalid")
        files = entry["files"]
        if type(files) is not dict or set(files) != _BUNDLE_MEMBER_NAMES:
            raise RunBundleError("format", f"{where} file set is not exact")
        for name in sorted(_BUNDLE_MEMBER_NAMES):
            info = files[name]
            if type(info) is not dict or set(info) != _FILE_INFO_KEYS:
                raise RunBundleError("format", f"{where} file metadata is not exact")
            if info["name"] != name:
                raise RunBundleError("format", f"{where} file name is invalid")
            size = info["bytes"]
            if type(size) is not int or isinstance(size, bool) or size <= 0:
                raise RunBundleError("format", f"{where} file size is invalid")
            digest = info["sha256"]
            if type(digest) is not str or _FILE_DIGEST.fullmatch(digest) is None:
                raise RunBundleError("format", f"{where} file digest is invalid")
        total_tokens += entry["token_count"]
        total_routing += entry["routing_count"]
    if seen_keys != sorted(seen_keys) or len(set(seen_keys)) != len(seen_keys):
        raise RunBundleError("format", "manifest shards are not uniquely ordered")
    if total_tokens != parsed["token_count"] or total_routing != parsed["routing_count"]:
        raise RunBundleError("format", "manifest totals do not match shard entries")
    if _canonical_json(parsed) != payload:
        raise RunBundleError("format", "manifest bytes are not canonical")
    return parsed, payload


def _member_label(hex_digest: str, name: str) -> str:
    return f"{_DATA_DIRECTORY}/{_SHARD_DIRECTORY_PREFIX}{hex_digest}/{name}"


def _load_bundle(
    source: str | Path,
    *,
    max_event_rows: int,
    max_file_bytes: int,
) -> _LoadedBundle:
    source_path = _resolve_directory(
        source, param="source", stage="workspace", require_empty=False
    )
    manifest, manifest_payload = _load_manifest(source_path, max_file_bytes=max_file_bytes)
    run_key = manifest["run_key"]

    total_rows = manifest["token_count"] + manifest["routing_count"]
    if total_rows > max_event_rows:
        raise RunBundleError("budget", "bundle event rows exceed the configured budget")

    loaded: list[_BundleShard] = []
    entries: list[RunBundleFileEntry] = []
    for entry in manifest["shards"]:
        hex_digest = entry["shard_key"].removeprefix("shard:")
        payloads: dict[str, bytes] = {}
        for name in sorted(_BUNDLE_MEMBER_NAMES):
            label = _member_label(hex_digest, name)
            info = entry["files"][name]
            payload = _read_member(
                source_path / label, label=label, max_file_bytes=max_file_bytes
            )
            if len(payload) != info["bytes"]:
                raise RunBundleError("format", f"bundle member size mismatch: {label}")
            if _digest_bytes(payload) != info["sha256"]:
                raise RunBundleError("format", f"bundle member digest mismatch: {label}")
            entries.append(
                RunBundleFileEntry(
                    name=label, bytes=len(payload), sha256=_digest_bytes(payload)
                )
            )
            payloads[name] = payload
        store_text = entry["token_text_stored"]
        token_events = _parse_token_records(
            payloads[_TOKENS_FILE],
            expected_count=entry["token_count"],
            store_text=store_text,
            run_key=run_key,
            label=_member_label(hex_digest, _TOKENS_FILE),
        )
        routing_events = _parse_routing_records(
            payloads[_ROUTING_FILE],
            expected_count=entry["routing_count"],
            label=_member_label(hex_digest, _ROUTING_FILE),
        )
        try:
            validate_routing_links(token_events, routing_events)
        except (TypeError, ValueError) as exc:
            raise RunBundleError(
                "format", f"{hex_digest} shard links failed validation", cause=exc
            ) from exc
        _, _, semantic = _semantic_rows(
            token_events, routing_events, store_token_text=store_text
        )
        if _shard_key(semantic) != entry["shard_key"]:
            raise RunBundleError(
                "format",
                f"{hex_digest} shard identity does not match its exported events",
            )
        loaded.append(
            _BundleShard(
                shard_key=entry["shard_key"],
                token_events=token_events,
                routing_events=routing_events,
                token_text_stored=store_text,
            )
        )
    entries.append(
        RunBundleFileEntry(
            name=_MANIFEST_FILE,
            bytes=len(manifest_payload),
            sha256=_digest_bytes(manifest_payload),
        )
    )
    return _LoadedBundle(
        run_key=run_key,
        shards=tuple(loaded),
        manifest_bytes=manifest_payload,
        entries=tuple(sorted(entries, key=lambda item: item.name)),
    )


def _receipt_from(loaded: _LoadedBundle) -> RunBundleReceipt:
    token_count = sum(len(shard.token_events) for shard in loaded.shards)
    routing_count = sum(len(shard.routing_events) for shard in loaded.shards)
    return RunBundleReceipt(
        schema_version=RUN_EXPORT_SCHEMA_VERSION,
        manifest_type=BUNDLE_MANIFEST_TYPE,
        run_key=loaded.run_key,
        shard_count=len(loaded.shards),
        token_count=token_count,
        routing_count=routing_count,
        manifest_sha256=_digest_bytes(loaded.manifest_bytes),
        entries=loaded.entries,
    )


def _events_from_shard_rows(
    data: _ShardData,
) -> tuple[tuple[TokenEvent, ...], tuple[RoutingEvent, ...]]:
    """Rebuild validated events from already-revalidated persisted rows."""

    store_text = data.receipt.token_text_stored
    token_events = [
        TokenEvent(
            schema_version=row[3],
            event_type=row[4],
            token_key=row[5],
            run_key=row[6],
            sequence_id=row[7],
            token_pos=row[8],
            token_id=row[9],
            token_text=row[10] if store_text else "",
            phase=row[12],
        )
        for row in data.token_rows
    ]
    routing_events = [
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
        for row in data.routing_rows
    ]
    return tuple(token_events), tuple(routing_events)


def _token_record(index: int, event: TokenEvent, *, store_text: bool) -> dict[str, object]:
    return {
        "event_index": index,
        "schema_version": event.schema_version,
        "type": "token",
        "token_key": event.token_key,
        "run_key": event.run_key,
        "sequence_id": event.sequence_id,
        "token_pos": event.token_pos,
        "token_id": event.token_id,
        "token_text": event.token_text if store_text else None,
        "phase": event.phase.value,
    }


def _routing_record(index: int, event: RoutingEvent) -> dict[str, object]:
    return {
        "event_index": index,
        "schema_version": event.schema_version,
        "type": "routing",
        "token_key": event.token_key,
        "layer_key": event.layer_key,
        "rank": event.rank,
        "expert_key": event.expert_key,
        "router_logit": event.router_logit,
        "probability": event.probability,
        "weight": event.weight,
        "selected": event.selected,
    }


def _cleanup_stage(stage: Path) -> None:
    try:
        if stage.exists() and stage.is_dir() and not stage.is_symlink():
            shutil.rmtree(stage)
    except Exception:
        return


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_member(stage: Path, relative: str, payload: bytes) -> None:
    target = stage / relative
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            if os.name == "posix":
                os.chmod(target, 0o600)
            os.fsync(stream.fileno())
        _fsync_directory(target.parent)
    except Exception as exc:
        raise RunBundleError(
            "write", f"bundle member write failed: {relative}", cause=exc
        ) from exc


def _publish_bundle(
    destination: Path, payloads: dict[str, bytes], manifest_bytes: bytes
) -> None:
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.export-staging-", dir=str(destination.parent)
        )
    )
    published = False
    try:
        for relative, payload in sorted(payloads.items()):
            _write_member(stage, relative, payload)
        try:
            manifest_path = stage / _MANIFEST_FILE
            with manifest_path.open("wb") as stream:
                stream.write(manifest_bytes)
                stream.flush()
                if os.name == "posix":
                    os.chmod(manifest_path, 0o600)
                os.fsync(stream.fileno())
            if os.name == "posix":
                os.chmod(stage, 0o700)
            _fsync_directory(stage)
        except Exception as exc:
            raise RunBundleError("write", "bundle manifest write failed", cause=exc)
        try:
            os.rename(stage, destination)
            published = True
        except Exception as exc:
            raise RunBundleError("publish", "bundle rename failed", cause=exc) from exc
    except BaseException:
        if not published:
            _cleanup_stage(stage)
        raise
    try:
        _fsync_directory(destination.parent)
    except Exception as exc:
        raise RunBundleError("publish", "bundle parent fsync failed", cause=exc) from exc


def export_run_bundle(
    workspace: str | Path,
    destination: str | Path,
    *,
    run_key: str,
    max_event_rows: int = _DEFAULT_MAX_EVENT_ROWS,
    max_file_bytes: int = _DEFAULT_MAX_FILE_BYTES,
) -> RunBundleReceipt:
    """Export every committed shard of one run as a bounded evidence bundle.

    The destination must be nonexistent or an empty real directory; publication
    stages a sibling temporary directory, fsyncs every member plus the manifest
    last, and atomically renames it into place.  Two exports of the same
    committed run state produce byte-identical bundles.
    """

    max_event_rows = _validate_budget(max_event_rows, "max_event_rows")
    max_file_bytes = _validate_budget(max_file_bytes, "max_file_bytes")
    if not isinstance(destination, str | Path):
        raise TypeError("destination must be a string or pathlib.Path")
    target = Path(destination)
    _check_parent(target, param="destination")
    _resolve_directory(target, param="destination", stage="workspace", require_empty=True)

    source = _validate_workspace(workspace)
    stable_run_key = validate_stable_identifier(run_key, field_name="run_key")
    try:
        duckdb = _load_duckdb()
    except RoutingShardError as exc:
        raise RunBundleError(
            "dependency", "storage engine unavailable for export", cause=exc
        ) from exc

    try:
        run_parent = _existing_run_parent(source, stable_run_key)
        if run_parent is None:
            raise RunBundleError("source", f"run has no committed shards: {stable_run_key}")
        shards = _existing_shards(run_parent, stable_run_key, duckdb)
        if not shards:
            raise RunBundleError("source", f"run has no committed shards: {stable_run_key}")
    except RoutingShardError as exc:
        raise _wrap_storage(exc) from exc

    total_rows = sum(
        data.receipt.token_count + data.receipt.routing_count for data in shards
    )
    if total_rows > max_event_rows:
        raise RunBundleError("budget", "run event rows exceed the configured budget")

    payloads: dict[str, bytes] = {}
    shard_entries: list[dict[str, object]] = []
    token_total = 0
    routing_total = 0
    for data in shards:
        hex_digest = data.receipt.shard_key.removeprefix("shard:")
        prefix = f"data/{_SHARD_DIRECTORY_PREFIX}{hex_digest}"
        store_text = data.receipt.token_text_stored
        token_events, routing_events = _events_from_shard_rows(data)
        token_payload = b"".join(
            _canonical_json(_token_record(index, event, store_text=store_text))
            for index, event in enumerate(token_events)
        )
        routing_payload = b"".join(
            _canonical_json(_routing_record(index, event))
            for index, event in enumerate(routing_events)
        )
        members = ((_TOKENS_FILE, token_payload), (_ROUTING_FILE, routing_payload))
        for name, payload in members:
            if len(payload) > max_file_bytes:
                raise RunBundleError(
                    "budget", f"exported file exceeds the byte budget: {prefix}/{name}"
                )
            payloads[f"{prefix}/{name}"] = payload
        token_total += data.receipt.token_count
        routing_total += data.receipt.routing_count
        shard_entries.append(
            {
                "shard_key": data.receipt.shard_key,
                "token_count": data.receipt.token_count,
                "routing_count": data.receipt.routing_count,
                "token_text_stored": store_text,
                "files": {
                    _TOKENS_FILE: {
                        "name": _TOKENS_FILE,
                        "bytes": len(token_payload),
                        "sha256": _digest_bytes(token_payload),
                    },
                    _ROUTING_FILE: {
                        "name": _ROUTING_FILE,
                        "bytes": len(routing_payload),
                        "sha256": _digest_bytes(routing_payload),
                    },
                },
            }
        )

    manifest = {
        "manifest_type": BUNDLE_MANIFEST_TYPE,
        "schema_version": RUN_EXPORT_SCHEMA_VERSION,
        "store_schema_version": STORE_SCHEMA_VERSION,
        "event_schema_version": EVENT_SCHEMA_VERSION,
        "run_key": stable_run_key,
        "writer_name": _WRITER_NAME,
        "writer_version": __version__,
        "token_count": token_total,
        "routing_count": routing_total,
        "shards": shard_entries,
    }
    manifest_bytes = _canonical_json(manifest)

    _publish_bundle(target, payloads, manifest_bytes)

    entries = [
        RunBundleFileEntry(name=name, bytes=len(payload), sha256=_digest_bytes(payload))
        for name, payload in payloads.items()
    ]
    entries.append(
        RunBundleFileEntry(
            name=_MANIFEST_FILE,
            bytes=len(manifest_bytes),
            sha256=_digest_bytes(manifest_bytes),
        )
    )
    return RunBundleReceipt(
        schema_version=RUN_EXPORT_SCHEMA_VERSION,
        manifest_type=BUNDLE_MANIFEST_TYPE,
        run_key=stable_run_key,
        shard_count=len(shards),
        token_count=token_total,
        routing_count=routing_total,
        manifest_sha256=_digest_bytes(manifest_bytes),
        entries=tuple(sorted(entries, key=lambda item: item.name)),
    )


def verify_run_bundle(
    source: str | Path,
    *,
    max_event_rows: int = _DEFAULT_MAX_EVENT_ROWS,
    max_file_bytes: int = _DEFAULT_MAX_FILE_BYTES,
) -> RunBundleReceipt:
    """Fully validate a bundle's digests, canonicality, and event contracts.

    Verification recomputes every file digest, enforces canonical encoding,
    revalidates each shard's events and their links, and re-derives each shard's
    content-addressed identity from the exported events themselves.
    """

    max_event_rows = _validate_budget(max_event_rows, "max_event_rows")
    max_file_bytes = _validate_budget(max_file_bytes, "max_file_bytes")
    loaded = _load_bundle(
        source, max_event_rows=max_event_rows, max_file_bytes=max_file_bytes
    )
    return _receipt_from(loaded)


def import_run_bundle(
    source: str | Path,
    workspace: str | Path,
    *,
    max_event_rows: int = _DEFAULT_MAX_EVENT_ROWS,
    max_file_bytes: int = _DEFAULT_MAX_FILE_BYTES,
) -> tuple[RoutingShardReceipt, ...]:
    """Import a verified bundle into a workspace as immutable shards.

    Import composes full bundle verification with the standard shard appender:
    each exported shard is reconstructed and appended through the same
    content-addressed publication path used by forward capture, so importing a
    bundle into the workspace it was exported from is idempotent and importing
    into a fresh workspace reproduces identical shard identities.
    """

    max_event_rows = _validate_budget(max_event_rows, "max_event_rows")
    max_file_bytes = _validate_budget(max_file_bytes, "max_file_bytes")
    loaded = _load_bundle(
        source, max_event_rows=max_event_rows, max_file_bytes=max_file_bytes
    )
    target = _validate_workspace(workspace)
    try:
        duckdb = _load_duckdb()
    except RoutingShardError as exc:
        raise RunBundleError(
            "dependency", "storage engine unavailable for import", cause=exc
        ) from exc
    receipts: list[RoutingShardReceipt] = []
    for shard in loaded.shards:
        try:
            receipts.append(
                _append_internal(
                    target,
                    shard.token_events,
                    shard.routing_events,
                    store_token_text=shard.token_text_stored,
                    duckdb=duckdb,
                )
            )
        except RoutingShardError as exc:
            raise _wrap_storage(exc) from exc
    return tuple(receipts)


__all__ = [
    "BUNDLE_MANIFEST_TYPE",
    "RUN_EXPORT_SCHEMA_VERSION",
    "RunBundleError",
    "RunBundleFileEntry",
    "RunBundleReceipt",
    "export_run_bundle",
    "import_run_bundle",
    "verify_run_bundle",
]
