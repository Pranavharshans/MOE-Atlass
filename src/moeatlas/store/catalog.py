"""Bounded versioned workspace catalog and run registry.

The catalog is one canonical JSON manifest per workspace at
``<workspace>/.moeatlas/catalog.json``. It registers known runs and their
observed storage totals; lifecycle truth stays in committed ``RunRecord``
manifests and storage truth stays in the routing shards. Every write is
atomic and every failure names a stage from the same vocabulary as shard
storage (``dependency``, ``workspace``, ``write``, ``publish``, ``reopen``,
``conflict``).
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import (
    Field,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)

from ..core import StrictManifestModel, VersionedManifest, validate_stable_identifier
from ..runs import RunState
from .routing_shards import list_routing_runs

WORKSPACE_CATALOG_SCHEMA_VERSION = "1.0"
CATALOG_MAX_RUNS = 10_000

_STAGES = frozenset(
    {"dependency", "workspace", "write", "publish", "reopen", "conflict"}
)
_CATALOG_DIR_NAME = ".moeatlas"
_CATALOG_FILE_NAME = "catalog.json"
_STAGING_PREFIX = ".staging-catalog-"
_RUN_KEY_PATTERN = re.compile(r"^run:[0-9a-f]{64}$")
_MAX_TIMESTAMP = 64


class WorkspaceCatalogError(RuntimeError):
    """Raised when a catalog operation fails at a specific stage."""

    def __init__(
        self,
        stage: str,
        message: str | None = None,
        *,
        cause: BaseException | None = None,
    ) -> None:
        if type(stage) is not str or stage not in _STAGES:
            raise ValueError(f"unsupported workspace catalog stage: {stage!r}")
        self.stage = stage
        text = f"workspace catalog failed at {stage}"
        if message is not None:
            text = f"{text}: {message}"
        super().__init__(text)
        if cause is not None:
            self.__cause__ = cause


class RunRegistryEntry(StrictManifestModel):
    """One registered run in a workspace catalog."""

    run_key: StrictStr
    specification_fingerprint: StrictStr | None = None
    state: StrictStr | None = None
    attempt: StrictInt = Field(default=1, ge=1)
    shard_count: StrictInt = Field(default=0, ge=0)
    token_event_count: StrictInt = Field(default=0, ge=0)
    routing_event_count: StrictInt = Field(default=0, ge=0)
    token_text_policy: Literal["redacted", "stored", "mixed"] | None = None
    registered_at: StrictStr | None = None
    updated_at: StrictStr | None = None

    @field_validator("run_key")
    @classmethod
    def _check_run_key(cls, value: str) -> str:
        # Registry keys mirror the shard-storage identity vocabulary, not the
        # stricter RunSpecification fingerprint form: rebuild must accept any
        # run key the storage layer already committed.
        validate_stable_identifier(value, field_name="run_key")
        return value

    @field_validator("specification_fingerprint")
    @classmethod
    def _check_fingerprint(cls, value: str | None) -> str | None:
        if value is not None and _RUN_KEY_PATTERN.fullmatch(value) is None:
            raise ValueError(
                "specification_fingerprint must use the run:<64 lowercase hex> form"
            )
        return value

    @field_validator("state")
    @classmethod
    def _check_state(cls, value: str | None) -> str | None:
        if value is None:
            return value
        allowed = sorted(state.value for state in RunState)
        if value not in allowed:
            raise ValueError(f"state must be one of: {', '.join(allowed)}")
        return value

    @field_validator("registered_at", "updated_at")
    @classmethod
    def _check_timestamp(cls, value: str | None) -> str | None:
        return _validate_timestamp_value(value)


class WorkspaceCatalog(VersionedManifest):
    """Versioned snapshot of a workspace's run registry."""

    manifest_type: ClassVar[str] = "workspace_catalog"

    created_at: StrictStr | None = None
    updated_at: StrictStr | None = None
    runs: tuple[RunRegistryEntry, ...] = ()

    @field_validator("created_at", "updated_at")
    @classmethod
    def _check_timestamp(cls, value: str | None) -> str | None:
        return _validate_timestamp_value(value)

    @model_validator(mode="after")
    def _check_registry_order(self) -> WorkspaceCatalog:
        keys = [entry.run_key for entry in self.runs]
        if keys != sorted(keys):
            raise ValueError("runs must be sorted ascending by run_key")
        if len(set(keys)) != len(keys):
            raise ValueError("runs must have unique run_key values")
        return self


@dataclass(frozen=True, slots=True)
class CatalogRebuildReceipt:
    """Sorted reconciliation outcome of one catalog rebuild."""

    added: tuple[str, ...] = ()
    updated: tuple[str, ...] = ()
    unchanged: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    run_count: int = 0


def catalog_path(workspace: str | Path) -> Path:
    """Return the catalog manifest path without touching the filesystem."""

    return Path(workspace) / _CATALOG_DIR_NAME / _CATALOG_FILE_NAME


def initialize_catalog(
    workspace: str | Path,
    *,
    at: str | None = None,
    max_runs: int = CATALOG_MAX_RUNS,
) -> WorkspaceCatalog:
    """Create the empty catalog manifest for a workspace."""

    _require_strict_positive_int(max_runs, "max_runs")
    timestamp = _require_optional_timestamp(at)
    root = _validate_workspace(workspace)
    final = catalog_path(root)
    directory = _ensure_catalog_directory(root)
    if final.is_symlink() or final.exists():
        raise WorkspaceCatalogError("conflict", "workspace catalog already exists")
    catalog = WorkspaceCatalog(created_at=timestamp, updated_at=timestamp, runs=())
    _publish_catalog(directory, catalog)
    return catalog


def read_catalog(workspace: str | Path) -> WorkspaceCatalog:
    """Read and validate the catalog manifest for a workspace."""

    root = _validate_workspace(workspace)
    payload = _read_catalog_payload(catalog_path(root))
    # manifest_type is persisted as a stability marker but is a ClassVar,
    # not a model field; extra="forbid" rejects it during validation.
    payload.pop("manifest_type")
    try:
        return WorkspaceCatalog.model_validate(payload)
    except ValidationError as exc:
        raise WorkspaceCatalogError(
            "reopen", "catalog failed schema validation", cause=exc
        ) from exc


def upsert_run_entry(
    workspace: str | Path,
    entry: RunRegistryEntry,
    *,
    at: str | None = None,
    max_runs: int = CATALOG_MAX_RUNS,
) -> WorkspaceCatalog:
    """Insert or merge one registry entry, writing only on change."""

    if type(entry) is not RunRegistryEntry:
        raise TypeError(f"entry must be a RunRegistryEntry, got {type(entry).__name__}")
    _require_strict_positive_int(max_runs, "max_runs")
    timestamp = _require_optional_timestamp(at)
    root = _validate_workspace(workspace)
    catalog = read_catalog(root)
    existing = {item.run_key: item for item in catalog.runs}
    old = existing.get(entry.run_key)
    if old is None and len(catalog.runs) >= max_runs:
        raise WorkspaceCatalogError("conflict", "run registry is full")
    data = entry.model_dump()
    if old is not None:
        # Registration time is immutable history: an update never overrides it.
        data["registered_at"] = old.registered_at
        if data["updated_at"] is None:
            data["updated_at"] = old.updated_at
    if timestamp is not None:
        data["updated_at"] = timestamp
    merged = RunRegistryEntry.model_validate(data)
    by_key = dict(existing)
    by_key[merged.run_key] = merged
    result = WorkspaceCatalog(
        created_at=catalog.created_at,
        updated_at=timestamp if timestamp is not None else catalog.updated_at,
        runs=tuple(sorted(by_key.values(), key=lambda item: item.run_key)),
    )
    if result == catalog:
        return catalog
    _publish_catalog(_ensure_catalog_directory(root), result)
    return result


def rebuild_catalog(
    workspace: str | Path,
    *,
    at: str | None = None,
    max_runs: int = 1_000,
    max_shards: int = 10_000,
    max_event_rows: int = 1_000_000,
    max_source_bytes: int = 1_000_000_000,
) -> tuple[WorkspaceCatalog, CatalogRebuildReceipt]:
    """Reconcile the registry against committed shards through the inventory."""

    for name, value in (
        ("max_runs", max_runs),
        ("max_shards", max_shards),
        ("max_event_rows", max_event_rows),
        ("max_source_bytes", max_source_bytes),
    ):
        _require_strict_positive_int(value, name)
    timestamp = _require_optional_timestamp(at)
    root = _validate_workspace(workspace)
    catalog = read_catalog(root)
    inventory = list_routing_runs(
        root,
        max_runs=max_runs,
        max_shards=max_shards,
        max_event_rows=max_event_rows,
        max_source_bytes=max_source_bytes,
    )
    existing = {entry.run_key: entry for entry in catalog.runs}
    merged = dict(existing)
    added: list[str] = []
    updated: list[str] = []
    unchanged: list[str] = []
    for summary in inventory.runs:
        old = existing.get(summary.run_key)
        refreshed = RunRegistryEntry(
            run_key=summary.run_key,
            specification_fingerprint=(
                old.specification_fingerprint if old is not None else None
            ),
            state=old.state if old is not None else None,
            attempt=old.attempt if old is not None else 1,
            shard_count=summary.shard_count,
            token_event_count=summary.token_count,
            routing_event_count=summary.routing_count,
            token_text_policy=summary.token_text_policy,
            registered_at=old.registered_at if old is not None else timestamp,
            updated_at=(
                timestamp
                if timestamp is not None
                else (old.updated_at if old is not None else None)
            ),
        )
        if old is None:
            added.append(summary.run_key)
        elif _rebuild_entry_changed(old, refreshed):
            updated.append(summary.run_key)
        else:
            unchanged.append(summary.run_key)
        merged[summary.run_key] = refreshed
    observed = {summary.run_key for summary in inventory.runs}
    for key in existing:
        if key not in observed:
            unchanged.append(key)
    if len(merged) > max_runs:
        raise WorkspaceCatalogError("conflict", "run registry exceeds max_runs")
    result = WorkspaceCatalog(
        created_at=catalog.created_at,
        updated_at=timestamp if timestamp is not None else catalog.updated_at,
        runs=tuple(sorted(merged.values(), key=lambda item: item.run_key)),
    )
    receipt = CatalogRebuildReceipt(
        added=tuple(sorted(added)),
        updated=tuple(sorted(updated)),
        unchanged=tuple(sorted(unchanged)),
        removed=(),
        run_count=len(result.runs),
    )
    if result == catalog:
        return catalog, receipt
    _publish_catalog(_ensure_catalog_directory(root), result)
    return result, receipt


def _validate_timestamp_value(value: str | None) -> str | None:
    if value is None:
        return value
    if type(value) is not str or not value:
        raise ValueError("timestamp must be a non-empty string")
    if len(value) > _MAX_TIMESTAMP:
        raise ValueError("timestamp must be at most 64 characters")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
        raise ValueError("timestamp must not contain control characters")
    return value


def _require_optional_timestamp(at: str | None) -> str | None:
    try:
        return _validate_timestamp_value(at)
    except ValueError as exc:
        raise WorkspaceCatalogError("dependency", str(exc), cause=exc) from exc


def _require_strict_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkspaceCatalogError("dependency", f"{name} must be an exact integer")
    if value <= 0:
        raise WorkspaceCatalogError("dependency", f"{name} must be positive")


def _validate_workspace(workspace: str | Path) -> Path:
    if isinstance(workspace, bool) or not isinstance(workspace, str | Path):
        raise TypeError(
            f"workspace must be a string or Path, got {type(workspace).__name__}"
        )
    root = Path(workspace)
    try:
        if root.is_symlink():
            raise ValueError("workspace symlink rejected")
        if not root.exists():
            raise ValueError("workspace directory does not exist")
        if not root.is_dir():
            raise ValueError("workspace path is not a directory")
    except Exception as exc:
        raise WorkspaceCatalogError("workspace", str(exc), cause=exc) from exc
    return root


def _ensure_catalog_directory(root: Path) -> Path:
    directory = root / _CATALOG_DIR_NAME
    try:
        if directory.is_symlink():
            raise ValueError("managed directory symlink rejected")
        if directory.exists():
            if not directory.is_dir():
                raise ValueError("managed path is not a directory")
        else:
            directory.mkdir(mode=0o700)
        if os.name == "posix":
            os.chmod(directory, 0o700)
    except Exception as exc:
        raise WorkspaceCatalogError("workspace", str(exc), cause=exc) from exc
    return directory


def _read_catalog_payload(path: Path) -> dict[str, Any]:
    try:
        if path.is_symlink():
            raise ValueError("catalog manifest symlink rejected")
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise WorkspaceCatalogError(
            "reopen", "workspace catalog is not initialized", cause=exc
        ) from exc
    except Exception as exc:
        raise WorkspaceCatalogError("reopen", str(exc), cause=exc) from exc
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise WorkspaceCatalogError(
            "reopen", "catalog is not valid JSON", cause=exc
        ) from exc
    if not isinstance(payload, dict):
        raise WorkspaceCatalogError("reopen", "catalog payload must be a JSON object")
    version = payload.get("schema_version")
    if version != WORKSPACE_CATALOG_SCHEMA_VERSION:
        raise WorkspaceCatalogError(
            "reopen", f"unsupported catalog schema_version: {version!r}"
        )
    if payload.get("manifest_type") != WorkspaceCatalog.manifest_type:
        raise WorkspaceCatalogError(
            "reopen", "payload is not a workspace catalog manifest"
        )
    return payload


def _rebuild_entry_changed(old: RunRegistryEntry, refreshed: RunRegistryEntry) -> bool:
    old_data = old.model_dump()
    old_data.pop("updated_at")
    new_data = refreshed.model_dump()
    new_data.pop("updated_at")
    return old_data != new_data


def _publish_catalog(directory: Path, catalog: WorkspaceCatalog) -> None:
    final = directory / _CATALOG_FILE_NAME
    payload: dict[str, Any] = {"manifest_type": catalog.manifest_type}
    payload.update(catalog.to_dict())
    body = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    try:
        descriptor, staging_name = tempfile.mkstemp(prefix=_STAGING_PREFIX, dir=directory)
    except Exception as exc:
        raise WorkspaceCatalogError(
            "write", "staging file creation failed", cause=exc
        ) from exc
    staging = Path(staging_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(staging, 0o600)
    except Exception as exc:
        staging.unlink(missing_ok=True)
        raise WorkspaceCatalogError("write", "staging write failed", cause=exc) from exc
    try:
        os.replace(staging, final)
    except Exception as exc:
        staging.unlink(missing_ok=True)
        raise WorkspaceCatalogError(
            "publish", "atomic rename failed", cause=exc
        ) from exc
    try:
        handle = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(handle)
        finally:
            os.close(handle)
    except Exception as exc:
        raise WorkspaceCatalogError(
            "publish", "directory fsync failed", cause=exc
        ) from exc
