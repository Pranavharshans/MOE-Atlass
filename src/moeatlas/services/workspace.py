"""Application-layer workspace and run-registry orchestration.

This module composes the bounded persistence layer (catalog) with the
port/adapter interfaces for routing-run storage. It adds no persistence of its
own: all state lives in the workspace directory's catalog file and shard
storage. The layer is callable by the CLI, the Python API, and a future local
server.

All timestamps are caller-supplied strings; the service never reads a clock.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from moeatlas.store.catalog import (
    CatalogRebuildReceipt,
    RunRegistryEntry,
    WorkspaceCatalog,
    initialize_catalog,
    read_catalog,
    rebuild_catalog,
    upsert_run_entry,
)

if TYPE_CHECKING:
    from moeatlas.runs.lifecycle import RunRecord
    from moeatlas.runs.specs import RunSpecification


@dataclass(frozen=True, slots=True)
class WorkspaceSnapshot:
    """A frozen snapshot of a workspace's catalog at a point in time."""

    path: Path
    catalog: WorkspaceCatalog


def _resolve(workspace: str | Path) -> Path:
    if not isinstance(workspace, str | Path):
        raise TypeError(
            f"workspace must be a string or Path, got {type(workspace).__name__}"
        )
    return Path(workspace).resolve()


def initialize_workspace(
    workspace: str | Path, *, at: str | None = None
) -> WorkspaceCatalog:
    """Initialize a new catalog in the workspace directory.

    The workspace must already be a real, non-symlink directory; the underlying
    catalog layer validates this and raises ``WorkspaceCatalogError("workspace")``
    otherwise.
    """
    return initialize_catalog(_resolve(workspace), at=at)


def open_workspace(workspace: str | Path) -> WorkspaceSnapshot:
    """Open an existing workspace and return a snapshot of its catalog.

    Raises ``WorkspaceCatalogError("reopen")`` if the catalog has not been
    initialized.
    """
    path = _resolve(workspace)
    catalog = read_catalog(path)
    return WorkspaceSnapshot(path=path, catalog=catalog)


def register_run(
    workspace: str | Path,
    specification: RunSpecification,
    *,
    at: str | None = None,
) -> RunRegistryEntry:
    """Register a planned run from a specification.

    The entry is created with state ``"planned"``, attempt ``1``, and the
    specification's run key as the fingerprint. Re-registering the same run key
    is idempotent: the original ``registered_at`` is preserved (the catalog
    layer handles this).
    """
    from moeatlas.runs.specs import RunSpecification as _RunSpec

    if not isinstance(specification, _RunSpec):
        raise TypeError(
            f"specification must be a RunSpecification, "
            f"got {type(specification).__name__}"
        )
    entry = RunRegistryEntry(
        run_key=specification.run_key,
        run_name=specification.run_name,
        specification_fingerprint=specification.run_key,
        state="planned",
        attempt=1,
        registered_at=at,
        updated_at=at,
    )
    catalog = upsert_run_entry(_resolve(workspace), entry, at=at)
    for e in catalog.runs:
        if e.run_key == entry.run_key:
            return e
    raise RuntimeError(f"run {entry.run_key} not found after upsert")


def record_run_record(
    workspace: str | Path,
    record: RunRecord,
    *,
    at: str | None = None,
) -> RunRegistryEntry:
    """Record a lifecycle state update from a ``RunRecord``.

    If the run key is unknown in the catalog, a new entry is auto-registered
    with the record's state, fingerprint, and attempt. If the run is already
    known, the state, attempt, and ``updated_at`` are updated while preserving
    counts, token text policy, fingerprint, and ``registered_at``.
    """
    from moeatlas.runs.lifecycle import RunRecord as _RunRecord

    if not isinstance(record, _RunRecord):
        raise TypeError(
            f"record must be a RunRecord, got {type(record).__name__}"
        )
    path = _resolve(workspace)
    catalog = read_catalog(path)

    existing = {e.run_key: e for e in catalog.runs}
    if record.run_key in existing:
        old = existing[record.run_key]
        fingerprint = (
            old.specification_fingerprint
            if old.specification_fingerprint is not None
            else record.specification_fingerprint
        )
        entry = RunRegistryEntry(
            run_key=record.run_key,
            run_name=old.run_name,
            specification_fingerprint=fingerprint,
            state=record.state.value if record.state else None,
            attempt=record.attempt,
            shard_count=old.shard_count,
            token_event_count=old.token_event_count,
            routing_event_count=old.routing_event_count,
            token_text_policy=old.token_text_policy,
            registered_at=old.registered_at,
            updated_at=at,
        )
    else:
        entry = RunRegistryEntry(
            run_key=record.run_key,
            specification_fingerprint=record.specification_fingerprint,
            state=record.state.value if record.state else None,
            attempt=record.attempt,
            registered_at=at,
            updated_at=at,
        )

    catalog = upsert_run_entry(path, entry, at=at)
    for e in catalog.runs:
        if e.run_key == entry.run_key:
            return e
    raise RuntimeError(f"run {entry.run_key} not found after upsert")


def sync_runs_from_shards(
    workspace: str | Path,
    *,
    at: str | None = None,
    max_runs: int = 1_000,
    max_shards: int = 10_000,
    max_event_rows: int = 1_000_000,
    max_source_bytes: int = 1_000_000_000,
) -> CatalogRebuildReceipt:
    """Synchronise the run registry from shard storage.

    Delegates to the catalog's ``rebuild_catalog`` which reads shard storage
    and reconciles the registry against observed shards.
    """
    _, receipt = rebuild_catalog(
        _resolve(workspace),
        at=at,
        max_runs=max_runs,
        max_shards=max_shards,
        max_event_rows=max_event_rows,
        max_source_bytes=max_source_bytes,
    )
    return receipt


def query_runs(
    workspace: str | Path,
    *,
    state: str | None = None,
    max_results: int = 100,
) -> tuple[RunRegistryEntry, ...]:
    """Query run registry entries, optionally filtered by state.

    When ``state`` is given, it must be a valid ``RunState`` value string.
    ``max_results`` must be a positive ``int``.

    Returns entries in catalog order (sorted by ``run_key``), truncated to
    ``max_results``.
    """
    from moeatlas.runs.lifecycle import RunState as _RunState

    if state is not None:
        if not isinstance(state, str):
            raise TypeError(f"state must be a string, got {type(state).__name__}")
        valid_states = tuple(s.value for s in _RunState)
        try:
            _RunState(state)
        except ValueError:
            raise ValueError(
                f"state must be one of: {', '.join(valid_states)}"
            ) from None

    if not isinstance(max_results, int) or isinstance(max_results, bool):
        raise TypeError(
            f"max_results must be an int, got {type(max_results).__name__}"
        )
    if max_results <= 0:
        raise ValueError("max_results must be a positive int")

    path = _resolve(workspace)
    catalog = read_catalog(path)
    if state is None:
        return catalog.runs[:max_results]
    return tuple(e for e in catalog.runs if e.state == state)[:max_results]
