"""Bounded per-layer expert-activity summaries over immutable shards.

This module turns persisted ``experts.parquet`` rows into one canonical,
frozen, round-trippable activation summary value. It reads exclusively
through the public storage query seam (:func:`moeatlas.store.
query_expert_activity`), aggregates mean/max contribution norms per layer
and expert with explicit zero-activity accounting for universe cells that
never fired, and retains no raw rows anywhere.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core import parse_component_key, validate_stable_identifier
from ..events import EVENT_SCHEMA_VERSION
from ..store import STORE_SCHEMA_VERSION
from ..store import routing_shards as _storage

EXPERT_ACTIVITY_SUMMARY_SCHEMA_VERSION = "1.0"

_EXPERT_ACTIVITY_ARTIFACT_TYPE = "moeatlas.expert_activity_summary"

_SHARD = re.compile(r"^shard:[0-9a-f]{64}$")


def _strict_positive_budget(value: object, name: str) -> int:
    if type(value) is not int or isinstance(value, bool):
        raise TypeError(f"{name} must be a strict positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a strict positive integer")
    return value


def _strict_string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if type(value) is not tuple or not value:
        raise TypeError(f"{field_name} must be a non-empty exact tuple")
    for entry in value:
        if type(entry) is not str:
            raise TypeError(f"{field_name} entries must be exact strings")
        parse_component_key(entry)
    return value


def _strict_shard_keys(value: object, field_name: str) -> tuple[str, ...]:
    if type(value) is not tuple or not value:
        raise TypeError(f"{field_name} must be a non-empty exact tuple")
    for entry in value:
        if type(entry) is not str or _SHARD.fullmatch(entry) is None:
            raise ValueError(f"{field_name} must be canonical shard keys")
    return value


def _strict_universe_rows(value: object, field_name: str) -> tuple[tuple[str, ...], ...]:
    if type(value) is not tuple or not value:
        raise TypeError(f"{field_name} must be a non-empty exact tuple of rows")
    rows: list[tuple[str, ...]] = []
    seen: set[str] = set()
    for row in value:
        if type(row) is not tuple or not row:
            raise TypeError(f"{field_name} rows must be non-empty exact tuples")
        for key in row:
            if type(key) is not str:
                raise TypeError(f"{field_name} keys must be exact strings")
            parse_component_key(key)
            if key in seen:
                raise ValueError(f"{field_name} keys must be globally unique")
            seen.add(key)
        rows.append(tuple(row))
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class LayerExpertActivity:
    """One canonical per-layer activity row over the discovered experts."""

    layer_key: str
    event_counts: tuple[int, ...]
    mean_contributions: tuple[float | None, ...]
    max_contributions: tuple[float | None, ...]

    def __post_init__(self) -> None:
        if type(self.layer_key) is not str:
            raise TypeError("layer_key must be an exact string")
        parse_component_key(self.layer_key)
        axes = (
            ("event_counts", self.event_counts),
            ("mean_contributions", self.mean_contributions),
            ("max_contributions", self.max_contributions),
        )
        width = len(self.event_counts)
        if width == 0:
            raise ValueError("activity rows must cover at least one expert cell")
        for name, values in axes:
            if type(values) is not tuple or len(values) != width:
                raise ValueError(f"{name} must match the expert-cell axis")
        for count in self.event_counts:
            if type(count) is not int or isinstance(count, bool) or count < 0:
                raise ValueError("event_counts must contain nonnegative integers")
        for name in ("mean_contributions", "max_contributions"):
            for value in getattr(self, name):
                if value is None:
                    continue
                if type(value) is not float or not math.isfinite(value) or value < 0.0:
                    raise ValueError(f"{name} must contain finite nonnegative floats or null")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible data without runtime objects."""

        return {
            "layer_key": self.layer_key,
            "event_counts": list(self.event_counts),
            "mean_contributions": list(self.mean_contributions),
            "max_contributions": list(self.max_contributions),
        }


@dataclass(frozen=True, slots=True)
class ExpertActivitySummary:
    """Canonical frozen summary of one run's per-layer expert activity."""

    schema_version: str
    store_schema_version: str
    event_schema_version: str
    run_key: str
    shard_keys: tuple[str, ...]
    layer_keys: tuple[str, ...]
    expert_keys: tuple[tuple[str, ...], ...]
    layers: tuple[LayerExpertActivity, ...]
    active_expert_cells: int
    inactive_expert_cells: int
    total_event_count: int

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not str
            or self.schema_version != EXPERT_ACTIVITY_SUMMARY_SCHEMA_VERSION
        ):
            raise ValueError("schema_version is not the exact expert-activity version")
        if (
            type(self.store_schema_version) is not str
            or self.store_schema_version != STORE_SCHEMA_VERSION
        ):
            raise ValueError("store_schema_version is not the exact store version")
        if (
            type(self.event_schema_version) is not str
            or self.event_schema_version != EVENT_SCHEMA_VERSION
        ):
            raise ValueError("event_schema_version is not the exact event version")
        if type(self.run_key) is not str:
            raise TypeError("run_key must be an exact string")
        validate_stable_identifier(self.run_key, field_name="run_key")
        if type(self.shard_keys) is not tuple or any(
            type(key) is not str or _SHARD.fullmatch(key) is None for key in self.shard_keys
        ):
            raise ValueError("shard_keys must be canonical shard keys")
        if tuple(sorted(self.shard_keys)) != self.shard_keys:
            raise ValueError("shard_keys must be sorted and unique")
        if type(self.layer_keys) is not tuple or not self.layer_keys:
            raise ValueError("layer_keys must be a non-empty tuple")
        for key in self.layer_keys:
            if type(key) is not str:
                raise TypeError("layer_keys must contain exact strings")
            parse_component_key(key)
        if len(set(self.layer_keys)) != len(self.layer_keys):
            raise ValueError("layer_keys must be unique")
        if type(self.expert_keys) is not tuple or len(self.expert_keys) != len(self.layer_keys):
            raise ValueError("expert_keys must match layer_keys exactly")
        seen_experts: set[str] = set()
        for row in self.expert_keys:
            if type(row) is not tuple or not row:
                raise ValueError("expert_keys rows must be non-empty tuples")
            for key in row:
                if type(key) is not str:
                    raise TypeError("expert_keys must contain exact strings")
                parse_component_key(key)
                if key in seen_experts:
                    raise ValueError("expert_keys must be globally unique")
                seen_experts.add(key)
        if type(self.layers) is not tuple or any(
            type(row) is not LayerExpertActivity for row in self.layers
        ):
            raise TypeError("layers must be a tuple of exact LayerExpertActivity rows")
        if len(self.layers) != len(self.layer_keys):
            raise ValueError("layers must match the layer axis")
        for position, row in enumerate(self.layers):
            if row.layer_key != self.layer_keys[position]:
                raise ValueError("layer rows must use the canonical layer order")
            if len(row.event_counts) != len(self.expert_keys[position]):
                raise ValueError("layer rows must match their expert axis")
        cells = sum(len(row) for row in self.expert_keys)
        for name in ("active_expert_cells", "inactive_expert_cells", "total_event_count"):
            value = getattr(self, name)
            if type(value) is not int or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.active_expert_cells + self.inactive_expert_cells != cells:
            raise ValueError("active and inactive cells must partition the expert universe")
        observed_active = sum(
            1 for row in self.layers for count in row.event_counts if count > 0
        )
        if self.active_expert_cells != observed_active:
            raise ValueError("active_expert_cells does not match the layer rows")
        if self.total_event_count != sum(
            count for row in self.layers for count in row.event_counts
        ):
            raise ValueError("total_event_count does not match the layer rows")
        for row in self.layers:
            for count, mean_value, max_value in zip(
                row.event_counts, row.mean_contributions, row.max_contributions
            ):
                if count == 0 and (mean_value is not None or max_value is not None):
                    raise ValueError("zero-activity cells must carry null contributions")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible data without runtime objects."""

        return {
            "artifact_type": _EXPERT_ACTIVITY_ARTIFACT_TYPE,
            "schema_version": self.schema_version,
            "store_schema_version": self.store_schema_version,
            "event_schema_version": self.event_schema_version,
            "run_key": self.run_key,
            "shard_keys": list(self.shard_keys),
            "layer_keys": list(self.layer_keys),
            "expert_keys": [list(row) for row in self.expert_keys],
            "layers": [row.to_dict() for row in self.layers],
            "active_expert_cells": self.active_expert_cells,
            "inactive_expert_cells": self.inactive_expert_cells,
            "total_event_count": self.total_event_count,
        }

    def to_json(self) -> str:
        """Serialize this summary with deterministic key order."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> ExpertActivitySummary:
        """Validate one canonical JSON document into an exact summary value."""

        try:
            document = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("expert activity summary document is not valid JSON") from exc
        if type(document) is not dict:
            raise ValueError("expert activity summary document must be a JSON object")
        if (
            document.get("artifact_type") != _EXPERT_ACTIVITY_ARTIFACT_TYPE
            or document.get("schema_version") != EXPERT_ACTIVITY_SUMMARY_SCHEMA_VERSION
        ):
            raise ValueError("document is not an expert activity summary artifact")

        def optional_floats(value: object, field_name: str) -> tuple[float | None, ...]:
            if type(value) is not list:
                raise TypeError(f"{field_name} must be a list")
            result: list[float | None] = []
            for entry in value:
                if entry is None:
                    result.append(None)
                    continue
                if type(entry) is not float or not math.isfinite(entry) or entry < 0.0:
                    raise ValueError(f"{field_name} must contain finite nonnegative floats")
                result.append(entry)
            return tuple(result)

        try:
            layers = tuple(
                LayerExpertActivity(
                    layer_key=row["layer_key"],
                    event_counts=tuple(row["event_counts"]),
                    mean_contributions=optional_floats(
                        row["mean_contributions"], "mean_contributions"
                    ),
                    max_contributions=optional_floats(
                        row["max_contributions"], "max_contributions"
                    ),
                )
                for row in document["layers"]
            )
            return cls(
                schema_version=document["schema_version"],
                store_schema_version=document["store_schema_version"],
                event_schema_version=document["event_schema_version"],
                run_key=document["run_key"],
                shard_keys=_strict_shard_keys(tuple(document["shard_keys"]), "shard_keys"),
                layer_keys=_strict_string_tuple(
                    tuple(document["layer_keys"]), "layer_keys"
                ),
                expert_keys=_strict_universe_rows(
                    tuple(tuple(row) for row in document["expert_keys"]), "expert_keys"
                ),
                layers=layers,
                active_expert_cells=document["active_expert_cells"],
                inactive_expert_cells=document["inactive_expert_cells"],
                total_event_count=document["total_event_count"],
            )
        except KeyError as exc:
            raise ValueError("expert activity summary document is missing fields") from exc
        except (TypeError, ValueError):
            raise
        except Exception as exc:
            raise ValueError("expert activity summary document is not usable") from exc


def summarize_expert_activity(
    workspace: str | Path,
    *,
    run_key: str,
    layer_keys: tuple[str, ...],
    expert_keys: tuple[tuple[str, ...], ...],
    max_expert_rows: int,
    max_source_bytes: int,
) -> ExpertActivitySummary:
    """Summarize persisted expert events for one run over its layer universe.

    Every committed shard is reopened and fully validated by the storage query
    seam; contribution aggregates are computed per layer and expert with
    explicit zero-activity accounting (cells that never fired contribute a
    zero count and null statistics). No raw expert rows are retained.
    """

    if not isinstance(workspace, str | Path):
        raise TypeError("workspace must be a string or pathlib.Path")
    if type(run_key) is not str:
        raise TypeError("run_key must be an exact string")
    validate_stable_identifier(run_key, field_name="run_key")
    _strict_positive_budget(max_expert_rows, "max_expert_rows")
    _strict_positive_budget(max_source_bytes, "max_source_bytes")
    stable_layers = _strict_string_tuple(layer_keys, "layer_keys")
    stable_universe = _strict_universe_rows(expert_keys, "expert_keys")

    # Read through the public storage query seam; analysis owns the bounded
    # in-memory connection lifecycle exactly once, mirroring routing_load.
    primary: BaseException | None = None
    records: tuple[_storage.RoutingShardExpertActivityQuery, ...] | None = None
    connection: Any | None = None
    try:
        duckdb = _storage._load_duckdb()
        connection = duckdb.connect(database=":memory:")
        records = _storage.query_expert_activity(
            workspace,
            run_key=run_key,
            layer_keys=stable_layers,
            expert_keys=stable_universe,
            max_expert_rows=max_expert_rows,
            max_source_bytes=max_source_bytes,
            duckdb=duckdb,
            connection=connection,
        )
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt | SystemExit):
            primary = exc
        elif isinstance(exc, _storage.RoutingShardError):
            primary = exc
        elif isinstance(exc, _storage.RoutingRunInventoryError):
            primary = ValueError("expert activity budgets were exceeded")
            primary.__cause__ = exc
        else:
            primary = ValueError("expert activity sources are unusable")
            primary.__cause__ = exc
    finally:
        if connection is not None:
            try:
                connection.close()
            except BaseException as close_error:
                if primary is None:
                    primary = close_error
    if primary is not None:
        raise primary
    assert records is not None

    totals_events: dict[tuple[str, str], int] = {}
    totals_measured: dict[tuple[str, str], int] = {}
    totals_sums: dict[tuple[str, str], list[float]] = {}
    totals_peaks: dict[tuple[str, str], list[float]] = {}
    shard_keys: list[str] = []
    for record in records:
        shard_keys.append(record.shard_key)
        for cell_layer, cell_expert, event_count, measured, total, peak in (
            record.activity_cells
        ):
            cell = (cell_layer, cell_expert)
            totals_events[cell] = totals_events.get(cell, 0) + event_count
            totals_measured[cell] = totals_measured.get(cell, 0) + measured
            totals_sums.setdefault(cell, []).append(total)
            totals_peaks.setdefault(cell, []).append(peak)

    layer_rows: list[LayerExpertActivity] = []
    active_cells = 0
    total_events = 0
    for layer_position, layer_key in enumerate(stable_layers):
        counts: list[int] = []
        means: list[float | None] = []
        maxima: list[float | None] = []
        for expert_key in stable_universe[layer_position]:
            cell = (layer_key, expert_key)
            event_count = totals_events.get(cell)
            if event_count is None:
                counts.append(0)
                means.append(None)
                maxima.append(None)
                continue
            measured = totals_measured[cell]
            sums = totals_sums[cell]
            peaks = totals_peaks[cell]
            counts.append(event_count)
            means.append(math.fsum(sums) / measured if measured else None)
            maxima.append(max(peaks) if peaks else None)
            total_events += event_count
            if event_count > 0:
                active_cells += 1
        layer_rows.append(
            LayerExpertActivity(
                layer_key=layer_key,
                event_counts=tuple(counts),
                mean_contributions=tuple(means),
                max_contributions=tuple(maxima),
            )
        )
    cells = sum(len(row) for row in stable_universe)
    return ExpertActivitySummary(
        schema_version=EXPERT_ACTIVITY_SUMMARY_SCHEMA_VERSION,
        store_schema_version=STORE_SCHEMA_VERSION,
        event_schema_version=EVENT_SCHEMA_VERSION,
        run_key=run_key,
        shard_keys=tuple(sorted(shard_keys)),
        layer_keys=stable_layers,
        expert_keys=stable_universe,
        layers=tuple(layer_rows),
        active_expert_cells=active_cells,
        inactive_expert_cells=cells - active_cells,
        total_event_count=total_events,
    )


__all__ = [
    "EXPERT_ACTIVITY_SUMMARY_SCHEMA_VERSION",
    "ExpertActivitySummary",
    "LayerExpertActivity",
    "summarize_expert_activity",
]
