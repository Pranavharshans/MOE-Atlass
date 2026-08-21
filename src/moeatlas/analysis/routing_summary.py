"""Bounded, read-only summary metrics over one routing-load matrix."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

from ..core import parse_component_key, parse_model_key, validate_stable_identifier
from ..events import EVENT_SCHEMA_VERSION
from ..store import STORE_SCHEMA_VERSION
from .routing_load import RoutingLoadMatrix

ROUTING_SUMMARY_SCHEMA_VERSION = "1.0"

_ROUTING_SUMMARY_ARTIFACT_TYPE = "moeatlas.routing_load_summary"

_TOLERANCE = 1e-12
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_SHARD = re.compile(r"^shard:[0-9a-f]{64}$")
_LAYOUTS = frozenset({"legacy_indexed", "packed"})


def _strict_positive_cells(value: object) -> int:
    if type(value) is not int or isinstance(value, bool):
        raise TypeError("max_cells must be an exact positive integer")
    if value <= 0:
        raise ValueError("max_cells must be positive")
    return value


def _fresh_matrix(matrix: RoutingLoadMatrix) -> RoutingLoadMatrix:
    return RoutingLoadMatrix(
        schema_version=matrix.schema_version,
        store_schema_version=matrix.store_schema_version,
        event_schema_version=matrix.event_schema_version,
        run_key=matrix.run_key,
        model_key=matrix.model_key,
        adapter_name=matrix.adapter_name,
        adapter_version=matrix.adapter_version,
        inspection_digest=matrix.inspection_digest,
        layout=matrix.layout,
        shard_keys=matrix.shard_keys,
        token_count=matrix.token_count,
        assignment_count=matrix.assignment_count,
        routed_top_k=matrix.routed_top_k,
        layer_keys=matrix.layer_keys,
        layer_indices=matrix.layer_indices,
        expert_keys=matrix.expert_keys,
        assignment_counts=matrix.assignment_counts,
        assignment_shares=matrix.assignment_shares,
        load_ratios=matrix.load_ratios,
    )


def _entropy(row_shares: tuple[float, ...]) -> float:
    """Shannon entropy in nats; zero shares contribute exactly zero."""

    total = 0.0
    for share in row_shares:
        if share > 0.0:
            total -= share * math.log(share)
    return total


def _gini(counts: tuple[int, ...]) -> float:
    """Exact ascending-rank Gini over one layer's integer counts."""

    ordered = sorted(counts)
    denominator = len(ordered) * sum(ordered)
    if denominator == 0:
        return 0.0
    weighted = sum((position + 1) * count for position, count in enumerate(ordered))
    return (2 * weighted) / denominator - (len(ordered) + 1) / len(ordered)


def _coefficient_of_variation(counts: tuple[int, ...]) -> float:
    expert_count = len(counts)
    mean = sum(counts) / expert_count
    variance = sum((count - mean) ** 2 for count in counts) / expert_count
    return math.sqrt(variance) / mean


@dataclass(frozen=True, slots=True)
class RoutingLoadSummary:
    schema_version: str
    store_schema_version: str
    event_schema_version: str
    run_key: str
    model_key: str
    adapter_name: str
    adapter_version: str
    inspection_digest: str
    layout: str
    token_count: int
    routed_top_k: int
    assignment_count: int
    shard_keys: tuple[str, ...]
    layer_keys: tuple[str, ...]
    layer_indices: tuple[int, ...]
    expert_keys: tuple[tuple[str, ...], ...]
    layer_entropies: tuple[float, ...]
    normalized_layer_entropies: tuple[float, ...]
    effective_expert_counts: tuple[float, ...]
    normalized_diversities: tuple[float, ...]
    layer_gini_coefficients: tuple[float, ...]
    layer_cv_counts: tuple[float, ...]
    top_expert_shares: tuple[float, ...]
    dead_expert_count: int
    dead_expert_fraction: float

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not str
            or self.schema_version != ROUTING_SUMMARY_SCHEMA_VERSION
        ):
            raise ValueError("schema_version is not the exact routing-summary version")
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
        parse_model_key(self.model_key)
        if (
            type(self.adapter_name) is not str
            or type(self.adapter_version) is not str
            or not self.adapter_name
            or not self.adapter_version
            or self.adapter_name != self.adapter_name.strip()
            or self.adapter_version != self.adapter_version.strip()
        ):
            raise ValueError("adapter identity is not complete")
        if (
            type(self.inspection_digest) is not str
            or _DIGEST.fullmatch(self.inspection_digest) is None
        ):
            raise ValueError("inspection_digest must be sha256:<64hex>")
        if type(self.layout) is not str or self.layout not in _LAYOUTS:
            raise ValueError("layout must be legacy_indexed or packed")
        for field_name in ("token_count", "routed_top_k", "assignment_count"):
            value = getattr(self, field_name)
            if type(value) is not int or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field_name} must be a strict positive integer")
        if type(self.shard_keys) is not tuple or not self.shard_keys or any(
            type(key) is not str or _SHARD.fullmatch(key) is None for key in self.shard_keys
        ):
            raise ValueError("shard_keys must be non-empty canonical shard keys")
        if (
            tuple(sorted(self.shard_keys)) != self.shard_keys
            or len(set(self.shard_keys)) != len(self.shard_keys)
        ):
            raise ValueError("shard_keys must be sorted and unique")
        axes = (
            ("layer_keys", self.layer_keys),
            ("layer_indices", self.layer_indices),
            ("expert_keys", self.expert_keys),
            ("layer_entropies", self.layer_entropies),
            ("normalized_layer_entropies", self.normalized_layer_entropies),
            ("effective_expert_counts", self.effective_expert_counts),
            ("normalized_diversities", self.normalized_diversities),
            ("layer_gini_coefficients", self.layer_gini_coefficients),
            ("layer_cv_counts", self.layer_cv_counts),
            ("top_expert_shares", self.top_expert_shares),
        )
        for field_name, value in axes:
            if type(value) is not tuple:
                raise TypeError(f"{field_name} must be an exact tuple")
        layer_count = len(self.layer_keys)
        if not self.layer_keys or any(
            len(getattr(self, name)) != layer_count
            for name, _ in axes[3:]
        ) or len(self.layer_indices) != layer_count or len(self.expert_keys) != layer_count:
            raise ValueError("summary rows must match the layer axis")
        if any(type(key) is not str for key in self.layer_keys):
            raise TypeError("layer_keys must contain exact strings")
        for key in self.layer_keys:
            parse_component_key(key)
        if len(set(self.layer_keys)) != layer_count:
            raise ValueError("layer_keys must be unique")
        if any(
            type(index) is not int or isinstance(index, bool) or index < 0
            for index in self.layer_indices
        ):
            raise TypeError("layer_indices must contain strict nonnegative integers")
        if tuple(sorted(set(self.layer_indices))) != self.layer_indices:
            raise ValueError("layer_indices must be strictly ascending")
        expert_count: int | None = None
        seen_expert_keys: set[str] = set()
        for row in self.expert_keys:
            if type(row) is not tuple or not row:
                raise TypeError("expert_keys rows must be non-empty exact tuples")
            for key in row:
                if type(key) is not str:
                    raise TypeError("expert_keys must contain exact strings")
                parse_component_key(key)
            if len(set(row)) != len(row):
                raise ValueError("expert_keys rows must be unique")
            if seen_expert_keys.intersection(row):
                raise ValueError("expert_keys must be globally unique")
            seen_expert_keys.update(row)
            if expert_count is None:
                expert_count = len(row)
            elif len(row) != expert_count:
                raise ValueError("expert_keys must be rectangular")
        assert expert_count is not None
        if self.routed_top_k > expert_count:
            raise ValueError("routed_top_k cannot exceed the expert axis")
        maximum_entropy = math.log(expert_count)
        cells = layer_count * expert_count
        for field_name, low, high in (
            ("layer_entropies", -_TOLERANCE, maximum_entropy + _TOLERANCE),
            ("normalized_layer_entropies", -_TOLERANCE, 1.0 + _TOLERANCE),
            ("normalized_diversities", -_TOLERANCE, 1.0 + _TOLERANCE),
            ("layer_gini_coefficients", -_TOLERANCE, 1.0 + _TOLERANCE),
            ("top_expert_shares", -_TOLERANCE, 1.0 + _TOLERANCE),
        ):
            for value in getattr(self, field_name):
                if type(value) is not float or not math.isfinite(value):
                    raise TypeError(f"{field_name} must contain finite floats")
                if value < low or value > high:
                    raise ValueError(f"{field_name} contains an out-of-range value")
        for value in self.effective_expert_counts:
            if type(value) is not float or not math.isfinite(value):
                raise TypeError("effective_expert_counts must contain finite floats")
            if value < -_TOLERANCE or value > expert_count + _TOLERANCE:
                raise ValueError("effective_expert_counts contains an out-of-range value")
        for value in self.layer_cv_counts:
            if type(value) is not float or not math.isfinite(value) or value < -_TOLERANCE:
                raise TypeError("layer_cv_counts must contain finite nonnegative floats")
        if type(self.dead_expert_count) is not int or isinstance(
            self.dead_expert_count, bool
        ) or self.dead_expert_count < 0 or self.dead_expert_count > cells:
            raise ValueError("dead_expert_count must be within the cell universe")
        if type(self.dead_expert_fraction) is not float or not math.isfinite(
            self.dead_expert_fraction
        ):
            raise TypeError("dead_expert_fraction must be a finite float")
        if abs(self.dead_expert_fraction - self.dead_expert_count / cells) > _TOLERANCE:
            raise ValueError("dead_expert_fraction does not match dead_expert_count")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible data without runtime objects."""

        return {
            "artifact_type": _ROUTING_SUMMARY_ARTIFACT_TYPE,
            "schema_version": self.schema_version,
            "store_schema_version": self.store_schema_version,
            "event_schema_version": self.event_schema_version,
            "run_key": self.run_key,
            "model_key": self.model_key,
            "adapter_name": self.adapter_name,
            "adapter_version": self.adapter_version,
            "inspection_digest": self.inspection_digest,
            "layout": self.layout,
            "token_count": self.token_count,
            "routed_top_k": self.routed_top_k,
            "assignment_count": self.assignment_count,
            "shard_keys": list(self.shard_keys),
            "layer_keys": list(self.layer_keys),
            "layer_indices": list(self.layer_indices),
            "expert_keys": [list(row) for row in self.expert_keys],
            "layer_entropies": list(self.layer_entropies),
            "normalized_layer_entropies": list(self.normalized_layer_entropies),
            "effective_expert_counts": list(self.effective_expert_counts),
            "normalized_diversities": list(self.normalized_diversities),
            "layer_gini_coefficients": list(self.layer_gini_coefficients),
            "layer_cv_counts": list(self.layer_cv_counts),
            "top_expert_shares": list(self.top_expert_shares),
            "dead_expert_count": self.dead_expert_count,
            "dead_expert_fraction": self.dead_expert_fraction,
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
    def from_json(cls, payload: str | bytes | bytearray) -> RoutingLoadSummary:
        """Validate one canonical JSON document into an exact summary value."""

        from .routing_load import (
            _strict_float_rows,
            _strict_index_tuple,
            _strict_row_tuple,
            _strict_string_tuple,
        )

        def strict_float_tuple(value: object, field_name: str) -> tuple[float, ...]:
            rows = _strict_float_rows([value], field_name)
            return rows[0]

        try:
            document = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("routing load summary document is not valid JSON") from exc
        if type(document) is not dict:
            raise ValueError("routing load summary document must be a JSON object")
        if (
            document.get("artifact_type") != _ROUTING_SUMMARY_ARTIFACT_TYPE
            or document.get("schema_version") != ROUTING_SUMMARY_SCHEMA_VERSION
        ):
            raise ValueError("document is not a routing load summary artifact")
        try:
            return cls(
                schema_version=document["schema_version"],
                store_schema_version=document["store_schema_version"],
                event_schema_version=document["event_schema_version"],
                run_key=document["run_key"],
                model_key=document["model_key"],
                adapter_name=document["adapter_name"],
                adapter_version=document["adapter_version"],
                inspection_digest=document["inspection_digest"],
                layout=document["layout"],
                token_count=document["token_count"],
                routed_top_k=document["routed_top_k"],
                assignment_count=document["assignment_count"],
                shard_keys=_strict_string_tuple(document["shard_keys"], "shard_keys"),
                layer_keys=_strict_string_tuple(document["layer_keys"], "layer_keys"),
                layer_indices=_strict_index_tuple(document["layer_indices"], "layer_indices"),
                expert_keys=_strict_row_tuple(document["expert_keys"], "expert_keys"),
                layer_entropies=strict_float_tuple(
                    document["layer_entropies"], "layer_entropies"
                ),
                normalized_layer_entropies=strict_float_tuple(
                    document["normalized_layer_entropies"], "normalized_layer_entropies"
                ),
                effective_expert_counts=strict_float_tuple(
                    document["effective_expert_counts"], "effective_expert_counts"
                ),
                normalized_diversities=strict_float_tuple(
                    document["normalized_diversities"], "normalized_diversities"
                ),
                layer_gini_coefficients=strict_float_tuple(
                    document["layer_gini_coefficients"], "layer_gini_coefficients"
                ),
                layer_cv_counts=strict_float_tuple(
                    document["layer_cv_counts"], "layer_cv_counts"
                ),
                top_expert_shares=strict_float_tuple(
                    document["top_expert_shares"], "top_expert_shares"
                ),
                dead_expert_count=document["dead_expert_count"],
                dead_expert_fraction=document["dead_expert_fraction"],
            )
        except KeyError as exc:
            raise ValueError("routing load summary document is missing fields") from exc
        except (TypeError, ValueError):
            raise
        except Exception as exc:
            raise ValueError("routing load summary document is not usable") from exc


def summarize_routing_load(
    matrix: RoutingLoadMatrix,
    *,
    max_cells: int,
) -> RoutingLoadSummary:
    """Summarize one accepted routing-load matrix with bounded load metrics."""

    _strict_positive_cells(max_cells)
    if type(matrix) is not RoutingLoadMatrix:
        raise TypeError("matrix must be an exact RoutingLoadMatrix")
    fresh = _fresh_matrix(matrix)
    expert_count = len(fresh.expert_keys[0])
    cells = len(fresh.layer_keys) * expert_count
    if cells > max_cells:
        raise ValueError("matrix cells exceed max_cells")

    layer_entropies = tuple(_entropy(row) for row in fresh.assignment_shares)
    normalized = tuple(value / math.log(expert_count) for value in layer_entropies)
    effective = tuple(math.exp(value) for value in layer_entropies)
    diversities = tuple(value / expert_count for value in effective)
    ginis = tuple(_gini(row) for row in fresh.assignment_counts)
    cvs = tuple(_coefficient_of_variation(row) for row in fresh.assignment_counts)
    tops = tuple(max(row) for row in fresh.assignment_shares)
    dead = sum(1 for row in fresh.assignment_counts for count in row if count == 0)
    return RoutingLoadSummary(
        schema_version=ROUTING_SUMMARY_SCHEMA_VERSION,
        store_schema_version=fresh.store_schema_version,
        event_schema_version=fresh.event_schema_version,
        run_key=fresh.run_key,
        model_key=fresh.model_key,
        adapter_name=fresh.adapter_name,
        adapter_version=fresh.adapter_version,
        inspection_digest=fresh.inspection_digest,
        layout=fresh.layout,
        token_count=fresh.token_count,
        routed_top_k=fresh.routed_top_k,
        assignment_count=fresh.assignment_count,
        shard_keys=fresh.shard_keys,
        layer_keys=fresh.layer_keys,
        layer_indices=fresh.layer_indices,
        expert_keys=fresh.expert_keys,
        layer_entropies=layer_entropies,
        normalized_layer_entropies=normalized,
        effective_expert_counts=effective,
        normalized_diversities=diversities,
        layer_gini_coefficients=ginis,
        layer_cv_counts=cvs,
        top_expert_shares=tops,
        dead_expert_count=dead,
        dead_expert_fraction=dead / cells,
    )


__all__ = [
    "ROUTING_SUMMARY_SCHEMA_VERSION",
    "RoutingLoadSummary",
    "summarize_routing_load",
]
