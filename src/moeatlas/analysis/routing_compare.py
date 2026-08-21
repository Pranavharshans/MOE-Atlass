"""Bounded, read-only comparison of two routing-load matrices."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from ..core import parse_component_key, parse_model_key, validate_stable_identifier
from ..events import EVENT_SCHEMA_VERSION
from ..store import STORE_SCHEMA_VERSION
from .routing_load import RoutingLoadMatrix

ROUTING_COMPARE_SCHEMA_VERSION = "1.0"

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


def _require_equal(baseline: object, comparison: object, field_name: str) -> None:
    if baseline != comparison:
        raise ValueError(f"{field_name} differs between the two runs")


@dataclass(frozen=True, slots=True)
class RoutingLoadComparison:
    schema_version: str
    store_schema_version: str
    event_schema_version: str
    baseline_run_key: str
    comparison_run_key: str
    model_key: str
    adapter_name: str
    adapter_version: str
    inspection_digest: str
    layout: str
    token_count: int
    routed_top_k: int
    baseline_shard_keys: tuple[str, ...]
    comparison_shard_keys: tuple[str, ...]
    baseline_assignment_count: int
    comparison_assignment_count: int
    layer_keys: tuple[str, ...]
    layer_indices: tuple[int, ...]
    expert_keys: tuple[tuple[str, ...], ...]
    count_deltas: tuple[tuple[int, ...], ...]
    share_deltas: tuple[tuple[float, ...], ...]
    ratio_deltas: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not str
            or self.schema_version != ROUTING_COMPARE_SCHEMA_VERSION
        ):
            raise ValueError("schema_version is not the exact routing-compare version")
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
        for field_name in ("baseline_run_key", "comparison_run_key"):
            value = getattr(self, field_name)
            if type(value) is not str:
                raise TypeError(f"{field_name} must be an exact string")
            validate_stable_identifier(value, field_name=field_name)
        if self.baseline_run_key == self.comparison_run_key:
            raise ValueError("baseline and comparison run keys must differ")
        if type(self.model_key) is not str:
            raise TypeError("model_key must be an exact string")
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
        for field_name in (
            "token_count",
            "routed_top_k",
            "baseline_assignment_count",
            "comparison_assignment_count",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field_name} must be a strict positive integer")
        for field_name in ("baseline_shard_keys", "comparison_shard_keys"):
            shard_keys = getattr(self, field_name)
            if type(shard_keys) is not tuple or not shard_keys or any(
                type(key) is not str or _SHARD.fullmatch(key) is None for key in shard_keys
            ):
                raise ValueError(f"{field_name} must be non-empty canonical shard keys")
            if tuple(sorted(shard_keys)) != shard_keys or len(set(shard_keys)) != len(shard_keys):
                raise ValueError(f"{field_name} must be sorted and unique")
        expected_assignments = self.token_count * len(self.layer_keys) * self.routed_top_k
        if self.baseline_assignment_count != expected_assignments:
            raise ValueError(
                "baseline_assignment_count does not match the token/layer/top-k formula"
            )
        if self.comparison_assignment_count != expected_assignments:
            raise ValueError(
                "comparison_assignment_count does not match the token/layer/top-k formula"
            )
        layer_keys = self.layer_keys
        layer_indices = self.layer_indices
        expert_keys = self.expert_keys
        count_rows = self.count_deltas
        share_rows = self.share_deltas
        ratio_rows = self.ratio_deltas
        for field_name, value in (
            ("layer_keys", layer_keys),
            ("layer_indices", layer_indices),
            ("expert_keys", expert_keys),
            ("count_deltas", count_rows),
            ("share_deltas", share_rows),
            ("ratio_deltas", ratio_rows),
        ):
            if type(value) is not tuple:
                raise TypeError(f"{field_name} must be an exact tuple")
        if (
            not layer_keys
            or len(layer_keys) != len(layer_indices)
            or len(layer_keys) != len(expert_keys)
        ):
            raise ValueError("layer axes must have one non-empty row per layer")
        if (
            len(layer_keys) != len(count_rows)
            or len(layer_keys) != len(share_rows)
            or len(layer_keys) != len(ratio_rows)
        ):
            raise ValueError("delta row counts must match layer axes")
        if any(type(key) is not str for key in layer_keys):
            raise TypeError("layer_keys must contain exact strings")
        for key in layer_keys:
            parse_component_key(key)
        if len(set(layer_keys)) != len(layer_keys):
            raise ValueError("layer_keys must be unique")
        if any(
            type(index) is not int or isinstance(index, bool) or index < 0
            for index in layer_indices
        ):
            raise TypeError("layer_indices must contain strict nonnegative integers")
        if tuple(sorted(set(layer_indices))) != layer_indices:
            raise ValueError("layer_indices must be strictly ascending")
        expert_count: int | None = None
        all_expert_keys: set[str] = set()
        for row in expert_keys:
            if type(row) is not tuple or not row:
                raise TypeError("expert_keys rows must be non-empty exact tuples")
            for key in row:
                if type(key) is not str:
                    raise TypeError("expert_keys must contain exact strings")
                parse_component_key(key)
            keys = row
            if len(set(keys)) != len(keys):
                raise ValueError("expert_keys rows must be unique")
            if all_expert_keys.intersection(keys):
                raise ValueError("expert_keys must be globally unique")
            all_expert_keys.update(keys)
            if expert_count is None:
                expert_count = len(keys)
            elif len(keys) != expert_count:
                raise ValueError("expert_keys must be rectangular")
        assert expert_count is not None
        if self.routed_top_k > expert_count:
            raise ValueError("routed_top_k cannot exceed the expert axis")
        for row_index, (count_row, share_row, ratio_row) in enumerate(
            zip(count_rows, share_rows, ratio_rows, strict=True)
        ):
            if type(count_row) is not tuple or len(count_row) != expert_count:
                raise ValueError("count_deltas rows must match the expert axis")
            if type(share_row) is not tuple or len(share_row) != expert_count:
                raise ValueError("share_deltas rows must match the expert axis")
            if type(ratio_row) is not tuple or len(ratio_row) != expert_count:
                raise ValueError("ratio_deltas rows must match the expert axis")
            for delta in count_row:
                if type(delta) is not int or isinstance(delta, bool):
                    raise TypeError("count_deltas must contain exact integers")
            for delta in share_row:
                if type(delta) is not float or not math.isfinite(delta):
                    raise TypeError("share_deltas must contain finite floats")
                if abs(delta) > 1.0 + _TOLERANCE:
                    raise ValueError("share_deltas exceed the unit interval")
            for delta in ratio_row:
                if type(delta) is not float or not math.isfinite(delta):
                    raise TypeError("ratio_deltas must contain finite floats")
                if abs(delta) > expert_count + _TOLERANCE:
                    raise ValueError("ratio_deltas exceed the load-ratio range")
            if sum(count_row) != 0:
                raise ValueError("count_deltas must sum to zero per layer")
            if abs(sum(share_row)) > _TOLERANCE:
                raise ValueError("share_deltas must sum to zero per layer")
            if abs(sum(ratio_row) / expert_count) > _TOLERANCE:
                raise ValueError("ratio_deltas must have mean zero per layer")


def compare_routing_load(
    baseline: RoutingLoadMatrix,
    comparison: RoutingLoadMatrix,
    *,
    max_cells: int,
) -> RoutingLoadComparison:
    """Compare two accepted routing-load matrices over one identical universe."""

    _strict_positive_cells(max_cells)
    if type(baseline) is not RoutingLoadMatrix or type(comparison) is not RoutingLoadMatrix:
        raise TypeError("baseline and comparison must be exact RoutingLoadMatrix values")
    fresh_baseline = _fresh_matrix(baseline)
    fresh_comparison = _fresh_matrix(comparison)
    for field_name in (
        "schema_version",
        "store_schema_version",
        "event_schema_version",
        "model_key",
        "adapter_name",
        "adapter_version",
        "inspection_digest",
        "layout",
        "routed_top_k",
        "token_count",
        "layer_keys",
        "layer_indices",
        "expert_keys",
    ):
        _require_equal(
            getattr(fresh_baseline, field_name),
            getattr(fresh_comparison, field_name),
            field_name,
        )
    if fresh_baseline.run_key == fresh_comparison.run_key:
        raise ValueError("baseline and comparison run keys must differ")
    cells = len(fresh_baseline.layer_keys) * len(fresh_baseline.expert_keys[0])
    if cells > max_cells:
        raise ValueError("matrix cells exceed max_cells")

    count_deltas = tuple(
        tuple(comparison_count - baseline_count for comparison_count, baseline_count in zip(
            comparison_counts, baseline_counts, strict=True
        ))
        for comparison_counts, baseline_counts in zip(
            fresh_comparison.assignment_counts, fresh_baseline.assignment_counts, strict=True
        )
    )
    share_deltas = tuple(
        tuple(comparison_share - baseline_share for comparison_share, baseline_share in zip(
            comparison_shares, baseline_shares, strict=True
        ))
        for comparison_shares, baseline_shares in zip(
            fresh_comparison.assignment_shares, fresh_baseline.assignment_shares, strict=True
        )
    )
    ratio_deltas = tuple(
        tuple(comparison_ratio - baseline_ratio for comparison_ratio, baseline_ratio in zip(
            comparison_ratios, baseline_ratios, strict=True
        ))
        for comparison_ratios, baseline_ratios in zip(
            fresh_comparison.load_ratios, fresh_baseline.load_ratios, strict=True
        )
    )
    return RoutingLoadComparison(
        schema_version=ROUTING_COMPARE_SCHEMA_VERSION,
        store_schema_version=fresh_baseline.store_schema_version,
        event_schema_version=fresh_baseline.event_schema_version,
        baseline_run_key=fresh_baseline.run_key,
        comparison_run_key=fresh_comparison.run_key,
        model_key=fresh_baseline.model_key,
        adapter_name=fresh_baseline.adapter_name,
        adapter_version=fresh_baseline.adapter_version,
        inspection_digest=fresh_baseline.inspection_digest,
        layout=fresh_baseline.layout,
        token_count=fresh_baseline.token_count,
        routed_top_k=fresh_baseline.routed_top_k,
        baseline_shard_keys=fresh_baseline.shard_keys,
        comparison_shard_keys=fresh_comparison.shard_keys,
        baseline_assignment_count=fresh_baseline.assignment_count,
        comparison_assignment_count=fresh_comparison.assignment_count,
        layer_keys=fresh_baseline.layer_keys,
        layer_indices=fresh_baseline.layer_indices,
        expert_keys=fresh_baseline.expert_keys,
        count_deltas=count_deltas,
        share_deltas=share_deltas,
        ratio_deltas=ratio_deltas,
    )


__all__ = [
    "ROUTING_COMPARE_SCHEMA_VERSION",
    "RoutingLoadComparison",
    "compare_routing_load",
]
