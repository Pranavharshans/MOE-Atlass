"""Model-neutral similarity evidence for two routing-load distributions.

Unlike the exact delta comparison, this analysis permits different token
counts.  It compares normalized per-layer expert distributions only after the
immutable model, adapter, inspection, layout, and routing universe match.
That makes it suitable for cross-dataset or cross-task comparisons without
turning sample size into a routing effect.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from .routing_load import RoutingLoadMatrix

ROUTING_SIMILARITY_SCHEMA_VERSION = "1.0"
_ROUTING_SIMILARITY_ARTIFACT_TYPE = "moeatlas.routing_similarity"
_TOLERANCE = 1e-12


def _finite_unit(value: object, field_name: str) -> float:
    if type(value) not in (float, int):
        raise TypeError(f"{field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < -_TOLERANCE or result > 1.0 + _TOLERANCE:
        raise ValueError(f"{field_name} must be within the unit interval")
    return min(1.0, max(0.0, result))


def _finite_correlation(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    if type(value) not in (float, int):
        raise TypeError(f"{field_name} must be numeric or null")
    result = float(value)
    if not math.isfinite(result) or result < -1.0 - _TOLERANCE or result > 1.0 + _TOLERANCE:
        raise ValueError(f"{field_name} must be within [-1, 1]")
    return min(1.0, max(-1.0, result))


@dataclass(frozen=True, slots=True)
class RoutingSimilarity:
    """Per-layer distribution similarity with explicit rank uncertainty."""

    schema_version: str
    baseline_run_key: str
    comparison_run_key: str
    model_key: str
    adapter_name: str
    adapter_version: str
    inspection_digest: str
    layout: str
    baseline_token_count: int
    comparison_token_count: int
    top_n: int
    layer_keys: tuple[str, ...]
    js_divergence_rows: tuple[float, ...]
    spearman_rows: tuple[float | None, ...]
    top_n_jaccard_rows: tuple[float, ...]
    baseline_top1_margin_rows: tuple[float | None, ...]
    comparison_top1_margin_rows: tuple[float | None, ...]
    mean_js_divergence: float
    mean_spearman: float | None
    mean_top_n_jaccard: float
    undefined_spearman_layers: int

    def __post_init__(self) -> None:
        if self.schema_version != ROUTING_SIMILARITY_SCHEMA_VERSION:
            raise ValueError("schema_version is not the exact routing-similarity version")
        for name in (
            "baseline_run_key",
            "comparison_run_key",
            "model_key",
            "adapter_name",
            "adapter_version",
            "inspection_digest",
            "layout",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(f"{name} must be a non-empty exact string")
        if self.baseline_run_key == self.comparison_run_key:
            raise ValueError("comparison runs must differ")
        for name in ("baseline_token_count", "comparison_token_count", "top_n"):
            value = getattr(self, name)
            if type(value) is not int or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a strict positive integer")
        if type(self.layer_keys) is not tuple or not self.layer_keys:
            raise ValueError("layer_keys must be a non-empty tuple")
        if len(set(self.layer_keys)) != len(self.layer_keys) or any(
            type(key) is not str or not key for key in self.layer_keys
        ):
            raise ValueError("layer_keys must be unique non-empty strings")
        width = len(self.layer_keys)
        for name in (
            "js_divergence_rows",
            "spearman_rows",
            "top_n_jaccard_rows",
            "baseline_top1_margin_rows",
            "comparison_top1_margin_rows",
        ):
            rows = getattr(self, name)
            if type(rows) is not tuple or len(rows) != width:
                raise ValueError(f"{name} must align with layer_keys")
        object.__setattr__(
            self,
            "js_divergence_rows",
            tuple(_finite_unit(value, "js_divergence_rows") for value in self.js_divergence_rows),
        )
        object.__setattr__(
            self,
            "top_n_jaccard_rows",
            tuple(_finite_unit(value, "top_n_jaccard_rows") for value in self.top_n_jaccard_rows),
        )
        for name in ("baseline_top1_margin_rows", "comparison_top1_margin_rows"):
            object.__setattr__(
                self,
                name,
                tuple(
                    None if value is None else _finite_unit(value, name)
                    for value in getattr(self, name)
                ),
            )
        object.__setattr__(
            self,
            "spearman_rows",
            tuple(_finite_correlation(value, "spearman_rows") for value in self.spearman_rows),
        )
        object.__setattr__(
            self,
            "mean_js_divergence",
            _finite_unit(self.mean_js_divergence, "mean_js_divergence"),
        )
        object.__setattr__(
            self,
            "mean_top_n_jaccard",
            _finite_unit(self.mean_top_n_jaccard, "mean_top_n_jaccard"),
        )
        object.__setattr__(
            self,
            "mean_spearman",
            _finite_correlation(self.mean_spearman, "mean_spearman"),
        )
        if (
            type(self.undefined_spearman_layers) is not int
            or isinstance(self.undefined_spearman_layers, bool)
            or self.undefined_spearman_layers < 0
            or self.undefined_spearman_layers > width
            or self.undefined_spearman_layers
            != sum(value is None for value in self.spearman_rows)
        ):
            raise ValueError("undefined_spearman_layers does not match the rows")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": _ROUTING_SIMILARITY_ARTIFACT_TYPE,
            "schema_version": self.schema_version,
            "baseline_run_key": self.baseline_run_key,
            "comparison_run_key": self.comparison_run_key,
            "model_key": self.model_key,
            "adapter_name": self.adapter_name,
            "adapter_version": self.adapter_version,
            "inspection_digest": self.inspection_digest,
            "layout": self.layout,
            "baseline_token_count": self.baseline_token_count,
            "comparison_token_count": self.comparison_token_count,
            "top_n": self.top_n,
            "layer_keys": list(self.layer_keys),
            "js_divergence_rows": list(self.js_divergence_rows),
            "spearman_rows": list(self.spearman_rows),
            "top_n_jaccard_rows": list(self.top_n_jaccard_rows),
            "baseline_top1_margin_rows": list(self.baseline_top1_margin_rows),
            "comparison_top1_margin_rows": list(self.comparison_top1_margin_rows),
            "mean_js_divergence": self.mean_js_divergence,
            "mean_spearman": self.mean_spearman,
            "mean_top_n_jaccard": self.mean_top_n_jaccard,
            "undefined_spearman_layers": self.undefined_spearman_layers,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> RoutingSimilarity:
        try:
            document = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("routing similarity document is not valid JSON") from exc
        if type(document) is not dict:
            raise ValueError("routing similarity document must be a JSON object")
        if document.get("artifact_type") != _ROUTING_SIMILARITY_ARTIFACT_TYPE:
            raise ValueError("document is not a routing similarity artifact")
        try:
            return cls(
                schema_version=document["schema_version"],
                baseline_run_key=document["baseline_run_key"],
                comparison_run_key=document["comparison_run_key"],
                model_key=document["model_key"],
                adapter_name=document["adapter_name"],
                adapter_version=document["adapter_version"],
                inspection_digest=document["inspection_digest"],
                layout=document["layout"],
                baseline_token_count=document["baseline_token_count"],
                comparison_token_count=document["comparison_token_count"],
                top_n=document["top_n"],
                layer_keys=tuple(document["layer_keys"]),
                js_divergence_rows=tuple(document["js_divergence_rows"]),
                spearman_rows=tuple(document["spearman_rows"]),
                top_n_jaccard_rows=tuple(document["top_n_jaccard_rows"]),
                baseline_top1_margin_rows=tuple(document["baseline_top1_margin_rows"]),
                comparison_top1_margin_rows=tuple(document["comparison_top1_margin_rows"]),
                mean_js_divergence=document["mean_js_divergence"],
                mean_spearman=document["mean_spearman"],
                mean_top_n_jaccard=document["mean_top_n_jaccard"],
                undefined_spearman_layers=document["undefined_spearman_layers"],
            )
        except KeyError as exc:
            raise ValueError("routing similarity document is missing fields") from exc


def _entropy(distribution: tuple[float, ...]) -> float:
    return -sum(value * math.log2(value) for value in distribution if value > 0.0)


def _js_divergence(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    midpoint = tuple((a + b) / 2.0 for a, b in zip(left, right, strict=True))
    value = _entropy(midpoint) - (_entropy(left) + _entropy(right)) / 2.0
    return min(1.0, max(0.0, value))


def _average_ranks(values: tuple[float, ...]) -> tuple[float, ...]:
    ordered = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[cursor]]:
            end += 1
        average = ((cursor + 1) + end) / 2.0
        for position in range(cursor, end):
            ranks[ordered[position]] = average
        cursor = end
    return tuple(ranks)


def _spearman(left: tuple[float, ...], right: tuple[float, ...]) -> float | None:
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = sum(left_ranks) / len(left_ranks)
    right_mean = sum(right_ranks) / len(right_ranks)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left_ranks, right_ranks, strict=True)
    )
    left_scale = sum((value - left_mean) ** 2 for value in left_ranks)
    right_scale = sum((value - right_mean) ** 2 for value in right_ranks)
    if left_scale == 0.0 or right_scale == 0.0:
        return None
    return min(1.0, max(-1.0, numerator / math.sqrt(left_scale * right_scale)))


def _top_set(
    shares: tuple[float, ...], expert_keys: tuple[str, ...], top_n: int
) -> frozenset[str]:
    ordered = sorted(
        range(len(shares)), key=lambda index: (-shares[index], expert_keys[index])
    )
    return frozenset(expert_keys[index] for index in ordered[:top_n])


def _top1_margin(shares: tuple[float, ...]) -> float | None:
    if len(shares) < 2:
        return None
    ordered = sorted(shares, reverse=True)
    return ordered[0] - ordered[1]


def compare_routing_similarity(
    baseline: RoutingLoadMatrix,
    comparison: RoutingLoadMatrix,
    *,
    top_n: int = 5,
    max_cells: int = 100_000,
) -> RoutingSimilarity:
    """Compare normalized routing distributions over one exact topology."""

    if type(baseline) is not RoutingLoadMatrix or type(comparison) is not RoutingLoadMatrix:
        raise TypeError("baseline and comparison must be exact RoutingLoadMatrix values")
    if type(top_n) is not int or isinstance(top_n, bool) or top_n <= 0:
        raise ValueError("top_n must be a strict positive integer")
    if type(max_cells) is not int or isinstance(max_cells, bool) or max_cells <= 0:
        raise ValueError("max_cells must be a strict positive integer")
    baseline = RoutingLoadMatrix.from_json(baseline.to_json())
    comparison = RoutingLoadMatrix.from_json(comparison.to_json())
    if baseline.run_key == comparison.run_key:
        raise ValueError("comparison runs must differ")
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
        "layer_keys",
        "layer_indices",
        "expert_keys",
    ):
        if getattr(baseline, field_name) != getattr(comparison, field_name):
            raise ValueError(f"{field_name} differs between the two runs")
    expert_count = len(baseline.expert_keys[0])
    cells = len(baseline.layer_keys) * expert_count
    if cells > max_cells:
        raise ValueError("matrix cells exceed max_cells")
    effective_top_n = min(top_n, expert_count)

    js_rows = []
    spearman_rows = []
    jaccard_rows = []
    baseline_margins = []
    comparison_margins = []
    for index, (left, right) in enumerate(
        zip(baseline.assignment_shares, comparison.assignment_shares, strict=True)
    ):
        js_rows.append(_js_divergence(left, right))
        spearman_rows.append(_spearman(left, right))
        left_top = _top_set(left, baseline.expert_keys[index], effective_top_n)
        right_top = _top_set(right, comparison.expert_keys[index], effective_top_n)
        jaccard_rows.append(len(left_top & right_top) / len(left_top | right_top))
        baseline_margins.append(_top1_margin(left))
        comparison_margins.append(_top1_margin(right))

    defined_spearman = [value for value in spearman_rows if value is not None]
    return RoutingSimilarity(
        schema_version=ROUTING_SIMILARITY_SCHEMA_VERSION,
        baseline_run_key=baseline.run_key,
        comparison_run_key=comparison.run_key,
        model_key=baseline.model_key,
        adapter_name=baseline.adapter_name,
        adapter_version=baseline.adapter_version,
        inspection_digest=baseline.inspection_digest,
        layout=baseline.layout,
        baseline_token_count=baseline.token_count,
        comparison_token_count=comparison.token_count,
        top_n=effective_top_n,
        layer_keys=baseline.layer_keys,
        js_divergence_rows=tuple(js_rows),
        spearman_rows=tuple(spearman_rows),
        top_n_jaccard_rows=tuple(jaccard_rows),
        baseline_top1_margin_rows=tuple(baseline_margins),
        comparison_top1_margin_rows=tuple(comparison_margins),
        mean_js_divergence=sum(js_rows) / len(js_rows),
        mean_spearman=(
            sum(defined_spearman) / len(defined_spearman)
            if defined_spearman
            else None
        ),
        mean_top_n_jaccard=sum(jaccard_rows) / len(jaccard_rows),
        undefined_spearman_layers=len(spearman_rows) - len(defined_spearman),
    )


__all__ = [
    "ROUTING_SIMILARITY_SCHEMA_VERSION",
    "RoutingSimilarity",
    "compare_routing_similarity",
]
