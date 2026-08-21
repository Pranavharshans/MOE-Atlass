"""Expert co-routing graph summaries (PRD §11.3).

Co-routing records how often pairs of experts at one layer were selected
together — the model-neutral core of the PRD's co-activation and conditional
co-routing graphs. Callers supply, per layer, a symmetric pair-count matrix
over that layer's experts (zero diagonal); the analysis derives total
co-selection mass, how many experts are actually coupled, and a
deterministically ranked top-pair list with normalized shares.

The layer stays pure: no storage reads, clocks, randomness, or model
knowledge. Co-routing is association evidence only — it never implies
specialization or causality.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

COROUTING_SCHEMA_VERSION = "1.0"
"""Schema version of the co-routing contracts."""

_COROUTING_ARTIFACT_TYPE = "moeatlas.corouting"

_ERROR_STAGES = frozenset({"contract", "budget"})


class CoRoutingError(RuntimeError):
    """Safe fixed-stage failure for co-routing handling."""

    def __init__(
        self, stage: str, message: str | None = None, *, cause: BaseException | None = None
    ) -> None:
        if stage not in _ERROR_STAGES:
            raise ValueError("co-routing error stage is not supported")
        self.stage = stage
        text = f"co-routing failed at {stage}"
        if message:
            text = f"{text}: {message}"
        super().__init__(text)
        if cause is not None:
            self.__cause__ = cause


def _strict_key_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{field_name} must be a tuple of strings")
    for entry in value:
        if type(entry) is not str:
            raise TypeError(f"{field_name} entries must be strings")
    keys = tuple(value)
    if not keys or list(keys) != sorted(set(keys)):
        raise ValueError(f"{field_name} must be non-empty, unique, and sorted")
    return keys


@dataclass(frozen=True, slots=True)
class ExpertCoRoutingCounts:
    """Per-layer symmetric co-selection counts over that layer's experts.

    ``pair_counts[layer][i][j]`` counts the steps where experts i and j of
    the layer were selected together. The matrix must be square over
    ``expert_keys[layer]``, symmetric, non-negative, and zero-diagonal (an
    expert is never paired with itself).
    """

    layer_keys: tuple[str, ...]
    expert_keys: tuple[tuple[str, ...], ...]
    pair_counts: tuple[tuple[tuple[int, ...], ...], ...]

    def __post_init__(self) -> None:
        _strict_key_tuple(self.layer_keys, "layer_keys")
        if type(self.expert_keys) is not tuple:
            raise TypeError("expert_keys must be a tuple of per-layer key tuples")
        for row in self.expert_keys:
            _strict_key_tuple(row, "expert_keys")
        if len(self.expert_keys) != len(self.layer_keys):
            raise ValueError("expert_keys must hold exactly one row per layer")
        if type(self.pair_counts) is not tuple:
            raise TypeError("pair_counts must be a tuple of per-layer tables")
        if len(self.pair_counts) != len(self.layer_keys):
            raise ValueError("pair_counts must hold exactly one table per layer")
        for index, layer in enumerate(self.layer_keys):
            keys = self.expert_keys[index]
            table = self.pair_counts[index]
            field_name = f"pair_counts[{layer!r}]"
            if type(table) is not tuple:
                raise TypeError(f"{field_name} must be a tuple of count rows")
            if len(table) != len(keys):
                raise ValueError(f"{field_name} must be square over its expert keys")
            for row in table:
                if type(row) is not tuple:
                    raise TypeError(f"{field_name} rows must be tuples of counts")
                if len(row) != len(keys):
                    raise ValueError(
                        f"{field_name} rows must match the layer's expert keys exactly"
                    )
                for value in row:
                    if type(value) is not int or isinstance(value, bool):
                        raise TypeError(f"{field_name} counts must be integers")
                    if value < 0:
                        raise ValueError(f"{field_name} counts must be non-negative")
            for i in range(len(keys)):
                if table[i][i] != 0:
                    raise ValueError(f"{field_name} diagonal must stay zero")
                for j in range(i + 1, len(keys)):
                    if table[i][j] != table[j][i]:
                        raise ValueError(f"{field_name} must be symmetric")

    @property
    def cell_count(self) -> int:
        return sum(len(row) ** 2 for row in self.expert_keys)


@dataclass(frozen=True, slots=True)
class CoRoutingGraph:
    """Per-layer co-routing summaries with deterministically ranked pairs.

    ``top_pairs[layer]`` holds ``(expert_a, expert_b, count, share)`` tuples
    sorted by descending count then ascending keys, bounded by ``max_pairs``;
    ``share`` normalizes each pair's count by the layer's total
    co-selection mass. ``total_pair_selections`` hold that mass per layer,
    and ``coupled_expert_rows`` count experts appearing in at least one
    co-selection.
    """

    schema_version: str
    layer_keys: tuple[str, ...]
    top_pairs: tuple[tuple[tuple[str, str, int, float], ...], ...]
    total_pair_selections: tuple[int, ...]
    coupled_expert_rows: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": _COROUTING_ARTIFACT_TYPE,
            "schema_version": self.schema_version,
            "layer_keys": list(self.layer_keys),
            "top_pairs": [
                [list(pair) for pair in layer_pairs]
                for layer_pairs in self.top_pairs
            ],
            "total_pair_selections": list(self.total_pair_selections),
            "coupled_expert_rows": list(self.coupled_expert_rows),
        }

    def to_json(self) -> str:
        """Serialize this graph with deterministic key order and no whitespace."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> CoRoutingGraph:
        """Validate one canonical JSON document into an exact graph value."""

        try:
            document = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("co-routing document is not valid JSON") from exc
        if type(document) is not dict:
            raise ValueError("co-routing document must be a JSON object")
        if (
            document.get("artifact_type") != _COROUTING_ARTIFACT_TYPE
            or document.get("schema_version") != COROUTING_SCHEMA_VERSION
        ):
            raise ValueError("document is not a co-routing artifact")
        try:
            return cls(
                schema_version=document["schema_version"],
                layer_keys=tuple(document["layer_keys"]),
                top_pairs=tuple(
                    tuple((pair[0], pair[1], pair[2], pair[3]) for pair in layer_pairs)
                    for layer_pairs in document["top_pairs"]
                ),
                total_pair_selections=tuple(document["total_pair_selections"]),
                coupled_expert_rows=tuple(document["coupled_expert_rows"]),
            )
        except KeyError as exc:
            raise ValueError("co-routing document is missing fields") from exc


def summarize_co_routing(
    counts: ExpertCoRoutingCounts,
    *,
    max_cells: int = 1_000_000,
    max_pairs: int = 100,
) -> CoRoutingGraph:
    """Derive per-layer co-routing summaries from symmetric pair counts."""

    if type(counts) is not ExpertCoRoutingCounts:
        raise TypeError("counts must be an ExpertCoRoutingCounts")
    for name, value in (("max_cells", max_cells), ("max_pairs", max_pairs)):
        if type(value) is not int or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
    if max_cells <= 0:
        raise CoRoutingError("budget", "max_cells must be strictly positive")
    if max_pairs <= 0:
        raise CoRoutingError("budget", "max_pairs must be strictly positive")
    if counts.cell_count > max_cells:
        raise CoRoutingError(
            "budget",
            f"count tables have {counts.cell_count} cells; budget is {max_cells}",
        )
    totals: list[int] = []
    coupled_rows: list[int] = []
    top_pair_rows: list[tuple[tuple[str, str, int, float], ...]] = []
    for index, layer in enumerate(counts.layer_keys):
        keys = counts.expert_keys[index]
        table = counts.pair_counts[index]
        pairs: list[tuple[str, str, int]] = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                if table[i][j] > 0:
                    pairs.append((keys[i], keys[j], table[i][j]))
        total = sum(count for _, _, count in pairs)
        totals.append(total)
        coupled = {expert for pair in pairs for expert in pair[:2]}
        coupled_rows.append(len(coupled))
        pairs.sort(key=lambda pair: (-pair[2], pair[0], pair[1]))
        top_pair_rows.append(
            tuple(
                (expert_a, expert_b, count, count / total)
                for expert_a, expert_b, count in pairs[:max_pairs]
            )
        )
    return CoRoutingGraph(
        schema_version=COROUTING_SCHEMA_VERSION,
        layer_keys=counts.layer_keys,
        top_pairs=tuple(top_pair_rows),
        total_pair_selections=tuple(totals),
        coupled_expert_rows=tuple(coupled_rows),
    )
