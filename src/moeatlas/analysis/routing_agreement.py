"""Prompt-level vs rollout-level routing agreement (PRD §11.2).

Compares, per routed layer, the expert-selection distribution over prompt
tokens with the distribution over rollout (generated) tokens. The two phases
are supplied as count vectors over one shared per-layer expert universe; the
analysis derives base-2 Jensen-Shannon divergence, its bounded agreement
complement, and total-variation distance — all deterministic, budget-bounded,
and free of clocks, randomness, storage reads, and model knowledge.

Agreement is a similarity between routing *distributions*. Like every
association metric in this package it never implies specialization or
causality: identical prompt and rollout behavior is evidence of consistency,
not of why.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

ROUTING_AGREEMENT_SCHEMA_VERSION = "1.0"
"""Schema version of the routing-agreement contracts."""

_ROUTING_AGREEMENT_ARTIFACT_TYPE = "moeatlas.routing_agreement"

_ERROR_STAGES = frozenset({"contract", "budget"})


class RoutingAgreementError(RuntimeError):
    """Safe fixed-stage failure for routing-agreement handling."""

    def __init__(
        self, stage: str, message: str | None = None, *, cause: BaseException | None = None
    ) -> None:
        if stage not in _ERROR_STAGES:
            raise ValueError("routing agreement error stage is not supported")
        self.stage = stage
        text = f"routing agreement failed at {stage}"
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


def _strict_count_row(value: object, width: int, field_name: str) -> tuple[int, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{field_name} must be a tuple of integer counts")
    if len(value) != width:
        raise ValueError(f"{field_name} rows must align with the layer's experts")
    for entry in value:
        if type(entry) is not int or isinstance(entry, bool):
            raise TypeError(f"{field_name} counts must be integers")
        if entry < 0:
            raise ValueError(f"{field_name} counts must be non-negative")
    return value  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class PromptRolloutCounts:
    """Paired selection-count vectors over one shared per-layer universe.

    ``prompt_counts`` and ``rollout_counts`` are rectangular
    ``(layer, expert)`` tables over the same ``expert_keys`` topology; every
    layer needs a positive total in both phases so both conditional
    distributions are defined. Zero counts for individual experts are fine.
    """

    layer_keys: tuple[str, ...]
    expert_keys: tuple[tuple[str, ...], ...]
    prompt_counts: tuple[tuple[int, ...], ...]
    rollout_counts: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "layer_keys", _strict_key_tuple(self.layer_keys, "layer_keys")
        )
        if type(self.expert_keys) is not tuple:
            raise TypeError("expert_keys must be a tuple of per-layer key tuples")
        for row in self.expert_keys:
            _strict_key_tuple(row, "expert_keys")
        if len(self.expert_keys) != len(self.layer_keys):
            raise ValueError("expert_keys must hold exactly one row per layer")
        if type(self.prompt_counts) is not tuple or len(self.prompt_counts) != len(
            self.layer_keys
        ):
            raise ValueError("prompt_counts must hold exactly one row per layer")
        if type(self.rollout_counts) is not tuple or len(self.rollout_counts) != len(
            self.layer_keys
        ):
            raise ValueError("rollout_counts must hold exactly one row per layer")
        for index, layer in enumerate(self.layer_keys):
            width = len(self.expert_keys[index])
            prompt_row = _strict_count_row(
                self.prompt_counts[index], width, f"prompt_counts[{layer!r}]"
            )
            rollout_row = _strict_count_row(
                self.rollout_counts[index], width, f"rollout_counts[{layer!r}]"
            )
            if sum(prompt_row) <= 0:
                raise ValueError(
                    f"prompt_counts[{layer!r}] total must be strictly positive"
                )
            if sum(rollout_row) <= 0:
                raise ValueError(
                    f"rollout_counts[{layer!r}] total must be strictly positive"
                )

    @property
    def cell_count(self) -> int:
        return sum(len(row) for row in self.expert_keys)


@dataclass(frozen=True, slots=True)
class RoutingAgreement:
    """Per-layer agreement between prompt and rollout routing distributions.

    ``js_divergence_rows`` hold base-2 Jensen-Shannon divergence in ``[0, 1]``,
    ``agreement_rows`` its complement ``1 - JSD``, and ``tv_distance_rows``
    half the L1 distance between the distributions, also in ``[0, 1]``. All
    values are defined by construction: positive phase totals are a
    construction precondition, so no cell is ever null.
    """

    schema_version: str
    layer_keys: tuple[str, ...]
    expert_keys: tuple[tuple[str, ...], ...]
    js_divergence_rows: tuple[float, ...]
    agreement_rows: tuple[float, ...]
    tv_distance_rows: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": _ROUTING_AGREEMENT_ARTIFACT_TYPE,
            "schema_version": self.schema_version,
            "layer_keys": list(self.layer_keys),
            "expert_keys": [list(row) for row in self.expert_keys],
            "js_divergence_rows": list(self.js_divergence_rows),
            "agreement_rows": list(self.agreement_rows),
            "tv_distance_rows": list(self.tv_distance_rows),
        }

    def to_json(self) -> str:
        """Serialize this result with deterministic key order and no whitespace."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> RoutingAgreement:
        """Validate one canonical JSON document into an exact result value."""

        try:
            document = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("routing agreement document is not valid JSON") from exc
        if type(document) is not dict:
            raise ValueError("routing agreement document must be a JSON object")
        if (
            document.get("artifact_type") != _ROUTING_AGREEMENT_ARTIFACT_TYPE
            or document.get("schema_version") != ROUTING_AGREEMENT_SCHEMA_VERSION
        ):
            raise ValueError("document is not a routing agreement artifact")
        try:
            return cls(
                schema_version=document["schema_version"],
                layer_keys=tuple(document["layer_keys"]),
                expert_keys=tuple(tuple(row) for row in document["expert_keys"]),
                js_divergence_rows=tuple(document["js_divergence_rows"]),
                agreement_rows=tuple(document["agreement_rows"]),
                tv_distance_rows=tuple(document["tv_distance_rows"]),
            )
        except KeyError as exc:
            raise ValueError("routing agreement document is missing fields") from exc


def _entropy(distribution: tuple[float, ...]) -> float:
    total = 0.0
    for probability in distribution:
        if probability > 0.0:
            total -= probability * math.log2(probability)
    return total


def analyze_routing_agreement(
    counts: PromptRolloutCounts, *, max_cells: int = 1_000_000
) -> RoutingAgreement:
    """Derive per-layer prompt-vs-rollout agreement from paired counts."""

    if type(counts) is not PromptRolloutCounts:
        raise TypeError("counts must be a PromptRolloutCounts")
    if type(max_cells) is not int or isinstance(max_cells, bool):
        raise TypeError("max_cells must be an integer")
    if max_cells <= 0:
        raise RoutingAgreementError("budget", "max_cells must be strictly positive")
    if counts.cell_count > max_cells:
        raise RoutingAgreementError(
            "budget",
            f"count table has {counts.cell_count} cells; budget is {max_cells}",
        )
    js_rows: list[float] = []
    agreement_rows: list[float] = []
    tv_rows: list[float] = []
    for index, layer in enumerate(counts.layer_keys):
        prompt_total = sum(counts.prompt_counts[index])
        rollout_total = sum(counts.rollout_counts[index])
        p = tuple(value / prompt_total for value in counts.prompt_counts[index])
        q = tuple(value / rollout_total for value in counts.rollout_counts[index])
        midpoint = tuple((left + right) / 2.0 for left, right in zip(p, q))
        jsd = _entropy(midpoint) - (_entropy(p) + _entropy(q)) / 2.0
        # Clamp float noise so the documented [0, 1] bounds hold exactly.
        jsd = min(1.0, max(0.0, jsd))
        js_rows.append(jsd)
        agreement_rows.append(1.0 - jsd)
        tv_rows.append(
            min(1.0, max(0.0, 0.5 * sum(abs(left - right) for left, right in zip(p, q))))
        )
    return RoutingAgreement(
        schema_version=ROUTING_AGREEMENT_SCHEMA_VERSION,
        layer_keys=counts.layer_keys,
        expert_keys=counts.expert_keys,
        js_divergence_rows=tuple(js_rows),
        agreement_rows=tuple(agreement_rows),
        tv_distance_rows=tuple(tv_rows),
    )
