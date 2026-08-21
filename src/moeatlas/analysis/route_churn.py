"""Route-churn routing-stability summaries (PRD §11.1).

Route churn measures how expert selection changes across adjacent steps —
adjacent generated tokens, prompt perturbations, or any caller-defined
ordering of routed steps (token keys are content digests, so adjacency is a
caller-supplied sequence, never inferred). For every adjacent pair the
analysis records whether the selected-expert set changed and its Jaccard
distance ``1 - |A ∩ B| / |A ∪ B|``, with the documented conventions that an
empty-to-empty pair is no change and empty-to-nonempty is full change.

Layers with fewer than two steps have undefined churn — reported as ``null``,
never inferred. The layer stays pure: no storage reads, clocks, randomness,
or model knowledge, and churn explains routing stability only — never
specialization or causality.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

ROUTE_CHURN_SCHEMA_VERSION = "1.0"
"""Schema version of the route-churn contracts."""

_ROUTE_CHURN_ARTIFACT_TYPE = "moeatlas.route_churn"

_ERROR_STAGES = frozenset({"contract", "budget"})


class RouteChurnError(RuntimeError):
    """Safe fixed-stage failure for route-churn handling."""

    def __init__(
        self, stage: str, message: str | None = None, *, cause: BaseException | None = None
    ) -> None:
        if stage not in _ERROR_STAGES:
            raise ValueError("route churn error stage is not supported")
        self.stage = stage
        text = f"route churn failed at {stage}"
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


def _strict_step_row(value: object, field_name: str) -> tuple[tuple[str, ...], ...]:
    if type(value) is not tuple:
        raise TypeError(f"{field_name} must be a tuple of step tuples")
    for step in value:
        if type(step) is not tuple:
            raise TypeError(f"{field_name} entries must be tuples of expert keys")
        for expert in step:
            if type(expert) is not str:
                raise TypeError(f"{field_name} expert keys must be strings")
        if len(set(step)) != len(step):
            raise ValueError(f"{field_name} steps must not repeat expert keys")
    return value  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class RouteChurnSequences:
    """Per-layer ordered selected-expert sets, one entry per routed step.

    ``step_experts[layer]`` holds one inner tuple per step in caller-defined
    adjacency order; order inside a step is irrelevant (each step is a set).
    Empty steps are legal and mean "no experts selected at this step" — they
    follow the documented empty-set conventions instead of being skipped.
    """

    layer_keys: tuple[str, ...]
    step_experts: tuple[tuple[tuple[str, ...], ...], ...]

    def __post_init__(self) -> None:
        _strict_key_tuple(self.layer_keys, "layer_keys")
        if type(self.step_experts) is not tuple:
            raise TypeError("step_experts must be a tuple of per-layer rows")
        if len(self.step_experts) != len(self.layer_keys):
            raise ValueError("step_experts must hold exactly one row per layer")
        for index, layer in enumerate(self.layer_keys):
            _strict_step_row(self.step_experts[index], f"step_experts[{layer!r}]")

    @property
    def step_count(self) -> int:
        return sum(len(row) for row in self.step_experts)


@dataclass(frozen=True, slots=True)
class RouteChurnSummary:
    """Per-layer churn over adjacent steps with explicit pair counts.

    ``churn_rate_rows`` hold the fraction of adjacent pairs whose selected
    sets differ; ``mean_jaccard_rows`` hold the mean Jaccard distance over
    those pairs; ``pair_rows`` count the adjacent pairs. Layers with fewer
    than two steps report ``null`` rates and distances with zero pairs.
    """

    schema_version: str
    layer_keys: tuple[str, ...]
    churn_rate_rows: tuple[float | None, ...]
    mean_jaccard_rows: tuple[float | None, ...]
    pair_rows: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": _ROUTE_CHURN_ARTIFACT_TYPE,
            "schema_version": self.schema_version,
            "layer_keys": list(self.layer_keys),
            "churn_rate_rows": list(self.churn_rate_rows),
            "mean_jaccard_rows": list(self.mean_jaccard_rows),
            "pair_rows": list(self.pair_rows),
        }

    def to_json(self) -> str:
        """Serialize this summary with deterministic key order and no whitespace."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> RouteChurnSummary:
        """Validate one canonical JSON document into an exact summary value."""

        try:
            document = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("route churn document is not valid JSON") from exc
        if type(document) is not dict:
            raise ValueError("route churn document must be a JSON object")
        if (
            document.get("artifact_type") != _ROUTE_CHURN_ARTIFACT_TYPE
            or document.get("schema_version") != ROUTE_CHURN_SCHEMA_VERSION
        ):
            raise ValueError("document is not a route churn artifact")
        try:
            return cls(
                schema_version=document["schema_version"],
                layer_keys=tuple(document["layer_keys"]),
                churn_rate_rows=tuple(document["churn_rate_rows"]),
                mean_jaccard_rows=tuple(document["mean_jaccard_rows"]),
                pair_rows=tuple(document["pair_rows"]),
            )
        except KeyError as exc:
            raise ValueError("route churn document is missing fields") from exc


def _jaccard_distance(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    if not union:
        return 0.0  # empty -> empty: no change
    return 1.0 - len(left & right) / len(union)


def analyze_route_churn(
    sequences: RouteChurnSequences, *, max_steps: int = 1_000_000
) -> RouteChurnSummary:
    """Derive per-layer route-churn summaries from ordered step sequences."""

    if type(sequences) is not RouteChurnSequences:
        raise TypeError("sequences must be a RouteChurnSequences")
    if type(max_steps) is not int or isinstance(max_steps, bool):
        raise TypeError("max_steps must be an integer")
    if max_steps <= 0:
        raise RouteChurnError("budget", "max_steps must be strictly positive")
    if sequences.step_count > max_steps:
        raise RouteChurnError(
            "budget",
            f"sequence table has {sequences.step_count} steps; budget is {max_steps}",
        )
    churn_rates: list[float | None] = []
    jaccard_means: list[float | None] = []
    pair_counts: list[int] = []
    for index, layer in enumerate(sequences.layer_keys):
        steps = [frozenset(step) for step in sequences.step_experts[index]]
        pairs = len(steps) - 1
        pair_counts.append(pairs)
        if pairs <= 0:
            churn_rates.append(None)
            jaccard_means.append(None)
            continue
        distances = [
            _jaccard_distance(steps[position], steps[position + 1])
            for position in range(pairs)
        ]
        changed = sum(1 for distance in distances if distance > 0.0)
        churn_rates.append(changed / pairs)
        jaccard_means.append(sum(distances) / pairs)
    return RouteChurnSummary(
        schema_version=ROUTE_CHURN_SCHEMA_VERSION,
        layer_keys=sequences.layer_keys,
        churn_rate_rows=tuple(churn_rates),
        mean_jaccard_rows=tuple(jaccard_means),
        pair_rows=tuple(pair_counts),
    )
