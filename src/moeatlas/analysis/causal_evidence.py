"""Causal effect summaries over paired observations (PRD §11.4).

Reduces caller-supplied baseline/intervention metric pairs — one pair per
metric label and replication index — into per-label effect summaries with
explicit stability evidence: mean baseline/intervened values, absolute and
relative effects, direction consistency across replications, and a strict
``stable`` marker that is true only when every replication reproduces the
same nonzero direction.

The layer is pure and deterministic: no storage reads, clocks, randomness,
or model knowledge. An effect summary describes paired observations only —
it never by itself proves specialization, and real-model causal claims stay
deferred to the validation ledger. ``null`` marks undefined values (zero
baseline means, undefined effect direction) as evidence, never inferred.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

CAUSAL_EVIDENCE_SCHEMA_VERSION = "1.0"
"""Schema version of the causal-evidence contracts."""

_CAUSAL_EVIDENCE_ARTIFACT_TYPE = "moeatlas.causal_evidence"

_ERROR_STAGES = frozenset({"contract", "budget"})

_MAX_LABEL = 200


class CausalEvidenceError(RuntimeError):
    """Safe fixed-stage failure for causal-evidence handling."""

    def __init__(
        self, stage: str, message: str | None = None, *, cause: BaseException | None = None
    ) -> None:
        if stage not in _ERROR_STAGES:
            raise ValueError("causal evidence error stage is not supported")
        self.stage = stage
        text = f"causal evidence failed at {stage}"
        if message:
            text = f"{text}: {message}"
        super().__init__(text)
        if cause is not None:
            self.__cause__ = cause


def _strict_label(value: object) -> str:
    if type(value) is not str:
        raise TypeError("labels must be strings")
    if not value:
        raise ValueError("labels must not be empty")
    if len(value) > _MAX_LABEL:
        raise ValueError(f"labels must hold at most {_MAX_LABEL} characters")
    return value


def _strict_replication(value: object) -> int:
    if type(value) is not int or isinstance(value, bool):
        raise TypeError("replication must be an integer")
    if value < 0:
        raise ValueError("replication must be non-negative")
    return value


def _strict_value(value: object, field_name: str) -> float:
    if type(value) is not float and type(value) is not int:
        raise TypeError(f"{field_name} must be a number")
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"{field_name} must be finite")
    return float(value)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class CausalPair:
    """One baseline/intervention metric pair from one replication.

    ``label`` is a caller-owned metric identity (for example an expert's
    output mass or a task score); the analysis never interprets it beyond
    grouping. ``replication`` indexes repeated executions of the same
    comparison.
    """

    label: str
    replication: int
    baseline_value: float
    intervened_value: float

    def __post_init__(self) -> None:
        _strict_label(self.label)
        _strict_replication(self.replication)
        _strict_value(self.baseline_value, "baseline_value")
        _strict_value(self.intervened_value, "intervened_value")


@dataclass(frozen=True, slots=True)
class CausalEvidence:
    """Per-label paired-effect summaries with explicit stability evidence.

    ``labels`` is sorted and unique. Per label (same tuple positions):
    ``mean_baseline``/``mean_intervened`` are replication means,
    ``absolute_effects`` is intervened minus baseline, ``relative_effects``
    divides by ``abs(mean_baseline)`` and is ``null`` where the baseline
    mean is zero, ``direction_consistency`` is the share of replication
    effects matching the mean-effect sign and is ``null`` where the mean
    effect is exactly zero, ``stable_labels`` is true only when every
    replication effect shares one nonzero sign, and ``zero_effect_labels``
    marks labels whose every replication effect is exactly zero.
    """

    schema_version: str
    labels: tuple[str, ...]
    replication_counts: tuple[int, ...]
    mean_baseline: tuple[float, ...]
    mean_intervened: tuple[float, ...]
    absolute_effects: tuple[float, ...]
    relative_effects: tuple[float | None, ...]
    direction_consistency: tuple[float | None, ...]
    stable_labels: tuple[bool, ...]
    zero_effect_labels: tuple[bool, ...]

    def __post_init__(self) -> None:
        if type(self.labels) is not tuple:
            raise TypeError("labels must be a tuple")
        widths = (
            len(self.replication_counts),
            len(self.mean_baseline),
            len(self.mean_intervened),
            len(self.absolute_effects),
            len(self.relative_effects),
            len(self.direction_consistency),
            len(self.stable_labels),
            len(self.zero_effect_labels),
        )
        if any(width != len(self.labels) for width in widths):
            raise ValueError("every summary row must hold exactly one entry per label")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": _CAUSAL_EVIDENCE_ARTIFACT_TYPE,
            "schema_version": self.schema_version,
            "labels": list(self.labels),
            "replication_counts": list(self.replication_counts),
            "mean_baseline": list(self.mean_baseline),
            "mean_intervened": list(self.mean_intervened),
            "absolute_effects": list(self.absolute_effects),
            "relative_effects": list(self.relative_effects),
            "direction_consistency": list(self.direction_consistency),
            "stable_labels": list(self.stable_labels),
            "zero_effect_labels": list(self.zero_effect_labels),
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
    def from_json(cls, payload: str | bytes | bytearray) -> CausalEvidence:
        """Validate one canonical JSON document into an exact result value."""

        try:
            document = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("causal evidence document is not valid JSON") from exc
        if type(document) is not dict:
            raise ValueError("causal evidence document must be a JSON object")
        if (
            document.get("artifact_type") != _CAUSAL_EVIDENCE_ARTIFACT_TYPE
            or document.get("schema_version") != CAUSAL_EVIDENCE_SCHEMA_VERSION
        ):
            raise ValueError("document is not a causal evidence artifact")
        try:
            return cls(
                schema_version=document["schema_version"],
                labels=tuple(document["labels"]),
                replication_counts=tuple(document["replication_counts"]),
                mean_baseline=tuple(document["mean_baseline"]),
                mean_intervened=tuple(document["mean_intervened"]),
                absolute_effects=tuple(document["absolute_effects"]),
                relative_effects=tuple(document["relative_effects"]),
                direction_consistency=tuple(document["direction_consistency"]),
                stable_labels=tuple(document["stable_labels"]),
                zero_effect_labels=tuple(document["zero_effect_labels"]),
            )
        except KeyError as exc:
            raise ValueError("causal evidence document is missing fields") from exc


def analyze_causal_evidence(
    pairs: tuple[CausalPair, ...], *, max_pairs: int = 100_000
) -> CausalEvidence:
    """Summarize paired baseline/intervention effects per metric label."""

    if type(pairs) is not tuple:
        raise TypeError("pairs must be a tuple of CausalPair entries")
    for entry in pairs:
        if type(entry) is not CausalPair:
            raise TypeError("pairs entries must be CausalPair values")
    if type(max_pairs) is not int or isinstance(max_pairs, bool):
        raise TypeError("max_pairs must be an integer")
    if max_pairs <= 0:
        raise CausalEvidenceError("budget", "max_pairs must be strictly positive")
    if not pairs:
        raise CausalEvidenceError("contract", "pairs must not be empty")
    if len(pairs) > max_pairs:
        raise CausalEvidenceError(
            "budget", f"pairs hold {len(pairs)} entries; budget is {max_pairs}"
        )
    grouped: dict[str, dict[int, tuple[float, float]]] = {}
    for pair in pairs:
        replications = grouped.setdefault(pair.label, {})
        if pair.replication in replications:
            raise CausalEvidenceError(
                "contract",
                f"label {pair.label!r} repeats replication {pair.replication}",
            )
        replications[pair.replication] = (pair.baseline_value, pair.intervened_value)
    labels = tuple(sorted(grouped))
    replication_counts: list[int] = []
    mean_baseline_row: list[float] = []
    mean_intervened_row: list[float] = []
    absolute_row: list[float] = []
    relative_row: list[float | None] = []
    direction_row: list[float | None] = []
    stable_row: list[bool] = []
    zero_row: list[bool] = []
    for label in labels:
        replications = grouped[label]
        order = sorted(replications)
        baselines = [replications[index][0] for index in order]
        intervened = [replications[index][1] for index in order]
        effects = [after - before for before, after in zip(baselines, intervened)]
        count = len(order)
        mean_before = sum(baselines) / count
        mean_after = sum(intervened) / count
        mean_effect = sum(effects) / count
        replication_counts.append(count)
        mean_baseline_row.append(mean_before)
        mean_intervened_row.append(mean_after)
        absolute_row.append(mean_effect)
        relative_row.append(
            mean_effect / abs(mean_before) if mean_before != 0.0 else None
        )
        if mean_effect > 0.0:
            direction_row.append(sum(1 for effect in effects if effect > 0.0) / count)
            stable_row.append(all(effect > 0.0 for effect in effects))
        elif mean_effect < 0.0:
            direction_row.append(sum(1 for effect in effects if effect < 0.0) / count)
            stable_row.append(all(effect < 0.0 for effect in effects))
        else:
            direction_row.append(None)
            stable_row.append(False)
        zero_row.append(all(effect == 0.0 for effect in effects))
    return CausalEvidence(
        schema_version=CAUSAL_EVIDENCE_SCHEMA_VERSION,
        labels=labels,
        replication_counts=tuple(replication_counts),
        mean_baseline=tuple(mean_baseline_row),
        mean_intervened=tuple(mean_intervened_row),
        absolute_effects=tuple(absolute_row),
        relative_effects=tuple(relative_row),
        direction_consistency=tuple(direction_row),
        stable_labels=tuple(stable_row),
        zero_effect_labels=tuple(zero_row),
    )
