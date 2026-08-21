"""Cross-run stability of expert-task association (PRD §11.2).

Compares two runs' association evidence over one identical (layer, task,
expert) topology. For every (layer, task) cell the per-run conditional
routing distributions P(expert | task) are compared with base-2
Jensen-Shannon divergence; the analysis reports the divergence, its bounded
agreement complement, and the per-layer mean agreement — deterministic,
budget-bounded, and free of clocks, randomness, storage reads, and model
knowledge.

Stability is a property of association *distributions*. Two runs agreeing
here is evidence of reproducible routing behavior, never a claim of
specialization or causality.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from .task_association import TaskExpertCounts

ASSOCIATION_STABILITY_SCHEMA_VERSION = "1.0"
"""Schema version of the association-stability contracts."""

_ASSOCIATION_STABILITY_ARTIFACT_TYPE = "moeatlas.association_stability"

_ERROR_STAGES = frozenset({"contract", "budget"})


class AssociationStabilityError(RuntimeError):
    """Safe fixed-stage failure for association-stability handling."""

    def __init__(
        self, stage: str, message: str | None = None, *, cause: BaseException | None = None
    ) -> None:
        if stage not in _ERROR_STAGES:
            raise ValueError("association stability error stage is not supported")
        self.stage = stage
        text = f"association stability failed at {stage}"
        if message:
            text = f"{text}: {message}"
        super().__init__(text)
        if cause is not None:
            self.__cause__ = cause


@dataclass(frozen=True, slots=True)
class AssociationStability:
    """Per-(layer, task) agreement between two runs' association evidence.

    Rows are index-aligned with ``layer_keys`` and each row with
    ``task_keys``. ``js_divergence_rows`` hold base-2 Jensen-Shannon
    divergence between the runs' P(expert | task) distributions in ``[0, 1]``,
    ``agreement_rows`` its complement ``1 - JSD``, and
    ``mean_agreement_rows`` the per-layer mean agreement. Every cell is
    defined by construction: both inputs guarantee positive task totals.
    """

    schema_version: str
    layer_keys: tuple[str, ...]
    task_keys: tuple[str, ...]
    js_divergence_rows: tuple[tuple[float, ...], ...]
    agreement_rows: tuple[tuple[float, ...], ...]
    mean_agreement_rows: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": _ASSOCIATION_STABILITY_ARTIFACT_TYPE,
            "schema_version": self.schema_version,
            "layer_keys": list(self.layer_keys),
            "task_keys": list(self.task_keys),
            "js_divergence_rows": [list(row) for row in self.js_divergence_rows],
            "agreement_rows": [list(row) for row in self.agreement_rows],
            "mean_agreement_rows": list(self.mean_agreement_rows),
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
    def from_json(cls, payload: str | bytes | bytearray) -> AssociationStability:
        """Validate one canonical JSON document into an exact result value."""

        try:
            document = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("association stability document is not valid JSON") from exc
        if type(document) is not dict:
            raise ValueError("association stability document must be a JSON object")
        if (
            document.get("artifact_type") != _ASSOCIATION_STABILITY_ARTIFACT_TYPE
            or document.get("schema_version") != ASSOCIATION_STABILITY_SCHEMA_VERSION
        ):
            raise ValueError("document is not an association stability artifact")
        try:
            return cls(
                schema_version=document["schema_version"],
                layer_keys=tuple(document["layer_keys"]),
                task_keys=tuple(document["task_keys"]),
                js_divergence_rows=tuple(
                    tuple(row) for row in document["js_divergence_rows"]
                ),
                agreement_rows=tuple(tuple(row) for row in document["agreement_rows"]),
                mean_agreement_rows=tuple(document["mean_agreement_rows"]),
            )
        except KeyError as exc:
            raise ValueError("association stability document is missing fields") from exc


def _entropy(distribution: tuple[float, ...]) -> float:
    total = 0.0
    for probability in distribution:
        if probability > 0.0:
            total -= probability * math.log2(probability)
    return total


def _js_divergence(
    p: tuple[float, ...], q: tuple[float, ...]
) -> float:
    midpoint = tuple((left + right) / 2.0 for left, right in zip(p, q))
    divergence = _entropy(midpoint) - (_entropy(p) + _entropy(q)) / 2.0
    # Clamp float noise so the documented [0, 1] bounds hold exactly.
    return min(1.0, max(0.0, divergence))


def analyze_association_stability(
    counts_a: TaskExpertCounts,
    counts_b: TaskExpertCounts,
    *,
    max_cells: int = 1_000_000,
) -> AssociationStability:
    """Compare two runs' association tables over one identical topology."""

    for name, value in (("counts_a", counts_a), ("counts_b", counts_b)):
        if type(value) is not TaskExpertCounts:
            raise TypeError(f"{name} must be a TaskExpertCounts")
    if type(max_cells) is not int or isinstance(max_cells, bool):
        raise TypeError("max_cells must be an integer")
    if max_cells <= 0:
        raise AssociationStabilityError(
            "budget", "max_cells must be strictly positive"
        )
    if (
        counts_a.layer_keys != counts_b.layer_keys
        or counts_a.task_keys != counts_b.task_keys
        or counts_a.expert_keys != counts_b.expert_keys
    ):
        raise AssociationStabilityError(
            "contract",
            "both count tables must share one identical (layer, task, expert)"
            " topology",
        )
    if counts_a.cell_count > max_cells:
        raise AssociationStabilityError(
            "budget",
            f"count tables have {counts_a.cell_count} cells; budget is {max_cells}",
        )
    js_rows: list[tuple[float, ...]] = []
    agreement_rows: list[tuple[float, ...]] = []
    mean_rows: list[float] = []
    for layer_index, table_a in enumerate(counts_a.counts):
        table_b = counts_b.counts[layer_index]
        layer_js: list[float] = []
        layer_agreement: list[float] = []
        for task_index, row_a in enumerate(table_a):
            total_a = sum(row_a)
            total_b = sum(table_b[task_index])
            p = tuple(value / total_a for value in row_a)
            q = tuple(value / total_b for value in table_b[task_index])
            divergence = _js_divergence(p, q)
            layer_js.append(divergence)
            layer_agreement.append(1.0 - divergence)
        js_rows.append(tuple(layer_js))
        agreement_rows.append(tuple(layer_agreement))
        mean_rows.append(sum(layer_agreement) / len(layer_agreement))
    return AssociationStability(
        schema_version=ASSOCIATION_STABILITY_SCHEMA_VERSION,
        layer_keys=counts_a.layer_keys,
        task_keys=counts_a.task_keys,
        js_divergence_rows=tuple(js_rows),
        agreement_rows=tuple(agreement_rows),
        mean_agreement_rows=tuple(mean_rows),
    )
