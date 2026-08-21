"""Task-association analyses over layered expert assignment count tables.

This module implements the model-neutral core of PRD §11.2: enrichment
(``P(expert | task) / P(expert)``), pointwise and mutual information between
expert selection and task labels, per-layer task separability from routing
distributions, and expert exclusivity versus generality across tasks. The
input is a strict frozen contingency table of selected-route counts per
(layer, task, expert); the output is a frozen, canonically serializable
matrix whose every metric documents its denominators and its undefined-cell
policy.

Association is not specialization or causality: these numbers say a task
routes somewhere more or less often than baseline — nothing more. Absence is
explicit evidence: cells with an unusable denominator (an expert never
selected anywhere in the layer) serialize as ``null``, never ``NaN``.
Per-token task-labeled evidence arrives with task-labeled executors in later
sequences; until then this contract is exercised over synthetic tables, and
real-checkpoint behavior remains deferred MV evidence.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

TASK_ASSOCIATION_SCHEMA_VERSION = "1.0"
"""Schema version of the task-association contracts."""

_TASK_ASSOCIATION_ARTIFACT_TYPE = "moeatlas.task_association"

_ERROR_STAGES = frozenset({"contract", "budget"})


class TaskAssociationError(RuntimeError):
    """Safe fixed-stage failure for task-association analysis."""

    def __init__(
        self, stage: str, message: str | None = None, *, cause: BaseException | None = None
    ) -> None:
        if stage not in _ERROR_STAGES:
            raise ValueError("task association error stage is not supported")
        self.stage = stage
        text = f"task association failed at {stage}"
        if message:
            text = f"{text}: {message}"
        super().__init__(text)
        if cause is not None:
            self.__cause__ = cause


def _strict_key_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{field_name} must be a tuple of strings")
    for key in value:
        if type(key) is not str or not key:
            raise TypeError(f"{field_name} entries must be non-empty strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{field_name} entries must be unique")
    if list(value) != sorted(value):
        raise ValueError(f"{field_name} must be sorted")
    return value  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class TaskExpertCounts:
    """Selected-route counts per (layer, task, expert), fully validated.

    Every task must carry evidence (a strictly positive total) so that all
    conditional distributions are defined; experts without any selection stay
    legal and surface as explicit undefined cells downstream.
    """

    layer_keys: tuple[str, ...]
    task_keys: tuple[str, ...]
    expert_keys: tuple[tuple[str, ...], ...]
    counts: tuple[tuple[tuple[int, ...], ...], ...]

    def __post_init__(self) -> None:
        _strict_key_tuple(self.layer_keys, "layer_keys")
        _strict_key_tuple(self.task_keys, "task_keys")
        if not self.layer_keys or not self.task_keys:
            raise ValueError("layer_keys and task_keys must be non-empty")
        if type(self.expert_keys) is not tuple or len(self.expert_keys) != len(
            self.layer_keys
        ):
            raise TypeError("expert_keys must be one tuple per layer")
        for index, row in enumerate(self.expert_keys):
            try:
                _strict_key_tuple(row, f"expert_keys[{index}]")
            except TypeError as exc:
                raise TypeError(str(exc)) from exc
            except ValueError as exc:
                raise ValueError(
                    f"expert_keys[{index}] must be sorted and unique"
                ) from exc
            if not row:
                raise ValueError(f"expert_keys[{index}] must be non-empty")
        if type(self.counts) is not tuple:
            raise TypeError("counts must be a tuple of per-layer tables")
        if len(self.counts) != len(self.layer_keys):
            raise ValueError("counts must contain one table per layer")
        for layer_index, table in enumerate(self.counts):
            if type(table) is not tuple:
                raise TypeError("counts tables must be tuples of task rows")
            if len(table) != len(self.task_keys):
                raise ValueError("counts must contain one row per task")
            for task_index, row in enumerate(table):
                expected = len(self.expert_keys[layer_index])
                if type(row) is not tuple:
                    raise TypeError("counts rows must be tuples of integers")
                if len(row) != expected:
                    raise ValueError(
                        "counts rows must match the layer's expert keys exactly"
                    )
                for value in row:
                    if type(value) is not int or isinstance(value, bool):
                        raise TypeError("counts must be integers")
                    if value < 0:
                        raise ValueError("counts must be non-negative integers")
            totals = [sum(row) for row in table]
            if any(total == 0 for total in totals):
                raise ValueError("every task must have at least one assignment")

    @property
    def cell_count(self) -> int:
        return sum(
            len(self.task_keys) * len(experts) for experts in self.expert_keys
        )


@dataclass(frozen=True, slots=True)
class TaskAssociationMatrix:
    """Deterministic association metrics over one validated count table.

    Rows are canonically ordered. ``enrichment_rows`` hold
    ``P(expert | task) / P(expert)``; ``pmi_rows`` hold ``log2`` of the same
    ratio. A cell is ``None`` exactly when ``P(expert) == 0`` (the expert is
    unused in the layer); PMI is additionally ``None`` when the conditional
    probability is zero, since the pointwise information would be negative
    infinity. ``mutual_information_rows`` hold per-layer MI(task; expert) in
    bits; ``specific_mi_rows`` hold each task's contribution-weightable
    ``sum_e p(e|t) log2(p(e|t)/p(e))``; ``separability_rows`` hold the mean
    pairwise base-2 Jensen-Shannon divergence between the tasks' routing
    distributions (``None`` with fewer than two tasks);
    ``exclusivity_rows`` hold ``max_t P(task | expert)`` plus the number of
    tasks the expert actually receives (``None`` exclusivity when unused).
    """

    schema_version: str
    layer_keys: tuple[str, ...]
    task_keys: tuple[str, ...]
    expert_keys: tuple[tuple[str, ...], ...]
    assignment_totals: tuple[tuple[int, ...], ...]
    enrichment_rows: tuple[tuple[str, str, str, float | None], ...]
    pmi_rows: tuple[tuple[str, str, str, float | None], ...]
    mutual_information_rows: tuple[tuple[str, float], ...]
    specific_mi_rows: tuple[tuple[str, str, float], ...]
    separability_rows: tuple[tuple[str, float | None], ...]
    exclusivity_rows: tuple[tuple[str, str, float | None, int], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible data without runtime objects."""

        return {
            "artifact_type": _TASK_ASSOCIATION_ARTIFACT_TYPE,
            "schema_version": self.schema_version,
            "layer_keys": list(self.layer_keys),
            "task_keys": list(self.task_keys),
            "expert_keys": [list(row) for row in self.expert_keys],
            "assignment_totals": [list(row) for row in self.assignment_totals],
            "enrichment_rows": [list(row) for row in self.enrichment_rows],
            "pmi_rows": [list(row) for row in self.pmi_rows],
            "mutual_information_rows": [
                list(row) for row in self.mutual_information_rows
            ],
            "specific_mi_rows": [list(row) for row in self.specific_mi_rows],
            "separability_rows": [list(row) for row in self.separability_rows],
            "exclusivity_rows": [list(row) for row in self.exclusivity_rows],
        }

    def to_json(self) -> str:
        """Serialize this matrix with deterministic key order and no whitespace."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> TaskAssociationMatrix:
        """Validate one canonical JSON document into an exact matrix value."""

        try:
            document = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("task association document is not valid JSON") from exc
        if type(document) is not dict:
            raise ValueError("task association document must be a JSON object")
        if (
            document.get("artifact_type") != _TASK_ASSOCIATION_ARTIFACT_TYPE
            or document.get("schema_version") != TASK_ASSOCIATION_SCHEMA_VERSION
        ):
            raise ValueError("document is not a task association artifact")
        try:
            return cls(
                schema_version=document["schema_version"],
                layer_keys=_strict_key_tuple(
                    tuple(document["layer_keys"]), "layer_keys"
                ),
                task_keys=_strict_key_tuple(tuple(document["task_keys"]), "task_keys"),
                expert_keys=tuple(
                    _strict_key_tuple(tuple(row), f"expert_keys[{index}]")
                    for index, row in enumerate(document["expert_keys"])
                ),
                assignment_totals=_totals_rows(
                    document["assignment_totals"], "assignment_totals"
                ),
                enrichment_rows=_metric_rows(document["enrichment_rows"]),
                pmi_rows=_metric_rows(document["pmi_rows"]),
                mutual_information_rows=_scalar_rows(
                    document["mutual_information_rows"]
                ),
                specific_mi_rows=_specific_rows(document["specific_mi_rows"]),
                separability_rows=_optional_scalar_rows(document["separability_rows"]),
                exclusivity_rows=_exclusivity_rows(document["exclusivity_rows"]),
            )
        except KeyError as exc:
            raise ValueError("task association document is missing fields") from exc
        except (TypeError, ValueError):
            raise
        except Exception as exc:
            raise ValueError("task association document is not usable") from exc


def _totals_rows(
    value: object, field_name: str
) -> tuple[tuple[int, ...], ...]:
    if type(value) is not list:
        raise TypeError(f"{field_name} must be a list of integer rows")
    rows = []
    for row in value:
        if type(row) is not list:
            raise TypeError(f"{field_name} rows must be lists")
        for entry in row:
            if type(entry) is not int or isinstance(entry, bool) or entry < 0:
                raise ValueError(f"{field_name} entries must be non-negative integers")
        rows.append(tuple(row))
    return tuple(rows)


def _metric_rows(value: object) -> tuple[tuple[str, str, str, float | None], ...]:
    if type(value) is not list:
        raise TypeError("metric rows must be lists")
    rows: list[tuple[str, str, str, float | None]] = []
    for row in value:
        if type(row) is not list or len(row) != 4:
            raise TypeError("metric rows must be (layer, task, expert, value)")
        layer, task, expert, metric = row
        for key in (layer, task, expert):
            if type(key) is not str or not key:
                raise TypeError("metric keys must be non-empty strings")
        if metric is not None and (type(metric) is not float and type(metric) is not int):
            raise TypeError("metric values must be numbers or null")
        rows.append((layer, task, expert, float(metric) if metric is not None else None))
    return tuple(rows)


def _scalar_rows(value: object) -> tuple[tuple[str, float], ...]:
    if type(value) is not list:
        raise TypeError("scalar rows must be lists")
    rows: list[tuple[str, float]] = []
    for row in value:
        if type(row) is not list or len(row) != 2 or type(row[0]) is not str:
            raise TypeError("scalar rows must be (key, value)")
        if type(row[1]) is not float and type(row[1]) is not int:
            raise TypeError("scalar values must be numbers")
        rows.append((row[0], float(row[1])))
    return tuple(rows)


def _optional_scalar_rows(value: object) -> tuple[tuple[str, float | None], ...]:
    if type(value) is not list:
        raise TypeError("optional scalar rows must be lists")
    rows: list[tuple[str, float | None]] = []
    for row in value:
        if type(row) is not list or len(row) != 2 or type(row[0]) is not str:
            raise TypeError("optional scalar rows must be (key, value)")
        metric = row[1]
        if metric is not None and type(metric) is not float and type(metric) is not int:
            raise TypeError("optional scalar values must be numbers or null")
        rows.append((row[0], float(metric) if metric is not None else None))
    return tuple(rows)


def _specific_rows(value: object) -> tuple[tuple[str, str, float], ...]:
    if type(value) is not list:
        raise TypeError("specific rows must be lists")
    rows: list[tuple[str, str, float]] = []
    for row in value:
        if (
            type(row) is not list
            or len(row) != 3
            or type(row[0]) is not str
            or type(row[1]) is not str
        ):
            raise TypeError("specific rows must be (layer, task, value)")
        if type(row[2]) is not float and type(row[2]) is not int:
            raise TypeError("specific values must be numbers")
        rows.append((row[0], row[1], float(row[2])))
    return tuple(rows)


def _exclusivity_rows(
    value: object,
) -> tuple[tuple[str, str, float | None, int], ...]:
    if type(value) is not list:
        raise TypeError("exclusivity rows must be lists")
    rows: list[tuple[str, str, float | None, int]] = []
    for row in value:
        if (
            type(row) is not list
            or len(row) != 4
            or type(row[0]) is not str
            or type(row[1]) is not str
        ):
            raise TypeError("exclusivity rows must be (layer, expert, value, spread)")
        metric = row[2]
        if metric is not None and type(metric) is not float and type(metric) is not int:
            raise TypeError("exclusivity values must be numbers or null")
        spread = row[3]
        if type(spread) is not int or isinstance(spread, bool) or spread < 0:
            raise ValueError("exclusivity spreads must be non-negative integers")
        rows.append(
            (row[0], row[1], float(metric) if metric is not None else None, spread)
        )
    return tuple(rows)


def _entropy(distribution: list[float]) -> float:
    total = 0.0
    for probability in distribution:
        if probability > 0.0:
            total -= probability * math.log2(probability)
    return total


def analyze_task_association(
    counts: TaskExpertCounts, *, max_cells: int = 1_000_000
) -> TaskAssociationMatrix:
    """Compute all §11.2 association metrics over one validated count table."""

    if not isinstance(counts, TaskExpertCounts):
        raise TypeError(f"counts must be a TaskExpertCounts, got {type(counts).__name__}")
    if type(max_cells) is not int or isinstance(max_cells, bool):
        raise TypeError("max_cells must be an integer")
    if max_cells <= 0:
        raise TaskAssociationError("budget", "max_cells must be a strict positive integer")
    if counts.cell_count > max_cells:
        raise TaskAssociationError("budget", "count-table cells exceed max_cells")

    enrichment_rows: list[tuple[str, str, str, float | None]] = []
    pmi_rows: list[tuple[str, str, str, float | None]] = []
    mutual_information_rows: list[tuple[str, float]] = []
    specific_mi_rows: list[tuple[str, str, float]] = []
    separability_rows: list[tuple[str, float | None]] = []
    exclusivity_rows: list[tuple[str, str, float | None, int]] = []

    for layer_index, layer_key in enumerate(counts.layer_keys):
        experts = counts.expert_keys[layer_index]
        table = counts.counts[layer_index]
        task_totals = [sum(row) for row in table]
        layer_total = sum(task_totals)
        overall = [
            sum(table[task_index][expert_index] for task_index in range(len(table)))
            / layer_total
            for expert_index in range(len(experts))
        ]

        # Enrichment and PMI per (task, expert), in canonical order.
        conditionals: list[list[float]] = []
        for task_index, task_key in enumerate(counts.task_keys):
            conditional = [
                table[task_index][expert_index] / task_totals[task_index]
                for expert_index in range(len(experts))
            ]
            conditionals.append(conditional)
            for expert_index, expert_key in enumerate(experts):
                baseline = overall[expert_index]
                if baseline == 0.0:
                    enrichment_rows.append((layer_key, task_key, expert_key, None))
                    pmi_rows.append((layer_key, task_key, expert_key, None))
                    continue
                ratio = conditional[expert_index] / baseline
                enrichment_rows.append((layer_key, task_key, expert_key, ratio))
                if conditional[expert_index] == 0.0:
                    pmi_rows.append((layer_key, task_key, expert_key, None))
                else:
                    pmi_rows.append((layer_key, task_key, expert_key, math.log2(ratio)))

        # Mutual information over the (task, expert) joint within the layer.
        mi = 0.0
        for task_index, task_key in enumerate(counts.task_keys):
            specific = 0.0
            for expert_index in range(len(experts)):
                joint = table[task_index][expert_index] / layer_total
                if joint == 0.0:
                    continue
                baseline = (
                    task_totals[task_index] / layer_total * overall[expert_index]
                )
                term = math.log2(joint / baseline)
                mi += joint * term
                specific += conditionals[task_index][expert_index] * term
            specific_mi_rows.append((layer_key, task_key, specific))
        mutual_information_rows.append((layer_key, mi))

        # Separability: mean pairwise JS divergence between task distributions.
        if len(conditionals) < 2:
            separability_rows.append((layer_key, None))
        else:
            divergences = []
            for first in range(len(conditionals)):
                for second in range(first + 1, len(conditionals)):
                    midpoint = [
                        (conditionals[first][index] + conditionals[second][index]) / 2.0
                        for index in range(len(experts))
                    ]
                    divergence = _entropy(midpoint) - (
                        _entropy(conditionals[first]) + _entropy(conditionals[second])
                    ) / 2.0
                    divergences.append(divergence)
            separability_rows.append((layer_key, sum(divergences) / len(divergences)))

        # Exclusivity: how concentrated an expert's assignments are by task.
        for expert_index, expert_key in enumerate(experts):
            expert_total = sum(
                table[task_index][expert_index] for task_index in range(len(table))
            )
            if expert_total == 0:
                exclusivity_rows.append((layer_key, expert_key, None, 0))
                continue
            shares = [
                table[task_index][expert_index] / expert_total
                for task_index in range(len(table))
            ]
            spread = sum(1 for share in shares if share > 0.0)
            exclusivity_rows.append((layer_key, expert_key, max(shares), spread))

    return TaskAssociationMatrix(
        schema_version=TASK_ASSOCIATION_SCHEMA_VERSION,
        layer_keys=counts.layer_keys,
        task_keys=counts.task_keys,
        expert_keys=counts.expert_keys,
        assignment_totals=tuple(tuple(sum(row) for row in table) for table in counts.counts),
        enrichment_rows=tuple(enrichment_rows),
        pmi_rows=tuple(pmi_rows),
        mutual_information_rows=tuple(mutual_information_rows),
        specific_mi_rows=tuple(specific_mi_rows),
        separability_rows=tuple(separability_rows),
        exclusivity_rows=tuple(exclusivity_rows),
    )
