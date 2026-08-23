"""Evidence-backed expert candidates for controlled intervention studies.

The ranking is descriptive: it combines validated routing assignments with
persisted expert contribution norms. It deliberately does not call an expert
important, specialized, redundant, or safe to prune. Those claims require a
paired intervention on the same task and run contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..core import parse_component_key, validate_stable_identifier
from .expert_activity import ExpertActivitySummary
from .routing_load import RoutingLoadMatrix

EXPERT_CANDIDATE_SCHEMA_VERSION = "1.0"


def _strict_positive(value: object, field_name: str) -> int:
    if type(value) is not int or isinstance(value, bool):
        raise TypeError(f"{field_name} must be a strict integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


@dataclass(frozen=True, slots=True)
class ExpertInterventionCandidate:
    """One fully reconciled routing and contribution observation."""

    layer_index: int
    expert_index: int
    layer_key: str
    expert_key: str
    routing_count: int
    routing_share: float
    event_count: int
    mean_contribution: float
    contribution_variance: float
    total_contribution: float

    def __post_init__(self) -> None:
        for name in ("layer_index", "expert_index", "routing_count", "event_count"):
            value = getattr(self, name)
            if type(value) is not int or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a strict nonnegative integer")
        if self.routing_count == 0 or self.event_count != self.routing_count:
            raise ValueError("candidate evidence must cover every routed assignment")
        for name in (
            "routing_share",
            "mean_contribution",
            "contribution_variance",
            "total_contribution",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite nonnegative float")
        if self.routing_share > 1.0:
            raise ValueError("routing_share must not exceed one")
        if not math.isclose(
            self.total_contribution,
            self.event_count * self.mean_contribution,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("total_contribution must match the observed evidence")
        if type(self.layer_key) is not str or type(self.expert_key) is not str:
            raise TypeError("candidate keys must be exact strings")
        parse_component_key(self.layer_key)
        parse_component_key(self.expert_key)

    def to_dict(self) -> dict[str, object]:
        return {
            "layer_index": self.layer_index,
            "expert_index": self.expert_index,
            "layer_key": self.layer_key,
            "expert_key": self.expert_key,
            "routing_count": self.routing_count,
            "routing_share": self.routing_share,
            "event_count": self.event_count,
            "mean_contribution": self.mean_contribution,
            "contribution_variance": self.contribution_variance,
            "total_contribution": self.total_contribution,
        }


@dataclass(frozen=True, slots=True)
class ExpertCandidateRanking:
    """Bounded high/low observed-contribution candidates for one run."""

    schema_version: str
    run_key: str
    ranked_cell_count: int
    incomplete_cell_count: int
    high_observed: tuple[ExpertInterventionCandidate, ...]
    low_observed: tuple[ExpertInterventionCandidate, ...]
    evidence_complete: bool

    def __post_init__(self) -> None:
        if self.schema_version != EXPERT_CANDIDATE_SCHEMA_VERSION:
            raise ValueError("schema_version is not the exact candidate-ranking version")
        if type(self.run_key) is not str:
            raise TypeError("run_key must be an exact string")
        validate_stable_identifier(self.run_key, field_name="run_key")
        for name in ("ranked_cell_count", "incomplete_cell_count"):
            value = getattr(self, name)
            if type(value) is not int or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a strict nonnegative integer")
        for name in ("high_observed", "low_observed"):
            values = getattr(self, name)
            if type(values) is not tuple or any(
                type(value) is not ExpertInterventionCandidate for value in values
            ):
                raise TypeError(f"{name} must contain exact candidate values")
            if len(values) > self.ranked_cell_count:
                raise ValueError(f"{name} exceeds the ranked cell universe")
        if type(self.evidence_complete) is not bool:
            raise TypeError("evidence_complete must be an exact boolean")
        if self.evidence_complete != (self.incomplete_cell_count == 0):
            raise ValueError("evidence_complete must match incomplete_cell_count")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_key": self.run_key,
            "ranked_cell_count": self.ranked_cell_count,
            "incomplete_cell_count": self.incomplete_cell_count,
            "high_observed": [value.to_dict() for value in self.high_observed],
            "low_observed": [value.to_dict() for value in self.low_observed],
            "evidence_complete": self.evidence_complete,
            "claim_boundary": (
                "Descriptive observed contribution only; paired intervention is required "
                "to establish a causal task effect."
            ),
        }


def rank_expert_candidates(
    matrix: RoutingLoadMatrix,
    activity: ExpertActivitySummary,
    *,
    max_candidates: int = 8,
) -> ExpertCandidateRanking:
    """Rank only cells whose routing and expert-event counts reconcile exactly."""

    if type(matrix) is not RoutingLoadMatrix:
        raise TypeError("matrix must be an exact RoutingLoadMatrix")
    if type(activity) is not ExpertActivitySummary:
        raise TypeError("activity must be an exact ExpertActivitySummary")
    limit = _strict_positive(max_candidates, "max_candidates")
    if matrix.run_key != activity.run_key:
        raise ValueError("run_key does not match expert activity evidence")
    if matrix.layer_keys != activity.layer_keys:
        raise ValueError("layer_keys do not match expert activity evidence")
    if matrix.expert_keys != activity.expert_keys:
        raise ValueError("expert_keys do not match expert activity evidence")

    candidates: list[ExpertInterventionCandidate] = []
    incomplete = 0
    for layer_position, layer_key in enumerate(matrix.layer_keys):
        activity_row = activity.layers[layer_position]
        for expert_position, expert_key in enumerate(matrix.expert_keys[layer_position]):
            routing_count = matrix.assignment_counts[layer_position][expert_position]
            event_count = activity_row.event_counts[expert_position]
            mean = activity_row.mean_contributions[expert_position]
            variance = activity_row.variance_contributions[expert_position]
            if (
                routing_count == 0
                or event_count != routing_count
                or mean is None
                or variance is None
            ):
                incomplete += 1
                continue
            candidates.append(
                ExpertInterventionCandidate(
                    layer_index=matrix.layer_indices[layer_position],
                    expert_index=expert_position,
                    layer_key=layer_key,
                    expert_key=expert_key,
                    routing_count=routing_count,
                    routing_share=matrix.assignment_shares[layer_position][expert_position],
                    event_count=event_count,
                    mean_contribution=mean,
                    contribution_variance=variance,
                    total_contribution=float(event_count * mean),
                )
            )

    high = tuple(
        sorted(
            candidates,
            key=lambda item: (
                -item.total_contribution,
                item.layer_index,
                item.expert_index,
                item.expert_key,
            ),
        )[:limit]
    )
    low = tuple(
        sorted(
            candidates,
            key=lambda item: (
                item.total_contribution,
                item.layer_index,
                item.expert_index,
                item.expert_key,
            ),
        )[:limit]
    )
    return ExpertCandidateRanking(
        schema_version=EXPERT_CANDIDATE_SCHEMA_VERSION,
        run_key=matrix.run_key,
        ranked_cell_count=len(candidates),
        incomplete_cell_count=incomplete,
        high_observed=high,
        low_observed=low,
        evidence_complete=incomplete == 0,
    )


__all__ = [
    "EXPERT_CANDIDATE_SCHEMA_VERSION",
    "ExpertCandidateRanking",
    "ExpertInterventionCandidate",
    "rank_expert_candidates",
]
