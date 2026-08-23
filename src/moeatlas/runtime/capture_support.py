"""Honest, structure-only capture support classification.

Discovery can prove that hook targets exist, but it cannot prove that an
unseen router payload will decode or that experts will execute until a real
forward runs.  This module keeps those states distinct and model-family
neutral: it grades only the static :class:`DiscoveryReport` evidence consumed
by the generic capture runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ..discovery import DiscoveryReport
from .generic_capture import (
    StructuredCaptureError,
    structured_expert_targets,
    structured_router_targets,
)

CaptureSupportGrade = Literal[
    "not_moe",
    "topology_only",
    "routing_candidate",
    "routing_and_activity_candidate",
]


@dataclass(frozen=True, slots=True)
class CaptureSupportReport:
    """Static support evidence without forward-time claims."""

    grade: CaptureSupportGrade
    topology_discovered: bool
    routing_capture: Literal["candidate", "unavailable"]
    expert_activity_capture: Literal["candidate", "unavailable"]
    router_target_count: int
    expert_target_count: int
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "grade": self.grade,
            "topology_discovered": self.topology_discovered,
            "routing_capture": self.routing_capture,
            "expert_activity_capture": self.expert_activity_capture,
            "router_target_count": self.router_target_count,
            "expert_target_count": self.expert_target_count,
            "limitations": list(self.limitations),
        }


def classify_capture_support(report: DiscoveryReport) -> CaptureSupportReport:
    """Classify the generic capture seams proven by one static scan.

    ``candidate`` deliberately means only that safe hook targets resolved.
    A completed run must still validate payload shape, complete top-k routing,
    and expert activity before those evidence types become ``captured``.
    """

    if type(report) is not DiscoveryReport:
        raise TypeError("report must be an exact DiscoveryReport")
    fresh = DiscoveryReport.model_validate(report.model_dump(mode="json"))
    facts = fresh.facts
    topology = (
        type(facts.expert_count) is int
        and facts.expert_count > 0
        and type(facts.routed_top_k) is int
        and 0 < facts.routed_top_k <= facts.expert_count
    )
    if not topology:
        return CaptureSupportReport(
            grade="not_moe",
            topology_discovered=False,
            routing_capture="unavailable",
            expert_activity_capture="unavailable",
            router_target_count=0,
            expert_target_count=0,
            limitations=("static scan did not prove routed MoE topology",),
        )

    limitations: list[str] = []
    try:
        routers = structured_router_targets(fresh)
    except StructuredCaptureError as exc:
        routers = ()
        limitations.append(str(exc))
    if not routers:
        return CaptureSupportReport(
            grade="topology_only",
            topology_discovered=True,
            routing_capture="unavailable",
            expert_activity_capture="unavailable",
            router_target_count=0,
            expert_target_count=0,
            limitations=tuple(limitations or ("no trusted router hook target resolved",)),
        )

    try:
        experts = structured_expert_targets(fresh)
    except StructuredCaptureError as exc:
        experts = ()
        limitations.append(str(exc))
    limitations.insert(
        0,
        "router payload compatibility remains unproven until a real forward validates events",
    )
    if not experts:
        limitations.append("no routed-expert activity hook targets resolved")
        return CaptureSupportReport(
            grade="routing_candidate",
            topology_discovered=True,
            routing_capture="candidate",
            expert_activity_capture="unavailable",
            router_target_count=len(routers),
            expert_target_count=0,
            limitations=tuple(limitations),
        )
    limitations.append("expert activity remains unproven until selected experts fire")
    return CaptureSupportReport(
        grade="routing_and_activity_candidate",
        topology_discovered=True,
        routing_capture="candidate",
        expert_activity_capture="candidate",
        router_target_count=len(routers),
        expert_target_count=len(experts),
        limitations=tuple(limitations),
    )


__all__ = ["CaptureSupportGrade", "CaptureSupportReport", "classify_capture_support"]
