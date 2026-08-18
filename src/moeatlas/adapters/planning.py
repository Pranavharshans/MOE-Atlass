"""Compile validated static adapter inspections into inert routing plans.

This boundary only translates portable inspection evidence into a
serializable :class:`~moeatlas.probe.ProbePlan`.  It does not call an adapter,
inspect a model, resolve a module, install hooks, decode tensors, or retain
runtime objects.
"""

from __future__ import annotations

from ..core import ComponentKind
from ..probe.plan import (
    CaptureMode,
    CapturePolicy,
    HookPoint,
    ProbeLevel,
    ProbePlan,
    ProbeTarget,
    ReductionPolicy,
)
from .contracts import AdapterInspection

_STAGES = frozenset({"inspection", "targets", "plan"})


class AdapterProbePlanError(ValueError):
    """Safe fixed-stage failure raised by routing-plan compilation."""

    def __init__(self, stage: str) -> None:
        if stage not in _STAGES:
            raise ValueError("adapter probe-plan error stage must be inspection, targets, or plan")
        self.stage = stage
        super().__init__(f"adapter routing probe planning failed at {stage}")


def _inspection(value: object) -> AdapterInspection:
    if type(value) is not AdapterInspection:
        raise AdapterProbePlanError("inspection")
    try:
        payload = value.model_dump(mode="json")
        fresh = AdapterInspection.model_validate(payload)
    except Exception as exc:
        raise AdapterProbePlanError("inspection") from exc
    if type(fresh) is not AdapterInspection:
        raise AdapterProbePlanError("inspection")
    return fresh


def _targets(inspection: AdapterInspection) -> tuple[ProbeTarget, ...]:
    try:
        routers = [
            component
            for component in inspection.report.components
            if component.kind is ComponentKind.ROUTER
        ]
        if not routers:
            raise AdapterProbePlanError("targets")
        paths = [component.module_path for component in routers]
        if len(set(paths)) != len(paths):
            raise AdapterProbePlanError("targets")
        return tuple(
            ProbeTarget(
                module_path=component.module_path,
                component_key=component.component_key,
                component_kind=ComponentKind.ROUTER,
            )
            for component in routers
        )
    except AdapterProbePlanError:
        raise
    except Exception as exc:
        raise AdapterProbePlanError("targets") from exc


def build_routing_probe_plan(inspection: AdapterInspection) -> ProbePlan:
    """Build the canonical passive ROUTING plan for every inspected router."""

    validated = _inspection(inspection)
    targets = _targets(validated)
    try:
        capture = CapturePolicy(
            mode=CaptureMode.REDUCED,
            reduction=ReductionPolicy.TOP_K,
            include_inputs=False,
            include_outputs=True,
            include_gradients=False,
            raw_opt_in=False,
            max_items=None,
            max_bytes=None,
            sample_rate=1.0,
            sample_seed=None,
        )
        return ProbePlan(
            level=ProbeLevel.ROUTING,
            hook_points=(HookPoint.FORWARD,),
            targets=targets,
            include=(),
            exclude=(),
            capture=capture,
            intervention_opt_in=False,
        )
    except Exception as exc:
        raise AdapterProbePlanError("plan") from exc


__all__ = ["AdapterProbePlanError", "build_routing_probe_plan"]
