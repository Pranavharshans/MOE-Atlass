"""Structure-driven expert interventions for loaded Transformers models.

This module intentionally contains no model-family allowlist.  It resolves a
human layer/expert coordinate against the immutable discovery report and owns
only temporary forward-hook handles.  Models that do not expose each routed
expert as an independently hookable module are rejected explicitly; packed
storage and fused execution are recorded as separate facts and neither is
reported as successfully ablated without runtime evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from moeatlas.core import ComponentKind
from moeatlas.discovery import DiscoveryReport
from moeatlas.interventions.recipes import InterventionOperation, InterventionRecipe
from moeatlas.runtime.generic_capture import StructuredCaptureError, structured_expert_targets

_TARGET = re.compile(r"^layer:(0|[1-9][0-9]*)/expert:(0|[1-9][0-9]*)$")


class TransformersInterventionError(RuntimeError):
    """Safe failure for unsupported or invalid live expert interventions."""


class InterventionSupportTier(str, Enum):
    """How a discovered implementation can be manipulated safely."""

    EXPOSED_EXPERTS = "exposed_experts"
    PACKED_EXPERTS = "packed_experts"
    OPAQUE_EXPERTS = "opaque_experts"
    UNAVAILABLE = "unavailable"


class ExpertWeightLayout(str, Enum):
    """How routed expert parameters are represented in the loaded module tree."""

    INDEXED_MODULES = "indexed_modules"
    PACKED_TENSORS = "packed_tensors"
    OPAQUE = "opaque"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class InterventionCapabilityReport:
    """Model-neutral, evidence-bound intervention support declaration."""

    tier: InterventionSupportTier
    operations: tuple[InterventionOperation, ...]
    target_count: int
    target_format: str | None
    reason: str
    weight_layout: ExpertWeightLayout
    execution_backend: str | None = None
    fused_backend: bool | None = None

    @property
    def live_supported(self) -> bool:
        return bool(self.operations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "live_supported": self.live_supported,
            "operations": [operation.value for operation in self.operations],
            "reason": self.reason,
            "target_count": self.target_count,
            "target_format": self.target_format,
            "tier": self.tier.value,
            "weight_layout": self.weight_layout.value,
            "execution_backend": self.execution_backend,
            "fused_backend": self.fused_backend,
        }


@dataclass(frozen=True, slots=True)
class ExpertInterventionTarget:
    """One user-facing layer × expert coordinate bound to a real module."""

    label: str
    layer_index: int
    expert_index: int
    layer_key: str
    expert_key: str
    module_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "layer_index": self.layer_index,
            "expert_index": self.expert_index,
            "layer_key": self.layer_key,
            "expert_key": self.expert_key,
            "module_path": self.module_path,
        }


def intervention_targets(report: DiscoveryReport) -> tuple[ExpertInterventionTarget, ...]:
    """Return stable layer × expert coordinates for independently exposed experts."""

    try:
        discovered = structured_expert_targets(report)
    except StructuredCaptureError as exc:
        raise TransformersInterventionError(
            "model does not expose independently hookable routed experts"
        ) from exc
    grouped: dict[int, list[Any]] = {}
    for target in discovered:
        grouped.setdefault(target.layer_index, []).append(target)
    resolved: list[ExpertInterventionTarget] = []
    for layer_index in sorted(grouped):
        row = grouped[layer_index]
        for expert_index, target in enumerate(row):
            resolved.append(
                ExpertInterventionTarget(
                    label=f"layer:{layer_index}/expert:{expert_index}",
                    layer_index=layer_index,
                    expert_index=expert_index,
                    layer_key=target.layer_key,
                    expert_key=target.component_key,
                    module_path=target.module_path,
                )
            )
    if not resolved:
        raise TransformersInterventionError(
            "model does not expose independently hookable routed experts"
        )
    return tuple(resolved)


def classify_intervention_capability(report: DiscoveryReport) -> InterventionCapabilityReport:
    """Declare intervention support without guessing packed tensor semantics."""

    if type(report) is not DiscoveryReport:
        raise TypeError("report must be an exact DiscoveryReport")
    try:
        targets = intervention_targets(report)
    except TransformersInterventionError:
        targets = ()
    if targets:
        return InterventionCapabilityReport(
            tier=InterventionSupportTier.EXPOSED_EXPERTS,
            operations=(InterventionOperation.ABLATE, InterventionOperation.SCALE),
            target_count=len(targets),
            target_format="layer:N/expert:M",
            reason="routed experts are independently exposed as hookable modules",
            weight_layout=ExpertWeightLayout.INDEXED_MODULES,
        )
    has_moe = report.facts.expert_count is not None or any(
        component.kind in {ComponentKind.MOE_LAYER, ComponentKind.EXPERT_CONTAINER}
        for component in report.components
    )
    containers = tuple(
        component
        for component in report.components
        if component.kind is ComponentKind.EXPERT_CONTAINER
    )
    expert_count = report.facts.expert_count
    has_packed_weights = type(expert_count) is int and any(
        any(len(shape) >= 3 and expert_count in shape for shape in component.tensor_shapes.values())
        for component in containers
    )
    if has_moe and has_packed_weights:
        return InterventionCapabilityReport(
            tier=InterventionSupportTier.PACKED_EXPERTS,
            operations=(),
            target_count=0,
            target_format=None,
            reason=(
                "routed expert weights are packed tensors; the active execution backend "
                "has not been inspected"
            ),
            weight_layout=ExpertWeightLayout.PACKED_TENSORS,
        )
    if has_moe and containers:
        return InterventionCapabilityReport(
            tier=InterventionSupportTier.OPAQUE_EXPERTS,
            operations=(),
            target_count=0,
            target_format=None,
            reason="routed expert storage is opaque to static discovery",
            weight_layout=ExpertWeightLayout.OPAQUE,
        )
    return InterventionCapabilityReport(
        tier=InterventionSupportTier.UNAVAILABLE,
        operations=(),
        target_count=0,
        target_format=None,
        reason="no routed expert intervention targets were discovered",
        weight_layout=ExpertWeightLayout.UNAVAILABLE,
    )


def _scale_output(output: object, factor: float) -> object:
    """Scale one tensor-like expert output without guessing nested semantics."""

    if output is None or isinstance(output, bool | int | float | str | bytes):
        raise TransformersInterventionError("expert output is not a tensor-like value")
    multiply = getattr(output, "mul", None)
    if callable(multiply):
        return multiply(factor)
    try:
        scaled = output * factor  # type: ignore[operator]
    except Exception as exc:
        raise TransformersInterventionError("expert output cannot be scaled safely") from exc
    if scaled is NotImplemented:
        raise TransformersInterventionError("expert output cannot be scaled safely")
    return scaled


class TransformersExpertInterventionCapability:
    """Temporary ablate/scale hooks resolved only from discovery evidence."""

    def __init__(self, report: DiscoveryReport) -> None:
        if type(report) is not DiscoveryReport:
            raise TypeError("report must be an exact DiscoveryReport")
        self._targets = intervention_targets(report)
        self._handles: list[object] = []
        self._invocations: dict[str, int] = {}

    @property
    def target_inventory(self) -> tuple[ExpertInterventionTarget, ...]:
        return self._targets

    @property
    def invocation_counts(self) -> dict[str, int]:
        return dict(sorted(self._invocations.items()))

    def capture(self, module: object) -> tuple[()]:
        del module
        if self._handles:
            raise TransformersInterventionError("intervention capability is already active")
        self._invocations = {}
        return ()

    def apply(self, module: object, recipe: InterventionRecipe) -> None:
        if recipe.operation not in {InterventionOperation.ABLATE, InterventionOperation.SCALE}:
            raise TransformersInterventionError(
                "live execution currently supports only expert ablation and scaling"
            )
        named_modules = getattr(module, "named_modules", None)
        if not callable(named_modules):
            raise TransformersInterventionError("loaded model does not expose named_modules()")
        try:
            modules = dict(named_modules())
        except Exception as exc:
            raise TransformersInterventionError("model module inventory is unavailable") from exc
        by_label = {target.label: target for target in self._targets}
        unknown = tuple(label for label in recipe.targets if label not in by_label)
        if unknown:
            raise TransformersInterventionError(
                "recipe targets are outside the discovered routed-expert universe"
            )
        factor = 0.0 if recipe.operation is InterventionOperation.ABLATE else recipe.factor
        assert factor is not None
        try:
            for label in recipe.targets:
                target = by_label[label]
                expert = modules.get(target.module_path)
                register = getattr(expert, "register_forward_hook", None)
                if not callable(register):
                    raise TransformersInterventionError(
                        "a selected expert does not support temporary forward hooks"
                    )

                def intervention_hook(
                    _module: object,
                    _inputs: object,
                    output: object,
                    *,
                    target_label: str = label,
                    scale_factor: float = factor,
                ) -> object:
                    self._invocations[target_label] = self._invocations.get(target_label, 0) + 1
                    return _scale_output(output, scale_factor)

                handle = register(intervention_hook)
                if not callable(getattr(handle, "remove", None)):
                    raise TransformersInterventionError(
                        "selected expert returned an unremovable hook handle"
                    )
                self._handles.append(handle)
        except BaseException:
            self._remove_handles()
            raise

    def restore(self, module: object, snapshot: object) -> None:
        del module
        if snapshot != ():
            raise TransformersInterventionError("intervention snapshot is invalid")
        self._remove_handles()

    def _remove_handles(self) -> None:
        failures: list[BaseException] = []
        while self._handles:
            handle = self._handles.pop()
            try:
                handle.remove()
            except BaseException as exc:
                failures.append(exc)
        if failures:
            raise TransformersInterventionError(
                f"failed to remove {len(failures)} intervention hook(s)"
            ) from failures[0]


def parse_intervention_target(label: str) -> tuple[int, int]:
    """Parse the public ``layer:N/expert:M`` coordinate format."""

    if type(label) is not str:
        raise TypeError("intervention target label must be a string")
    match = _TARGET.fullmatch(label)
    if match is None:
        raise ValueError("intervention target must use layer:N/expert:M")
    return int(match.group(1)), int(match.group(2))


__all__ = [
    "ExpertInterventionTarget",
    "ExpertWeightLayout",
    "InterventionCapabilityReport",
    "InterventionSupportTier",
    "TransformersExpertInterventionCapability",
    "TransformersInterventionError",
    "intervention_targets",
    "classify_intervention_capability",
    "parse_intervention_target",
]
