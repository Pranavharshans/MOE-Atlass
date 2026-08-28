"""Structure-driven expert interventions for loaded Transformers models.

This module intentionally contains no model-family allowlist.  It resolves a
human layer/expert coordinate against the immutable discovery report and owns
temporary forward hooks or reversible Hugging Face backend delegates. Packed
storage and fused execution remain separate facts; packed contribution zeroing
requires both a proven expert axis and a live backend declaration.
"""

from __future__ import annotations

import importlib
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from moeatlas.core import ComponentKind
from moeatlas.discovery import DiscoveryReport
from moeatlas.interventions.recipes import InterventionOperation, InterventionRecipe
from moeatlas.runtime.generic_capture import (
    StructuredCaptureError,
    structured_expert_targets,
    structured_router_targets,
)

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


class ExpertExecutionMode(str, Enum):
    """Observed execution strategy declared by a loaded expert backend."""

    REFERENCE = "reference"
    BATCHED_MATMUL = "batched_matmul"
    GROUPED_MATMUL = "grouped_matmul"
    ACCELERATED = "accelerated"
    FUSED = "fused"
    CUSTOM = "custom"
    UNRESOLVED = "unresolved"


class ExpertBackendDiscoveryStatus(str, Enum):
    """Whether a loaded model supplied trustworthy backend declarations."""

    OBSERVED = "observed"
    UNRESOLVED = "unresolved"
    UNAVAILABLE = "unavailable"
    INVALID = "invalid"


class ExpertOperation(str, Enum):
    """Stable user-facing expert observation and intervention operations."""

    CAPTURE_ROUTING = "capture_routing"
    ZERO_CONTRIBUTION = "zero_contribution"
    SCALE_CONTRIBUTION = "scale_contribution"
    EXCLUDE_AND_RENORMALIZE = "exclude_and_renormalize"
    REROUTE_NEXT_BEST = "reroute_next_best"
    SKIP_COMPUTE = "skip_compute"


class OperationCapabilityStatus(str, Enum):
    """Whether one exact operation can be executed with current evidence."""

    AVAILABLE = "available"
    RUN_VALIDATION_REQUIRED = "run_validation_required"
    NOT_IMPLEMENTED = "not_implemented"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class OperationCapability:
    """One precise operation verdict with evidence and semantic boundaries."""

    operation: ExpertOperation
    label: str
    status: OperationCapabilityStatus
    reason: str
    evidence: tuple[str, ...]
    changes_routing: bool
    skips_compute: bool | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation.value,
            "label": self.label,
            "status": self.status.value,
            "reason": self.reason,
            "evidence": list(self.evidence),
            "changes_routing": self.changes_routing,
            "skips_compute": self.skips_compute,
        }


@dataclass(frozen=True, slots=True)
class ExpertBackendEvidence:
    """One bounded Hugging Face model/submodel backend declaration."""

    scope: str
    implementation: str | None
    mode: ExpertExecutionMode
    fused: bool | None
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "implementation": self.implementation,
            "mode": self.mode.value,
            "fused": self.fused,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class ExpertBackendDiscovery:
    """Non-fatal result of inspecting the optional Transformers interface."""

    status: ExpertBackendDiscoveryStatus
    backends: tuple[ExpertBackendEvidence, ...]
    reason: str
    source: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "backends": [item.to_dict() for item in self.backends],
            "reason": self.reason,
            "source": self.source,
        }


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
    execution_backends: tuple[ExpertBackendEvidence, ...] = ()
    backend_discovery: ExpertBackendDiscovery | None = None
    operation_capabilities: tuple[OperationCapability, ...] = ()

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
            "execution_backends": [item.to_dict() for item in self.execution_backends],
            "backend_discovery": (
                self.backend_discovery.to_dict() if self.backend_discovery is not None else None
            ),
            "operation_capabilities": [
                operation.to_dict() for operation in self.operation_capabilities
            ],
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
    """Return stable layer × expert coordinates for exposed or packed experts."""

    try:
        discovered = structured_expert_targets(report)
    except StructuredCaptureError:
        discovered = ()
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
        expert_count = report.facts.expert_count
        containers = {
            component.layer_index: component
            for component in report.components
            if component.kind is ComponentKind.EXPERT_CONTAINER
            and component.layer_index is not None
            and type(expert_count) is int
            and any(
                len(shape) >= 3 and expert_count in shape
                for shape in component.tensor_shapes.values()
            )
        }
        if type(expert_count) is int and expert_count > 0:
            layers = {
                component.layer_index: component
                for component in report.components
                if component.kind is ComponentKind.MOE_LAYER and component.layer_index is not None
            }
            routers = sorted(
                (
                    component
                    for component in report.components
                    if component.kind is ComponentKind.ROUTER and component.layer_index is not None
                ),
                key=lambda component: component.layer_index,  # type: ignore[arg-type]
            )
            for router in routers:
                assert router.layer_index is not None
                container = containers.get(router.layer_index)
                layer = layers.get(router.layer_index)
                if container is None or layer is None:
                    continue
                for expert_index in range(expert_count):
                    resolved.append(
                        ExpertInterventionTarget(
                            label=f"layer:{router.layer_index}/expert:{expert_index}",
                            layer_index=router.layer_index,
                            expert_index=expert_index,
                            layer_key=layer.component_key,
                            expert_key=f"{container.component_key}.packed.{expert_index}",
                            module_path=container.module_path,
                        )
                    )
    if not resolved:
        raise TransformersInterventionError(
            "model does not expose independently hookable or proven packed routed experts"
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
    has_indexed_experts = any(
        component.kind is ComponentKind.EXPERT for component in report.components
    )
    if targets and has_indexed_experts:
        capability = InterventionCapabilityReport(
            tier=InterventionSupportTier.EXPOSED_EXPERTS,
            operations=(InterventionOperation.ABLATE, InterventionOperation.SCALE),
            target_count=len(targets),
            target_format="layer:N/expert:M",
            reason="routed experts are independently exposed as hookable modules",
            weight_layout=ExpertWeightLayout.INDEXED_MODULES,
        )
    else:
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
            any(
                len(shape) >= 3 and expert_count in shape
                for shape in component.tensor_shapes.values()
            )
            for component in containers
        )
        if has_moe and has_packed_weights:
            capability = InterventionCapabilityReport(
                tier=InterventionSupportTier.PACKED_EXPERTS,
                operations=(),
                target_count=len(targets),
                target_format="layer:N/expert:M" if targets else None,
                reason=(
                    "routed expert weights are packed tensors; the active execution backend "
                    "has not been inspected"
                ),
                weight_layout=ExpertWeightLayout.PACKED_TENSORS,
            )
        elif has_moe and containers:
            capability = InterventionCapabilityReport(
                tier=InterventionSupportTier.OPAQUE_EXPERTS,
                operations=(),
                target_count=0,
                target_format=None,
                reason="routed expert storage is opaque to static discovery",
                weight_layout=ExpertWeightLayout.OPAQUE,
            )
        else:
            capability = InterventionCapabilityReport(
                tier=InterventionSupportTier.UNAVAILABLE,
                operations=(),
                target_count=0,
                target_format=None,
                reason="no routed expert intervention targets were discovered",
                weight_layout=ExpertWeightLayout.UNAVAILABLE,
            )
    return replace(
        capability,
        operation_capabilities=_operation_capabilities(capability, report),
    )


def _router_target_count(report: DiscoveryReport) -> int:
    try:
        return len(structured_router_targets(report))
    except StructuredCaptureError:
        return 0


def _operation_capabilities(
    capability: InterventionCapabilityReport,
    report: DiscoveryReport,
) -> tuple[OperationCapability, ...]:
    router_count = _router_target_count(report)
    has_backend = (
        capability.backend_discovery is not None
        and capability.backend_discovery.status is ExpertBackendDiscoveryStatus.OBSERVED
    )
    can_zero = InterventionOperation.ABLATE in capability.operations
    can_scale = InterventionOperation.SCALE in capability.operations
    has_moe_structure = capability.weight_layout is not ExpertWeightLayout.UNAVAILABLE
    expert_evidence = (
        (f"structure.exposed_experts={capability.target_count}",)
        if capability.weight_layout is ExpertWeightLayout.INDEXED_MODULES
        else (
            f"structure.weight_layout={capability.weight_layout.value}",
            f"structure.logical_targets={capability.target_count}",
        )
    )
    backend_evidence = (
        (f"runtime.expert_backend={capability.execution_backend or 'mixed'}",)
        if has_backend
        else ()
    )

    def contribution(
        operation: ExpertOperation,
        label: str,
        available: bool,
    ) -> OperationCapability:
        if available:
            status = OperationCapabilityStatus.AVAILABLE
            reason = (
                "the live expert backend supports temporary routing-weight masking"
                if capability.weight_layout is ExpertWeightLayout.PACKED_TENSORS
                else "independent expert modules support temporary output hooks"
            )
        elif (
            operation is ExpertOperation.ZERO_CONTRIBUTION
            and capability.weight_layout is ExpertWeightLayout.PACKED_TENSORS
            and not has_backend
        ):
            status = OperationCapabilityStatus.RUN_VALIDATION_REQUIRED
            reason = "packed ablation requires a reversible live expert backend"
        elif has_backend or capability.weight_layout is ExpertWeightLayout.PACKED_TENSORS:
            status = OperationCapabilityStatus.NOT_IMPLEMENTED
            reason = "packed contribution control requires a supported live expert backend"
        else:
            status = OperationCapabilityStatus.UNAVAILABLE
            reason = "no independently controllable expert contribution seam was proven"
        return OperationCapability(
            operation=operation,
            label=label,
            status=status,
            reason=reason,
            evidence=(*expert_evidence, *backend_evidence),
            changes_routing=False,
            skips_compute=False,
        )

    routing_status = (
        OperationCapabilityStatus.RUN_VALIDATION_REQUIRED
        if router_count
        else OperationCapabilityStatus.UNAVAILABLE
    )
    routing_reason = (
        "static router targets exist; a real forward must validate their payload"
        if router_count
        else "no structurally addressable router target was discovered"
    )
    writable_status = (
        OperationCapabilityStatus.NOT_IMPLEMENTED
        if router_count
        else OperationCapabilityStatus.UNAVAILABLE
    )
    return (
        OperationCapability(
            operation=ExpertOperation.CAPTURE_ROUTING,
            label="Capture routing",
            status=routing_status,
            reason=routing_reason,
            evidence=(f"structure.router_targets={router_count}",),
            changes_routing=False,
            skips_compute=False,
        ),
        contribution(ExpertOperation.ZERO_CONTRIBUTION, "Zero contribution", can_zero),
        contribution(ExpertOperation.SCALE_CONTRIBUTION, "Scale contribution", can_scale),
        OperationCapability(
            operation=ExpertOperation.EXCLUDE_AND_RENORMALIZE,
            label="Exclude and renormalize",
            status=writable_status,
            reason=(
                "router capture exists, but writable top-k exclusion and weight "
                "renormalization are not implemented"
                if router_count
                else "no writable router seam was proven"
            ),
            evidence=(f"structure.router_targets={router_count}",),
            changes_routing=True,
            skips_compute=None,
        ),
        OperationCapability(
            operation=ExpertOperation.REROUTE_NEXT_BEST,
            label="Reroute to next best",
            status=writable_status,
            reason=(
                "router capture exists, but next-best logits and writable dispatch are "
                "not implemented"
                if router_count
                else "next-best router logits are unavailable"
            ),
            evidence=(f"structure.router_targets={router_count}",),
            changes_routing=True,
            skips_compute=None,
        ),
        OperationCapability(
            operation=ExpertOperation.SKIP_COMPUTE,
            label="Skip expert compute",
            status=(
                OperationCapabilityStatus.NOT_IMPLEMENTED
                if has_moe_structure
                else OperationCapabilityStatus.UNAVAILABLE
            ),
            reason=(
                "expert structure exists, but no pre-dispatch compute-skipping adapter is "
                "implemented"
                if has_moe_structure
                else "no expert dispatch seam was discovered"
            ),
            evidence=(*expert_evidence, *backend_evidence),
            changes_routing=False,
            skips_compute=True,
        ),
    )


_KNOWN_HUGGINGFACE_BACKENDS: dict[str, tuple[ExpertExecutionMode, bool | None]] = {
    "eager": (ExpertExecutionMode.REFERENCE, False),
    "batched_mm": (ExpertExecutionMode.BATCHED_MATMUL, False),
    "grouped_mm": (ExpertExecutionMode.GROUPED_MATMUL, False),
    # DeepGEMM accelerates grouped expert matrix multiplications. It is not
    # sufficient evidence that routing, activation, and combination are one
    # fused operation.
    "deepgemm": (ExpertExecutionMode.ACCELERATED, None),
    # SonicMoE is registered by Transformers as a fused expert implementation.
    "sonicmoe": (ExpertExecutionMode.FUSED, True),
}


def _safe_backend_text(value: str, *, field: str, allow_empty: bool) -> str:
    if len(value) > 128 or any(ord(character) < 32 for character in value):
        raise TransformersInterventionError(f"Hugging Face expert backend {field} is malformed")
    normalized = value.strip()
    if not allow_empty and not normalized:
        raise TransformersInterventionError(f"Hugging Face expert backend {field} is malformed")
    return normalized


def discover_huggingface_expert_backends(
    model: object,
) -> ExpertBackendDiscovery:
    """Read the loaded model's public Transformers expert-backend snapshot.

    The implementation is deliberately duck-typed: importing MoEAtlas never
    imports Transformers, while real ``PreTrainedModel`` instances can expose
    ``get_experts_implementation()`` at runtime. Unknown registered backends
    remain custom and their fusion status remains unresolved.
    """

    try:
        getter = getattr(model, "get_experts_implementation", None)
    except Exception:
        return ExpertBackendDiscovery(
            status=ExpertBackendDiscoveryStatus.INVALID,
            backends=(),
            reason="loaded model expert backend interface could not be inspected",
            source=None,
        )
    if not callable(getter):
        return ExpertBackendDiscovery(
            status=ExpertBackendDiscoveryStatus.UNAVAILABLE,
            backends=(),
            reason="loaded model does not expose the Hugging Face expert backend interface",
            source=None,
        )
    try:
        snapshot = getter()
    except Exception:
        return ExpertBackendDiscovery(
            status=ExpertBackendDiscoveryStatus.INVALID,
            backends=(),
            reason="loaded model expert backend snapshot failed",
            source="model.get_experts_implementation",
        )
    if not isinstance(snapshot, Mapping) or len(snapshot) > 64:
        return ExpertBackendDiscovery(
            status=ExpertBackendDiscoveryStatus.INVALID,
            backends=(),
            reason="loaded model returned an invalid expert backend snapshot",
            source="model.get_experts_implementation",
        )
    evidence: list[ExpertBackendEvidence] = []
    scopes: set[str] = set()
    for raw_scope, raw_implementation in snapshot.items():
        if type(raw_scope) is not str or (
            raw_implementation is not None and type(raw_implementation) is not str
        ):
            return ExpertBackendDiscovery(
                status=ExpertBackendDiscoveryStatus.INVALID,
                backends=(),
                reason="loaded model returned an invalid expert backend snapshot",
                source="model.get_experts_implementation",
            )
        try:
            scope = _safe_backend_text(raw_scope, field="scope", allow_empty=True)
        except TransformersInterventionError:
            return ExpertBackendDiscovery(
                status=ExpertBackendDiscoveryStatus.INVALID,
                backends=(),
                reason="loaded model returned an invalid expert backend scope",
                source="model.get_experts_implementation",
            )
        if scope in scopes:
            return ExpertBackendDiscovery(
                status=ExpertBackendDiscoveryStatus.INVALID,
                backends=(),
                reason="loaded model returned duplicate expert backend scopes",
                source="model.get_experts_implementation",
            )
        scopes.add(scope)
        if raw_implementation is None:
            implementation = None
            mode = ExpertExecutionMode.UNRESOLVED
            fused = None
        else:
            try:
                implementation = _safe_backend_text(
                    raw_implementation, field="implementation", allow_empty=False
                )
            except TransformersInterventionError:
                return ExpertBackendDiscovery(
                    status=ExpertBackendDiscoveryStatus.INVALID,
                    backends=(),
                    reason="loaded model returned an invalid expert backend implementation",
                    source="model.get_experts_implementation",
                )
            mode, fused = _KNOWN_HUGGINGFACE_BACKENDS.get(
                implementation,
                (ExpertExecutionMode.CUSTOM, None),
            )
        evidence.append(
            ExpertBackendEvidence(
                scope=scope,
                implementation=implementation,
                mode=mode,
                fused=fused,
                source="model.get_experts_implementation",
            )
        )
    backends = tuple(sorted(evidence, key=lambda item: item.scope))
    status = (
        ExpertBackendDiscoveryStatus.OBSERVED
        if any(item.implementation is not None for item in backends)
        else ExpertBackendDiscoveryStatus.UNRESOLVED
    )
    reason = (
        "loaded model declared its active Hugging Face expert backend"
        if status is ExpertBackendDiscoveryStatus.OBSERVED
        else "loaded model exposed the interface but no active expert backend"
    )
    return ExpertBackendDiscovery(
        status=status,
        backends=backends,
        reason=reason,
        source="model.get_experts_implementation",
    )


def inspect_intervention_capability(
    report: DiscoveryReport,
    model: object,
) -> InterventionCapabilityReport:
    """Combine static storage evidence with the live HF backend declaration."""

    capability = classify_intervention_capability(report)
    backend_discovery = discover_huggingface_expert_backends(model)
    backends = backend_discovery.backends
    implementations = {item.implementation for item in backends if item.implementation is not None}
    execution_backend = None
    if len(implementations) == 1:
        execution_backend = next(iter(implementations))
    elif len(implementations) > 1:
        execution_backend = "mixed"
    fused_values = {item.fused for item in backends if item.fused is not None}
    fused_backend = next(iter(fused_values)) if len(fused_values) == 1 else None
    supported_packed_backends = frozenset({"batched_mm", "grouped_mm"})
    packed_ablation = (
        capability.weight_layout is ExpertWeightLayout.PACKED_TENSORS
        and backend_discovery.status is ExpertBackendDiscoveryStatus.OBSERVED
        and bool(backends)
        and bool(implementations)
        and implementations <= supported_packed_backends
    )
    enriched = replace(
        capability,
        operations=(InterventionOperation.ABLATE,) if packed_ablation else capability.operations,
        reason=(
            "packed expert contributions can be zeroed at the active Hugging Face backend seam"
            if packed_ablation
            else capability.reason
        ),
        execution_backend=execution_backend,
        fused_backend=fused_backend,
        execution_backends=backends,
        backend_discovery=backend_discovery,
        operation_capabilities=(),
    )
    return replace(
        enriched,
        operation_capabilities=_operation_capabilities(enriched, report),
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
    """Temporary exposed hooks or packed-backend weight masking."""

    def __init__(self, report: DiscoveryReport, *, registry: object | None = None) -> None:
        if type(report) is not DiscoveryReport:
            raise TypeError("report must be an exact DiscoveryReport")
        self._targets = intervention_targets(report)
        self._packed = (
            classify_intervention_capability(report).weight_layout
            is ExpertWeightLayout.PACKED_TENSORS
        )
        self._registry = registry
        self._handles: list[object] = []
        self._invocations: dict[str, int] = {}
        self._backend_snapshot: dict[str, str | None] | None = None
        self._registered_backends: list[str] = []
        self._setter: Any = None

    @property
    def target_inventory(self) -> tuple[ExpertInterventionTarget, ...]:
        return self._targets

    @property
    def invocation_counts(self) -> dict[str, int]:
        return dict(sorted(self._invocations.items()))

    def capture(self, module: object) -> object:
        if self._handles or self._backend_snapshot is not None:
            raise TransformersInterventionError("intervention capability is already active")
        self._invocations = {}
        if self._packed:
            getter = getattr(module, "get_experts_implementation", None)
            setter = getattr(module, "set_experts_implementation", None)
            if not callable(getter) or not callable(setter):
                raise TransformersInterventionError("loaded model cannot switch expert backends")
            snapshot = getter()
            if not isinstance(snapshot, Mapping):
                raise TransformersInterventionError("expert backend snapshot is invalid")
            self._backend_snapshot = dict(snapshot)
            self._setter = setter
            return dict(snapshot)
        return ()

    def apply(self, module: object, recipe: InterventionRecipe) -> None:
        if self._packed:
            self._apply_packed(module, recipe)
            return
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
        if self._packed:
            self._restore_packed(snapshot)
            return
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

    def _apply_packed(self, module: object, recipe: InterventionRecipe) -> None:
        if recipe.operation is not InterventionOperation.ABLATE:
            raise TransformersInterventionError("packed experts currently support ablation only")
        by_label = {target.label: target for target in self._targets}
        unknown = tuple(label for label in recipe.targets if label not in by_label)
        if unknown:
            raise TransformersInterventionError(
                "recipe targets are outside the discovered routed-expert universe"
            )
        named_modules = getattr(module, "named_modules", None)
        if not callable(named_modules):
            raise TransformersInterventionError("loaded model does not expose named_modules()")
        modules = dict(named_modules())
        selected: dict[int, tuple[tuple[str, int], ...]] = {}
        grouped: dict[int, list[tuple[str, int]]] = {}
        for label in recipe.targets:
            target = by_label[label]
            container = modules.get(target.module_path)
            if container is None:
                raise TransformersInterventionError("packed expert container is unavailable")
            grouped.setdefault(id(container), []).append((label, target.expert_index))
        selected = {key: tuple(value) for key, value in grouped.items()}

        registry = self._registry
        if registry is None:
            try:
                registry = getattr(
                    importlib.import_module("transformers.integrations.moe"),
                    "ALL_EXPERTS_FUNCTIONS",
                )
            except Exception as exc:
                raise TransformersInterventionError(
                    "Hugging Face expert backend registry is unavailable"
                ) from exc
        if not callable(getattr(type(registry), "__setitem__", None)) or not callable(
            getattr(type(registry), "__delitem__", None)
        ):
            raise TransformersInterventionError("expert backend registry is not reversible")
        assert self._backend_snapshot is not None and callable(self._setter)
        replacements = dict(self._backend_snapshot)
        implementations = sorted(
            {value for value in replacements.values() if isinstance(value, str) and value}
        )
        for implementation in implementations:
            if implementation not in {"batched_mm", "grouped_mm"}:
                raise TransformersInterventionError(
                    f"expert backend {implementation!r} is not certified for packed ablation"
                )
            try:
                original = registry[implementation]  # type: ignore[index]
            except Exception as exc:
                raise TransformersInterventionError(
                    "active expert backend is absent from the registry"
                ) from exc
            if not callable(original):
                raise TransformersInterventionError("active expert backend is not callable")
            temporary_name = f"moeatlas_ablate_{uuid.uuid4().hex}"

            def ablate_backend(
                expert_module: object,
                hidden_states: object,
                top_k_index: object,
                top_k_weights: object,
                *args: object,
                _original: Any = original,
                **kwargs: object,
            ) -> object:
                adjusted = top_k_weights
                for label, expert_index in selected.get(id(expert_module), ()):
                    try:
                        mask = top_k_index == expert_index
                        any_selected = bool(mask.any().item())
                        adjusted = adjusted.masked_fill(mask, 0.0)
                    except Exception as exc:
                        raise TransformersInterventionError(
                            "expert routing tensors do not support safe contribution masking"
                        ) from exc
                    if any_selected:
                        self._invocations[label] = self._invocations.get(label, 0) + 1
                return _original(
                    expert_module,
                    hidden_states,
                    top_k_index,
                    adjusted,
                    *args,
                    **kwargs,
                )

            registry[temporary_name] = ablate_backend  # type: ignore[index]
            self._registered_backends.append(temporary_name)
            replacements = {
                scope: temporary_name if active == implementation else active
                for scope, active in replacements.items()
            }
        self._registry = registry
        self._setter(replacements)

    def _restore_packed(self, snapshot: object) -> None:
        if not isinstance(snapshot, Mapping) or self._backend_snapshot is None:
            raise TransformersInterventionError("intervention snapshot is invalid")
        failures: list[BaseException] = []
        try:
            self._setter(dict(snapshot))
        except BaseException as exc:
            failures.append(exc)
        while self._registered_backends:
            name = self._registered_backends.pop()
            try:
                del self._registry[name]  # type: ignore[index]
            except BaseException as exc:
                failures.append(exc)
        self._backend_snapshot = None
        self._setter = None
        if failures:
            raise TransformersInterventionError(
                f"packed expert restoration failed ({len(failures)} error(s))"
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
    "ExpertBackendDiscovery",
    "ExpertBackendDiscoveryStatus",
    "ExpertBackendEvidence",
    "ExpertExecutionMode",
    "ExpertInterventionTarget",
    "ExpertOperation",
    "ExpertWeightLayout",
    "InterventionCapabilityReport",
    "InterventionSupportTier",
    "OperationCapability",
    "OperationCapabilityStatus",
    "TransformersExpertInterventionCapability",
    "TransformersInterventionError",
    "intervention_targets",
    "classify_intervention_capability",
    "discover_huggingface_expert_backends",
    "inspect_intervention_capability",
    "parse_intervention_target",
]
