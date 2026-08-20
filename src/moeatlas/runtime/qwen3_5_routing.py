"""Model-free routing decoding for the official Qwen3.5-MoE gate tuple.

The Qwen3.5 gate in the supported Transformers v5.14 surface returns
``(router_logits, router_scores, router_indices)``. This module validates and
normalizes that tuple at the boundary. It intentionally has no dependency on
PyTorch, Transformers, or the Mixtral implementation: family-specific tensor
layouts belong here, while the returned :class:`RoutingEvent` values remain
model-neutral.

The decoder is a passive, single-use-per-router callback. It stores only
freshly validated inspection facts, token identities, and router bindings;
hook modules, callback arguments, and tensor payloads are never retained.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from .. import __version__
from ..adapters import AdapterInspection
from ..core import CapabilityLabel, CaptureSource, ComponentKind
from ..discovery import DiscoverySignal
from ..events import RoutingEvent, TokenEvent
from ..probe import ProbeTarget
from .routing import RoutingCaptureTarget

_ADAPTER_NAME: Final = "huggingface-qwen3.5-moe-static"
_ADAPTER_VERSION: Final = "1.0"
_ADAPTER_FAMILIES: Final = ("qwen3_5_moe",)
_METHOD: Final = "qwen3.5-moe-static-structure-v1"
_LAYOUT: Final = "packed"
_SCORE_TOLERANCE: Final = 1e-6
_DESCRIPTOR_NOTES: Final = (
    "official Transformers v5.14 packed conditional and text surfaces are supported",
    "shared experts are structural and not router targets",
    "structure-only; routing and model certification are not provided",
)
_REPORT_WARNING: Final = "packed expert slices are logical and are not independently hookable"
_PROVENANCE_METADATA: Final = {
    "layout": "packed",
    "evidence": ["config", "topology", "shapes"],
}
_CANDIDATE_EVIDENCE: Final = ((DiscoverySignal.CHILD_STRUCTURE, "Qwen3.5-MoE packed layout", 1.0),)


@dataclass(frozen=True, slots=True)
class _RouterBinding:
    """Canonical identity and routed-expert facts for one Qwen gate."""

    target: ProbeTarget
    layer_key: str
    expert_keys: tuple[str, ...]
    routed_top_k: int


def _fresh_inspection(inspection: object) -> AdapterInspection:
    """Revalidate the caller's inspection and detach it from caller state."""

    if type(inspection) is not AdapterInspection:
        raise TypeError("inspection must be an exact AdapterInspection")
    payload = inspection.model_dump(mode="json")
    fresh = AdapterInspection.model_validate(payload)
    if type(fresh) is not AdapterInspection or fresh is inspection:
        raise TypeError("inspection revalidation returned an unexpected type")
    return fresh


def _fresh_token_events(token_events: object) -> tuple[TokenEvent, ...]:
    """Revalidate immutable token identities without retaining caller objects."""

    if type(token_events) is not tuple:
        raise TypeError("token_events must be an exact tuple")
    if not token_events:
        raise ValueError("token_events must be non-empty")

    fresh_events: list[TokenEvent] = []
    seen_keys: set[str] = set()
    run_key: str | None = None
    sequence_id: str | None = None
    phase: object | None = None
    for value in token_events:
        if type(value) is not TokenEvent:
            raise TypeError("token_events must contain exact TokenEvent objects")
        fresh = TokenEvent.model_validate(value.model_dump(mode="json"))
        if type(fresh) is not TokenEvent or fresh is value:
            raise TypeError("token event revalidation returned an unexpected type")
        if fresh.token_key in seen_keys:
            raise ValueError("token_events must have unique token keys")
        seen_keys.add(fresh.token_key)
        if run_key is None:
            run_key = fresh.run_key
            sequence_id = fresh.sequence_id
            phase = fresh.phase
        elif fresh.run_key != run_key or fresh.phase != phase:
            raise ValueError("token_events must share one run_key and phase")
        elif fresh.sequence_id != sequence_id:
            raise ValueError("token_events must share one sequence_id")
        fresh_events.append(fresh)
    positions = [event.token_pos for event in fresh_events]
    if positions != list(range(len(positions))):
        raise ValueError("token_events must contain one contiguous canonical sequence 0..N-1")
    return tuple(fresh_events)


def _strict_int(value: object, *, field_name: str) -> int:
    if type(value) is not int or isinstance(value, bool):
        raise ValueError(f"{field_name} must be a strict integer")
    return value


def _validate_exact_qwen_report(inspection: AdapterInspection) -> None:
    """Reject reports that are valid schemas but not this exact Qwen seam."""

    descriptor = inspection.descriptor
    if (
        descriptor.name != _ADAPTER_NAME
        or descriptor.version != _ADAPTER_VERSION
        or descriptor.architecture_families != _ADAPTER_FAMILIES
        or descriptor.compatibility_notes != _DESCRIPTOR_NOTES
    ):
        raise ValueError("inspection descriptor is not the exact Qwen3.5 static descriptor")

    detection = inspection.detection
    allowed_detection_evidence = {
        tuple(
            sorted(
                (
                    "architecture:qwen3.5-conditional-allowlist",
                    "config:strict-fields-and-schedule",
                    "model_type:qwen3_5_moe",
                    "shapes:exact",
                    "topology:packed-shared-expert",
                )
            )
        ),
        tuple(
            sorted(
                (
                    "architecture:qwen3.5-text-allowlist",
                    "config:strict-fields-and-schedule",
                    "model_type:qwen3_5_moe_text",
                    "shapes:exact",
                    "topology:packed-shared-expert",
                )
            )
        ),
    }
    if (
        type(detection.score) is not float
        or detection.score != 1.0
        or detection.warnings != ()
        or tuple(detection.evidence) not in allowed_detection_evidence
    ):
        raise ValueError("Qwen3.5 detection evidence is not exact")

    report = inspection.report
    if report.model_manifest.architecture != "qwen3_5_moe":
        raise ValueError("Qwen3.5 report manifest has the wrong architecture")
    if report.scanner_version != __version__:
        raise ValueError("Qwen3.5 report scanner version is not exact")
    if report.warnings != [_REPORT_WARNING]:
        raise ValueError("Qwen3.5 report warnings are not exact")
    facts = report.facts
    expert_count = _strict_int(facts.expert_count, field_name="expert_count")
    routed_top_k = _strict_int(facts.routed_top_k, field_name="routed_top_k")
    shared_count = _strict_int(facts.shared_expert_count, field_name="shared_expert_count")
    if (
        facts.expert_count_source != "config.num_experts"
        or facts.routed_top_k_source != "config.num_experts_per_tok"
        or facts.shared_expert_count_source != "topology.shared_expert"
        or shared_count != 1
        or expert_count <= 0
        or routed_top_k <= 0
        or routed_top_k > expert_count
    ):
        raise ValueError("Qwen3.5 report facts are not exact")

    components = report.components
    candidates = report.candidates
    if len(candidates) != len(components) or any(
        candidate.component_key != component.component_key
        for candidate, component in zip(candidates, components, strict=False)
    ):
        raise ValueError("Qwen3.5 candidates and components are not canonically paired")
    expected_kinds = {
        ComponentKind.MOE_LAYER,
        ComponentKind.ROUTER,
        ComponentKind.EXPERT_CONTAINER,
        ComponentKind.EXPERT,
        ComponentKind.SHARED_EXPERT,
    }
    kind_order = {
        ComponentKind.MOE_LAYER: 0,
        ComponentKind.ROUTER: 1,
        ComponentKind.EXPERT_CONTAINER: 2,
        ComponentKind.EXPERT: 3,
        ComponentKind.SHARED_EXPERT: 4,
    }
    expected_layers: set[int] = set()
    router_layers: set[int] = set()
    for candidate, component in zip(candidates, components, strict=True):
        if component.kind not in expected_kinds:
            raise ValueError("Qwen3.5 report contains an unsupported component kind")
        if component.capabilities != [CapabilityLabel.STRUCTURE]:
            raise ValueError("Qwen3.5 components must be STRUCTURE-only")
        capture = component.capture
        if capture is None or (
            capture.source is not CaptureSource.STATIC_STRUCTURE
            or capture.method != _METHOD
            or capture.adapter != _ADAPTER_NAME
            or capture.adapter_version != _ADAPTER_VERSION
            or capture.verified is not False
            or capture.metadata != {"layout": _LAYOUT}
        ):
            raise ValueError("Qwen3.5 component capture provenance is not exact")
        provenance = component.provenance
        if provenance is None or (
            provenance.source != _METHOD
            or provenance.tool_version != __version__
            or provenance.metadata != _PROVENANCE_METADATA
        ):
            raise ValueError("Qwen3.5 component provenance is not exact")
        if (
            candidate.model_key != report.model_key
            or candidate.confidence != 1.0
            or tuple((item.signal, item.detail, item.weight) for item in candidate.evidence)
            != _CANDIDATE_EVIDENCE
            or candidate.routed != component.routed
            or candidate.shared != component.shared
            or candidate.warnings != component.warnings
        ):
            raise ValueError("Qwen3.5 candidate evidence is not exact")
        expected_routed, expected_shared = {
            ComponentKind.EXPERT: (True, False),
            ComponentKind.SHARED_EXPERT: (False, True),
        }.get(component.kind, (None, None))
        if component.routed != expected_routed or component.shared != expected_shared:
            raise ValueError("Qwen3.5 component routed/shared flags are not exact")
        if component.kind is ComponentKind.EXPERT:
            expected_warning = [_REPORT_WARNING]
        else:
            expected_warning = []
        if component.warnings != expected_warning:
            raise ValueError("Qwen3.5 component warnings are not exact")
        layer_index = component.layer_index
        if type(layer_index) is not int or isinstance(layer_index, bool) or layer_index < 0:
            raise ValueError("Qwen3.5 components must have numeric layer indices")
        expected_layers.add(layer_index)
        if component.kind is ComponentKind.ROUTER:
            router_layers.add(layer_index)

    if not router_layers or router_layers != set(range(len(router_layers))):
        raise ValueError("Qwen3.5 router layers must be contiguous and zero-based")
    if expected_layers != router_layers:
        raise ValueError("Qwen3.5 components must cover exactly the router layers")
    expected_order = sorted(
        components,
        key=lambda component: (
            component.layer_index,
            kind_order[component.kind],
            component.expert_index if component.expert_index is not None else -1,
        ),
    )
    if components != expected_order:
        raise ValueError("Qwen3.5 components are not in canonical layer order")
    for layer_index in sorted(router_layers):
        layer_components = [
            component for component in components if component.layer_index == layer_index
        ]
        counts = {
            kind: sum(component.kind is kind for component in layer_components)
            for kind in expected_kinds
        }
        if counts != {
            ComponentKind.MOE_LAYER: 1,
            ComponentKind.ROUTER: 1,
            ComponentKind.EXPERT_CONTAINER: 1,
            ComponentKind.EXPERT: expert_count,
            ComponentKind.SHARED_EXPERT: shared_count,
        }:
            raise ValueError("Qwen3.5 report has an incomplete layer topology")
        experts = sorted(
            (component for component in layer_components if component.kind is ComponentKind.EXPERT),
            key=lambda component: component.expert_index,
        )
        if [component.expert_index for component in experts] != list(range(expert_count)):
            raise ValueError("Qwen3.5 expert indices must be contiguous and zero-based")


def _inspection_bindings(
    inspection: AdapterInspection,
) -> tuple[_RouterBinding, ...]:
    """Validate the Qwen3.5 static report and build router bindings.

    Packed expert slices are represented by one module path but have one
    component key per logical expert. The shared expert is deliberately
    checked and then excluded from every binding.
    """

    _validate_exact_qwen_report(inspection)

    facts = inspection.report.facts
    expert_count = _strict_int(facts.expert_count, field_name="expert_count")
    routed_top_k = _strict_int(facts.routed_top_k, field_name="routed_top_k")
    shared_count = _strict_int(facts.shared_expert_count, field_name="shared_expert_count")
    if expert_count <= 0 or routed_top_k <= 0 or routed_top_k > expert_count:
        raise ValueError("inspection routing facts have an invalid expert count or top-k")
    if shared_count != 1:
        raise ValueError("Qwen3.5 inspection must contain exactly one shared expert per layer")

    components = inspection.report.components
    routers = [component for component in components if component.kind is ComponentKind.ROUTER]
    if not routers:
        raise ValueError("inspection must contain at least one router")

    bindings: list[_RouterBinding] = []
    seen_paths: set[str] = set()
    seen_layers: set[int] = set()
    for router in routers:
        capture = router.capture
        if capture is None or (
            capture.source is not CaptureSource.STATIC_STRUCTURE
            or capture.method != _METHOD
            or capture.adapter != _ADAPTER_NAME
            or capture.adapter_version != _ADAPTER_VERSION
            or capture.verified is not False
            or capture.metadata.keys() != {"layout"}
            or capture.metadata.get("layout") != _LAYOUT
        ):
            raise ValueError("router capture provenance is not exact Qwen3.5 metadata")
        layer_index = _strict_int(router.layer_index, field_name="router layer_index")
        if layer_index < 0:
            raise ValueError("router layer_index must be non-negative")
        if router.module_path in seen_paths or layer_index in seen_layers:
            raise ValueError("Qwen3.5 router paths and layers must be unique")
        seen_paths.add(router.module_path)
        seen_layers.add(layer_index)
        if not router.module_path.endswith(".mlp.gate"):
            raise ValueError("Qwen3.5 router module path must end with .mlp.gate")

        same_layer = [
            component
            for component in components
            if component.kind is ComponentKind.MOE_LAYER and component.layer_index == layer_index
        ]
        if len(same_layer) != 1:
            raise ValueError("router must bind exactly one same-layer MoE component")
        layer = same_layer[0]
        if not layer.module_path.endswith(".mlp"):
            raise ValueError("Qwen3.5 MoE layer path must end with .mlp")
        layer_root = layer.module_path[: -len(".mlp")]
        if not layer_root:
            raise ValueError("Qwen3.5 MoE layer must have a canonical root")
        expected_router_path = f"{layer_root}.mlp.gate"
        if router.module_path != expected_router_path:
            raise ValueError("router module path does not match the canonical layer root")

        containers = [
            component
            for component in components
            if component.kind is ComponentKind.EXPERT_CONTAINER
            and component.layer_index == layer_index
        ]
        if len(containers) != 1:
            raise ValueError("router must bind exactly one packed expert container")
        container = containers[0]
        expected_container_path = f"{layer_root}.mlp.experts"
        if container.module_path != expected_container_path:
            raise ValueError("expert container path does not match the canonical layer root")

        shared = [
            component
            for component in components
            if component.kind is ComponentKind.SHARED_EXPERT
            and component.layer_index == layer_index
        ]
        if len(shared) != shared_count:
            raise ValueError("router must bind exactly one same-layer shared expert")
        shared_component = shared[0]
        if (
            shared_component.shared is not True
            or shared_component.routed is not False
            or shared_component.expert_index is not None
            or shared_component.module_path != f"{layer_root}.mlp.shared_expert"
        ):
            raise ValueError("shared expert does not match the canonical layer root")

        experts = [
            component
            for component in components
            if component.kind is ComponentKind.EXPERT and component.layer_index == layer_index
        ]
        if len(experts) != expert_count:
            raise ValueError("router same-layer expert count does not match inspection facts")
        if any(component.routed is not True or component.shared is True for component in experts):
            raise ValueError("router experts must be routed and non-shared")
        indices = [component.expert_index for component in experts]
        if any(type(index) is not int or isinstance(index, bool) for index in indices):
            raise ValueError("router experts must have strict indices")
        if sorted(indices) != list(range(expert_count)):
            raise ValueError("router expert indices must be contiguous and zero-based")
        if any(component.module_path != expected_container_path for component in experts):
            raise ValueError("Qwen3.5 packed experts must match the canonical layer root")
        experts.sort(key=lambda component: component.expert_index)

        target = ProbeTarget(
            module_path=router.module_path,
            component_key=router.component_key,
            component_kind=ComponentKind.ROUTER,
        )
        bindings.append(
            _RouterBinding(
                target=target,
                layer_key=layer.component_key,
                expert_keys=tuple(component.component_key for component in experts),
                routed_top_k=routed_top_k,
            )
        )

    bindings.sort(key=lambda binding: binding.target.module_path)
    return tuple(bindings)


def _as_float_rows(value: object, *, field_name: str) -> list[list[float]]:
    """Detach, move to CPU, cast to float, and materialize a 2-D value."""

    try:
        converted = value.detach().cpu().float().tolist()  # type: ignore[attr-defined]
    except AttributeError as exc:
        raise TypeError(f"{field_name} must be tensor-like") from exc
    if type(converted) is not list:
        raise TypeError(f"{field_name} must convert to an exact list")
    rows: list[list[float]] = []
    for row in converted:
        if type(row) is not list:
            raise TypeError(f"{field_name} must have exact two-dimensional list shape")
        checked: list[float] = []
        for item in row:
            if type(item) is not float or not math.isfinite(item):
                raise ValueError(f"{field_name} values must be finite strict floats")
            checked.append(item)
        rows.append(checked)
    return rows


def _as_index_rows(value: object) -> list[list[int]]:
    """Detach, move to CPU, and materialize strict integer indices."""

    try:
        converted = value.detach().cpu().tolist()  # type: ignore[attr-defined]
    except AttributeError as exc:
        raise TypeError("router_indices must be tensor-like") from exc
    if type(converted) is not list:
        raise TypeError("router_indices must convert to an exact list")
    rows: list[list[int]] = []
    for row in converted:
        if type(row) is not list:
            raise TypeError("router_indices must have exact two-dimensional list shape")
        checked: list[int] = []
        for item in row:
            if type(item) is not int or isinstance(item, bool):
                raise ValueError("router_indices must contain strict integers")
            checked.append(item)
        rows.append(checked)
    return rows


def _softmax(values: list[float]) -> list[float]:
    """Compute a stable finite softmax from already validated logits."""

    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    total = sum(exponentials)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("router logits do not produce a finite softmax")
    return [value / total for value in exponentials]


def _top_k(values: list[float], count: int) -> list[int]:
    """Return deterministic index order while rejecting ambiguous ties."""

    ordered = sorted(range(len(values)), key=lambda index: (-values[index], index))
    selected_values = [values[index] for index in ordered[:count]]
    selected_tie = len(set(selected_values)) != len(selected_values)
    cutoff_tie = count < len(values) and values[ordered[count - 1]] == values[ordered[count]]
    if selected_tie or cutoff_tie:
        raise ValueError("router logits contain an ambiguous selected/cutoff tie")
    return ordered[:count]


def _check_shape(
    rows: list[list[object]], expected_rows: int, expected_columns: int, name: str
) -> None:
    if len(rows) != expected_rows or any(len(row) != expected_columns for row in rows):
        raise ValueError(f"{name} must have exact shape [{expected_rows},{expected_columns}]")


class Qwen3_5RoutingDecoder:
    """Decode one exact, single-use Qwen3.5 packed router capture per path."""

    __slots__ = ("_bindings", "_token_events", "_used_paths")

    def __init__(
        self,
        inspection: AdapterInspection,
        token_events: tuple[TokenEvent, ...],
    ) -> None:
        fresh_inspection = _fresh_inspection(inspection)
        fresh_token_events = _fresh_token_events(token_events)
        bindings = _inspection_bindings(fresh_inspection)
        self._bindings = {binding.target.module_path: binding for binding in bindings}
        self._token_events = fresh_token_events
        self._used_paths: set[str] = set()

    def __call__(
        self,
        context: RoutingCaptureTarget,
        module: object,
        inputs: tuple[object, ...],
        output: object,
    ) -> tuple[RoutingEvent, ...]:
        del module
        if type(context) is not RoutingCaptureTarget:
            raise TypeError("context must be an exact RoutingCaptureTarget")
        if type(inputs) is not tuple:
            raise TypeError("inputs must be an exact tuple")

        path = context.router.module_path
        binding = self._bindings.get(path)
        if binding is None:
            raise ValueError("context router path is not bound to this inspection")
        if (
            context.router != binding.target
            or context.layer_key != binding.layer_key
            or context.expert_keys != binding.expert_keys
            or context.routed_top_k != binding.routed_top_k
        ):
            raise ValueError("context does not match the exact router binding")
        if path in self._used_paths:
            raise RuntimeError("router decoder invocation is single-use per router")

        token_count = len(self._token_events)
        expert_count = len(binding.expert_keys)
        top_k = binding.routed_top_k
        if type(output) is not tuple or len(output) != 3:
            raise TypeError(
                "Qwen3.5 router output must be an exact "
                "(router_logits, router_scores, router_indices) tuple"
            )

        logits = _as_float_rows(output[0], field_name="router_logits")
        scores = _as_float_rows(output[1], field_name="router_scores")
        indices = _as_index_rows(output[2])
        _check_shape(logits, token_count, expert_count, "router_logits")
        _check_shape(scores, token_count, top_k, "router_scores")
        _check_shape(indices, token_count, top_k, "router_indices")

        events: list[RoutingEvent] = []
        for token_event, logit_row, score_row, index_row in zip(
            self._token_events, logits, scores, indices, strict=True
        ):
            selected = _top_k(logit_row, top_k)
            if len(set(index_row)) != len(index_row) or any(
                index < 0 or index >= expert_count for index in index_row
            ):
                raise ValueError("router_indices must be unique and in range")
            if index_row != selected:
                raise ValueError("router_indices do not match deterministic top-k")
            if any(score < 0.0 or score > 1.0 for score in score_row):
                raise ValueError("router_scores must be within [0, 1]")
            if any(left < right for left, right in zip(score_row, score_row[1:])):
                raise ValueError("router_scores must be non-increasing in native top-k order")
            if abs(sum(score_row) - 1.0) > _SCORE_TOLERANCE:
                raise ValueError("router_scores must sum to one within tolerance")
            probabilities = _softmax(logit_row)
            selected_total = sum(probabilities[index] for index in selected)
            expected_scores = [probabilities[index] / selected_total for index in selected]
            if any(
                not math.isclose(
                    score,
                    expected,
                    rel_tol=_SCORE_TOLERANCE,
                    abs_tol=_SCORE_TOLERANCE,
                )
                for score, expected in zip(score_row, expected_scores, strict=True)
            ):
                raise ValueError("router_scores do not match softmax top-k renormalization")
            for rank, (expert_index, score) in enumerate(zip(index_row, score_row, strict=True)):
                events.append(
                    RoutingEvent(
                        token_key=token_event.token_key,
                        layer_key=binding.layer_key,
                        rank=rank,
                        expert_key=binding.expert_keys[expert_index],
                        router_logit=logit_row[expert_index],
                        probability=None,
                        weight=score,
                        selected=True,
                    )
                )

        result = tuple(events)
        self._used_paths.add(path)
        return result


__all__ = ["Qwen3_5RoutingDecoder"]
