"""Strict, model-runtime-independent Mixtral routing decoding.

The decoder is intentionally a small boundary around a router hook callback.
It accepts only the exact static Mixtral inspection, exact token rows, and
opaque tensor-like values supplied by the caller.  It never imports a tensor
runtime, retains hook payloads, or infers token identities from model input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from ..adapters import AdapterInspection
from ..core import CaptureSource, ComponentKind
from ..events import RoutingEvent, TokenEvent
from ..probe import ProbeTarget
from .capabilities import RouterPayloadShape, ScoreSemantics
from .routing import RoutingCaptureTarget

_ADAPTER_NAME: Final = "huggingface-mixtral-static"
_ADAPTER_VERSION: Final = "1.0"
_ADAPTER_FAMILIES: Final = ("mixtral",)
_LAYOUT_LEGACY: Final = "legacy_indexed"
_LAYOUT_PACKED: Final = "packed"
_LAYOUTS: Final = frozenset({_LAYOUT_LEGACY, _LAYOUT_PACKED})
_SCORE_TOLERANCE: Final = 1e-6


@dataclass(frozen=True, slots=True)
class _RouterBinding:
    """Canonical identity and routing facts for one router hook."""

    target: ProbeTarget
    layer_key: str
    expert_keys: tuple[str, ...]
    routed_top_k: int


def _fresh_inspection(inspection: object) -> AdapterInspection:
    if type(inspection) is not AdapterInspection:
        raise TypeError("inspection must be an exact AdapterInspection")
    payload = inspection.model_dump(mode="json")
    fresh = AdapterInspection.model_validate(payload)
    if type(fresh) is not AdapterInspection or fresh is inspection:
        raise TypeError("inspection revalidation returned an unexpected type")
    return fresh


def _fresh_token_events(token_events: object) -> tuple[TokenEvent, ...]:
    if type(token_events) is not tuple:
        raise TypeError("token_events must be an exact tuple")
    if not token_events:
        raise ValueError("token_events must be non-empty")

    fresh_events: list[TokenEvent] = []
    seen_keys: set[str] = set()
    run_key: str | None = None
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
            phase = fresh.phase
        elif fresh.run_key != run_key or fresh.phase != phase:
            raise ValueError("token_events must share one run_key and phase")
        fresh_events.append(fresh)
    return tuple(fresh_events)


def _inspection_layout_and_bindings(
    inspection: AdapterInspection,
) -> tuple[str, tuple[_RouterBinding, ...]]:
    descriptor = inspection.descriptor
    if (
        descriptor.name != _ADAPTER_NAME
        or descriptor.version != _ADAPTER_VERSION
        or descriptor.architecture_families != _ADAPTER_FAMILIES
    ):
        raise ValueError("inspection descriptor is not the exact Mixtral static descriptor")

    facts = inspection.report.facts
    expert_count = facts.expert_count
    routed_top_k = facts.routed_top_k
    if type(expert_count) is not int or isinstance(expert_count, bool):
        raise ValueError("inspection must provide a strict expert_count")
    if type(routed_top_k) is not int or isinstance(routed_top_k, bool):
        raise ValueError("inspection must provide a strict routed_top_k")
    if expert_count <= 0 or routed_top_k <= 0 or routed_top_k > expert_count:
        raise ValueError("inspection routing facts have an invalid expert count or top-k")

    components = inspection.report.components
    routers = [component for component in components if component.kind is ComponentKind.ROUTER]
    if not routers:
        raise ValueError("inspection must contain at least one router")

    layouts: list[str] = []
    bindings: list[_RouterBinding] = []
    seen_paths: set[str] = set()
    for router in routers:
        if router.capture is None:
            raise ValueError("router is missing capture provenance")
        capture = router.capture
        if (
            capture.source is not CaptureSource.STATIC_STRUCTURE
            or capture.method != "mixtral-static-structure-v1"
            or capture.adapter != _ADAPTER_NAME
            or capture.adapter_version != _ADAPTER_VERSION
            or capture.verified is not False
            or capture.metadata.keys() != {"layout"}
        ):
            raise ValueError("router capture provenance is not exact Mixtral metadata")
        layout = capture.metadata.get("layout")
        if type(layout) is not str or layout not in _LAYOUTS:
            raise ValueError("router capture metadata must name legacy_indexed or packed")
        layouts.append(layout)

        if router.layer_index is None or type(router.layer_index) is not int:
            raise ValueError("router must have a strict layer index")
        if router.module_path in seen_paths:
            raise ValueError("router module paths must be unique")
        seen_paths.add(router.module_path)

        same_layer = [
            component
            for component in components
            if component.kind is ComponentKind.MOE_LAYER
            and component.layer_index == router.layer_index
        ]
        if len(same_layer) != 1:
            raise ValueError("router must bind exactly one same-layer MoE component")

        experts = [
            component
            for component in components
            if component.kind is ComponentKind.EXPERT
            and component.layer_index == router.layer_index
        ]
        if len(experts) != expert_count:
            raise ValueError("router same-layer expert count does not match inspection facts")
        if any(component.routed is not True or component.shared is True for component in experts):
            raise ValueError("router experts must be routed and non-shared")
        indices = [component.expert_index for component in experts]
        if any(type(index) is not int for index in indices):
            raise ValueError("router experts must have strict indices")
        if sorted(indices) != list(range(expert_count)):
            raise ValueError("router expert indices must be contiguous and zero-based")
        experts.sort(key=lambda component: component.expert_index)

        target = ProbeTarget(
            module_path=router.module_path,
            component_key=router.component_key,
            component_kind=ComponentKind.ROUTER,
        )
        bindings.append(
            _RouterBinding(
                target=target,
                layer_key=same_layer[0].component_key,
                expert_keys=tuple(component.component_key for component in experts),
                routed_top_k=routed_top_k,
            )
        )

    if len(set(layouts)) != 1:
        raise ValueError("all router capture metadata must use one consistent layout")
    bindings.sort(key=lambda binding: binding.target.module_path)
    return layouts[0], tuple(bindings)


def _as_float_rows(value: object, *, field_name: str) -> list[list[float]]:
    """Detach, move to CPU, cast to float, and convert one tensor-like value."""

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
    """Detach, move to CPU, and convert one integer-index tensor-like value."""

    try:
        converted = value.detach().cpu().tolist()  # type: ignore[attr-defined]
    except AttributeError as exc:
        raise TypeError("indices must be tensor-like") from exc
    if type(converted) is not list:
        raise TypeError("indices must convert to an exact list")
    rows: list[list[int]] = []
    for row in converted:
        if type(row) is not list:
            raise TypeError("indices must have exact two-dimensional list shape")
        checked: list[int] = []
        for item in row:
            if type(item) is not int:
                raise ValueError("indices must contain strict integers")
            checked.append(item)
        rows.append(checked)
    return rows


def _softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def _top_k(values: list[float], count: int) -> list[int]:
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


class MixtralRoutingDecoder:
    """Decode one exact, single-use Mixtral router capture per router path."""

    __slots__ = ("_bindings", "_inspection_layout", "_token_events", "_used_paths")

    @property
    def payload_shape(self) -> RouterPayloadShape:
        """The declared raw payload vocabulary of this decoder's layout."""

        if self._inspection_layout == _LAYOUT_LEGACY:
            return RouterPayloadShape.LOGITS_TUPLE
        return RouterPayloadShape.SCORES_INDICES_TUPLE

    @property
    def score_semantics(self) -> ScoreSemantics:
        """Emitted rows always carry observed router logits."""

        return ScoreSemantics.LOGITS

    def __init__(
        self,
        inspection: AdapterInspection,
        token_events: tuple[TokenEvent, ...],
    ) -> None:
        fresh_inspection = _fresh_inspection(inspection)
        fresh_token_events = _fresh_token_events(token_events)
        layout, bindings = _inspection_layout_and_bindings(fresh_inspection)
        self._inspection_layout = layout
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
        events: list[RoutingEvent] = []

        if self._inspection_layout == _LAYOUT_LEGACY:
            logits = _as_float_rows(output, field_name="legacy router output")
            _check_shape(logits, token_count, expert_count, "legacy router output")
            for token_event, row in zip(self._token_events, logits, strict=True):
                _softmax(row)
                selected = _top_k(row, top_k)
                for rank, expert_index in enumerate(selected):
                    events.append(
                        RoutingEvent(
                            token_key=token_event.token_key,
                            layer_key=binding.layer_key,
                            rank=rank,
                            expert_key=binding.expert_keys[expert_index],
                            router_logit=row[expert_index],
                            probability=None,
                            weight=None,
                            selected=True,
                        )
                    )
        else:
            if type(output) is not tuple or len(output) != 3:
                raise TypeError(
                    "packed router output must be an exact (logits, scores, indices) tuple"
                )
            logits = _as_float_rows(output[0], field_name="packed router logits")
            scores = _as_float_rows(output[1], field_name="packed router scores")
            indices = _as_index_rows(output[2])
            _check_shape(logits, token_count, expert_count, "packed router logits")
            _check_shape(scores, token_count, top_k, "packed router scores")
            _check_shape(indices, token_count, top_k, "packed router indices")
            for token_event, logit_row, score_row, index_row in zip(
                self._token_events, logits, scores, indices, strict=True
            ):
                selected = _top_k(logit_row, top_k)
                if len(set(index_row)) != len(index_row) or any(
                    index < 0 or index >= expert_count for index in index_row
                ):
                    raise ValueError("packed router indices must be unique and in range")
                if index_row != selected:
                    raise ValueError("packed router indices do not match deterministic top-k")
                if any(score < 0.0 or score > 1.0 for score in score_row):
                    raise ValueError("packed router scores must be within [0, 1]")
                if abs(sum(score_row) - 1.0) > _SCORE_TOLERANCE:
                    raise ValueError("packed router scores must sum to one within tolerance")
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
                    raise ValueError(
                        "packed router scores do not match softmax top-k renormalization"
                    )
                for rank, (expert_index, score) in enumerate(
                    zip(index_row, score_row, strict=True)
                ):
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


__all__ = ["MixtralRoutingDecoder"]
