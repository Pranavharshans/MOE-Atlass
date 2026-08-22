"""Structure-driven, model-family-agnostic routing capture composition.

This module closes the foreign-family capture gap documented in
``docs/runtime.md``: the certified :class:`~moeatlas.runtime.RoutingCaptureSession`
requires an exact ``AdapterInspection``, so unknown architectures could only be
observed through caller-owned hooks. Here that caller-side pattern becomes a
product seam. Router modules are discovered from a static ``[STRUCTURE]``
:class:`~moeatlas.discovery.DiscoveryReport`, hooks are attached through the
existing passive :class:`~moeatlas.probe.HookManager`, and router payloads are
decoded generically from the published expert count and routed top-k — with no
adapter name, module-path convention, or certified descriptor anywhere.

Score normalization is driven by the model config where determinable (a
``score_function`` of ``softmax`` or ``sigmoid``); when it is not determinable,
raw logits are emitted and a capability note is returned alongside the events.
The module imports no model stack and downloads nothing; real checkpoint and
payload equivalence remain deferred to the final VM phase.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ..core import ComponentKind, parse_component_key
from ..discovery import DiscoveryReport
from ..event_validation import fresh_routing_events, fresh_token_events
from ..events import RoutingEvent, TokenEvent
from ..probe import (
    CaptureMode,
    CapturePolicy,
    HookBinding,
    HookManager,
    HookPoint,
    ProbeLevel,
    ProbePlan,
    ProbeTarget,
    ReductionPolicy,
)

_STAGES = frozenset({"preflight", "resolution", "decode", "events", "lifecycle"})

_CONFIG_SCORE_KEYS = ("score_function", "routing_norm")


class StructuredCaptureError(RuntimeError):
    """Safe fixed-stage failure for structure-driven routing capture."""

    def __init__(self, stage: str, message: str | None = None) -> None:
        if stage not in _STAGES:
            raise ValueError("structured capture error stage is not supported")
        self.stage = stage
        if message is None:
            super().__init__(f"structured routing capture failed at {stage}")
        else:
            super().__init__(f"structured routing capture failed at {stage}: {message}")


@dataclass(frozen=True, slots=True)
class StructuredRouterTarget:
    """One generic hook target derived purely from structure evidence."""

    module_path: str
    component_key: str
    layer_key: str
    expert_keys: tuple[str, ...]
    routed_top_k: int
    layer_index: int

    def __post_init__(self) -> None:
        for name in ("module_path", "component_key", "layer_key"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise TypeError(f"{name} must be a non-empty string")
        parse_component_key(self.component_key)
        parse_component_key(self.layer_key)
        if type(self.expert_keys) is not tuple or not self.expert_keys:
            raise ValueError("expert_keys must be a non-empty tuple")
        for key in self.expert_keys:
            if type(key) is not str:
                raise TypeError("expert_keys must contain strings")
            parse_component_key(key)
        if len(set(self.expert_keys)) != len(self.expert_keys):
            raise ValueError("expert_keys must be unique")
        for name in ("routed_top_k", "layer_index"):
            value = getattr(self, name)
            if type(value) is not int or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.routed_top_k <= 0 or self.routed_top_k > len(self.expert_keys):
            raise ValueError("routed_top_k must be positive within the expert count")


def _fresh_report(value: object) -> DiscoveryReport:
    if type(value) is not DiscoveryReport:
        raise TypeError("report must be an exact DiscoveryReport")
    fresh = DiscoveryReport.model_validate(value.model_dump(mode="json"))
    if type(fresh) is not DiscoveryReport or fresh is value:
        raise TypeError("structure report revalidation returned an unexpected type")
    return fresh


def structured_router_targets(report: DiscoveryReport) -> tuple[StructuredRouterTarget, ...]:
    """Resolve generic hook targets from one static structure report.

    Every router candidate contributes exactly one target bound to its
    same-index MoE layer key and its contiguous zero-based routed experts.
    The report's strict ``expert_count``/``routed_top_k`` facts drive every
    count; shared-expert components are excluded by kind, never by name.
    """

    fresh = _fresh_report(report)
    facts = fresh.facts
    expert_count = facts.expert_count
    routed_top_k = facts.routed_top_k
    if type(expert_count) is not int or type(routed_top_k) is not int:
        raise StructuredCaptureError(
            "resolution",
            "structure report does not publish strict expert_count and routed_top_k facts",
        )
    if expert_count <= 0 or routed_top_k <= 0 or routed_top_k > expert_count:
        raise StructuredCaptureError(
            "resolution",
            "structure report top-k must be positive within the expert count",
        )

    components = fresh.components
    routers = [
        component
        for component in components
        if component.kind is ComponentKind.ROUTER and component.layer_index is not None
    ]
    if not routers:
        raise StructuredCaptureError(
            "resolution", "structure report does not publish any routed router"
        )
    targets: list[StructuredRouterTarget] = []
    seen_layers: set[int] = set()
    for router in sorted(routers, key=lambda item: item.module_path):
        layer_index = router.layer_index
        if layer_index in seen_layers:
            raise StructuredCaptureError(
                "resolution", f"layer {layer_index} publishes more than one router"
            )
        seen_layers.add(layer_index)
        layers = [
            component
            for component in components
            if component.kind is ComponentKind.MOE_LAYER and component.layer_index == layer_index
        ]
        if len(layers) != 1:
            raise StructuredCaptureError(
                "resolution",
                f"router on layer {layer_index} must bind exactly one MoE layer",
            )
        experts = [
            component
            for component in components
            if component.kind is ComponentKind.EXPERT and component.layer_index == layer_index
        ]
        if len(experts) != expert_count:
            raise StructuredCaptureError(
                "resolution",
                f"layer {layer_index} publishes {len(experts)} of {expert_count} routed experts",
            )
        indices = [component.expert_index for component in experts]
        if any(type(index) is not int or isinstance(index, bool) for index in indices) or sorted(
            indices  # type: ignore[type-var]
        ) != list(range(expert_count)):
            raise StructuredCaptureError(
                "resolution",
                f"layer {layer_index} expert indices must be contiguous and zero-based",
            )
        experts.sort(key=lambda component: component.expert_index)  # type: ignore[arg-type, return-value]
        targets.append(
            StructuredRouterTarget(
                module_path=router.module_path,
                component_key=router.component_key,
                layer_key=layers[0].component_key,
                expert_keys=tuple(component.component_key for component in experts),
                routed_top_k=routed_top_k,
                layer_index=layer_index,
            )
        )
    return tuple(targets)


def _materialize_floats(value: object) -> list[list[float]]:
    """Convert a tensor-like matrix via detach -> cpu -> float -> tolist."""

    current = value
    for method_name in ("detach", "cpu", "float"):
        method = getattr(current, method_name, None)
        current = method() if callable(method) else current
    tolist = getattr(current, "tolist", None)
    rows = tolist() if callable(tolist) else current
    return _validate_float_matrix(rows)


def _materialize_ints(value: object) -> list[list[int]]:
    """Convert a tensor-like integer matrix via detach -> cpu -> tolist."""

    current = value
    for method_name in ("detach", "cpu"):
        method = getattr(current, method_name, None)
        current = method() if callable(method) else current
    tolist = getattr(current, "tolist", None)
    rows = tolist() if callable(tolist) else current
    return _validate_int_matrix(rows)


def _validate_float_matrix(rows: object) -> list[list[float]]:
    if type(rows) is not list or not rows or type(rows[0]) is not list:
        raise StructuredCaptureError("decode", "router payload must be a non-empty matrix")
    width = len(rows[0])
    matrix: list[list[float]] = []
    for row in rows:
        if type(row) is not list or len(row) != width or not row:
            raise StructuredCaptureError("decode", "router payload rows must share one width")
        values: list[float] = []
        for entry in row:
            if type(entry) is not float or not math.isfinite(entry):
                raise StructuredCaptureError("decode", "router payload entries must be finite")
            values.append(entry)
        matrix.append(values)
    return matrix


def _validate_int_matrix(rows: object) -> list[list[int]]:
    if type(rows) is not list or not rows or type(rows[0]) is not list:
        raise StructuredCaptureError("decode", "index payload must be a non-empty matrix")
    width = len(rows[0])
    matrix: list[list[int]] = []
    for row in rows:
        if type(row) is not list or len(row) != width or not row:
            raise StructuredCaptureError("decode", "index payload rows must share one width")
        for entry in row:
            if type(entry) is not int or isinstance(entry, bool) or entry < 0:
                raise StructuredCaptureError(
                    "decode", "index payload entries must be non-negative integers"
                )
        matrix.append(row)  # type: ignore[arg-type]
    return matrix


def _config_field(config: object, name: str) -> Any:
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get(name)
    try:
        return getattr(config, name, None)
    except Exception:
        return None


def _score_normalization(config: object) -> str | None:
    """Return 'softmax', 'sigmoid', or None when config does not decide."""

    for key in _CONFIG_SCORE_KEYS:
        value = _config_field(config, key)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if "sigmoid" in normalized:
                return "sigmoid"
            if "softmax" in normalized:
                return "softmax"
    return None


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    inverse = math.exp(value)
    return inverse / (1.0 + inverse)


def _softmax_row(values: list[float]) -> list[float]:
    peak = max(values)
    exponents = [math.exp(value - peak) for value in values]
    total = sum(exponents)
    return [exponent / total for exponent in exponents]


def decode_structured_payload(
    payload: object,
    *,
    target: StructuredRouterTarget,
    token_events: tuple[TokenEvent, ...],
    config: object = None,
) -> tuple[tuple[RoutingEvent, ...], str | None]:
    """Decode one opaque router hook payload into generic top-k event rows.

    Packed ``(logits, scores, indices)`` tuples use their native scores as
    probabilities; flat ``[tokens, experts]`` logit matrices are reduced with
    deterministic tie-rejecting top-k whose score columns follow the config's
    declared ``score_function`` when present. The optional second return value
    is a capability note describing evidence limits.
    """

    if type(payload) is tuple:
        if len(payload) != 3:
            raise StructuredCaptureError(
                "decode", "packed router payloads must contain exactly logits, scores, indices"
            )
        logits = _materialize_floats(payload[0])
        scores = _materialize_floats(payload[1])
        indices = _materialize_ints(payload[2])
        token_count = len(token_events)
        if not (len(logits) == len(scores) == len(indices) == token_count):
            raise StructuredCaptureError(
                "decode",
                "packed router payloads must carry exactly one row per captured token",
            )
        top_k = target.routed_top_k
        events: list[RoutingEvent] = []
        for position, token in enumerate(token_events):
            for rank in range(top_k):
                native_index = indices[position][rank]
                if native_index >= len(target.expert_keys):
                    raise StructuredCaptureError(
                        "decode",
                        f"native expert index {native_index} is outside the discovered universe",
                    )
                score = scores[position][rank]
                logit = (
                    logits[position][native_index]
                    if native_index < len(logits[position])
                    else None
                )
                events.append(
                    RoutingEvent(
                        token_key=token.token_key,
                        layer_key=target.layer_key,
                        rank=rank,
                        expert_key=target.expert_keys[native_index],
                        router_logit=logit,
                        probability=score,
                        weight=score,
                        selected=True,
                    )
                )
        return tuple(events), None

    logits = _materialize_floats(payload)
    token_count = len(token_events)
    if len(logits) != token_count:
        raise StructuredCaptureError(
            "decode", "flat router payloads must carry exactly one row per captured token"
        )
    normalization = _score_normalization(config)
    note: str | None = None
    if normalization is None:
        note = (
            "router score normalization was not determinable from the model config; "
            "raw router logits are recorded without probability claims"
        )
    top_k = target.routed_top_k
    events = []
    for position, token in enumerate(token_events):
        row = logits[position]
        ordered = sorted(range(len(row)), key=lambda index: (-row[index], index))
        selected = ordered[:top_k]
        if len({row[index] for index in selected}) != top_k or (
            top_k < len(row) and row[selected[-1]] == row[ordered[top_k]]
        ):
            raise StructuredCaptureError(
                "decode", "selected/cutoff router scores are tied; refusing arbitrary selection"
            )
        full = _softmax_row(row) if normalization == "softmax" else None
        for rank, expert_index in enumerate(selected):
            if expert_index >= len(target.expert_keys):
                raise StructuredCaptureError(
                    "decode",
                    f"router column {expert_index} is outside the discovered expert universe",
                )
            logit = row[expert_index]
            if normalization == "softmax":
                probability = full[expert_index]  # type: ignore[index]
            elif normalization == "sigmoid":
                probability = _sigmoid(logit)
            else:
                probability = None
            events.append(
                RoutingEvent(
                    token_key=token.token_key,
                    layer_key=target.layer_key,
                    rank=rank,
                    expert_key=target.expert_keys[expert_index],
                    router_logit=logit,
                    probability=probability,
                    weight=None,
                    selected=True,
                )
            )
    return tuple(events), note


@dataclass(frozen=True, slots=True)
class StructuredRoutingForwardResult:
    """Caller-owned output plus complete generic routing evidence."""

    output: object = field(repr=False)
    token_events: tuple[TokenEvent, ...]
    routing_events: tuple[RoutingEvent, ...]
    capability_notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        fresh_tokens = fresh_token_events(self.token_events)
        fresh_routes = fresh_routing_events(self.routing_events)
        for note in self.capability_notes:
            if type(note) is not str or not note or note != note.strip():
                raise TypeError("capability_notes must be non-empty trimmed strings")
        object.__setattr__(self, "token_events", fresh_tokens)
        object.__setattr__(self, "routing_events", fresh_routes)
        object.__setattr__(self, "capability_notes", tuple(self.capability_notes))


def run_structured_routing_forward(
    model: object,
    report: DiscoveryReport,
    token_events: tuple[TokenEvent, ...],
    model_kwargs: dict[str, object],
    *,
    max_events: int,
    config: object = None,
) -> StructuredRoutingForwardResult:
    """Run exactly one forward with structure-driven routing capture.

    The structure report supplies every target; hooks attach through the
    passive :class:`~moeatlas.probe.HookManager`; each router payload decodes
    generically against the discovered expert universe. The complete-event
    budget ``len(token_events) * len(targets) * routed_top_k`` is checked
    before hooks exist. Every router must fire exactly once and every token
    must receive every layer's complete rank schedule, otherwise the capture
    fails without publishing anything.
    """

    if not callable(model):
        raise TypeError("model must be callable")
    if type(max_events) is not int or isinstance(max_events, bool):
        raise TypeError("max_events must be a strict integer")
    if max_events <= 0:
        raise ValueError("max_events must be positive")
    if type(model_kwargs) is not dict:
        raise TypeError("model_kwargs must be an exact dict")
    for key in model_kwargs:
        if type(key) is not str or not key or key != key.strip():
            raise TypeError("model_kwargs keys must be non-empty trimmed exact strings")

    fresh_tokens = fresh_token_events(token_events)
    targets = structured_router_targets(report)
    expected_events = len(fresh_tokens) * len(targets) * targets[0].routed_top_k
    if max_events < expected_events:
        raise StructuredCaptureError(
            "preflight", "max_events is insufficient for complete structured routing capture"
        )

    plan = ProbePlan(
        level=ProbeLevel.ROUTING,
        hook_points=(HookPoint.FORWARD,),
        targets=tuple(
            ProbeTarget(
                module_path=target.module_path,
                component_key=target.component_key,
                component_kind=ComponentKind.ROUTER,
            )
            for target in targets
        ),
        capture=CapturePolicy(mode=CaptureMode.STATS, reduction=ReductionPolicy.COUNTS),
    )

    captured: dict[str, tuple[RoutingEvent, ...]] = {}
    notes: set[str] = set()

    def callback_for(target: StructuredRouterTarget):
        def callback(module: object, inputs: object, output: object) -> None:
            del module, inputs
            events, note = decode_structured_payload(
                output,
                target=target,
                token_events=fresh_tokens,
                config=config,
            )
            if note:
                notes.add(note)
            if target.module_path in captured:
                raise StructuredCaptureError(
                    "events", f"router {target.module_path!r} fired more than once"
                )
            captured[target.module_path] = events

        return callback

    callbacks = {
        HookBinding(target.module_path, HookPoint.FORWARD): callback_for(target)
        for target in targets
    }
    manager = HookManager(model, plan, callbacks)
    entered = False
    try:
        manager.__enter__()
        entered = True
        output = model(**dict(model_kwargs))
    except BaseException:
        if entered:
            try:
                manager.close()
            except Exception:
                pass  # cleanup failures never replace the primary body exception
        raise
    manager.__exit__(None, None, None)

    missing = [target.module_path for target in targets if target.module_path not in captured]
    if missing:
        raise StructuredCaptureError(
            "events", f"routers did not fire during the forward: {missing}"
        )
    ordered_targets = sorted(targets, key=lambda target: target.layer_index)
    routing: list[RoutingEvent] = []
    for target in ordered_targets:
        routing.extend(captured[target.module_path])

    token_keys = [token.token_key for token in fresh_tokens]
    expected_pairs = {(key, target.layer_key) for key in token_keys for target in targets}
    observed_pairs = {(event.token_key, event.layer_key) for event in routing}
    if observed_pairs != expected_pairs:
        raise StructuredCaptureError(
            "events", "routing capture did not publish complete events for every token and layer"
        )
    grouped: dict[tuple[str, str], list[RoutingEvent]] = {}
    for event in routing:
        grouped.setdefault((event.token_key, event.layer_key), []).append(event)
    for group in grouped.values():
        ranks = sorted(event.rank for event in group)
        if ranks != list(range(targets[0].routed_top_k)):
            raise StructuredCaptureError(
                "events",
                f"routing capture ranks are not exactly 0..{targets[0].routed_top_k - 1}",
            )
        experts = [event.expert_key for event in sorted(group, key=lambda item: item.rank)]
        if len(set(experts)) != len(experts):
            raise StructuredCaptureError(
                "events", "routing capture selected the same expert twice for one token"
            )

    return StructuredRoutingForwardResult(
        output=output,
        token_events=fresh_tokens,
        routing_events=tuple(routing),
        capability_notes=tuple(sorted(notes)),
    )


__all__ = [
    "StructuredCaptureError",
    "StructuredRoutingForwardResult",
    "StructuredRouterTarget",
    "decode_structured_payload",
    "run_structured_routing_forward",
    "structured_router_targets",
]
