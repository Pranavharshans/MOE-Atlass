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
Router targets bind to the structure the scan proved — published expert
containers and their sibling topology — so noisy name-token candidates on
foreign families (SwiGLU ``gate_proj`` modules, ``...Moe...`` class names)
never become hook points; strict name guards apply only when such evidence is
absent.

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
from ..discovery import (
    DiscoveryReport,
    bind_moe_layer_key,
    bind_routed_expert_keys,
    trusted_routers,
)
from ..event_validation import (
    fresh_expert_events,
    fresh_routing_events,
    fresh_token_events,
    validate_expert_links,
)
from ..events import ExpertEvent, RoutingEvent, TokenEvent
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

_DEFAULT_MAX_EXPERT_EVENTS = 65536


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


@dataclass(frozen=True, slots=True)
class StructuredExpertTarget:
    """One generic expert hook target derived purely from structure evidence."""

    module_path: str
    component_key: str
    layer_key: str
    layer_index: int

    def __post_init__(self) -> None:
        for name in ("module_path", "component_key", "layer_key"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise TypeError(f"{name} must be a non-empty string")
        parse_component_key(self.component_key)
        parse_component_key(self.layer_key)
        if type(self.layer_index) is not int or isinstance(self.layer_index, bool):
            raise ValueError("layer_index must be a strict integer")
        if self.layer_index < 0:
            raise ValueError("layer_index must be non-negative")


def _fresh_report(value: object) -> DiscoveryReport:
    if type(value) is not DiscoveryReport:
        raise TypeError("report must be an exact DiscoveryReport")
    fresh = DiscoveryReport.model_validate(value.model_dump(mode="json"))
    if type(fresh) is not DiscoveryReport or fresh is value:
        raise TypeError("structure report revalidation returned an unexpected type")
    return fresh


def structured_router_targets(report: DiscoveryReport) -> tuple[StructuredRouterTarget, ...]:
    """Resolve generic hook targets from one static structure report.

    Routers bind to the report's own structural evidence, never to name
    guesses: a ROUTER component is trusted when its parent block publishes an
    ``EXPERT_CONTAINER`` component, so SwiGLU ``gate_proj`` modules inside
    individual experts never qualify, and each trusted router binds the
    MOE_LAYER identity the scanner published at its parent-block path. When a
    report publishes no such structure, a strictly guarded fallback applies:
    the final dotted path segment must be exactly ``gate`` and the router's
    layer must carry expert-container or routed-expert evidence. The report's
    strict ``expert_count``/``routed_top_k`` facts drive every count;
    shared-expert components are excluded by kind, never by name.
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
    routers = trusted_routers(components)
    if not routers:
        raise StructuredCaptureError(
            "resolution", "structure report does not publish any routed router"
        )
    targets: list[StructuredRouterTarget] = []
    seen_layers: set[int] = set()
    for router in routers:
        layer_index = router.layer_index
        if layer_index is None:  # pragma: no cover - excluded by trusted_routers
            raise StructuredCaptureError(
                "resolution", "structure report does not publish any routed router"
            )
        if layer_index in seen_layers:
            raise StructuredCaptureError(
                "resolution", f"layer {layer_index} publishes more than one router"
            )
        seen_layers.add(layer_index)
        try:
            layer_key = bind_moe_layer_key(fresh.model_key, components, router)
        except ValueError as exc:
            raise StructuredCaptureError("resolution", str(exc)) from exc
        try:
            expert_keys = bind_routed_expert_keys(
                fresh.model_key, components, router, expert_count
            )
        except ValueError as exc:
            raise StructuredCaptureError("resolution", str(exc)) from exc
        targets.append(
            StructuredRouterTarget(
                module_path=router.module_path,
                component_key=router.component_key,
                layer_key=layer_key,
                expert_keys=expert_keys,
                routed_top_k=routed_top_k,
                layer_index=layer_index,
            )
        )
    return tuple(targets)


def structured_expert_targets(report: DiscoveryReport) -> tuple[StructuredExpertTarget, ...]:
    """Resolve one generic expert hook target per routed expert component.

    The routing universe from :func:`structured_router_targets` drives every
    count; each discovered expert component key must bind exactly one expert
    module path, and the resulting targets are ordered by layer index and
    zero-based expert index. Shared-expert components never appear because the
    router universe only publishes routed experts.
    """

    fresh = _fresh_report(report)
    router_targets = structured_router_targets(fresh)
    components = {
        component.component_key: component
        for component in fresh.components
        if component.kind is ComponentKind.EXPERT
    }
    targets: list[StructuredExpertTarget] = []
    seen_paths: set[str] = set()
    for router_target in sorted(router_targets, key=lambda target: target.layer_index):
        for expert_key in router_target.expert_keys:
            component = components.get(expert_key)
            if component is None:
                raise StructuredCaptureError(
                    "resolution",
                    f"routed expert {expert_key!r} is missing an expert component",
                )
            if type(component.module_path) is not str or not component.module_path:
                raise StructuredCaptureError(
                    "resolution",
                    f"expert {expert_key!r} does not publish a module path",
                )
            if component.module_path in seen_paths:
                raise StructuredCaptureError(
                    "resolution",
                    f"module path {component.module_path!r} hosts more than one routed expert",
                )
            seen_paths.add(component.module_path)
            targets.append(
                StructuredExpertTarget(
                    module_path=component.module_path,
                    component_key=expert_key,
                    layer_key=router_target.layer_key,
                    layer_index=router_target.layer_index,
                )
            )
    return tuple(targets)


def _materialized(value: object) -> object:
    """Detach/move/cast a tensor-like value via duck-typed method chaining."""

    current = value
    for method_name in ("detach", "cpu", "float"):
        method = getattr(current, method_name, None)
        current = method() if callable(method) else current
    return current


def _flatten_finite_floats(value: object, *, what: str) -> list[float]:
    """Recursively flatten a materialized payload into finite float values."""

    current = _materialized(value)
    tolist = getattr(current, "tolist", None)
    if callable(tolist):
        try:
            current = tolist()
        except Exception as exc:
            raise StructuredCaptureError("decode", f"{what} payload is not materializable") from exc
    flat: list[float] = []

    def walk(node: object) -> None:
        if isinstance(node, list | tuple):
            for item in node:
                walk(item)
            return
        if type(node) is not float or not math.isfinite(node):
            raise StructuredCaptureError("decode", f"{what} payload entries must be finite floats")
        flat.append(node)

    try:
        walk(current)
    except RecursionError as exc:
        raise StructuredCaptureError("decode", f"{what} payload nesting is too deep") from exc
    if not flat:
        raise StructuredCaptureError("decode", f"{what} payload must contain at least one value")
    return flat


def _shape_signature(value: object) -> tuple[int, ...] | None:
    """Return the nested list shape of an already-materialized payload."""

    if isinstance(value, list | tuple):
        if not value:
            return None
        first = _shape_signature(value[0])
        if first is None or any(_shape_signature(item) != first for item in value[1:]):
            return None
        return (len(value),) + first
    if type(value) is float:
        return ()
    return None


def decode_expert_activity(
    inputs: object,
    output: object,
) -> tuple[float, float, float | None]:
    """Decode one opaque expert hook invocation into L2 norm measurements.

    The input side uses the first positional forward argument (the torch
    forward-hook convention); norms are computed over detached, materialized
    values without importing any model stack. ``contribution_norm`` requires
    matching input/output shapes and is ``None`` otherwise — expert FFN blocks
    legitimately change width.
    """

    input_value = inputs[0] if isinstance(inputs, tuple | list) and inputs else inputs
    flat_input = _flatten_finite_floats(input_value, what="expert input")
    flat_output = _flatten_finite_floats(output, what="expert output")
    input_norm = math.sqrt(sum(value * value for value in flat_input))
    output_norm = math.sqrt(sum(value * value for value in flat_output))
    contribution_norm: float | None = None
    input_materialized = _materialized(input_value)
    output_materialized = _materialized(output)
    input_shape = _shape_signature(input_materialized)
    output_shape = _shape_signature(output_materialized)
    if input_shape == output_shape and len(flat_input) == len(flat_output):
        difference = (
            out_value - in_value for in_value, out_value in zip(flat_input, flat_output)
        )
        contribution_norm = math.sqrt(sum(value * value for value in difference))
    return input_norm, output_norm, contribution_norm


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
    probabilities.  Ling-style ``(indices, weights, logits)`` tuples preserve
    native weights separately and never relabel them as probabilities: router
    implementations commonly apply a scaling factor, so a value above one is
    valid weight evidence but not a probability claim. Flat ``[tokens,
    experts]`` logit matrices are reduced with deterministic tie-rejecting
    top-k whose score columns follow the config's declared ``score_function``
    when present. The optional second return value is a capability note
    describing evidence limits.
    """

    if type(payload) is tuple:
        if len(payload) != 3:
            raise StructuredCaptureError(
                "decode", "packed router payloads must contain exactly logits, scores, indices"
            )
        # There are two widely used three-column contracts.  Do not infer the
        # order from a family name or module path: integer first-column data is
        # the unambiguous marker for Ling-style ``(indices, weights, logits)``;
        # the historical packed contract starts with floating-point logits.
        native_indices: list[list[int]] | None = None
        try:
            native_indices = _materialize_ints(payload[0])
        except StructuredCaptureError:
            pass
        if native_indices is not None:
            indices = native_indices
            weights = _materialize_floats(payload[1])
            logits = _materialize_floats(payload[2])
            token_count = len(token_events)
            if not (len(indices) == len(weights) == len(logits) == token_count):
                raise StructuredCaptureError(
                    "decode",
                    "native router tuples must carry exactly one row per captured token",
                )
            top_k = target.routed_top_k
            if len(indices[0]) < top_k or len(weights[0]) < top_k:
                raise StructuredCaptureError(
                    "decode", "native router tuples do not carry the discovered routed top-k"
                )
            events: list[RoutingEvent] = []
            for position, token in enumerate(token_events):
                for rank in range(top_k):
                    native_index = indices[position][rank]
                    if native_index >= len(target.expert_keys):
                        raise StructuredCaptureError(
                            "decode",
                            "native expert index "
                            f"{native_index} is outside the discovered universe",
                        )
                    weight = weights[position][rank]
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
                            probability=None,
                            weight=weight,
                            selected=True,
                        )
                    )
            return (
                tuple(events),
                "native router weights were retained as weights; probability "
                "normalization was not claimed",
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
    """Caller-owned output plus complete generic routing evidence.

    ``expert_events`` carries optional per-invoked-expert activity evidence;
    when present it revalidates and links to the captured tokens exactly like
    routing rows do.
    """

    output: object = field(repr=False)
    token_events: tuple[TokenEvent, ...]
    routing_events: tuple[RoutingEvent, ...]
    capability_notes: tuple[str, ...] = ()
    expert_events: tuple[ExpertEvent, ...] = ()

    def __post_init__(self) -> None:
        fresh_tokens = fresh_token_events(self.token_events)
        fresh_routes = fresh_routing_events(self.routing_events)
        for note in self.capability_notes:
            if type(note) is not str or not note or note != note.strip():
                raise TypeError("capability_notes must be non-empty trimmed strings")
        fresh_experts = fresh_expert_events(self.expert_events)
        validate_expert_links(fresh_tokens, fresh_experts)
        object.__setattr__(self, "token_events", fresh_tokens)
        object.__setattr__(self, "routing_events", fresh_routes)
        object.__setattr__(self, "capability_notes", tuple(self.capability_notes))
        object.__setattr__(self, "expert_events", fresh_experts)


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


def run_structured_expert_forward(
    model: object,
    report: DiscoveryReport,
    token_events: tuple[TokenEvent, ...],
    model_kwargs: dict[str, object],
    *,
    max_events: int,
    config: object = None,
    max_expert_events: int = _DEFAULT_MAX_EXPERT_EVENTS,
) -> StructuredRoutingForwardResult:
    """Run exactly one forward with structure-driven routing AND expert capture.

    Router targets decode exactly as :func:`run_structured_routing_forward`;
    each discovered routed-expert module additionally receives one passive
    ``EXPERT_ACTIVITY`` forward hook whose payload is reduced to input/output/
    contribution L2 norms for the whole invocation. Every selected expert must
    fire exactly once, every fired expert must be selected at least once, and
    the projected expert-event count must fit ``max_expert_events`` — otherwise
    the capture fails without publishing anything.
    """

    if not callable(model):
        raise TypeError("model must be callable")
    if type(max_events) is not int or isinstance(max_events, bool):
        raise TypeError("max_events must be a strict integer")
    if max_events <= 0:
        raise ValueError("max_events must be positive")
    if type(max_expert_events) is not int or isinstance(max_expert_events, bool):
        raise TypeError("max_expert_events must be a strict integer")
    if max_expert_events <= 0:
        raise ValueError("max_expert_events must be positive")
    if type(model_kwargs) is not dict:
        raise TypeError("model_kwargs must be an exact dict")
    for key in model_kwargs:
        if type(key) is not str or not key or key != key.strip():
            raise TypeError("model_kwargs keys must be non-empty trimmed exact strings")

    fresh_tokens = fresh_token_events(token_events)
    router_targets = structured_router_targets(report)
    expert_targets = structured_expert_targets(report)
    top_k = router_targets[0].routed_top_k
    expected_events = len(fresh_tokens) * len(router_targets) * top_k
    # Every selected (token, layer) pair contributes one expert event per rank.
    expected_expert_events = expected_events
    if max_events < expected_events:
        raise StructuredCaptureError(
            "preflight", "max_events is insufficient for complete structured routing capture"
        )
    if max_expert_events < expected_expert_events:
        raise StructuredCaptureError(
            "preflight",
            "max_expert_events is insufficient for complete structured expert capture",
        )

    plan = ProbePlan(
        level=ProbeLevel.EXPERT_ACTIVITY,
        hook_points=(HookPoint.FORWARD,),
        targets=(
            *[ProbeTarget(
                module_path=target.module_path,
                component_key=target.component_key,
                component_kind=ComponentKind.ROUTER,
            ) for target in router_targets],
            *[ProbeTarget(
                module_path=target.module_path,
                component_key=target.component_key,
                component_kind=ComponentKind.EXPERT,
            ) for target in expert_targets],
        ),
        capture=CapturePolicy(mode=CaptureMode.STATS, reduction=ReductionPolicy.COUNTS),
    )

    captured: dict[str, tuple[RoutingEvent, ...]] = {}
    captured_experts: dict[str, tuple[StructuredExpertTarget, float, float, float | None]] = {}
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

    def expert_callback_for(target: StructuredExpertTarget):
        def callback(module: object, inputs: object, output: object) -> None:
            del module
            input_norm, output_norm, contribution_norm = decode_expert_activity(inputs, output)
            if target.module_path in captured_experts:
                raise StructuredCaptureError(
                    "events", f"expert {target.module_path!r} fired more than once"
                )
            captured_experts[target.module_path] = (
                target,
                input_norm,
                output_norm,
                contribution_norm,
            )

        return callback

    callbacks: dict[HookBinding, Any] = {
        HookBinding(target.module_path, HookPoint.FORWARD): callback_for(target)
        for target in router_targets
    }
    for target in expert_targets:
        callbacks[HookBinding(target.module_path, HookPoint.FORWARD)] = expert_callback_for(target)
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

    missing = [
        target.module_path for target in router_targets if target.module_path not in captured
    ]
    if missing:
        raise StructuredCaptureError(
            "events", f"routers did not fire during the forward: {missing}"
        )
    ordered_targets = sorted(router_targets, key=lambda target: target.layer_index)
    routing: list[RoutingEvent] = []
    for target in ordered_targets:
        routing.extend(captured[target.module_path])

    token_keys = [token.token_key for token in fresh_tokens]
    expected_pairs = {(key, target.layer_key) for key in token_keys for target in router_targets}
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
        if ranks != list(range(top_k)):
            raise StructuredCaptureError(
                "events",
                f"routing capture ranks are not exactly 0..{top_k - 1}",
            )
        experts = [event.expert_key for event in sorted(group, key=lambda item: item.rank)]
        if len(set(experts)) != len(experts):
            raise StructuredCaptureError(
                "events", "routing capture selected the same expert twice for one token"
            )

    # Expert completeness: every fired expert must be bound to a known routed
    # expert of its layer and selected by at least one token; every selected
    # (layer, expert) pair must have fired exactly once.
    layer_by_key = {target.layer_key: target.layer_index for target in router_targets}
    experts_by_layer: dict[str, tuple[str, ...]] = {}
    for target in expert_targets:
        experts_by_layer.setdefault(target.layer_key, ())
        experts_by_layer[target.layer_key] = experts_by_layer[target.layer_key] + (
            target.component_key,
        )
    fired_by_layer: dict[str, dict[str, tuple[float, float, float | None]]] = {}
    for module_path, (target, input_norm, output_norm, contribution_norm) in (
        captured_experts.items()
    ):
        known = experts_by_layer.get(target.layer_key, ())
        if target.component_key not in known or target.layer_key not in layer_by_key:
            raise StructuredCaptureError(
                "events",
                f"fired expert {module_path!r} is outside the discovered routing universe",
            )
        fired_by_layer.setdefault(target.layer_key, {})[target.component_key] = (
            input_norm,
            output_norm,
            contribution_norm,
        )
    selected_pairs: set[tuple[str, str]] = set()
    for event in routing:
        selected_pairs.add((event.layer_key, event.expert_key))
    for pair in sorted(selected_pairs):
        if pair[1] not in fired_by_layer.get(pair[0], {}):
            raise StructuredCaptureError(
                "events",
                f"selected expert at layer {pair[0]!r} did not fire during the forward",
            )

    expert_events: list[ExpertEvent] = []
    for target in sorted(expert_targets, key=lambda item: item.layer_index):
        # Sparse dispatch executes only experts selected by at least one token.
        # The completeness check above already proves that every selected
        # expert fired, so an absent entry here is a valid inactive expert,
        # not a missing-capture error.
        norms = fired_by_layer.get(target.layer_key, {}).get(target.component_key)
        if norms is None:
            continue
        selected_tokens = [
            event.token_key
            for event in routing
            if event.layer_key == target.layer_key and event.expert_key == target.component_key
        ]
        if not selected_tokens:
            # An expert module legitimately fires without any captured token
            # selecting it (batch dispatch); it contributes no events.
            continue
        for token_key in selected_tokens:
            expert_events.append(
                ExpertEvent(
                    token_key=token_key,
                    expert_key=target.component_key,
                    input_norm=norms[0],
                    output_norm=norms[1],
                    contribution_norm=norms[2],
                    metadata={"invocation_token_count": len(selected_tokens)},
                )
            )

    return StructuredRoutingForwardResult(
        output=output,
        token_events=fresh_tokens,
        routing_events=tuple(routing),
        capability_notes=tuple(sorted(notes)),
        expert_events=tuple(expert_events),
    )


__all__ = [
    "StructuredCaptureError",
    "StructuredExpertTarget",
    "StructuredRouterTarget",
    "StructuredRoutingForwardResult",
    "decode_expert_activity",
    "decode_structured_payload",
    "run_structured_expert_forward",
    "run_structured_routing_forward",
    "structured_expert_targets",
    "structured_router_targets",
]
