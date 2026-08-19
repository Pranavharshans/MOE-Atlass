"""One-forward Mixtral routing execution and publication boundary."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..adapters import AdapterInspection, build_routing_probe_plan
from ..core import ComponentKind
from ..events import RoutingEvent, TokenEvent
from ..probe import ProbePlan
from .contracts import (
    PendingRuntimeCleanup,
    RuntimeCleanupError,
    _add_cleanup_note,
    _attach_pending_cleanup,
)
from .mixtral_routing import MixtralRoutingDecoder
from .routing import RoutingCaptureError, RoutingCaptureSession


def _fresh_inspection(value: object) -> AdapterInspection:
    if type(value) is not AdapterInspection:
        raise TypeError("inspection must be an exact AdapterInspection")
    fresh = AdapterInspection.model_validate(value.model_dump(mode="json"))
    if type(fresh) is not AdapterInspection or fresh is value:
        raise TypeError("inspection revalidation returned an unexpected type")
    return fresh


def _fresh_plan(value: object) -> ProbePlan:
    if type(value) is not ProbePlan:
        raise TypeError("plan must be an exact ProbePlan")
    fresh = ProbePlan.model_validate(value.model_dump(mode="json"))
    if type(fresh) is not ProbePlan or fresh is value:
        raise TypeError("plan revalidation returned an unexpected type")
    return fresh


def _fresh_token_events(value: object) -> tuple[TokenEvent, ...]:
    if type(value) is not tuple:
        raise TypeError("token_events must be an exact tuple")
    if not value:
        raise ValueError("token_events must be non-empty")
    fresh_events: list[TokenEvent] = []
    seen: set[str] = set()
    run_key: str | None = None
    phase: object | None = None
    for event in value:
        if type(event) is not TokenEvent:
            raise TypeError("token_events must contain exact TokenEvent objects")
        fresh = TokenEvent.model_validate(event.model_dump(mode="json"))
        if type(fresh) is not TokenEvent or fresh is event:
            raise TypeError("token event revalidation returned an unexpected type")
        if fresh.token_key in seen:
            raise ValueError("token_events must have unique token keys")
        seen.add(fresh.token_key)
        if run_key is None:
            run_key = fresh.run_key
            phase = fresh.phase
        elif fresh.run_key != run_key or fresh.phase != phase:
            raise ValueError("token_events must share one run_key and phase")
        fresh_events.append(fresh)
    return tuple(fresh_events)


def _fresh_routing_events(value: object) -> tuple[RoutingEvent, ...]:
    if type(value) is not tuple:
        raise TypeError("routing_events must be an exact tuple")
    if not value:
        raise ValueError("routing_events must be non-empty")
    fresh_events: list[RoutingEvent] = []
    for event in value:
        if type(event) is not RoutingEvent:
            raise TypeError("routing_events must contain exact RoutingEvent objects")
        fresh = RoutingEvent.model_validate(event.model_dump(mode="json"))
        if type(fresh) is not RoutingEvent or fresh is event:
            raise TypeError("routing event revalidation returned an unexpected type")
        fresh_events.append(fresh)
    return tuple(fresh_events)


def _validate_routing_links(
    token_events: tuple[TokenEvent, ...],
    routing_events: tuple[RoutingEvent, ...],
) -> None:
    token_keys = tuple(event.token_key for event in token_events)
    token_key_set = set(token_keys)
    token_positions = {token_key: index for index, token_key in enumerate(token_keys)}
    if any(event.selected is not True for event in routing_events):
        raise ValueError("routing_events must contain only selected events")
    seen_links: set[tuple[str, str, int]] = set()
    represented: set[str] = set()
    layer_order: dict[str, int] = {}
    layer_last_token: dict[str, int] = {}
    layer_last_rank: dict[str, int] = {}
    current_layer = -1
    for event in routing_events:
        if event.token_key not in token_key_set:
            raise ValueError("every routing event must reference a supplied token")
        link = (event.token_key, event.layer_key, event.rank)
        if link in seen_links:
            raise ValueError("routing_events must have unique token-layer-rank links")
        seen_links.add(link)
        represented.add(event.token_key)
        if event.layer_key not in layer_order:
            current_layer += 1
            layer_order[event.layer_key] = current_layer
            layer_last_token[event.layer_key] = -1
            layer_last_rank[event.layer_key] = -1
        elif layer_order[event.layer_key] != current_layer:
            raise ValueError("routing_events must use deterministic layer order")
        token_position = token_positions[event.token_key]
        if token_position < layer_last_token[event.layer_key] or (
            token_position == layer_last_token[event.layer_key]
            and event.rank <= layer_last_rank[event.layer_key]
        ):
            raise ValueError("routing_events must use deterministic token/rank order")
        if token_position != layer_last_token[event.layer_key]:
            layer_last_rank[event.layer_key] = -1
        layer_last_token[event.layer_key] = token_position
        layer_last_rank[event.layer_key] = event.rank
    if represented != token_key_set:
        raise ValueError("every supplied token must be represented by routing events")


@dataclass(frozen=True, slots=True, eq=False)
class MixtralRoutingForwardResult:
    """Caller-owned output and complete fresh routing evidence for one forward."""

    output: object = field(repr=False)
    token_events: tuple[TokenEvent, ...]
    routing_events: tuple[RoutingEvent, ...]

    def __post_init__(self) -> None:
        fresh_tokens = _fresh_token_events(self.token_events)
        fresh_routes = _fresh_routing_events(self.routing_events)
        _validate_routing_links(fresh_tokens, fresh_routes)
        object.__setattr__(self, "token_events", fresh_tokens)
        object.__setattr__(self, "routing_events", fresh_routes)


def _validate_kwargs(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError("model_kwargs must be an exact dict")
    for key in value:
        if type(key) is not str or not key or key != key.strip():
            raise TypeError("model_kwargs keys must be non-empty trimmed exact strings")
    return dict(value)


def _canonical_layer_keys(
    inspection: AdapterInspection,
    canonical_plan: ProbePlan,
) -> tuple[str, ...]:
    """Resolve the expected layer block order from exact static identities."""

    components = inspection.report.components
    layer_keys: list[str] = []
    for target in canonical_plan.targets:
        routers = [
            component
            for component in components
            if component.component_key == target.component_key
            and component.module_path == target.module_path
            and component.kind is ComponentKind.ROUTER
        ]
        if len(routers) != 1:
            raise ValueError("canonical target must bind one exact router component")
        router = routers[0]
        layer_index = router.layer_index
        if type(layer_index) is not int or isinstance(layer_index, bool):
            raise ValueError("router must have a strict layer index")
        layers = [
            component
            for component in components
            if component.kind is ComponentKind.MOE_LAYER and component.layer_index == layer_index
        ]
        if len(layers) != 1:
            raise ValueError("router must bind one exact same-index MoE layer")
        layer_key = layers[0].component_key
        if layer_key in layer_keys:
            raise ValueError("canonical targets must bind distinct layer components")
        layer_keys.append(layer_key)
    if not layer_keys:
        raise ValueError("canonical routing plan must contain targets")
    return tuple(layer_keys)


def _cleanup_after_failure(
    session: RoutingCaptureSession,
    primary: BaseException,
) -> None:
    """Retry session cleanup once while preserving the primary failure."""

    try:
        session.close()
    except BaseException as cleanup_exception:
        if (
            type(cleanup_exception) is RoutingCaptureError
            and cleanup_exception.stage == "lifecycle"
            and cleanup_exception.__cause__ is None
        ):
            return
        cleanup_error = (
            cleanup_exception
            if isinstance(cleanup_exception, RuntimeCleanupError)
            else RuntimeCleanupError((cleanup_exception,))
        )
        pending = PendingRuntimeCleanup(session.close)
        _attach_pending_cleanup(primary, pending)
        _add_cleanup_note(primary, cleanup_error)


def run_mixtral_routing_forward(
    model: object,
    inspection: AdapterInspection,
    plan: ProbePlan,
    token_events: tuple[TokenEvent, ...],
    model_kwargs: dict[str, object],
    *,
    max_events: int,
) -> MixtralRoutingForwardResult:
    """Run exactly one caller-supplied Mixtral forward with complete routing capture."""

    if not callable(model):
        raise TypeError("model must be callable")
    fresh_inspection = _fresh_inspection(inspection)
    fresh_plan = _fresh_plan(plan)
    fresh_tokens = _fresh_token_events(token_events)
    copied_model_kwargs = _validate_kwargs(model_kwargs)
    if type(max_events) is not int or isinstance(max_events, bool):
        raise TypeError("max_events must be a strict integer")
    if max_events <= 0:
        raise ValueError("max_events must be positive")

    canonical_plan = build_routing_probe_plan(fresh_inspection)
    if type(canonical_plan) is not ProbePlan:
        raise TypeError("routing plan compiler returned an unexpected type")
    if (
        fresh_plan != canonical_plan
        or fresh_plan.to_json() != canonical_plan.to_json()
        or fresh_plan.plan_id != canonical_plan.plan_id
    ):
        raise ValueError("supplied plan is not the canonical routing plan")
    routed_top_k = fresh_inspection.report.facts.routed_top_k
    if type(routed_top_k) is not int or isinstance(routed_top_k, bool) or routed_top_k <= 0:
        raise ValueError("inspection must provide a strict positive routed_top_k")
    expected_events = len(fresh_tokens) * len(canonical_plan.targets) * routed_top_k
    if max_events < expected_events:
        raise ValueError("max_events is insufficient for complete routing capture")
    expected_layer_keys = _canonical_layer_keys(fresh_inspection, canonical_plan)

    decoder = MixtralRoutingDecoder(fresh_inspection, fresh_tokens)
    session = RoutingCaptureSession(
        model,
        fresh_inspection,
        fresh_plan,
        decoder,
        max_events=max_events,
    )
    try:
        with session:
            output = model(**copied_model_kwargs)
    except BaseException as primary:
        _cleanup_after_failure(session, primary)
        raise

    captured = session.events
    if len(captured) != expected_events or session.truncated or session.dropped_invocations != 0:
        raise ValueError("routing capture did not publish complete events")
    captured_layer_keys: list[str] = []
    for event in captured:
        if not captured_layer_keys or captured_layer_keys[-1] != event.layer_key:
            captured_layer_keys.append(event.layer_key)
    if tuple(captured_layer_keys) != expected_layer_keys:
        raise ValueError("routing events do not use the canonical layer block order")
    return MixtralRoutingForwardResult(output, fresh_tokens, captured)


__all__ = ["MixtralRoutingForwardResult", "run_mixtral_routing_forward"]
