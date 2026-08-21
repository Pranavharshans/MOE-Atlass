"""One-forward routing execution and publication boundary.

The shared :func:`run_routing_forward` seam is model-neutral: it composes the
canonical plan compiler, the adapter-published routing universe, any conforming
:class:`~moeatlas.runtime.RoutingHookDecoder`, and the passive capture session
without ever inspecting an adapter name, module-path convention, or payload
type identity.  Family knowledge lives entirely in the decoder factories the
caller supplies; ``run_mixtral_routing_forward`` and
``run_qwen3_5_routing_forward`` are thin compatibility wrappers over this seam
with their historical decoders.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ..adapters import (
    AdapterInspection,
    RoutingUniverse,
    build_routing_probe_plan,
    project_rectangular_universe,
    publish_routing_universe,
)
from ..core import ComponentKind
from ..event_validation import (
    fresh_routing_events,
    fresh_token_events,
    validate_routing_links,
)
from ..events import RoutingEvent, TokenEvent
from ..probe import ProbePlan
from .capabilities import (
    RouterPayloadShape,
    RoutingHookDecoder,
    ScoreSemantics,
    TokenSequencePolicy,
    validate_observed_routing,
)
from .contracts import (
    PendingRuntimeCleanup,
    RuntimeCleanupError,
    _add_cleanup_note,
    _attach_pending_cleanup,
)
from .mixtral_routing import MixtralRoutingDecoder
from .qwen3_5_routing import Qwen3_5RoutingDecoder
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


_fresh_token_events = fresh_token_events
_fresh_routing_events = fresh_routing_events
_validate_routing_links = validate_routing_links


@dataclass(frozen=True, slots=True, eq=False)
class RoutingForwardResult:
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


def _declared_capabilities(decoder: object) -> ScoreSemantics:
    """Require the declared payload-shape and score-semantics capabilities."""

    if not isinstance(decoder, RoutingHookDecoder):
        raise TypeError(
            "decoder must declare payload_shape, score_semantics, "
            "and decode one router hook invocation into RoutingEvent rows"
        )
    shape = decoder.payload_shape
    if type(shape) is not RouterPayloadShape:
        raise TypeError("decoder payload_shape must be an exact RouterPayloadShape")
    semantics = decoder.score_semantics
    if type(semantics) is not ScoreSemantics:
        raise TypeError("decoder score_semantics must be an exact ScoreSemantics")
    return semantics


def _capture_universe(inspection: AdapterInspection) -> RoutingUniverse:
    """Publish and explicitly project the inspection's routing universe.

    Hook capture consumes complete per-layer payloads, so the session's
    uniform capture contexts require the rectangular reduction.  Non-
    rectangular families are first-class contracts elsewhere; here the
    limitation is surfaced as this named gate instead of a hidden invariant.
    """

    universe = publish_routing_universe(inspection)
    project_rectangular_universe(universe)
    return universe


MixtralRoutingForwardResult = RoutingForwardResult


def run_routing_forward(
    model: object,
    inspection: AdapterInspection,
    plan: ProbePlan,
    token_events: tuple[TokenEvent, ...],
    model_kwargs: dict[str, object],
    *,
    decoder: Callable[[AdapterInspection, tuple[TokenEvent, ...]], object],
    token_sequence: TokenSequencePolicy = TokenSequencePolicy.SHARED_RUN,
    max_events: int,
) -> RoutingForwardResult:
    """Run exactly one caller-supplied model forward with complete routing capture.

    This is the universal execution boundary for every MoE family.  The
    ``decoder`` factory is invoked with the freshly revalidated inspection and
    token rows and must return a conforming
    :class:`~moeatlas.runtime.RoutingHookDecoder` whose declared score
    semantics drive the shared postcondition verification of everything it
    decodes.  ``token_sequence`` selects how strictly the supplied token
    identities are validated before execution.  The inspection's published
    routing universe must pass the explicit rectangular projection, because
    hook capture binds one uniform capture context per router today.
    """

    if not callable(model):
        raise TypeError("model must be callable")
    if not callable(decoder):
        raise TypeError("decoder must be callable")
    if type(token_sequence) is not TokenSequencePolicy:
        raise TypeError("token_sequence must be an exact TokenSequencePolicy")
    fresh_inspection = _fresh_inspection(inspection)
    fresh_plan = _fresh_plan(plan)
    fresh_tokens = _fresh_token_events(
        token_events,
        strict_sequence=token_sequence is TokenSequencePolicy.CANONICAL_SEQUENCE,
    )
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

    universe = _capture_universe(fresh_inspection)

    hook_decoder = decoder(fresh_inspection, fresh_tokens)
    semantics = _declared_capabilities(hook_decoder)
    session = RoutingCaptureSession(
        model,
        fresh_inspection,
        fresh_plan,
        hook_decoder,
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
    validate_observed_routing(
        captured,
        universe=universe,
        token_keys=tuple(event.token_key for event in fresh_tokens),
        score_semantics=semantics,
    )
    return RoutingForwardResult(output, fresh_tokens, captured)


def run_mixtral_routing_forward(
    model: object,
    inspection: AdapterInspection,
    plan: ProbePlan,
    token_events: tuple[TokenEvent, ...],
    model_kwargs: dict[str, object],
    *,
    max_events: int,
) -> RoutingForwardResult:
    """Run exactly one caller-supplied Mixtral forward with complete routing capture.

    Thin compatibility wrapper over :func:`run_routing_forward` composing the
    declared :class:`MixtralRoutingDecoder` capability.
    """

    return run_routing_forward(
        model,
        inspection,
        plan,
        token_events,
        model_kwargs,
        decoder=MixtralRoutingDecoder,
        token_sequence=TokenSequencePolicy.SHARED_RUN,
        max_events=max_events,
    )


def run_qwen3_5_routing_forward(
    model: object,
    inspection: AdapterInspection,
    plan: ProbePlan,
    token_events: tuple[TokenEvent, ...],
    model_kwargs: dict[str, object],
    *,
    max_events: int,
) -> RoutingForwardResult:
    """Run exactly one caller-supplied Qwen3.5 forward with routing capture.

    Thin compatibility wrapper over :func:`run_routing_forward` composing the
    declared :class:`Qwen3_5RoutingDecoder` capability; packed whole-sequence
    gate tuples decode against the contiguous canonical token sequence.
    """

    return run_routing_forward(
        model,
        inspection,
        plan,
        token_events,
        model_kwargs,
        decoder=Qwen3_5RoutingDecoder,
        token_sequence=TokenSequencePolicy.CANONICAL_SEQUENCE,
        max_events=max_events,
    )


__all__ = [
    "RoutingForwardResult",
    "MixtralRoutingForwardResult",
    "run_routing_forward",
    "run_mixtral_routing_forward",
    "run_qwen3_5_routing_forward",
]
