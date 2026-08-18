"""Cleanup-safe routing-event capture over a validated probe plan.

This module is deliberately a thin runtime boundary.  It consumes static
adapter evidence and a canonical forward-hook plan, delegates resolution and
hook lifecycle to :mod:`moeatlas.probe`, and retains only validated
``RoutingEvent`` objects.  The caller-owned model and decoder may remain
available for the session lifetime, but synchronous callback payloads are
never retained after the callback returns.  The decoder owns detaching and
reduction; this boundary never decodes tensors itself.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..adapters import AdapterInspection, build_routing_probe_plan
from ..core import ComponentKind, parse_component_key
from ..events import RoutingEvent
from ..probe import HookBinding, HookManager, HookPoint, ProbePlan, ProbeTarget

_STAGES = frozenset({"preflight", "decode", "events", "lifecycle"})


class RoutingCaptureError(RuntimeError):
    """Safe fixed-stage error for one routing-capture boundary."""

    def __init__(self, stage: str) -> None:
        if stage not in _STAGES:
            raise ValueError(
                "routing capture error stage must be preflight, decode, events, or lifecycle"
            )
        self.stage = stage
        super().__init__(f"routing capture failed at {stage}")


@dataclass(frozen=True, slots=True)
class RoutingCaptureTarget:
    """The static identity and top-k budget for one router hook."""

    router: ProbeTarget
    layer_key: str
    expert_keys: tuple[str, ...]
    routed_top_k: int

    def __post_init__(self) -> None:
        if type(self.router) is not ProbeTarget:
            raise TypeError("router must be an exact ProbeTarget")
        if self.router.component_kind is not ComponentKind.ROUTER:
            raise ValueError("router target must have ComponentKind.ROUTER")
        if type(self.layer_key) is not str:
            raise TypeError("layer_key must be a string")
        parse_component_key(self.layer_key)
        if type(self.expert_keys) is not tuple:
            raise TypeError("expert_keys must be a tuple")
        if not self.expert_keys:
            raise ValueError("expert_keys must contain at least one routed expert")
        for expert_key in self.expert_keys:
            if type(expert_key) is not str:
                raise TypeError("expert_keys must contain strings")
            parse_component_key(expert_key)
        if len(set(self.expert_keys)) != len(self.expert_keys):
            raise ValueError("expert_keys must be unique")
        if type(self.routed_top_k) is not int or isinstance(self.routed_top_k, bool):
            raise TypeError("routed_top_k must be a strict integer")
        if self.routed_top_k <= 0:
            raise ValueError("routed_top_k must be positive")
        if self.routed_top_k > len(self.expert_keys):
            raise ValueError("routed_top_k cannot exceed the routed expert count")


class RoutingCaptureSession:
    """One single-use, passive routing capture session.

    Only validated events survive a successful body and hook cleanup.  The
    supplied decoder owns architecture-specific interpretation, detaching,
    and reduction of the opaque hook arguments; this session only validates
    identity, shape, and a retained-event quota.  The caller-owned model and
    decoder may remain available for the session lifetime, while callback
    payloads are never retained after invocation.  Ordinary decoder and event
    failures are chained to fixed stage errors; body and control-flow failures
    remain the exact primary exception.
    """

    __slots__ = (
        "_active_body",
        "_contexts",
        "_decoder",
        "_dropped_invocations",
        "_entered",
        "_event_ids",
        "_events",
        "_in_context",
        "_manager",
        "_max_events",
        "_model",
        "_normal_body",
        "_plan",
        "_started",
        "_state",
        "_truncated",
        "_body_failed",
        "_closed_early",
    )

    def __init__(
        self,
        model: object,
        inspection: AdapterInspection,
        plan: ProbePlan,
        decoder: Callable[
            [RoutingCaptureTarget, object, tuple[object, ...], object],
            tuple[RoutingEvent, ...],
        ],
        *,
        max_events: int,
    ) -> None:
        try:
            if type(inspection) is not AdapterInspection:
                raise TypeError("inspection must be an exact AdapterInspection")
            if type(plan) is not ProbePlan:
                raise TypeError("plan must be an exact ProbePlan")
            if not callable(decoder):
                raise TypeError("decoder must be callable")
            if type(max_events) is not int or isinstance(max_events, bool):
                raise TypeError("max_events must be a strict integer")
            if max_events <= 0:
                raise ValueError("max_events must be positive")

            inspection_payload = inspection.model_dump(mode="json")
            fresh_inspection = AdapterInspection.model_validate(inspection_payload)
            if type(fresh_inspection) is not AdapterInspection:
                raise TypeError("inspection revalidation returned an unexpected type")

            plan_payload = plan.model_dump(mode="json")
            fresh_plan = ProbePlan.model_validate(plan_payload)
            if type(fresh_plan) is not ProbePlan:
                raise TypeError("plan revalidation returned an unexpected type")

            canonical_plan = build_routing_probe_plan(fresh_inspection)
            if type(canonical_plan) is not ProbePlan:
                raise TypeError("routing plan compiler returned an unexpected type")
            if (
                fresh_plan != canonical_plan
                or fresh_plan.to_json() != canonical_plan.to_json()
                or fresh_plan.plan_id != canonical_plan.plan_id
            ):
                raise ValueError("supplied plan is not the canonical routing plan")
            contexts = _build_contexts(fresh_inspection, fresh_plan)
        except Exception as exc:
            raise RoutingCaptureError("preflight") from exc

        self._contexts = contexts
        self._active_body = False
        self._decoder = decoder
        self._dropped_invocations = 0
        self._entered = False
        self._event_ids: set[tuple[str, str, int, str]] = set()
        self._events: list[RoutingEvent] = []
        self._in_context = False
        self._manager: HookManager | None = None
        self._max_events = max_events
        self._normal_body = False
        self._plan = fresh_plan
        self._started = False
        self._state = "ready"
        self._truncated = False
        self._body_failed = False
        self._closed_early = False
        self._model = model

    @property
    def plan(self) -> ProbePlan:
        return self._plan

    @property
    def events(self) -> tuple[RoutingEvent, ...]:
        self._require_published()
        return tuple(self._events)

    @property
    def truncated(self) -> bool:
        self._require_published()
        return self._truncated

    @property
    def dropped_invocations(self) -> int:
        self._require_published()
        return self._dropped_invocations

    def __enter__(self) -> RoutingCaptureSession:
        if self._started or self._state != "ready":
            raise RoutingCaptureError("lifecycle")
        self._started = True
        try:
            callbacks = {
                HookBinding(path, HookPoint.FORWARD): self._callback_for(context)
                for path, context in self._contexts.items()
            }
            self._manager = HookManager(self._model, self._plan, callbacks)
            self._manager.__enter__()
        except BaseException:
            self._state = "failed"
            self._active_body = False
            self._clear_staging()
            raise
        self._entered = True
        self._in_context = True
        self._active_body = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> bool:
        del traceback
        self._in_context = False
        self._active_body = False
        if self._manager is None or not self._entered:
            return False
        if exc_value is not None:
            self._body_failed = True
            self._normal_body = False
            self._clear_staging()
            self._manager.__exit__(exc_type, exc_value, None)
            self._state = "failed"
            return False

        self._normal_body = True
        if self._closed_early:
            self._state = "failed"
            self._clear_staging()
            return False
        try:
            self._manager.__exit__(None, None, None)
        except Exception as exc:
            self._state = "cleanup_failed"
            raise RoutingCaptureError("lifecycle") from exc
        self._state = "published"
        return False

    def close(self) -> None:
        """Retry failed hook removals and publish only after clean completion."""

        if self._state == "published":
            return
        self._active_body = False
        if self._manager is None or not self._started:
            raise RoutingCaptureError("lifecycle")
        if self._in_context:
            self._closed_early = True
        try:
            self._manager.close()
        except Exception as exc:
            self._state = "cleanup_failed"
            raise RoutingCaptureError("lifecycle") from exc
        if self._entered and self._normal_body and not self._body_failed and not self._closed_early:
            self._state = "published"
        else:
            self._clear_staging()
            self._state = "failed"

    def _callback_for(self, context: RoutingCaptureTarget) -> Callable[..., None]:
        def callback(*args: object, **kwargs: object) -> None:
            self._invoke(context, args, kwargs)
            return None

        return callback

    def _invoke(
        self,
        context: RoutingCaptureTarget,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        if not self._active_body:
            return
        try:
            if kwargs or len(args) != 3:
                raise TypeError("routing callback requires exactly three positional arguments")
            module, inputs, output = args
            if type(inputs) is not tuple:
                raise TypeError("routing callback inputs must be an exact tuple")
        except Exception as exc:
            raise RoutingCaptureError("decode") from exc
        if len(self._events) >= self._max_events:
            self._truncated = True
            self._dropped_invocations += 1
            return
        try:
            decoded = self._decoder(context, module, inputs, output)
        except Exception as exc:
            raise RoutingCaptureError("decode") from exc

        try:
            if type(decoded) is not tuple:
                raise TypeError("decoder must return an exact tuple of RoutingEvent")
            remaining = self._max_events - len(self._events)
            if len(decoded) > remaining:
                self._truncated = True
                self._dropped_invocations += 1
                return
            fresh_events = self._validate_events(context, decoded)
        except Exception as exc:
            raise RoutingCaptureError("events") from exc

        self._events.extend(fresh_events)
        self._event_ids.update(_event_identity(event) for event in fresh_events)

    def _validate_events(
        self,
        context: RoutingCaptureTarget,
        decoded: tuple[object, ...],
    ) -> tuple[RoutingEvent, ...]:
        fresh_events: list[RoutingEvent] = []
        seen = set(self._event_ids)
        for value in decoded:
            if type(value) is not RoutingEvent:
                raise TypeError("decoder must return exact RoutingEvent objects")
            payload = value.model_dump(mode="json")
            event = RoutingEvent.model_validate(payload)
            if type(event) is not RoutingEvent:
                raise TypeError("event revalidation returned an unexpected type")
            if event.layer_key != context.layer_key:
                raise ValueError("routing event layer does not match the hook context")
            if event.expert_key not in context.expert_keys:
                raise ValueError("routing event expert is not allowed by the hook context")
            identity = _event_identity(event)
            if identity in seen:
                raise ValueError("duplicate routing event identity")
            seen.add(identity)
            fresh_events.append(event)
        return tuple(fresh_events)

    def _clear_staging(self) -> None:
        self._events.clear()
        self._event_ids.clear()
        self._truncated = False
        self._dropped_invocations = 0

    def _require_published(self) -> None:
        if self._state != "published":
            raise RoutingCaptureError("lifecycle")


def _event_identity(event: RoutingEvent) -> tuple[str, str, int, str]:
    return (event.token_key, event.layer_key, event.rank, event.expert_key)


def _build_contexts(
    inspection: AdapterInspection,
    plan: ProbePlan,
) -> dict[str, RoutingCaptureTarget]:
    components = inspection.report.components
    component_by_key = {component.component_key: component for component in components}
    facts = inspection.report.facts
    expert_count = facts.expert_count
    routed_top_k = facts.routed_top_k
    if type(expert_count) is not int or type(routed_top_k) is not int:
        raise ValueError("routing facts must provide strict expert_count and routed_top_k")
    if expert_count <= 0 or routed_top_k <= 0 or routed_top_k > expert_count:
        raise ValueError("routing facts must have positive top-k within expert count")

    contexts: dict[str, RoutingCaptureTarget] = {}
    for target in plan.targets:
        router = component_by_key.get(target.component_key or "")
        if router is None or router.kind is not ComponentKind.ROUTER:
            raise ValueError("routing target is not bound to one router component")
        if router.module_path != target.module_path or router.layer_index is None:
            raise ValueError("routing target has incomplete router identity")
        layer_index = router.layer_index
        layers = [
            component
            for component in components
            if component.kind is ComponentKind.MOE_LAYER and component.layer_index == layer_index
        ]
        if len(layers) != 1:
            raise ValueError("each router must have one same-layer MoE layer")
        if any(
            component.kind is ComponentKind.SHARED_EXPERT and component.layer_index == layer_index
            for component in components
        ):
            raise ValueError("shared experts are not valid routing targets")
        experts = [
            component
            for component in components
            if component.kind is ComponentKind.EXPERT and component.layer_index == layer_index
        ]
        if len(experts) != expert_count:
            raise ValueError("same-layer routed expert count does not match discovery facts")
        if any(component.routed is not True or component.shared is True for component in experts):
            raise ValueError("same-layer experts must be routed, indexed, and non-shared")
        indices = [component.expert_index for component in experts]
        if any(type(index) is not int for index in indices) or sorted(indices) != list(
            range(expert_count)
        ):
            raise ValueError(
                "same-layer expert indices must be exact contiguous zero-based indices"
            )
        experts.sort(key=lambda component: component.expert_index)
        context = RoutingCaptureTarget(
            router=target,
            layer_key=layers[0].component_key,
            expert_keys=tuple(component.component_key for component in experts),
            routed_top_k=routed_top_k,
        )
        if target.module_path in contexts:
            raise ValueError("routing target paths must be unique")
        contexts[target.module_path] = context
    if not contexts:
        raise ValueError("routing plan must contain at least one target")
    return contexts


__all__ = ["RoutingCaptureError", "RoutingCaptureSession", "RoutingCaptureTarget"]
