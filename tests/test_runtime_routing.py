from __future__ import annotations

import ast
import inspect
import socket
import sys
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path
from typing import get_type_hints
from urllib import request

import pytest

import moeatlas.runtime.routing as routing_module
from moeatlas import runtime as runtime_package
from moeatlas.adapters import (
    AdapterInspection,
    MixtralStaticAdapter,
    Qwen3_5MoeStaticAdapter,
    Qwen3MoeStaticAdapter,
    build_routing_probe_plan,
    inspect_static_adapter,
)
from moeatlas.core import (
    ComponentKind,
    DType,
    ModelManifest,
    TokenizerIdentity,
    make_config_hash,
    make_model_key,
)
from moeatlas.events import RoutingEvent
from moeatlas.probe import HookBinding, HookPoint, ProbeLevel, ProbePlan
from moeatlas.runtime import (
    RoutingCaptureError,
    RoutingCaptureSession,
    RoutingCaptureTarget,
)

from .fixtures import MixtralForCausalLM, Qwen3MoeForCausalLM
from .fixtures.qwen3_5_moe import (
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeForConditionalGeneration,
)


def _manifest(architecture: str) -> ModelManifest:
    revision = "r1"
    return ModelManifest(
        model_key=make_model_key(f"acme/{architecture}", revision),
        architecture=architecture,
        revision=revision,
        config_hash=make_config_hash({"architecture": architecture, "revision": revision}),
        tokenizer=TokenizerIdentity(identifier=f"acme/{architecture}-tokenizer", revision=revision),
        dtype=DType.FLOAT32,
        device_map={"": "cpu"},
    )


def _inspection(architecture: str, layout: str) -> AdapterInspection:
    if architecture == "mixtral":
        model = MixtralForCausalLM(layout=layout)
        adapter = MixtralStaticAdapter()
    elif architecture == "qwen3_5_moe":
        model = Qwen3_5MoeForConditionalGeneration(layout=layout)
        adapter = Qwen3_5MoeStaticAdapter()
    elif architecture == "qwen3_5_moe_text":
        model = Qwen3_5MoeForCausalLM(layout=layout)
        adapter = Qwen3_5MoeStaticAdapter()
    else:
        model = Qwen3MoeForCausalLM(layout=layout)
        adapter = Qwen3MoeStaticAdapter()
    return inspect_static_adapter(adapter, model, model.config, _manifest(architecture))


def _future_inspection(architecture: str = "mixtral", layout: str = "legacy") -> AdapterInspection:
    inspection = _inspection(architecture, layout)
    descriptor = inspection.descriptor.model_copy(
        update={
            "name": "future-family-static",
            "version": "9.0",
            "architecture_families": ("future-family",),
        }
    )
    components = [
        component.model_copy(
            update={
                "capture": component.capture.model_copy(
                    update={"adapter": descriptor.name, "adapter_version": descriptor.version}
                )
            }
        )
        for component in inspection.report.components
    ]
    report = inspection.report.model_copy(update={"components": components})
    return inspection.model_copy(update={"descriptor": descriptor, "report": report})


class _Handle:
    def __init__(self, owner: _HookModule, callback: object, failures: int = 0) -> None:
        self.owner = owner
        self.callback = callback
        self.failures = failures
        self.removed = False

    def remove(self) -> None:
        if self.failures:
            self.failures -= 1
            raise OSError("transient handle removal")
        if not self.removed:
            self.removed = True
            self.owner.callbacks.remove(self.callback)
            self.owner.removals.append(self.owner.path)
            if self.owner.removal_log is not None:
                self.owner.removal_log.append(self.owner.path)


class _TupleSubclass(tuple):
    pass


class _HookModule:
    def __init__(
        self,
        path: str,
        *,
        removal_failures: int = 0,
        registration_error: BaseException | None = None,
        removal_log: list[str] | None = None,
    ) -> None:
        self.path = path
        self.callbacks: list[object] = []
        self.removals: list[str] = []
        self.removal_failures = removal_failures
        self.registration_error = registration_error
        self.removal_log = removal_log

    def register_forward_hook(self, callback: object) -> _Handle:
        if self.registration_error is not None:
            raise self.registration_error
        self.callbacks.append(callback)
        return _Handle(self, callback, self.removal_failures)

    def fire(self, inputs: tuple[object, ...], output: object) -> object:
        result: object = None
        for callback in tuple(self.callbacks):
            result = callback(self, inputs, output)
        return result


class _HookModel:
    def __init__(self, architecture: str, layout: str) -> None:
        if architecture == "mixtral":
            source = MixtralForCausalLM(layout=layout)
        elif architecture == "qwen3_5_moe":
            source = Qwen3_5MoeForConditionalGeneration(layout=layout)
        elif architecture == "qwen3_5_moe_text":
            source = Qwen3_5MoeForCausalLM(layout=layout)
        else:
            source = Qwen3MoeForCausalLM(layout=layout)
        self.config = source.config
        self._entries = list(source.named_modules())
        self.nodes: dict[str, _HookModule] = {}
        self.removal_log: list[str] = []
        for path, module in self._entries:
            if path and (path.endswith(".gate") or path.endswith(".mlp.gate")):
                node = _HookModule(path, removal_log=self.removal_log)
                self.nodes[path] = node
                self._entries[self._entries.index((path, module))] = (path, node)

    def named_modules(self):
        return iter(self._entries)


def _session(
    architecture: str = "mixtral",
    layout: str = "legacy",
    *,
    max_events: int = 8,
    decoder=None,
    model: object | None = None,
) -> tuple[RoutingCaptureSession, _HookModel, AdapterInspection]:
    inspection = _inspection(architecture, layout)
    plan = build_routing_probe_plan(inspection)
    hook_model = _HookModel(architecture, layout) if model is None else model
    if decoder is None:

        def decoder(
            context: RoutingCaptureTarget,
            module: object,
            inputs: tuple[object, ...],
            output: object,
        ):
            del module, inputs, output
            return (
                RoutingEvent(
                    token_key="token:" + "1" * 64,
                    layer_key=context.layer_key,
                    rank=0,
                    expert_key=context.expert_keys[0],
                    probability=0.5,
                    selected=True,
                ),
            )

    session = RoutingCaptureSession(
        hook_model,
        inspection,
        plan,
        decoder,
        max_events=max_events,
    )
    return session, hook_model, inspection


def _assert_failed_session_is_unpublished(
    session: RoutingCaptureSession, model: _HookModel
) -> None:
    assert all(node.callbacks == [] for node in model.nodes.values())
    assert session._events == []
    assert session._event_ids == set()
    assert session._truncated is False
    assert session._dropped_invocations == 0
    for accessor in (
        lambda: session.events,
        lambda: session.truncated,
        lambda: session.dropped_invocations,
    ):
        with pytest.raises(RoutingCaptureError) as exc_info:
            accessor()
        assert exc_info.value.stage == "lifecycle"


def test_public_api_and_exact_constructor() -> None:
    assert routing_module.__all__ == [
        "RoutingCaptureError",
        "RoutingCaptureSession",
        "RoutingCaptureTarget",
    ]
    constructor = inspect.signature(RoutingCaptureSession)
    assert tuple(constructor.parameters) == (
        "model",
        "inspection",
        "plan",
        "decoder",
        "max_events",
    )
    expected_kinds = (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )
    assert tuple(parameter.kind for parameter in constructor.parameters.values()) == expected_kinds
    assert all(
        parameter.default is inspect.Parameter.empty
        for parameter in constructor.parameters.values()
    )
    constructor_hints = get_type_hints(RoutingCaptureSession.__init__)
    assert constructor_hints == {
        "model": object,
        "inspection": AdapterInspection,
        "plan": ProbePlan,
        "decoder": Callable[
            [RoutingCaptureTarget, object, tuple[object, ...], object],
            tuple[RoutingEvent, ...],
        ],
        "max_events": int,
        "return": type(None),
    }
    assert issubclass(RoutingCaptureError, RuntimeError)
    for stage in ("preflight", "decode", "events", "lifecycle"):
        error = RoutingCaptureError(stage)
        assert error.stage == stage
        assert str(error) == f"routing capture failed at {stage}"
    with pytest.raises(ValueError, match="routing capture error stage"):
        RoutingCaptureError("secret-stage")


def test_public_surface_signatures_slots_and_exports_are_exact() -> None:
    runtime_module_exports = {
        name: getattr(runtime_package, name)
        for name in ("RoutingCaptureError", "RoutingCaptureSession", "RoutingCaptureTarget")
    }
    assert runtime_module_exports["RoutingCaptureError"] is RoutingCaptureError
    assert runtime_module_exports["RoutingCaptureSession"] is RoutingCaptureSession
    assert runtime_module_exports["RoutingCaptureTarget"] is RoutingCaptureTarget
    assert tuple(field.name for field in fields(RoutingCaptureTarget)) == (
        "router",
        "layer_key",
        "expert_keys",
        "routed_top_k",
    )
    assert RoutingCaptureTarget.__slots__ == (
        "router",
        "layer_key",
        "expert_keys",
        "routed_top_k",
    )
    public_session_members = tuple(
        name for name in dir(RoutingCaptureSession) if not name.startswith("_")
    )
    assert public_session_members == (
        "close",
        "dropped_invocations",
        "events",
        "plan",
        "truncated",
    )
    target_signature = inspect.signature(RoutingCaptureTarget)
    assert tuple(target_signature.parameters) == (
        "router",
        "layer_key",
        "expert_keys",
        "routed_top_k",
    )
    assert (
        tuple(parameter.kind for parameter in target_signature.parameters.values())
        == (inspect.Parameter.POSITIONAL_OR_KEYWORD,) * 4
    )
    assert all(
        parameter.default is inspect.Parameter.empty
        for parameter in target_signature.parameters.values()
    )
    assert get_type_hints(RoutingCaptureTarget) == {
        "router": routing_module.ProbeTarget,
        "layer_key": str,
        "expert_keys": tuple[str, ...],
        "routed_top_k": int,
    }
    enter_signature = inspect.signature(RoutingCaptureSession.__enter__)
    assert tuple(enter_signature.parameters) == ("self",)
    assert tuple(parameter.kind for parameter in enter_signature.parameters.values()) == (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    assert all(
        parameter.default is inspect.Parameter.empty
        for parameter in enter_signature.parameters.values()
    )
    assert get_type_hints(RoutingCaptureSession.__enter__) == {
        "return": RoutingCaptureSession,
    }
    exit_signature = inspect.signature(RoutingCaptureSession.__exit__)
    assert tuple(exit_signature.parameters) == (
        "self",
        "exc_type",
        "exc_value",
        "traceback",
    )
    assert (
        tuple(parameter.kind for parameter in exit_signature.parameters.values())
        == (inspect.Parameter.POSITIONAL_OR_KEYWORD,) * 4
    )
    assert all(
        parameter.default is inspect.Parameter.empty
        for parameter in exit_signature.parameters.values()
    )
    assert get_type_hints(RoutingCaptureSession.__exit__) == {
        "exc_type": type[BaseException] | None,
        "exc_value": BaseException | None,
        "traceback": object,
        "return": bool,
    }
    close_signature = inspect.signature(RoutingCaptureSession.close)
    assert tuple(close_signature.parameters) == ("self",)
    assert tuple(parameter.kind for parameter in close_signature.parameters.values()) == (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    assert get_type_hints(RoutingCaptureSession.close) == {"return": type(None)}
    assert get_type_hints(RoutingCaptureSession.plan.fget) == {"return": ProbePlan}
    assert get_type_hints(RoutingCaptureSession.events.fget) == {"return": tuple[RoutingEvent, ...]}
    assert get_type_hints(RoutingCaptureSession.truncated.fget) == {"return": bool}
    assert get_type_hints(RoutingCaptureSession.dropped_invocations.fget) == {"return": int}


@pytest.mark.parametrize(
    ("architecture", "layout"),
    [
        ("mixtral", "legacy"),
        ("mixtral", "packed"),
        ("qwen3_moe", "legacy_indexed"),
        ("qwen3_moe", "packed"),
        ("qwen3_5_moe", "packed"),
        ("qwen3_5_moe_text", "packed"),
    ],
)
def test_successful_session_passes_exact_context_and_hook_arguments(
    architecture: str, layout: str
) -> None:
    calls: list[tuple[RoutingCaptureTarget, object, tuple[object, ...], object]] = []

    def decoder(context, module, inputs, output):
        calls.append((context, module, inputs, output))
        return (
            RoutingEvent(
                token_key="token:" + "2" * 64,
                layer_key=context.layer_key,
                rank=0,
                expert_key=context.expert_keys[0],
                probability=0.75,
                selected=True,
            ),
        )

    session, model, inspection = _session(architecture, layout, decoder=decoder, max_events=16)
    expected_routers = {
        component.module_path: component
        for component in inspection.report.components
        if component.kind.value == "router"
    }
    assert set(session._contexts) == set(expected_routers)
    for path, context in session._contexts.items():
        router = expected_routers[path]
        plan_target = next(target for target in session.plan.targets if target.module_path == path)
        expected_layer = next(
            component
            for component in inspection.report.components
            if component.kind.value == "moe_layer" and component.layer_index == router.layer_index
        )
        expected_experts = sorted(
            (
                component
                for component in inspection.report.components
                if component.kind.value == "expert" and component.layer_index == router.layer_index
            ),
            key=lambda component: component.expert_index,
        )
        assert context.router is plan_target
        assert context.router.component_key == router.component_key
        assert context.layer_key == expected_layer.component_key
        assert context.expert_keys == tuple(
            component.component_key for component in expected_experts
        )
        assert context.routed_top_k == inspection.report.facts.routed_top_k
    original_inputs = ("input",)
    original_output = object()
    with session as entered:
        assert entered is session
        for path, node in model.nodes.items():
            assert node.fire(original_inputs, original_output) is None
            assert calls[-1][1] is node
            assert calls[-1][2] is original_inputs
            assert calls[-1][3] is original_output

        with pytest.raises(RoutingCaptureError) as exc_info:
            _ = session.events
        assert exc_info.value.stage == "lifecycle"

    assert len(session.events) == len(model.nodes)
    assert session.truncated is False
    assert session.dropped_invocations == 0
    assert session.plan == build_routing_probe_plan(inspection)
    assert session._manager is not None
    assert session._manager.resolved_plan is not None
    assert tuple(
        (path, hook_point, module)
        for path, hook_point, module in session._manager.resolved_plan.bindings
    ) == tuple((path, HookPoint.FORWARD, model.nodes[path]) for path in expected_routers)
    assert all(node.callbacks == [] for node in model.nodes.values())


@pytest.mark.parametrize("architecture", ["qwen3_5_moe", "qwen3_5_moe_text"])
def test_shared_experts_are_validated_but_excluded_from_routing_context(
    architecture: str,
) -> None:
    session, model, inspection = _session(architecture, "packed")
    shared = [
        component
        for component in inspection.report.components
        if component.kind is ComponentKind.SHARED_EXPERT
    ]
    assert shared
    assert all(component.shared is True and component.routed is False for component in shared)
    layer_by_key = {
        component.component_key: component.layer_index
        for component in inspection.report.components
        if component.kind is ComponentKind.MOE_LAYER
    }
    for context in session._contexts.values():
        layer_index = layer_by_key[context.layer_key]
        same_layer_shared = [
            component for component in shared if component.layer_index == layer_index
        ]
        assert same_layer_shared
        assert all(
            component.component_key not in context.expert_keys for component in same_layer_shared
        )
    assert all(node.callbacks == [] for node in model.nodes.values())


def _tamper_qwen35_shared_report(inspection: AdapterInspection, case: str) -> AdapterInspection:
    components = list(inspection.report.components)
    shared_position = next(
        index
        for index, component in enumerate(components)
        if component.kind is ComponentKind.SHARED_EXPERT
    )
    shared = components[shared_position]
    report_update: dict[str, object] = {}
    if case == "count_mismatch":
        report_update["facts"] = inspection.report.facts.model_copy(
            update={"shared_expert_count": 0, "shared_expert_count_source": "fixture.none"}
        )
    elif case == "missing_count_with_shared":
        report_update["facts"] = inspection.report.facts.model_copy(
            update={"shared_expert_count": None, "shared_expert_count_source": None}
        )
    elif case == "positive_count_missing":
        del components[shared_position]
    elif case == "positive_count_extra":
        components.append(shared.model_copy(update={"module_path": shared.module_path + ".extra"}))
    elif case == "duplicate_path":
        components.append(shared)
        report_update["facts"] = inspection.report.facts.model_copy(
            update={"shared_expert_count": 2, "shared_expert_count_source": "fixture.duplicate"}
        )
    elif case == "wrong_shared_flag":
        components[shared_position] = shared.model_copy(update={"shared": False})
    elif case == "wrong_routed_flag":
        components[shared_position] = shared.model_copy(update={"routed": True})
    elif case == "wrong_index":
        components[shared_position] = shared.model_copy(update={"expert_index": 0})
    elif case == "wrong_layer":
        components[shared_position] = shared.model_copy(update={"layer_index": 1})
    else:
        raise AssertionError(f"unknown shared tamper case: {case}")
    report_update["components"] = components
    return inspection.model_copy(
        update={"report": inspection.report.model_copy(update=report_update)}
    )


@pytest.mark.parametrize(
    "case",
    ["count_mismatch", "wrong_shared_flag", "wrong_routed_flag", "wrong_index", "wrong_layer"],
)
def test_shared_count_flags_index_and_layer_are_model_neutral_contracts(
    case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ForbiddenModel:
        @property
        def named_modules(self):
            raise AssertionError("invalid shared-expert evidence must not traverse the model")

    base = _inspection("qwen3_5_moe", "packed")
    inspection = _tamper_qwen35_shared_report(base, case)
    plan = build_routing_probe_plan(base)

    def return_tampered(cls, payload: object) -> AdapterInspection:
        del cls, payload
        return inspection

    monkeypatch.setattr(AdapterInspection, "model_validate", classmethod(return_tampered))
    with pytest.raises(RoutingCaptureError) as exc_info:
        RoutingCaptureSession(ForbiddenModel(), base, plan, lambda *args: (), max_events=1)
    assert exc_info.value.stage == "preflight"


@pytest.mark.parametrize(
    "case", ["missing_count_with_shared", "positive_count_missing", "positive_count_extra"]
)
def test_shared_count_mismatches_fail_during_preflight_before_traversal(
    case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ForbiddenModel:
        @property
        def named_modules(self):
            raise AssertionError("shared count validation must precede model traversal")

    base = _inspection("qwen3_5_moe", "packed")
    inspection = _tamper_qwen35_shared_report(base, case)
    plan = build_routing_probe_plan(base)

    def return_tampered(cls, payload: object) -> AdapterInspection:
        del cls, payload
        return inspection

    monkeypatch.setattr(AdapterInspection, "model_validate", classmethod(return_tampered))
    with pytest.raises(RoutingCaptureError) as exc_info:
        RoutingCaptureSession(ForbiddenModel(), base, plan, lambda *args: (), max_events=1)
    assert exc_info.value.stage == "preflight"


def test_duplicate_shared_module_paths_fail_during_preflight_before_traversal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ForbiddenModel:
        @property
        def named_modules(self):
            raise AssertionError("duplicate shared paths must precede model traversal")

    base = _inspection("qwen3_5_moe", "packed")
    inspection = _tamper_qwen35_shared_report(base, "duplicate_path")
    plan = build_routing_probe_plan(base)

    def return_tampered(cls, payload: object) -> AdapterInspection:
        del cls, payload
        return inspection

    monkeypatch.setattr(AdapterInspection, "model_validate", classmethod(return_tampered))
    with pytest.raises(RoutingCaptureError) as exc_info:
        RoutingCaptureSession(ForbiddenModel(), base, plan, lambda *args: (), max_events=1)
    assert exc_info.value.stage == "preflight"


def test_missing_shared_count_normalizes_to_zero_and_explicit_zero_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _inspection("mixtral", "legacy")
    assert base.report.facts.shared_expert_count is None
    zero_facts = base.report.facts.model_copy(
        update={"shared_expert_count": 0, "shared_expert_count_source": "fixture.none"}
    )
    zero_inspection = base.model_copy(
        update={"report": base.report.model_copy(update={"facts": zero_facts})}
    )
    plan = build_routing_probe_plan(base)
    model = _HookModel("mixtral", "legacy")

    def return_zero(cls, payload: object) -> AdapterInspection:
        del cls, payload
        return zero_inspection

    monkeypatch.setattr(AdapterInspection, "model_validate", classmethod(return_zero))
    session = RoutingCaptureSession(model, base, plan, lambda *args: (), max_events=1)
    with session:
        pass
    assert session.events == ()


def test_shared_component_cannot_be_a_routing_target() -> None:
    inspection = _inspection("qwen3_5_moe", "packed")
    plan = build_routing_probe_plan(inspection)
    shared = next(
        component
        for component in inspection.report.components
        if component.kind is ComponentKind.SHARED_EXPERT
    )
    target = plan.targets[0]
    shared_target = target.model_copy(
        update={"component_key": shared.component_key, "module_path": shared.module_path}
    )
    tampered_plan = plan.model_copy(update={"targets": (shared_target, *plan.targets[1:])})
    with pytest.raises(ValueError, match="router"):
        routing_module._build_contexts(inspection, tampered_plan)


def test_shared_component_cannot_emit_a_routing_event() -> None:
    session, model, inspection = _session("qwen3_5_moe", "packed", max_events=4)
    first_path = next(iter(model.nodes))
    router = next(
        component
        for component in inspection.report.components
        if component.kind is ComponentKind.ROUTER and component.module_path == first_path
    )
    shared_key = next(
        component.component_key
        for component in inspection.report.components
        if component.kind is ComponentKind.SHARED_EXPERT
        and component.layer_index == router.layer_index
    )

    def shared_event_decoder(context, module, inputs, output):
        del module, inputs, output
        return (
            RoutingEvent(
                token_key="token:" + "f" * 64,
                layer_key=context.layer_key,
                rank=0,
                expert_key=shared_key,
                probability=0.5,
                selected=True,
            ),
        )

    session = RoutingCaptureSession(
        model,
        inspection,
        build_routing_probe_plan(inspection),
        shared_event_decoder,
        max_events=4,
    )
    with pytest.raises(RoutingCaptureError) as exc_info:
        with session:
            model.nodes[first_path].fire((), object())
    assert exc_info.value.stage == "events"
    assert session._events == []
    assert all(node.callbacks == [] for node in model.nodes.values())


@pytest.mark.parametrize("architecture", ["qwen3_5_moe", "qwen3_5_moe_text"])
def test_qwen35_both_roots_multi_layer_cleanup_pending_is_retryable(architecture: str) -> None:
    session, model, _ = _session(architecture, "packed", max_events=8)
    first_path = next(iter(model.nodes))
    model.nodes[first_path].removal_failures = 1
    with pytest.raises(RoutingCaptureError) as exc_info:
        with session:
            next(iter(model.nodes.values())).fire((), object())
    assert exc_info.value.stage == "lifecycle"
    assert all(node.callbacks == [] for path, node in model.nodes.items() if path != first_path)
    assert model.nodes[first_path].callbacks
    with pytest.raises(RoutingCaptureError):
        _ = session.events
    session.close()
    assert all(node.callbacks == [] for node in model.nodes.values())
    assert len(session.events) == 1
    session.close()


def test_future_descriptor_family_is_accepted_without_allowlist() -> None:
    inspection = _future_inspection()
    model = _HookModel("mixtral", "legacy")
    plan = build_routing_probe_plan(inspection)
    session = RoutingCaptureSession(
        model,
        inspection,
        plan,
        lambda context, module, inputs, output: (),
        max_events=1,
    )
    with session:
        pass
    assert session.events == ()


def test_normal_cleanup_is_reverse_registration_order_and_close_is_idempotent() -> None:
    session, model, _ = _session()
    expected_order = list(reversed(tuple(model.nodes)))
    with session:
        pass
    assert model.removal_log == expected_order
    session.close()
    assert model.removal_log == expected_order


def test_quota_is_atomic_and_skips_decoder_after_full() -> None:
    calls = 0

    def one_event(context, module, inputs, output):
        nonlocal calls
        calls += 1
        return (
            RoutingEvent(
                token_key="token:" + "3" * 64,
                layer_key=context.layer_key,
                rank=0,
                expert_key=context.expert_keys[0],
                probability=0.5,
                selected=True,
            ),
        )

    session, model, _ = _session(max_events=1, decoder=one_event)
    with session:
        next(iter(model.nodes.values())).fire((), object())
        next(iter(model.nodes.values())).fire((), object())
    assert calls == 1
    assert len(session.events) == 1
    assert session.truncated is True
    assert session.dropped_invocations == 1


def test_over_quota_tuple_is_dropped_before_event_iteration_or_revalidation() -> None:
    class ForbiddenEvent:
        def model_dump(self, *args: object, **kwargs: object) -> object:
            raise AssertionError("over-quota events must not be inspected")

    calls = 0

    def decoder(context, module, inputs, output):
        nonlocal calls
        calls += 1
        del context, module, inputs, output
        return (ForbiddenEvent(), ForbiddenEvent())

    session, model, _ = _session(max_events=1, decoder=decoder)
    with session:
        next(iter(model.nodes.values())).fire((), object())
    assert calls == 1
    assert session.events == ()
    assert session.truncated is True
    assert session.dropped_invocations == 1


def test_exact_fit_is_retained_and_event_values_are_freshly_revalidated() -> None:
    session, model, _ = _session(max_events=1)
    with session:
        next(iter(model.nodes.values())).fire((), object())
    assert len(session.events) == 1
    assert type(session.events[0]) is RoutingEvent


@pytest.mark.parametrize(
    "bad_result",
    [
        ["not-a-tuple"],
        (object(),),
    ],
)
def test_decoder_result_and_event_binding_fail_at_events(bad_result: object) -> None:
    def decoder(context, module, inputs, output):
        return bad_result

    session, model, _ = _session(decoder=decoder)
    with pytest.raises(RoutingCaptureError) as exc_info:
        with session:
            next(iter(model.nodes.values())).fire((), object())
    assert exc_info.value.stage == "events"
    with pytest.raises(RoutingCaptureError) as lifecycle_error:
        _ = session.events
    assert lifecycle_error.value.stage == "lifecycle"


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
def test_decoder_control_flow_exceptions_are_exact(error_type: type[BaseException]) -> None:
    error = error_type("SECRET_ROUTING_CONTROL_FLOW")

    def decoder(context, module, inputs, output):
        raise error

    session, model, _ = _session(decoder=decoder)
    with pytest.raises(error_type) as exc_info:
        with session:
            next(iter(model.nodes.values())).fire((), object())
    assert exc_info.value is error


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
def test_decoder_control_flow_after_staging_rolls_back_and_cleans_every_hook(
    error_type: type[BaseException],
) -> None:
    error = error_type("SECRET_ROUTING_CONTROL_FLOW_AFTER_EVENT")
    calls = 0

    def decoder(context, module, inputs, output):
        nonlocal calls
        del module, inputs, output
        calls += 1
        if calls == 2:
            raise error
        return (
            RoutingEvent(
                token_key="token:" + "e" * 64,
                layer_key=context.layer_key,
                rank=0,
                expert_key=context.expert_keys[0],
                probability=0.5,
                selected=True,
            ),
        )

    session, model, _ = _session(decoder=decoder)
    nodes = tuple(model.nodes.values())
    with pytest.raises(error_type) as exc_info:
        with session:
            nodes[0].fire((), object())
            nodes[1].fire((), object())
    assert exc_info.value is error
    assert calls == 2
    _assert_failed_session_is_unpublished(session, model)
    assert model.removal_log == list(reversed(tuple(model.nodes)))


def test_decoder_ordinary_error_is_safe_and_chained() -> None:
    error = ValueError("TOP_SECRET_ROUTING_VALUE")

    def decoder(context, module, inputs, output):
        raise error

    session, model, _ = _session(decoder=decoder)
    with pytest.raises(RoutingCaptureError) as exc_info:
        with session:
            next(iter(model.nodes.values())).fire((), object())
    assert exc_info.value.stage == "decode"
    assert str(exc_info.value) == "routing capture failed at decode"
    assert exc_info.value.__cause__ is error
    assert "TOP_SECRET_ROUTING_VALUE" not in str(exc_info.value)


def test_decoder_routing_capture_error_is_rewrapped_at_decode_boundary() -> None:
    error = RoutingCaptureError("decode")

    def decoder(context, module, inputs, output):
        del context, module, inputs, output
        raise error

    session, model, _ = _session(decoder=decoder)
    with pytest.raises(RoutingCaptureError) as exc_info:
        with session:
            next(iter(model.nodes.values())).fire((), object())
    assert exc_info.value is not error
    assert exc_info.value.stage == "decode"
    assert str(exc_info.value) == "routing capture failed at decode"
    assert exc_info.value.__cause__ is error


def test_event_validation_routing_capture_error_is_rewrapped_at_events_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = RoutingCaptureError("events")

    def decoder(context, module, inputs, output):
        del module, inputs, output
        return (
            RoutingEvent(
                token_key="token:" + "b" * 64,
                layer_key=context.layer_key,
                rank=0,
                expert_key=context.expert_keys[0],
                probability=0.5,
                selected=True,
            ),
        )

    session, model, _ = _session(decoder=decoder)

    def failing_dump(self, *args: object, **kwargs: object) -> object:
        del self, args, kwargs
        raise error

    monkeypatch.setattr(RoutingEvent, "model_dump", failing_dump)
    with pytest.raises(RoutingCaptureError) as exc_info:
        with session:
            next(iter(model.nodes.values())).fire((), object())
    assert exc_info.value is not error
    assert exc_info.value.stage == "events"
    assert exc_info.value.__cause__ is error


@pytest.mark.parametrize("container", ["list", "generator", "mapping", "tuple_subclass"])
def test_decoder_output_requires_an_exact_tuple_before_event_validation(container: str) -> None:
    def decoder(context, module, inputs, output):
        event = RoutingEvent(
            token_key="token:" + "7" * 64,
            layer_key=context.layer_key,
            rank=0,
            expert_key=context.expert_keys[0],
            probability=0.5,
            selected=True,
        )
        if container == "list":
            return [event]
        if container == "generator":
            return (value for value in (event,))
        if container == "mapping":
            return {"event": event}

        class TupleSubclass(tuple):
            pass

        return TupleSubclass((event,))

    session, model, _ = _session(decoder=decoder)
    with pytest.raises(RoutingCaptureError) as exc_info:
        with session:
            next(iter(model.nodes.values())).fire((), object())
    assert exc_info.value.stage == "events"


def test_event_subclasses_and_within_invocation_duplicates_are_rejected() -> None:
    class EventSubclass(RoutingEvent):
        pass

    def decoder(context, module, inputs, output):
        del module, inputs, output
        event = EventSubclass(
            token_key="token:" + "8" * 64,
            layer_key=context.layer_key,
            rank=0,
            expert_key=context.expert_keys[0],
            probability=0.5,
            selected=True,
        )
        return (event,)

    session, model, _ = _session(decoder=decoder)
    with pytest.raises(RoutingCaptureError) as exc_info:
        with session:
            next(iter(model.nodes.values())).fire((), object())
    assert exc_info.value.stage == "events"


def test_event_wrong_layer_unknown_expert_and_cross_model_are_rejected() -> None:
    inspection = _inspection("mixtral", "legacy")
    other_inspection = _inspection("qwen3_moe", "legacy_indexed")
    wrong_layer = next(
        component
        for component in inspection.report.components
        if component.kind.value == "moe_layer" and component.layer_index == 1
    )
    unknown_expert = "component:" + "e" * 64
    other_layer = next(
        component
        for component in other_inspection.report.components
        if component.kind.value == "moe_layer" and component.layer_index == 1
    )
    other_expert = next(
        component
        for component in other_inspection.report.components
        if component.kind.value == "expert" and component.layer_index == 1
    )

    for layer_key, expert_key in (
        (wrong_layer.component_key, None),
        (None, unknown_expert),
        (other_layer.component_key, other_expert.component_key),
    ):
        session, model, _ = _session()

        def decoder(context, module, inputs, output, *, layer_key=layer_key, expert_key=expert_key):
            del module, inputs, output
            return (
                RoutingEvent(
                    token_key="token:" + "a" * 64,
                    layer_key=layer_key or context.layer_key,
                    rank=0,
                    expert_key=expert_key or context.expert_keys[0],
                    probability=0.5,
                    selected=True,
                ),
            )

        # Construct the session directly so the event can be varied without
        # changing its canonical preflight evidence.
        session = RoutingCaptureSession(
            model,
            inspection,
            build_routing_probe_plan(inspection),
            decoder,
            max_events=1,
        )
        with pytest.raises(RoutingCaptureError) as exc_info:
            with session:
                next(iter(model.nodes.values())).fire((), object())
        assert exc_info.value.stage == "events"

    def duplicate_decoder(context, module, inputs, output):
        del module, inputs, output
        event = RoutingEvent(
            token_key="token:" + "9" * 64,
            layer_key=context.layer_key,
            rank=0,
            expert_key=context.expert_keys[0],
            probability=0.5,
            selected=True,
        )
        return (event, event)

    session, model, _ = _session(decoder=duplicate_decoder)
    with pytest.raises(RoutingCaptureError) as exc_info:
        with session:
            next(iter(model.nodes.values())).fire((), object())
    assert exc_info.value.stage == "events"


@pytest.mark.parametrize(
    "failure_kind",
    [
        "bad_container",
        "bad_object",
        "tuple_subclass",
        "event_subclass",
        "wrong_layer",
        "unknown_expert",
        "cross_model",
        "duplicate",
        "invalid_selected",
    ],
)
def test_event_failure_families_rollback_staged_events_and_remove_all_hooks(
    failure_kind: str,
) -> None:
    base_inspection = _inspection("mixtral", "legacy")
    other_inspection = _inspection("qwen3_moe", "legacy_indexed")
    wrong_layer = next(
        component
        for component in base_inspection.report.components
        if component.kind is ComponentKind.MOE_LAYER and component.layer_index == 0
    ).component_key
    other_expert = next(
        component
        for component in other_inspection.report.components
        if component.kind is ComponentKind.EXPERT and component.layer_index == 1
    ).component_key
    unknown_expert = "component:" + "f" * 64
    calls = 0

    def make_event(context, *, token_hex: str, layer_key=None, expert_key=None):
        return RoutingEvent(
            token_key="token:" + token_hex,
            layer_key=layer_key or context.layer_key,
            rank=0,
            expert_key=expert_key or context.expert_keys[0],
            probability=0.5,
            selected=True,
        )

    def decoder(context, module, inputs, output):
        nonlocal calls
        del module, inputs, output
        calls += 1
        if calls == 1:
            return (make_event(context, token_hex="1" * 64),)
        event = make_event(context, token_hex="2" * 64)
        if failure_kind == "bad_container":
            return [event]
        if failure_kind == "bad_object":
            return (object(),)
        if failure_kind == "tuple_subclass":

            class TupleSubclass(tuple):
                pass

            return TupleSubclass((event,))
        if failure_kind == "event_subclass":

            class EventSubclass(RoutingEvent):
                pass

            return (EventSubclass.model_validate(event.model_dump(mode="json")),)
        if failure_kind == "wrong_layer":
            return (make_event(context, token_hex="2" * 64, layer_key=wrong_layer),)
        if failure_kind == "unknown_expert":
            return (make_event(context, token_hex="2" * 64, expert_key=unknown_expert),)
        if failure_kind == "cross_model":
            return (make_event(context, token_hex="2" * 64, expert_key=other_expert),)
        if failure_kind == "duplicate":
            return (make_event(context, token_hex="1" * 64),)
        if failure_kind == "invalid_selected":
            return (event.model_copy(update={"selected": "not-bool"}),)
        raise AssertionError(f"unknown failure kind: {failure_kind}")

    session, model, _ = _session(decoder=decoder, max_events=4)
    nodes = tuple(model.nodes.values())
    with pytest.raises(RoutingCaptureError) as exc_info:
        with session:
            nodes[0].fire((), object())
            (nodes[0] if failure_kind == "duplicate" else nodes[1]).fire((), object())
    assert exc_info.value.stage == "events"
    assert calls == 2
    _assert_failed_session_is_unpublished(session, model)
    assert model.removal_log == list(reversed(tuple(model.nodes)))


def test_injected_event_validation_failure_after_staging_rolls_back_everything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    injected = ValueError("TOP_SECRET_EVENT_VALIDATION")
    original_dump = RoutingEvent.model_dump
    dump_calls = 0

    def failing_dump(self, *args: object, **kwargs: object) -> object:
        nonlocal dump_calls
        dump_calls += 1
        if dump_calls == 2:
            raise injected
        return original_dump(self, *args, **kwargs)

    monkeypatch.setattr(RoutingEvent, "model_dump", failing_dump)
    calls = 0

    def decoder(context, module, inputs, output):
        nonlocal calls
        del module, inputs, output
        calls += 1
        return (
            RoutingEvent(
                token_key="token:" + ("3" if calls == 1 else "4") * 64,
                layer_key=context.layer_key,
                rank=0,
                expert_key=context.expert_keys[0],
                probability=0.5,
                selected=True,
            ),
        )

    session, model, _ = _session(decoder=decoder, max_events=4)
    nodes = tuple(model.nodes.values())
    with pytest.raises(RoutingCaptureError) as exc_info:
        with session:
            nodes[0].fire((), object())
            nodes[1].fire((), object())
    assert exc_info.value.stage == "events"
    assert exc_info.value.__cause__ is injected
    assert calls == 2
    _assert_failed_session_is_unpublished(session, model)
    assert model.removal_log == list(reversed(tuple(model.nodes)))


@pytest.mark.parametrize("invocation", [((object(),), {}, "decode"), ((), {"x": 1}, "decode")])
def test_callback_arity_and_input_tuple_are_decode_errors(invocation) -> None:
    args, kwargs, stage = invocation
    session, model, _ = _session()
    node = next(iter(model.nodes.values()))
    with pytest.raises(RoutingCaptureError) as exc_info:
        with session:
            for installed in tuple(node.callbacks):
                installed(*(args), **kwargs)
    assert exc_info.value.stage == stage


@pytest.mark.parametrize("inputs", [["not", "tuple"], _TupleSubclass(("x",))])
def test_non_tuple_inputs_are_decode_error(inputs: object) -> None:
    session, model, _ = _session()
    with pytest.raises(RoutingCaptureError) as exc_info:
        with session:
            next(iter(model.nodes.values())).fire(inputs, object())
    assert exc_info.value.stage == "decode"


def test_cross_layer_unknown_and_duplicate_events_are_rejected() -> None:
    session, model, _ = _session()
    node = next(iter(model.nodes.values()))
    with pytest.raises(RoutingCaptureError) as exc_info:
        with session:
            node.fire((), object())
            node.fire((), object())
    assert exc_info.value.stage == "events"
    assert session._events == []
    assert session._event_ids == set()


def test_preflight_rejects_tampered_plan_before_model_traversal() -> None:
    class ForbiddenModel:
        @property
        def named_modules(self):
            raise AssertionError("model traversal must not occur during preflight")

    inspection = _inspection("mixtral", "legacy")
    plan = build_routing_probe_plan(inspection)
    tampered = plan.model_copy(update={"include": ("layers.999.gate",)})
    with pytest.raises(RoutingCaptureError) as exc_info:
        RoutingCaptureSession(
            ForbiddenModel(), inspection, tampered, lambda *args: (), max_events=1
        )
    assert exc_info.value.stage == "preflight"


class _InspectionSubclass(AdapterInspection):
    pass


class _PlanSubclass(ProbePlan):
    pass


def test_preflight_requires_exact_inspection_and_plan_types_before_traversal() -> None:
    class ForbiddenModel:
        @property
        def named_modules(self):
            raise AssertionError("preflight must not traverse the model")

        def forward(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("preflight must not forward")

        def generate(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("preflight must not generate")

        @property
        def tokenizer(self):
            raise AssertionError("preflight must not touch tokenizer")

    inspection = _inspection("mixtral", "legacy")
    plan = build_routing_probe_plan(inspection)
    inspection_subclass = _InspectionSubclass.model_validate(inspection.model_dump(mode="json"))

    plan_subclass = _PlanSubclass.model_validate(plan.model_dump(mode="json"))
    cases = (
        (inspection.report, plan),
        (inspection_subclass, plan),
        (inspection, object()),
        (inspection, plan_subclass),
    )
    for bad_inspection, bad_plan in cases:
        with pytest.raises(RoutingCaptureError) as exc_info:
            RoutingCaptureSession(
                ForbiddenModel(), bad_inspection, bad_plan, lambda *args: (), max_events=1
            )
        assert exc_info.value.stage == "preflight"


@pytest.mark.parametrize(
    "update",
    [
        {"include": ("missing",)},
        {"exclude": ("missing",)},
        {"targets": ()},
        {"hook_points": ()},
        {"level": ProbeLevel.EXPERT_ACTIVITY},
    ],
)
def test_all_plan_tampering_is_rejected_before_traversal(update: dict[str, object]) -> None:
    class ForbiddenModel:
        @property
        def named_modules(self):
            raise AssertionError("plan validation must precede model traversal")

    inspection = _inspection("mixtral", "legacy")
    plan = build_routing_probe_plan(inspection).model_copy(update=update)
    with pytest.raises(RoutingCaptureError) as exc_info:
        RoutingCaptureSession(ForbiddenModel(), inspection, plan, lambda *args: (), max_events=1)
    assert exc_info.value.stage == "preflight"


@pytest.mark.parametrize(
    "update_kind",
    [
        "descriptor",
        "detection",
        "report",
    ],
)
def test_tampered_inspection_is_rejected_before_traversal(update_kind: str) -> None:
    class ForbiddenModel:
        @property
        def named_modules(self):
            raise AssertionError("inspection validation must precede model traversal")

    base = _inspection("mixtral", "legacy")
    updates: dict[str, object]
    if update_kind == "descriptor":
        updates = {"descriptor": base.descriptor.model_copy(update={"name": "tampered"})}
    elif update_kind == "detection":
        updates = {
            "detection": base.detection.model_copy(
                update={"score": 0.0, "evidence": (), "warnings": ("ambiguous",)}
            )
        }
    else:
        updates = {"report": base.report.model_copy(update={"model_key": "model:tampered@r1"})}
    inspection = base.model_copy(update=updates)
    plan = build_routing_probe_plan(base)
    with pytest.raises(RoutingCaptureError) as exc_info:
        RoutingCaptureSession(ForbiddenModel(), inspection, plan, lambda *args: (), max_events=1)
    assert exc_info.value.stage == "preflight"


def _tamper_report(inspection: AdapterInspection, case: str) -> AdapterInspection:
    components = list(inspection.report.components)
    router_positions = [
        index
        for index, component in enumerate(components)
        if component.kind is ComponentKind.ROUTER
    ]
    expert_positions = [
        index
        for index, component in enumerate(components)
        if component.kind is ComponentKind.EXPERT and component.layer_index == 0
    ]
    first_router = router_positions[0]
    first_expert = expert_positions[0]
    if case == "missing_router":
        components = [
            component for component in components if component.kind is not ComponentKind.ROUTER
        ]
    elif case == "multiple_router_path":
        second_router = router_positions[1]
        components[second_router] = components[second_router].model_copy(
            update={"module_path": components[first_router].module_path}
        )
    elif case == "wrong_kind":
        components[first_router] = components[first_router].model_copy(
            update={"kind": ComponentKind.EXPERT}
        )
    elif case == "missing_layer":
        components[first_router] = components[first_router].model_copy(update={"layer_index": None})
    elif case == "shared":
        components[first_expert] = components[first_expert].model_copy(update={"shared": True})
    elif case == "nonrouted":
        components[first_expert] = components[first_expert].model_copy(update={"routed": False})
    elif case == "nonindexed":
        components[first_expert] = components[first_expert].model_copy(
            update={"expert_index": None}
        )
    elif case == "duplicate_index":
        second_expert = expert_positions[1]
        components[first_expert] = components[first_expert].model_copy(
            update={"expert_index": components[second_expert].expert_index}
        )
    elif case == "noncontiguous_index":
        components[first_expert] = components[first_expert].model_copy(update={"expert_index": 99})
    elif case == "zero_layer":
        components = [
            component
            for component in components
            if not (component.kind is ComponentKind.MOE_LAYER and component.layer_index == 0)
        ]
    elif case == "multiple_layer":
        layer = next(
            component
            for component in components
            if component.kind is ComponentKind.MOE_LAYER and component.layer_index == 0
        )
        components.append(layer.model_copy(update={"component_key": "component:" + "d" * 64}))
    elif case == "missing_expert":
        del components[first_expert]
    elif case == "extra_expert":
        expert = components[first_expert]
        components.append(
            expert.model_copy(update={"component_key": "component:" + "e" * 64, "expert_index": 99})
        )
    elif case == "bad_path":
        components[first_router] = components[first_router].model_copy(
            update={"module_path": "layers..router"}
        )
    elif case not in {
        "bad_count",
        "bad_count_none",
        "bad_count_bool",
        "bad_count_string",
        "bad_count_zero",
        "bad_count_negative",
        "bad_top_k_bool",
        "bad_top_k_zero",
        "bad_top_k_negative",
        "bad_top_k_large",
        "bad_top_k_none",
    }:
        raise AssertionError(f"unknown tamper case: {case}")

    report_update: dict[str, object] = {"components": components}
    if case == "bad_count":
        report_update["facts"] = inspection.report.facts.model_copy(
            update={"expert_count": inspection.report.facts.expert_count + 1}
        )
    if case.startswith("bad_count"):
        value = {
            "bad_count_none": None,
            "bad_count_bool": True,
            "bad_count_string": "4",
            "bad_count_zero": 0,
            "bad_count_negative": -1,
        }.get(case)
        if value is not None or case == "bad_count_none":
            report_update["facts"] = inspection.report.facts.model_copy(
                update={"expert_count": value}
            )
    if case.startswith("bad_top_k"):
        value = {
            "bad_top_k_bool": True,
            "bad_top_k_zero": 0,
            "bad_top_k_negative": -1,
            "bad_top_k_large": inspection.report.facts.expert_count + 1,
            "bad_top_k_none": None,
        }[case]
        report_update["facts"] = inspection.report.facts.model_copy(update={"routed_top_k": value})
    return inspection.model_copy(
        update={"report": inspection.report.model_copy(update=report_update)}
    )


@pytest.mark.parametrize(
    "case",
    [
        "missing_router",
        "multiple_router_path",
        "wrong_kind",
        "missing_layer",
        "shared",
        "nonrouted",
        "nonindexed",
        "duplicate_index",
        "noncontiguous_index",
        "zero_layer",
        "multiple_layer",
        "missing_expert",
        "extra_expert",
        "bad_path",
        "bad_count",
        "bad_count_none",
        "bad_count_bool",
        "bad_count_string",
        "bad_count_zero",
        "bad_count_negative",
        "bad_top_k_bool",
        "bad_top_k_zero",
        "bad_top_k_negative",
        "bad_top_k_large",
        "bad_top_k_none",
    ],
)
def test_invalid_component_mapping_and_facts_fail_before_model_traversal(
    case: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ForbiddenModel:
        @property
        def named_modules(self):
            raise AssertionError("invalid static evidence must not traverse the model")

    inspection = _tamper_report(_inspection("mixtral", "legacy"), case)
    plan = build_routing_probe_plan(_inspection("mixtral", "legacy"))

    def return_tampered(cls, payload: object) -> AdapterInspection:
        del cls, payload
        return inspection

    # The unvalidated copy intentionally represents an adversarial report.  It
    # is injected only after the public schema boundary so this test exercises
    # context validation without allowing model traversal.
    monkeypatch.setattr(AdapterInspection, "model_validate", classmethod(return_tampered))
    with pytest.raises(RoutingCaptureError) as exc_info:
        RoutingCaptureSession(ForbiddenModel(), inspection, plan, lambda *args: (), max_events=1)
    assert exc_info.value.stage == "preflight"


def test_missing_component_mapping_is_rejected_before_traversal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ForbiddenModel:
        @property
        def named_modules(self):
            raise AssertionError("missing component mapping must not traverse the model")

    inspection = _inspection("mixtral", "legacy")
    plan = build_routing_probe_plan(inspection)
    missing_target = plan.targets[0].model_copy(update={"component_key": "component:" + "f" * 64})
    tampered_plan = plan.model_copy(update={"targets": (missing_target, *plan.targets[1:])})
    monkeypatch.setattr(routing_module, "build_routing_probe_plan", lambda value: tampered_plan)
    with pytest.raises(RoutingCaptureError) as exc_info:
        RoutingCaptureSession(
            ForbiddenModel(), inspection, tampered_plan, lambda *args: (), max_events=1
        )
    assert exc_info.value.stage == "preflight"


@pytest.mark.parametrize("decoder", [None, object(), 1])
def test_decoder_must_be_callable_before_model_traversal(decoder: object) -> None:
    class ForbiddenModel:
        @property
        def named_modules(self):
            raise AssertionError("decoder validation must precede model traversal")

    inspection = _inspection("mixtral", "legacy")
    plan = build_routing_probe_plan(inspection)
    with pytest.raises(RoutingCaptureError) as exc_info:
        RoutingCaptureSession(ForbiddenModel(), inspection, plan, decoder, max_events=1)
    assert exc_info.value.stage == "preflight"


@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize(
    "boundary", ["inspection_dump", "inspection_validate", "plan_dump", "compiler"]
)
def test_preflight_injected_failures_are_safe_or_exact(
    boundary: str,
    error_type: type[BaseException],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ForbiddenModel:
        @property
        def named_modules(self):
            raise AssertionError("injected preflight failure must precede traversal")

        def forward(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("preflight must not forward")

        def generate(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("preflight must not generate")

        @property
        def tokenizer(self):
            raise AssertionError("preflight must not touch tokenizer")

    inspection = _inspection("mixtral", "legacy")
    plan = build_routing_probe_plan(inspection)
    error = error_type("TOP_SECRET_PREFLIGHT_FAILURE")

    if boundary == "inspection_dump":

        def failing_dump(self, *args: object, **kwargs: object) -> object:
            raise error

        monkeypatch.setattr(AdapterInspection, "model_dump", failing_dump)
    elif boundary == "inspection_validate":

        def failing_validate(cls, *args: object, **kwargs: object) -> object:
            del cls, args, kwargs
            raise error

        monkeypatch.setattr(AdapterInspection, "model_validate", classmethod(failing_validate))
    elif boundary == "plan_dump":

        def failing_plan_dump(self, *args: object, **kwargs: object) -> object:
            raise error

        monkeypatch.setattr(ProbePlan, "model_dump", failing_plan_dump)
    else:

        def failing_compiler(value: AdapterInspection) -> ProbePlan:
            del value
            raise error

        monkeypatch.setattr(routing_module, "build_routing_probe_plan", failing_compiler)

    if error_type in (KeyboardInterrupt, SystemExit):
        with pytest.raises(error_type) as exc_info:
            RoutingCaptureSession(
                ForbiddenModel(), inspection, plan, lambda *args: (), max_events=1
            )
        assert exc_info.value is error
    else:
        with pytest.raises(RoutingCaptureError) as exc_info:
            RoutingCaptureSession(
                ForbiddenModel(), inspection, plan, lambda *args: (), max_events=1
            )
        assert exc_info.value.stage == "preflight"
        assert exc_info.value.__cause__ is error
        assert "TOP_SECRET_PREFLIGHT_FAILURE" not in str(exc_info.value)


def test_preflight_injected_routing_capture_error_is_rewrapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspection = _inspection("mixtral", "legacy")
    plan = build_routing_probe_plan(inspection)
    injected = RoutingCaptureError("preflight")

    def failing_compiler(value: AdapterInspection) -> ProbePlan:
        del value
        raise injected

    monkeypatch.setattr(routing_module, "build_routing_probe_plan", failing_compiler)
    with pytest.raises(RoutingCaptureError) as exc_info:
        RoutingCaptureSession(object(), inspection, plan, lambda *args: (), max_events=1)
    assert exc_info.value is not injected
    assert exc_info.value.stage == "preflight"
    assert exc_info.value.__cause__ is injected


@pytest.mark.parametrize("value", [True, 0, -1, "1"])
def test_preflight_rejects_non_strict_max_events(value: object) -> None:
    inspection = _inspection("mixtral", "legacy")
    plan = build_routing_probe_plan(inspection)
    with pytest.raises(RoutingCaptureError) as exc_info:
        RoutingCaptureSession(object(), inspection, plan, lambda *args: (), max_events=value)
    assert exc_info.value.stage == "preflight"


def test_cleanup_failure_is_retryable_and_publishes_after_retry() -> None:
    inspection = _inspection("mixtral", "legacy")
    plan = build_routing_probe_plan(inspection)
    model = _HookModel("mixtral", "legacy")
    first_path = next(iter(model.nodes))
    model.nodes[first_path].removal_failures = 1

    def decoder(context, module, inputs, output):
        return (
            RoutingEvent(
                token_key="token:" + "6" * 64,
                layer_key=context.layer_key,
                rank=0,
                expert_key=context.expert_keys[0],
                probability=0.2,
                selected=True,
            ),
        )

    session = RoutingCaptureSession(model, inspection, plan, decoder, max_events=2)
    with pytest.raises(RoutingCaptureError) as exc_info:
        with session:
            next(iter(model.nodes.values())).fire((), object())
    assert exc_info.value.stage == "lifecycle"
    expected_order = list(reversed(tuple(model.nodes)))
    assert model.removal_log == [path for path in expected_order if path != first_path]
    with pytest.raises(RoutingCaptureError):
        _ = session.events
    session.close()
    assert model.removal_log == expected_order
    assert len(session.events) == 1
    session.close()


@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt, SystemExit])
def test_registration_failure_preserves_exact_primary_and_manager_rollback(
    error_type: type[BaseException],
) -> None:
    model = _HookModel("mixtral", "legacy")
    failing_path = tuple(model.nodes)[1]
    error = error_type("TOP_SECRET_REGISTRATION_FAILURE")
    model.nodes[failing_path].registration_error = error
    session, _, _ = _session(model=model)

    with pytest.raises(error_type) as exc_info:
        session.__enter__()
    assert exc_info.value is error
    assert all(not node.callbacks for node in model.nodes.values())
    session.close()
    with pytest.raises(RoutingCaptureError) as lifecycle_error:
        _ = session.events
    assert lifecycle_error.value.stage == "lifecycle"


def test_partial_registration_failure_retains_only_failed_removal_for_close() -> None:
    model = _HookModel("mixtral", "legacy")
    first_path, failing_path = tuple(model.nodes)[:2]
    model.nodes[first_path].removal_failures = 1
    registration_error = ValueError("TOP_SECRET_PARTIAL_REGISTRATION")
    model.nodes[failing_path].registration_error = registration_error
    session, _, _ = _session(model=model)

    with pytest.raises(ValueError) as exc_info:
        session.__enter__()
    assert exc_info.value is registration_error
    assert any("hook cleanup failures" in note for note in exc_info.value.__notes__)
    assert model.nodes[first_path].callbacks
    assert model.removal_log == []
    session.close()
    assert not model.nodes[first_path].callbacks
    assert model.removal_log == [first_path]
    session.close()


def test_body_error_is_primary_and_prevents_publication() -> None:
    session, model, _ = _session()
    body_error = ValueError("TOP_SECRET_BODY")
    with pytest.raises(ValueError) as exc_info:
        with session:
            next(iter(model.nodes.values())).fire((), object())
            raise body_error
    assert exc_info.value is body_error
    with pytest.raises(RoutingCaptureError) as lifecycle_error:
        _ = session.events
    assert lifecycle_error.value.stage == "lifecycle"


@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt, SystemExit])
def test_body_control_flow_and_ordinary_failures_remain_primary_with_cleanup_retry(
    error_type: type[BaseException],
) -> None:
    session, model, _ = _session()
    first_path = next(iter(model.nodes))
    model.nodes[first_path].removal_failures = 1
    body_error = error_type("TOP_SECRET_BODY_FAILURE")
    with pytest.raises(error_type) as exc_info:
        with session:
            next(iter(model.nodes.values())).fire((), object())
            raise body_error
    assert exc_info.value is body_error
    session.close()
    with pytest.raises(RoutingCaptureError) as lifecycle_error:
        _ = session.events
    assert lifecycle_error.value.stage == "lifecycle"


def test_decoder_failure_and_cleanup_failure_keep_decoder_boundary_primary() -> None:
    decoder_error = ValueError("TOP_SECRET_DECODER_BODY")

    def decoder(context, module, inputs, output):
        del context, module, inputs, output
        raise decoder_error

    session, model, _ = _session(decoder=decoder)
    first_path = next(iter(model.nodes))
    model.nodes[first_path].removal_failures = 1
    with pytest.raises(RoutingCaptureError) as exc_info:
        with session:
            next(iter(model.nodes.values())).fire((), object())
    assert exc_info.value.stage == "decode"
    assert exc_info.value.__cause__ is decoder_error
    session.close()
    with pytest.raises(RoutingCaptureError) as lifecycle_error:
        _ = session.events
    assert lifecycle_error.value.stage == "lifecycle"


def test_session_single_use_close_before_enter_and_close_during_context() -> None:
    session, model, _ = _session()
    with pytest.raises(RoutingCaptureError) as exc_info:
        session.close()
    assert exc_info.value.stage == "lifecycle"

    with session:
        with pytest.raises(RoutingCaptureError) as reentry:
            session.__enter__()
        assert reentry.value.stage == "lifecycle"
        session.close()
        with pytest.raises(RoutingCaptureError) as unavailable:
            _ = session.events
        assert unavailable.value.stage == "lifecycle"
    assert all(not node.callbacks for node in model.nodes.values())
    with pytest.raises(RoutingCaptureError) as reuse:
        session.__enter__()
    assert reuse.value.stage == "lifecycle"


@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt, SystemExit])
def test_hook_manager_constructor_failures_are_exact_and_leave_deterministic_failed_state(
    error_type: type[BaseException], monkeypatch: pytest.MonkeyPatch
) -> None:
    error = error_type("TOP_SECRET_MANAGER_CONSTRUCTOR")

    class FailingManager:
        def __init__(self, model: object, plan: ProbePlan, callbacks: object) -> None:
            del model, plan, callbacks
            raise error

    monkeypatch.setattr(routing_module, "HookManager", FailingManager)
    session, model, _ = _session()
    with pytest.raises(error_type) as exc_info:
        session.__enter__()
    assert exc_info.value is error
    assert session._manager is None
    assert session._state == "failed"
    assert session._active_body is False
    assert all(not node.callbacks for node in model.nodes.values())
    with pytest.raises(RoutingCaptureError) as close_error:
        session.close()
    assert close_error.value.stage == "lifecycle"
    with pytest.raises(RoutingCaptureError) as reuse_error:
        session.__enter__()
    assert reuse_error.value.stage == "lifecycle"


def test_hook_manager_receives_session_plan_identity_and_deterministic_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_manager = routing_module.HookManager
    calls: list[tuple[object, object, tuple[HookBinding, ...]]] = []

    class SpyManager:
        def __init__(self, model: object, plan: ProbePlan, callbacks: object) -> None:
            assert isinstance(callbacks, dict)
            calls.append((model, plan, tuple(callbacks)))
            self._delegate = original_manager(model, plan, callbacks)

        @property
        def resolved_plan(self):
            return self._delegate.resolved_plan

        def __enter__(self):
            self._delegate.__enter__()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return self._delegate.__exit__(exc_type, exc_value, traceback)

        def close(self) -> None:
            self._delegate.close()

    monkeypatch.setattr(routing_module, "HookManager", SpyManager)
    session, model, _ = _session()
    with session:
        pass
    assert len(calls) == 1
    received_model, received_plan, bindings = calls[0]
    assert received_model is model
    assert received_plan is session.plan
    assert bindings == tuple(HookBinding(path, HookPoint.FORWARD) for path in session._contexts)


def test_callbacks_are_inert_after_cleanup_failure_retry_and_publication() -> None:
    decoder_calls = 0

    def decoder(context, module, inputs, output):
        nonlocal decoder_calls
        decoder_calls += 1
        return ()

    session, model, _ = _session(decoder=decoder, max_events=2)
    first_path = next(iter(model.nodes))
    model.nodes[first_path].removal_failures = 1
    with pytest.raises(RoutingCaptureError):
        with session:
            saved_callback = next(iter(model.nodes.values())).callbacks[0]
    snapshot = (
        decoder_calls,
        tuple(session._events),
        session._truncated,
        session._dropped_invocations,
    )
    saved_callback(model.nodes[first_path], (), object())
    assert (
        decoder_calls,
        tuple(session._events),
        session._truncated,
        session._dropped_invocations,
    ) == snapshot
    session.close()
    assert session.events == ()
    saved_callback(model.nodes[first_path], (), object())
    assert decoder_calls == snapshot[0]
    assert session.events == ()


def test_callback_saved_before_close_during_body_is_inert() -> None:
    calls = 0

    def decoder(context, module, inputs, output):
        nonlocal calls
        calls += 1
        return ()

    session, model, _ = _session(decoder=decoder)
    with session:
        saved_callback = next(iter(model.nodes.values())).callbacks[0]
        session.close()
        saved_callback(model, (), object())
        assert calls == 0
        assert session._events == []
        assert session._dropped_invocations == 0
    with pytest.raises(RoutingCaptureError):
        _ = session.events
    saved_callback(model, (), object())
    assert calls == 0


@pytest.mark.parametrize(
    "update",
    [{"selected": "not-bool"}, {"rank": -1}, {"probability": 2.0}],
)
def test_retained_events_are_fresh_and_invalid_model_copy_values_reject(
    update: dict[str, object],
) -> None:
    supplied: list[RoutingEvent] = []

    def decoder(context, module, inputs, output):
        event = RoutingEvent(
            token_key="token:" + "c" * 64,
            layer_key=context.layer_key,
            rank=0,
            expert_key=context.expert_keys[0],
            probability=0.5,
            selected=True,
        )
        supplied.append(event)
        return (event.model_copy(update=update),)

    session, model, _ = _session(decoder=decoder)
    with pytest.raises(RoutingCaptureError) as exc_info:
        with session:
            next(iter(model.nodes.values())).fire((), object())
    assert exc_info.value.stage == "events"
    assert session._events == []

    session, model, _ = _session(max_events=1)
    valid_supplied: list[RoutingEvent] = []

    def valid_decoder(context, module, inputs, output):
        del module, inputs, output
        event = RoutingEvent(
            token_key="token:" + "d" * 64,
            layer_key=context.layer_key,
            rank=0,
            expert_key=context.expert_keys[0],
            probability=0.5,
            selected=True,
        )
        valid_supplied.append(event)
        return (event,)

    session = RoutingCaptureSession(
        model,
        _inspection("mixtral", "legacy"),
        build_routing_probe_plan(_inspection("mixtral", "legacy")),
        valid_decoder,
        max_events=1,
    )
    with session:
        node = next(iter(model.nodes.values()))
        node.fire((), object())
    assert session.events[0] is not valid_supplied[0]


def _assert_context_matrix_rejects_ambiguous_layers_experts_and_facts_before_traversal(
    case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ForbiddenModel:
        @property
        def named_modules(self):
            raise AssertionError("invalid routing context must not traverse the model")

    original = _inspection("mixtral", "legacy")
    inspection = _tamper_report(original, case)
    plan = build_routing_probe_plan(original)

    def return_tampered(cls, payload: object) -> AdapterInspection:
        del cls, payload
        return inspection

    monkeypatch.setattr(AdapterInspection, "model_validate", classmethod(return_tampered))
    with pytest.raises(RoutingCaptureError) as exc_info:
        RoutingCaptureSession(ForbiddenModel(), inspection, plan, lambda *args: (), max_events=1)
    assert exc_info.value.stage == "preflight"


@pytest.mark.parametrize(
    "case",
    [
        "zero_layer",
        "multiple_layer",
        "missing_expert",
        "extra_expert",
        "bad_count_none",
        "bad_count_bool",
        "bad_count_string",
        "bad_count_zero",
        "bad_count_negative",
    ],
)
def test_context_matrix_cases(case: str, monkeypatch: pytest.MonkeyPatch) -> None:
    _assert_context_matrix_rejects_ambiguous_layers_experts_and_facts_before_traversal(
        case, monkeypatch
    )


def test_no_network_or_optional_model_imports_and_no_filesystem_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(request, "urlopen", forbidden)
    cache = tmp_path / "cache"
    cache.mkdir()
    for variable in (
        "HF_HOME",
        "TRANSFORMERS_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "TORCH_HOME",
    ):
        monkeypatch.setenv(variable, str(cache))
    before = tuple(sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")))
    optional_before = {
        name
        for name in ("torch", "transformers", "safetensors", "accelerate")
        if name in sys.modules
    }
    session, model, _ = _session(max_events=1)
    with session:
        next(iter(model.nodes.values())).fire((), object())
    assert tuple(sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))) == before
    assert optional_before == {
        name
        for name in ("torch", "transformers", "safetensors", "accelerate")
        if name in sys.modules
    }


def test_routing_module_has_no_optional_runtime_or_io_execution_surface() -> None:
    tree = ast.parse(inspect.getsource(routing_module))
    optional = {"torch", "transformers", "safetensors", "accelerate"}
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.lstrip(".").split(".", 1)[0])
    assert imported.isdisjoint(optional)
    call_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert call_names.isdisjoint(
        {
            "open",
            "urlopen",
            "create_connection",
            "resolve_probe_plan",
            "register_forward_hook",
            "forward",
            "generate",
            "named_modules",
            "named_parameters",
            "read_text",
            "write_text",
        }
    )
    forbidden_attributes = {
        "forward",
        "generate",
        "tokenizer",
        "named_modules",
        "named_parameters",
        "register_forward_hook",
        "register_forward_pre_hook",
        "register_full_backward_hook",
        "detach",
        "cpu",
        "numpy",
        "tolist",
        "storage",
        "open",
        "read_text",
        "write_text",
        "unlink",
        "replace",
    }
    assert {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }.isdisjoint(forbidden_attributes)


def test_target_is_frozen_and_strict() -> None:
    inspection = _inspection("mixtral", "legacy")
    plan = build_routing_probe_plan(inspection)
    target = plan.targets[0]
    context = RoutingCaptureTarget(
        target,
        next(
            component.component_key
            for component in inspection.report.components
            if component.kind.value == "moe_layer" and component.layer_index == 0
        ),
        tuple(
            component.component_key
            for component in inspection.report.components
            if component.kind.value == "expert" and component.layer_index == 0
        ),
        2,
    )
    with pytest.raises((AttributeError, TypeError)):
        context.routed_top_k = 1  # type: ignore[misc]
    assert context.router is target
