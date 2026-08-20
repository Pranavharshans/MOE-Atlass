from __future__ import annotations

import sys

import pytest
from pydantic import ValidationError

from moeatlas.core import ComponentKind, make_component_key, parse_component_key
from moeatlas.probe import (
    CaptureMode,
    CapturePolicy,
    HookBinding,
    HookCleanupError,
    HookLifecycleError,
    HookManager,
    HookPoint,
    HookRegistrationError,
    ProbeLevel,
    ProbePlan,
    ProbeResolutionError,
    ProbeTarget,
    ReductionPolicy,
    ResolvedProbePlan,
    ResolvedTarget,
    resolve_probe_plan,
)

from .fixtures.synthetic_hooks import SyntheticHookModel, SyntheticHookModule


def routing_plan(*paths: str, **kwargs: object) -> ProbePlan:
    return ProbePlan(
        level=ProbeLevel.ROUTING,
        hook_points=[HookPoint.FORWARD],
        targets=[ProbeTarget(module_path=path) for path in paths],
        **kwargs,
    )


def test_probe_levels_and_plan_json_round_trip_are_stable() -> None:
    assert [level.value for level in ProbeLevel] == [0, 1, 2, 3, 4, 5]
    policy = CapturePolicy(
        mode=CaptureMode.REDUCED,
        reduction=ReductionPolicy.MEAN,
        include_inputs=True,
        sample_rate=0.25,
        sample_seed=17,
        max_items=32,
    )
    first = ProbePlan(
        level=ProbeLevel.ROUTING,
        hook_points=[HookPoint.FORWARD, HookPoint.FORWARD_PRE],
        targets=[
            ProbeTarget(module_path="layers.1.router"),
            ProbeTarget(module_path="layers.0.router"),
        ],
        include=["layers.1.router", "layers.0.router"],
        capture=policy,
    )
    second = ProbePlan(
        level=ProbeLevel.ROUTING,
        hook_points=[HookPoint.FORWARD_PRE, HookPoint.FORWARD],
        targets=list(reversed(first.targets)),
        include=list(reversed(first.include)),
        capture=policy,
    )

    assert first.plan_id == second.plan_id
    assert first.plan_hash == second.plan_hash
    assert first.hook_points == (HookPoint.FORWARD_PRE, HookPoint.FORWARD)
    assert first.targets[0].module_path == "layers.0.router"
    decoded = ProbePlan.from_json(first.to_json(indent=2))
    assert decoded == first
    assert decoded.to_dict() == first.to_dict()


def test_probe_targets_use_natural_numeric_path_order() -> None:
    plan = routing_plan(
        "model.layers.10.mlp.gate",
        "model.layers.2.mlp.gate",
        "model.layers.1.mlp.gate",
    )
    assert [target.module_path for target in plan.targets] == [
        "model.layers.1.mlp.gate",
        "model.layers.2.mlp.gate",
        "model.layers.10.mlp.gate",
    ]
    decoded = ProbePlan.from_json(plan.to_json())
    assert decoded == plan
    assert decoded.plan_id == plan.plan_id


def test_probe_natural_order_is_model_neutral_and_does_not_pad_paths() -> None:
    plan = routing_plan("block.10.router", "block.2.router")
    assert [target.module_path for target in plan.targets] == [
        "block.2.router",
        "block.10.router",
    ]
    assert all(".02." not in target.module_path for target in plan.targets)


def test_resolved_plan_preserves_source_numeric_order_and_filters() -> None:
    registration_log: list[tuple[str, str]] = []
    removal_log: list[tuple[str, str]] = []
    paths = tuple(f"layers.{index}.router" for index in range(11))
    modules = {
        path: SyntheticHookModule(
            path,
            registration_log=registration_log,
            removal_log=removal_log,
        )
        for path in reversed(paths)
    }

    class PermutedModel:
        def named_modules(self):
            yield "", self
            yield from modules.items()

    model = PermutedModel()
    plan = routing_plan(*reversed(paths))
    assert tuple(target.module_path for target in plan.targets) == paths
    resolved = resolve_probe_plan(plan, model)
    assert tuple(target.target.module_path for target in resolved.targets) == paths
    assert tuple(path for path, _, _ in resolved.bindings) == paths

    supplied_in_reverse = tuple(
        ResolvedTarget(target=target, module=modules[target.module_path])
        for target in reversed(plan.targets)
    )
    reordered = ResolvedProbePlan(plan=plan, targets=supplied_in_reverse)
    assert tuple(target.target.module_path for target in reordered.targets) == paths

    filtered_plan = routing_plan(
        *reversed(paths),
        include=("layers.10.router", "layers.2.router", "layers.0.router"),
        exclude=("layers.9.router",),
    )
    filtered = resolve_probe_plan(filtered_plan, model)
    assert tuple(target.target.module_path for target in filtered.targets) == (
        "layers.0.router",
        "layers.2.router",
        "layers.10.router",
    )
    assert tuple(path for path, _, _ in filtered.bindings) == (
        "layers.0.router",
        "layers.2.router",
        "layers.10.router",
    )


def test_probe_target_component_identity_is_canonical_and_paired() -> None:
    component_key = make_component_key(
        "model:org/model@main",
        ComponentKind.ROUTER.value,
        "layers.0.router",
    )
    target = ProbeTarget(
        module_path="layers.0.router",
        component_key=component_key,
        component_kind=ComponentKind.ROUTER,
    )
    assert parse_component_key(target.component_key or "") == component_key.removeprefix(
        "component:"
    )

    with pytest.raises(ValidationError, match="canonical component"):
        ProbeTarget(
            module_path="layers.0.router",
            component_key="component:a",
            component_kind=ComponentKind.ROUTER,
        )
    with pytest.raises(ValidationError, match="must be provided together"):
        ProbeTarget(module_path="layers.0.router", component_key=component_key)
    with pytest.raises(ValidationError, match="must be provided together"):
        ProbeTarget(module_path="layers.0.router", component_kind=ComponentKind.ROUTER)


def test_probe_plan_rejects_shared_module_paths_and_is_immutable() -> None:
    with pytest.raises(ValidationError, match="share a module_path"):
        ProbePlan(
            level=ProbeLevel.ROUTING,
            hook_points=[HookPoint.FORWARD],
            targets=[
                ProbeTarget(module_path="layers.0.router"),
                ProbeTarget(module_path="layers.0.router"),
            ],
        )

    plan = routing_plan("layers.0.router")
    original_id = plan.plan_id
    with pytest.raises(AttributeError):
        plan.hook_points.append(HookPoint.FORWARD_PRE)  # type: ignore[attr-defined]
    with pytest.raises(TypeError):
        plan.targets[0] = ProbeTarget(module_path="layers.1.router")  # type: ignore[index]
    assert plan.plan_id == original_id


def test_probe_plan_is_strict_and_enforces_capture_boundaries() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ProbePlan.model_validate(routing_plan("layers.0.router").to_dict() | {"unexpected": True})

    with pytest.raises(ValidationError, match="incompatible"):
        ProbePlan(
            level=ProbeLevel.ROUTING,
            hook_points=[HookPoint.FULL_BACKWARD],
            targets=[ProbeTarget(module_path="layers.0.router")],
        )

    with pytest.raises(ValidationError, match="RAW.*NONE"):
        CapturePolicy(mode=CaptureMode.RAW)
    with pytest.raises(ValidationError, match="requires raw_opt_in"):
        CapturePolicy(mode=CaptureMode.RAW, reduction=ReductionPolicy.NONE)
    with pytest.raises(ValidationError, match="RAW.*NONE"):
        CapturePolicy(reduction=ReductionPolicy.NONE)
    with pytest.raises(ValidationError, match="positive max_items"):
        CapturePolicy(raw_opt_in=True)
    with pytest.raises(ValidationError, match="sample_seed"):
        CapturePolicy(sample_rate=0.5)
    with pytest.raises(ValidationError, match="FULL_ACTIVATIONS requires"):
        ProbePlan(
            level=ProbeLevel.FULL_ACTIVATIONS,
            hook_points=[HookPoint.FORWARD],
            targets=[ProbeTarget(module_path="layers.0.router")],
        )
    with pytest.raises(ValidationError, match="CaptureMode.RAW"):
        ProbePlan(
            level=ProbeLevel.FULL_ACTIVATIONS,
            hook_points=[HookPoint.FORWARD],
            targets=[ProbeTarget(module_path="layers.0.router")],
            capture=CapturePolicy(raw_opt_in=True, max_items=8),
        )

    full_plan = ProbePlan(
        level=ProbeLevel.FULL_ACTIVATIONS,
        hook_points=[HookPoint.FORWARD],
        targets=[ProbeTarget(module_path="layers.0.router")],
        capture=CapturePolicy(
            raw_opt_in=True,
            mode=CaptureMode.RAW,
            reduction=ReductionPolicy.NONE,
            max_items=8,
        ),
    )
    assert full_plan.level is ProbeLevel.FULL_ACTIVATIONS

    with pytest.raises(ValidationError, match="GRADIENTS requires"):
        ProbePlan(
            level=ProbeLevel.GRADIENTS,
            hook_points=[HookPoint.FULL_BACKWARD],
            targets=[ProbeTarget(module_path="layers.0.router")],
            capture=CapturePolicy(
                raw_opt_in=True,
                mode=CaptureMode.RAW,
                reduction=ReductionPolicy.NONE,
                max_items=8,
            ),
        )

    with pytest.raises(ValidationError, match="INTERVENTION requires"):
        ProbePlan(
            level=ProbeLevel.INTERVENTION,
            hook_points=[HookPoint.FORWARD_PRE],
            targets=[ProbeTarget(module_path="layers.0.router")],
        )
    with pytest.raises(ValidationError, match="exactly the GRADIENTS"):
        ProbePlan(
            level=ProbeLevel.INTERVENTION,
            hook_points=[HookPoint.FORWARD_PRE],
            targets=[ProbeTarget(module_path="layers.0.router")],
            capture=CapturePolicy(
                include_gradients=True,
                raw_opt_in=True,
                max_items=8,
            ),
            intervention_opt_in=True,
        )


def test_resolution_is_deterministic_and_rejects_bad_targets_or_surfaces() -> None:
    model = SyntheticHookModel()
    plan = routing_plan("layers.1.router", "layers.0.router")
    resolved = resolve_probe_plan(plan, model)

    assert [item.target.module_path for item in resolved.targets] == [
        "layers.0.router",
        "layers.1.router",
    ]
    assert [path for path, _, _ in resolved.bindings] == [
        "layers.0.router",
        "layers.1.router",
    ]

    with pytest.raises(ProbeResolutionError, match="missing"):
        resolve_probe_plan(routing_plan("layers.9.router"), model)
    with pytest.raises(ProbeResolutionError, match="selected no targets"):
        resolve_probe_plan(routing_plan("layers.0.router", include=["layers.1.router"]), model)

    class DuplicateModel(SyntheticHookModel):
        def named_modules(self):
            yield "", self
            yield "layers.0.router", self.modules["layers.0.router"]
            yield "layers.0.router", self.modules["layers.0.router"]

    with pytest.raises(ProbeResolutionError, match="duplicate"):
        resolve_probe_plan(routing_plan("layers.0.router"), DuplicateModel())

    with pytest.raises(ProbeResolutionError, match="duplicate module paths"):
        ResolvedProbePlan(
            plan=routing_plan("layers.0.router", "layers.1.router"),
            targets=(
                ResolvedTarget(
                    target=ProbeTarget(module_path="layers.0.router"),
                    module=model.modules["layers.0.router"],
                ),
                ResolvedTarget(
                    target=ProbeTarget(module_path="layers.0.router"),
                    module=model.modules["layers.0.router"],
                ),
            ),
        )

    filtered_plan = routing_plan(
        "layers.0.router",
        "layers.1.router",
        include=["layers.0.router"],
    )
    filtered = resolve_probe_plan(filtered_plan, model)
    assert [target.target.module_path for target in filtered.targets] == ["layers.0.router"]
    with pytest.raises(ProbeResolutionError, match="exactly match"):
        ResolvedProbePlan(plan=filtered_plan, targets=tuple(resolved.targets))

    component_key = make_component_key(
        "model:org/model@main",
        ComponentKind.ROUTER.value,
        "layers.0.router",
    )
    tampered_target = ProbeTarget(
        module_path="layers.0.router",
        component_key=component_key,
        component_kind=ComponentKind.ROUTER,
    )
    with pytest.raises(ProbeResolutionError, match="exactly match"):
        ResolvedProbePlan(
            plan=plan,
            targets=(
                ResolvedTarget(
                    target=tampered_target,
                    module=model.modules["layers.0.router"],
                ),
                resolved.targets[1],
            ),
        )

    class ForwardOnlyModel:
        class ForwardOnlyModule:
            def register_forward_hook(self, callback: object) -> object:
                return callback

        def named_modules(self):
            yield "", self
            yield "layers.0.router", self.ForwardOnlyModule()

    with pytest.raises(ProbeResolutionError, match="does not support"):
        resolve_probe_plan(
            ProbePlan(
                level=ProbeLevel.GRADIENTS,
                hook_points=[HookPoint.FULL_BACKWARD],
                targets=[ProbeTarget(module_path="layers.0.router")],
                capture=CapturePolicy(
                    raw_opt_in=True,
                    mode=CaptureMode.RAW,
                    reduction=ReductionPolicy.NONE,
                    max_items=8,
                    include_gradients=True,
                ),
            ),
            ForwardOnlyModel(),
        )


def test_hook_manager_registers_in_order_and_cleans_in_reverse() -> None:
    model = SyntheticHookModel()
    plan = routing_plan("layers.1.router", "layers.0.router")
    seen: list[tuple[str, object]] = []
    callbacks = {
        HookBinding(path, HookPoint.FORWARD): (
            lambda *args, path=path: seen.append((path, args[1]))
        )
        for path in ("layers.0.router", "layers.1.router")
    }
    input_value = object()

    manager = HookManager(model, plan, callbacks)
    with manager:
        assert manager.active
        assert manager.installed_count == 2
        model.modules["layers.0.router"].fire("forward", "module", input_value, "output")
        model.modules["layers.1.router"].fire("forward", "module", input_value, "output")

    assert seen == [("layers.0.router", input_value), ("layers.1.router", input_value)]
    assert model.registration_log == [
        ("layers.0.router", "forward"),
        ("layers.1.router", "forward"),
    ]
    assert model.removal_log == [
        ("layers.1.router", "forward"),
        ("layers.0.router", "forward"),
    ]
    assert not manager.active
    manager.close()
    with pytest.raises(HookLifecycleError, match="single-use"):
        manager.__enter__()


def test_hook_manager_rejects_resolved_plan_from_another_model() -> None:
    model_a = SyntheticHookModel()
    model_b = SyntheticHookModel()
    plan = routing_plan("layers.0.router")
    resolved_from_b = resolve_probe_plan(plan, model_b)
    manager = HookManager(
        model_a,
        resolved_from_b,
        {HookBinding("layers.0.router", HookPoint.FORWARD): lambda *args: None},
    )

    with pytest.raises(HookRegistrationError, match="module identity"):
        manager.__enter__()
    assert model_a.registration_log == []
    assert model_b.registration_log == []


def test_hook_manager_callbacks_are_passive_across_hook_points() -> None:
    module = SyntheticHookModule(
        "layers.0.router",
        registration_log=[],
        removal_log=[],
    )
    model = SyntheticHookModel(modules={"layers.0.router": module})
    sentinel = object()
    seen: list[tuple[str, tuple[object, ...]]] = []

    def callback(*args: object) -> object:
        seen.append(("routing", args))
        return sentinel

    routing = ProbePlan(
        level=ProbeLevel.ROUTING,
        hook_points=[HookPoint.FORWARD_PRE, HookPoint.FORWARD],
        targets=[ProbeTarget(module_path="layers.0.router")],
    )
    routing_callbacks = {
        HookBinding("layers.0.router", HookPoint.FORWARD_PRE): callback,
        HookBinding("layers.0.router", HookPoint.FORWARD): callback,
    }
    with HookManager(model, routing, routing_callbacks):
        input_value = object()
        output_value = object()
        assert module.fire("forward_pre", "module", input_value) == [None]
        assert module.fire("forward", "module", input_value, output_value) == [None]

    assert seen == [
        ("routing", ("module", input_value)),
        ("routing", ("module", input_value, output_value)),
    ]

    gradient = ProbePlan(
        level=ProbeLevel.GRADIENTS,
        hook_points=[HookPoint.FORWARD, HookPoint.FULL_BACKWARD],
        targets=[ProbeTarget(module_path="layers.0.router")],
        capture=CapturePolicy(
            mode=CaptureMode.RAW,
            reduction=ReductionPolicy.NONE,
            raw_opt_in=True,
            max_items=8,
            include_gradients=True,
        ),
    )

    def gradient_callback(*args: object) -> object:
        seen.append(("gradient", args))
        return sentinel

    with HookManager(
        model,
        gradient,
        {
            HookBinding("layers.0.router", HookPoint.FORWARD): gradient_callback,
            HookBinding("layers.0.router", HookPoint.FULL_BACKWARD): gradient_callback,
        },
    ):
        assert module.fire("forward", "module", "input", "output") == [None]
        assert module.fire("full_backward", "module", "grad_input", "grad_output") == [None]

    assert seen[-2:] == [
        ("gradient", ("module", "input", "output")),
        ("gradient", ("module", "grad_input", "grad_output")),
    ]


def test_hook_manager_preserves_body_callback_and_baseexception_errors() -> None:
    model = SyntheticHookModel()

    def callback(*args: object) -> None:
        raise ValueError("callback failure")

    with pytest.raises(ValueError, match="callback failure"):
        with HookManager(
            model,
            routing_plan("layers.0.router"),
            {HookBinding("layers.0.router", HookPoint.FORWARD): callback},
        ):
            model.modules["layers.0.router"].fire("forward", "module", (), "output")
    assert model.removal_log == [("layers.0.router", "forward")]

    with pytest.raises(ValueError, match="body failure"):
        with HookManager(
            model,
            routing_plan("layers.0.router"),
            {HookBinding("layers.0.router", HookPoint.FORWARD): lambda *args: None},
        ):
            raise ValueError("body failure")

    class SyntheticBase(BaseException):
        pass

    with pytest.raises(SyntheticBase):
        with HookManager(
            model,
            routing_plan("layers.0.router"),
            {HookBinding("layers.0.router", HookPoint.FORWARD): lambda *args: None},
        ):
            raise SyntheticBase()


def test_partial_registration_failure_cleans_prior_handles_and_prevents_reuse() -> None:
    registration_log: list[tuple[str, str]] = []
    removal_log: list[tuple[str, str]] = []
    modules = {
        "layers.0.router": SyntheticHookModule(
            "layers.0.router",
            registration_log=registration_log,
            removal_log=removal_log,
        ),
        "layers.1.router": SyntheticHookModule(
            "layers.1.router",
            registration_log=registration_log,
            removal_log=removal_log,
            fail_registrations={"forward"},
        ),
    }
    model = SyntheticHookModel(modules=modules)
    plan = routing_plan("layers.0.router", "layers.1.router")
    callbacks = {HookBinding(path, HookPoint.FORWARD): lambda *args: None for path in modules}
    manager = HookManager(model, plan, callbacks)

    with pytest.raises(RuntimeError, match="registration failure"):
        with manager:
            pass
    assert removal_log == [("layers.0.router", "forward")]
    with pytest.raises(HookLifecycleError, match="single-use"):
        manager.__enter__()


def test_cleanup_failures_are_aggregated_and_do_not_hide_body_error() -> None:
    registration_log: list[tuple[str, str]] = []
    removal_log: list[tuple[str, str]] = []
    transient_module = SyntheticHookModule(
        "layers.0.router",
        registration_log=registration_log,
        removal_log=removal_log,
        transient_removals={"forward": 1},
    )
    model = SyntheticHookModel(modules={"layers.0.router": transient_module})
    plan = routing_plan("layers.0.router")
    callback = {HookBinding("layers.0.router", HookPoint.FORWARD): lambda *args: None}

    manager = HookManager(model, plan, callback)
    with pytest.raises(HookCleanupError) as cleanup_error:
        with manager:
            pass
    assert len(cleanup_error.value.failures) == 1
    assert manager.installed_count == 1
    manager.close()
    assert manager.installed_count == 0
    manager.close()

    permanent_module = SyntheticHookModule(
        "layers.0.router",
        registration_log=registration_log,
        removal_log=removal_log,
        fail_removals={"forward"},
    )
    model = SyntheticHookModel(modules={"layers.0.router": permanent_module})
    manager = HookManager(model, plan, callback)
    with pytest.raises(ValueError, match="original body") as body_error:
        with manager:
            raise ValueError("original body")
    assert manager.installed_count == 1
    assert "cleanup failures" in " ".join(getattr(body_error.value, "__notes__", []))
    with pytest.raises(HookCleanupError):
        manager.close()
    assert manager.installed_count == 1
    permanent_module.fail_removals.clear()
    manager.close()
    assert manager.installed_count == 0
    manager.close()


def test_registration_handle_without_remove_is_retained_for_cleanup() -> None:
    class LateHandle:
        remove = None

    late_handle = LateHandle()

    class InvalidHandleModule(SyntheticHookModule):
        def register_forward_hook(self, callback: object) -> LateHandle:
            self.registration_log.append((self.name, "forward"))
            self.callbacks["forward"].append(callback)
            return late_handle

    registration_log: list[tuple[str, str]] = []
    removal_log: list[tuple[str, str]] = []
    module = InvalidHandleModule(
        "layers.0.router",
        registration_log=registration_log,
        removal_log=removal_log,
    )
    model = SyntheticHookModel(modules={"layers.0.router": module})
    manager = HookManager(
        model,
        routing_plan("layers.0.router"),
        {HookBinding("layers.0.router", HookPoint.FORWARD): lambda *args: None},
    )

    with pytest.raises(HookRegistrationError, match="without remove") as registration_error:
        manager.__enter__()
    assert manager.installed_count == 1
    assert "cleanup failures" in " ".join(getattr(registration_error.value, "__notes__", []))
    with pytest.raises(HookCleanupError):
        manager.close()
    assert manager.installed_count == 1

    late_handle.remove = lambda: None
    manager.close()
    assert manager.installed_count == 0
    manager.close()


def test_probe_import_does_not_load_model_runtime() -> None:
    assert not any(name in sys.modules for name in ("torch", "transformers", "safetensors"))
