from __future__ import annotations

import ast
import inspect
import json
import os
import socket
import sys
from pathlib import Path
from typing import get_type_hints
from urllib import request

import pytest

import moeatlas.adapters as adapters
import moeatlas.adapters.planning as planning
from moeatlas.adapters import (
    AdapterDescriptor,
    AdapterInspection,
    AdapterProbePlanError,
    MixtralStaticAdapter,
    Qwen3MoeStaticAdapter,
    build_routing_probe_plan,
    inspect_static_adapter,
)
from moeatlas.core import (
    CapabilityLabel,
    CaptureSource,
    ComponentKind,
    DType,
    ModelManifest,
    TokenizerIdentity,
    make_component_key,
    make_config_hash,
    make_model_key,
)
from moeatlas.probe import (
    CaptureMode,
    HookPoint,
    ProbeLevel,
    ProbePlan,
    ReductionPolicy,
)

from .fixtures import MixtralForCausalLM, Qwen3MoeForCausalLM


def _manifest(architecture: str) -> ModelManifest:
    revision = "r1"
    model_id = f"acme/{architecture}"
    return ModelManifest(
        model_key=make_model_key(model_id, revision),
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
    else:
        model = Qwen3MoeForCausalLM(layout=layout)
        adapter = Qwen3MoeStaticAdapter()
    manifest = _manifest(architecture)
    return inspect_static_adapter(adapter, model, model.config, manifest)


def _routers(inspection: AdapterInspection) -> list[object]:
    return [
        component
        for component in inspection.report.components
        if component.kind is ComponentKind.ROUTER
    ]


def test_public_api_signature_exports_and_error_contract() -> None:
    assert planning.__all__ == ["AdapterProbePlanError", "build_routing_probe_plan"]
    assert adapters.AdapterProbePlanError is planning.AdapterProbePlanError
    assert adapters.build_routing_probe_plan is planning.build_routing_probe_plan

    signature = inspect.signature(planning.build_routing_probe_plan)
    assert tuple(signature.parameters) == ("inspection",)
    assert get_type_hints(planning.build_routing_probe_plan) == {
        "inspection": AdapterInspection,
        "return": ProbePlan,
    }
    assert issubclass(AdapterProbePlanError, ValueError)
    for stage in ("inspection", "targets", "plan"):
        error = AdapterProbePlanError(stage)
        assert error.stage == stage
        assert str(error) == f"adapter routing probe planning failed at {stage}"
    with pytest.raises(ValueError, match="adapter probe-plan error stage"):
        AdapterProbePlanError("invalid")


@pytest.mark.parametrize(
    ("architecture", "layout"),
    [
        ("mixtral", "legacy"),
        ("mixtral", "packed"),
        ("qwen3_moe", "legacy_indexed"),
        ("qwen3_moe", "packed"),
    ],
)
def test_builds_every_router_with_exact_inert_routing_fields(
    architecture: str, layout: str
) -> None:
    inspection = _inspection(architecture, layout)
    plan = build_routing_probe_plan(inspection)
    routers = _routers(inspection)
    expected_paths = sorted(component.module_path for component in routers)

    assert type(plan) is ProbePlan
    assert plan.level is ProbeLevel.ROUTING
    assert plan.hook_points == (HookPoint.FORWARD,)
    assert plan.include == ()
    assert plan.exclude == ()
    assert plan.intervention_opt_in is False
    assert plan.capture.mode is CaptureMode.REDUCED
    assert plan.capture.reduction is ReductionPolicy.TOP_K
    assert plan.capture.include_inputs is False
    assert plan.capture.include_outputs is True
    assert plan.capture.include_gradients is False
    assert plan.capture.raw_opt_in is False
    assert plan.capture.max_items is None
    assert plan.capture.max_bytes is None
    assert plan.capture.sample_rate == 1.0
    assert plan.capture.sample_seed is None
    assert [target.module_path for target in plan.targets] == expected_paths
    assert all(target.component_kind is ComponentKind.ROUTER for target in plan.targets)
    expected_keys = {component.module_path: component.component_key for component in routers}
    assert {target.module_path: target.component_key for target in plan.targets} == expected_keys
    assert plan.plan_id.startswith("plan:")


def test_qwen_plan_selects_sparse_routers_only() -> None:
    inspection = _inspection("qwen3_moe", "legacy_indexed")
    plan = build_routing_probe_plan(inspection)

    assert len(plan.targets) == 2
    assert all(".mlp.gate" in target.module_path for target in plan.targets)
    assert all(not target.module_path.endswith("gate_proj") for target in plan.targets)


def test_plan_json_roundtrip_is_deterministic_and_returns_fresh_objects() -> None:
    inspection = _inspection("mixtral", "packed")
    before = inspection.to_json()
    first = build_routing_probe_plan(inspection)
    second = build_routing_probe_plan(inspection)

    assert first is not second
    assert first.plan_id == second.plan_id
    assert first.to_json() == second.to_json()
    assert ProbePlan.from_json(first.to_json()) == first
    assert json.loads(first.to_json())["manifest_type"] == "probe_plan"
    assert inspection.to_json() == before


def test_family_neutral_future_descriptor_is_accepted() -> None:
    base = _inspection("mixtral", "legacy")
    descriptor = AdapterDescriptor(
        name="future-routing-static",
        version="9.0",
        architecture_families=("future_family",),
        compatibility_notes=("future structure",),
    )
    components = [
        component.model_copy(
            update={
                "capture": component.capture.model_copy(
                    update={"adapter": descriptor.name, "adapter_version": descriptor.version}
                )
            }
        )
        for component in base.report.components
    ]
    report = base.report.model_copy(update={"components": components})
    future = AdapterInspection.model_construct(
        schema_version="1.0",
        descriptor=descriptor,
        detection=base.detection,
        report=report,
    )

    plan = build_routing_probe_plan(future)

    assert len(plan.targets) == len(_routers(base))
    assert plan.targets == build_routing_probe_plan(base).targets


@pytest.mark.parametrize("value", [object(), None, "inspection"])
def test_input_must_be_exact_adapter_inspection(value: object) -> None:
    with pytest.raises(AdapterProbePlanError) as exc_info:
        build_routing_probe_plan(value)
    assert exc_info.value.stage == "inspection"
    assert str(exc_info.value) == "adapter routing probe planning failed at inspection"


class _InspectionSubclass(AdapterInspection):
    pass


def test_subclass_and_duck_inspections_are_rejected_at_inspection_stage() -> None:
    base = _inspection("mixtral", "legacy")
    subclass = _InspectionSubclass.model_validate(base.model_dump(mode="json"))

    class DuckInspection:
        def model_dump(self, **kwargs: object) -> dict[str, object]:
            return base.model_dump(mode="json")

    for value in (subclass, DuckInspection()):
        with pytest.raises(AdapterProbePlanError) as exc_info:
            build_routing_probe_plan(value)
        assert exc_info.value.stage == "inspection"


def _tampered_inspection(base: AdapterInspection, update: dict[str, object]) -> AdapterInspection:
    return base.model_copy(update=update)


@pytest.mark.parametrize(
    "tamper",
    [
        "zero_detection",
        "elevated",
        "wrong_source",
        "verified",
        "wrong_adapter",
        "wrong_adapter_version",
    ],
)
def test_tampered_inspection_is_revalidated_before_target_selection(tamper: str) -> None:
    base = _inspection("mixtral", "legacy")
    if tamper == "zero_detection":
        value = _tampered_inspection(
            base,
            {"detection": base.detection.model_copy(update={"score": 0.0, "warnings": ("none",)})},
        )
    elif tamper == "elevated":
        component = base.report.components[0].model_copy(
            update={"capabilities": [CapabilityLabel.ROUTING]}
        )
        value = _tampered_inspection(
            base,
            {
                "report": base.report.model_copy(
                    update={"components": [component, *base.report.components[1:]]}
                )
            },
        )
    elif tamper == "wrong_source":
        component = base.report.components[0]
        capture = component.capture.model_copy(update={"source": CaptureSource.MODULE_HOOK})
        value = _tampered_inspection(
            base,
            {
                "report": base.report.model_copy(
                    update={
                        "components": [
                            component.model_copy(update={"capture": capture}),
                            *base.report.components[1:],
                        ]
                    }
                )
            },
        )
    elif tamper == "verified":
        component = base.report.components[0]
        capture = component.capture.model_copy(update={"verified": True})
        value = _tampered_inspection(
            base,
            {
                "report": base.report.model_copy(
                    update={
                        "components": [
                            component.model_copy(update={"capture": capture}),
                            *base.report.components[1:],
                        ]
                    }
                )
            },
        )
    else:
        component = base.report.components[0]
        field = "adapter" if tamper == "wrong_adapter" else "adapter_version"
        capture = component.capture.model_copy(update={field: "tampered-value"})
        value = _tampered_inspection(
            base,
            {
                "report": base.report.model_copy(
                    update={
                        "components": [
                            component.model_copy(update={"capture": capture}),
                            *base.report.components[1:],
                        ]
                    }
                )
            },
        )

    with pytest.raises(AdapterProbePlanError) as exc_info:
        build_routing_probe_plan(value)
    assert exc_info.value.stage == "inspection"


def test_tampered_descriptor_report_binding_missing_parity_and_identity_are_rejected() -> None:
    base = _inspection("mixtral", "legacy")
    other_manifest = _manifest("qwen3_moe")
    cases = [
        base.model_copy(update={"descriptor": object()}),
        base.model_copy(
            update={"report": base.report.model_copy(update={"model_manifest": other_manifest})}
        ),
        base.model_copy(
            update={
                "report": base.report.model_copy(update={"components": base.report.components[:-1]})
            }
        ),
        base.model_copy(
            update={
                "report": base.report.model_copy(
                    update={
                        "components": [
                            base.report.components[0].model_copy(
                                update={"module_path": "bad path"}
                            ),
                            *base.report.components[1:],
                        ]
                    }
                )
            }
        ),
    ]
    for value in cases:
        with pytest.raises(AdapterProbePlanError) as exc_info:
            build_routing_probe_plan(value)
        assert exc_info.value.stage == "inspection"


def test_discovery_report_input_is_rejected_at_inspection_stage() -> None:
    inspection = _inspection("mixtral", "legacy")

    with pytest.raises(AdapterProbePlanError) as exc_info:
        build_routing_probe_plan(inspection.report)

    assert exc_info.value.stage == "inspection"


def test_valid_inspection_with_only_non_router_components_fails_at_targets() -> None:
    base = _inspection("mixtral", "legacy")
    non_router_components = [
        component
        for component in base.report.components
        if component.kind is not ComponentKind.ROUTER
    ]
    non_router_keys = {component.component_key for component in non_router_components}
    non_router_candidates = [
        candidate
        for candidate in base.report.candidates
        if candidate.component_key in non_router_keys
    ]
    report = base.report.model_copy(
        update={"components": non_router_components, "candidates": non_router_candidates}
    )
    value = AdapterInspection.model_construct(
        schema_version="1.0",
        descriptor=base.descriptor,
        detection=base.detection,
        report=report,
    )

    with pytest.raises(AdapterProbePlanError) as exc_info:
        build_routing_probe_plan(value)

    assert exc_info.value.stage == "targets"


def test_missing_router_components_is_a_target_stage_failure() -> None:
    base = _inspection("mixtral", "legacy")
    report = base.report.model_copy(
        update={"candidates": [], "components": [], "warnings": ["no routers"]}
    )
    value = AdapterInspection.model_construct(
        schema_version="1.0", descriptor=base.descriptor, detection=base.detection, report=report
    )

    with pytest.raises(AdapterProbePlanError) as exc_info:
        build_routing_probe_plan(value)
    assert exc_info.value.stage == "targets"


def test_duplicate_router_module_paths_are_a_target_stage_failure() -> None:
    base = _inspection("mixtral", "legacy")
    router = next(
        component for component in base.report.components if component.kind is ComponentKind.ROUTER
    )
    duplicate = router.model_copy(
        update={
            "layer_index": (router.layer_index or 0) + 10,
            "component_key": make_component_key(
                router.model_key,
                ComponentKind.ROUTER.value,
                router.module_path,
                layer_index=(router.layer_index or 0) + 10,
            ),
        }
    )
    candidate = next(
        item for item in base.report.candidates if item.component_key == router.component_key
    )
    duplicate_candidate = candidate.model_copy(
        update={
            "layer_index": (candidate.layer_index or 0) + 10,
            "component_key": duplicate.component_key,
        }
    )
    report = base.report.model_copy(
        update={
            "components": [*base.report.components, duplicate],
            "candidates": [*base.report.candidates, duplicate_candidate],
        }
    )
    value = AdapterInspection.model_construct(
        schema_version="1.0", descriptor=base.descriptor, detection=base.detection, report=report
    )

    with pytest.raises(AdapterProbePlanError) as exc_info:
        build_routing_probe_plan(value)
    assert exc_info.value.stage == "targets"


@pytest.mark.parametrize("boundary", ["dump", "revalidate", "targets", "capture", "plan"])
@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
def test_control_flow_exceptions_are_never_wrapped(
    boundary: str,
    error_type: type[BaseException],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _inspection("mixtral", "legacy")
    error = error_type(f"TOP_SECRET_{boundary}")
    if boundary == "dump":

        def raising_dump(self: object, **kwargs: object) -> object:
            raise error

        monkeypatch.setattr(AdapterInspection, "model_dump", raising_dump)
    elif boundary == "revalidate":

        def raising_validate(cls: object, payload: object) -> object:
            raise error

        monkeypatch.setattr(AdapterInspection, "model_validate", classmethod(raising_validate))
    elif boundary == "targets":

        def raising_target(*args: object, **kwargs: object) -> object:
            raise error

        import moeatlas.adapters.planning as planning

        monkeypatch.setattr(planning, "ProbeTarget", raising_target)
    elif boundary == "capture":

        def raising_capture(*args: object, **kwargs: object) -> object:
            raise error

        import moeatlas.adapters.planning as planning

        monkeypatch.setattr(planning, "CapturePolicy", raising_capture)
    else:

        def raising_plan(*args: object, **kwargs: object) -> object:
            raise error

        import moeatlas.adapters.planning as planning

        monkeypatch.setattr(planning, "ProbePlan", raising_plan)

    with pytest.raises(error_type) as exc_info:
        build_routing_probe_plan(base)
    assert exc_info.value is error


@pytest.mark.parametrize("boundary", ["dump", "revalidate", "targets", "capture", "plan"])
def test_ordinary_failures_are_safe_fixed_stage_errors(
    boundary: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = _inspection("mixtral", "legacy")
    error = ValueError("TOP_SECRET_PLANNING_VALUE")
    if boundary == "dump":

        def raising_dump(self: object, **kwargs: object) -> object:
            raise error

        monkeypatch.setattr(AdapterInspection, "model_dump", raising_dump)
        expected_stage = "inspection"
    elif boundary == "revalidate":

        def raising_validate(cls: object, payload: object) -> object:
            raise error

        monkeypatch.setattr(AdapterInspection, "model_validate", classmethod(raising_validate))
        expected_stage = "inspection"
    elif boundary == "targets":

        def raising_target(*args: object, **kwargs: object) -> object:
            raise error

        import moeatlas.adapters.planning as planning

        monkeypatch.setattr(planning, "ProbeTarget", raising_target)
        expected_stage = "targets"
    elif boundary == "capture":

        def raising_capture(*args: object, **kwargs: object) -> object:
            raise error

        import moeatlas.adapters.planning as planning

        monkeypatch.setattr(planning, "CapturePolicy", raising_capture)
        expected_stage = "plan"
    else:

        def raising_plan(*args: object, **kwargs: object) -> object:
            raise error

        import moeatlas.adapters.planning as planning

        monkeypatch.setattr(planning, "ProbePlan", raising_plan)
        expected_stage = "plan"

    with pytest.raises(AdapterProbePlanError) as exc_info:
        build_routing_probe_plan(base)
    assert exc_info.value.stage == expected_stage
    assert str(exc_info.value) == f"adapter routing probe planning failed at {expected_stage}"
    assert "TOP_SECRET_PLANNING_VALUE" not in str(exc_info.value)
    assert exc_info.value.__cause__ is error


def test_planning_source_has_no_runtime_or_side_effect_operations() -> None:
    source_path = Path(__file__).parents[1] / "src" / "moeatlas" / "adapters" / "planning.py"
    tree = ast.parse(source_path.read_text())
    forbidden_import_roots = {
        "accelerate",
        "importlib",
        "safetensors",
        "subprocess",
        "torch",
        "transformers",
    }
    forbidden_calls = {
        "__import__",
        "create_connection",
        "find_spec",
        "forward",
        "generate",
        "import_module",
        "makedirs",
        "mkdir",
        "named_modules",
        "named_parameters",
        "open",
        "register_forward_hook",
        "register_forward_pre_hook",
        "register_full_backward_hook",
        "subprocess",
        "unlink",
        "urlopen",
        "write",
        "write_bytes",
        "write_text",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots = {alias.name.split(".", 1)[0] for alias in node.names}
            assert imported_roots.isdisjoint(forbidden_import_roots)
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            assert root not in forbidden_import_roots
        elif isinstance(node, ast.Call):
            function_name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr
            assert function_name not in forbidden_calls


def test_compiler_does_not_call_adapters_or_touch_runtime_surfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspection = _inspection("qwen3_moe", "packed")
    model_methods_called: list[str] = []

    def forbidden(*args: object, **kwargs: object) -> object:
        model_methods_called.append("called")
        raise AssertionError("runtime/model method must not be called")

    monkeypatch.setattr(Qwen3MoeStaticAdapter, "detect", forbidden)
    monkeypatch.setattr(Qwen3MoeStaticAdapter, "discover", forbidden)
    plan = build_routing_probe_plan(inspection)

    assert plan.level is ProbeLevel.ROUTING
    assert model_methods_called == []


def test_compiler_has_no_network_cache_or_file_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inspection = _inspection("mixtral", "packed")

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access must not occur")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(request, "urlopen", forbidden)
    cache_dirs = [tmp_path / name for name in ("hf", "hub", "torch")]
    for cache_dir in cache_dirs:
        cache_dir.mkdir()
    for env_name, cache_dir in zip(
        ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TORCH_HOME"), cache_dirs, strict=True
    ):
        monkeypatch.setenv(env_name, str(cache_dir))
    before = tuple(sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*")))
    optional_before = {
        name
        for name in ("torch", "transformers", "safetensors", "accelerate")
        if name in sys.modules
    }

    build_routing_probe_plan(inspection)

    after = tuple(sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*")))
    optional_after = {
        name
        for name in ("torch", "transformers", "safetensors", "accelerate")
        if name in sys.modules
    }
    assert after == before
    assert optional_after == optional_before
    assert os.environ["HF_HOME"] == str(cache_dirs[0])
