from __future__ import annotations

import ast
import gc
import inspect
import weakref
from collections.abc import Mapping
from pathlib import Path

import pytest

from moeatlas.adapters import (
    Qwen3_5MoeStaticAdapter,
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

from .fixtures.qwen3_5_moe import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeForConditionalGeneration,
    Qwen3_5MoeModel,
)


def _manifest() -> ModelManifest:
    return ModelManifest(
        model_key=make_model_key("acme/qwen3.5", "r1"),
        architecture="qwen3_5_moe",
        revision="r1",
        config_hash=make_config_hash({"fixture": "qwen3.5"}),
        tokenizer=TokenizerIdentity(identifier="acme/tok", revision="r1"),
        dtype=DType.FLOAT32,
        device_map={"": "cpu"},
    )


def _mapping_model(*, conditional: bool = True) -> object:
    model = Qwen3_5MoeForConditionalGeneration() if conditional else Qwen3_5MoeForCausalLM()
    if conditional:
        nested_source = model.config.text_config
        nested = dict(vars(nested_source))
        nested.pop("architectures")
        outer = dict(vars(model.config))
        outer["text_config"] = nested
        model.config = outer
        model._modules["model.language_model"].config = nested
    else:
        model.config = dict(vars(model.config))
    return model


class _HostileMapping(dict[str, object]):
    def __init__(self, source: Mapping[str, object], mode: str) -> None:
        super().__init__(source)
        self.mode = mode

    def __contains__(self, key: object) -> bool:
        if self.mode == "contains":
            raise ValueError("TOP_SECRET_CONTAINS")
        return super().__contains__(key)

    def __getitem__(self, key: str) -> object:
        if self.mode == "getitem":
            raise KeyError("TOP_SECRET_GETITEM")
        return super().__getitem__(key)


class _FailingIterator:
    def __init__(self, error: type[BaseException]) -> None:
        self.error = error

    def __iter__(self) -> _FailingIterator:
        return self

    def __next__(self) -> object:
        if isinstance(self.error, type):
            raise self.error("QWEN35_ITERATION_BOUNDARY")
        raise self.error


class _ShapeBoundaryParameter:
    def __init__(self, error: type[BaseException]) -> None:
        self.error = error

    @property
    def shape(self) -> tuple[int, ...]:
        if isinstance(self.error, type):
            raise self.error("QWEN35_SHAPE_BOUNDARY")
        raise self.error


@pytest.mark.parametrize(
    "factory",
    [Qwen3_5MoeForConditionalGeneration, Qwen3_5MoeForCausalLM],
)
def test_current_conditional_and_text_surfaces_are_positive(factory: type[object]) -> None:
    model = factory()
    adapter = Qwen3_5MoeStaticAdapter()
    detection = adapter.detect(model, model.config)
    assert detection.score == 1.0
    assert "shapes:exact" in detection.evidence
    inspection = inspect_static_adapter(adapter, model, model.config, _manifest())
    assert inspection.detection == detection
    assert inspection.report.facts.expert_count == 4
    assert inspection.report.facts.routed_top_k == 2
    assert inspection.report.facts.shared_expert_count == 1


def test_nested_text_architecture_may_be_omitted_but_foreign_conflicts_reject() -> None:
    model = Qwen3_5MoeForConditionalGeneration()
    del model.config.text_config.architectures
    assert Qwen3_5MoeStaticAdapter().detect(model, model.config).score == 1.0
    model.config.text_config.architectures = ("ForeignTextModel",)
    assert Qwen3_5MoeStaticAdapter().detect(model, model.config).score == 0.0


@pytest.mark.parametrize("conditional", [True, False])
def test_mapping_config_surfaces_are_supported_with_identity(conditional: bool) -> None:
    model = _mapping_model(conditional=conditional)
    assert Qwen3_5MoeStaticAdapter().detect(model, model.config).score == 1.0


@pytest.mark.parametrize("mode", ["contains", "getitem"])
def test_hostile_mapping_config_access_is_safe_and_redacted(mode: str) -> None:
    source = _mapping_model(conditional=False)
    config = _HostileMapping(source.config, mode)
    source.config = config
    detection = Qwen3_5MoeStaticAdapter().detect(source, config)
    assert detection.score == 0.0
    assert "TOP_SECRET" not in " ".join(detection.warnings)


def test_descriptor_is_independent_and_frozen() -> None:
    adapter = Qwen3_5MoeStaticAdapter()
    descriptor = adapter.descriptor
    assert descriptor.name == "huggingface-qwen3.5-moe-static"
    assert descriptor.version == "1.0"
    assert descriptor.architecture_families == ("qwen3_5_moe",)
    assert descriptor.compatibility_notes == (
        "official Transformers v5.14 packed conditional and text surfaces are supported",
        "shared experts are structural and not router targets",
        "structure-only; routing and model certification are not provided",
    )
    assert Qwen3_5MoeStaticAdapter.__slots__ == ()
    assert Qwen3_5MoeStaticAdapter.__bases__ == (object,)
    assert set(inspect.signature(adapter.detect).parameters) == {"model", "config"}
    assert set(inspect.signature(adapter.discover).parameters) == {"model", "model_manifest"}


def test_text_base_surface_with_bare_layers_is_accepted() -> None:
    model = Qwen3_5MoeModel(surface="bare")
    # The bare base surface is only valid with an exact official text identity.
    model.config.model_type = "qwen3_5_moe_text"
    model.config.architectures = ("Qwen3_5MoeTextModel",)
    detection = Qwen3_5MoeStaticAdapter().detect(model, model.config)
    assert detection.score == 1.0


def test_report_has_only_model_neutral_structure_and_shared_non_routed() -> None:
    model = Qwen3_5MoeForConditionalGeneration()
    inspection = inspect_static_adapter(Qwen3_5MoeStaticAdapter(), model, model.config, _manifest())
    components = inspection.report.components
    assert all(component.capabilities == [CapabilityLabel.STRUCTURE] for component in components)
    assert all(
        component.capture is not None and not component.capture.verified for component in components
    )
    assert {component.capture.method for component in components if component.capture} == {
        "qwen3.5-moe-static-structure-v1"
    }
    per_layer = 1 + 1 + 1 + 4 + 1
    assert len(components) == 2 * per_layer
    assert sum(component.kind is ComponentKind.ROUTER for component in components) == 2
    assert sum(component.kind is ComponentKind.SHARED_EXPERT for component in components) == 2
    shared = [
        component for component in components if component.kind is ComponentKind.SHARED_EXPERT
    ]
    assert all(component.shared is True and component.routed is False for component in shared)
    experts = [component for component in components if component.kind is ComponentKind.EXPERT]
    assert len(experts) == 8
    assert all(component.routed is True and component.shared is False for component in experts)
    assert all("shared_expert_gate.weight" in component.tensor_shapes for component in shared)
    assert all("denominator" not in str(component.tensor_shapes).lower() for component in shared)


def test_official_v514_descendants_are_required_and_foreign_mlp_children_reject() -> None:
    model = Qwen3_5MoeForConditionalGeneration()
    assert Qwen3_5MoeStaticAdapter().detect(model, model.config).score == 1.0
    model._modules.pop("model.language_model.layers.0.mlp.experts.act_fn")
    assert Qwen3_5MoeStaticAdapter().detect(model, model.config).score == 0.0
    model = Qwen3_5MoeForConditionalGeneration()
    model._modules["model.language_model.layers.1.mlp.shared_expert.foreign"] = object()
    assert Qwen3_5MoeStaticAdapter().detect(model, model.config).score == 0.0


def test_conditional_language_model_exposes_exact_nested_text_config() -> None:
    model = Qwen3_5MoeForConditionalGeneration()
    language_model = model._modules["model.language_model"]
    language_model.config = object()
    assert Qwen3_5MoeStaticAdapter().detect(model, model.config).score == 0.0


def test_layer_indices_are_numeric_and_reported_in_numeric_order() -> None:
    text_config = Qwen3_5MoeConfig().text_config
    text_config.num_hidden_layers = 11
    text_config.layer_types = ("full_attention",) * 11
    model = Qwen3_5MoeForConditionalGeneration(
        config=Qwen3_5MoeConfig(text_config=text_config),
        num_layers=11,
    )
    report = Qwen3_5MoeStaticAdapter().discover(model, _manifest())
    routers = [
        component for component in report.components if component.kind is ComponentKind.ROUTER
    ]
    assert [component.layer_index for component in routers] == list(range(11))
    assert routers[2].module_path.endswith("layers.2.mlp.gate")
    assert routers[10].module_path.endswith("layers.10.mlp.gate")
    inspection = inspect_static_adapter(Qwen3_5MoeStaticAdapter(), model, model.config, _manifest())
    plan = build_routing_probe_plan(inspection)
    assert len(plan.targets) == 11
    assert all("shared_expert" not in target.module_path for target in plan.targets)
    expected_paths = [f"model.language_model.layers.{index}.mlp.gate" for index in range(11)]
    assert {target.module_path for target in plan.targets} == set(expected_paths)
    assert [target.module_path for target in plan.targets] == expected_paths


def test_packed_expert_components_are_logical_slices() -> None:
    model = Qwen3_5MoeForCausalLM()
    report = Qwen3_5MoeStaticAdapter().discover(model, _manifest())
    experts = [
        component for component in report.components if component.kind is ComponentKind.EXPERT
    ]
    assert {component.module_path for component in experts} == {
        "model.layers.0.mlp.experts",
        "model.layers.1.mlp.experts",
    }
    assert {component.expert_index for component in experts} == {0, 1, 2, 3}
    assert all(component.warnings for component in experts)
    assert "logical" in " ".join(report.warnings)


def test_report_facts_component_order_shapes_provenance_and_json_digest() -> None:
    model = Qwen3_5MoeForConditionalGeneration()
    report = Qwen3_5MoeStaticAdapter().discover(model, _manifest())
    assert report.facts.model_dump(mode="json") == {
        "expert_count": 4,
        "expert_count_source": "config.num_experts",
        "routed_top_k": 2,
        "routed_top_k_source": "config.num_experts_per_tok",
        "shared_expert_count": 1,
        "shared_expert_count_source": "topology.shared_expert",
    }
    expected_kinds = [
        ComponentKind.MOE_LAYER,
        ComponentKind.ROUTER,
        ComponentKind.EXPERT_CONTAINER,
        ComponentKind.EXPERT,
        ComponentKind.EXPERT,
        ComponentKind.EXPERT,
        ComponentKind.EXPERT,
        ComponentKind.SHARED_EXPERT,
    ]
    assert [candidate.kind for candidate in report.candidates[:8]] == expected_kinds
    assert [component.kind for component in report.components[:8]] == expected_kinds
    assert report.warnings == [
        "packed expert slices are logical and are not independently hookable"
    ]
    for candidate, component in zip(report.candidates, report.components, strict=True):
        assert candidate.component_key == component.component_key
        assert candidate.component_key == make_component_key(
            report.model_key,
            candidate.kind.value,
            candidate.module_path,
            layer_index=candidate.layer_index,
            expert_index=candidate.expert_index,
        )
        assert component.capabilities == [CapabilityLabel.STRUCTURE]
        assert component.capture is not None
        assert component.capture.source is CaptureSource.STATIC_STRUCTURE
        assert component.capture.method == "qwen3.5-moe-static-structure-v1"
        assert component.capture.adapter == "huggingface-qwen3.5-moe-static"
        assert component.capture.adapter_version == "1.0"
        assert component.capture.verified is False
        assert component.capture.metadata == {"layout": "packed"}
        assert component.provenance is not None
        assert component.provenance.source == "qwen3.5-moe-static-structure-v1"
        assert component.provenance.metadata == {
            "layout": "packed",
            "evidence": ["config", "topology", "shapes"],
        }
    container = next(
        component
        for component in report.components
        if component.kind is ComponentKind.EXPERT_CONTAINER
    )
    assert container.tensor_shapes == {
        "gate_up_proj": [4, 24, 8],
        "down_proj": [4, 8, 12],
    }
    shared = next(
        component
        for component in report.components
        if component.kind is ComponentKind.SHARED_EXPERT
    )
    assert shared.tensor_shapes == {
        "gate_proj.weight": [16, 8],
        "up_proj.weight": [16, 8],
        "down_proj.weight": [8, 16],
        "shared_expert_gate.weight": [1, 8],
    }
    assert report.to_json() == type(report).from_json(report.to_json()).to_json()
    assert report.model_manifest.config_hash == _manifest().config_hash


def test_detection_evidence_is_exact_and_deterministic() -> None:
    model = Qwen3_5MoeForConditionalGeneration()
    adapter = Qwen3_5MoeStaticAdapter()
    first = adapter.detect(model, model.config)
    second = adapter.detect(model, model.config)
    assert first.score == 1.0
    assert first.evidence == (
        "architecture:qwen3.5-conditional-allowlist",
        "config:strict-fields-and-schedule",
        "model_type:qwen3_5_moe",
        "shapes:exact",
        "topology:packed-shared-expert",
    )
    assert first.warnings == ()
    assert first.to_json() == second.to_json()


def test_invalid_report_is_empty_safe_and_sorted() -> None:
    model = Qwen3_5MoeForConditionalGeneration()
    model.config.model_type = "foreign"
    report = Qwen3_5MoeStaticAdapter().discover(model, _manifest())
    assert report.candidates == []
    assert report.components == []
    assert report.warnings == sorted(set(report.warnings))
    assert "Qwen3.5-MoE family identity" in " ".join(report.warnings)


def test_probe_plan_contains_routers_only_in_layer_order() -> None:
    model = Qwen3_5MoeForConditionalGeneration()
    inspection = inspect_static_adapter(Qwen3_5MoeStaticAdapter(), model, model.config, _manifest())
    plan = build_routing_probe_plan(inspection)
    assert [target.module_path for target in plan.targets] == [
        "model.language_model.layers.0.mlp.gate",
        "model.language_model.layers.1.mlp.gate",
    ]
    assert all(target.component_kind is ComponentKind.ROUTER for target in plan.targets)
    repeat = build_routing_probe_plan(inspection)
    assert plan.to_json() == repeat.to_json()
    assert plan.plan_id == repeat.plan_id


@pytest.mark.parametrize(
    "model",
    [
        Qwen3_5MoeForConditionalGeneration(layout="legacy_indexed"),
        Qwen3_5MoeForConditionalGeneration(layout="mixed"),
    ],
)
def test_indexed_or_mixed_experts_are_rejected(model: object) -> None:
    assert Qwen3_5MoeStaticAdapter().detect(model, model.config).score == 0.0


def test_wrong_family_and_fuzzy_architecture_are_rejected() -> None:
    model = Qwen3_5MoeForConditionalGeneration()
    model.config.model_type = "qwen3_moe"
    assert Qwen3_5MoeStaticAdapter().detect(model, model.config).score == 0.0
    model.config.model_type = "qwen3_5_moe"
    model.config.architectures = ("Qwen3_5MoeForConditionalGenerationV2",)
    assert Qwen3_5MoeStaticAdapter().detect(model, model.config).score == 0.0


def test_nested_text_identity_is_required_for_conditional_surface() -> None:
    model = Qwen3_5MoeForConditionalGeneration()
    model.config.text_config.model_type = "qwen3_moe"
    assert Qwen3_5MoeStaticAdapter().detect(model, model.config).score == 0.0


@pytest.mark.parametrize(
    "mutate",
    [
        lambda config: setattr(config, "model_type", "qwen3_moe"),
        lambda config: setattr(config, "model_type", "qwen3_5_moe_v2"),
        lambda config: setattr(config, "architectures", ("ForeignModel",)),
        lambda config: setattr(config, "architectures", None),
        lambda config: setattr(config, "architectures", ("Qwen3_5MoeForConditionalGenerationV2",)),
    ],
)
def test_outer_family_markers_are_exact_and_conflicts_reject(mutate) -> None:
    model = Qwen3_5MoeForConditionalGeneration()
    mutate(model.config)
    assert Qwen3_5MoeStaticAdapter().detect(model, model.config).score == 0.0


def test_conditional_text_config_is_required_and_must_be_mapping_or_attributes() -> None:
    model = Qwen3_5MoeForConditionalGeneration()
    delattr(model.config, "text_config")
    assert Qwen3_5MoeStaticAdapter().detect(model, model.config).score == 0.0
    model = Qwen3_5MoeForConditionalGeneration()
    model.config.text_config = object()
    assert Qwen3_5MoeStaticAdapter().detect(model, model.config).score == 0.0


@pytest.mark.parametrize(
    "field",
    [
        "num_hidden_layers",
        "hidden_size",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
        "num_experts",
        "num_experts_per_tok",
    ],
)
def test_positive_integer_configuration_is_strict(field: str) -> None:
    model = Qwen3_5MoeForConditionalGeneration()
    setattr(model.config.text_config, field, 0)
    assert Qwen3_5MoeStaticAdapter().detect(model, model.config).score == 0.0


@pytest.mark.parametrize("bad_value", [True, False, 0, -1, 1.5, "1", None])
@pytest.mark.parametrize(
    "field",
    [
        "num_hidden_layers",
        "hidden_size",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
        "num_experts",
        "num_experts_per_tok",
    ],
)
def test_every_structural_integer_field_rejects_non_positive_or_non_int(
    field: str, bad_value: object
) -> None:
    model = Qwen3_5MoeForConditionalGeneration()
    setattr(model.config.text_config, field, bad_value)
    assert Qwen3_5MoeStaticAdapter().detect(model, model.config).score == 0.0


@pytest.mark.parametrize(
    "field",
    [
        "num_hidden_layers",
        "hidden_size",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
        "num_experts",
        "num_experts_per_tok",
        "layer_types",
    ],
)
def test_missing_structural_config_fields_are_rejected(field: str) -> None:
    model = _mapping_model()
    model.config["text_config"].pop(field)
    assert Qwen3_5MoeStaticAdapter().detect(model, model.config).score == 0.0


@pytest.mark.parametrize(
    "layer_types",
    [
        ("full_attention",),
        ("full_attention", "foreign_attention"),
        ("full_attention", True),
        "full_attention,linear_attention",
        None,
    ],
)
def test_layer_types_is_an_exact_per_layer_allowlist(layer_types: object) -> None:
    model = Qwen3_5MoeForConditionalGeneration()
    model.config.text_config.layer_types = layer_types
    assert Qwen3_5MoeStaticAdapter().detect(model, model.config).score == 0.0


@pytest.mark.parametrize("mlp_only_layers", [None, [0], [True], "[]", {0}])
def test_mlp_only_layers_must_be_absent_or_exactly_empty(mlp_only_layers: object) -> None:
    model = Qwen3_5MoeForConditionalGeneration()
    model.config.text_config.mlp_only_layers = mlp_only_layers
    assert Qwen3_5MoeStaticAdapter().detect(model, model.config).score == 0.0


def test_top_k_must_not_exceed_expert_count() -> None:
    model = Qwen3_5MoeForConditionalGeneration()
    model.config.text_config.num_experts_per_tok = model.config.text_config.num_experts + 1
    assert Qwen3_5MoeStaticAdapter().detect(model, model.config).score == 0.0


def test_layer_schedule_and_mlp_only_layers_are_strict() -> None:
    model = Qwen3_5MoeForConditionalGeneration()
    model.config.text_config.layer_types = ("full_attention",)
    assert Qwen3_5MoeStaticAdapter().detect(model, model.config).score == 0.0
    model = Qwen3_5MoeForConditionalGeneration()
    model.config.text_config.mlp_only_layers = (0,)
    assert Qwen3_5MoeStaticAdapter().detect(model, model.config).score == 0.0


def test_missing_shared_expert_or_shared_gate_is_rejected() -> None:
    model = Qwen3_5MoeForConditionalGeneration()
    model._modules.pop("model.language_model.layers.0.mlp.shared_expert")
    assert Qwen3_5MoeStaticAdapter().detect(model, model.config).score == 0.0
    model = Qwen3_5MoeForConditionalGeneration()
    model._modules.pop("model.language_model.layers.1.mlp.shared_expert_gate")
    assert Qwen3_5MoeStaticAdapter().detect(model, model.config).score == 0.0


@pytest.mark.parametrize(
    "suffix",
    [
        "mlp",
        "mlp.gate",
        "mlp.experts",
        "mlp.experts.act_fn",
        "mlp.shared_expert",
        "mlp.shared_expert.gate_proj",
        "mlp.shared_expert.up_proj",
        "mlp.shared_expert.down_proj",
        "mlp.shared_expert.act_fn",
        "mlp.shared_expert_gate",
    ],
)
def test_every_official_mlp_descendant_is_required(suffix: str) -> None:
    model = Qwen3_5MoeForConditionalGeneration()
    model._modules.pop(f"model.language_model.layers.0.{suffix}")
    assert Qwen3_5MoeStaticAdapter().detect(model, model.config).score == 0.0


def test_layer_and_layers_root_modules_are_required() -> None:
    model = Qwen3_5MoeForConditionalGeneration()
    model._modules.pop("model.language_model.layers.0")
    assert Qwen3_5MoeStaticAdapter().detect(model, model.config).score == 0.0
    model = Qwen3_5MoeForConditionalGeneration()
    model._modules.pop("model.language_model.layers")
    assert Qwen3_5MoeStaticAdapter().detect(model, model.config).score == 0.0


def test_shape_mismatch_and_extra_mlp_parameter_are_rejected() -> None:
    model = Qwen3_5MoeForConditionalGeneration()
    model._parameters["model.language_model.layers.0.mlp.gate.weight"].shape = (1, 1)
    assert Qwen3_5MoeStaticAdapter().detect(model, model.config).score == 0.0


@pytest.mark.parametrize(
    "parameter_suffix",
    [
        "mlp.gate.weight",
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
        "mlp.shared_expert.gate_proj.weight",
        "mlp.shared_expert.up_proj.weight",
        "mlp.shared_expert.down_proj.weight",
        "mlp.shared_expert_gate.weight",
    ],
)
def test_each_of_seven_parameter_surfaces_is_required(parameter_suffix: str) -> None:
    model = Qwen3_5MoeForConditionalGeneration()
    model._parameters.pop(f"model.language_model.layers.0.{parameter_suffix}")
    assert Qwen3_5MoeStaticAdapter().detect(model, model.config).score == 0.0


@pytest.mark.parametrize(
    "bad_shape",
    [(), (4,), (4, 8, 1), (True, 8), (4, 8.0), (4, -1), "not-a-shape"],
)
def test_shape_rank_dimension_and_type_are_strict(bad_shape: object) -> None:
    model = Qwen3_5MoeForConditionalGeneration()
    parameter = model._parameters["model.language_model.layers.0.mlp.gate.weight"]
    parameter.shape = bad_shape
    assert Qwen3_5MoeStaticAdapter().detect(model, model.config).score == 0.0
    model = Qwen3_5MoeForConditionalGeneration()
    model._parameters["model.language_model.layers.0.mlp.extra.weight"] = type(
        "Parameter", (), {"shape": (1,)}
    )()
    assert Qwen3_5MoeStaticAdapter().detect(model, model.config).score == 0.0


def test_second_layer_root_is_rejected() -> None:
    model = Qwen3_5MoeForConditionalGeneration(extra_modules=("other.layers", "other.layers.0"))
    assert Qwen3_5MoeStaticAdapter().detect(model, model.config).score == 0.0


@pytest.mark.parametrize(
    "bad_module_path",
    [
        "foreign.layers.0.mlp",
        "model.language_model.layers.02",
        "model.language_model.layers.-1",
        "model.language_model.layers.0.mlp.experts.0",
        "model..language_model.layers.0",
        ".model.language_model.layers.0",
        "model.language_model.layers..0",
    ],
)
def test_module_root_layer_index_and_malformed_paths_reject(bad_module_path: str) -> None:
    model = Qwen3_5MoeForConditionalGeneration()
    model._modules[bad_module_path] = object()
    assert Qwen3_5MoeStaticAdapter().detect(model, model.config).score == 0.0


@pytest.mark.parametrize(
    "bad_parameter_path",
    [
        "foreign.layers.0.mlp.gate.weight",
        "model.language_model.layers.02.mlp.gate.weight",
        "model.language_model.layers.-1.mlp.gate.weight",
        "model.language_model.layers.0.mlp.experts.0.gate_proj.weight",
        "model..language_model.layers.0.mlp.gate.weight",
        ".model.language_model.layers.0.mlp.gate.weight",
        "model.language_model.layers..0.mlp.gate.weight",
    ],
)
def test_parameter_root_layer_index_and_malformed_paths_reject(bad_parameter_path: str) -> None:
    model = Qwen3_5MoeForConditionalGeneration()
    model._parameters[bad_parameter_path] = model._parameters[
        "model.language_model.layers.0.mlp.gate.weight"
    ]
    assert Qwen3_5MoeStaticAdapter().detect(model, model.config).score == 0.0


@pytest.mark.parametrize("surface", ["named_modules", "named_parameters"])
@pytest.mark.parametrize(
    "mode", ["missing", "noncallable", "call", "iteration", "malformed", "duplicate"]
)
def test_named_surface_boundaries_are_safe(surface: str, mode: str) -> None:
    source = Qwen3_5MoeForConditionalGeneration()
    if mode == "missing":

        class Missing:
            config = source.config

        model = Missing()
        setattr(model, "named_parameters", source.named_parameters)
    else:
        model = source
        if mode == "noncallable":
            setattr(model, surface, object())
        elif mode == "call":

            def fail() -> object:
                raise ValueError("TOP_SECRET_CALL")

            setattr(model, surface, fail)
        elif mode == "iteration":
            setattr(model, surface, lambda: _FailingIterator(ValueError))
        elif mode == "malformed":
            setattr(model, surface, lambda: (("malformed",),))
        else:
            setattr(model, surface, lambda: (("", object()), ("", object())))
    detection = Qwen3_5MoeStaticAdapter().detect(model, getattr(model, "config", object()))
    assert detection.score == 0.0
    assert "TOP_SECRET_CALL" not in " ".join(detection.warnings)


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize(
    "boundary",
    [
        "model_config",
        "config_field",
        "modules_call",
        "modules_iteration",
        "parameters_call",
        "parameters_iteration",
        "shape",
        "language_config",
    ],
)
def test_keyboard_interrupt_and_system_exit_propagate_at_every_boundary(
    error_type: type[BaseException], boundary: str
) -> None:
    error = error_type(f"QWEN35_{boundary}")
    source = Qwen3_5MoeForConditionalGeneration()
    if boundary == "model_config":

        class Model:
            @property
            def config(self) -> object:
                raise error

        model = Model()
        config = object()
    elif boundary == "config_field":

        class Config:
            def __getattr__(self, name: str) -> object:
                if name == "model_type":
                    raise error
                return getattr(source.config, name)

        config = Config()
        model = source
        model.config = config
    elif boundary == "modules_call":
        model = source

        def fail_modules_call() -> object:
            raise error

        model.named_modules = fail_modules_call
        config = model.config
    elif boundary == "modules_iteration":
        model = source
        model.named_modules = lambda: _FailingIterator(error)
        config = model.config
    elif boundary == "parameters_call":
        model = source

        def fail_parameters_call() -> object:
            raise error

        model.named_parameters = fail_parameters_call
        config = model.config
    elif boundary == "parameters_iteration":
        model = source
        model.named_parameters = lambda: _FailingIterator(error)
        config = model.config
    elif boundary == "shape":
        model = source
        model._parameters["model.language_model.layers.0.mlp.gate.weight"] = (
            _ShapeBoundaryParameter(error)
        )
        config = model.config
    else:
        model = source

        class LanguageModel:
            @property
            def config(self) -> object:
                raise error

        model._modules["model.language_model"] = LanguageModel()
        config = model.config
    with pytest.raises(error_type) as raised:
        Qwen3_5MoeStaticAdapter().detect(model, config)
    assert raised.value is error


def test_model_config_identity_and_discovery_are_safe() -> None:
    model = Qwen3_5MoeForConditionalGeneration()
    copied = Qwen3_5MoeConfig()
    assert Qwen3_5MoeStaticAdapter().detect(model, copied).score == 0.0
    adapter = Qwen3_5MoeStaticAdapter()
    first = adapter.discover(model, _manifest()).model_dump(mode="json")
    second = adapter.discover(model, _manifest()).model_dump(mode="json")
    assert first == second


def test_ordinary_hostile_language_config_access_is_redacted() -> None:
    model = Qwen3_5MoeForConditionalGeneration()

    class Hostile:
        def __getattribute__(self, name: str) -> object:
            if name == "config":
                raise ValueError("TOP_SECRET_LANGUAGE_CONFIG")
            return object.__getattribute__(self, name)

    model._modules["model.language_model"] = Hostile()
    detection = Qwen3_5MoeStaticAdapter().detect(model, model.config)
    assert detection.score == 0.0
    assert "TOP_SECRET_LANGUAGE_CONFIG" not in " ".join(detection.warnings)


def test_shape_only_probe_does_not_read_parameter_values() -> None:
    model = Qwen3_5MoeForCausalLM()

    class ForbiddenParameter:
        shape = (4, 8)

        def __getattribute__(self, name: str) -> object:
            if name in {"data", "device", "dtype", "detach", "cpu", "numpy"}:
                raise AssertionError(name)
            return object.__getattribute__(self, name)

    model._parameters["model.layers.0.mlp.gate.weight"] = ForbiddenParameter()
    assert Qwen3_5MoeStaticAdapter().detect(model, model.config).score == 1.0


def test_adapter_is_stateless_and_does_not_retain_model_objects() -> None:
    adapter = Qwen3_5MoeStaticAdapter()
    model = Qwen3_5MoeForConditionalGeneration()
    model_ref = weakref.ref(model)
    config_ref = weakref.ref(model.config)
    adapter.detect(model, model.config)
    adapter.discover(model, _manifest())
    del model
    gc.collect()
    assert model_ref() is None
    assert config_ref() is None
    assert not hasattr(adapter, "__dict__")


def test_source_has_no_runtime_or_older_qwen3_dependency() -> None:
    source_path = Path(__file__).parents[1] / "src" / "moeatlas" / "adapters" / "qwen3_5_moe.py"
    source = source_path.read_text()
    tree = ast.parse(source)
    imported_names = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in node.names
    }
    assert imported_names.isdisjoint(
        {
            "torch",
            "transformers",
            "numpy",
            "safetensors",
            "os",
            "pathlib",
            "socket",
            "subprocess",
            "tempfile",
            "shutil",
            "requests",
            "urllib",
            "importlib",
            "pickle",
            "sqlite3",
            "duckdb",
            "fastapi",
            "flask",
            "starlette",
            "httpx",
            "websocket",
            "redis",
            "cachetools",
        }
    )
    assert "Qwen3MoeStaticAdapter" not in source
    forbidden_call_names = {
        "__import__",
        "eval",
        "exec",
        "open",
        "forward",
        "generate",
        "from_pretrained",
        "register_forward_hook",
        "register_full_backward_hook",
        "load",
        "save",
        "connect",
        "urlopen",
        "mkdtemp",
        "cache",
        "write",
        "unlink",
        "remove",
        "rmtree",
        "request",
        "post",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function_name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else node.func.attr
                if isinstance(node.func, ast.Attribute)
                else None
            )
            assert function_name not in forbidden_call_names
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"forward", "generate", "register_forward_hook"}
    class_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    adapter_class = next(node for node in class_nodes if node.name == "Qwen3_5MoeStaticAdapter")
    assert adapter_class.bases == []
