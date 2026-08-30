from __future__ import annotations

import ast
import gc
import inspect
import weakref
from pathlib import Path

import pytest

from moeatlas.adapters import Qwen4ExpStaticAdapter, inspect_static_adapter
from moeatlas.core import (
    CapabilityLabel,
    ComponentKind,
    DType,
    ModelManifest,
    TokenizerIdentity,
    make_config_hash,
    make_model_key,
)

from .fixtures.qwen4_exp import (
    FakeParameter,
    Qwen4ExpConfig,
    Qwen4ExpForConditionalGeneration,
)


def _manifest() -> ModelManifest:
    return ModelManifest(
        model_key=make_model_key("Qwen/Qwen3.8-Flash-Next-FP8", "fixture"),
        architecture="qwen4_exp",
        revision="fixture",
        config_hash=make_config_hash({"fixture": "qwen4_exp"}),
        tokenizer=TokenizerIdentity(identifier="fixture/tokenizer", revision="fixture"),
        dtype=DType.FLOAT32,
        device_map={"": "cpu"},
    )


def test_official_conditional_surface_is_positive_and_structural() -> None:
    model = Qwen4ExpForConditionalGeneration()
    adapter = Qwen4ExpStaticAdapter()
    detection = adapter.detect(model, model.config)
    assert detection.score == 1.0
    assert detection.evidence == (
        "architecture:qwen4-exp-conditional-allowlist",
        "config:strict-fields-and-schedule",
        "model_type:qwen4_exp",
        "shapes:exact",
        "topology:packed-shared-expert",
    )
    inspection = inspect_static_adapter(adapter, model, model.config, _manifest())
    assert inspection.report.facts.expert_count == 4
    assert inspection.report.facts.routed_top_k == 2
    assert inspection.report.facts.shared_expert_count == 1
    assert all(
        component.capabilities == [CapabilityLabel.STRUCTURE]
        for component in inspection.report.components
    )
    assert all(
        component.capture is not None and component.capture.verified is False
        for component in inspection.report.components
    )


def test_descriptor_is_frozen_and_adapter_does_not_retain_state() -> None:
    adapter = Qwen4ExpStaticAdapter()
    descriptor = adapter.descriptor
    assert descriptor.name == "huggingface-qwen4-exp-static"
    assert descriptor.version == "1.0"
    assert descriptor.architecture_families == ("qwen4_exp",)
    assert Qwen4ExpStaticAdapter.__slots__ == ()
    assert not hasattr(adapter, "__dict__")
    assert set(inspect.signature(adapter.detect).parameters) == {"model", "config"}
    assert set(inspect.signature(adapter.discover).parameters) == {"model", "model_manifest"}
    model = Qwen4ExpForConditionalGeneration()
    model_ref = weakref.ref(model)
    config_ref = weakref.ref(model.config)
    adapter.detect(model, model.config)
    adapter.discover(model, _manifest())
    del model
    gc.collect()
    assert model_ref() is None
    assert config_ref() is None


@pytest.mark.parametrize(
    ("model_type", "architecture"),
    [
        ("qwen4_exp_v2", "Qwen4ExpForConditionalGeneration"),
        ("qwen4_exp", "Qwen4ExpForConditionalGeneratio"),
        ("qwen4_exp", "Qwen4ExpForCausalLM"),
    ],
)
def test_foreign_or_fuzzy_outer_identity_is_rejected(model_type: str, architecture: str) -> None:
    model = Qwen4ExpForConditionalGeneration()
    model.config.model_type = model_type
    model.config.architectures = (architecture,)
    assert Qwen4ExpStaticAdapter().detect(model, model.config).score == 0.0


def test_nested_identity_and_config_object_identity_are_strict() -> None:
    model = Qwen4ExpForConditionalGeneration()
    model.config.text_config.model_type = "qwen4_exp_text_v2"
    assert Qwen4ExpStaticAdapter().detect(model, model.config).score == 0.0
    model = Qwen4ExpForConditionalGeneration()
    model.config.text_config.model_type = "qwen4_exp_text"
    supplied = Qwen4ExpConfig()
    assert Qwen4ExpStaticAdapter().detect(model, supplied).score == 0.0


@pytest.mark.parametrize(
    "field, value",
    [
        ("num_hidden_layers", 0),
        ("hidden_size", True),
        ("moe_intermediate_size", -1),
        ("shared_expert_intermediate_size", 0),
        ("num_experts", 0),
        ("num_experts_per_tok", 5),
    ],
)
def test_bad_positive_integer_config_or_topk_is_rejected(field: str, value: object) -> None:
    model = Qwen4ExpForConditionalGeneration()
    setattr(model.config.text_config, field, value)
    assert Qwen4ExpStaticAdapter().detect(model, model.config).score == 0.0


def test_layer_schedule_must_match_exact_count_and_allowed_kinds() -> None:
    model = Qwen4ExpForConditionalGeneration()
    model.config.text_config.layer_types = ("linear_attention",)
    assert Qwen4ExpStaticAdapter().detect(model, model.config).score == 0.0
    model = Qwen4ExpForConditionalGeneration()
    model.config.text_config.layer_types = ("qwen_sparse_attention", "full_attention")
    assert Qwen4ExpStaticAdapter().detect(model, model.config).score == 0.0


def test_topology_requires_exact_qwen4_root_and_mlp_children() -> None:
    model = Qwen4ExpForConditionalGeneration()
    model._modules.pop("model.language_model.layers.0.mlp.experts.act_fn")
    assert Qwen4ExpStaticAdapter().detect(model, model.config).score == 0.0
    model = Qwen4ExpForConditionalGeneration()
    model._modules["model.language_model.layers.1.mlp.foreign"] = object()
    assert Qwen4ExpStaticAdapter().detect(model, model.config).score == 0.0
    model = Qwen4ExpForConditionalGeneration()
    model._modules["model.layers"] = object()
    assert Qwen4ExpStaticAdapter().detect(model, model.config).score == 0.0


def test_packed_parameter_shapes_are_exact() -> None:
    model = Qwen4ExpForConditionalGeneration()
    model._parameters["model.language_model.layers.0.mlp.experts.down_proj"] = FakeParameter(
        (4, 8, 13)
    )
    assert Qwen4ExpStaticAdapter().detect(model, model.config).score == 0.0
    model = Qwen4ExpForConditionalGeneration()
    model._parameters["model.language_model.layers.1.mlp.gate.weight"] = FakeParameter((4, 8))
    model._parameters["model.language_model.layers.1.mlp.gate.extra"] = FakeParameter((1,))
    assert Qwen4ExpStaticAdapter().detect(model, model.config).score == 0.0


def test_report_excludes_shared_expert_from_routed_logical_experts() -> None:
    model = Qwen4ExpForConditionalGeneration()
    report = Qwen4ExpStaticAdapter().discover(model, _manifest())
    experts = [
        component for component in report.components if component.kind is ComponentKind.EXPERT
    ]
    shared = [
        component
        for component in report.components
        if component.kind is ComponentKind.SHARED_EXPERT
    ]
    assert len(experts) == 8
    assert all(component.routed is True and component.shared is False for component in experts)
    assert len(shared) == 2
    assert all(component.routed is False and component.shared is True for component in shared)
    assert all("shared_expert" not in component.module_path for component in experts)
    assert all("shared_expert_gate.weight" in component.tensor_shapes for component in shared)


def test_language_model_exposes_the_exact_nested_config() -> None:
    model = Qwen4ExpForConditionalGeneration()
    model._modules["model.language_model"].config = object()
    assert Qwen4ExpStaticAdapter().detect(model, model.config).score == 0.0


def test_shape_probe_reads_metadata_only() -> None:
    model = Qwen4ExpForConditionalGeneration()

    class Parameter:
        shape = (4, 8)

        def __getattribute__(self, name: str) -> object:
            if name in {"data", "device", "dtype", "detach", "cpu", "numpy", "item"}:
                raise AssertionError(name)
            return object.__getattribute__(self, name)

    model._parameters["model.language_model.layers.0.mlp.gate.weight"] = Parameter()
    assert Qwen4ExpStaticAdapter().detect(model, model.config).score == 1.0


def test_source_has_no_runtime_imports_or_execution_calls() -> None:
    source = (
        Path(__file__).parents[1] / "src" / "moeatlas" / "adapters" / "qwen4_exp.py"
    ).read_text()
    tree = ast.parse(source)
    imported_names = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in node.names
    }
    assert imported_names.isdisjoint(
        {"torch", "transformers", "numpy", "safetensors", "os", "subprocess", "urllib"}
    )
    forbidden = {
        "forward",
        "generate",
        "from_pretrained",
        "register_forward_hook",
        "__import__",
        "eval",
        "exec",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else node.func.attr
                if isinstance(node.func, ast.Attribute)
                else None
            )
            assert name not in forbidden
        if isinstance(node, ast.Attribute):
            assert node.attr not in forbidden
