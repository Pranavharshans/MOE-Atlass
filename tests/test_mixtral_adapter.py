from __future__ import annotations

import dataclasses
import socket
from pathlib import Path
from urllib import request

import pytest

from moeatlas.adapters import MixtralStaticAdapter, inspect_static_adapter
from moeatlas.core import (
    CapabilityLabel,
    CaptureSource,
    ComponentKind,
    DType,
    ModelManifest,
    TokenizerIdentity,
    make_config_hash,
    make_model_key,
)

from .fixtures import FakeParameter, MixtralConfig, MixtralConfigMapping, MixtralForCausalLM


def _manifest() -> ModelManifest:
    return ModelManifest(
        model_key=make_model_key("acme/mixtral", "main"),
        architecture="mixtral",
        revision="main",
        config_hash=make_config_hash({"experts": 4, "top_k": 2}),
        tokenizer=TokenizerIdentity(identifier="acme/tokenizer", revision="main"),
        dtype=DType.FLOAT32,
        device_map={"": "cpu"},
    )


def _parameter_entries(model: MixtralForCausalLM) -> list[tuple[str, FakeParameter]]:
    return list(model.named_parameters())


def _module_entries(model: MixtralForCausalLM) -> list[tuple[str, object]]:
    return list(model.named_modules())


def _structural_target(layout: str, category: str, layer_index: int = 1) -> tuple[str, str]:
    prefix = f"layers.{layer_index}"
    if layout == "legacy":
        return f"{prefix}.block_sparse_moe", category
    if category == "mlp.top_k":
        return f"{prefix}.mlp", "top_k"
    if category == "gate.num_experts":
        return f"{prefix}.mlp.gate", "num_experts"
    if category == "gate.top_k":
        return f"{prefix}.mlp.gate", "top_k"
    return f"{prefix}.mlp.experts", "num_experts"


def test_descriptor_is_exact_and_publicly_selectable() -> None:
    adapter = MixtralStaticAdapter()

    assert adapter.descriptor.name == "huggingface-mixtral-static"
    assert adapter.descriptor.version == "1.0"
    assert adapter.descriptor.architecture_families == ("mixtral",)
    assert adapter.descriptor.compatibility_notes == tuple(
        sorted(adapter.descriptor.compatibility_notes)
    )
    assert "4.50" in " ".join(adapter.descriptor.compatibility_notes)
    assert "packed" in " ".join(adapter.descriptor.compatibility_notes)
    assert "routing certification" in " ".join(adapter.descriptor.compatibility_notes)


@pytest.mark.parametrize("layout", ["legacy", "packed"])
@pytest.mark.parametrize("prefix", ["", "model.transformer"])
def test_valid_layouts_have_deterministic_positive_detection(layout: str, prefix: str) -> None:
    model = MixtralForCausalLM(layout=layout, prefix=prefix)
    adapter = MixtralStaticAdapter()

    first = adapter.detect(model, model.config)
    second = adapter.detect(model, model.config)

    assert 0.75 <= first.score <= 1.0
    assert first.score == second.score
    assert first.evidence == second.evidence
    assert first.warnings == ()
    assert first.to_json() == second.to_json()


def test_fixtures_match_official_child_and_parameter_surfaces() -> None:
    legacy = MixtralForCausalLM(layout="legacy")
    legacy_modules = {path for path, _ in _module_entries(legacy)}
    legacy_parameters = {path for path, _ in _parameter_entries(legacy)}
    for layer_index in range(2):
        root = f"layers.{layer_index}.block_sparse_moe.experts"
        for expert_index in range(4):
            expert = f"{root}.{expert_index}"
            assert {f"{expert}.{name}" for name in ("w1", "w2", "w3", "act_fn")} <= (legacy_modules)
        assert f"layers.{layer_index}.block_sparse_moe.gate.weight" in legacy_parameters

    packed = MixtralForCausalLM(layout="packed")
    packed_modules = {path for path, _ in _module_entries(packed)}
    packed_parameters = {path for path, _ in _parameter_entries(packed)}
    for layer_index in range(2):
        root = f"layers.{layer_index}.mlp"
        assert f"{root}.experts.act_fn" in packed_modules
        assert f"{root}.experts.gate_up_proj" in packed_parameters
        assert f"{root}.experts.down_proj" in packed_parameters
    assert MixtralStaticAdapter().detect(packed, packed.config).score > 0


def test_legacy_discovery_reports_exact_order_paths_facts_and_provenance() -> None:
    model = MixtralForCausalLM(layout="legacy")
    report = MixtralStaticAdapter().discover(model, _manifest())

    assert report.facts.expert_count == 4
    assert report.facts.expert_count_source == "config.num_local_experts"
    assert report.facts.routed_top_k == 2
    assert report.facts.routed_top_k_source == "config.num_experts_per_tok"
    assert report.facts.shared_expert_count is None
    assert report.warnings == []
    assert len(report.candidates) == 2 * (3 + 4)
    for layer_index in range(2):
        start = layer_index * 7
        layer_candidates = report.candidates[start : start + 7]
        assert [candidate.kind for candidate in layer_candidates] == [
            ComponentKind.MOE_LAYER,
            ComponentKind.ROUTER,
            ComponentKind.EXPERT_CONTAINER,
            ComponentKind.EXPERT,
            ComponentKind.EXPERT,
            ComponentKind.EXPERT,
            ComponentKind.EXPERT,
        ]
        assert layer_candidates[0].module_path == f"layers.{layer_index}.block_sparse_moe"
        assert layer_candidates[1].module_path == (f"layers.{layer_index}.block_sparse_moe.gate")
        assert layer_candidates[2].module_path == (f"layers.{layer_index}.block_sparse_moe.experts")
        assert [candidate.expert_index for candidate in layer_candidates[3:]] == [0, 1, 2, 3]
        assert [candidate.routed for candidate in layer_candidates[3:]] == [True] * 4
        for candidate, component in zip(
            layer_candidates, report.components[start : start + 7], strict=True
        ):
            assert candidate.component_key == component.component_key
            assert component.capabilities == [CapabilityLabel.STRUCTURE]
            assert component.capture is not None
            assert component.capture.source is CaptureSource.STATIC_STRUCTURE
            assert component.capture.method == "mixtral-static-structure-v1"
            assert component.capture.adapter == "huggingface-mixtral-static"
            assert component.capture.adapter_version == "1.0"
            assert component.capture.verified is False
            assert component.capture.metadata == {"layout": "legacy_indexed"}
            assert component.provenance is not None
            assert component.provenance.metadata == {
                "layout": "legacy_indexed",
                "evidence": ["config", "topology", "shapes"],
            }
            assert candidate.evidence[0].detail == "Mixtral legacy_indexed layout"
            assert component.shared is None
    expert = report.components[3]
    assert expert.tensor_shapes == {
        "w1.weight": [16, 8],
        "w2.weight": [8, 16],
        "w3.weight": [16, 8],
    }


def test_packed_discovery_uses_logical_expert_slices_and_fixed_warning() -> None:
    model = MixtralForCausalLM(layout="packed", prefix="model")
    report = MixtralStaticAdapter().discover(model, _manifest())

    assert report.warnings == [
        "packed expert slices are logical and are not independently hookable"
    ]
    packed_experts = [
        component for component in report.components if component.kind is ComponentKind.EXPERT
    ]
    assert len(packed_experts) == 8
    assert {component.module_path for component in packed_experts} == {
        "model.layers.0.mlp.experts",
        "model.layers.1.mlp.experts",
    }
    assert [component.expert_index for component in packed_experts[:4]] == [0, 1, 2, 3]
    assert packed_experts[0].tensor_shapes == {
        "gate_up_proj": [32, 8],
        "down_proj": [8, 16],
    }
    assert packed_experts[0].capture is not None
    assert packed_experts[0].capture.metadata == {"layout": "packed"}
    assert packed_experts[0].provenance is not None
    assert packed_experts[0].provenance.metadata == {
        "layout": "packed",
        "evidence": ["config", "topology", "shapes"],
    }
    assert packed_experts[0].warnings == report.warnings
    assert all(component.capture is not None for component in packed_experts)


def test_mapping_config_identity_is_required_and_supported() -> None:
    config = MixtralConfigMapping()
    model = MixtralForCausalLM(config=config)
    adapter = MixtralStaticAdapter()

    assert adapter.detect(model, config).score > 0
    assert adapter.discover(model, _manifest()).facts.expert_count == 4
    assert adapter.detect(model, dict(config)).score == 0.0


@pytest.mark.parametrize(
    "config_update",
    [
        {"model_type": "qwen2"},
        {"model_type": ""},
        {"num_hidden_layers": True},
        {"hidden_size": 8.0},
        {"intermediate_size": "16"},
        {"num_local_experts": 0},
        {"num_experts_per_tok": 5},
    ],
)
def test_family_and_strict_config_negatives_are_zero(config_update: dict[str, object]) -> None:
    config = dataclasses.replace(MixtralConfig(), **config_update)
    model = MixtralForCausalLM(config=config)

    detection = MixtralStaticAdapter().detect(model, config)

    assert detection.score == 0.0
    assert detection.evidence == ()
    assert detection.warnings
    assert all("TOP_SECRET" not in warning for warning in detection.warnings)


class Qwen2ForCausalLM(MixtralForCausalLM):
    pass


def test_model_class_name_does_not_change_exact_config_family_detection() -> None:
    config = MixtralConfig()
    model = Qwen2ForCausalLM(config=config)
    baseline = MixtralForCausalLM(config=config)

    detection = MixtralStaticAdapter().detect(model, config)
    baseline_detection = MixtralStaticAdapter().detect(baseline, config)
    assert detection.score == baseline_detection.score
    assert detection.evidence == baseline_detection.evidence


def test_class_name_cannot_spoof_missing_family_markers() -> None:
    config = MixtralConfigMapping()
    model = MixtralForCausalLM(config=config)
    config.pop("model_type")
    config.pop("architectures")

    detection = MixtralStaticAdapter().detect(model, config)
    assert detection.score == 0.0
    assert detection.evidence == ()
    assert detection.warnings


@pytest.mark.parametrize(
    "config_update",
    [
        {"model_type": "qwen2"},
        {"architectures": ("QwenForCausalLM",)},
        {"architectures": ("MixtralForCausalLM", "QwenForCausalLM")},
    ],
)
def test_conflicting_family_markers_are_zero(config_update: dict[str, object]) -> None:
    config = dataclasses.replace(MixtralConfig(), **config_update)
    model = MixtralForCausalLM(config=config)

    detection = MixtralStaticAdapter().detect(model, config)
    assert detection.score == 0.0
    assert detection.evidence == ()


@pytest.mark.parametrize(
    ("layout", "category"),
    [
        ("legacy", "num_experts"),
        ("legacy", "top_k"),
        ("packed", "mlp.top_k"),
        ("packed", "gate.num_experts"),
        ("packed", "gate.top_k"),
        ("packed", "experts.num_experts"),
    ],
)
@pytest.mark.parametrize("invalid_value", [None, True, 2.0, "2"])
def test_structural_attributes_are_strict_and_configuration_bound(
    layout: str, category: str, invalid_value: object
) -> None:
    model = MixtralForCausalLM(layout=layout)
    path, attribute = _structural_target(layout, category)
    modules = dict(_module_entries(model))
    target = modules[path]
    if invalid_value is None:
        delattr(target, attribute)
    else:
        setattr(target, attribute, invalid_value)

    detection = MixtralStaticAdapter().detect(model, model.config)
    assert detection.score == 0.0
    assert detection.evidence == ()
    assert "TOP_SECRET" not in " ".join(detection.warnings)


@pytest.mark.parametrize(
    ("layout", "category"),
    [
        ("legacy", "num_experts"),
        ("legacy", "top_k"),
        ("packed", "mlp.top_k"),
        ("packed", "gate.num_experts"),
        ("packed", "gate.top_k"),
        ("packed", "experts.num_experts"),
    ],
)
def test_structural_attributes_must_match_on_every_layer(layout: str, category: str) -> None:
    model = MixtralForCausalLM(layout=layout)
    path, attribute = _structural_target(layout, category, layer_index=1)
    target = dict(_module_entries(model))[path]
    setattr(target, attribute, 99)

    detection = MixtralStaticAdapter().detect(model, model.config)
    assert detection.score == 0.0
    assert any("disagrees with config" in warning for warning in detection.warnings)


@pytest.mark.parametrize(
    ("layout", "category"),
    [
        ("legacy", "num_experts"),
        ("legacy", "top_k"),
        ("packed", "mlp.top_k"),
        ("packed", "gate.num_experts"),
        ("packed", "gate.top_k"),
        ("packed", "experts.num_experts"),
    ],
)
def test_ordinary_structural_attribute_errors_are_redacted(layout: str, category: str) -> None:
    model = MixtralForCausalLM(layout=layout)
    path, attribute = _structural_target(layout, category)
    expected = RuntimeError("TOP_SECRET_STRUCTURAL_ATTRIBUTE")

    class RaisingModule:
        def __getattr__(self, name: str) -> object:
            if name == attribute:
                raise expected
            raise AttributeError(name)

    model._modules = [
        (module_path, RaisingModule() if module_path == path else module)
        for module_path, module in model._modules
    ]
    detection = MixtralStaticAdapter().detect(model, model.config)
    assert detection.score == 0.0
    assert "TOP_SECRET" not in " ".join(detection.warnings)
    assert any("RuntimeError" in warning for warning in detection.warnings)


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize(
    ("layout", "category"),
    [
        ("legacy", "num_experts"),
        ("legacy", "top_k"),
        ("packed", "mlp.top_k"),
        ("packed", "gate.num_experts"),
        ("packed", "gate.top_k"),
        ("packed", "experts.num_experts"),
    ],
)
def test_control_flow_from_structural_attributes_is_unchanged(
    exception_type: type[BaseException], layout: str, category: str
) -> None:
    model = MixtralForCausalLM(layout=layout)
    path, attribute = _structural_target(layout, category)
    expected = exception_type("TOP_SECRET_STRUCTURAL_CONTROL_FLOW")

    class RaisingModule:
        def __getattr__(self, name: str) -> object:
            if name == attribute:
                raise expected
            raise AttributeError(name)

    model._modules = [
        (module_path, RaisingModule() if module_path == path else module)
        for module_path, module in model._modules
    ]
    with pytest.raises(exception_type) as caught:
        MixtralStaticAdapter().detect(model, model.config)
    assert caught.value is expected


@pytest.mark.parametrize(
    "field_name",
    [
        "model_type",
        "architectures",
        "num_hidden_layers",
        "hidden_size",
        "intermediate_size",
        "num_local_experts",
        "num_experts_per_tok",
    ],
)
def test_ordinary_config_field_errors_are_redacted(field_name: str) -> None:
    model = MixtralForCausalLM()
    expected = RuntimeError("TOP_SECRET_CONFIG_FIELD")
    source = model.config

    class RaisingConfig:
        def __getattr__(self, name: str) -> object:
            if name == field_name:
                raise expected
            return getattr(source, name)

    model.config = RaisingConfig()
    detection = MixtralStaticAdapter().detect(model, model.config)
    assert detection.score == 0.0
    assert "TOP_SECRET" not in " ".join(detection.warnings)
    assert any("RuntimeError" in warning for warning in detection.warnings)


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize(
    "field_name",
    [
        "model_type",
        "architectures",
        "num_hidden_layers",
        "hidden_size",
        "intermediate_size",
        "num_local_experts",
        "num_experts_per_tok",
    ],
)
def test_control_flow_from_config_fields_is_unchanged(
    exception_type: type[BaseException], field_name: str
) -> None:
    model = MixtralForCausalLM()
    expected = exception_type("TOP_SECRET_CONFIG_CONTROL_FLOW")
    source = model.config

    class RaisingConfig:
        def __getattr__(self, name: str) -> object:
            if name == field_name:
                raise expected
            return getattr(source, name)

    model.config = RaisingConfig()
    with pytest.raises(exception_type) as caught:
        MixtralStaticAdapter().detect(model, model.config)
    assert caught.value is expected


def test_canonical_paths_without_family_identity_do_not_false_positive() -> None:
    class CanonicalPathModel(MixtralForCausalLM):
        pass

    config = dataclasses.replace(MixtralConfig(), model_type="not-mixtral", architectures=())
    model = CanonicalPathModel(config=config)

    assert MixtralStaticAdapter().detect(model, config).score == 0.0


def test_missing_family_fields_and_structure_are_zero() -> None:
    class GenericModel:
        def __init__(self) -> None:
            self.config = {
                "num_hidden_layers": 2,
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_local_experts": 4,
                "num_experts_per_tok": 2,
            }

        def named_modules(self):
            return iter((("", self), ("layers.0.gate", object())))

        def named_parameters(self):
            return iter((("layers.0.gate.weight", FakeParameter((4, 8))),))

    model = GenericModel()
    assert MixtralStaticAdapter().detect(model, model.config).score == 0.0


@pytest.mark.parametrize("variant", ["missing", "mixed", "multiple", "noncontiguous", "expert"])
def test_topology_negatives_are_zero(variant: str) -> None:
    model = MixtralForCausalLM(layout="legacy")
    modules = _module_entries(model)
    parameters = _parameter_entries(model)
    if variant == "missing":
        modules = [item for item in modules if not item[0].endswith(".experts")]
    elif variant == "mixed":
        modules.append(("layers.0.mlp", object()))
        modules.append(("layers.0.mlp.gate", object()))
        modules.append(("layers.0.mlp.experts", object()))
    elif variant == "multiple":
        modules.append(("other.layers.0.mlp", object()))
    elif variant == "noncontiguous":
        modules = [item for item in modules if not item[0].startswith("layers.1")]
        parameters = [item for item in parameters if not item[0].startswith("layers.1")]
    elif variant == "expert":
        modules = [item for item in modules if not item[0].endswith("experts.3")]
    model.named_modules = lambda: iter(modules)  # type: ignore[method-assign]
    model.named_parameters = lambda: iter(parameters)  # type: ignore[method-assign]

    assert MixtralStaticAdapter().detect(model, model.config).score == 0.0


def test_shape_mismatch_and_transformed_surface_are_zero() -> None:
    model = MixtralForCausalLM(layout="packed")
    parameters = _parameter_entries(model)
    parameters[0] = (parameters[0][0], FakeParameter((3, 8)))
    model.named_parameters = lambda: iter(parameters)  # type: ignore[method-assign]

    assert MixtralStaticAdapter().detect(model, model.config).score == 0.0


@pytest.mark.parametrize("surface", ["modules", "parameters"])
def test_partial_iteration_failure_is_safe_zero(surface: str) -> None:
    model = MixtralForCausalLM()
    original_modules = _module_entries(model)
    original_parameters = _parameter_entries(model)
    if surface == "modules":

        def failing_modules():
            yield from original_modules[:3]
            raise RuntimeError("TOP_SECRET_ITERATION")

        model.named_modules = failing_modules  # type: ignore[method-assign]
    else:

        def failing_parameters():
            yield from original_parameters[:1]
            raise RuntimeError("TOP_SECRET_ITERATION")

        model.named_parameters = failing_parameters  # type: ignore[method-assign]

    detection = MixtralStaticAdapter().detect(model, model.config)
    assert detection.score == 0.0
    assert "TOP_SECRET" not in " ".join(detection.warnings)
    assert any("RuntimeError" in warning for warning in detection.warnings)


def test_discover_unsupported_is_manifest_bound_and_fixed_warning() -> None:
    class Unsupported:
        config = object()

    report = MixtralStaticAdapter().discover(Unsupported(), _manifest())

    assert report.model_manifest == _manifest()
    assert report.model_key == _manifest().model_key
    assert report.components == []
    assert report.candidates == []
    assert report.warnings


def test_inspection_binds_manifest_and_preserves_read_only_boundary() -> None:
    model = MixtralForCausalLM()
    before_modules = _module_entries(model)
    before_parameters = _parameter_entries(model)
    manifest = _manifest()

    inspection = inspect_static_adapter(MixtralStaticAdapter(), model, model.config, manifest)

    assert inspection.report.model_manifest == manifest
    assert all(
        component.capabilities == [CapabilityLabel.STRUCTURE]
        for component in inspection.report.components
    )
    assert all(
        component.capture is not None
        and component.capture.source is CaptureSource.STATIC_STRUCTURE
        and component.capture.verified is False
        for component in inspection.report.components
    )
    assert _module_entries(model) == before_modules
    assert _parameter_entries(model) == before_parameters


def test_forbidden_runtime_actions_network_and_cache_are_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Guarded(MixtralForCausalLM):
        def forward(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("forward must not run")

        def generate(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("generate must not run")

        def register_forward_hook(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("hooks must not run")

        def parameters(self):
            raise AssertionError("tensor parameters must not be read")

    def fail_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("network must not run")

    monkeypatch.setattr(socket, "create_connection", fail_network)
    monkeypatch.setattr(request, "urlopen", fail_network)
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    monkeypatch.setenv("HF_HOME", str(cache_root / "hf"))
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(cache_root / "transformers"))
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    model = Guarded()
    assert MixtralStaticAdapter().detect(model, model.config).score > 0
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == before


def test_unrelated_layer_modules_do_not_change_exact_moe_layout_detection() -> None:
    model = MixtralForCausalLM()
    modules = [item for item in _module_entries(model) if item[0] not in {"layers.0", "layers.1"}]
    modules.extend(
        (
            ("layers.0", object()),
            ("layers.1", object()),
            ("layers.0.self_attn", object()),
            ("layers.1.input_layernorm", object()),
        )
    )
    model.named_modules = lambda: iter(modules)  # type: ignore[method-assign]

    assert MixtralStaticAdapter().detect(model, model.config).score > 0


@pytest.mark.parametrize(
    "malformed_path",
    [
        "layers.00.block_sparse_moe",
        "layers.0.block_sparse_moe.experts.00",
        "layers.0.block_sparse_moe.fused",
    ],
)
def test_noncanonical_or_transformed_layout_paths_are_zero(malformed_path: str) -> None:
    model = MixtralForCausalLM()
    modules = _module_entries(model)
    modules.append((malformed_path, object()))
    model.named_modules = lambda: iter(modules)  # type: ignore[method-assign]

    assert MixtralStaticAdapter().detect(model, model.config).score == 0.0


def test_extra_parameter_inside_moe_surface_is_zero() -> None:
    model = MixtralForCausalLM()
    parameters = _parameter_entries(model)
    parameters.append(
        (
            "layers.0.block_sparse_moe.experts.fused.weight",
            FakeParameter((16, 8)),
        )
    )
    model.named_parameters = lambda: iter(parameters)  # type: ignore[method-assign]

    assert MixtralStaticAdapter().detect(model, model.config).score == 0.0


@pytest.mark.parametrize("surface", ["modules", "parameters"])
@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit])
def test_control_flow_during_structural_iteration_propagates_unchanged(
    surface: str, exception_type: type[BaseException]
) -> None:
    model = MixtralForCausalLM()
    expected = exception_type("TOP_SECRET_CONTROL_FLOW")
    if surface == "modules":

        def failing_modules():
            raise expected

        model.named_modules = failing_modules  # type: ignore[method-assign]
    else:

        def failing_parameters():
            raise expected

        model.named_parameters = failing_parameters  # type: ignore[method-assign]

    with pytest.raises(exception_type) as caught:
        MixtralStaticAdapter().detect(model, model.config)
    assert caught.value is expected


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit])
def test_control_flow_from_model_config_propagates_unchanged(
    exception_type: type[BaseException],
) -> None:
    expected = exception_type("TOP_SECRET_CONFIG")

    class ConfigFailure:
        @property
        def config(self) -> object:
            raise expected

    with pytest.raises(exception_type) as caught:
        MixtralStaticAdapter().detect(ConfigFailure(), object())
    assert caught.value is expected


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit])
def test_control_flow_from_parameter_shape_propagates_unchanged(
    exception_type: type[BaseException],
) -> None:
    model = MixtralForCausalLM()
    expected = exception_type("TOP_SECRET_SHAPE")

    class ShapeFailure:
        @property
        def shape(self) -> tuple[int, ...]:
            raise expected

    model.named_parameters = lambda: iter(
        [("layers.0.block_sparse_moe.gate.weight", ShapeFailure())]
    )  # type: ignore[method-assign]

    with pytest.raises(exception_type) as caught:
        MixtralStaticAdapter().detect(model, model.config)
    assert caught.value is expected


def test_ordinary_structural_failures_are_safe_and_typed() -> None:
    config_exception = RuntimeError("TOP_SECRET_CONFIG_VALUE")

    class ConfigFailure:
        @property
        def config(self) -> object:
            raise config_exception

    config_detection = MixtralStaticAdapter().detect(ConfigFailure(), object())
    assert config_detection.score == 0.0
    assert "TOP_SECRET" not in " ".join(config_detection.warnings)
    assert any("RuntimeError" in warning for warning in config_detection.warnings)

    model = MixtralForCausalLM()
    shape_exception = RuntimeError("TOP_SECRET_SHAPE_VALUE")

    class ShapeFailure:
        @property
        def shape(self) -> tuple[int, ...]:
            raise shape_exception

    model.named_parameters = lambda: iter(
        [("layers.0.block_sparse_moe.gate.weight", ShapeFailure())]
    )  # type: ignore[method-assign]
    shape_detection = MixtralStaticAdapter().detect(model, model.config)
    assert shape_detection.score == 0.0
    assert "TOP_SECRET" not in " ".join(shape_detection.warnings)
    assert any("RuntimeError" in warning for warning in shape_detection.warnings)


def test_repository_has_no_model_or_checkpoint_artifacts() -> None:
    root = Path(__file__).resolve().parents[1]
    excluded_parts = {".git", ".venv", ".pytest_cache", "build", "dist"}
    prohibited_suffixes = {
        ".bin",
        ".ckpt",
        ".gguf",
        ".onnx",
        ".pt",
        ".pth",
        ".safetensors",
    }
    artifacts = [
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file()
        and not excluded_parts.intersection(path.parts)
        and path.suffix.casefold() in prohibited_suffixes
    ]
    assert artifacts == []
