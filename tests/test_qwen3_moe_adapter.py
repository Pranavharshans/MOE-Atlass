from __future__ import annotations

import dataclasses
import os
import socket
import sys
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

from moeatlas.adapters import Qwen3MoeStaticAdapter, inspect_static_adapter
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

from .fixtures import (
    FakeParameter,
    Qwen3MoeConfig,
    Qwen3MoeConfigMapping,
    Qwen3MoeForCausalLM,
)


def _manifest() -> ModelManifest:
    return ModelManifest(
        model_key=make_model_key("acme/qwen3-moe", "r1"),
        architecture="qwen3_moe",
        revision="r1",
        config_hash=make_config_hash({"experts": 4, "top_k": 2}),
        tokenizer=TokenizerIdentity(identifier="acme/tokenizer", revision="r1"),
        dtype=DType.FLOAT32,
        device_map={"": "cpu"},
    )


def _modules(model: Qwen3MoeForCausalLM) -> list[tuple[str, object]]:
    return list(model.named_modules())


def _parameters(model: Qwen3MoeForCausalLM) -> list[tuple[str, FakeParameter]]:
    return list(model.named_parameters())


def _replace_module(model: Qwen3MoeForCausalLM, path: str, value: object) -> None:
    model._modules = [
        (current_path, value if current_path == path else current_value)
        for current_path, current_value in model._modules
    ]


def _sparse_layers(config: Qwen3MoeConfig) -> tuple[int, ...]:
    only_dense = set(config.mlp_only_layers)
    return tuple(
        index
        for index in range(config.num_hidden_layers)
        if index not in only_dense and (index + 1) % config.decoder_sparse_step == 0
    )


class _ConfigWithout:
    def __init__(self, source: object, *omitted: str) -> None:
        self._source = source
        self._omitted = set(omitted)

    def __getattr__(self, name: str) -> object:
        if name in self._omitted:
            raise AttributeError(name)
        return getattr(self._source, name)


def test_descriptor_export_and_exact_notes() -> None:
    adapter = Qwen3MoeStaticAdapter()

    assert adapter.__class__.__slots__ == ()
    assert adapter.descriptor.name == "huggingface-qwen3-moe-static"
    assert adapter.descriptor.version == "1.0"
    assert adapter.descriptor.architecture_families == ("qwen3_moe",)
    assert adapter.descriptor.compatibility_notes == tuple(
        sorted(adapter.descriptor.compatibility_notes)
    )
    assert adapter.descriptor.compatibility_notes == (
        "official Transformers 4.51.3 and 4.57.1 indexed reference layouts are supported",
        "official Transformers 5.0.0 packed reference layout is supported as logical slices",
        "structure-only; routing certification is not provided",
    )
    assert "Qwen3MoeStaticAdapter" in sys.modules["moeatlas.adapters"].__all__


@pytest.mark.parametrize("layout", ["legacy_indexed", "packed"])
@pytest.mark.parametrize("prefix", ["", "model.transformer"])
def test_official_layouts_and_schedule_are_positive(layout: str, prefix: str) -> None:
    model = Qwen3MoeForCausalLM(layout=layout, prefix=prefix)
    detection = Qwen3MoeStaticAdapter().detect(model, model.config)

    assert detection.score == 1.0
    assert detection.evidence == (
        "architecture:qwen3-moe-allowlist",
        "config:strict-fields-and-schedule",
        "model_type:qwen3_moe",
        "shapes:exact",
        "topology:complete-layout",
    )
    assert detection.warnings == ()


def test_mapping_config_is_supported_but_identity_is_required() -> None:
    config = Qwen3MoeConfigMapping()
    model = Qwen3MoeForCausalLM(layout="packed", config=config)
    adapter = Qwen3MoeStaticAdapter()

    assert adapter.detect(model, config).score == 1.0
    assert adapter.detect(model, dict(config)).score == 0.0
    assert adapter.detect(model, dict(config)).evidence == ()


def test_detection_scores_single_family_markers() -> None:
    adapter = Qwen3MoeStaticAdapter()
    model_type_only = Qwen3MoeForCausalLM()
    model_type_only.config = _ConfigWithout(model_type_only.config, "architectures")
    architecture_only = Qwen3MoeForCausalLM()
    architecture_only.config = _ConfigWithout(architecture_only.config, "model_type")

    assert adapter.detect(model_type_only, model_type_only.config).score == 0.85
    assert adapter.detect(architecture_only, architecture_only.config).score == 0.75


def test_class_name_never_spoofs_absent_family_markers() -> None:
    model = Qwen3MoeForCausalLM()
    model.config = _ConfigWithout(model.config, "model_type", "architectures")

    detection = Qwen3MoeStaticAdapter().detect(model, model.config)

    assert detection.score == 0.0
    assert detection.evidence == ()
    assert detection.warnings


@pytest.mark.parametrize(
    "config_update",
    [
        {"model_type": "qwen2_moe"},
        {"model_type": "qwen3.5_moe"},
        {"model_type": "Qwen3Moe"},
        {"architectures": ["Qwen3MoeForCausalLM", "Qwen3MoeForCausalLM"]},
        {"architectures": ["Qwen2MoeForCausalLM"]},
        {"architectures": ["Qwen3MoeForCausalLM", "Qwen3MoeForCausalLMExtra"]},
        {"architectures": "Qwen3MoeForCausalLM"},
        {"architectures": [1]},
    ],
)
def test_family_marker_negatives_are_zero(config_update: dict[str, object]) -> None:
    config = dataclasses.replace(Qwen3MoeConfig(), **config_update)
    model = Qwen3MoeForCausalLM(config=config)
    detection = Qwen3MoeStaticAdapter().detect(model, config)

    assert detection.score == 0.0
    assert detection.evidence == ()
    assert detection.warnings
    assert all("TOP_SECRET" not in warning for warning in detection.warnings)


@pytest.mark.parametrize(
    "config_update",
    [
        {"num_hidden_layers": True},
        {"hidden_size": 8.0},
        {"intermediate_size": "16"},
        {"moe_intermediate_size": 0},
        {"num_experts": False},
        {"num_experts_per_tok": 5},
        {"decoder_sparse_step": 0},
        {"norm_topk_prob": 1},
        {"mlp_only_layers": (0,)},
        {"mlp_only_layers": [0, 0]},
        {"mlp_only_layers": [4]},
    ],
)
def test_strict_config_and_schedule_negatives_are_zero(
    config_update: dict[str, object],
) -> None:
    config = dataclasses.replace(Qwen3MoeConfig(), **config_update)
    if config_update.get("decoder_sparse_step") == 0:
        model = Qwen3MoeForCausalLM()
        model.config.decoder_sparse_step = 0
        config = model.config
    else:
        model = Qwen3MoeForCausalLM(config=config)

    detection = Qwen3MoeStaticAdapter().detect(model, config)

    assert detection.score == 0.0
    assert detection.evidence == ()
    assert detection.warnings


def test_no_sparse_layers_are_rejected() -> None:
    config = Qwen3MoeConfig(
        num_hidden_layers=2,
        decoder_sparse_step=3,
        mlp_only_layers=[],
    )
    model = Qwen3MoeForCausalLM(config=config)

    assert Qwen3MoeStaticAdapter().detect(model, config).score == 0.0


def test_reports_only_sparse_layers_with_exact_legacy_shapes_and_metadata() -> None:
    model = Qwen3MoeForCausalLM(layout="legacy_indexed")
    report = Qwen3MoeStaticAdapter().discover(model, _manifest())

    assert report.facts.expert_count == 4
    assert report.facts.expert_count_source == "config.num_experts"
    assert report.facts.routed_top_k == 2
    assert report.facts.routed_top_k_source == "config.num_experts_per_tok"
    assert report.facts.shared_expert_count is None
    assert len(report.candidates) == len(_sparse_layers(model.config)) * 7
    assert [candidate.kind for candidate in report.candidates[:7]] == [
        ComponentKind.MOE_LAYER,
        ComponentKind.ROUTER,
        ComponentKind.EXPERT_CONTAINER,
        ComponentKind.EXPERT,
        ComponentKind.EXPERT,
        ComponentKind.EXPERT,
        ComponentKind.EXPERT,
    ]
    assert report.warnings == []
    for candidate, component in zip(report.candidates, report.components, strict=True):
        assert candidate.component_key == component.component_key
        assert component.capabilities == [CapabilityLabel.STRUCTURE]
        assert component.capture is not None
        assert component.capture.source is CaptureSource.STATIC_STRUCTURE
        assert component.capture.method == "qwen3-moe-static-structure-v1"
        assert component.capture.adapter == "huggingface-qwen3-moe-static"
        assert component.capture.adapter_version == "1.0"
        assert component.capture.verified is False
        assert component.capture.metadata == {"layout": "legacy_indexed"}
        assert component.provenance is not None
        assert component.provenance.source == "qwen3-moe-static"
        assert component.provenance.metadata == {
            "layout": "legacy_indexed",
            "evidence": ["config", "topology", "shapes"],
        }
        assert candidate.evidence[0].detail == "Qwen3-MoE legacy_indexed layout"
    expert = report.components[3]
    assert expert.tensor_shapes == {
        "gate_proj.weight": [12, 8],
        "up_proj.weight": [12, 8],
        "down_proj.weight": [8, 12],
    }
    assert all(component.kind is not ComponentKind.MODULE for component in report.components)


def test_packed_report_uses_physical_container_and_logical_slices() -> None:
    model = Qwen3MoeForCausalLM(layout="packed", prefix="model")
    report = Qwen3MoeStaticAdapter().discover(model, _manifest())

    assert report.warnings == [
        "packed expert slices are logical and are not independently hookable"
    ]
    containers = [
        component
        for component in report.components
        if component.kind is ComponentKind.EXPERT_CONTAINER
    ]
    experts = [
        component for component in report.components if component.kind is ComponentKind.EXPERT
    ]
    assert len(containers) == len(_sparse_layers(model.config))
    assert len(experts) == len(_sparse_layers(model.config)) * 4
    assert containers[0].tensor_shapes == {
        "gate_up_proj": [4, 24, 8],
        "down_proj": [4, 8, 12],
    }
    assert experts[0].module_path == "model.layers.1.mlp.experts"
    assert experts[0].expert_index == 0
    assert experts[0].tensor_shapes == {
        "gate_up_proj": [24, 8],
        "down_proj": [8, 12],
    }
    assert experts[0].warnings == report.warnings
    assert all(component.capture is not None for component in experts)


def test_report_is_deterministic_fresh_and_json_roundtrippable() -> None:
    model = Qwen3MoeForCausalLM(layout="packed")
    adapter = Qwen3MoeStaticAdapter()
    first = adapter.discover(model, _manifest())
    second = adapter.discover(model, _manifest())

    assert first.to_json() == second.to_json()
    assert first is not second
    assert type(first).from_json(first.to_json()) == first
    assert first.model_manifest == _manifest()


@pytest.mark.parametrize(
    "layout, path",
    [
        ("legacy_indexed", "layers.1.mlp.experts.0.gate_proj.weight"),
        ("packed", "layers.1.mlp.experts.gate_up_proj"),
    ],
)
def test_exact_parameter_names_and_shapes_are_required(layout: str, path: str) -> None:
    model = Qwen3MoeForCausalLM(layout=layout)
    for index, (name, parameter) in enumerate(model._parameters):
        if name == path:
            model._parameters[index] = (name, FakeParameter((999, *parameter.shape[1:])))
            break
    else:
        raise AssertionError(path)

    assert Qwen3MoeStaticAdapter().detect(model, model.config).score == 0.0


def test_packed_weight_suffix_is_rejected() -> None:
    model = Qwen3MoeForCausalLM(layout="packed")
    model._parameters = [
        (f"{name}.weight" if name.endswith("gate_up_proj") else name, parameter)
        for name, parameter in model._parameters
    ]

    assert Qwen3MoeStaticAdapter().detect(model, model.config).score == 0.0


def test_dense_sparse_swap_and_mixed_sparse_layout_are_rejected() -> None:
    model = Qwen3MoeForCausalLM(layout="legacy_indexed")
    model._modules = [
        (path.replace("layers.0.mlp", "layers.0.mlp.bad"), value)
        if path.startswith("layers.0.mlp")
        else (path, value)
        for path, value in model._modules
    ]
    assert Qwen3MoeStaticAdapter().detect(model, model.config).score == 0.0

    mixed = Qwen3MoeForCausalLM(layout="legacy_indexed")
    for index, (path, value) in enumerate(mixed._modules):
        if path.startswith("layers.3.mlp"):
            mixed._modules[index] = (path.replace("layers.3.mlp", "layers.3.mlp"), value)
    mixed._modules = [
        item for item in mixed._modules if not item[0].startswith("layers.3.mlp.experts.")
    ]
    mixed._modules.extend(
        [
            ("layers.3.mlp.experts", object()),
            ("layers.3.mlp.experts.act_fn", object()),
        ]
    )
    assert Qwen3MoeStaticAdapter().detect(mixed, mixed.config).score == 0.0


@pytest.mark.parametrize(
    "path, attribute",
    [
        ("layers.1.mlp", "num_experts"),
        ("layers.1.mlp", "top_k"),
        ("layers.1.mlp", "norm_topk_prob"),
    ],
)
def test_legacy_structural_attributes_are_required(path: str, attribute: str) -> None:
    model = Qwen3MoeForCausalLM(layout="legacy_indexed")
    module = dict(_modules(model))[path]
    setattr(
        module,
        attribute,
        getattr(module, attribute) + 1 if attribute != "norm_topk_prob" else False,
    )

    assert Qwen3MoeStaticAdapter().detect(model, model.config).score == 0.0


def test_packed_structural_attributes_are_required() -> None:
    model = Qwen3MoeForCausalLM(layout="packed")
    modules = dict(_modules(model))
    modules["layers.1.mlp.gate"].hidden_dim = 99

    assert Qwen3MoeStaticAdapter().detect(model, model.config).score == 0.0


def test_extra_or_missing_module_inside_mlp_is_rejected() -> None:
    model = Qwen3MoeForCausalLM()
    model._modules.pop()
    assert Qwen3MoeStaticAdapter().detect(model, model.config).score == 0.0

    model = Qwen3MoeForCausalLM()
    model._modules.append(("layers.1.mlp.unexpected", object()))
    assert Qwen3MoeStaticAdapter().detect(model, model.config).score == 0.0


def test_conflicting_roots_and_noncontiguous_layers_are_rejected() -> None:
    model = Qwen3MoeForCausalLM()
    model._modules.append(("other.layers.0.mlp", object()))
    assert Qwen3MoeStaticAdapter().detect(model, model.config).score == 0.0

    model = Qwen3MoeForCausalLM()
    model._modules.append(("other.layers.0.self_attn", object()))
    assert Qwen3MoeStaticAdapter().detect(model, model.config).score == 0.0

    model = Qwen3MoeForCausalLM()
    model._modules = [
        (path.replace("layers.3", "layers.4"), value) for path, value in model._modules
    ]
    assert Qwen3MoeStaticAdapter().detect(model, model.config).score == 0.0


def test_alternate_parameter_layer_stack_prefix_is_rejected() -> None:
    model = Qwen3MoeForCausalLM()
    model._parameters.append(("other.layers.0.self_attn.q_proj.weight", FakeParameter((8, 8))))
    assert Qwen3MoeStaticAdapter().detect(model, model.config).score == 0.0


def test_unrelated_siblings_do_not_change_validity() -> None:
    model = Qwen3MoeForCausalLM()
    model._modules.extend(
        [("layers.0.self_attn", object()), ("layers.0.self_attn.q_proj", object())]
    )
    model._parameters.append(("layers.0.self_attn.q_proj.weight", FakeParameter((8, 8))))

    assert Qwen3MoeStaticAdapter().detect(model, model.config).score == 1.0


def test_inspection_revalidates_exact_static_report() -> None:
    model = Qwen3MoeForCausalLM()
    inspection = inspect_static_adapter(
        Qwen3MoeStaticAdapter(),
        model,
        model.config,
        _manifest(),
    )

    assert inspection.detection.score == 1.0
    assert inspection.report.model_manifest == _manifest()
    assert all(
        component.capabilities == [CapabilityLabel.STRUCTURE]
        and component.capture is not None
        and component.capture.source is CaptureSource.STATIC_STRUCTURE
        and component.capture.verified is False
        for component in inspection.report.components
    )


def test_unsupported_discovery_is_manifest_bound_and_safe() -> None:
    model = Qwen3MoeForCausalLM()
    model.config.model_type = "qwen2"
    report = Qwen3MoeStaticAdapter().discover(model, _manifest())

    assert report.model_key == _manifest().model_key
    assert report.model_manifest == _manifest()
    assert report.components == []
    assert report.warnings
    assert "qwen2" not in " ".join(report.warnings).lower()


class _OrdinaryFailureModule:
    @property
    def num_experts(self) -> int:
        raise ValueError("TOP_SECRET_ATTRIBUTE")


class _KeyboardFailureModule:
    @property
    def num_experts(self) -> int:
        raise KeyboardInterrupt


class _SystemExitFailureModule:
    @property
    def num_experts(self) -> int:
        raise SystemExit(7)


@pytest.mark.parametrize("failure_type", [KeyboardInterrupt, SystemExit])
def test_control_flow_from_structural_attributes_is_not_wrapped(
    failure_type: type[BaseException],
) -> None:
    model = Qwen3MoeForCausalLM()
    path = "layers.1.mlp"
    value: object = (
        _KeyboardFailureModule()
        if failure_type is KeyboardInterrupt
        else _SystemExitFailureModule()
    )
    _replace_module(model, path, value)

    with pytest.raises(failure_type):
        Qwen3MoeStaticAdapter().detect(model, model.config)


def test_ordinary_attribute_errors_are_type_only() -> None:
    model = Qwen3MoeForCausalLM()
    _replace_module(model, "layers.1.mlp", _OrdinaryFailureModule())

    detection = Qwen3MoeStaticAdapter().detect(model, model.config)

    assert detection.score == 0.0
    assert "TOP_SECRET_ATTRIBUTE" not in " ".join(detection.warnings)
    assert any("ValueError" in warning for warning in detection.warnings)


class _IteratorModel(Qwen3MoeForCausalLM):
    def __init__(self, *, mode: str) -> None:
        super().__init__()
        self.mode = mode

    def named_modules(self) -> Iterator[tuple[str, object]]:
        if self.mode == "keyboard":
            raise KeyboardInterrupt
        if self.mode == "system":
            raise SystemExit(8)
        if self.mode == "ordinary":
            raise ValueError("TOP_SECRET_ITERATION")
        return super().named_modules()


@pytest.mark.parametrize("mode, error", [("keyboard", KeyboardInterrupt), ("system", SystemExit)])
def test_control_flow_from_surface_methods_is_not_wrapped(
    mode: str, error: type[BaseException]
) -> None:
    model = _IteratorModel(mode=mode)
    with pytest.raises(error):
        Qwen3MoeStaticAdapter().detect(model, model.config)


def test_ordinary_surface_errors_are_redacted() -> None:
    model = _IteratorModel(mode="ordinary")
    detection = Qwen3MoeStaticAdapter().detect(model, model.config)
    assert detection.score == 0.0
    assert "TOP_SECRET_ITERATION" not in " ".join(detection.warnings)


class _ShapeFailure:
    @property
    def shape(self) -> tuple[int, ...]:
        raise KeyboardInterrupt


class _ShapeSystemExit:
    @property
    def shape(self) -> tuple[int, ...]:
        raise SystemExit(9)


@pytest.mark.parametrize(
    "value, error", [(_ShapeFailure(), KeyboardInterrupt), (_ShapeSystemExit(), SystemExit)]
)
def test_control_flow_from_parameter_shapes_is_not_wrapped(
    value: object, error: type[BaseException]
) -> None:
    model = Qwen3MoeForCausalLM()
    model._parameters[0] = (model._parameters[0][0], value)  # type: ignore[assignment]
    with pytest.raises(error):
        Qwen3MoeStaticAdapter().detect(model, model.config)


def test_static_adapter_does_not_import_model_runtime_or_mutate_surface() -> None:
    model = Qwen3MoeForCausalLM(layout="packed")
    before_modules = list(model._modules)
    before_parameters = list(model._parameters)
    before_config = dataclasses.asdict(model.config)
    adapter = Qwen3MoeStaticAdapter()

    adapter.detect(model, model.config)
    adapter.discover(model, _manifest())

    assert model._modules == before_modules
    assert model._parameters == before_parameters
    assert dataclasses.asdict(model.config) == before_config
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules
    assert "safetensors" not in sys.modules


def _module_map(model: Qwen3MoeForCausalLM) -> dict[str, object]:
    return dict(_modules(model))


def _parameter_map(model: Qwen3MoeForCausalLM) -> dict[str, FakeParameter]:
    return dict(_parameters(model))


def _remove_module_prefix(model: Qwen3MoeForCausalLM, prefix: str) -> None:
    model._modules = [
        (path, value)
        for path, value in model._modules
        if path != prefix and not path.startswith(f"{prefix}.")
    ]


def _remove_parameter_prefix(model: Qwen3MoeForCausalLM, prefix: str) -> None:
    model._parameters = [
        (name, value)
        for name, value in model._parameters
        if name != prefix and not name.startswith(f"{prefix}.")
    ]


def _copy_layer_surface(
    target: Qwen3MoeForCausalLM,
    source: Qwen3MoeForCausalLM,
    *,
    target_layer: int,
    source_layer: int,
) -> None:
    target_prefix = f"{target.prefix + '.' if target.prefix else ''}layers.{target_layer}.mlp"
    source_prefix = f"{source.prefix + '.' if source.prefix else ''}layers.{source_layer}.mlp"
    _remove_module_prefix(target, target_prefix)
    _remove_parameter_prefix(target, target_prefix)
    for path, value in source._modules:
        if path == source_prefix or path.startswith(f"{source_prefix}."):
            target._modules.append((target_prefix + path[len(source_prefix) :], value))
    for name, value in source._parameters:
        if name == source_prefix or name.startswith(f"{source_prefix}."):
            target._parameters.append((target_prefix + name[len(source_prefix) :], value))


def _tree_snapshot(root: Path) -> tuple[str, ...]:
    if not root.exists():
        return ()
    return tuple(sorted(str(path.relative_to(root)) for path in root.rglob("*")))


class _AttributeFailure:
    def __init__(self, attribute: str, error: BaseException) -> None:
        self._attribute = attribute
        self._error = error

    def __getattribute__(self, name: str) -> object:
        if name not in {"_attribute", "_error", "__dict__", "__class__"}:
            attribute = object.__getattribute__(self, "_attribute")
            if name == attribute:
                raise object.__getattribute__(self, "_error")
        return object.__getattribute__(self, name)


class _ConfigFailure:
    def __init__(self, source: object, field_name: str, error: BaseException) -> None:
        self._source = source
        self._field_name = field_name
        self._error = error

    def __getattribute__(self, name: str) -> object:
        if name in {"_source", "_field_name", "_error", "__dict__", "__class__"}:
            return object.__getattribute__(self, name)
        field_name = object.__getattribute__(self, "_field_name")
        if name == field_name:
            raise object.__getattribute__(self, "_error")
        return getattr(object.__getattribute__(self, "_source"), name)


class _ConfigAccessFailureModel:
    def __init__(self, source: Qwen3MoeForCausalLM, error: BaseException) -> None:
        self._source = source
        self._error = error

    @property
    def config(self) -> object:
        raise self._error

    def named_modules(self) -> Iterator[tuple[str, object]]:
        return self._source.named_modules()

    def named_parameters(self) -> Iterator[tuple[str, object]]:
        return self._source.named_parameters()


class _SurfaceFailureModel:
    def __init__(
        self,
        source: Qwen3MoeForCausalLM,
        *,
        failing_surface: str,
        mode: str,
        error: BaseException | None = None,
        entries: list[object] | None = None,
    ) -> None:
        self.config = source.config
        self._source = source
        self._failing_surface = failing_surface
        self._mode = mode
        self._error = error
        self._entries = entries

    def _surface(self, name: str) -> Iterator[tuple[str, object]]:
        if name != self._failing_surface:
            return getattr(self._source, name)()
        if self._mode == "method":
            raise self._error  # type: ignore[misc]
        if self._entries is not None:
            return iter(self._entries)  # type: ignore[return-value]
        return _FailingIterator(
            getattr(self._source, name)(),
            self._error,  # type: ignore[arg-type]
        )

    def named_modules(self) -> Iterator[tuple[str, object]]:
        return self._surface("named_modules")

    def named_parameters(self) -> Iterator[tuple[str, object]]:
        return self._surface("named_parameters")


class _FailingIterator:
    def __init__(self, source: Iterator[tuple[str, object]], error: BaseException) -> None:
        self._source = iter(source)
        self._error = error
        self._count = 0

    def __iter__(self) -> _FailingIterator:
        return self

    def __next__(self) -> tuple[str, object]:
        if self._count == 1:
            raise self._error
        self._count += 1
        return next(self._source)


class _MalformedThenFailureIterator:
    def __init__(self, error: BaseException) -> None:
        self._error = error
        self._first = True

    def __iter__(self) -> _MalformedThenFailureIterator:
        return self

    def __next__(self) -> object:
        if self._first:
            self._first = False
            return ("malformed",)
        raise self._error


@pytest.mark.parametrize(
    "layout, path, attribute, expected",
    [
        ("legacy_indexed", "layers.2.mlp", "hidden_size", 8),
        ("legacy_indexed", "layers.2.mlp", "intermediate_size", 16),
        ("legacy_indexed", "layers.1.mlp", "num_experts", 4),
        ("legacy_indexed", "layers.1.mlp", "top_k", 2),
        ("legacy_indexed", "layers.1.mlp", "norm_topk_prob", True),
        ("legacy_indexed", "layers.3.mlp.experts.3", "hidden_size", 8),
        ("legacy_indexed", "layers.3.mlp.experts.3", "intermediate_size", 12),
        ("packed", "layers.1.mlp.gate", "top_k", 2),
        ("packed", "layers.1.mlp.gate", "num_experts", 4),
        ("packed", "layers.1.mlp.gate", "norm_topk_prob", True),
        ("packed", "layers.1.mlp.gate", "hidden_dim", 8),
        ("packed", "layers.3.mlp.experts", "num_experts", 4),
        ("packed", "layers.3.mlp.experts", "hidden_dim", 8),
        ("packed", "layers.3.mlp.experts", "intermediate_dim", 12),
    ],
)
@pytest.mark.parametrize("mutation", ["missing", "wrong_type", "mismatch"])
def test_every_structural_attribute_is_strict(
    layout: str,
    path: str,
    attribute: str,
    expected: object,
    mutation: str,
) -> None:
    model = Qwen3MoeForCausalLM(layout=layout)
    module = _module_map(model)[path]
    if mutation == "missing":
        delattr(module, attribute)
    elif mutation == "wrong_type":
        setattr(module, attribute, "TOP_SECRET_ATTRIBUTE")
    elif isinstance(expected, bool):
        setattr(module, attribute, not expected)
    else:
        setattr(module, attribute, int(expected) + 1)

    detection = Qwen3MoeStaticAdapter().detect(model, model.config)

    assert detection.score == 0.0
    assert "TOP_SECRET_ATTRIBUTE" not in " ".join(detection.warnings)


@pytest.mark.parametrize(
    "layout, path, attribute",
    [
        ("legacy_indexed", "layers.2.mlp", "hidden_size"),
        ("legacy_indexed", "layers.2.mlp", "intermediate_size"),
        ("legacy_indexed", "layers.1.mlp", "num_experts"),
        ("legacy_indexed", "layers.1.mlp", "top_k"),
        ("legacy_indexed", "layers.1.mlp", "norm_topk_prob"),
        ("legacy_indexed", "layers.3.mlp.experts.3", "hidden_size"),
        ("legacy_indexed", "layers.3.mlp.experts.3", "intermediate_size"),
        ("packed", "layers.1.mlp.gate", "top_k"),
        ("packed", "layers.1.mlp.gate", "num_experts"),
        ("packed", "layers.1.mlp.gate", "norm_topk_prob"),
        ("packed", "layers.1.mlp.gate", "hidden_dim"),
        ("packed", "layers.3.mlp.experts", "num_experts"),
        ("packed", "layers.3.mlp.experts", "hidden_dim"),
        ("packed", "layers.3.mlp.experts", "intermediate_dim"),
    ],
)
@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
def test_every_structural_attribute_preserves_control_flow_identity(
    layout: str,
    path: str,
    attribute: str,
    error_type: type[BaseException],
) -> None:
    model = Qwen3MoeForCausalLM(layout=layout)
    error = error_type("QWEN3_ATTRIBUTE_BOUNDARY")
    _replace_module(model, path, _AttributeFailure(attribute, error))

    with pytest.raises(error_type) as raised:
        Qwen3MoeStaticAdapter().detect(model, model.config)
    assert raised.value is error


@pytest.mark.parametrize(
    "layout, path, attribute",
    [
        ("legacy_indexed", "layers.2.mlp", "hidden_size"),
        ("legacy_indexed", "layers.2.mlp", "intermediate_size"),
        ("legacy_indexed", "layers.1.mlp", "num_experts"),
        ("legacy_indexed", "layers.1.mlp", "top_k"),
        ("legacy_indexed", "layers.1.mlp", "norm_topk_prob"),
        ("legacy_indexed", "layers.3.mlp.experts.3", "hidden_size"),
        ("legacy_indexed", "layers.3.mlp.experts.3", "intermediate_size"),
        ("packed", "layers.1.mlp.gate", "top_k"),
        ("packed", "layers.1.mlp.gate", "num_experts"),
        ("packed", "layers.1.mlp.gate", "norm_topk_prob"),
        ("packed", "layers.1.mlp.gate", "hidden_dim"),
        ("packed", "layers.3.mlp.experts", "num_experts"),
        ("packed", "layers.3.mlp.experts", "hidden_dim"),
        ("packed", "layers.3.mlp.experts", "intermediate_dim"),
    ],
)
def test_every_structural_attribute_redacts_ordinary_exceptions(
    layout: str, path: str, attribute: str
) -> None:
    model = Qwen3MoeForCausalLM(layout=layout)
    _replace_module(
        model,
        path,
        _AttributeFailure(attribute, ValueError("TOP_SECRET_STRUCTURAL")),
    )
    detection = Qwen3MoeStaticAdapter().detect(model, model.config)
    assert detection.score == 0.0
    assert "TOP_SECRET_STRUCTURAL" not in " ".join(detection.warnings)


@pytest.mark.parametrize(
    "layout, parameter_name",
    [
        ("legacy_indexed", "layers.0.mlp.gate_proj.weight"),
        ("legacy_indexed", "layers.0.mlp.up_proj.weight"),
        ("legacy_indexed", "layers.0.mlp.down_proj.weight"),
        ("legacy_indexed", "layers.1.mlp.gate.weight"),
        ("legacy_indexed", "layers.1.mlp.experts.3.gate_proj.weight"),
        ("legacy_indexed", "layers.1.mlp.experts.3.up_proj.weight"),
        ("legacy_indexed", "layers.1.mlp.experts.3.down_proj.weight"),
        ("packed", "layers.1.mlp.gate.weight"),
        ("packed", "layers.1.mlp.experts.gate_up_proj"),
        ("packed", "layers.1.mlp.experts.down_proj"),
    ],
)
@pytest.mark.parametrize("mutation", ["wrong", "missing", "extra"])
def test_every_layout_shape_category_rejects_wrong_missing_and_extra(
    layout: str, parameter_name: str, mutation: str
) -> None:
    model = Qwen3MoeForCausalLM(layout=layout)
    target = parameter_name
    if mutation == "wrong":
        parameters = _parameter_map(model)
        original = parameters[target]
        index = next(index for index, (name, _) in enumerate(model._parameters) if name == target)
        model._parameters[index] = (
            target,
            FakeParameter((original.shape[0] + 1, *original.shape[1:])),
        )
    elif mutation == "missing":
        model._parameters = [
            (name, parameter) for name, parameter in model._parameters if name != target
        ]
    else:
        if target.endswith(".weight"):
            extra = f"{target.rsplit('.', 1)[0]}.bias"
        else:
            extra = f"{target}.weight"
        model._parameters.append((extra, FakeParameter((1,))))

    assert Qwen3MoeStaticAdapter().detect(model, model.config).score == 0.0


@pytest.mark.parametrize(
    "layout, extra",
    [
        ("legacy_indexed", "layers.1.mlp.experts.gate_up_proj"),
        ("legacy_indexed", "layers.1.mlp.gate.bias"),
        ("packed", "layers.1.mlp.experts.gate_up_proj.weight"),
        ("packed", "layers.1.mlp.experts.down_proj.weight"),
    ],
)
def test_legacy_and_packed_parameter_names_cannot_be_confused(layout: str, extra: str) -> None:
    model = Qwen3MoeForCausalLM(layout=layout)
    model._parameters.append((extra, FakeParameter((1,))))
    assert Qwen3MoeStaticAdapter().detect(model, model.config).score == 0.0


@pytest.mark.parametrize("layout", ["legacy_indexed", "packed"])
def test_official_fixture_surface_is_exact_and_complete(layout: str) -> None:
    model = Qwen3MoeForCausalLM(layout=layout, prefix="arbitrary.backbone")
    modules = {path for path, _ in model._modules}
    parameters = {name for name, _ in model._parameters}
    config = model.config
    root = "arbitrary.backbone.layers"
    expected_modules = {root}
    expected_parameters: set[str] = set()
    for index in range(config.num_hidden_layers):
        layer = f"{root}.{index}"
        expected_modules.add(layer)
        mlp = f"{layer}.mlp"
        if index in _sparse_layers(config):
            expected_modules.update({mlp, f"{mlp}.gate", f"{mlp}.experts"})
            if layout == "legacy_indexed":
                for expert in range(config.num_experts):
                    expert_path = f"{mlp}.experts.{expert}"
                    expected_modules.update(
                        {
                            expert_path,
                            f"{expert_path}.gate_proj",
                            f"{expert_path}.up_proj",
                            f"{expert_path}.down_proj",
                            f"{expert_path}.act_fn",
                        }
                    )
                expected_parameters.add(f"{mlp}.gate.weight")
                for expert in range(config.num_experts):
                    expert_path = f"{mlp}.experts.{expert}"
                    expected_parameters.update(
                        {
                            f"{expert_path}.gate_proj.weight",
                            f"{expert_path}.up_proj.weight",
                            f"{expert_path}.down_proj.weight",
                        }
                    )
            else:
                expected_modules.add(f"{mlp}.experts.act_fn")
                expected_parameters.update(
                    {
                        f"{mlp}.gate.weight",
                        f"{mlp}.experts.gate_up_proj",
                        f"{mlp}.experts.down_proj",
                    }
                )
        else:
            expected_modules.update(
                {
                    mlp,
                    f"{mlp}.gate_proj",
                    f"{mlp}.up_proj",
                    f"{mlp}.down_proj",
                    f"{mlp}.act_fn",
                }
            )
            expected_parameters.update(
                {
                    f"{mlp}.gate_proj.weight",
                    f"{mlp}.up_proj.weight",
                    f"{mlp}.down_proj.weight",
                }
            )
    assert modules == expected_modules
    assert parameters == expected_parameters
    assert Qwen3MoeStaticAdapter().detect(model, config).score == 1.0


@pytest.mark.parametrize("layout", ["legacy_indexed", "packed"])
def test_all_sparse_schedule_and_order_are_valid(layout: str) -> None:
    config = Qwen3MoeConfig(num_hidden_layers=4, decoder_sparse_step=1)
    model = Qwen3MoeForCausalLM(layout=layout, config=config)
    report = Qwen3MoeStaticAdapter().discover(model, _manifest())
    layer_indices = [candidate.layer_index for candidate in report.candidates]
    assert layer_indices == sorted(layer_indices)
    assert [candidate.kind for candidate in report.candidates[:4]] == [
        ComponentKind.MOE_LAYER,
        ComponentKind.ROUTER,
        ComponentKind.EXPERT_CONTAINER,
        ComponentKind.EXPERT,
    ]
    assert [
        candidate.expert_index
        for candidate in report.candidates
        if candidate.kind is ComponentKind.EXPERT
    ][:4] == [0, 1, 2, 3]
    assert len(report.candidates) == 4 * 7


def test_true_schedule_swaps_are_rejected_after_complete_surface_validation() -> None:
    model = Qwen3MoeForCausalLM(layout="legacy_indexed")
    all_sparse = Qwen3MoeForCausalLM(
        layout="legacy_indexed",
        config=Qwen3MoeConfig(num_hidden_layers=4, decoder_sparse_step=1),
    )
    _copy_layer_surface(model, all_sparse, target_layer=0, source_layer=0)
    assert Qwen3MoeStaticAdapter().detect(model, model.config).score == 0.0

    model = Qwen3MoeForCausalLM(layout="legacy_indexed")
    dense_source = Qwen3MoeForCausalLM(layout="legacy_indexed")
    _copy_layer_surface(model, dense_source, target_layer=1, source_layer=0)
    assert Qwen3MoeStaticAdapter().detect(model, model.config).score == 0.0


def test_complete_mixed_legacy_and_packed_sparse_layers_are_rejected() -> None:
    model = Qwen3MoeForCausalLM(layout="legacy_indexed")
    packed = Qwen3MoeForCausalLM(layout="packed")
    _copy_layer_surface(model, packed, target_layer=3, source_layer=1)
    assert Qwen3MoeStaticAdapter().detect(model, model.config).score == 0.0


@pytest.mark.parametrize(
    "field_name",
    [
        "model_type",
        "architectures",
        "num_hidden_layers",
        "hidden_size",
        "intermediate_size",
        "moe_intermediate_size",
        "num_experts",
        "num_experts_per_tok",
        "decoder_sparse_step",
        "norm_topk_prob",
        "mlp_only_layers",
    ],
)
def test_every_config_boundary_redacts_ordinary_errors(field_name: str) -> None:
    model = Qwen3MoeForCausalLM()
    config = _ConfigFailure(model.config, field_name, ValueError("TOP_SECRET_CONFIG"))
    model.config = config

    detection = Qwen3MoeStaticAdapter().detect(model, config)

    assert detection.score == 0.0
    assert "TOP_SECRET_CONFIG" not in " ".join(detection.warnings)
    assert all("Qwen3MoeConfig" not in warning for warning in detection.warnings)


@pytest.mark.parametrize(
    "field_name",
    [
        "model_type",
        "architectures",
        "num_hidden_layers",
        "hidden_size",
        "intermediate_size",
        "moe_intermediate_size",
        "num_experts",
        "num_experts_per_tok",
        "decoder_sparse_step",
        "norm_topk_prob",
        "mlp_only_layers",
    ],
)
@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
def test_every_config_boundary_preserves_control_flow_identity(
    field_name: str, error_type: type[BaseException]
) -> None:
    model = Qwen3MoeForCausalLM()
    error = error_type("QWEN3_CONFIG_BOUNDARY")
    config = _ConfigFailure(model.config, field_name, error)
    model.config = config

    with pytest.raises(error_type) as raised:
        Qwen3MoeStaticAdapter().detect(model, config)
    assert raised.value is error


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
def test_model_config_access_preserves_control_flow_identity(
    error_type: type[BaseException],
) -> None:
    source = Qwen3MoeForCausalLM()
    error = error_type("QWEN3_MODEL_CONFIG_BOUNDARY")

    with pytest.raises(error_type) as raised:
        Qwen3MoeStaticAdapter().detect(_ConfigAccessFailureModel(source, error), object())
    assert raised.value is error


def test_ordinary_model_config_access_is_safe_and_redacted() -> None:
    source = Qwen3MoeForCausalLM()
    detection = Qwen3MoeStaticAdapter().detect(
        _ConfigAccessFailureModel(source, ValueError("TOP_SECRET_MODEL_CONFIG")), object()
    )
    assert detection.score == 0.0
    assert "TOP_SECRET_MODEL_CONFIG" not in " ".join(detection.warnings)


@pytest.mark.parametrize("surface", ["named_modules", "named_parameters"])
@pytest.mark.parametrize("mode", ["method", "iterator"])
def test_surface_malformed_and_duplicate_entries_are_rejected(surface: str, mode: str) -> None:
    source = Qwen3MoeForCausalLM()
    if mode == "method":
        entries = (
            [("malformed",)]
            if surface == "named_modules"
            else [("same", object()), ("same", object())]
        )
    else:
        entries = (
            [("", source)]
            if surface == "named_modules"
            else [("same", object()), ("same", object())]
        )
    model = _SurfaceFailureModel(
        source,
        failing_surface=surface,
        mode="entries",
        entries=entries,
    )
    assert Qwen3MoeStaticAdapter().detect(model, model.config).score == 0.0


def test_duplicate_named_module_paths_are_rejected() -> None:
    source = Qwen3MoeForCausalLM()
    model = _SurfaceFailureModel(
        source,
        failing_surface="named_modules",
        mode="entries",
        entries=[("", source), ("", source)],
    )
    detection = Qwen3MoeStaticAdapter().detect(model, model.config)
    assert detection.score == 0.0


def test_malformed_named_parameter_entries_are_rejected() -> None:
    source = Qwen3MoeForCausalLM()
    model = _SurfaceFailureModel(
        source,
        failing_surface="named_parameters",
        mode="entries",
        entries=[("malformed",)],
    )
    detection = Qwen3MoeStaticAdapter().detect(model, model.config)
    assert detection.score == 0.0


@pytest.mark.parametrize("surface", ["named_modules", "named_parameters"])
def test_malformed_surface_entry_stops_before_later_iterator_failure(surface: str) -> None:
    source = Qwen3MoeForCausalLM()
    model = _SurfaceFailureModel(
        source,
        failing_surface=surface,
        mode="entries",
        entries=[],
    )
    error = KeyboardInterrupt("must not be reached")
    iterator = _MalformedThenFailureIterator(error)
    if surface == "named_modules":
        model.named_modules = lambda: iterator  # type: ignore[method-assign,assignment]
    else:
        model.named_parameters = lambda: iterator  # type: ignore[method-assign,assignment]
    assert Qwen3MoeStaticAdapter().detect(model, model.config).score == 0.0


@pytest.mark.parametrize("surface", ["named_modules", "named_parameters"])
@pytest.mark.parametrize("mode", ["method", "iterator"])
def test_surface_ordinary_failures_are_type_only(surface: str, mode: str) -> None:
    source = Qwen3MoeForCausalLM()
    error = ValueError("TOP_SECRET_SURFACE")
    model = _SurfaceFailureModel(
        source,
        failing_surface=surface,
        mode=mode,
        error=error,
    )
    detection = Qwen3MoeStaticAdapter().detect(model, model.config)
    assert detection.score == 0.0
    assert "TOP_SECRET_SURFACE" not in " ".join(detection.warnings)


@pytest.mark.parametrize("surface", ["named_modules", "named_parameters"])
@pytest.mark.parametrize("mode", ["method", "iterator"])
@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
def test_surface_control_flow_preserves_exact_identity(
    surface: str, mode: str, error_type: type[BaseException]
) -> None:
    source = Qwen3MoeForCausalLM()
    error = error_type("QWEN3_SURFACE_BOUNDARY")
    model = _SurfaceFailureModel(
        source,
        failing_surface=surface,
        mode=mode,
        error=error,
    )
    with pytest.raises(error_type) as raised:
        Qwen3MoeStaticAdapter().detect(model, model.config)
    assert raised.value is error


def test_malformed_modules_fail_fast_before_named_parameters() -> None:
    source = Qwen3MoeForCausalLM()
    calls: list[str] = []
    model = _SurfaceFailureModel(
        source,
        failing_surface="named_modules",
        mode="entries",
        entries=[("malformed",)],
    )

    def forbidden_parameters() -> Iterator[tuple[str, object]]:
        calls.append("parameters")
        raise AssertionError("named_parameters must not be called")

    model.named_parameters = forbidden_parameters  # type: ignore[method-assign]
    assert Qwen3MoeStaticAdapter().detect(model, model.config).score == 0.0
    assert calls == []


@pytest.mark.parametrize("field_name", ["model_type", "num_hidden_layers", "norm_topk_prob"])
def test_invalid_config_fails_fast_before_any_surface_iteration(field_name: str) -> None:
    source = Qwen3MoeForCausalLM()
    config = _ConfigFailure(source.config, field_name, ValueError("TOP_SECRET_CONFIG"))
    calls: list[str] = []
    source.config = config

    def forbidden_modules() -> Iterator[tuple[str, object]]:
        calls.append("modules")
        raise AssertionError("named_modules must not be called")

    def forbidden_parameters() -> Iterator[tuple[str, object]]:
        calls.append("parameters")
        raise AssertionError("named_parameters must not be called")

    source.named_modules = forbidden_modules  # type: ignore[method-assign]
    source.named_parameters = forbidden_parameters  # type: ignore[method-assign]
    assert Qwen3MoeStaticAdapter().detect(source, config).score == 0.0
    assert calls == []


class _ShapeBoundaryParameter:
    def __init__(self, error: BaseException) -> None:
        self._error = error

    @property
    def shape(self) -> tuple[int, ...]:
        raise self._error


@pytest.mark.parametrize(
    "layout, parameter_name",
    [
        ("legacy_indexed", "layers.0.mlp.gate_proj.weight"),
        ("legacy_indexed", "layers.0.mlp.up_proj.weight"),
        ("legacy_indexed", "layers.0.mlp.down_proj.weight"),
        ("legacy_indexed", "layers.1.mlp.gate.weight"),
        ("legacy_indexed", "layers.1.mlp.experts.3.gate_proj.weight"),
        ("legacy_indexed", "layers.1.mlp.experts.3.up_proj.weight"),
        ("legacy_indexed", "layers.1.mlp.experts.3.down_proj.weight"),
        ("packed", "layers.1.mlp.gate.weight"),
        ("packed", "layers.1.mlp.experts.gate_up_proj"),
        ("packed", "layers.1.mlp.experts.down_proj"),
    ],
)
@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
def test_every_shape_category_preserves_control_flow_identity(
    layout: str, parameter_name: str, error_type: type[BaseException]
) -> None:
    model = Qwen3MoeForCausalLM(layout=layout)
    error = error_type("QWEN3_SHAPE_BOUNDARY")
    index = next(
        index for index, (name, _) in enumerate(model._parameters) if name == parameter_name
    )
    model._parameters[index] = (parameter_name, _ShapeBoundaryParameter(error))  # type: ignore[assignment]

    with pytest.raises(error_type) as raised:
        Qwen3MoeStaticAdapter().detect(model, model.config)
    assert raised.value is error


def test_shape_failure_is_redacted() -> None:
    model = Qwen3MoeForCausalLM()
    index = next(
        index
        for index, (name, _) in enumerate(model._parameters)
        if name == "layers.1.mlp.gate.weight"
    )
    model._parameters[index] = (
        "layers.1.mlp.gate.weight",
        _ShapeBoundaryParameter(ValueError("TOP_SECRET_SHAPE")),
    )  # type: ignore[assignment]
    detection = Qwen3MoeStaticAdapter().detect(model, model.config)
    assert detection.score == 0.0
    assert "TOP_SECRET_SHAPE" not in " ".join(detection.warnings)


@pytest.mark.parametrize("bad_root", ["model..layers", ".model.layers", "model.layers.."])
def test_empty_components_in_module_paths_are_rejected(bad_root: str) -> None:
    model = Qwen3MoeForCausalLM(prefix="model")
    model._modules = [
        (path.replace("model.layers", bad_root), value) for path, value in model._modules
    ]
    assert Qwen3MoeStaticAdapter().detect(model, model.config).score == 0.0


@pytest.mark.parametrize("bad_root", ["model..layers", ".model.layers", "model.layers.."])
def test_empty_components_in_parameter_paths_are_rejected(bad_root: str) -> None:
    model = Qwen3MoeForCausalLM(prefix="model")
    model._parameters = [
        (name.replace("model.layers", bad_root), value) for name, value in model._parameters
    ]
    assert Qwen3MoeStaticAdapter().detect(model, model.config).score == 0.0


def test_out_of_range_module_siblings_and_parameter_roots_are_rejected() -> None:
    model = Qwen3MoeForCausalLM()
    model._modules.append(("layers.99.self_attn", object()))
    assert Qwen3MoeStaticAdapter().detect(model, model.config).score == 0.0

    model = Qwen3MoeForCausalLM()
    model._parameters.append(("layers.99.mlp.gate_proj.weight", FakeParameter((16, 8))))
    assert Qwen3MoeStaticAdapter().detect(model, model.config).score == 0.0


@pytest.mark.parametrize(
    "malformed_name",
    ["lm_head..weight", "lm_head.weight.", "lm_head...weight"],
)
def test_malformed_unrelated_parameter_names_are_rejected(malformed_name: str) -> None:
    model = Qwen3MoeForCausalLM()
    model._parameters.append((malformed_name, FakeParameter((8, 8))))
    assert Qwen3MoeStaticAdapter().detect(model, model.config).score == 0.0


def test_unsupported_report_has_candidate_component_parity_and_safe_capabilities() -> None:
    model = Qwen3MoeForCausalLM()
    model.config.model_type = "qwen2"
    report = Qwen3MoeStaticAdapter().discover(model, _manifest())
    assert report.candidates == []
    assert report.components == []
    assert report.model_key == report.model_manifest.model_key
    assert report.warnings == sorted(set(report.warnings))


def test_positive_report_candidate_semantics_and_exact_order() -> None:
    model = Qwen3MoeForCausalLM(layout="packed")
    report = Qwen3MoeStaticAdapter().discover(model, _manifest())
    assert [candidate.layer_index for candidate in report.candidates] == sorted(
        candidate.layer_index for candidate in report.candidates
    )
    for candidate, component in zip(report.candidates, report.components, strict=True):
        assert candidate.confidence == 1.0
        assert candidate.shared is None
        if candidate.kind is ComponentKind.EXPERT:
            assert candidate.routed is True
        else:
            assert candidate.routed is None
        assert candidate.evidence[0].detail == "Qwen3-MoE packed layout"
        assert component.capabilities == [CapabilityLabel.STRUCTURE]
        assert component.routed == candidate.routed
        assert component.shared == candidate.shared
        assert component.capture is not None
        assert component.capture.source is CaptureSource.STATIC_STRUCTURE
        assert component.capture.verified is False


def test_public_qwen_module_surface_has_no_extra_exports() -> None:
    from moeatlas.adapters import qwen3_moe

    assert qwen3_moe.__all__ == ["Qwen3MoeStaticAdapter"]
    public_class_members = {
        name for name in Qwen3MoeStaticAdapter.__dict__ if not name.startswith("_")
    }
    assert public_class_members == {"descriptor", "detect", "discover"}


def test_qwen_source_has_no_model_artifact_or_runtime_imports() -> None:
    source = Path("src/moeatlas/adapters/qwen3_moe.py").read_text()
    assert "import torch" not in source
    assert "import transformers" not in source
    assert ".safetensors" not in source
    assert "checkpoint" not in source.lower()


class _ValueForbiddenParameter:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self._shape = shape

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def __getattr__(self, name: str) -> object:
        if name in {"item", "detach", "cpu", "numpy", "tolist", "storage"}:
            raise AssertionError(f"parameter value operation forbidden: {name}")
        raise AttributeError(name)


def test_static_scan_has_no_network_cache_or_runtime_side_effects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = Qwen3MoeForCausalLM(layout="packed")
    model._parameters = [
        (name, _ValueForbiddenParameter(parameter.shape)) for name, parameter in model._parameters
    ]  # type: ignore[list-item]

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    cache_dirs = [tmp_path / name for name in ("hf", "transformers", "hub", "torch")]
    for cache_dir in cache_dirs:
        cache_dir.mkdir()
    for env_name, cache_dir in zip(
        ("HF_HOME", "TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE", "TORCH_HOME"),
        cache_dirs,
        strict=True,
    ):
        monkeypatch.setenv(env_name, str(cache_dir))

    for method_name in (
        "forward",
        "generate",
        "register_forward_hook",
        "register_forward_pre_hook",
        "register_full_backward_hook",
    ):
        setattr(model, method_name, forbidden)
    before_modules = list(model._modules)
    before_parameters = list(model._parameters)
    before_config = dataclasses.asdict(model.config)
    before_tree = tuple(_tree_snapshot(tmp_path))
    optional_before = {
        name
        for name in ("torch", "transformers", "safetensors", "accelerate")
        if name in sys.modules
    }

    adapter = Qwen3MoeStaticAdapter()
    adapter.detect(model, model.config)
    adapter.discover(model, _manifest())
    inspect_static_adapter(adapter, model, model.config, _manifest())

    assert _tree_snapshot(tmp_path) == before_tree
    assert model._modules == before_modules
    assert model._parameters == before_parameters
    assert dataclasses.asdict(model.config) == before_config
    optional_after = {
        name
        for name in ("torch", "transformers", "safetensors", "accelerate")
        if name in sys.modules
    }
    assert optional_after == optional_before
    assert os.environ["HF_HOME"] == str(cache_dirs[0])
