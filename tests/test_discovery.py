from __future__ import annotations

import sys
from dataclasses import dataclass, field

import pytest
from pydantic import ValidationError

from moeatlas.core import (
    CapabilityLabel,
    ComponentKind,
    ComponentManifest,
    DType,
    ModelManifest,
    TokenizerIdentity,
    make_component_key,
    make_config_hash,
    make_model_key,
)
from moeatlas.discovery import (
    DiscoveryReport,
    bind_moe_layer_key,
    has_whole_word_moe_marker,
    scan,
    trusted_routers,
)

from .fixtures import SyntheticMoE


def _model_manifest() -> ModelManifest:
    model_key = make_model_key("acme/demo-moe", "main")
    return ModelManifest(
        model_key=model_key,
        architecture="demo_moe",
        revision="main",
        config_hash=make_config_hash({"experts": 4, "top_k": 2}),
        tokenizer=TokenizerIdentity(identifier="acme/demo-tokenizer", revision="main"),
        dtype=DType.BFLOAT16,
        device_map={"": "cpu"},
    )


def test_synthetic_scan_detects_expected_components_and_facts() -> None:
    report = scan(SyntheticMoE(), _model_manifest())

    assert report.schema_version == "1.0"
    assert report.manifest_type == "discovery_report"
    assert report.facts.expert_count == 4
    assert report.facts.routed_top_k == 2
    assert report.facts.shared_expert_count == 1
    assert report.facts.expert_count_source == "config.num_local_experts"
    assert report.facts.routed_top_k_source == "config.num_experts_per_tok"

    kinds = {candidate.kind for candidate in report.candidates}
    assert kinds == {
        ComponentKind.MOE_LAYER,
        ComponentKind.ROUTER,
        ComponentKind.EXPERT_CONTAINER,
        ComponentKind.EXPERT,
        ComponentKind.SHARED_EXPERT,
    }
    paths = {candidate.module_path for candidate in report.candidates}
    assert "layers.0" in paths
    assert "layers.0.router" in paths
    assert "layers.0.experts" in paths
    assert {path for path in paths if path.startswith("layers.0.experts.")} == {
        "layers.0.experts.0",
        "layers.0.experts.1",
        "layers.0.experts.2",
        "layers.0.experts.3",
    }
    assert {path for path in paths if path.startswith("layers.1.experts.")} == {
        "layers.1.experts.0",
        "layers.1.experts.1",
        "layers.1.experts.2",
        "layers.1.experts.3",
    }
    assert "layers.0.shared_expert" in paths
    assert "layers.1.shared_expert" in paths

    for candidate, component in zip(report.candidates, report.components, strict=True):
        assert candidate.component_key == make_component_key(
            report.model_key,
            candidate.kind.value,
            candidate.module_path,
            layer_index=candidate.layer_index,
            expert_index=candidate.expert_index,
        )
        assert candidate.component_key == component.component_key
        assert component.capabilities == [CapabilityLabel.STRUCTURE]
        assert component.provenance is not None
        assert component.provenance.source == "static-discovery"
        assert candidate.evidence

    expert_candidates = [
        candidate for candidate in report.candidates if candidate.kind is ComponentKind.EXPERT
    ]
    assert len(expert_candidates) == 8
    assert {
        candidate.layer_index for candidate in expert_candidates if candidate.layer_index == 0
    } == {0}
    assert {
        candidate.layer_index for candidate in expert_candidates if candidate.layer_index == 1
    } == {1}
    assert [candidate.expert_index for candidate in expert_candidates[:4]] == [0, 1, 2, 3]
    assert [candidate.expert_index for candidate in expert_candidates[4:]] == [0, 1, 2, 3]
    assert all(candidate.routed is True for candidate in expert_candidates)

    shared_candidates = [
        candidate
        for candidate in report.candidates
        if candidate.kind is ComponentKind.SHARED_EXPERT
    ]
    assert len(shared_candidates) == 2
    assert {candidate.layer_index for candidate in shared_candidates} == {0, 1}


def test_nested_composite_config_uses_text_topology_and_exact_sources() -> None:
    """Composite configs are read structurally, without a family allowlist."""

    @dataclass
    class TextConfig:
        num_experts: int = 4
        num_experts_per_tok: int = 2

    @dataclass
    class VisionConfig:
        # A multimodal config can carry unrelated expert-like fields.
        num_experts: int = 8
        num_experts_per_tok: int = 1

    @dataclass
    class CompositeConfig:
        text_config: TextConfig = field(default_factory=TextConfig)
        vision_config: VisionConfig = field(default_factory=VisionConfig)

    class CompositeMoE(SyntheticMoE):
        def __init__(self) -> None:
            super().__init__()
            self.config = CompositeConfig()

    report = scan(CompositeMoE(), _model_manifest())

    assert report.facts.expert_count == 4
    assert report.facts.routed_top_k == 2
    assert report.facts.expert_count_source == "config.text_config.num_experts"
    assert report.facts.routed_top_k_source == "config.text_config.num_experts_per_tok"
    assert not any("conflicting expert_count" in warning for warning in report.warnings)
    assert not any("conflicting routed_top_k" in warning for warning in report.warnings)


def test_nested_config_conflicts_remain_explicit_within_same_role() -> None:
    @dataclass
    class TextConfig:
        num_experts: int = 4
        num_local_experts: int = 6
        num_experts_per_tok: int = 2

    @dataclass
    class CompositeConfig:
        text_config: TextConfig = field(default_factory=TextConfig)

    class CompositeModel:
        config = CompositeConfig()

        def named_modules(self):
            yield "", self

        def named_parameters(self):
            return iter(())

    report = scan(CompositeModel(), _model_manifest())

    assert report.facts.expert_count == 6
    assert report.facts.expert_count_source == "config.text_config.num_local_experts"
    assert any("conflicting expert_count configuration" in warning for warning in report.warnings)


def test_discovery_report_is_json_round_trip_safe_and_deterministic() -> None:
    first = scan(SyntheticMoE(), _model_manifest())
    second = scan(SyntheticMoE(), _model_manifest())

    assert first.to_json() == second.to_json()
    decoded = DiscoveryReport.from_json(first.to_json(indent=2))
    assert decoded == first
    assert decoded.to_dict() == first.to_dict()


def test_scan_does_not_mutate_the_inspected_object() -> None:
    model = SyntheticMoE()

    def snapshot(node: object) -> tuple[object, ...]:
        children = getattr(node, "_children", {})
        parameters = getattr(node, "_parameters", {})
        config = getattr(node, "config", None)
        return (
            type(node).__qualname__,
            tuple(sorted((name, parameter.shape) for name, parameter in parameters.items())),
            tuple(sorted((name, snapshot(child)) for name, child in children.items())),
            repr(config),
        )

    before = snapshot(model)

    scan(model, _model_manifest())

    assert snapshot(model) == before


def test_dense_objects_do_not_become_moe_candidates() -> None:
    class DenseModel:
        def named_modules(self):
            yield "", self
            yield "layers.0.linear", object()

        def named_parameters(self):
            return iter(())

    report = scan(DenseModel(), _model_manifest())

    assert report.candidates == []
    assert report.components == []
    assert report.warnings == []


def test_realistic_dense_gated_mlp_stays_ambiguous_and_never_moe() -> None:
    class DenseProjection:
        pass

    class DenseGatedMLP:
        def named_modules(self):
            yield "", self
            yield "layers.0.mlp.gate_proj", DenseProjection()
            yield "layers.0.mlp.up_proj", DenseProjection()
            yield "layers.0.mlp.down_proj", DenseProjection()

    report = scan(DenseGatedMLP(), _model_manifest())

    assert all(candidate.confidence < 0.60 for candidate in report.candidates)
    assert all(candidate.kind is ComponentKind.ROUTER for candidate in report.candidates)
    assert not any(
        candidate.kind
        in {
            ComponentKind.MOE_LAYER,
            ComponentKind.EXPERT_CONTAINER,
            ComponentKind.EXPERT,
            ComponentKind.SHARED_EXPERT,
        }
        for candidate in report.candidates
    )
    assert any("ambiguous" in warning for warning in report.warnings)


def test_ambiguous_gate_is_reported_with_low_confidence_and_warning() -> None:
    class AmbiguousGateModel:
        config = {"num_experts": 4}

        def named_modules(self):
            yield "", self
            yield "gate", object()

    report = scan(AmbiguousGateModel(), _model_manifest())

    assert len(report.candidates) == 1
    candidate = report.candidates[0]
    assert candidate.kind is ComponentKind.ROUTER
    assert candidate.confidence < 0.60
    assert any("ambiguous" in warning for warning in candidate.warnings)
    assert any("ambiguous" in warning for warning in report.warnings)


def test_missing_optional_surfaces_are_explicitly_warned() -> None:
    report = scan(object(), _model_manifest())

    assert report.candidates == []
    assert any("named_modules" in warning for warning in report.warnings)


def test_iteration_failures_keep_prior_entries_and_warn_deterministically() -> None:
    class FailingIterationModel:
        config = {"num_experts": 2, "top_k": 1}

        def named_modules(self):
            yield "", self
            yield ("malformed",)
            yield "router", object()
            raise RuntimeError("synthetic iterator failure")

        def named_parameters(self):
            yield ("malformed",)
            yield "router.weight", type("Parameter", (), {"shape": (2, 8)})()
            raise LookupError("synthetic parameter failure")

    report = scan(FailingIterationModel(), _model_manifest())

    assert any(candidate.module_path == "router" for candidate in report.candidates)
    assert any(
        "named_modules() iteration failed after 2 retained pair(s)" in warning
        for warning in report.warnings
    )
    assert any(
        "named_parameters() iteration failed after 1 retained pair(s)" in warning
        for warning in report.warnings
    )
    assert any(
        "named_modules() item 1 is not a (name, value) pair" in warning
        for warning in report.warnings
    )
    assert any(
        "named_parameters() item 0 is not a (name, value) pair" in warning
        for warning in report.warnings
    )
    assert report.warnings == sorted(report.warnings)


def test_semantic_shape_evidence_requires_configured_expert_axis() -> None:
    class PackedExperts:
        def __init__(self, shape: tuple[int, ...]) -> None:
            self.shape = shape

    class PackedModel:
        config = {"num_experts": 4, "top_k": 2}

        def __init__(self, shape: tuple[int, ...]) -> None:
            self.shape = shape

        def named_modules(self):
            yield "", self
            yield "layers.0.moe", self
            yield "layers.0.moe.router", self
            yield "layers.0.moe.experts", PackedExperts(self.shape)

        def named_parameters(self):
            yield "layers.0.moe.router.weight", PackedExperts((self.shape[0], 8))
            yield "layers.0.moe.experts.weight", PackedExperts(self.shape)

    matching = scan(PackedModel((4, 16, 8)), _model_manifest())
    packed = next(
        candidate
        for candidate in matching.candidates
        if candidate.kind is ComponentKind.EXPERT_CONTAINER
    )
    assert any(evidence.signal.value == "parameter_shape" for evidence in packed.evidence)
    router = next(
        candidate for candidate in matching.candidates if candidate.kind is ComponentKind.ROUTER
    )
    assert any(evidence.signal.value == "parameter_shape" for evidence in router.evidence)

    mismatch = scan(PackedModel((3, 16, 8)), _model_manifest())
    packed_mismatch = next(
        candidate
        for candidate in mismatch.candidates
        if candidate.kind is ComponentKind.EXPERT_CONTAINER
    )
    assert not any(
        evidence.signal.value == "parameter_shape" for evidence in packed_mismatch.evidence
    )
    assert any("packed expert-container shapes" in warning for warning in packed_mismatch.warnings)
    router_mismatch = next(
        candidate for candidate in mismatch.candidates if candidate.kind is ComponentKind.ROUTER
    )
    assert not any(
        evidence.signal.value == "parameter_shape" for evidence in router_mismatch.evidence
    )
    assert any("router parameter shapes" in warning for warning in router_mismatch.warnings)


def test_report_invariants_reject_capability_and_duplicate_field_drift() -> None:
    report = scan(SyntheticMoE(), _model_manifest())
    values = report.to_dict()

    values["components"][0]["capabilities"] = ["STRUCTURE", "MODULE"]
    with pytest.raises(ValidationError, match=r"exactly \[STRUCTURE\]"):
        DiscoveryReport.model_validate(values)

    values = report.to_dict()
    values["candidates"][0]["routed"] = not bool(values["candidates"][0]["routed"])
    with pytest.raises(ValidationError, match="routed/shared fields"):
        DiscoveryReport.model_validate(values)

    values = report.to_dict()
    values["candidates"][0]["warnings"] = ["drift"]
    with pytest.raises(ValidationError, match="warnings must agree"):
        DiscoveryReport.model_validate(values)

    values = report.to_dict()
    values["candidates"][0]["confidence"] = 0.0
    with pytest.raises(ValidationError, match="confidence must equal"):
        DiscoveryReport.model_validate(values)

    values = report.to_dict()
    values["candidates"][0]["evidence"].append(values["candidates"][0]["evidence"][0])
    with pytest.raises(ValidationError, match="duplicate identical entries"):
        DiscoveryReport.model_validate(values)

    class GateOnly:
        def named_modules(self):
            yield "", self
            yield "gate", object()

    ambiguous_values = scan(GateOnly(), _model_manifest()).to_dict()
    ambiguous_values["candidates"][0]["warnings"] = []
    with pytest.raises(ValidationError, match="requires an ambiguity warning"):
        DiscoveryReport.model_validate(ambiguous_values)


def test_discovery_fact_values_and_sources_are_paired() -> None:
    report = scan(SyntheticMoE(), _model_manifest())
    values = report.to_dict()
    values["facts"]["expert_count_source"] = None
    with pytest.raises(ValidationError, match="expert_count and its source"):
        DiscoveryReport.model_validate(values)

    values = report.to_dict()
    values["facts"]["expert_count"] = None
    with pytest.raises(ValidationError, match="expert_count and its source"):
        DiscoveryReport.model_validate(values)


def test_conflicting_per_layer_structural_counts_warn_deterministically() -> None:
    class UnevenExperts:
        def named_modules(self):
            yield "", self
            for layer_index, count in ((0, 2), (1, 3)):
                yield f"layers.{layer_index}.experts", self
                for expert_index in range(count):
                    yield f"layers.{layer_index}.experts.{expert_index}", object()

    report = scan(UnevenExperts(), _model_manifest())

    assert report.facts.expert_count is None
    assert report.facts.expert_count_source is None
    assert report.warnings == sorted(report.warnings)
    assert (
        "conflicting per-layer/container expert_count counts: "
        "layers.0.experts=2, layers.1.experts=3"
    ) in report.warnings


def test_routed_top_k_above_expert_count_warns() -> None:
    class InvalidRoutingConfig:
        config = {"num_experts": 2, "top_k": 3}

        def named_modules(self):
            yield "", self
            yield "layers.0.moe", self
            yield "layers.0.moe.router", self
            yield "layers.0.moe.experts", self
            yield "layers.0.moe.experts.0", object()
            yield "layers.0.moe.experts.1", object()

    report = scan(InvalidRoutingConfig(), _model_manifest())

    assert report.facts.expert_count == 2
    assert report.facts.routed_top_k == 3
    assert "routed_top_k configuration=3 exceeds expert_count=2" in report.warnings


def test_trusted_router_resolution_binds_to_published_container_evidence() -> None:
    model = SyntheticMoE()
    report = scan(model, _model_manifest())

    trusted = trusted_routers(report.components)

    assert [router.module_path for router in trusted] == [
        "layers.0.router",
        "layers.1.router",
    ]


def test_fallback_selection_requires_an_exact_gate_leaf_and_expert_evidence() -> None:
    model_key = make_model_key("acme/legacy-report", "r1")

    def component(
        kind: ComponentKind,
        path: str,
        *,
        layer_index: int,
        expert_index: int | None = None,
        **extra: object,
    ):
        return ComponentManifest(
            component_key=make_component_key(
                model_key,
                kind.value,
                path,
                layer_index=layer_index,
                expert_index=expert_index,
            ),
            model_key=model_key,
            kind=kind,
            module_path=path,
            layer_index=layer_index,
            expert_index=expert_index,
            capabilities=[CapabilityLabel.STRUCTURE],
            **extra,
        )

    components = [
        component(ComponentKind.ROUTER, "stack.layers.7.moe.gate", layer_index=7),
        # SwiGLU-style noise: tokenizes like a router but hosts no experts.
        component(ComponentKind.ROUTER, "stack.layers.7.moe.experts.0.gate_proj", layer_index=7),
    ]
    components.extend(
        component(
            ComponentKind.EXPERT,
            f"stack.layers.7.moe.experts.{index}",
            layer_index=7,
            expert_index=index,
            routed=True,
            shared=False,
        )
        for index in range(4)
    )

    trusted = trusted_routers(components)
    assert [router.module_path for router in trusted] == ["stack.layers.7.moe.gate"]
    assert bind_moe_layer_key(model_key, components, trusted[0]) == make_component_key(
        model_key, "moe_layer", "stack.layers.7.moe", layer_index=7
    )

    renamed = [
        item.model_copy(
            update={
                "module_path": "stack.layers.7.moe.gating_unit",
                "component_key": make_component_key(
                    model_key,
                    "router",
                    "stack.layers.7.moe.gating_unit",
                    layer_index=7,
                ),
            }
        )
        for item in components
        if item.kind is ComponentKind.ROUTER and item.module_path.endswith("moe.gate")
    ]
    survivors = [
        item for item in components if not item.module_path.endswith("moe.gate")
    ] + renamed
    assert trusted_routers(survivors) == ()


def test_layer_binding_refuses_to_guess_without_whole_word_moe_marker() -> None:
    model_key = make_model_key("acme/legacy-report", "r1")

    def component(
        kind: ComponentKind,
        path: str,
        *,
        layer_index: int,
        expert_index: int | None = None,
        **extra: object,
    ):
        return ComponentManifest(
            component_key=make_component_key(
                model_key,
                kind.value,
                path,
                layer_index=layer_index,
                expert_index=expert_index,
            ),
            model_key=model_key,
            kind=kind,
            module_path=path,
            layer_index=layer_index,
            expert_index=expert_index,
            capabilities=[CapabilityLabel.STRUCTURE],
            **extra,
        )

    router = component(ComponentKind.ROUTER, "stack.layers.7.core.gate", layer_index=7)
    components = [router]
    components.extend(
        component(
            ComponentKind.EXPERT,
            f"stack.layers.7.core.bays.{index}",
            layer_index=7,
            expert_index=index,
            routed=True,
            shared=False,
        )
        for index in range(4)
    )

    assert [item.module_path for item in trusted_routers(components)] == [
        "stack.layers.7.core.gate"
    ]
    with pytest.raises(ValueError, match="must bind exactly one MoE layer"):
        bind_moe_layer_key(model_key, components, router)


def test_whole_word_moe_marker_never_matches_camelcase_substrings() -> None:
    assert has_whole_word_moe_marker("moe.layers.0")
    assert has_whole_word_moe_marker("blocks.3.sparse-moe")
    assert has_whole_word_moe_marker("moe_block.router")
    assert not has_whole_word_moe_marker("BailingMoeV3RMSNorm")
    assert not has_whole_word_moe_marker("model.layers.1.mlp")


def test_discovery_import_does_not_load_model_runtime() -> None:
    runtime_names = {"torch", "transformers", "safetensors"}
    assert not any(name in sys.modules for name in runtime_names)
