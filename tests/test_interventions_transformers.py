"""Model-free contract tests for real structure-driven expert interventions."""

from __future__ import annotations

import pytest

from moeatlas.core import ComponentKind
from moeatlas.interventions import (
    ExpertWeightLayout,
    InterventionOperation,
    InterventionRecipe,
    InterventionSupportTier,
    TransformersExpertInterventionCapability,
    TransformersInterventionError,
    classify_intervention_capability,
    intervention_targets,
    run_intervention,
)

from .test_runtime_generic_capture import _flat_logits
from .test_runtime_generic_expert_capture import _ExpertHookedModel, scan_report


class _TensorLike:
    def __init__(self, value: float) -> None:
        self.value = value

    def mul(self, factor: float) -> _TensorLike:
        return _TensorLike(self.value * factor)


def _model() -> _ExpertHookedModel:
    outputs = {
        f"layers.{layer}.experts.{expert}": _TensorLike(4.0)
        for layer in range(2)
        for expert in range(4)
    }
    return _ExpertHookedModel(_flat_logits(), expert_outputs=outputs)


def test_inventory_exposes_stable_layer_expert_coordinates() -> None:
    targets = intervention_targets(scan_report(_model()))

    assert len(targets) == 8
    assert targets[0].label == "layer:0/expert:0"
    assert targets[-1].label == "layer:1/expert:3"
    assert targets[0].module_path == "layers.0.experts.0"


def test_capability_report_declares_exposed_expert_operations() -> None:
    capability = classify_intervention_capability(scan_report(_model()))

    assert capability.tier is InterventionSupportTier.EXPOSED_EXPERTS
    assert [operation.value for operation in capability.operations] == ["ablate", "scale"]
    assert capability.target_count == 8
    assert capability.live_supported is True
    assert capability.weight_layout is ExpertWeightLayout.INDEXED_MODULES
    assert capability.execution_backend is None
    assert capability.fused_backend is None


def test_ablation_hooks_are_exercised_and_always_removed() -> None:
    model = _model()
    capability = TransformersExpertInterventionCapability(scan_report(model))
    recipe = InterventionRecipe(
        operation=InterventionOperation.ABLATE,
        targets=("layer:0/expert:1",),
    )
    observed: list[object] = []

    outcome = run_intervention(
        model,
        recipe,
        capability,
        lambda active_model: observed.append(active_model()),
    )

    assert outcome.recipe_fingerprint == recipe.fingerprint
    assert capability.invocation_counts == {"layer:0/expert:1": 1}
    selected = dict(model.named_modules())["layers.0.experts.1"]
    assert selected.callbacks == []


def test_unsupported_operation_and_unknown_target_fail_before_execution() -> None:
    model = _model()
    capability = TransformersExpertInterventionCapability(scan_report(model))
    reroute = InterventionRecipe(
        operation=InterventionOperation.REROUTE,
        targets=("layer:0/expert:0",),
        alternates=(("layer:0/expert:0", "layer:0/expert:1"),),
    )
    with pytest.raises(Exception, match="recipe application failed"):
        run_intervention(model, reroute, capability, lambda _: None)

    unknown = InterventionRecipe(
        operation=InterventionOperation.ABLATE,
        targets=("layer:99/expert:0",),
    )
    with pytest.raises(Exception, match="recipe application failed"):
        run_intervention(model, unknown, capability, lambda _: None)
    assert all(
        not node.callbacks for _, node in model.named_modules() if hasattr(node, "callbacks")
    )


def test_inventory_rejects_reports_without_independent_experts() -> None:
    class DenseModel:
        config = type("Config", (), {"num_hidden_layers": 2})()

        def named_modules(self):
            yield "", self

    report = scan_report(DenseModel())
    with pytest.raises(TransformersInterventionError, match="independently hookable"):
        intervention_targets(report)
    capability = classify_intervention_capability(report)
    assert capability.tier is InterventionSupportTier.UNAVAILABLE
    assert capability.live_supported is False


def test_packed_experts_are_separate_from_unresolved_execution_backend() -> None:
    report = scan_report(_model())
    expert_count = report.facts.expert_count
    assert expert_count == 4
    packed = report.model_copy(
        update={
            "candidates": [
                candidate
                for candidate in report.candidates
                if candidate.kind is not ComponentKind.EXPERT
            ],
            "components": [
                component
                for component in report.components
                if component.kind is not ComponentKind.EXPERT
            ],
        }
    )
    packed_container = next(
        component
        for component in packed.components
        if component.kind is ComponentKind.EXPERT_CONTAINER
    )
    packed = packed.model_copy(
        update={
            "components": [
                component.model_copy(
                    update={"tensor_shapes": {"gate_up_proj": [4, 16, 8]}}
                )
                if component.component_key == packed_container.component_key
                else component
                for component in packed.components
            ]
        }
    )

    capability = classify_intervention_capability(packed)

    assert capability.tier is InterventionSupportTier.PACKED_EXPERTS
    assert capability.operations == ()
    assert capability.weight_layout is ExpertWeightLayout.PACKED_TENSORS
    assert capability.execution_backend is None
    assert capability.fused_backend is None
    assert capability.to_dict()["weight_layout"] == "packed_tensors"


def test_opaque_expert_storage_is_not_mislabeled_as_fused() -> None:
    report = scan_report(_model())
    opaque = report.model_copy(
        update={
            "candidates": [
                candidate
                for candidate in report.candidates
                if candidate.kind is not ComponentKind.EXPERT
            ],
            "components": [
                component.model_copy(update={"tensor_shapes": {}})
                for component in report.components
                if component.kind is not ComponentKind.EXPERT
            ],
        }
    )

    capability = classify_intervention_capability(opaque)

    assert capability.tier is InterventionSupportTier.OPAQUE_EXPERTS
    assert capability.weight_layout is ExpertWeightLayout.OPAQUE
    assert capability.fused_backend is None
