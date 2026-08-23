"""Model-free contract tests for real structure-driven expert interventions."""

from __future__ import annotations

import pytest

from moeatlas.interventions import (
    InterventionOperation,
    InterventionRecipe,
    TransformersExpertInterventionCapability,
    TransformersInterventionError,
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
