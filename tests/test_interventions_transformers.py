"""Model-free contract tests for real structure-driven expert interventions."""

from __future__ import annotations

import pytest

from moeatlas.core import ComponentKind
from moeatlas.interventions import (
    ExpertBackendDiscoveryStatus,
    ExpertExecutionMode,
    ExpertOperation,
    ExpertWeightLayout,
    InterventionOperation,
    InterventionRecipe,
    InterventionSupportTier,
    OperationCapabilityStatus,
    TransformersExpertInterventionCapability,
    TransformersInterventionError,
    classify_intervention_capability,
    discover_huggingface_expert_backends,
    inspect_intervention_capability,
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
                component.model_copy(update={"tensor_shapes": {"gate_up_proj": [4, 16, 8]}})
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


def test_huggingface_backend_discovery_preserves_each_submodel_scope() -> None:
    model = _model()
    model.get_experts_implementation = lambda: {  # type: ignore[attr-defined]
        "": "grouped_mm",
        "language_model": "sonicmoe",
        "vision_model": None,
    }

    evidence = discover_huggingface_expert_backends(model)

    assert evidence.status is ExpertBackendDiscoveryStatus.OBSERVED
    assert [item.scope for item in evidence.backends] == [
        "",
        "language_model",
        "vision_model",
    ]
    assert evidence.backends[0].mode is ExpertExecutionMode.GROUPED_MATMUL
    assert evidence.backends[0].fused is False
    assert evidence.backends[1].mode is ExpertExecutionMode.FUSED
    assert evidence.backends[1].fused is True
    assert evidence.backends[2].mode is ExpertExecutionMode.UNRESOLVED
    assert evidence.backends[2].fused is None


def test_unknown_huggingface_backend_is_not_guessed_as_fused() -> None:
    model = _model()
    model.get_experts_implementation = lambda: {"": "vendor_kernel"}  # type: ignore[attr-defined]

    evidence = discover_huggingface_expert_backends(model)

    assert evidence.status is ExpertBackendDiscoveryStatus.OBSERVED
    assert len(evidence.backends) == 1
    assert evidence.backends[0].mode is ExpertExecutionMode.CUSTOM
    assert evidence.backends[0].fused is None


@pytest.mark.parametrize(
    "snapshot",
    [[], {"": 7}, {7: "eager"}, {"": "\n"}, {"text": "eager", " text ": "eager"}],
)
def test_invalid_huggingface_backend_snapshot_fails_closed(snapshot: object) -> None:
    model = _model()
    model.get_experts_implementation = lambda: snapshot  # type: ignore[attr-defined]

    evidence = discover_huggingface_expert_backends(model)

    assert evidence.status is ExpertBackendDiscoveryStatus.INVALID
    assert evidence.backends == ()


def test_missing_huggingface_backend_interface_is_nonfatal() -> None:
    evidence = discover_huggingface_expert_backends(_model())

    assert evidence.status is ExpertBackendDiscoveryStatus.UNAVAILABLE
    assert evidence.backends == ()


def test_live_capability_combines_static_layout_and_backend_evidence() -> None:
    model = _model()
    model.get_experts_implementation = lambda: {"": "grouped_mm"}  # type: ignore[attr-defined]

    capability = inspect_intervention_capability(scan_report(model), model)

    assert capability.weight_layout is ExpertWeightLayout.INDEXED_MODULES
    assert capability.execution_backend == "grouped_mm"
    assert capability.fused_backend is False
    assert capability.backend_discovery is not None
    assert capability.backend_discovery.status is ExpertBackendDiscoveryStatus.OBSERVED
    assert capability.to_dict()["execution_backends"] == [
        {
            "scope": "",
            "implementation": "grouped_mm",
            "mode": "grouped_matmul",
            "fused": False,
            "source": "model.get_experts_implementation",
        }
    ]


def test_operation_report_distinguishes_capture_contribution_and_compute() -> None:
    capability = classify_intervention_capability(scan_report(_model()))
    operations = {item.operation: item for item in capability.operation_capabilities}

    assert tuple(operations) == tuple(ExpertOperation)
    assert (
        operations[ExpertOperation.CAPTURE_ROUTING].status
        is OperationCapabilityStatus.RUN_VALIDATION_REQUIRED
    )
    assert (
        operations[ExpertOperation.ZERO_CONTRIBUTION].status is OperationCapabilityStatus.AVAILABLE
    )
    assert (
        operations[ExpertOperation.SCALE_CONTRIBUTION].status is OperationCapabilityStatus.AVAILABLE
    )
    assert (
        operations[ExpertOperation.EXCLUDE_AND_RENORMALIZE].status
        is OperationCapabilityStatus.NOT_IMPLEMENTED
    )
    assert operations[ExpertOperation.ZERO_CONTRIBUTION].changes_routing is False
    assert operations[ExpertOperation.ZERO_CONTRIBUTION].skips_compute is False
    assert operations[ExpertOperation.SKIP_COMPUTE].skips_compute is True


def test_packed_backend_report_does_not_claim_packed_intervention_support() -> None:
    report = scan_report(_model())
    packed = report.model_copy(
        update={
            "candidates": [
                candidate
                for candidate in report.candidates
                if candidate.kind is not ComponentKind.EXPERT
            ],
            "components": [
                component.model_copy(update={"tensor_shapes": {"weights": [4, 16, 8]}})
                if component.kind is ComponentKind.EXPERT_CONTAINER
                else component
                for component in report.components
                if component.kind is not ComponentKind.EXPERT
            ],
        }
    )
    model = _model()
    model.get_experts_implementation = lambda: {"": "grouped_mm"}  # type: ignore[attr-defined]

    capability = inspect_intervention_capability(packed, model)
    operations = {item.operation: item for item in capability.operation_capabilities}

    assert capability.weight_layout is ExpertWeightLayout.PACKED_TENSORS
    assert (
        operations[ExpertOperation.ZERO_CONTRIBUTION].status
        is OperationCapabilityStatus.NOT_IMPLEMENTED
    )
    assert (
        "runtime.expert_backend=grouped_mm"
        in operations[ExpertOperation.ZERO_CONTRIBUTION].evidence
    )


def test_dense_operation_report_is_explicitly_unavailable() -> None:
    class DenseModel:
        config = type("Config", (), {"num_hidden_layers": 2})()

        def named_modules(self):
            yield "", self

    capability = classify_intervention_capability(scan_report(DenseModel()))

    assert all(
        operation.status is OperationCapabilityStatus.UNAVAILABLE
        for operation in capability.operation_capabilities
    )
