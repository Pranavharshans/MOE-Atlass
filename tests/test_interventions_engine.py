"""Failure-safe intervention engine mechanics over synthetic modules."""

from __future__ import annotations

import pytest

from moeatlas.interventions import (
    INTERVENTION_ENGINE_SCHEMA_VERSION,
    InterventionBudget,
    InterventionEngineError,
    InterventionOperation,
    InterventionOutcome,
    InterventionRecipe,
    run_intervention,
)


class SyntheticModule:
    """A family-neutral stand-in with expert outputs and router logits."""

    def __init__(self, outputs: dict[str, float], biases: dict[str, float]) -> None:
        self.outputs = dict(outputs)
        self.biases = dict(biases)


class SyntheticCapability:
    """Snapshot/apply/restore primitives proving the engine contract."""

    def __init__(self) -> None:
        self.fail_apply = False
        self.fail_restore = False
        self.apply_count = 0
        self.restore_count = 0

    def capture(self, module: SyntheticModule) -> dict[str, dict[str, float]]:
        return {"outputs": dict(module.outputs), "biases": dict(module.biases)}

    def restore(self, module: SyntheticModule, snapshot: object) -> None:
        if self.fail_restore:
            raise RuntimeError("restore is broken")
        assert isinstance(snapshot, dict)
        module.outputs = dict(snapshot["outputs"])
        module.biases = dict(snapshot["biases"])
        self.restore_count += 1

    def apply(self, module: SyntheticModule, recipe: InterventionRecipe) -> None:
        if self.fail_apply:
            raise RuntimeError("apply is broken")
        self.apply_count += 1
        for target in recipe.targets:
            if recipe.operation is InterventionOperation.ABLATE:
                if target in module.outputs:
                    module.outputs[target] = 0.0
            elif recipe.operation is InterventionOperation.SCALE:
                assert recipe.factor is not None
                if target in module.outputs:
                    module.outputs[target] *= recipe.factor
            elif recipe.operation is InterventionOperation.ALTER_ROUTER:
                assert recipe.bias is not None
                if target in module.biases:
                    module.biases[target] += recipe.bias


@pytest.fixture()
def module() -> SyntheticModule:
    return SyntheticModule(
        outputs={"layer-0/expert-1": 3.0, "layer-0/expert-2": -1.5},
        biases={"layer-0/router": 0.25},
    )


def test_surface_is_pinned() -> None:
    assert INTERVENTION_ENGINE_SCHEMA_VERSION == "1.0"
    for stage in ("contract", "capture", "apply", "execute", "restore"):
        with pytest.raises(ValueError, match="stage is not supported"):
            InterventionEngineError(stage + "-nope")


def test_ablation_is_observed_then_restored(module: SyntheticModule) -> None:
    capability = SyntheticCapability()
    before_outputs = dict(module.outputs)
    observed: list[dict[str, float]] = []
    recipe = InterventionRecipe(
        operation=InterventionOperation.ABLATE, targets=("layer-0/expert-1",)
    )

    def observe(current: SyntheticModule) -> None:
        observed.append(dict(current.outputs))

    outcome = run_intervention(module, recipe, capability, observe)
    assert len(observed) == 1
    assert observed[0]["layer-0/expert-1"] == 0.0
    assert observed[0]["layer-0/expert-2"] == -1.5
    assert module.outputs == before_outputs
    assert outcome.schema_version == INTERVENTION_ENGINE_SCHEMA_VERSION
    assert outcome.operation == "ablate"
    assert outcome.targets == ("layer-0/expert-1",)
    assert outcome.recipe_fingerprint == recipe.fingerprint
    assert capability.apply_count == 1 and capability.restore_count == 1


def test_scaling_uses_the_recipe_factor(module: SyntheticModule) -> None:
    seen: list[float] = []
    recipe = InterventionRecipe(
        operation=InterventionOperation.SCALE,
        targets=("layer-0/expert-2",),
        factor=2.0,
    )

    def observe(current: SyntheticModule) -> None:
        seen.append(current.outputs["layer-0/expert-2"])

    run_intervention(module, recipe, SyntheticCapability(), observe)
    assert seen == [-3.0]
    assert module.outputs["layer-0/expert-2"] == -1.5


def test_router_alteration_touches_only_router_state(module: SyntheticModule) -> None:
    seen: list[float] = []
    recipe = InterventionRecipe(
        operation=InterventionOperation.ALTER_ROUTER,
        targets=("layer-0/router",),
        bias=1.0,
    )

    def observe(current: SyntheticModule) -> None:
        seen.append(current.biases["layer-0/router"])

    run_intervention(module, recipe, SyntheticCapability(), observe)
    assert seen == [1.25]
    assert module.biases == {"layer-0/router": 0.25}


def test_apply_failure_restores_and_reports_the_apply_stage(
    module: SyntheticModule,
) -> None:
    capability = SyntheticCapability()
    capability.fail_apply = True
    before_outputs = dict(module.outputs)
    recipe = InterventionRecipe(
        operation=InterventionOperation.ABLATE, targets=("layer-0/expert-1",)
    )
    with pytest.raises(InterventionEngineError) as caught:
        run_intervention(module, recipe, capability, lambda _module: None)
    assert caught.value.stage == "apply"
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert module.outputs == before_outputs


def test_execute_failure_restores_and_reports_the_execute_stage(
    module: SyntheticModule,
) -> None:
    def explode(_module: SyntheticModule) -> None:
        raise ValueError("observation exploded")

    recipe = InterventionRecipe(
        operation=InterventionOperation.ABLATE, targets=("layer-0/expert-1",)
    )
    with pytest.raises(InterventionEngineError) as caught:
        run_intervention(module, recipe, SyntheticCapability(), explode)
    assert caught.value.stage == "execute"
    assert module.outputs["layer-0/expert-1"] == 3.0


def test_failed_restoration_is_reported_and_never_publishes_an_outcome(
    module: SyntheticModule,
) -> None:
    capability = SyntheticCapability()
    capability.fail_restore = True
    recipe = InterventionRecipe(
        operation=InterventionOperation.ABLATE, targets=("layer-0/expert-1",)
    )
    with pytest.raises(InterventionEngineError) as caught:
        run_intervention(module, recipe, capability, lambda _module: None)
    assert caught.value.stage == "restore"
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_cancellation_propagates_after_restoration(module: SyntheticModule) -> None:
    def cancel(_module: SyntheticModule) -> None:
        raise KeyboardInterrupt

    recipe = InterventionRecipe(
        operation=InterventionOperation.ABLATE, targets=("layer-0/expert-1",)
    )
    with pytest.raises(KeyboardInterrupt):
        run_intervention(module, recipe, SyntheticCapability(), cancel)
    assert module.outputs["layer-0/expert-1"] == 3.0


def test_budget_violation_is_a_contract_failure(module: SyntheticModule) -> None:
    recipe = InterventionRecipe(
        operation=InterventionOperation.ABLATE,
        targets=("layer-0/expert-1", "layer-0/expert-2"),
    )
    with pytest.raises(InterventionEngineError) as caught:
        run_intervention(
            module,
            recipe,
            SyntheticCapability(),
            lambda _m: None,
            budget=InterventionBudget(max_targets=1),
        )
    assert caught.value.stage == "contract"
    assert "budget is 1" in str(caught.value)
    outcome = run_intervention(
        module,
        recipe,
        SyntheticCapability(),
        lambda _m: None,
        budget=InterventionBudget(max_targets=2),
    )
    assert outcome.targets == ("layer-0/expert-1", "layer-0/expert-2")


def test_contract_type_checks() -> None:
    recipe = InterventionRecipe(
        operation=InterventionOperation.ABLATE, targets=("a",)
    )
    capability = SyntheticCapability()

    class ApplyLess:
        def capture(self, module: object) -> object:
            return None

        def restore(self, module: object, snapshot: object) -> None:
            return None

    with pytest.raises(TypeError):
        run_intervention(None, recipe, capability, lambda _m: None)
    with pytest.raises(TypeError):
        run_intervention(object(), "not-a-recipe", capability, lambda _m: None)
    with pytest.raises(TypeError, match="callable apply"):
        run_intervention(object(), recipe, ApplyLess(), lambda _m: None)
    with pytest.raises(TypeError):
        run_intervention(object(), recipe, capability, "not-callable")
    with pytest.raises(TypeError):
        run_intervention(object(), recipe, capability, lambda _m: None, budget=64)


def test_capture_failure_reports_the_capture_stage(module: SyntheticModule) -> None:
    class BrokenCapture(SyntheticCapability):
        def capture(self, module: SyntheticModule) -> object:
            raise RuntimeError("capture is broken")

    recipe = InterventionRecipe(
        operation=InterventionOperation.ABLATE, targets=("layer-0/expert-1",)
    )
    with pytest.raises(InterventionEngineError) as caught:
        run_intervention(module, recipe, BrokenCapture(), lambda _m: None)
    assert caught.value.stage == "capture"


def test_outcome_canonical_round_trip() -> None:
    recipe = InterventionRecipe(
        operation=InterventionOperation.REROUTE,
        targets=("layer-0/expert-1",),
        alternates=(("layer-0/expert-1", "layer-0/expert-2"),),
    )
    outcome = InterventionOutcome(
        schema_version=INTERVENTION_ENGINE_SCHEMA_VERSION,
        recipe_fingerprint=recipe.fingerprint,
        operation="reroute",
        targets=("layer-0/expert-1",),
    )
    document = outcome.to_json()
    assert '"artifact_type":"moeatlas.intervention_outcome"' in document
    assert InterventionOutcome.from_json(document) == outcome
    assert InterventionOutcome.from_json(document.encode()) == outcome
    with pytest.raises(ValueError, match="not an intervention outcome artifact"):
        InterventionOutcome.from_json('{"schema_version":"1.0"}')
    with pytest.raises(ValueError, match="not valid JSON"):
        InterventionOutcome.from_json("nope")
