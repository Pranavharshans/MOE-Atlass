"""Contract tests for immutable intervention recipes and budgets."""

from __future__ import annotations

import pytest

from moeatlas.interventions import (
    INTERVENTION_SCHEMA_VERSION,
    InterventionBudget,
    InterventionBudgetError,
    InterventionOperation,
    InterventionRecipe,
    recipe_budget_from_json,
)
from moeatlas.runs.specs import InterventionLineage

_RECIPE_DOCUMENT = (
    '{"alternates":[],"artifact_type":"moeatlas.intervention_recipe",'
    '"bias":null,"factor":2.0,"operation":"scale",'
    '"schema_version":"1.0","targets":["layer-0/expert-1","layer-0/expert-3"]}'
)


def test_surface_is_pinned() -> None:
    assert INTERVENTION_SCHEMA_VERSION == "1.0"
    assert [op.value for op in InterventionOperation] == [
        "ablate",
        "scale",
        "reroute",
        "alter_router",
    ]
    assert str(InterventionOperation.ABLATE) == "ablate"
    with pytest.raises(ValueError, match="budget error stage is not supported"):
        InterventionBudgetError("unknown")


def test_valid_recipes_per_operation() -> None:
    ablate = InterventionRecipe(
        operation=InterventionOperation.ABLATE,
        targets=("layer-0/expert-1",),
    )
    assert ablate.factor is None and ablate.bias is None and ablate.alternates == ()

    scale = InterventionRecipe(
        operation=InterventionOperation.SCALE,
        targets=("layer-0/expert-1",),
        factor=0.5,
    )
    assert scale.factor == 0.5

    reroute = InterventionRecipe(
        operation=InterventionOperation.REROUTE,
        targets=("layer-0/expert-1", "layer-0/expert-2"),
        alternates=(
            ("layer-0/expert-1", "layer-0/expert-2"),
            ("layer-0/expert-2", "layer-0/expert-1"),
        ),
    )
    assert len(reroute.alternates) == 2

    alter = InterventionRecipe(
        operation=InterventionOperation.ALTER_ROUTER,
        targets=("layer-0/router",),
        bias=-1.25,
    )
    assert alter.bias == -1.25


def test_container_type_violations_raise_type_error() -> None:
    with pytest.raises(TypeError):
        InterventionRecipe(operation="ablate", targets=("a",))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        InterventionRecipe(operation=InterventionOperation.ABLATE, targets=["a"])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        InterventionRecipe(operation=InterventionOperation.ABLATE, targets=(1,))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        InterventionRecipe(
            operation=InterventionOperation.SCALE, targets=("a",), factor=True
        )
    with pytest.raises(TypeError):
        InterventionRecipe(
            operation=InterventionOperation.REROUTE,
            targets=("a",),
            alternates=(["a", "b"],),  # type: ignore[arg-type]
        )


def test_value_violations_raise_value_error() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        InterventionRecipe(operation=InterventionOperation.ABLATE, targets=())
    with pytest.raises(ValueError, match="unique and sorted"):
        InterventionRecipe(
            operation=InterventionOperation.ABLATE, targets=("b", "a")
        )
    with pytest.raises(ValueError, match="unique and sorted"):
        InterventionRecipe(
            operation=InterventionOperation.ABLATE, targets=("a", "a")
        )
    with pytest.raises(ValueError, match="must be finite"):
        InterventionRecipe(
            operation=InterventionOperation.SCALE, targets=("a",), factor=float("nan")
        )
    with pytest.raises(ValueError, match="must not be empty"):
        InterventionRecipe(
            operation=InterventionOperation.REROUTE,
            targets=("a",),
            alternates=(("a", ""),),
        )


def test_parameter_exclusivity_is_enforced_per_operation() -> None:
    with pytest.raises(ValueError, match="scale recipes require"):
        InterventionRecipe(operation=InterventionOperation.SCALE, targets=("a",))
    with pytest.raises(ValueError, match="only valid on scale"):
        InterventionRecipe(
            operation=InterventionOperation.ABLATE, targets=("a",), factor=1.0
        )
    with pytest.raises(ValueError, match="alter_router recipes require"):
        InterventionRecipe(operation=InterventionOperation.ALTER_ROUTER, targets=("a",))
    with pytest.raises(ValueError, match="only valid on alter_router"):
        InterventionRecipe(
            operation=InterventionOperation.ABLATE, targets=("a",), bias=1.0
        )
    with pytest.raises(ValueError, match="cover exactly the target set"):
        InterventionRecipe(
            operation=InterventionOperation.REROUTE,
            targets=("a", "b"),
            alternates=(("a", "b"),),
        )
    with pytest.raises(ValueError, match="differ from their targets"):
        InterventionRecipe(
            operation=InterventionOperation.REROUTE,
            targets=("a",),
            alternates=(("a", "a"),),
        )
    with pytest.raises(ValueError, match="only valid on reroute"):
        InterventionRecipe(
            operation=InterventionOperation.ABLATE, targets=("a",), alternates=(("a", "b"),)
        )


def test_canonical_round_trip_and_rejection() -> None:
    recipe = InterventionRecipe(
        operation=InterventionOperation.SCALE,
        targets=("layer-0/expert-1", "layer-0/expert-3"),
        factor=2.0,
    )
    assert recipe.to_json() == _RECIPE_DOCUMENT
    restored = InterventionRecipe.from_json(_RECIPE_DOCUMENT)
    assert restored == recipe
    assert InterventionRecipe.from_json(recipe.to_json().encode()) == recipe

    wrong_type = _RECIPE_DOCUMENT.replace("intervention_recipe", "something_else")
    with pytest.raises(ValueError, match="not an intervention recipe artifact"):
        InterventionRecipe.from_json(wrong_type)
    with pytest.raises(ValueError, match="not valid JSON"):
        InterventionRecipe.from_json("{not json")
    with pytest.raises(ValueError, match="must be a JSON object"):
        InterventionRecipe.from_json("[1]")
    with pytest.raises(ValueError, match="operation is not supported"):
        InterventionRecipe.from_json(_RECIPE_DOCUMENT.replace('"scale"', '"explode"'))
    with pytest.raises(ValueError, match="missing fields"):
        InterventionRecipe.from_json(
            '{"artifact_type":"moeatlas.intervention_recipe",'
            '"schema_version":"1.0","operation":"ablate"}'
        )


def test_budget_contracts() -> None:
    budget = InterventionBudget(max_targets=4)
    assert budget.to_json() == (
        '{"artifact_type":"moeatlas.intervention_budget",'
        '"max_targets":4,"schema_version":"1.0"}'
    )
    assert InterventionBudget.from_json(budget.to_json()) == budget
    assert recipe_budget_from_json(budget.to_json()) == budget
    assert InterventionBudget() == InterventionBudget(max_targets=64)
    with pytest.raises(TypeError):
        InterventionBudget(max_targets=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        InterventionBudget(max_targets="4")  # type: ignore[arg-type]
    with pytest.raises(InterventionBudgetError, match="strictly positive"):
        InterventionBudget(max_targets=0)
    with pytest.raises(ValueError, match="not an intervention budget artifact"):
        InterventionBudget.from_json('{"max_targets":4}')


def test_fingerprint_binds_lineage_to_exact_content() -> None:
    recipe = InterventionRecipe(
        operation=InterventionOperation.ABLATE, targets=("layer-0/expert-1",)
    )
    twin = InterventionRecipe(
        operation=InterventionOperation.ABLATE, targets=("layer-0/expert-1",)
    )
    changed = InterventionRecipe(
        operation=InterventionOperation.ABLATE, targets=("layer-0/expert-2",)
    )
    assert recipe.fingerprint == twin.fingerprint
    assert recipe.fingerprint != changed.fingerprint
    assert recipe.fingerprint.startswith("sha256:")
    assert len(recipe.fingerprint) == 71

    lineage = InterventionLineage(
        baseline_run_key="run:" + "a" * 64,
        recipe_fingerprint=recipe.fingerprint,
        operation=str(recipe.operation.value),
        targets=recipe.targets,
    )
    assert lineage.recipe_fingerprint == recipe.fingerprint
