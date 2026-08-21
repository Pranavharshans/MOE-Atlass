"""Contract tests for task-association analyses over expert count tables."""

from __future__ import annotations

import math

import pytest

from moeatlas.analysis import (
    TASK_ASSOCIATION_SCHEMA_VERSION,
    TaskAssociationError,
    TaskAssociationMatrix,
    TaskExpertCounts,
    analyze_task_association,
)

# ---------------------------------------------------------------------------
# Fixtures


def _counts(
    *,
    layer_keys: tuple[str, ...] = ("l0",),
    task_keys: tuple[str, ...] = ("math", "prose"),
    expert_keys: tuple[tuple[str, ...], ...] = (("e0", "e1"),),
    counts: tuple[tuple[tuple[int, ...], ...], ...] | None = None,
) -> TaskExpertCounts:
    if counts is None:
        # math routes only e0; prose routes only e1 (perfect specialization,
        # equal task totals so every baseline probability is an exact half).
        counts = (((10, 0), (0, 10)),)
    return TaskExpertCounts(
        layer_keys=layer_keys,
        task_keys=task_keys,
        expert_keys=expert_keys,
        counts=counts,
    )


def _uniform_counts() -> TaskExpertCounts:
    return _counts(counts=(((4, 4), (4, 4)),))


# ---------------------------------------------------------------------------
# Public surface and error contract


def test_surface_is_pinned() -> None:
    assert TASK_ASSOCIATION_SCHEMA_VERSION == "1.0"
    assert str(TaskAssociationError("budget")) == "task association failed at budget"
    with pytest.raises(ValueError):
        TaskAssociationError("cancelled")


def test_count_table_shape_is_strict() -> None:
    with pytest.raises(TypeError):
        TaskExpertCounts(layer_keys=["l0"], task_keys=("t",), expert_keys=(("e",),),
                         counts=(((1,),),))
    with pytest.raises(TypeError):
        _counts(task_keys=("math", 7))
    with pytest.raises(ValueError):
        _counts(layer_keys=())
    with pytest.raises(ValueError):
        _counts(task_keys=())
    with pytest.raises(ValueError):
        _counts(expert_keys=((),))
    with pytest.raises(ValueError):
        _counts(layer_keys=("l1", "l0"))
    with pytest.raises(ValueError):
        _counts(task_keys=("prose", "math"))
    with pytest.raises(ValueError):
        _counts(expert_keys=(("e1", "e0"),))
    with pytest.raises(ValueError):
        _counts(layer_keys=("l0", "l0"))
    with pytest.raises(ValueError):
        _counts(task_keys=("math", "math"))
    with pytest.raises(ValueError):
        _counts(expert_keys=(("e0", "e0"),))
    with pytest.raises(ValueError):
        _counts(counts=(((10, 0),),))  # ragged: one task row, two declared
    with pytest.raises(ValueError):
        _counts(counts=(((-1, 0), (0, 8)),))
    with pytest.raises(TypeError):
        _counts(counts=(((True, 0), (0, 8)),))
    with pytest.raises(ValueError):
        _counts(counts=((((10, 0, 1), (0, 8, 2)),),))  # wider than experts
    with pytest.raises(ValueError):
        _counts(counts=((((0, 0), (0, 0)),),))  # a task with no evidence


# ---------------------------------------------------------------------------
# Metric correctness on known tables


def test_perfect_specialization_yields_exact_metrics() -> None:
    matrix = analyze_task_association(_counts())
    assert matrix.schema_version == TASK_ASSOCIATION_SCHEMA_VERSION
    # e1 is used (by prose), so math/e1 enrichment is a defined 0.0; only its
    # PMI is undefined, because the conditional probability is zero.
    assert matrix.enrichment_rows == (
        ("l0", "math", "e0", 2.0),
        ("l0", "math", "e1", 0.0),
        ("l0", "prose", "e0", 0.0),
        ("l0", "prose", "e1", 2.0),
    )
    assert matrix.pmi_rows == (
        ("l0", "math", "e0", 1.0),
        ("l0", "math", "e1", None),
        ("l0", "prose", "e0", None),
        ("l0", "prose", "e1", 1.0),
    )
    assert math.isclose(matrix.mutual_information_rows[0][1], 1.0)
    assert matrix.separability_rows == (("l0", 1.0),)
    assert matrix.exclusivity_rows == (
        ("l0", "e0", 1.0, 1),
        ("l0", "e1", 1.0, 1),
    )


def test_uniform_sharing_yields_neutral_metrics() -> None:
    matrix = analyze_task_association(_uniform_counts())
    assert matrix.enrichment_rows == (
        ("l0", "math", "e0", 1.0),
        ("l0", "math", "e1", 1.0),
        ("l0", "prose", "e0", 1.0),
        ("l0", "prose", "e1", 1.0),
    )
    assert all(row[3] == 0.0 for row in matrix.pmi_rows)
    assert matrix.mutual_information_rows == (("l0", 0.0),)
    assert matrix.specific_mi_rows == (("l0", "math", 0.0), ("l0", "prose", 0.0))
    assert matrix.separability_rows == (("l0", 0.0),)
    assert all(row[2] == 0.5 and row[3] == 2 for row in matrix.exclusivity_rows)


def test_unused_expert_is_explicit_undefined_evidence() -> None:
    counts = _counts(counts=(((6, 0), (4, 0)),))
    matrix = analyze_task_association(counts)
    by_cell = {(r[1], r[2]): r[3] for r in matrix.enrichment_rows}
    assert by_cell["math", "e1"] is None
    assert by_cell["prose", "e1"] is None
    pmi = {(r[1], r[2]): r[3] for r in matrix.pmi_rows}
    assert pmi["math", "e1"] is None
    exclusivity = {r[1]: (r[2], r[3]) for r in matrix.exclusivity_rows}
    assert exclusivity["e1"] == (None, 0)
    # e0 serves both tasks 6:4, so it is shared rather than exclusive.
    assert exclusivity["e0"] == (0.6, 2)


def test_task_missing_an_expert_has_defined_enrichment_but_no_pmi() -> None:
    counts = _counts(counts=(((6, 0), (4, 4)),))
    matrix = analyze_task_association(counts)
    enrichment = {(r[1], r[2]): r[3] for r in matrix.enrichment_rows}
    # P(e0|math)=1, P(e0)=10/14 → enrichment 1.4; e1 unused by math.
    assert math.isclose(enrichment["math", "e0"], 1.4)
    assert enrichment["math", "e1"] == 0.0
    pmi = {(r[1], r[2]): r[3] for r in matrix.pmi_rows}
    assert pmi["math", "e1"] is None


def test_mutual_information_and_specific_mi_are_consistent() -> None:
    counts = _counts(counts=(((8, 2), (2, 8)),))
    matrix = analyze_task_association(counts)
    total_mi = matrix.mutual_information_rows[0][1]
    weighted = sum(
        (row[2] * matrix.assignment_totals[0][matrix.task_keys.index(row[1])] / 20)
        for row in matrix.specific_mi_rows
    )
    assert math.isclose(total_mi, weighted, rel_tol=1e-12)
    assert 0 < total_mi < 1


def test_separability_is_mean_pairwise_js_divergence() -> None:
    counts = _counts(counts=(((8, 2), (2, 8)),))
    matrix = analyze_task_association(counts)
    value = matrix.separability_rows[0][1]
    assert 0 < value < 1

    # a=(1,0), b=(0,1), c=(0.5,0.5): JS(a,b)=1 bit, JS(a,c)=JS(b,c)=H(3/4)-1/2.
    three = _counts(
        task_keys=("a", "b", "c"),
        expert_keys=(("e0", "e1"),),
        counts=(((6, 0), (0, 6), (3, 3)),),
    )
    matrix_three = analyze_task_association(three)
    h_three_quarter = -(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25))
    expected = (1.0 + 2 * (h_three_quarter - 0.5)) / 3
    assert math.isclose(matrix_three.separability_rows[0][1], expected, rel_tol=1e-12)


def test_single_task_layer_has_undefined_separability() -> None:
    counts = _counts(task_keys=("only",), counts=(((6, 4),),))
    matrix = analyze_task_association(counts)
    assert matrix.separability_rows == (("l0", None),)


def test_layers_are_independent() -> None:
    counts = TaskExpertCounts(
        layer_keys=("l0", "l1"),
        task_keys=("math", "prose"),
        expert_keys=(("e0", "e1"), ("e0", "e1")),
        counts=(
            ((10, 0), (0, 10)),
            ((5, 5), (4, 4)),
        ),
    )
    matrix = analyze_task_association(counts)
    assert matrix.mutual_information_rows[0][1] == 1.0
    assert matrix.mutual_information_rows[1][1] == 0.0
    assert matrix.separability_rows == (("l0", 1.0), ("l1", 0.0))


# ---------------------------------------------------------------------------
# Budgets, determinism, serialization


def test_budget_and_strictness_are_exact() -> None:
    with pytest.raises(TaskAssociationError) as budget:
        analyze_task_association(_counts(), max_cells=3)
    assert budget.value.stage == "budget"
    with pytest.raises(TypeError):
        analyze_task_association("not-counts")
    with pytest.raises(TypeError):
        analyze_task_association(_counts(), max_cells=True)
    with pytest.raises(TaskAssociationError) as zero:
        analyze_task_association(_counts(), max_cells=0)
    assert zero.value.stage == "budget"


def test_analysis_is_deterministic_byte_for_byte(tmp_path) -> None:
    first = analyze_task_association(_counts())
    second = analyze_task_association(_counts())
    assert first == second
    assert first.to_json() == second.to_json()


def test_matrix_round_trips_through_canonical_json() -> None:
    matrix = analyze_task_association(_counts())
    restored = TaskAssociationMatrix.from_json(matrix.to_json())
    assert restored == matrix
    document = matrix.to_dict()
    assert document["artifact_type"] == "moeatlas.task_association"
    with pytest.raises(ValueError):
        TaskAssociationMatrix.from_json('{"artifact_type": "other"}')
    with pytest.raises(ValueError):
        TaskAssociationMatrix.from_json("{not json")
    with pytest.raises(ValueError):
        TaskAssociationMatrix.from_json("[]")


def test_serialized_document_carries_null_not_nan() -> None:
    matrix = analyze_task_association(_counts())
    assert "NaN" not in matrix.to_json()
    assert "null" in matrix.to_json()
