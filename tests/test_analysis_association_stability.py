"""Contract tests for cross-run stability of expert-task association."""

from __future__ import annotations

import pytest

from moeatlas.analysis import (
    ASSOCIATION_STABILITY_SCHEMA_VERSION,
    AssociationStabilityError,
    TaskExpertCounts,
    analyze_association_stability,
)

# ---------------------------------------------------------------------------
# Fixtures


def _counts(
    counts: tuple[tuple[tuple[int, ...], ...], ...],
) -> TaskExpertCounts:
    """One layer with two tasks and two experts, from a nested table.

    ``counts[layer][task]`` is the per-expert row; layers, tasks, and experts
    get canonical keys in order.
    """

    return TaskExpertCounts(
        layer_keys=tuple(f"l{index}" for index in range(len(counts))),
        task_keys=("math", "prose"),
        expert_keys=tuple(
            tuple(f"e{index}" for index in range(len(layer[0]))) for layer in counts
        ),
        counts=counts,
    )


# ---------------------------------------------------------------------------
# Public surface


def test_surface_is_pinned() -> None:
    assert ASSOCIATION_STABILITY_SCHEMA_VERSION == "1.0"
    assert str(AssociationStabilityError("contract")) == (
        "association stability failed at contract"
    )
    with pytest.raises(ValueError):
        AssociationStabilityError("cancelled")


def test_analyze_rejects_mismatched_topologies() -> None:
    first = _counts((((10, 0), (0, 10)),))
    second = _counts((((6, 4), (4, 6)),))
    analyze_association_stability(first, second)
    with pytest.raises(TypeError):
        analyze_association_stability("not counts", second)
    with pytest.raises(AssociationStabilityError):
        analyze_association_stability(first, second, max_cells=0)
    with pytest.raises(AssociationStabilityError) as excinfo:
        analyze_association_stability(first, second, max_cells=3)
    assert excinfo.value.stage == "budget"
    # Different task vocabulary cannot be compared cell by cell.
    renamed = TaskExpertCounts(
        layer_keys=first.layer_keys,
        task_keys=("code", "prose"),
        expert_keys=first.expert_keys,
        counts=first.counts,
    )
    with pytest.raises(AssociationStabilityError):
        analyze_association_stability(renamed, second)
    # Different per-layer expert universes cannot be compared either.
    wider = TaskExpertCounts(
        layer_keys=first.layer_keys,
        task_keys=first.task_keys,
        expert_keys=(("e0", "e1", "e2"),),
        counts=((((10, 0, 0), (0, 10, 0)),)),
    )
    with pytest.raises(AssociationStabilityError):
        analyze_association_stability(wider, second)


# ---------------------------------------------------------------------------
# Stability math


def test_identical_runs_are_fully_stable() -> None:
    counts = _counts((((10, 0), (0, 10)),))
    result = analyze_association_stability(counts, counts)
    assert result.js_divergence_rows == ((0.0, 0.0),)
    assert result.agreement_rows == ((1.0, 1.0),)
    assert result.mean_agreement_rows == (1.0,)


def test_disjoint_routing_is_fully_unstable() -> None:
    first = _counts((((10, 0), (10, 0)),))
    second = _counts((((0, 10), (0, 10)),))
    result = analyze_association_stability(first, second)
    assert result.js_divergence_rows == ((1.0, 1.0),)
    assert result.agreement_rows == ((0.0, 0.0),)
    assert result.mean_agreement_rows == (0.0,)


def test_mixed_case_has_exact_values() -> None:
    # math: run A routes e0 only; run B splits evenly -> JSD2 = H2(3/4)-1/2.
    # prose: identical rows stay at agreement 1.0.
    first = _counts((((1, 0), (2, 2)),))
    second = _counts((((1, 1), (2, 2)),))
    result = analyze_association_stability(first, second)
    expected_jsd = 0.8112781244591328 - 0.5
    assert result.js_divergence_rows[0][0] == pytest.approx(expected_jsd)
    assert result.agreement_rows[0][0] == pytest.approx(1.0 - expected_jsd)
    assert result.agreement_rows[0][1] == 1.0
    assert result.mean_agreement_rows[0] == pytest.approx(1.0 - expected_jsd / 2)


def test_layers_are_independent_and_totals_are_scale_free() -> None:
    first = _counts((((8, 0), (3, 3)), ((5, 5), (1, 1))))
    second = _counts((((0, 8), (3, 3)), ((50, 50), (1, 1))))
    result = analyze_association_stability(first, second)
    assert result.agreement_rows[0][0] == pytest.approx(0.0)
    assert result.agreement_rows[0][1] == pytest.approx(1.0)
    assert result.agreement_rows[1] == (pytest.approx(1.0), pytest.approx(1.0))


# ---------------------------------------------------------------------------
# Serialization


def test_result_round_trips_through_canonical_json() -> None:
    first = _counts((((6, 2), (1, 1)),))
    second = _counts((((2, 6), (1, 1)),))
    result = analyze_association_stability(first, second)
    restored = type(result).from_json(result.to_json())
    assert restored == result
    document = result.to_dict()
    assert document["artifact_type"] == "moeatlas.association_stability"
    assert document["schema_version"] == ASSOCIATION_STABILITY_SCHEMA_VERSION
    assert document["layer_keys"] == ["l0"]
    assert document["task_keys"] == ["math", "prose"]
    with pytest.raises(ValueError):
        type(result).from_json('{"artifact_type": "other"}')
    with pytest.raises(ValueError):
        type(result).from_json("[]")
    with pytest.raises(ValueError):
        type(result).from_json("{not json")


def test_serialization_is_deterministic_and_nan_free() -> None:
    first = _counts((((6, 2), (1, 1)),))
    second = _counts((((2, 6), (1, 1)),))
    left = analyze_association_stability(first, second)
    right = analyze_association_stability(first, second)
    assert left.to_json() == right.to_json()
    assert "NaN" not in left.to_json()
