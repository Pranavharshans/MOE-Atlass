"""Contract tests for the route-churn routing-stability metric."""

from __future__ import annotations

import pytest

from moeatlas.analysis import (
    ROUTE_CHURN_SCHEMA_VERSION,
    RouteChurnError,
    RouteChurnSequences,
    RouteChurnSummary,
    analyze_route_churn,
)

# ---------------------------------------------------------------------------
# Fixtures


def _sequences(
    steps: tuple[tuple[tuple[str, ...], ...], ...],
    *,
    layers: tuple[str, ...] | None = None,
) -> RouteChurnSequences:
    if layers is None:
        layers = tuple(f"l{index}" for index in range(len(steps)))
    return RouteChurnSequences(layer_keys=layers, step_experts=steps)


# ---------------------------------------------------------------------------
# Public surface


def test_surface_is_pinned() -> None:
    assert ROUTE_CHURN_SCHEMA_VERSION == "1.0"
    assert str(RouteChurnError("contract")) == "route churn failed at contract"
    with pytest.raises(ValueError):
        RouteChurnError("cancelled")


def test_sequences_are_strict() -> None:
    sequences = _sequences(((("e0",), ("e0", "e1")),))
    assert sequences.step_count == 2
    # Container-type violations are TypeError.
    with pytest.raises(TypeError):
        RouteChurnSequences(layer_keys=["l0"], step_experts=((("e0",),),))
    with pytest.raises(TypeError):
        RouteChurnSequences(layer_keys=("l0",), step_experts=[(("e0",),)])
    with pytest.raises(TypeError):
        _sequences(((("e0", 7),),))
    with pytest.raises(TypeError):
        _sequences((("e0",),))
    # Value violations are ValueError.
    with pytest.raises(ValueError):
        _sequences((), layers=())
    with pytest.raises(ValueError):
        _sequences(((("e0",),), (("e1",),)), layers=("b", "a"))
    with pytest.raises(ValueError):
        _sequences(((("e0",),),), layers=("l0", "l1"))
    with pytest.raises(ValueError):
        _sequences(((("e0", "e0"),),))  # duplicate expert within one step


def test_analyze_rejects_wrong_types_and_budgets() -> None:
    with pytest.raises(TypeError):
        analyze_route_churn("not sequences")
    with pytest.raises(RouteChurnError) as excinfo:
        analyze_route_churn(_sequences(((("e0",),),)), max_steps=0)
    assert excinfo.value.stage == "budget"
    with pytest.raises(RouteChurnError) as excinfo:
        analyze_route_churn(_sequences(((("e0",), ("e1",)),)), max_steps=1)
    assert excinfo.value.stage == "budget"


# ---------------------------------------------------------------------------
# Churn math


def test_identical_adjacent_steps_are_stable() -> None:
    result = analyze_route_churn(_sequences(((("e0", "e1"), ("e0", "e1")),)))
    assert result.churn_rate_rows == (0.0,)
    assert result.mean_jaccard_rows == (0.0,)
    assert result.pair_rows == (1,)


def test_disjoint_adjacent_steps_maximally_churn() -> None:
    result = analyze_route_churn(_sequences(((("e0",), ("e1",)),)))
    assert result.churn_rate_rows == (1.0,)
    assert result.mean_jaccard_rows == (1.0,)
    assert result.pair_rows == (1,)


def test_partial_overlap_has_exact_jaccard_distance() -> None:
    # {e0, e1} -> {e1, e2}: intersection 1, union 3 -> distance 2/3.
    result = analyze_route_churn(_sequences(((("e0", "e1"), ("e2", "e1")),)))
    assert result.churn_rate_rows == (1.0,)
    assert result.mean_jaccard_rows == (pytest.approx(2.0 / 3.0),)


def test_empty_steps_follow_the_documented_conventions() -> None:
    # empty -> empty is no change; empty -> nonempty is full change.
    result = analyze_route_churn(_sequences((((), (), ("e0",)),)))
    assert result.churn_rate_rows == (pytest.approx(1.0 / 2.0),)
    assert result.mean_jaccard_rows == (pytest.approx(0.5),)
    assert result.pair_rows == (2,)


def test_single_step_layers_have_undefined_churn() -> None:
    result = analyze_route_churn(_sequences(((("e0",),),)))
    assert result.churn_rate_rows == (None,)
    assert result.mean_jaccard_rows == (None,)
    assert result.pair_rows == (0,)


def test_layers_are_independent_and_order_within_steps_is_irrelevant() -> None:
    result = analyze_route_churn(
        _sequences((
            (("e0",), ("e1",)),
            (("e0", "e1"), ("e1", "e0")),
        ))
    )
    assert result.churn_rate_rows[0] == pytest.approx(1.0)
    assert result.churn_rate_rows[1] == pytest.approx(0.0)
    assert result.mean_jaccard_rows[1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Serialization


def test_summary_round_trips_through_canonical_json() -> None:
    result = analyze_route_churn(
        _sequences(((("e0",), ("e1",)), (("solo",), ("solo",))))
    )
    restored = RouteChurnSummary.from_json(result.to_json())
    assert restored == result
    document = result.to_dict()
    assert document["artifact_type"] == "moeatlas.route_churn"
    assert document["schema_version"] == ROUTE_CHURN_SCHEMA_VERSION
    assert document["churn_rate_rows"] == [1.0, 0.0]
    with pytest.raises(ValueError):
        RouteChurnSummary.from_json('{"artifact_type": "other"}')
    with pytest.raises(ValueError):
        RouteChurnSummary.from_json("[]")
    with pytest.raises(ValueError):
        RouteChurnSummary.from_json("{not json")


def test_serialization_is_deterministic_and_nan_free() -> None:
    first = analyze_route_churn(_sequences(((("e0",), ("e1",)),)))
    second = analyze_route_churn(_sequences(((("e0",), ("e1",)),)))
    assert first.to_json() == second.to_json()
    assert "NaN" not in first.to_json()
