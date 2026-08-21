"""Contract tests for expert co-routing graph summaries."""

from __future__ import annotations

import pytest

from moeatlas.analysis import (
    COROUTING_SCHEMA_VERSION,
    CoRoutingError,
    CoRoutingGraph,
    ExpertCoRoutingCounts,
    summarize_co_routing,
)

# ---------------------------------------------------------------------------
# Fixtures


def _counts(
    tables: tuple[tuple[tuple[int, ...], ...], ...],
    *,
    layers: tuple[str, ...] | None = None,
) -> ExpertCoRoutingCounts:
    """One table per layer over that layer's experts, upper triangle only.

    ``tables[layer][i][j]`` counts steps where expert i and j were selected
    together; the diagonal must be zero and the matrix symmetric.
    """

    if layers is None:
        layers = tuple(f"l{index}" for index in range(len(tables)))
    return ExpertCoRoutingCounts(
        layer_keys=layers,
        expert_keys=tuple(
            tuple(f"e{index}" for index in range(len(row))) for row in tables
        ),
        pair_counts=tables,
    )


# ---------------------------------------------------------------------------
# Public surface


def test_surface_is_pinned() -> None:
    assert COROUTING_SCHEMA_VERSION == "1.0"
    assert str(CoRoutingError("contract")) == "co-routing failed at contract"
    with pytest.raises(ValueError):
        CoRoutingError("cancelled")


def test_counts_are_strict() -> None:
    counts = _counts((((0, 3), (3, 0)),))
    assert counts.cell_count == 4
    # Container-type violations are TypeError.
    with pytest.raises(TypeError):
        ExpertCoRoutingCounts(
            layer_keys=["l0"], expert_keys=(("e0", "e1"),), pair_counts=(((0, 3), (3, 0)),)
        )
    with pytest.raises(TypeError):
        _counts(([[0, 3], [3, 0]],))
    with pytest.raises(TypeError):
        _counts((((0, True), (3, 0)),))
    with pytest.raises(TypeError):
        _counts((((0, 3.0), (3.0, 0)),))
    # Shape/value violations are ValueError.
    with pytest.raises(ValueError):
        _counts((), layers=())
    with pytest.raises(ValueError):
        _counts((((0, 3), (3, 0)), ((0, 1), (1, 0))), layers=("b", "a"))
    with pytest.raises(ValueError):
        _counts((((0, 3), (3, 0)),), layers=("l0", "l1"))
    with pytest.raises(ValueError):
        _counts((((0, 3),),))  # ragged rows
    with pytest.raises(ValueError):
        _counts((((1, 3), (3, 0)),))  # nonzero diagonal
    with pytest.raises(ValueError):
        _counts((((0, 3), (2, 0)),))  # asymmetric
    with pytest.raises(ValueError):
        _counts((((0, -1), (-1, 0)),))  # negative


def test_summarize_rejects_wrong_types_and_budgets() -> None:
    counts = _counts((((0, 3), (3, 0)),))
    with pytest.raises(TypeError):
        summarize_co_routing("not counts")
    with pytest.raises(CoRoutingError):
        summarize_co_routing(counts, max_cells=0)
    with pytest.raises(CoRoutingError) as excinfo:
        summarize_co_routing(counts, max_cells=3)
    assert excinfo.value.stage == "budget"
    with pytest.raises(CoRoutingError) as excinfo:
        summarize_co_routing(counts, max_pairs=0)
    assert excinfo.value.stage == "budget"


# ---------------------------------------------------------------------------
# Graph math


def test_single_pair_layer_has_exact_summary() -> None:
    result = summarize_co_routing(_counts((((0, 3), (3, 0)),)))
    assert result.total_pair_selections == (3,)
    assert result.coupled_expert_rows == (2,)
    assert result.top_pairs == ((("e0", "e1", 3, 1.0),),)


def test_top_pairs_are_ranked_and_share_normalized() -> None:
    # e0-e1 coupled 4 times, e0-e2 once; shares 0.8 / 0.2 of total mass 5.
    result = summarize_co_routing(_counts((((0, 4, 1), (4, 0, 0), (1, 0, 0)),)))
    top = result.top_pairs[0]
    assert top[0][:3] == ("e0", "e1", 4)
    assert top[0][3] == pytest.approx(0.8)
    assert top[1][:3] == ("e0", "e2", 1)
    assert top[1][3] == pytest.approx(0.2)
    assert result.total_pair_selections == (5,)
    assert result.coupled_expert_rows == (3,)


def test_max_pairs_bounds_output_deterministically() -> None:
    counts = _counts((((0, 4, 1), (4, 0, 0), (1, 0, 0)),))
    trimmed = summarize_co_routing(counts, max_pairs=1)
    assert len(trimmed.top_pairs[0]) == 1
    assert trimmed.top_pairs[0][0][:3] == ("e0", "e1", 4)


def test_uncoupled_experts_and_empty_layers_stay_explicit() -> None:
    # Zero matrix: no co-selections anywhere.
    result = summarize_co_routing(_counts((((0, 0), (0, 0)),)))
    assert result.total_pair_selections == (0,)
    assert result.coupled_expert_rows == (0,)
    assert result.top_pairs == ((),)


def test_layers_are_independent() -> None:
    result = summarize_co_routing(
        _counts((
            ((0, 2), (2, 0)),
            ((0, 0), (0, 0)),
        ))
    )
    assert result.total_pair_selections == (2, 0)
    assert len(result.top_pairs[0]) == 1
    assert result.top_pairs[1] == ()


# ---------------------------------------------------------------------------
# Serialization


def test_graph_round_trips_through_canonical_json() -> None:
    result = summarize_co_routing(_counts((((0, 3), (3, 0)),)))
    restored = CoRoutingGraph.from_json(result.to_json())
    assert restored == result
    document = result.to_dict()
    assert document["artifact_type"] == "moeatlas.corouting"
    assert document["schema_version"] == COROUTING_SCHEMA_VERSION
    assert document["total_pair_selections"] == [3]
    with pytest.raises(ValueError):
        CoRoutingGraph.from_json('{"artifact_type": "other"}')
    with pytest.raises(ValueError):
        CoRoutingGraph.from_json("[]")
    with pytest.raises(ValueError):
        CoRoutingGraph.from_json("{not json")


def test_serialization_is_deterministic_and_nan_free() -> None:
    first = summarize_co_routing(_counts((((0, 3), (3, 0)),)))
    second = summarize_co_routing(_counts((((0, 3), (3, 0)),)))
    assert first.to_json() == second.to_json()
    assert "NaN" not in first.to_json()
