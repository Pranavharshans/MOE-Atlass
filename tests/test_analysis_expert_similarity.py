"""Contract tests for expert weight/representation similarity summaries."""

from __future__ import annotations

import pytest

from moeatlas.analysis import (
    EXPERT_SIMILARITY_SCHEMA_VERSION,
    ExpertSimilarity,
    ExpertSimilarityError,
    ExpertVectors,
    analyze_expert_similarity,
)

# ---------------------------------------------------------------------------
# Fixtures


def _vectors(
    rows: tuple[tuple[tuple[float, ...], ...], ...],
    *,
    layers: tuple[str, ...] | None = None,
) -> ExpertVectors:
    if layers is None:
        layers = tuple(f"l{index}" for index in range(len(rows)))
    return ExpertVectors(
        layer_keys=layers,
        expert_keys=tuple(
            tuple(f"e{index}" for index in range(len(row))) for row in rows
        ),
        vectors=rows,
    )


# ---------------------------------------------------------------------------
# Public surface


def test_surface_is_pinned() -> None:
    assert EXPERT_SIMILARITY_SCHEMA_VERSION == "1.0"
    assert str(ExpertSimilarityError("contract")) == (
        "expert similarity failed at contract"
    )
    with pytest.raises(ValueError):
        ExpertSimilarityError("cancelled")


def test_vectors_are_strict() -> None:
    vectors = _vectors((((1.0, 0.0), (0.0, 1.0)),))
    assert vectors.cell_count == 2
    # Container-type violations are TypeError.
    with pytest.raises(TypeError):
        ExpertVectors(
            layer_keys=["l0"],
            expert_keys=(("e0", "e1"),),
            vectors=(((1.0, 0.0), (0.0, 1.0)),),
        )
    with pytest.raises(TypeError):
        _vectors(([[1.0, 0.0]],))
    with pytest.raises(TypeError):
        _vectors(((( "1.0", 0.0 ),),))
    # Shape/value violations are ValueError.
    with pytest.raises(ValueError):
        _vectors((), layers=())
    with pytest.raises(ValueError):
        _vectors((((1.0,),), ((0.0,),)), layers=("b", "a"))
    with pytest.raises(ValueError):
        _vectors((((1.0, 0.0),),), layers=("l0", "l1"))
    with pytest.raises(ValueError):
        _vectors((((1.0, 0.0), (0.0,)),))  # unequal vector lengths in one layer
    with pytest.raises(ValueError):
        _vectors((((float("nan"), 0.0),),))
    with pytest.raises(ValueError):
        _vectors((((1.0, float("inf")),),))


def test_analyze_rejects_wrong_types_and_budgets() -> None:
    vectors = _vectors((((1.0, 0.0),),))
    with pytest.raises(TypeError):
        analyze_expert_similarity("not vectors")
    with pytest.raises(ExpertSimilarityError):
        analyze_expert_similarity(vectors, max_cells=0)
    with pytest.raises(ExpertSimilarityError) as excinfo:
        analyze_expert_similarity(vectors, max_cells=0)
    assert excinfo.value.stage == "budget"


# ---------------------------------------------------------------------------
# Similarity math


def test_identical_opposite_and_orthogonal_directions() -> None:
    result = analyze_expert_similarity(
        _vectors((((2.0, 0.0), (3.0, 0.0), (0.0, 5.0), (-1.0, 0.0)),))
    )
    row = result.similarity_rows[0][0]  # expert e0's similarity row
    assert row[0] == pytest.approx(1.0)  # self
    assert row[1] == pytest.approx(1.0)  # same direction
    assert row[2] == pytest.approx(0.0)  # orthogonal
    assert row[3] == pytest.approx(-1.0)  # opposite direction
    assert result.undefined_expert_rows == (0,)


def test_diagonal_is_one_and_matrix_is_symmetric() -> None:
    result = analyze_expert_similarity(
        _vectors((((1.0, 2.0), (3.0, 4.0), (5.0, 6.0)),))
    )
    matrix = result.similarity_rows[0]
    for i in range(3):
        assert matrix[i][i] == pytest.approx(1.0)
        for j in range(3):
            assert matrix[i][j] == pytest.approx(matrix[j][i])


def test_known_angle_has_exact_cosine() -> None:
    # (1, 0) vs (1, 1): cos = 1/sqrt(2).
    result = analyze_expert_similarity(_vectors((((1.0, 0.0), (1.0, 1.0)),)))
    assert result.similarity_rows[0][0][1] == pytest.approx(0.7071067811865476)


def test_zero_norm_experts_are_explicitly_undefined() -> None:
    result = analyze_expert_similarity(
        _vectors((((0.0, 0.0), (1.0, 1.0)),))
    )
    matrix = result.similarity_rows[0]
    assert matrix[0][0] is None
    assert matrix[0][1] is None
    assert matrix[1][0] is None
    assert matrix[1][1] == pytest.approx(1.0)
    assert result.undefined_expert_rows == (1,)


def test_layers_are_independent() -> None:
    result = analyze_expert_similarity(
        _vectors((
            ((1.0, 0.0), (1.0, 0.0)),
            ((1.0, 0.0), (-1.0, 0.0)),
        ))
    )
    assert result.similarity_rows[0][0][1] == pytest.approx(1.0)
    assert result.similarity_rows[1][0][1] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Serialization


def test_result_round_trips_through_canonical_json() -> None:
    result = analyze_expert_similarity(_vectors((((0.0, 0.0), (1.0, 1.0)),)))
    restored = ExpertSimilarity.from_json(result.to_json())
    assert restored == result
    document = result.to_dict()
    assert document["artifact_type"] == "moeatlas.expert_similarity"
    assert document["schema_version"] == EXPERT_SIMILARITY_SCHEMA_VERSION
    assert document["layer_keys"] == ["l0"]
    with pytest.raises(ValueError):
        ExpertSimilarity.from_json('{"artifact_type": "other"}')
    with pytest.raises(ValueError):
        ExpertSimilarity.from_json("[]")
    with pytest.raises(ValueError):
        ExpertSimilarity.from_json("{not json")


def test_serialization_is_deterministic_and_nan_free() -> None:
    first = analyze_expert_similarity(_vectors((((1.0, 2.0), (3.0, 4.0)),)))
    second = analyze_expert_similarity(_vectors((((1.0, 2.0), (3.0, 4.0)),)))
    assert first.to_json() == second.to_json()
    assert "NaN" not in first.to_json()
