"""Contract tests for the router-margin routing-confidence metric."""

from __future__ import annotations

import pytest

from moeatlas.analysis import (
    ROUTER_MARGIN_SCHEMA_VERSION,
    RouterMarginError,
    RouterMarginSamples,
    RouterMarginSummary,
    analyze_router_margin,
)

# ---------------------------------------------------------------------------
# Fixtures


def _samples(
    token_scores: tuple[tuple[tuple[float, ...], ...], ...],
    *,
    layers: tuple[str, ...] | None = None,
) -> RouterMarginSamples:
    if layers is None:
        layers = tuple(f"l{index}" for index in range(len(token_scores)))
    return RouterMarginSamples(layer_keys=layers, token_scores=token_scores)


# ---------------------------------------------------------------------------
# Public surface


def test_surface_is_pinned() -> None:
    assert ROUTER_MARGIN_SCHEMA_VERSION == "1.0"
    assert str(RouterMarginError("contract")) == "router margin failed at contract"
    with pytest.raises(ValueError):
        RouterMarginError("cancelled")


def test_samples_are_strict() -> None:
    samples = _samples((((3.0, 1.0), (2.0,)),))
    assert samples.cell_count == 2
    # Container-type violations are TypeError.
    with pytest.raises(TypeError):
        RouterMarginSamples(layer_keys=["l0"], token_scores=(((3.0, 1.0),),))
    with pytest.raises(TypeError):
        RouterMarginSamples(layer_keys=("l0",), token_scores=[((3.0, 1.0),)])
    with pytest.raises(TypeError):
        _samples(((("3.0", 1.0),),))
    with pytest.raises(TypeError):
        _samples((((3.0,), 7),))
    # Value violations are ValueError.
    with pytest.raises(ValueError):
        _samples((), layers=())
    with pytest.raises(ValueError):
        _samples((((3.0, 1.0),), ((2.0,),)), layers=("b", "a"))
    with pytest.raises(ValueError):
        _samples((((3.0, 1.0),),), layers=("l0", "l1"))
    with pytest.raises(ValueError):
        _samples((((float("nan"), 1.0),),))
    with pytest.raises(ValueError):
        _samples((((3.0, float("inf")),),))


def test_analyze_rejects_wrong_types_and_budgets() -> None:
    with pytest.raises(TypeError):
        analyze_router_margin("not samples")
    with pytest.raises(RouterMarginError) as excinfo:
        analyze_router_margin(_samples((((3.0, 1.0),),)), max_tokens=0)
    assert excinfo.value.stage == "budget"
    with pytest.raises(RouterMarginError) as excinfo:
        analyze_router_margin(_samples((((3.0, 1.0),),)), max_tokens=0)
    assert excinfo.value.stage == "budget"


# ---------------------------------------------------------------------------
# Margin math


def test_margin_is_top_two_score_difference() -> None:
    result = analyze_router_margin(_samples((((4.0, 1.5),),)))
    assert result.mean_margin_rows == (pytest.approx(2.5),)
    assert result.margin_token_rows == (1,)
    assert result.token_rows == (1,)


def test_undefined_margins_are_explicit() -> None:
    # Fewer than two scored ranks leaves that token's margin undefined.
    result = analyze_router_margin(_samples((((2.0,),),)))
    assert result.mean_margin_rows == (None,)
    assert result.margin_token_rows == (0,)
    assert result.token_rows == (1,)


def test_mean_skips_undefined_tokens_and_counts_are_exact() -> None:
    result = analyze_router_margin(
        _samples((((4.0, 1.0), (9.0,), (1.0, 0.0)),))
    )
    # Defined margins: 3.0 and 1.0; the rank-less token is skipped.
    assert result.mean_margin_rows == (pytest.approx(2.0),)
    assert result.margin_token_rows == (2,)
    assert result.token_rows == (3,)


def test_layers_are_independent_and_negative_scores_are_legal() -> None:
    result = analyze_router_margin(_samples((((-1.0, -3.0),), ((0.5, 0.25),))))
    assert result.mean_margin_rows[0] == pytest.approx(2.0)
    assert result.mean_margin_rows[1] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Serialization


def test_summary_round_trips_through_canonical_json() -> None:
    result = analyze_router_margin(_samples((((4.0, 1.0), (9.0,)),)))
    restored = RouterMarginSummary.from_json(result.to_json())
    assert restored == result
    document = result.to_dict()
    assert document["artifact_type"] == "moeatlas.router_margin"
    assert document["schema_version"] == ROUTER_MARGIN_SCHEMA_VERSION
    assert document["mean_margin_rows"] == [3.0]
    with pytest.raises(ValueError):
        RouterMarginSummary.from_json('{"artifact_type": "other"}')
    with pytest.raises(ValueError):
        RouterMarginSummary.from_json("[]")
    with pytest.raises(ValueError):
        RouterMarginSummary.from_json("{not json")


def test_serialization_is_deterministic_and_nan_free() -> None:
    first = analyze_router_margin(_samples((((4.0, 1.0),),)))
    second = analyze_router_margin(_samples((((4.0, 1.0),),)))
    assert first.to_json() == second.to_json()
    assert "NaN" not in first.to_json()
