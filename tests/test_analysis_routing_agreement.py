"""Contract tests for prompt-level vs rollout-level routing agreement."""

from __future__ import annotations

import pytest

from moeatlas.analysis import (
    ROUTING_AGREEMENT_SCHEMA_VERSION,
    PromptRolloutCounts,
    RoutingAgreement,
    RoutingAgreementError,
    analyze_routing_agreement,
)

# ---------------------------------------------------------------------------
# Fixtures


def _counts(
    prompt: tuple[tuple[int, ...], ...],
    rollout: tuple[tuple[int, ...], ...],
    *,
    layers: tuple[str, ...] | None = None,
    experts: tuple[tuple[str, ...], ...] | None = None,
) -> PromptRolloutCounts:
    if layers is None:
        layers = tuple(f"l{index}" for index in range(len(prompt)))
    if experts is None:
        experts = tuple(
            tuple(f"e{index}" for index in range(len(row))) for row in prompt
        )
    return PromptRolloutCounts(
        layer_keys=layers,
        expert_keys=experts,
        prompt_counts=prompt,
        rollout_counts=rollout,
    )


# ---------------------------------------------------------------------------
# Public surface


def test_surface_is_pinned() -> None:
    assert ROUTING_AGREEMENT_SCHEMA_VERSION == "1.0"
    assert str(RoutingAgreementError("contract")) == (
        "routing agreement failed at contract"
    )
    with pytest.raises(ValueError):
        RoutingAgreementError("cancelled")


def test_counts_shape_is_strict() -> None:
    counts = _counts(((4, 4),), ((2, 6),))
    assert counts.cell_count == 2
    # Container-type violations are TypeError.
    with pytest.raises(TypeError):
        PromptRolloutCounts(
            layer_keys=["l0"],
            expert_keys=(("e0", "e1"),),
            prompt_counts=((4, 4),),
            rollout_counts=((2, 6),),
        )
    with pytest.raises(TypeError):
        _counts(((4.0, 4.0),), ((2, 6),))
    with pytest.raises(TypeError):
        _counts(((True, 4),), ((2, 6),))
    with pytest.raises(TypeError):
        PromptRolloutCounts(
            layer_keys=("l0",),
            expert_keys=("e0", "e1"),
            prompt_counts=((4, 4),),
            rollout_counts=((2, 6),),
        )
    # Length/shape/range violations are ValueError.
    with pytest.raises(ValueError):
        _counts((), ())
    with pytest.raises(ValueError):
        _counts(((4, 4), (0, 8)), ((2, 6),))
    with pytest.raises(ValueError):
        _counts(
            ((4, 4), (0, 8)),
            ((2, 6), (1, 1)),
            layers=("b", "a"),
        )
    with pytest.raises(ValueError):
        _counts(((4, 4),), ((2, 6),), experts=(("e1", "e0"),))
    with pytest.raises(ValueError):
        _counts(((4, 4),), ((2, 6),), experts=(("e0", "e0"),))
    with pytest.raises(ValueError):
        _counts(((4, -1),), ((2, 6),))
    # A zero phase total leaves one conditional undefined; an unused expert
    # inside a positive-total phase is fine and still analyzes.
    assert len(analyze_routing_agreement(_counts(((4, 0),), ((2, 6),))).agreement_rows) == 1
    with pytest.raises(ValueError):
        _counts(((4, 4),), ((0, 0),))
    with pytest.raises(ValueError):
        _counts(((0, 0),), ((2, 6),))


def test_analyze_rejects_wrong_types_and_budgets() -> None:
    with pytest.raises(TypeError):
        analyze_routing_agreement("not counts")
    with pytest.raises(RoutingAgreementError):
        analyze_routing_agreement(_counts(((4, 4),), ((2, 6),)), max_cells=0)
    with pytest.raises(RoutingAgreementError) as excinfo:
        analyze_routing_agreement(_counts(((4, 4),), ((2, 6),)), max_cells=1)
    assert excinfo.value.stage == "budget"


# ---------------------------------------------------------------------------
# Agreement math


def test_identical_phases_agree_exactly() -> None:
    result = analyze_routing_agreement(_counts(((6, 2),), ((6, 2),)))
    assert result.js_divergence_rows == (0.0,)
    assert result.agreement_rows == (1.0,)
    assert result.tv_distance_rows == (0.0,)


def test_disjoint_supports_maximally_disagree() -> None:
    result = analyze_routing_agreement(_counts(((8, 0),), ((0, 8),)))
    assert result.js_divergence_rows == (1.0,)
    assert result.agreement_rows == (0.0,)
    assert result.tv_distance_rows == (1.0,)


def test_mixed_case_has_exact_values() -> None:
    # p = (1, 0), q = (1/2, 1/2): JSD2 = H2(3/4, 1/4) - 1/2.
    result = analyze_routing_agreement(_counts(((1, 0),), ((1, 1),)))
    expected_jsd = 0.8112781244591328 - 0.5
    assert result.js_divergence_rows[0] == pytest.approx(expected_jsd)
    assert result.agreement_rows[0] == pytest.approx(1.0 - expected_jsd)
    assert result.tv_distance_rows == (0.5,)


def test_layers_are_independent_and_totals_are_scale_free() -> None:
    counts = _counts(
        ((8, 0), (3, 3)),
        ((0, 8), (30, 30)),
    )
    result = analyze_routing_agreement(counts)
    assert result.agreement_rows[0] == pytest.approx(0.0)
    assert result.agreement_rows[1] == pytest.approx(1.0)
    # Doubling both phases leaves the distributions, hence agreement, fixed.
    doubled = analyze_routing_agreement(
        _counts(((16, 0), (6, 6)), ((0, 16), (60, 60)))
    )
    assert doubled.agreement_rows == result.agreement_rows


def test_single_expert_layers_agree_trivially() -> None:
    counts = _counts(((5,),), ((9,),), experts=(("solo",),))
    result = analyze_routing_agreement(counts)
    assert result.agreement_rows == (1.0,)


# ---------------------------------------------------------------------------
# Serialization


def test_result_round_trips_through_canonical_json() -> None:
    counts = _counts(((6, 2), (1, 0)), ((6, 2), (0, 1)))
    result = analyze_routing_agreement(counts)
    restored = RoutingAgreement.from_json(result.to_json())
    assert restored == result
    document = result.to_dict()
    assert document["artifact_type"] == "moeatlas.routing_agreement"
    assert document["schema_version"] == ROUTING_AGREEMENT_SCHEMA_VERSION
    assert document["layer_keys"] == ["l0", "l1"]
    with pytest.raises(ValueError):
        RoutingAgreement.from_json('{"artifact_type": "other"}')
    with pytest.raises(ValueError):
        RoutingAgreement.from_json("[]")
    with pytest.raises(ValueError):
        RoutingAgreement.from_json("{not json")


def test_serialization_is_deterministic_and_nan_free() -> None:
    first = analyze_routing_agreement(_counts(((6, 2),), ((2, 6),)))
    second = analyze_routing_agreement(_counts(((6, 2),), ((2, 6),)))
    assert first.to_json() == second.to_json()
    assert "NaN" not in first.to_json()
