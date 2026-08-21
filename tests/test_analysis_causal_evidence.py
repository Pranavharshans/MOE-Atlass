"""Contract tests for paired causal-effect summaries (PRD §11.4)."""

from __future__ import annotations

import pytest

from moeatlas.analysis import (
    CAUSAL_EVIDENCE_SCHEMA_VERSION,
    CausalEvidence,
    CausalEvidenceError,
    CausalPair,
    analyze_causal_evidence,
)


def _pair(
    label: str,
    replication: int,
    baseline_value: float,
    intervened_value: float,
) -> CausalPair:
    return CausalPair(
        label=label,
        replication=replication,
        baseline_value=baseline_value,
        intervened_value=intervened_value,
    )


def test_surface_is_pinned() -> None:
    assert CAUSAL_EVIDENCE_SCHEMA_VERSION == "1.0"
    with pytest.raises(ValueError, match="stage is not supported"):
        CausalEvidenceError("unknown")
    pair = _pair("mass", 0, 1.0, 2.0)
    assert pair.label == "mass"
    assert pair.replication == 0


def test_single_label_effect_summary() -> None:
    pairs = (
        _pair("expert/output_mass", 0, 10.0, 12.0),
        _pair("expert/output_mass", 1, 20.0, 23.0),
    )
    result = analyze_causal_evidence(pairs)
    index = result.labels.index("expert/output_mass")
    assert result.replication_counts[index] == 2
    assert result.mean_baseline[index] == pytest.approx(15.0)
    assert result.mean_intervened[index] == pytest.approx(17.5)
    assert result.absolute_effects[index] == pytest.approx(2.5)
    assert result.relative_effects[index] == pytest.approx(2.5 / 15.0)
    assert result.direction_consistency[index] == 1.0
    assert result.stable_labels[index] is True
    assert result.zero_effect_labels[index] is False


def test_labels_are_sorted_and_independent() -> None:
    pairs = (
        _pair("b/metric", 0, 1.0, 3.0),
        _pair("a/metric", 0, 5.0, 4.0),
        _pair("c/metric", 0, 2.0, 2.0),
    )
    result = analyze_causal_evidence(pairs)
    assert result.labels == ("a/metric", "b/metric", "c/metric")
    a = result.labels.index("a/metric")
    c = result.labels.index("c/metric")
    assert result.absolute_effects[a] < 0
    assert result.absolute_effects[c] == 0.0
    assert result.direction_consistency[a] == 1.0
    assert result.direction_consistency[c] is None
    assert result.stable_labels[a] is True
    assert result.stable_labels[c] is False
    assert result.zero_effect_labels[c] is True


def test_relative_effect_is_null_on_zero_baseline() -> None:
    pairs = (_pair("m", 0, 0.0, 4.0),)
    result = analyze_causal_evidence(pairs)
    assert result.relative_effects[0] is None
    assert result.absolute_effects[0] == 4.0


def test_mixed_directions_report_partial_consistency() -> None:
    pairs = (
        _pair("m", 0, 0.0, 5.0),
        _pair("m", 1, 0.0, -1.0),
        _pair("m", 2, 0.0, 6.0),
        _pair("m", 3, 0.0, -2.0),
    )
    result = analyze_causal_evidence(pairs)
    # Mean effect is +2.0; two of four replication effects are positive.
    assert result.direction_consistency[0] == 0.5
    assert result.stable_labels[0] is False
    assert result.zero_effect_labels[0] is False


def test_duplicate_replications_are_rejected() -> None:
    pairs = (
        _pair("m", 0, 1.0, 2.0),
        _pair("m", 0, 3.0, 4.0),
    )
    with pytest.raises(CausalEvidenceError) as caught:
        analyze_causal_evidence(pairs)
    assert caught.value.stage == "contract"


def test_container_type_violations_raise_type_error() -> None:
    good = _pair("m", 0, 1.0, 2.0)
    with pytest.raises(TypeError):
        analyze_causal_evidence([good])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        analyze_causal_evidence((good, "not-a-pair"))  # type: ignore[list-item]
    with pytest.raises(TypeError):
        analyze_causal_evidence((good,), max_pairs=True)
    with pytest.raises(TypeError):
        CausalPair(label=7, replication=0, baseline_value=1.0, intervened_value=2.0)
    with pytest.raises(TypeError):
        CausalPair(label="m", replication=True, baseline_value=1.0, intervened_value=2.0)
    with pytest.raises(TypeError):
        CausalPair(label="m", replication=0, baseline_value="1", intervened_value=2.0)


def test_value_violations_raise_contract_or_value_errors() -> None:
    with pytest.raises(ValueError):
        CausalPair(label="", replication=0, baseline_value=1.0, intervened_value=2.0)
    with pytest.raises(ValueError):
        CausalPair(label="m", replication=-1, baseline_value=1.0, intervened_value=2.0)
    with pytest.raises(ValueError):
        CausalPair(
            label="m",
            replication=0,
            baseline_value=float("nan"),
            intervened_value=2.0,
        )
    empty: tuple[CausalPair, ...] = ()
    with pytest.raises(CausalEvidenceError) as caught:
        analyze_causal_evidence(empty)
    assert caught.value.stage == "contract"


def test_budget_violations_are_stage_tagged() -> None:
    pair = _pair("m", 0, 1.0, 2.0)
    with pytest.raises(CausalEvidenceError) as zero:
        analyze_causal_evidence((pair,), max_pairs=0)
    assert zero.value.stage == "budget"
    with pytest.raises(CausalEvidenceError) as exceeded:
        analyze_causal_evidence((pair, pair), max_pairs=1)
    assert exceeded.value.stage == "budget"


def test_canonical_round_trip_and_rejection() -> None:
    pairs = (
        _pair("a/metric", 0, 5.0, 4.0),
        _pair("a/metric", 1, 10.0, 8.0),
        _pair("b/metric", 0, 0.0, 3.0),
    )
    result = analyze_causal_evidence(pairs)
    restored = CausalEvidence.from_json(result.to_json())
    assert restored == result
    assert CausalEvidence.from_json(result.to_json().encode()) == result
    document = result.to_json()
    assert '"artifact_type":"moeatlas.causal_evidence"' in document
    with pytest.raises(ValueError, match="not a causal evidence artifact"):
        CausalEvidence.from_json('{"schema_version":"1.0"}')
    with pytest.raises(ValueError, match="not valid JSON"):
        CausalEvidence.from_json("{oops")
    with pytest.raises(ValueError, match="must be a JSON object"):
        CausalEvidence.from_json("[]")


def test_row_width_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="one entry per label"):
        CausalEvidence(
            schema_version=CAUSAL_EVIDENCE_SCHEMA_VERSION,
            labels=("a",),
            replication_counts=(1, 2),
            mean_baseline=(1.0,),
            mean_intervened=(2.0,),
            absolute_effects=(1.0,),
            relative_effects=(1.0,),
            direction_consistency=(1.0,),
            stable_labels=(True,),
            zero_effect_labels=(False,),
        )
