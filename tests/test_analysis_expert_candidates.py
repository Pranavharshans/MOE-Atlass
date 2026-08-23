from __future__ import annotations

from dataclasses import replace

import pytest

from moeatlas.analysis import (
    EXPERT_ACTIVITY_SUMMARY_SCHEMA_VERSION,
    EXPERT_CANDIDATE_SCHEMA_VERSION,
    ExpertActivitySummary,
    ExpertCandidateRanking,
    LayerExpertActivity,
    rank_expert_candidates,
)
from moeatlas.events import EVENT_SCHEMA_VERSION
from moeatlas.store import STORE_SCHEMA_VERSION

from .test_analysis_routing_compare import _matrix


def _activity() -> ExpertActivitySummary:
    matrix = _matrix()
    return ExpertActivitySummary(
        schema_version=EXPERT_ACTIVITY_SUMMARY_SCHEMA_VERSION,
        store_schema_version=STORE_SCHEMA_VERSION,
        event_schema_version=EVENT_SCHEMA_VERSION,
        run_key=matrix.run_key,
        shard_keys=matrix.shard_keys,
        layer_keys=matrix.layer_keys,
        expert_keys=matrix.expert_keys,
        layers=(
            LayerExpertActivity(
                layer_key=matrix.layer_keys[0],
                event_counts=(0, 2, 0, 2),
                mean_contributions=(None, 1.0, None, 4.0),
                variance_contributions=(None, 0.25, None, 1.0),
                max_contributions=(None, 1.5, None, 5.0),
            ),
            LayerExpertActivity(
                layer_key=matrix.layer_keys[1],
                event_counts=(1, 1, 1, 1),
                mean_contributions=(3.0, 2.0, 0.5, 1.5),
                variance_contributions=(0.0, 0.0, 0.0, 0.0),
                max_contributions=(3.0, 2.0, 0.5, 1.5),
            ),
        ),
        active_expert_cells=6,
        inactive_expert_cells=2,
        total_event_count=8,
    )


def test_candidates_rank_reconciled_observed_contribution() -> None:
    ranking = rank_expert_candidates(_matrix(), _activity(), max_candidates=2)

    assert ranking.schema_version == EXPERT_CANDIDATE_SCHEMA_VERSION
    assert type(ranking) is ExpertCandidateRanking
    assert ranking.ranked_cell_count == 6
    assert ranking.incomplete_cell_count == 2
    assert ranking.evidence_complete is False
    assert [item.total_contribution for item in ranking.high_observed] == [8.0, 3.0]
    assert [item.total_contribution for item in ranking.low_observed] == [0.5, 1.5]
    assert ranking.high_observed[0].routing_count == 2
    assert ranking.high_observed[0].contribution_variance == 1.0
    assert "paired intervention" in str(ranking.to_dict()["claim_boundary"])


def test_count_mismatch_is_excluded_instead_of_silently_ranked() -> None:
    activity = _activity()
    first = activity.layers[0]
    incomplete_first = replace(
        first,
        event_counts=(0, 1, 0, 2),
        mean_contributions=(None, 1.0, None, 4.0),
        variance_contributions=(None, 0.25, None, 1.0),
        max_contributions=(None, 1.5, None, 5.0),
    )
    incomplete = replace(
        activity,
        layers=(incomplete_first, activity.layers[1]),
        total_event_count=7,
    )
    ranking = rank_expert_candidates(_matrix(), incomplete, max_candidates=8)
    assert ranking.ranked_cell_count == 5
    assert ranking.incomplete_cell_count == 3
    assert all(item.expert_index != 1 or item.layer_index != 0 for item in ranking.high_observed)


@pytest.mark.parametrize("max_candidates", [0, -1, True, 1.5, "8"])
def test_candidate_budget_is_strict(max_candidates: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        rank_expert_candidates(
            _matrix(),
            _activity(),
            max_candidates=max_candidates,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("run_key", "run-other", "run_key"),
        ("expert_keys", tuple(reversed(_matrix().expert_keys)), "expert_keys"),
    ],
)
def test_candidate_ranking_rejects_mismatched_evidence(
    field: str, value: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        rank_expert_candidates(_matrix(), replace(_activity(), **{field: value}))


def test_candidate_ranking_rejects_reordered_layer_evidence() -> None:
    activity = _activity()
    reordered = replace(
        activity,
        layer_keys=tuple(reversed(activity.layer_keys)),
        expert_keys=tuple(reversed(activity.expert_keys)),
        layers=tuple(reversed(activity.layers)),
    )
    with pytest.raises(ValueError, match="layer_keys"):
        rank_expert_candidates(_matrix(), reordered)


def test_candidate_ranking_rejects_non_artifact_inputs() -> None:
    with pytest.raises(TypeError, match="matrix"):
        rank_expert_candidates(object(), _activity())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="activity"):
        rank_expert_candidates(_matrix(), object())  # type: ignore[arg-type]
