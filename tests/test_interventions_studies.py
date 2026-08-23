"""Replicated intervention-study tests."""

from __future__ import annotations

import pytest

from moeatlas.interventions import (
    InterventionStudyError,
    build_intervention_study,
    publish_intervention_study,
    read_intervention_study,
)


def _evidence(index: int, delta: float, *, exercised: bool = True) -> dict[str, object]:
    recipe = {
        "artifact_type": "moeatlas.intervention_recipe",
        "schema_version": "1.0",
        "operation": "ablate",
        "targets": ["layer:1/expert:2"],
        "factor": None,
        "bias": None,
        "alternates": [],
    }
    return {
        "intervention_run_key": "run:" + f"{index:064x}",
        "recipe": recipe,
        "recipe_fingerprint": "sha256:" + "a" * 64,
        "score_name": "token_f1",
        "task_score_delta": delta,
        "all_targets_exercised": exercised,
    }


def test_replicated_effect_without_control_remains_replication_only() -> None:
    study = build_intervention_study([_evidence(1, -0.4), _evidence(2, -0.2)])

    assert study["claim_status"] == "replicated"
    assert study["task_effect"]["mean"] == pytest.approx(-0.3)
    assert study["task_effect"]["direction_consistency"] == 1.0


def test_negative_control_promotes_only_a_larger_replicated_effect(tmp_path) -> None:
    study = build_intervention_study(
        [_evidence(3, -0.4), _evidence(4, -0.3)],
        controls=[_evidence(5, -0.05), _evidence(6, 0.0)],
    )

    assert study["claim_status"] == "controlled"
    path = publish_intervention_study(tmp_path, study)
    assert path.is_file()
    assert read_intervention_study(tmp_path, study["study_id"]) == study


def test_inconsistent_or_unexercised_effect_is_inconclusive() -> None:
    inconsistent = build_intervention_study([_evidence(7, -0.2), _evidence(8, 0.1)])
    unexercised = build_intervention_study(
        [_evidence(9, -0.2), _evidence(10, -0.1, exercised=False)]
    )

    assert inconsistent["claim_status"] == "inconclusive"
    assert unexercised["claim_status"] == "inconclusive"


def test_duplicate_replication_run_is_rejected() -> None:
    evidence = _evidence(11, -0.2)
    with pytest.raises(InterventionStudyError, match="unique"):
        build_intervention_study([evidence, evidence])
