"""Paired evidence tests for baseline-derived intervention runs."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from moeatlas.interventions import (
    InterventionOperation,
    InterventionOutcome,
    InterventionRecipe,
    build_intervention_evidence,
    publish_intervention_evidence,
    read_intervention_evidence,
)
from moeatlas.services.run_engine import RowResult


def _execution(*rows: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(
        results=tuple(RowResult(index, 0, row) for index, row in enumerate(rows))
    )


def test_paired_evidence_reports_output_score_latency_and_exercise(tmp_path) -> None:
    recipe = InterventionRecipe(
        operation=InterventionOperation.ABLATE,
        targets=("layer:1/expert:2",),
    )
    outcome = InterventionOutcome(
        schema_version="1.0",
        recipe_fingerprint=recipe.fingerprint,
        operation="ablate",
        targets=recipe.targets,
    )
    baseline = _execution(
        {
            "output_digest": "sha256:before-a",
            "task_score": 1.0,
            "score_name": "normalized_exact_match",
            "generation_ms": 10.0,
        },
        {"output_digest": "sha256:same", "task_score": 1.0, "generation_ms": 20.0},
    )
    intervened = _execution(
        {"output_digest": "sha256:after-a", "task_score": 0.0, "generation_ms": 15.0},
        {"output_digest": "sha256:same", "task_score": 1.0, "generation_ms": 25.0},
    )

    document = build_intervention_evidence(
        baseline_run_key="run:" + "1" * 64,
        intervention_run_key="run:" + "2" * 64,
        baseline_execution=baseline,
        intervention_execution=intervened,
        recipe=recipe,
        outcome=outcome,
        invocation_counts={"layer:1/expert:2": 2},
    )

    assert document["changed_output_fraction"] == 0.5
    assert document["task_score_delta"] == -0.5
    assert document["latency_delta_percent"] == pytest.approx(33.3333333333)
    assert document["all_targets_exercised"] is True
    published = publish_intervention_evidence(tmp_path, document)
    assert published.is_file()
    assert read_intervention_evidence(tmp_path, "run:" + "2" * 64) == document


def test_paired_evidence_requires_matching_rows_and_output_digests() -> None:
    recipe = InterventionRecipe(
        operation=InterventionOperation.ABLATE,
        targets=("layer:0/expert:0",),
    )
    outcome = InterventionOutcome(
        schema_version="1.0",
        recipe_fingerprint=recipe.fingerprint,
        operation="ablate",
        targets=recipe.targets,
    )
    with pytest.raises(Exception, match="output digests"):
        build_intervention_evidence(
            baseline_run_key="run:" + "3" * 64,
            intervention_run_key="run:" + "4" * 64,
            baseline_execution=_execution({"forward_ms": 1.0}),
            intervention_execution=_execution({"output_digest": "sha256:value"}),
            recipe=recipe,
            outcome=outcome,
            invocation_counts={},
        )
