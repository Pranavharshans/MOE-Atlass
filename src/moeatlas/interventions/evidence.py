"""Bounded paired evidence for baseline and intervention runs."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from moeatlas.core import validate_stable_identifier
from moeatlas.interventions.engine import InterventionOutcome
from moeatlas.interventions.recipes import InterventionRecipe

INTERVENTION_EVIDENCE_SCHEMA_VERSION = "1.0"
INTERVENTION_EVIDENCE_ARTIFACT_TYPE = "moeatlas.intervention_evidence"
_DIRECTORY = "interventions"
_MAX_ROWS = 10_000
_MAX_BYTES = 10_000_000


class InterventionEvidenceError(RuntimeError):
    """Safe failure for paired evidence construction or persistence."""


def _result_rows(value: object) -> dict[int, Mapping[str, Any]]:
    results = getattr(value, "results", None)
    if type(results) is not tuple:
        raise TypeError("execution evidence must expose a tuple of results")
    if len(results) > _MAX_ROWS:
        raise InterventionEvidenceError("execution evidence exceeds the row budget")
    rows: dict[int, Mapping[str, Any]] = {}
    for result in results:
        row_index = getattr(result, "row_index", None)
        document = getattr(result, "result", None)
        if (
            type(row_index) is not int
            or isinstance(row_index, bool)
            or not isinstance(document, Mapping)
            or row_index in rows
        ):
            raise InterventionEvidenceError("execution evidence contains invalid row results")
        rows[row_index] = document
    return rows


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def build_intervention_evidence(
    *,
    baseline_run_key: str,
    intervention_run_key: str,
    baseline_execution: object,
    intervention_execution: object,
    recipe: InterventionRecipe,
    outcome: InterventionOutcome,
    invocation_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Compare the exact successful row identities of two deterministic runs."""

    for name, key in (
        ("baseline_run_key", baseline_run_key),
        ("intervention_run_key", intervention_run_key),
    ):
        if type(key) is not str:
            raise TypeError(f"{name} must be a string")
        validate_stable_identifier(key, field_name=name)
    if baseline_run_key == intervention_run_key:
        raise ValueError("baseline and intervention run keys must differ")
    if type(recipe) is not InterventionRecipe:
        raise TypeError("recipe must be an InterventionRecipe")
    if type(outcome) is not InterventionOutcome:
        raise TypeError("outcome must be an InterventionOutcome")
    if outcome.recipe_fingerprint != recipe.fingerprint:
        raise ValueError("intervention outcome does not match the recipe")
    baseline = _result_rows(baseline_execution)
    intervened = _result_rows(intervention_execution)
    if tuple(sorted(baseline)) != tuple(sorted(intervened)):
        raise InterventionEvidenceError("baseline and intervention row identities differ")

    rows: list[dict[str, Any]] = []
    baseline_scores: list[float] = []
    intervention_scores: list[float] = []
    baseline_latency: list[float] = []
    intervention_latency: list[float] = []
    changed = 0
    for row_index in sorted(baseline):
        before = baseline[row_index]
        after = intervened[row_index]
        before_input = before.get("input_digest")
        after_input = after.get("input_digest")
        if not isinstance(before_input, str) or not isinstance(after_input, str):
            raise InterventionEvidenceError(
                "baseline must publish exact input digests before intervention"
            )
        if before_input != after_input:
            raise InterventionEvidenceError(
                f"baseline and intervention input differ for row {row_index}"
            )
        before_method = before.get("evaluation_method")
        after_method = after.get("evaluation_method")
        if before_method != after_method:
            raise InterventionEvidenceError(
                f"baseline and intervention evaluators differ for row {row_index}"
            )
        before_digest = before.get("output_digest")
        after_digest = after.get("output_digest")
        if not isinstance(before_digest, str) or not isinstance(after_digest, str):
            raise InterventionEvidenceError(
                "baseline must be captured by a version that publishes output digests"
            )
        is_changed = before_digest != after_digest
        changed += int(is_changed)
        before_score = before.get("task_score")
        after_score = after.get("task_score")
        before_ms = before.get("generation_ms") or before.get("forward_ms")
        after_ms = after.get("generation_ms") or after.get("forward_ms")
        if isinstance(before_score, int | float) and isinstance(after_score, int | float):
            baseline_scores.append(float(before_score))
            intervention_scores.append(float(after_score))
        if isinstance(before_ms, int | float) and isinstance(after_ms, int | float):
            baseline_latency.append(float(before_ms))
            intervention_latency.append(float(after_ms))
        row = {
            "row_index": row_index,
            "input_digest": before_input,
            "evaluation_method": before_method,
            "baseline_output_digest": before_digest,
            "intervention_output_digest": after_digest,
            "output_changed": is_changed,
            "score_name": after.get("score_name") or before.get("score_name"),
            "baseline_score": float(before_score)
            if isinstance(before_score, int | float)
            else None,
            "intervention_score": float(after_score)
            if isinstance(after_score, int | float)
            else None,
            "baseline_latency_ms": float(before_ms) if isinstance(before_ms, int | float) else None,
            "intervention_latency_ms": float(after_ms)
            if isinstance(after_ms, int | float)
            else None,
        }
        if isinstance(before.get("output_preview"), str):
            row["baseline_output_preview"] = before["output_preview"]
        if isinstance(after.get("output_preview"), str):
            row["intervention_output_preview"] = after["output_preview"]
        rows.append(row)

    mean_before_score = _mean(baseline_scores)
    mean_after_score = _mean(intervention_scores)
    mean_before_latency = _mean(baseline_latency)
    mean_after_latency = _mean(intervention_latency)
    counts = {label: int(invocation_counts.get(label, 0)) for label in recipe.targets}
    return {
        "artifact_type": INTERVENTION_EVIDENCE_ARTIFACT_TYPE,
        "schema_version": INTERVENTION_EVIDENCE_SCHEMA_VERSION,
        "status": "available",
        "baseline_run_key": baseline_run_key,
        "intervention_run_key": intervention_run_key,
        "recipe": recipe.to_dict(),
        "recipe_fingerprint": recipe.fingerprint,
        "outcome": outcome.to_dict(),
        "restoration_status": "restored",
        "target_invocation_counts": counts,
        "all_targets_exercised": all(value > 0 for value in counts.values()),
        "row_count": len(rows),
        "changed_output_rows": changed,
        "unchanged_output_rows": len(rows) - changed,
        "changed_output_fraction": changed / len(rows) if rows else None,
        "score_name": next((row["score_name"] for row in rows if row["score_name"]), None),
        "baseline_task_score": mean_before_score,
        "intervention_task_score": mean_after_score,
        "task_score_delta": (
            mean_after_score - mean_before_score
            if mean_before_score is not None and mean_after_score is not None
            else None
        ),
        "baseline_mean_latency_ms": mean_before_latency,
        "intervention_mean_latency_ms": mean_after_latency,
        "latency_delta_percent": (
            ((mean_after_latency - mean_before_latency) / mean_before_latency) * 100.0
            if mean_before_latency is not None
            and mean_after_latency is not None
            and mean_before_latency > 0
            else None
        ),
        "rows": rows,
    }


def publish_intervention_evidence(workspace: str | Path, document: Mapping[str, Any]) -> Path:
    """Atomically publish one validated intervention evidence document."""

    if not isinstance(document, Mapping):
        raise TypeError("document must be a mapping")
    run_key = document.get("intervention_run_key")
    if type(run_key) is not str:
        raise InterventionEvidenceError("intervention evidence has no run key")
    validate_stable_identifier(run_key, field_name="intervention_run_key")
    payload = json.dumps(
        dict(document),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(payload) > _MAX_BYTES:
        raise InterventionEvidenceError("intervention evidence exceeds the byte budget")
    root = Path(workspace)
    directory = root / _DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink():
        raise InterventionEvidenceError("intervention evidence directory is unsafe")
    target = directory / f"{run_key}.json"
    fd, staged_name = tempfile.mkstemp(prefix=".staging-", suffix=".json", dir=directory)
    staged = Path(staged_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
        os.replace(staged, target)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return target


def read_intervention_evidence(
    workspace: str | Path, run_key: str, *, max_bytes: int = _MAX_BYTES
) -> dict[str, Any]:
    """Read one bounded intervention evidence artifact without following symlinks."""

    validate_stable_identifier(run_key, field_name="run_key")
    target = Path(workspace) / _DIRECTORY / f"{run_key}.json"
    if target.is_symlink() or not target.is_file() or target.stat().st_size > max_bytes:
        raise InterventionEvidenceError("intervention evidence is unavailable")
    document = json.loads(target.read_text(encoding="utf-8"))
    if (
        not isinstance(document, dict)
        or document.get("artifact_type") != INTERVENTION_EVIDENCE_ARTIFACT_TYPE
        or document.get("schema_version") != INTERVENTION_EVIDENCE_SCHEMA_VERSION
        or document.get("intervention_run_key") != run_key
    ):
        raise InterventionEvidenceError("intervention evidence is invalid")
    return document


__all__ = [
    "INTERVENTION_EVIDENCE_ARTIFACT_TYPE",
    "INTERVENTION_EVIDENCE_SCHEMA_VERSION",
    "InterventionEvidenceError",
    "build_intervention_evidence",
    "publish_intervention_evidence",
    "read_intervention_evidence",
]
