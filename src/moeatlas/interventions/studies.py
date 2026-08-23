"""Replicated intervention studies with explicit negative-control evidence."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from moeatlas.core import stable_digest, validate_stable_identifier

INTERVENTION_STUDY_SCHEMA_VERSION = "1.0"
INTERVENTION_STUDY_ARTIFACT_TYPE = "moeatlas.intervention_study"
_DIRECTORY = "intervention-studies"
_MAX_REPLICATIONS = 100
_MAX_BYTES = 2_000_000


class InterventionStudyError(RuntimeError):
    """Safe failure for replicated-study construction or persistence."""


def _documents(value: object, *, field_name: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise TypeError(f"{field_name} must be a sequence of evidence mappings")
    documents = tuple(value)
    if not documents:
        raise InterventionStudyError(f"{field_name} must not be empty")
    if len(documents) > _MAX_REPLICATIONS:
        raise InterventionStudyError(f"{field_name} exceeds the replication budget")
    if any(not isinstance(document, Mapping) for document in documents):
        raise TypeError(f"{field_name} must contain evidence mappings")
    return documents


def _validated_run_key(document: Mapping[str, Any]) -> str:
    run_key = document.get("intervention_run_key")
    if not isinstance(run_key, str):
        raise InterventionStudyError("replication has no intervention run key")
    return validate_stable_identifier(run_key, field_name="intervention_run_key")


def _finite_delta(document: Mapping[str, Any]) -> float | None:
    value = document.get("task_score_delta")
    if value is None:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise InterventionStudyError("task score delta must be numeric or null")
    result = float(value)
    if not math.isfinite(result):
        raise InterventionStudyError("task score delta must be finite")
    return result


def _summary(values: list[float]) -> dict[str, float | list[float] | None]:
    if not values:
        return {
            "mean": None,
            "standard_deviation": None,
            "confidence_interval_95": None,
            "direction_consistency": None,
        }
    average = mean(values)
    deviation = stdev(values) if len(values) > 1 else 0.0
    half_width = 1.96 * deviation / math.sqrt(len(values)) if len(values) > 1 else 0.0
    if average > 0:
        consistency = sum(value > 0 for value in values) / len(values)
    elif average < 0:
        consistency = sum(value < 0 for value in values) / len(values)
    else:
        consistency = 0.0
    return {
        "mean": average,
        "standard_deviation": deviation,
        "confidence_interval_95": [average - half_width, average + half_width],
        "direction_consistency": consistency,
    }


def build_intervention_study(
    replications: Sequence[Mapping[str, Any]],
    *,
    controls: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Reduce independently persisted paired runs into one cautious claim."""

    repeated = _documents(replications, field_name="replications")
    control_documents = _documents(controls, field_name="controls") if controls else ()
    recipe_fingerprint = repeated[0].get("recipe_fingerprint")
    score_name = repeated[0].get("score_name")
    recipe = repeated[0].get("recipe")
    if not isinstance(recipe_fingerprint, str) or not isinstance(recipe, Mapping):
        raise InterventionStudyError("replication recipe evidence is unavailable")
    run_keys = tuple(_validated_run_key(document) for document in repeated)
    if len(set(run_keys)) != len(run_keys):
        raise InterventionStudyError("replication run keys must be unique")
    for document in repeated:
        if document.get("recipe_fingerprint") != recipe_fingerprint:
            raise InterventionStudyError("replications do not share one recipe")
        if document.get("score_name") != score_name:
            raise InterventionStudyError("replications do not share one evaluator")
    control_keys = tuple(_validated_run_key(document) for document in control_documents)
    if set(run_keys).intersection(control_keys) or len(set(control_keys)) != len(control_keys):
        raise InterventionStudyError("control and replication run keys must be distinct")

    deltas = [delta for document in repeated if (delta := _finite_delta(document)) is not None]
    control_deltas = [
        delta
        for document in control_documents
        if (delta := _finite_delta(document)) is not None
    ]
    effect = _summary(deltas)
    control_effect = _summary(control_deltas)
    all_targets_exercised = all(
        document.get("all_targets_exercised") is True for document in repeated
    )
    replicated = (
        len(repeated) >= 2
        and len(deltas) == len(repeated)
        and all_targets_exercised
        and effect["mean"] not in {None, 0.0}
        and effect["direction_consistency"] == 1.0
    )
    controlled = False
    if replicated and control_documents and len(control_deltas) == len(control_documents):
        target_mean = abs(float(effect["mean"]))
        control_mean = abs(float(control_effect["mean"]))
        controlled = target_mean > control_mean
    if controlled:
        claim_status = "controlled"
        claim_reason = "replicated task effect exceeds the measured negative-control effect"
    elif replicated:
        claim_status = "replicated"
        claim_reason = "effect repeats consistently; a negative control is still required"
    else:
        claim_status = "inconclusive"
        claim_reason = "effect is missing, unexercised, zero, or inconsistent across replications"
    document = {
        "artifact_type": INTERVENTION_STUDY_ARTIFACT_TYPE,
        "schema_version": INTERVENTION_STUDY_SCHEMA_VERSION,
        "recipe": dict(recipe),
        "recipe_fingerprint": recipe_fingerprint,
        "score_name": score_name,
        "replication_run_keys": list(run_keys),
        "control_run_keys": list(control_keys),
        "replication_count": len(repeated),
        "control_count": len(control_documents),
        "all_targets_exercised": all_targets_exercised,
        "task_effect": effect,
        "control_effect": control_effect,
        "claim_status": claim_status,
        "claim_reason": claim_reason,
    }
    document["study_id"] = "study:" + stable_digest(document)
    return document


def publish_intervention_study(workspace: str | Path, document: Mapping[str, Any]) -> Path:
    """Atomically publish one content-addressed replicated study."""

    if not isinstance(document, Mapping):
        raise TypeError("document must be a mapping")
    study_id = document.get("study_id")
    if not isinstance(study_id, str):
        raise InterventionStudyError("study has no content-addressed identity")
    validate_stable_identifier(study_id, field_name="study_id")
    payload = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if len(payload) > _MAX_BYTES:
        raise InterventionStudyError("study exceeds the artifact byte budget")
    root = Path(workspace)
    if root.is_symlink() or not root.is_dir():
        raise InterventionStudyError("workspace must be an existing non-symlink directory")
    directory = root / _DIRECTORY
    if directory.exists() and directory.is_symlink():
        raise InterventionStudyError("study directory must not be a symlink")
    directory.mkdir(exist_ok=True)
    target = directory / f"{study_id.removeprefix('study:')}.json"
    if target.exists():
        if target.is_symlink() or target.read_bytes() != payload:
            raise InterventionStudyError("published study conflicts with its identity")
        return target
    descriptor, staged_name = tempfile.mkstemp(dir=directory, prefix=".study-", suffix=".tmp")
    staged = Path(staged_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
        os.replace(staged, target)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return target


def read_intervention_study(workspace: str | Path, study_id: str) -> dict[str, Any]:
    """Read and verify one content-addressed study."""

    validate_stable_identifier(study_id, field_name="study_id")
    path = Path(workspace) / _DIRECTORY / f"{study_id.removeprefix('study:')}.json"
    if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_BYTES:
        raise InterventionStudyError("intervention study is unavailable")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("study_id") != study_id:
        raise InterventionStudyError("intervention study is invalid")
    identity_input = {key: value for key, value in document.items() if key != "study_id"}
    if study_id != "study:" + stable_digest(identity_input):
        raise InterventionStudyError("intervention study identity does not match its content")
    return document


__all__ = [
    "INTERVENTION_STUDY_ARTIFACT_TYPE",
    "INTERVENTION_STUDY_SCHEMA_VERSION",
    "InterventionStudyError",
    "build_intervention_study",
    "publish_intervention_study",
    "read_intervention_study",
]
