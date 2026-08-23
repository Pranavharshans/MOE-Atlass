"""Deterministic, model-independent text evaluation contracts."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum


class EvaluationMethod(str, Enum):
    """Built-in scoring methods available to local research runs."""

    EXACT_MATCH = "normalized_exact_match"
    TOKEN_F1 = "token_f1"
    CONTAINS_REFERENCE = "contains_reference"
    MULTIPLE_CHOICE = "multiple_choice_accuracy"
    NUMERIC_MATCH = "numeric_match"


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    method: EvaluationMethod
    score: float

    def __post_init__(self) -> None:
        if not isinstance(self.method, EvaluationMethod):
            raise TypeError("method must be an EvaluationMethod")
        if not isinstance(self.score, float) or not math.isfinite(self.score):
            raise TypeError("score must be a finite float")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be between zero and one")


def normalize_text(value: object) -> str:
    """Return the shared case-folded, whitespace-normalized text form."""

    return " ".join(str(value).strip().casefold().split())


def _token_f1(output: str, reference: str) -> float:
    predicted = output.split()
    expected = reference.split()
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    common = Counter(predicted) & Counter(expected)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return (2.0 * precision * recall) / (precision + recall)


_CHOICE = re.compile(r"(?:^|\b)([a-z])(?:\b|$)", re.IGNORECASE)
_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def _last_match(pattern: re.Pattern[str], value: str) -> str | None:
    matches = pattern.findall(value)
    return matches[-1] if matches else None


def _numeric(value: str) -> Decimal | None:
    matched = _last_match(_NUMBER, value)
    if matched is None:
        return None
    try:
        parsed = Decimal(matched)
    except InvalidOperation:
        return None
    return parsed if parsed.is_finite() else None


def evaluate_text(
    output: object,
    reference: object,
    method: EvaluationMethod | str = EvaluationMethod.EXACT_MATCH,
) -> EvaluationResult:
    """Score one output/reference pair without executing caller code."""

    try:
        selected = method if isinstance(method, EvaluationMethod) else EvaluationMethod(method)
    except (TypeError, ValueError) as exc:
        raise ValueError("unknown evaluation method") from exc
    predicted = normalize_text(output)
    expected = normalize_text(reference)
    if selected is EvaluationMethod.EXACT_MATCH:
        score = float(predicted == expected)
    elif selected is EvaluationMethod.TOKEN_F1:
        score = _token_f1(predicted, expected)
    elif selected is EvaluationMethod.CONTAINS_REFERENCE:
        score = float(bool(expected) and expected in predicted)
    elif selected is EvaluationMethod.MULTIPLE_CHOICE:
        predicted_choice = _last_match(_CHOICE, predicted)
        expected_choice = _last_match(_CHOICE, expected)
        score = float(
            predicted_choice is not None
            and expected_choice is not None
            and predicted_choice.casefold() == expected_choice.casefold()
        )
    else:
        predicted_number = _numeric(predicted)
        expected_number = _numeric(expected)
        score = float(
            predicted_number is not None
            and expected_number is not None
            and predicted_number == expected_number
        )
    return EvaluationResult(method=selected, score=score)


__all__ = [
    "EvaluationMethod",
    "EvaluationResult",
    "evaluate_text",
    "normalize_text",
]
