"""Tests for deterministic built-in task evaluators."""

from __future__ import annotations

import pytest

from moeatlas.evaluation import EvaluationMethod, evaluate_text, normalize_text


def test_normalized_exact_match_and_contains_reference() -> None:
    assert normalize_text("  Hello\n WORLD ") == "hello world"
    assert evaluate_text("Yes", " yes ").score == 1.0
    assert evaluate_text("The answer is Paris.", "paris", "contains_reference").score == 1.0


def test_token_f1_counts_duplicate_tokens() -> None:
    result = evaluate_text("red red blue", "red blue blue", EvaluationMethod.TOKEN_F1)
    assert result.score == pytest.approx(2 / 3)


def test_multiple_choice_uses_the_last_bounded_choice() -> None:
    result = evaluate_text(
        "I considered B. Final answer: C", "C", "multiple_choice_accuracy"
    )
    assert result.score == 1.0
    assert evaluate_text("Final answer: B", "C", "multiple_choice_accuracy").score == 0.0


def test_numeric_match_uses_exact_decimal_values() -> None:
    assert evaluate_text("Result: 1.50", "1.5", "numeric_match").score == 1.0
    assert evaluate_text("Result: 1.51", "1.5", "numeric_match").score == 0.0


def test_unknown_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown evaluation method"):
        evaluate_text("a", "a", "unsafe-caller-code")
