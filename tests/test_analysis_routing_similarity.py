from __future__ import annotations

from dataclasses import replace

import pytest

from moeatlas.analysis import (
    ROUTING_SIMILARITY_SCHEMA_VERSION,
    RoutingSimilarity,
    compare_routing_similarity,
)

from .test_analysis_routing_compare import _comparison_matrix, _matrix


def test_similarity_reports_distribution_rank_and_top_set_evidence() -> None:
    result = compare_routing_similarity(
        _matrix(), _comparison_matrix(), top_n=2, max_cells=8
    )

    assert result.schema_version == ROUTING_SIMILARITY_SCHEMA_VERSION
    assert result.baseline_run_key == "run-baseline"
    assert result.comparison_run_key == "run-comparison"
    assert result.top_n == 2
    assert result.baseline_token_count == 2
    assert result.comparison_token_count == 2
    assert len(result.js_divergence_rows) == 2
    assert result.js_divergence_rows[0] > 0.0
    assert result.top_n_jaccard_rows == pytest.approx((1.0 / 3.0, 1.0 / 3.0))
    assert result.spearman_rows[0] is not None
    assert result.baseline_top1_margin_rows == pytest.approx((0.0, 0.0))
    assert result.comparison_top1_margin_rows == pytest.approx((0.25, 0.25))
    assert RoutingSimilarity.from_json(result.to_json()) == result


def test_similarity_allows_different_sample_sizes_after_normalization() -> None:
    larger = _matrix(
        run_key="run-larger",
        counts=((0, 4, 0, 4), (2, 2, 2, 2)),
        token_count=4,
        assignment_count=16,
        shard_digits="23",
    )
    result = compare_routing_similarity(_matrix(), larger, max_cells=8)
    assert result.baseline_token_count == 2
    assert result.comparison_token_count == 4
    assert result.mean_js_divergence == pytest.approx(0.0)
    assert result.mean_top_n_jaccard == pytest.approx(1.0)


def test_uniform_ranks_are_explicitly_undefined() -> None:
    uniform = _matrix(
        counts=((1, 1, 1, 1), (1, 1, 1, 1)),
        shares=((0.25, 0.25, 0.25, 0.25), (0.25, 0.25, 0.25, 0.25)),
        ratios=((1.0, 1.0, 1.0, 1.0), (1.0, 1.0, 1.0, 1.0)),
    )
    other = replace(
        uniform,
        run_key="run-uniform-other",
        shard_keys=_comparison_matrix().shard_keys,
    )
    result = compare_routing_similarity(uniform, other, max_cells=8)
    assert result.spearman_rows == (None, None)
    assert result.mean_spearman is None
    assert result.undefined_spearman_layers == 2


@pytest.mark.parametrize(
    "field_name,value",
    [
        ("model_key", "model:acme/other@r1"),
        ("inspection_digest", "sha256:" + "2" * 64),
        ("layout", "packed"),
    ],
)
def test_similarity_rejects_incompatible_evidence(field_name: str, value: object) -> None:
    with pytest.raises(ValueError, match=field_name):
        compare_routing_similarity(
            _matrix(), replace(_comparison_matrix(), **{field_name: value}), max_cells=8
        )


@pytest.mark.parametrize("top_n,max_cells", [(0, 8), (True, 8), (2, 0), (2, True)])
def test_similarity_rejects_invalid_budgets(top_n: object, max_cells: object) -> None:
    with pytest.raises(ValueError):
        compare_routing_similarity(
            _matrix(),
            _comparison_matrix(),
            top_n=top_n,  # type: ignore[arg-type]
            max_cells=max_cells,  # type: ignore[arg-type]
        )
