from __future__ import annotations

import ast
import dataclasses
import inspect
import math
from dataclasses import replace
from pathlib import Path

import pytest

import moeatlas.analysis.routing_compare as routing_compare
from moeatlas.analysis import (
    ROUTING_COMPARE_SCHEMA_VERSION,
    RoutingLoadComparison,
    RoutingLoadMatrix,
    compare_routing_load,
)
from moeatlas.events import EVENT_SCHEMA_VERSION
from moeatlas.store import STORE_SCHEMA_VERSION


def _component(digit: str) -> str:
    return "component:" + digit * 64


def _shard(digit: str) -> str:
    return "shard:" + digit * 64


_BASE_COUNTS = ((0, 2, 0, 2), (1, 1, 1, 1))
_BASE_SHARES = ((0.0, 0.5, 0.0, 0.5), (0.25, 0.25, 0.25, 0.25))
_BASE_RATIOS = ((0.0, 2.0, 0.0, 2.0), (1.0, 1.0, 1.0, 1.0))
_OTHER_COUNTS = ((1, 1, 0, 2), (2, 0, 1, 1))
_OTHER_SHARES = ((0.25, 0.25, 0.0, 0.5), (0.5, 0.0, 0.25, 0.25))
_OTHER_RATIOS = ((1.0, 1.0, 0.0, 2.0), (2.0, 0.0, 1.0, 1.0))


def _matrix(
    run_key: str = "run-baseline",
    *,
    counts: tuple[tuple[int, ...], ...] = _BASE_COUNTS,
    shares: tuple[tuple[float, ...], ...] = _BASE_SHARES,
    ratios: tuple[tuple[float, ...], ...] = _BASE_RATIOS,
    shard_digits: str = "19",
    **overrides: object,
) -> RoutingLoadMatrix:
    values: dict[str, object] = {
        "schema_version": "1.0",
        "store_schema_version": STORE_SCHEMA_VERSION,
        "event_schema_version": EVENT_SCHEMA_VERSION,
        "run_key": run_key,
        "model_key": "model:acme/mixtral@r1",
        "adapter_name": "huggingface-mixtral-static",
        "adapter_version": "1.0",
        "inspection_digest": "sha256:" + "1" * 64,
        "layout": "legacy_indexed",
        "shard_keys": tuple(_shard(digit) for digit in shard_digits),
        "token_count": 2,
        "assignment_count": 8,
        "routed_top_k": 2,
        "layer_keys": (_component("a"), _component("b")),
        "layer_indices": (0, 1),
        "expert_keys": (
            (_component("c"), _component("d"), _component("e"), _component("f")),
            (_component("1"), _component("2"), _component("3"), _component("4")),
        ),
        "assignment_counts": counts,
        "assignment_shares": shares,
        "load_ratios": ratios,
    }
    values.update(overrides)
    return RoutingLoadMatrix(**values)  # type: ignore[arg-type]


def _comparison_matrix() -> RoutingLoadMatrix:
    return _matrix(
        run_key="run-comparison",
        counts=_OTHER_COUNTS,
        shares=_OTHER_SHARES,
        ratios=_OTHER_RATIOS,
        shard_digits="23",
    )


def _compare(baseline: object | None = None, comparison: object | None = None, **kwargs: int):
    return compare_routing_load(
        _matrix() if baseline is None else baseline,
        _comparison_matrix() if comparison is None else comparison,
        max_cells=kwargs.get("max_cells", 8),
    )


def test_public_surface_signature_and_schema() -> None:
    assert ROUTING_COMPARE_SCHEMA_VERSION == "1.0"
    signature = inspect.signature(compare_routing_load)
    assert tuple(signature.parameters) == ("baseline", "comparison", "max_cells")
    assert signature.parameters["max_cells"].kind is inspect.Parameter.KEYWORD_ONLY
    assert RoutingLoadComparison.__dataclass_params__.frozen is True


def test_comparison_produces_exact_deltas_and_frozen_provenance() -> None:
    value = _compare()
    assert value.schema_version == ROUTING_COMPARE_SCHEMA_VERSION
    assert value.baseline_run_key == "run-baseline"
    assert value.comparison_run_key == "run-comparison"
    assert value.model_key == "model:acme/mixtral@r1"
    assert value.adapter_name == "huggingface-mixtral-static"
    assert value.adapter_version == "1.0"
    assert value.inspection_digest == "sha256:" + "1" * 64
    assert value.layout == "legacy_indexed"
    assert value.token_count == 2
    assert value.routed_top_k == 2
    assert value.baseline_shard_keys == (_shard("1"), _shard("9"))
    assert value.comparison_shard_keys == (_shard("2"), _shard("3"))
    assert value.baseline_assignment_count == 8
    assert value.comparison_assignment_count == 8
    assert value.layer_keys == (_component("a"), _component("b"))
    assert value.layer_indices == (0, 1)
    assert value.expert_keys[0] == (
        _component("c"),
        _component("d"),
        _component("e"),
        _component("f"),
    )
    assert value.count_deltas == ((1, -1, 0, 0), (1, -1, 0, 0))
    assert value.share_deltas == ((0.25, -0.25, 0.0, 0.0), (0.25, -0.25, 0.0, 0.0))
    assert value.ratio_deltas == ((1.0, -1.0, 0.0, 0.0), (1.0, -1.0, 0.0, 0.0))


def test_comparison_is_deterministic_and_frozen() -> None:
    first = _compare()
    second = _compare()
    assert first == second
    with pytest.raises(Exception):
        first.token_count = 5  # type: ignore[misc]


def test_identical_distributions_across_distinct_runs_yield_zero_deltas() -> None:
    value = _compare(_matrix(), _matrix(run_key="run-comparison"))
    assert value.count_deltas == ((0, 0, 0, 0), (0, 0, 0, 0))
    assert all(delta == 0.0 for row in value.share_deltas for delta in row)
    assert all(delta == 0.0 for row in value.ratio_deltas for delta in row)


@pytest.mark.parametrize("side", ["baseline", "comparison"])
@pytest.mark.parametrize("value", [42, None, "run-baseline", 3.5])
def test_non_matrix_inputs_are_rejected(side: str, value: object) -> None:
    with pytest.raises(TypeError):
        compare_routing_load(
            value if side == "baseline" else _matrix(),
            _comparison_matrix() if side == "baseline" else value,
            max_cells=8,
        )


@pytest.mark.parametrize("max_cells", [True, False, 0, -1, 1.5, "8", None])
def test_invalid_max_cells_is_rejected(max_cells: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _compare(max_cells=max_cells)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field_name, override",
    [
        ("model_key", "model:acme/other@r1"),
        ("adapter_name", "other-adapter"),
        ("adapter_version", "2.0"),
        ("inspection_digest", "sha256:" + "2" * 64),
        ("layout", "packed"),
        ("layer_indices", (0, 2)),
        (
            "expert_keys",
            (
                (_component("7"), _component("d"), _component("e"), _component("f")),
                (_component("1"), _component("2"), _component("3"), _component("4")),
            ),
        ),
    ],
)
def test_universe_identity_mismatch_is_rejected(field_name: str, override: object) -> None:
    baseline = _matrix()
    comparison = _comparison_matrix()
    tampered_baseline = replace(baseline, **{field_name: override})  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        compare_routing_load(tampered_baseline, comparison, max_cells=8)


def test_token_count_mismatch_is_rejected() -> None:
    four_token_counts = ((1, 1, 1, 5), (2, 2, 2, 2))
    four_token_shares = ((0.125, 0.125, 0.125, 0.625), (0.25, 0.25, 0.25, 0.25))
    four_token_ratios = ((0.5, 0.5, 0.5, 2.5), (1.0, 1.0, 1.0, 1.0))
    baseline = _matrix()
    comparison = _comparison_matrix()
    tampered_comparison = replace(
        comparison,
        token_count=4,
        assignment_count=16,
        assignment_counts=four_token_counts,
        assignment_shares=four_token_shares,
        load_ratios=four_token_ratios,
    )
    with pytest.raises(ValueError):
        compare_routing_load(baseline, tampered_comparison, max_cells=8)


def test_routed_top_k_mismatch_is_rejected() -> None:
    three_top_counts = ((1, 2, 1, 2), (0, 3, 1, 2))
    three_top_shares = (
        (1 / 6, 1 / 3, 1 / 6, 1 / 3),
        (0.0, 0.5, 1 / 6, 1 / 3),
    )
    three_top_ratios = tuple(tuple(share * 4 for share in row) for row in three_top_shares)
    baseline = _matrix()
    comparison = _comparison_matrix()
    tampered_comparison = replace(
        comparison,
        routed_top_k=3,
        assignment_count=12,
        assignment_counts=three_top_counts,
        assignment_shares=three_top_shares,
        load_ratios=three_top_ratios,
    )
    with pytest.raises(ValueError):
        compare_routing_load(baseline, tampered_comparison, max_cells=8)


def test_layer_axis_mismatch_is_rejected() -> None:
    baseline = _matrix()
    comparison = _comparison_matrix()
    tampered_comparison = replace(
        comparison,
        layer_keys=(_component("a"),),
        layer_indices=(0,),
        expert_keys=((_component("c"), _component("d"), _component("e"), _component("f")),),
        assignment_counts=((1, 1, 0, 2),),
        assignment_shares=(_OTHER_SHARES[0],),
        load_ratios=(_OTHER_RATIOS[0],),
        assignment_count=4,
    )
    with pytest.raises(ValueError):
        compare_routing_load(baseline, tampered_comparison, max_cells=8)


def test_same_run_key_is_rejected() -> None:
    with pytest.raises(ValueError):
        _compare(
            _matrix(),
            _matrix(counts=_OTHER_COUNTS, shares=_OTHER_SHARES, ratios=_OTHER_RATIOS),
        )


def test_cell_budget_is_enforced() -> None:
    with pytest.raises(ValueError):
        _compare(max_cells=7)
    _compare(max_cells=8)


def _tampered(**updates: object) -> RoutingLoadComparison:
    value = _compare()
    return replace(value, **updates)  # type: ignore[arg-type]


def test_value_rejects_inconsistent_count_deltas() -> None:
    with pytest.raises(ValueError):
        _tampered(count_deltas=((1, -1, 0, 1), (1, -1, 0, 0)))
    with pytest.raises(TypeError):
        _tampered(count_deltas=((1.0, -1.0, 0.0, 0.0), (1, -1, 0, 0)))
    with pytest.raises(ValueError):
        _tampered(count_deltas=((1, -1, 0, 0),))


def test_value_rejects_impossible_share_and_ratio_deltas() -> None:
    with pytest.raises(ValueError):
        _tampered(share_deltas=((1.5, -0.25, 0.0, -0.25), (0.25, -0.25, 0.0, 0.0)))
    with pytest.raises(TypeError):
        _tampered(share_deltas=((float("nan"), -0.25, 0.0, -0.25), (0.25, -0.25, 0.0, 0.0)))
    with pytest.raises(TypeError):
        _tampered(ratio_deltas=((float("inf"), -1.0, 0.0, 0.0), (1.0, -1.0, 0.0, 0.0)))
    with pytest.raises(ValueError):
        _tampered(ratio_deltas=((9.0, -1.0, 0.0, 0.0), (1.0, -1.0, 0.0, 0.0)))
    with pytest.raises(ValueError):
        _tampered(share_deltas=((0.25, -0.25, 0.0, 0.0), (0.25, -0.25, 0.0, 0.1)))


def test_value_rejects_identity_and_axis_tampering() -> None:
    with pytest.raises(ValueError):
        _tampered(baseline_run_key="run-comparison")
    with pytest.raises(ValueError):
        _tampered(comparison_run_key="run-baseline")
    with pytest.raises(ValueError):
        _tampered(baseline_run_key="run-same", comparison_run_key="run-same")
    with pytest.raises(ValueError):
        _tampered(baseline_assignment_count=7)
    with pytest.raises(ValueError):
        _tampered(comparison_assignment_count=9)
    with pytest.raises(ValueError):
        _tampered(baseline_shard_keys=())
    with pytest.raises(ValueError):
        _tampered(comparison_shard_keys=(_shard("3"), _shard("2")))
    with pytest.raises(ValueError):
        _tampered(layout="indexed")
    with pytest.raises(ValueError):
        _tampered(layer_indices=(1, 0))
    with pytest.raises(ValueError):
        _tampered(
            expert_keys=(
                (_component("c"), _component("c"), _component("e"), _component("f")),
                (_component("1"), _component("2"), _component("3"), _component("4")),
            )
        )


def test_comparison_retains_only_scalar_and_tuple_evidence() -> None:
    value = _compare()
    for field_name in (
        "baseline_shard_keys",
        "comparison_shard_keys",
        "layer_keys",
        "layer_indices",
        "expert_keys",
        "count_deltas",
        "share_deltas",
        "ratio_deltas",
    ):
        assert type(getattr(value, field_name)) is tuple
    assert not any(
        isinstance(getattr(value, field.name), RoutingLoadMatrix)
        for field in dataclasses.fields(value)
    )


def test_module_imports_are_model_free_and_offline() -> None:
    source_path = Path("src/moeatlas/analysis/routing_compare.py")
    source = source_path.read_text()
    tree = ast.parse(source)
    forbidden = {
        "torch",
        "transformers",
        "accelerate",
        "safetensors",
        "numpy",
        "pandas",
        "pyarrow",
        "polars",
        "duckdb",
        "socket",
        "urllib",
        "httpx",
        "requests",
        "importlib",
        "os",
        "shutil",
        "tempfile",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] not in forbidden for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in forbidden
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr in {
                "write_text",
                "write_bytes",
                "mkdir",
                "unlink",
                "rename",
            }:
                raise AssertionError("comparison must not mutate the filesystem")
            if isinstance(node.func, ast.Name) and node.func.id == "open":
                raise AssertionError("comparison must not touch the filesystem")
    assert "token_text" not in source
    assert "duckdb" not in source
    assert "connect(" not in source


def test_public_exports_are_reachable_from_the_package() -> None:
    import moeatlas.analysis as package

    assert package.compare_routing_load is compare_routing_load
    assert package.RoutingLoadComparison is RoutingLoadComparison
    assert package.ROUTING_COMPARE_SCHEMA_VERSION == "1.0"
    assert routing_compare.compare_routing_load.__module__ == "moeatlas.analysis.routing_compare"
    assert math.isfinite(1.0)
