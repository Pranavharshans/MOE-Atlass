from __future__ import annotations

import ast
import dataclasses
import inspect
import math
from dataclasses import replace
from pathlib import Path

import pytest

import moeatlas.analysis.routing_summary as routing_summary
from moeatlas.analysis import (
    ROUTING_SUMMARY_SCHEMA_VERSION,
    RoutingLoadMatrix,
    RoutingLoadSummary,
    summarize_routing_load,
)
from moeatlas.events import EVENT_SCHEMA_VERSION
from moeatlas.store import STORE_SCHEMA_VERSION

from .test_analysis_routing_compare import (
    _comparison_matrix,
    _component,
    _matrix,
    _shard,
    compare_routing_load,
)


def _uniform_matrix() -> RoutingLoadMatrix:
    return _matrix(
        run_key="run-uniform",
        counts=((1, 1, 1, 1), (1, 1, 1, 1)),
        shares=((0.25, 0.25, 0.25, 0.25), (0.25, 0.25, 0.25, 0.25)),
        ratios=((1.0, 1.0, 1.0, 1.0), (1.0, 1.0, 1.0, 1.0)),
    )


def _concentrated_matrix() -> RoutingLoadMatrix:
    return _matrix(
        run_key="run-concentrated",
        counts=((4, 0, 0, 0),),
        shares=((1.0, 0.0, 0.0, 0.0),),
        ratios=((4.0, 0.0, 0.0, 0.0),),
        layer_keys=(_component("a"),),
        layer_indices=(0,),
        expert_keys=((_component("c"), _component("d"), _component("e"), _component("f")),),
        assignment_count=4,
    )


def _summary(matrix: object | None = None, *, max_cells: int = 8) -> RoutingLoadSummary:
    return summarize_routing_load(_matrix() if matrix is None else matrix, max_cells=max_cells)


def test_public_surface_signature_and_schema() -> None:
    assert ROUTING_SUMMARY_SCHEMA_VERSION == "1.0"
    assert routing_summary.ROUTING_SUMMARY_SCHEMA_VERSION == "1.0"
    signature = inspect.signature(summarize_routing_load)
    assert tuple(signature.parameters) == ("matrix", "max_cells")
    assert signature.parameters["matrix"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert signature.parameters["max_cells"].kind is inspect.Parameter.KEYWORD_ONLY
    assert RoutingLoadSummary.__dataclass_params__.frozen is True


def test_exports_are_reachable_from_the_package() -> None:
    import moeatlas.analysis as package

    assert package.summarize_routing_load is summarize_routing_load
    assert package.RoutingLoadSummary is RoutingLoadSummary
    assert package.ROUTING_SUMMARY_SCHEMA_VERSION == "1.0"
    assert routing_summary.summarize_routing_load.__module__ == "moeatlas.analysis.routing_summary"


def test_default_fixture_happy_path_asserts_every_field() -> None:
    value = _summary()
    assert value.schema_version == "1.0"
    assert value.store_schema_version == STORE_SCHEMA_VERSION
    assert value.event_schema_version == EVENT_SCHEMA_VERSION
    assert value.run_key == "run-baseline"
    assert value.model_key == "model:acme/mixtral@r1"
    assert value.adapter_name == "huggingface-mixtral-static"
    assert value.adapter_version == "1.0"
    assert value.inspection_digest == "sha256:" + "1" * 64
    assert value.layout == "legacy_indexed"
    assert value.token_count == 2
    assert value.routed_top_k == 2
    assert value.assignment_count == 8
    assert value.shard_keys == (_shard("1"), _shard("9"))
    assert value.layer_keys == (_component("a"), _component("b"))
    assert value.layer_indices == (0, 1)
    assert value.expert_keys[0] == (
        _component("c"),
        _component("d"),
        _component("e"),
        _component("f"),
    )
    assert value.expert_keys[1] == (
        _component("1"),
        _component("2"),
        _component("3"),
        _component("4"),
    )
    assert value.layer_entropies == pytest.approx((math.log(2), math.log(4)))
    assert value.normalized_layer_entropies == pytest.approx((0.5, 1.0))
    assert value.effective_expert_counts == pytest.approx((2.0, 4.0))
    assert value.normalized_diversities == pytest.approx((0.5, 1.0))
    assert value.layer_gini_coefficients == pytest.approx((0.5, 0.0))
    assert value.layer_cv_counts == pytest.approx((1.0, 0.0))
    assert value.top_expert_shares == pytest.approx((0.5, 0.25))
    assert value.dead_expert_count == 2
    assert value.dead_expert_fraction == pytest.approx(0.25)


def test_uniform_distribution_matches_closed_form_metrics() -> None:
    value = summarize_routing_load(_uniform_matrix(), max_cells=8)
    assert value.layer_entropies == pytest.approx((math.log(4), math.log(4)))
    assert value.normalized_layer_entropies == pytest.approx((1.0, 1.0))
    assert value.effective_expert_counts == pytest.approx((4.0, 4.0))
    assert value.normalized_diversities == pytest.approx((1.0, 1.0))
    assert value.layer_gini_coefficients == (0.0, 0.0)
    assert value.layer_cv_counts == (0.0, 0.0)
    assert value.top_expert_shares == pytest.approx((0.25, 0.25))


def test_concentrated_distribution_matches_closed_form_metrics() -> None:
    value = summarize_routing_load(_concentrated_matrix(), max_cells=8)
    assert value.layer_entropies == (0.0,)
    assert value.normalized_layer_entropies == (0.0,)
    assert value.effective_expert_counts == (1.0,)
    assert value.normalized_diversities == (0.25,)
    assert value.layer_gini_coefficients == (0.75,)
    assert value.layer_cv_counts == pytest.approx((math.sqrt(3),))
    assert value.top_expert_shares == (1.0,)
    assert value.dead_expert_count == 3
    assert value.dead_expert_fraction == pytest.approx(0.75)


def test_gini_anchors_match_the_ascending_rank_formula() -> None:
    concentrated = summarize_routing_load(_concentrated_matrix(), max_cells=8)
    uniform = summarize_routing_load(_uniform_matrix(), max_cells=8)
    assert concentrated.layer_gini_coefficients[0] == 0.75
    assert uniform.layer_gini_coefficients[0] == 0.0


def test_dead_expert_fraction_uses_the_full_cell_universe() -> None:
    value = _summary()
    cells = len(value.layer_keys) * len(value.expert_keys[0])
    assert cells == 8
    assert value.dead_expert_count == 2
    assert value.dead_expert_fraction == pytest.approx(value.dead_expert_count / cells)


def test_zero_dead_experts_fixture_reports_no_dead_experts() -> None:
    value = summarize_routing_load(_uniform_matrix(), max_cells=8)
    assert value.dead_expert_count == 0
    assert value.dead_expert_fraction == 0.0


def test_summary_is_deterministic_and_frozen() -> None:
    first = _summary()
    second = _summary()
    assert first == second
    with pytest.raises(Exception):
        first.token_count = 5  # type: ignore[misc]


@pytest.mark.parametrize("max_cells", [True, False, 0, -1, 1.5, "8", None])
def test_invalid_max_cells_is_rejected(max_cells: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _summary(max_cells=max_cells)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [None, 42, "run-baseline"])
def test_non_matrix_inputs_are_rejected(value: object) -> None:
    with pytest.raises(TypeError):
        summarize_routing_load(value, max_cells=8)  # type: ignore[arg-type]


def test_comparison_object_input_is_rejected() -> None:
    comparison = compare_routing_load(_matrix(), _comparison_matrix(), max_cells=8)
    with pytest.raises(TypeError):
        summarize_routing_load(comparison, max_cells=8)  # type: ignore[arg-type]


def test_exact_type_check_rejects_matrix_subclasses() -> None:
    subclass = type("MatrixSubclass", (RoutingLoadMatrix,), {})
    forged = subclass(
        **{field: getattr(_matrix(), field) for field in RoutingLoadMatrix.__dataclass_fields__}
    )
    with pytest.raises(TypeError):
        summarize_routing_load(forged, max_cells=8)


def test_summary_revalidates_a_fresh_matrix_copy() -> None:
    invalid = object.__new__(RoutingLoadMatrix)
    for field in RoutingLoadMatrix.__dataclass_fields__:
        object.__setattr__(invalid, field, getattr(_matrix(), field))
    object.__setattr__(invalid, "assignment_counts", ((0, 2, 0, 1), (1, 1, 1, 1)))
    with pytest.raises((TypeError, ValueError)):
        summarize_routing_load(invalid, max_cells=8)


def test_validation_order_is_max_cells_then_type_then_budget() -> None:
    with pytest.raises((TypeError, ValueError)):
        summarize_routing_load(None, max_cells=0)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        summarize_routing_load(None, max_cells=8)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exceed max_cells"):
        summarize_routing_load(_matrix(), max_cells=7)


def test_cell_budget_is_enforced() -> None:
    with pytest.raises(ValueError, match="exceed max_cells"):
        _summary(max_cells=7)
    _summary(max_cells=8)


@pytest.mark.parametrize(
    "updates",
    [
        {"schema_version": "2.0"},
        {"run_key": 42},
        {"shard_keys": (_shard("9"), _shard("1"))},
        {"layer_indices": (1, 0)},
        {
            "expert_keys": (
                (_component("c"),),
                (_component("1"), _component("2"), _component("3"), _component("4")),
            )
        },
        {"layer_entropies": (float("nan"), math.log(4))},
        {"layer_gini_coefficients": (1.5, 0.0)},
        {"layer_cv_counts": (-1.0, 0.0)},
        {"dead_expert_fraction": 0.5},
        {"dead_expert_count": 9},
        {"top_expert_shares": (float("inf"), 0.25)},
    ],
)
def test_tampered_fields_fail_fresh_validation(updates: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(_summary(), **updates)  # type: ignore[arg-type]


def test_summary_retains_only_scalar_and_tuple_evidence() -> None:
    value = _summary()
    for field in dataclasses.fields(value):
        assert type(getattr(value, field.name)) in (str, int, float, tuple)
    assert not any(
        isinstance(getattr(value, field.name), RoutingLoadMatrix)
        for field in dataclasses.fields(value)
    )


def test_module_imports_are_model_free_and_offline() -> None:
    source_path = Path("src/moeatlas/analysis/routing_summary.py")
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
                raise AssertionError("summary must not mutate the filesystem")
            if isinstance(node.func, ast.Name) and node.func.id == "open":
                raise AssertionError("summary must not touch the filesystem")
    assert "token_text" not in source
    assert "connect(" not in source
