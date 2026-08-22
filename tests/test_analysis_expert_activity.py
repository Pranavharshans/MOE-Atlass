"""Model-free closed-form tests for the R3.3 expert-activity summary."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

try:
    import duckdb
except ImportError:  # pragma: no cover - exercised only without the store extra
    duckdb = None

import moeatlas.analysis.expert_activity as module
from moeatlas.analysis import (
    EXPERT_ACTIVITY_SUMMARY_SCHEMA_VERSION,
    ExpertActivitySummary,
    LayerExpertActivity,
    summarize_expert_activity,
)
from moeatlas.store import RoutingShardError

from .test_store_expert_shards import _expert_key, _layer_key, _result, _workspace


@pytest.fixture(autouse=True)
def _store_extra_required() -> None:
    if duckdb is None:
        pytest.skip("duckdb store extra is unavailable")


def _summary(tmp_path: Path):
    workspace = _workspace(tmp_path)
    receipt = None
    from moeatlas.store import append_structured_shard

    receipt = append_structured_shard(workspace, _result())
    layer_keys = (_layer_key(0), _layer_key(1))
    # The third layer-0 expert never fires anywhere: explicit zero activity.
    expert_keys = (
        (_expert_key(0, 0), _expert_key(0, 1), _expert_key(0, 2)),
        (_expert_key(1, 0), _expert_key(1, 1)),
    )
    return (
        summarize_expert_activity(
            workspace,
            run_key=receipt.run_key,
            layer_keys=layer_keys,
            expert_keys=expert_keys,
            max_expert_rows=64,
            max_source_bytes=10_000_000,
        ),
        workspace,
        receipt,
        layer_keys,
        expert_keys,
    )


def test_summary_version_constants_and_surface() -> None:
    assert EXPERT_ACTIVITY_SUMMARY_SCHEMA_VERSION == "1.0"
    assert module.EXPERT_ACTIVITY_SUMMARY_SCHEMA_VERSION == "1.0"


def test_closed_form_means_maxima_and_zero_activity_accounting(tmp_path: Path) -> None:
    summary, *_ = _summary(tmp_path)
    assert summary.run_key == "run-expert-1"
    assert summary.total_event_count == 4
    # One captured token selects each routed cell once; rank r contributes
    # exactly 0.5*(r+1).
    layer0 = summary.layers[0]
    layer1 = summary.layers[1]
    assert layer0.event_counts == (1, 1, 0)
    assert layer1.event_counts == (1, 1)
    assert layer0.mean_contributions == pytest.approx((0.5, 1.0, None))
    assert layer0.max_contributions == pytest.approx((0.5, 1.0, None))
    assert layer1.mean_contributions == pytest.approx((0.5, 1.0))
    assert summary.active_expert_cells == 4
    assert summary.inactive_expert_cells == 1
    # Zero-activity cells must carry null statistics.
    assert layer0.mean_contributions[2] is None and layer0.max_contributions[2] is None


def test_canonical_round_trip_through_json(tmp_path: Path) -> None:
    summary, *_ = _summary(tmp_path)
    document = json.loads(summary.to_json())
    assert document["artifact_type"] == "moeatlas.expert_activity_summary"
    assert "\n" not in summary.to_json()
    restored = ExpertActivitySummary.from_json(summary.to_json())
    assert restored.to_dict() == summary.to_dict()
    assert type(restored) is ExpertActivitySummary
    assert all(type(row) is LayerExpertActivity for row in restored.layers)


def test_from_json_rejects_foreign_or_malformed_documents() -> None:
    good = {
        "artifact_type": "moeatlas.expert_activity_summary",
        "schema_version": "1.0",
    }
    with pytest.raises(ValueError, match="not an expert activity summary"):
        ExpertActivitySummary.from_json(json.dumps(good | {"artifact_type": "other"}))
    with pytest.raises(ValueError, match="not valid JSON"):
        ExpertActivitySummary.from_json(b"{not json")
    with pytest.raises(ValueError, match="missing fields"):
        ExpertActivitySummary.from_json(json.dumps(good))
    with pytest.raises(ValueError, match="JSON object"):
        ExpertActivitySummary.from_json(b"[1]")


def test_constructor_invariants_are_strict() -> None:
    from moeatlas.store import STORE_SCHEMA_VERSION

    row = LayerExpertActivity(
        layer_key=_layer_key(0),
        event_counts=(1, 0),
        mean_contributions=(1.0, None),
        max_contributions=(1.0, None),
    )
    assert row.event_counts == (1, 0)
    with pytest.raises(ValueError, match="match the expert-cell axis"):
        LayerExpertActivity(
            layer_key=_layer_key(0),
            event_counts=(1,),
            mean_contributions=(1.0, None),
            max_contributions=(1.0, None),
        )
    # A zero-activity cell may pass the layer-level shape checks but must be
    # rejected by the summary when it carries non-null statistics.
    stale_zero = LayerExpertActivity(
        layer_key=_layer_key(0),
        event_counts=(0,),
        mean_contributions=(1.0,),
        max_contributions=(None,),
    )
    with pytest.raises(ValueError, match="zero-activity"):
        ExpertActivitySummary(
            schema_version=EXPERT_ACTIVITY_SUMMARY_SCHEMA_VERSION,
            store_schema_version=STORE_SCHEMA_VERSION,
            event_schema_version="1.0",
            run_key="run-1",
            shard_keys=("shard:" + "0" * 64,),
            layer_keys=(_layer_key(0),),
            expert_keys=((_expert_key(0, 0),),),
            layers=(stale_zero,),
            active_expert_cells=0,
            inactive_expert_cells=1,
            total_event_count=0,
        )
    with pytest.raises(ValueError, match="partition"):
        ExpertActivitySummary(
            schema_version=EXPERT_ACTIVITY_SUMMARY_SCHEMA_VERSION,
            store_schema_version=STORE_SCHEMA_VERSION,
            event_schema_version="1.0",
            run_key="run-1",
            shard_keys=("shard:" + "0" * 64,),
            layer_keys=(_layer_key(0),),
            expert_keys=((_expert_key(0, 0), _expert_key(0, 1)),),
            layers=(row,),
            active_expert_cells=1,
            inactive_expert_cells=2,
            total_event_count=1,
        )


def test_tampered_shards_fail_the_summary_with_storage_stage(tmp_path: Path) -> None:
    summary_context = _summary(tmp_path)
    summary, workspace, receipt, layer_keys, expert_keys = summary_context
    path = workspace / receipt.relative_path / "experts.parquet"
    original = path.read_bytes()
    path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(RoutingShardError) as caught:
        summarize_expert_activity(
            workspace,
            run_key=receipt.run_key,
            layer_keys=layer_keys,
            expert_keys=expert_keys,
            max_expert_rows=64,
            max_source_bytes=10_000_000,
        )
    assert caught.value.stage == "reopen"
    del summary


def test_row_budget_exhaustion_is_reported_as_budget(tmp_path: Path) -> None:
    _, workspace, receipt, layer_keys, expert_keys = _summary(tmp_path)
    with pytest.raises(ValueError, match="budget"):
        summarize_expert_activity(
            workspace,
            run_key=receipt.run_key,
            layer_keys=layer_keys,
            expert_keys=expert_keys,
            max_expert_rows=2,
            max_source_bytes=10_000_000,
        )


def test_unknown_universe_cells_fail_closed(tmp_path: Path) -> None:
    _, workspace, receipt, layer_keys, _ = _summary(tmp_path)
    stranger = "component:" + format(9999, "064x")
    with pytest.raises(ValueError, match="sources are unusable") as caught:
        summarize_expert_activity(
            workspace,
            run_key=receipt.run_key,
            layer_keys=layer_keys,
            expert_keys=((stranger,), (_expert_key(1, 0),)),
            max_expert_rows=64,
            max_source_bytes=10_000_000,
        )
    assert "unknown expert" in str(caught.value.__cause__)


def test_argument_validation_is_strict(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    for bad_budget in (0, -1, True, "8", 1.5):
        with pytest.raises((TypeError, ValueError)):
            summarize_expert_activity(
                workspace,
                run_key="run-1",
                layer_keys=(_layer_key(0),),
                expert_keys=((_expert_key(0, 0),),),
                max_expert_rows=bad_budget,  # type: ignore[arg-type]
                max_source_bytes=10_000_000,
            )
    with pytest.raises(TypeError, match="workspace"):
        summarize_expert_activity(
            object(),  # type: ignore[arg-type]
            run_key="run-1",
            layer_keys=(_layer_key(0),),
            expert_keys=((_expert_key(0, 0),),),
            max_expert_rows=8,
            max_source_bytes=10_000_000,
        )


def test_analysis_seam_stays_public_bounded_and_pure() -> None:
    source = Path("src/moeatlas/analysis/expert_activity.py").read_text()
    tree = ast.parse(source)
    forbidden = ("torch", "transformers", "numpy", "pandas", "pyarrow", "duckdb")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] not in forbidden for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in forbidden
    assert "_storage.query_expert_activity(" in source
    for private in (
        "_validate_sources",
        "_validate_file_metadata",
        "_read_shard_manifest",
        "_existing_run_parent",
        "_validate_workspace",
        "_reconstruct_shard",
    ):
        assert f"_storage.{private}" not in source
