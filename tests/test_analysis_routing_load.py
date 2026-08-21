from __future__ import annotations

import ast
import gc
import inspect
import socket
import urllib.request
import weakref
from dataclasses import fields, replace
from pathlib import Path

import pytest

import moeatlas.store.routing_shards as storage
from moeatlas.analysis import (
    ROUTING_LOAD_SCHEMA_VERSION,
    MixtralRoutingLoadMatrix,
    RoutingLoadError,
    RoutingLoadMatrix,
    aggregate_mixtral_routing_load,
    aggregate_routing_load,
)
from moeatlas.core import make_component_key, make_model_key, stable_digest
from moeatlas.events import EVENT_SCHEMA_VERSION, RoutingEvent, TokenEvent, TokenPhase
from moeatlas.store import (
    STORE_SCHEMA_VERSION,
    RoutingShardError,
    append_mixtral_routing_shard,
    list_mixtral_routing_runs,
)

from .test_mixtral_routing_decoder import _inspection, _qwen_inspection
from .test_qwen3_5_routing_decoder import _inspection as _qwen35_inspection
from .test_qwen3_5_routing_forward import _run_qwen
from .test_store_routing_shards import (
    _qwen_result,
    _refresh_manifest_file,
    _rewrite_parquet,
    _run_result,
    _tree_snapshot,
    _workspace,
)

try:
    import duckdb  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised without the store extra
    duckdb = None


def _aggregate(workspace: Path, inspection: object, run_key: str = "run-1", **limits: int):
    return aggregate_mixtral_routing_load(
        workspace,
        inspection,
        run_key=run_key,
        max_routing_rows=limits.get("max_routing_rows", 10_000),
        max_source_bytes=limits.get("max_source_bytes", 10_000_000),
        max_matrix_cells=limits.get("max_matrix_cells", 10_000),
    )


def _inspection_with_layer_indices(inspection: object, indices: tuple[int, ...]):
    report = inspection.report
    model_key = report.model_key
    replacements = dict(zip((0, 1), indices, strict=True))
    components = []
    for component in report.components:
        old_index = component.layer_index
        if old_index in replacements:
            new_index = replacements[old_index]
            key = make_component_key(
                model_key,
                component.kind.value,
                component.module_path,
                layer_index=new_index,
                expert_index=component.expert_index,
            )
            components.append(
                component.model_copy(update={"layer_index": new_index, "component_key": key})
            )
        else:
            components.append(component)
    tampered_report = report.model_copy(update={"components": components})
    return inspection.model_copy(update={"report": tampered_report})


def _inspection_with_many_layers(inspection: object, count: int = 11):
    """Clone the real two-layer static report into a valid numeric 0..count-1 report."""

    report = inspection.report
    model_key = report.model_key

    def clone(item: object, layer_index: int):
        module_path = item.module_path.replace("layers.0", f"layers.{layer_index}", 1)
        component_key = make_component_key(
            model_key,
            item.kind.value,
            module_path,
            layer_index=layer_index,
            expert_index=item.expert_index,
        )
        return item.model_copy(
            update={
                "component_key": component_key,
                "module_path": module_path,
                "layer_index": layer_index,
            }
        )

    layer_components = [item for item in report.components if item.layer_index == 0]
    static_components = [item for item in report.components if item.layer_index not in (0, 1)]
    layer_candidates = [item for item in report.candidates if item.layer_index == 0]
    static_candidates = [item for item in report.candidates if item.layer_index not in (0, 1)]
    components = [*static_components]
    candidates = [*static_candidates]
    for layer_index in range(count):
        components.extend(clone(item, layer_index) for item in layer_components)
        candidates.extend(clone(item, layer_index) for item in layer_candidates)
    tampered_report = report.model_copy(update={"components": components, "candidates": candidates})
    return inspection.model_copy(update={"report": tampered_report})


def _result_with_many_layers(result: object, inspection: object):
    routers = sorted(
        (item for item in inspection.report.components if item.kind.value == "router"),
        key=lambda item: item.layer_index,
    )
    layer_by_index = {
        item.layer_index: item
        for item in inspection.report.components
        if item.kind.value == "moe_layer"
    }
    experts_by_layer = {
        layer_index: tuple(
            item.component_key
            for item in sorted(
                (
                    expert
                    for expert in inspection.report.components
                    if expert.kind.value == "expert" and expert.layer_index == layer_index
                ),
                key=lambda item: item.expert_index,
            )
        )
        for layer_index in layer_by_index
    }
    routes_by_token = {}
    for token in result.token_events:
        routes_by_token[token.token_key] = sorted(
            (
                event
                for event in result.routing_events
                if event.token_key == token.token_key
                and event.layer_key
                == next(
                    item.component_key
                    for item in inspection.report.components
                    if item.kind.value == "moe_layer" and item.layer_index == 0
                )
            ),
            key=lambda event: event.rank,
        )
    routes: list[RoutingEvent] = []
    for token in result.token_events:
        templates = routes_by_token[token.token_key]
        for router in routers:
            layer_index = router.layer_index
            layer_key = layer_by_index[layer_index].component_key
            for template in templates:
                routes.append(
                    template.model_copy(
                        update={
                            "layer_key": layer_key,
                            "expert_key": experts_by_layer[layer_index][template.rank],
                        }
                    )
                )
    return result.__class__(object(), result.token_events, tuple(routes))


def _inspection_with_model_key(inspection: object, identifier: str):
    report = inspection.report
    model_key = make_model_key(identifier, "r1")

    def clone(item: object):
        component_key = make_component_key(
            model_key,
            item.kind.value,
            item.module_path,
            layer_index=item.layer_index,
            expert_index=item.expert_index,
        )
        return item.model_copy(update={"component_key": component_key, "model_key": model_key})

    model_manifest = report.model_manifest.model_copy(update={"model_key": model_key})
    tampered_report = report.model_copy(
        update={
            "model_key": model_key,
            "model_manifest": model_manifest,
            "components": [clone(item) for item in report.components],
            "candidates": [clone(item) for item in report.candidates],
        }
    )
    return inspection.model_copy(update={"report": tampered_report})


class _FetchOne:
    def __init__(self, value: object) -> None:
        self.value = value

    def fetchone(self) -> tuple[object]:
        return (self.value,)


class _ConnectionProxy:
    def __init__(
        self,
        connection: object,
        *,
        routing_count: int | None = None,
        count_failure: BaseException | None = None,
        failure: BaseException | None = None,
        close_failure: BaseException | None = None,
        close_calls: list[int],
    ) -> None:
        self.connection = connection
        self.routing_count = routing_count
        self.count_failure = count_failure
        self.failure = failure
        self.close_failure = close_failure
        self.close_calls = close_calls

    def execute(self, sql: str, parameters: list[str]):
        if self.count_failure is not None and "COUNT(*)" in sql:
            raise self.count_failure
        if self.failure is not None and "GROUP BY layer_key" in sql:
            raise self.failure
        if self.routing_count is not None and "COUNT(*)" in sql:
            if parameters and str(parameters[0]).endswith("routing.parquet"):
                return _FetchOne(self.routing_count)
        return self.connection.execute(sql, parameters)

    def close(self) -> None:
        self.close_calls.append(1)
        self.connection.close()
        if self.close_failure is not None:
            raise self.close_failure


class _DuckDBProxy:
    def __init__(
        self,
        *,
        routing_count: int | None = None,
        count_failure: BaseException | None = None,
        failure: BaseException | None = None,
        close_failure: BaseException | None = None,
        close_calls: list[int],
    ) -> None:
        self.__version__ = duckdb.__version__
        self.routing_count = routing_count
        self.count_failure = count_failure
        self.failure = failure
        self.close_failure = close_failure
        self.close_calls = close_calls

    def connect(self, *, database: str):
        return _ConnectionProxy(
            duckdb.connect(database=database),
            routing_count=self.routing_count,
            count_failure=self.count_failure,
            failure=self.failure,
            close_failure=self.close_failure,
            close_calls=self.close_calls,
        )


@pytest.fixture(autouse=True)
def _duckdb_required(request: pytest.FixtureRequest) -> None:
    if duckdb is None and request.node.get_closest_marker("no_duckdb") is None:
        if request.node.name not in {
            "test_public_surface_and_matrix_field_order",
            "test_matrix_constructor_revalidates_formula_and_axes",
            "test_inspection_revalidation_rejects_non_mixtral",
            "test_ast_and_forbidden_import_guards",
        }:
            pytest.skip("duckdb store extra is unavailable")


def test_public_surface_and_matrix_field_order() -> None:
    assert ROUTING_LOAD_SCHEMA_VERSION == "1.0"
    assert tuple(field.name for field in fields(MixtralRoutingLoadMatrix)) == (
        "schema_version",
        "store_schema_version",
        "event_schema_version",
        "run_key",
        "model_key",
        "adapter_name",
        "adapter_version",
        "inspection_digest",
        "layout",
        "shard_keys",
        "token_count",
        "assignment_count",
        "routed_top_k",
        "layer_keys",
        "layer_indices",
        "expert_keys",
        "assignment_counts",
        "assignment_shares",
        "load_ratios",
    )
    assert MixtralRoutingLoadMatrix.__slots__ == tuple(
        field.name for field in fields(MixtralRoutingLoadMatrix)
    )
    assert MixtralRoutingLoadMatrix.__dataclass_params__.frozen is True
    assert MixtralRoutingLoadMatrix.__dataclass_params__.eq is True
    assert tuple(inspect.signature(aggregate_mixtral_routing_load).parameters) == (
        "workspace",
        "inspection",
        "run_key",
        "max_routing_rows",
        "max_source_bytes",
        "max_matrix_cells",
    )
    signature = inspect.signature(aggregate_mixtral_routing_load)
    assert signature.parameters["run_key"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["max_routing_rows"].kind is inspect.Parameter.KEYWORD_ONLY


def test_matrix_constructor_revalidates_formula_and_axes() -> None:
    valid = MixtralRoutingLoadMatrix(
        schema_version="1.0",
        store_schema_version="1.0",
        event_schema_version="1.0",
        run_key="run-1",
        model_key="model:acme/mixtral@r1",
        adapter_name="huggingface-mixtral-static",
        adapter_version="1.0",
        inspection_digest="sha256:" + "0" * 64,
        layout="legacy_indexed",
        shard_keys=("shard:" + "1" * 64,),
        token_count=1,
        assignment_count=2,
        routed_top_k=2,
        layer_keys=("component:" + "2" * 64,),
        layer_indices=(0,),
        expert_keys=(tuple("component:" + digit * 64 for digit in "3456"),),
        assignment_counts=((1, 0, 1, 0),),
        assignment_shares=((0.5, 0.0, 0.5, 0.0),),
        load_ratios=((2.0, 0.0, 2.0, 0.0),),
    )
    assert valid.assignment_count == 2
    with pytest.raises((TypeError, ValueError)):
        replace(valid, assignment_counts=((1, 0, 1, 1),))
    with pytest.raises((TypeError, ValueError)):
        replace(valid, assignment_shares=((0.25, 0.0, 0.25, 0.500001),))
    with pytest.raises((TypeError, ValueError)):
        replace(valid, layer_indices=(0, 0))
    with pytest.raises((TypeError, ValueError)):
        replace(valid, expert_keys=((valid.expert_keys[0][0],) * 4,))
    with pytest.raises((TypeError, ValueError)):
        replace(valid, assignment_counts=((True, 0, 1, 2),))

    with pytest.raises((TypeError, ValueError)):
        replace(
            valid,
            assignment_count=4,
            layer_keys=(valid.layer_keys[0], "component:" + "7" * 64),
            layer_indices=(0, 1),
            expert_keys=(valid.expert_keys[0], valid.expert_keys[0]),
            assignment_counts=(valid.assignment_counts[0], valid.assignment_counts[0]),
            assignment_shares=(valid.assignment_shares[0], valid.assignment_shares[0]),
            load_ratios=(valid.load_ratios[0], valid.load_ratios[0]),
        )


def test_inspection_revalidation_rejects_non_mixtral() -> None:
    with pytest.raises(RoutingLoadError) as caught:
        _aggregate(Path("."), _qwen_inspection())
    assert caught.value.stage == "inspection"
    inspection = _inspection("legacy")
    descriptor = inspection.descriptor.model_copy(update={"name": "qwen-static"})
    tampered = inspection.model_copy(update={"descriptor": descriptor})
    with pytest.raises(RoutingLoadError) as caught:
        _aggregate(Path("."), tampered)
    assert caught.value.stage == "inspection"


@pytest.mark.parametrize("indices", [(0, 2), (2, 10), (10, 2)])
def test_inspection_axes_require_numeric_contiguous_layers(indices: tuple[int, int]) -> None:
    tampered = _inspection_with_layer_indices(_inspection("legacy"), indices)
    with pytest.raises(RoutingLoadError) as caught:
        _aggregate(Path("."), tampered)
    assert caught.value.stage == "inspection"


@pytest.mark.parametrize("layout", ["legacy", "packed"])
def test_known_layout_formulas_and_explicit_zero_experts(layout: str, tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    result, _, inspection = _run_result(layout, token_count=2)
    receipt = append_mixtral_routing_shard(workspace, result)
    matrix = _aggregate(workspace, inspection, receipt.run_key)
    assert matrix.layout == ("legacy_indexed" if layout == "legacy" else "packed")
    assert matrix.schema_version == ROUTING_LOAD_SCHEMA_VERSION
    assert matrix.store_schema_version == STORE_SCHEMA_VERSION
    assert matrix.event_schema_version == EVENT_SCHEMA_VERSION
    assert matrix.model_key == inspection.report.model_key
    assert matrix.adapter_name == "huggingface-mixtral-static"
    assert matrix.adapter_version == "1.0"
    assert matrix.inspection_digest == "sha256:" + stable_digest(inspection.model_dump(mode="json"))
    assert matrix.token_count == 2
    assert matrix.assignment_count == 2 * len(matrix.layer_keys) * matrix.routed_top_k
    assert matrix.shard_keys == (receipt.shard_key,)
    assert matrix.layer_indices == tuple(sorted(matrix.layer_indices))
    router_layers = sorted(
        {
            component.layer_index
            for component in inspection.report.components
            if component.kind.value == "router"
        }
    )
    expected_layer_keys = tuple(
        next(
            component.component_key
            for component in inspection.report.components
            if component.kind.value == "moe_layer" and component.layer_index == layer_index
        )
        for layer_index in router_layers
    )
    expected_expert_keys = tuple(
        tuple(
            component.component_key
            for component in sorted(
                (
                    component
                    for component in inspection.report.components
                    if component.kind.value == "expert" and component.layer_index == layer_index
                ),
                key=lambda component: component.expert_index,
            )
        )
        for layer_index in router_layers
    )
    assert matrix.layer_indices == tuple(router_layers)
    assert matrix.layer_keys == expected_layer_keys
    assert matrix.expert_keys == expected_expert_keys
    assert all(
        sum(row) == matrix.token_count * matrix.routed_top_k for row in matrix.assignment_counts
    )
    assert matrix.assignment_counts == ((0, 2, 0, 2), (0, 2, 0, 2))
    assert any(count == 0 for row in matrix.assignment_counts for count in row)
    for counts, shares, ratios in zip(
        matrix.assignment_counts, matrix.assignment_shares, matrix.load_ratios, strict=True
    ):
        assert sum(shares) == pytest.approx(1.0, abs=1e-12)
        assert sum(ratios) / len(ratios) == pytest.approx(1.0, abs=1e-12)
        for count, share, ratio in zip(counts, shares, ratios, strict=True):
            assert share == pytest.approx(count / (matrix.token_count * matrix.routed_top_k))
            assert ratio == share * len(ratios)
            if count == 0:
                assert share == 0.0 and ratio == 0.0


def test_budget_validation_precedes_dependency_and_filesystem(tmp_path: Path, monkeypatch) -> None:
    workspace = _workspace(tmp_path)
    inspection = _inspection("legacy")
    before = _tree_snapshot(workspace)

    def forbidden_loader() -> object:
        raise AssertionError("duckdb import must be deferred after cell budget")

    monkeypatch.setattr(storage, "_load_duckdb", forbidden_loader)
    with pytest.raises(RoutingLoadError) as caught:
        _aggregate(workspace, inspection, max_matrix_cells=1)
    assert caught.value.stage == "budget"
    assert _tree_snapshot(workspace) == before


def test_source_budgets_and_workspace_preflight_do_not_write(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    result, _, inspection = _run_result("legacy", token_count=1)
    receipt = append_mixtral_routing_shard(workspace, result)
    before = _tree_snapshot(workspace)
    with pytest.raises(RoutingLoadError) as caught:
        _aggregate(workspace, inspection, receipt.run_key, max_source_bytes=1)
    assert caught.value.stage == "budget"
    assert _tree_snapshot(workspace) == before


def test_source_budget_exact_boundaries_and_readonly_success(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    result, _, inspection = _run_result("legacy", token_count=1)
    receipt = append_mixtral_routing_shard(workspace, result)
    shard = workspace / receipt.relative_path
    source_bytes = sum(
        (shard / name).stat().st_size
        for name in ("manifest.json", "tokens.parquet", "routing.parquet")
    )
    before_tree = _tree_snapshot(workspace)
    before_bytes = {
        path.relative_to(workspace).as_posix(): path.read_bytes()
        for path in workspace.rglob("*")
        if path.is_file()
    }
    matrix = _aggregate(
        workspace,
        inspection,
        receipt.run_key,
        max_routing_rows=receipt.routing_count,
        max_source_bytes=source_bytes,
    )
    after_bytes = {
        path.relative_to(workspace).as_posix(): path.read_bytes()
        for path in workspace.rglob("*")
        if path.is_file()
    }
    assert matrix.assignment_count == receipt.routing_count
    assert _tree_snapshot(workspace) == before_tree
    assert after_bytes == before_bytes


def test_actual_routing_budget_precedes_manifest_and_source_fetchall(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = _workspace(tmp_path)
    result, _, inspection = _run_result("legacy", token_count=1)
    receipt = append_mixtral_routing_shard(workspace, result)
    close_calls: list[int] = []
    proxy = _DuckDBProxy(routing_count=999, close_calls=close_calls)
    monkeypatch.setattr(storage, "_load_duckdb", lambda: proxy)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("source materialization must follow actual row budget")

    monkeypatch.setattr(storage, "_validate_routing_load_source", forbidden)
    with pytest.raises(RoutingLoadError) as caught:
        _aggregate(workspace, inspection, receipt.run_key, max_routing_rows=4)
    assert caught.value.stage == "budget"
    assert close_calls == [1]


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (OSError("parquet parse"), RoutingShardError),
        (KeyboardInterrupt(), KeyboardInterrupt),
        (SystemExit(9), SystemExit),
    ],
)
def test_count_read_failures_are_reopen_or_exact_control_flow(
    failure: BaseException, expected: type[BaseException], tmp_path: Path, monkeypatch
) -> None:
    workspace = _workspace(tmp_path)
    result, _, inspection = _run_result("legacy", token_count=1)
    receipt = append_mixtral_routing_shard(workspace, result)
    close_calls: list[int] = []
    proxy = _DuckDBProxy(count_failure=failure, close_calls=close_calls)
    monkeypatch.setattr(storage, "_load_duckdb", lambda: proxy)
    with pytest.raises(expected) as caught:
        _aggregate(workspace, inspection, receipt.run_key)
    if isinstance(failure, OSError):
        assert caught.value.stage == "reopen"
    assert close_calls == [1]


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (OSError("query"), RoutingLoadError),
        (KeyboardInterrupt(), KeyboardInterrupt),
        (SystemExit(7), SystemExit),
    ],
)
def test_connection_proxy_primary_failures_close_once(
    failure: BaseException, expected: type[BaseException], tmp_path: Path, monkeypatch
) -> None:
    workspace = _workspace(tmp_path)
    result, _, inspection = _run_result("legacy", token_count=1)
    receipt = append_mixtral_routing_shard(workspace, result)
    close_calls: list[int] = []
    proxy = _DuckDBProxy(failure=failure, close_calls=close_calls)
    monkeypatch.setattr(storage, "_load_duckdb", lambda: proxy)
    with pytest.raises(expected) as caught:
        _aggregate(workspace, inspection, receipt.run_key)
    if isinstance(failure, OSError):
        assert isinstance(caught.value, RoutingLoadError)
        assert caught.value.stage == "query"
    assert close_calls == [1]


def test_connection_proxy_close_failure_and_primary_are_terminal(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = _workspace(tmp_path)
    result, _, inspection = _run_result("legacy", token_count=1)
    receipt = append_mixtral_routing_shard(workspace, result)
    close_calls: list[int] = []
    proxy = _DuckDBProxy(close_failure=OSError("close"), close_calls=close_calls)
    monkeypatch.setattr(storage, "_load_duckdb", lambda: proxy)
    with pytest.raises(RoutingLoadError) as caught:
        _aggregate(workspace, inspection, receipt.run_key)
    assert caught.value.stage == "query"
    assert close_calls == [1]


def test_connection_proxy_success_closes_once_and_returns_matrix(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = _workspace(tmp_path)
    result, _, inspection = _run_result("legacy", token_count=1)
    receipt = append_mixtral_routing_shard(workspace, result)
    close_calls: list[int] = []
    proxy = _DuckDBProxy(close_calls=close_calls)
    monkeypatch.setattr(storage, "_load_duckdb", lambda: proxy)
    matrix = _aggregate(workspace, inspection, receipt.run_key)
    assert isinstance(matrix, MixtralRoutingLoadMatrix)
    assert close_calls == [1]


@pytest.mark.parametrize("failure", [KeyboardInterrupt(), SystemExit(13)])
def test_connection_proxy_close_control_flow_is_exact_and_publishes_no_matrix(
    failure: BaseException, tmp_path: Path, monkeypatch
) -> None:
    workspace = _workspace(tmp_path)
    result, _, inspection = _run_result("legacy", token_count=1)
    receipt = append_mixtral_routing_shard(workspace, result)
    close_calls: list[int] = []
    proxy = _DuckDBProxy(close_failure=failure, close_calls=close_calls)
    monkeypatch.setattr(storage, "_load_duckdb", lambda: proxy)
    with pytest.raises(type(failure)) as caught:
        _aggregate(workspace, inspection, receipt.run_key)
    assert caught.value is failure
    assert close_calls == [1]

    close_calls.clear()
    proxy = _DuckDBProxy(
        failure=OSError("primary"), close_failure=OSError("close"), close_calls=close_calls
    )
    monkeypatch.setattr(storage, "_load_duckdb", lambda: proxy)
    with pytest.raises(RoutingLoadError) as caught:
        _aggregate(workspace, inspection, receipt.run_key)
    assert caught.value.stage == "query"
    assert close_calls == [1]


def test_dependency_failure_is_lazy_and_nonmutating(tmp_path: Path, monkeypatch) -> None:
    workspace = _workspace(tmp_path)
    result, _, inspection = _run_result("legacy", token_count=1)
    receipt = append_mixtral_routing_shard(workspace, result)
    before = _tree_snapshot(workspace)

    def unavailable() -> object:
        raise RoutingShardError("dependency")

    monkeypatch.setattr(storage, "_load_duckdb", unavailable)
    with pytest.raises(RoutingShardError) as caught:
        _aggregate(workspace, inspection, receipt.run_key)
    assert caught.value.stage == "dependency"
    assert _tree_snapshot(workspace) == before


def test_multiple_shards_are_read_across_the_run_and_sorted(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    first, _, inspection = _run_result("legacy", token_count=1)
    first_receipt = append_mixtral_routing_shard(workspace, first)
    token = TokenEvent(
        run_key="run-1",
        sequence_id="sequence-2",
        token_pos=0,
        token_id=99,
        token_text="redacted",
        phase=TokenPhase.DECODE,
    )
    routes = tuple(
        event.model_copy(update={"token_key": token.token_key}) for event in first.routing_events
    )
    second = first.__class__(object(), (token,), routes)
    second_receipt = append_mixtral_routing_shard(workspace, second)
    matrix = _aggregate(workspace, inspection, first_receipt.run_key)
    assert matrix.token_count == 2
    assert matrix.assignment_count == 2 * len(matrix.layer_keys) * matrix.routed_top_k
    assert matrix.shard_keys == tuple(sorted((first_receipt.shard_key, second_receipt.shard_key)))
    with pytest.raises(RoutingLoadError) as caught:
        _aggregate(workspace, inspection, first_receipt.run_key, max_routing_rows=4)
    assert caught.value.stage == "budget"


def test_absent_run_is_a_source_failure_without_workspace_mutation(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    inspection = _inspection("legacy")
    before = _tree_snapshot(workspace)
    with pytest.raises(RoutingLoadError) as caught:
        _aggregate(workspace, inspection, "run-with-no-shards")
    assert caught.value.stage == "source"
    assert _tree_snapshot(workspace) == before


def test_valid_contiguous_eleven_layer_source_uses_numeric_axis_order(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    base_result, _, base_inspection = _run_result("legacy", token_count=1)
    inspection = _inspection_with_many_layers(base_inspection)
    result = _result_with_many_layers(base_result, inspection)
    receipt = append_mixtral_routing_shard(workspace, result)
    matrix = _aggregate(workspace, inspection, receipt.run_key)
    assert matrix.layer_indices == tuple(range(11))
    assert matrix.layer_indices[2] == 2
    assert matrix.layer_indices[10] == 10
    assert matrix.layer_keys[2] != matrix.layer_keys[10]
    first_layer_key = next(
        item.component_key
        for item in inspection.report.components
        if item.kind.value == "moe_layer" and item.layer_index == 0
    )
    expected_row = [0] * len(
        tuple(
            item
            for item in inspection.report.components
            if item.kind.value == "expert" and item.layer_index == 0
        )
    )
    for event in base_result.routing_events:
        if event.layer_key == first_layer_key:
            expected_row[event.rank] += 1
    assert matrix.assignment_counts == tuple(tuple(expected_row) for _ in range(11))


def test_different_exact_mixtral_model_binding_is_rejected_at_source(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    result, _, inspection = _run_result("legacy", token_count=1)
    receipt = append_mixtral_routing_shard(workspace, result)
    different = _inspection_with_model_key(inspection, "other/mixtral")
    assert different.descriptor.name == "huggingface-mixtral-static"
    assert different.descriptor.architecture_families == ("mixtral",)
    assert different.report.model_key != inspection.report.model_key
    with pytest.raises(RoutingLoadError) as caught:
        _aggregate(workspace, different, receipt.run_key)
    assert caught.value.stage == "source"


def test_repeated_aggregation_is_value_equal(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    result, _, inspection = _run_result("legacy", token_count=1)
    receipt = append_mixtral_routing_shard(workspace, result)
    assert _aggregate(workspace, inspection, receipt.run_key) == _aggregate(
        workspace, inspection, receipt.run_key
    )


def test_redacted_and_opt_in_token_text_have_equal_routing_matrices(tmp_path: Path) -> None:
    (tmp_path / "redacted").mkdir()
    (tmp_path / "stored").mkdir()
    redacted_workspace = _workspace(tmp_path / "redacted")
    stored_workspace = _workspace(tmp_path / "stored")
    result, _, inspection = _run_result("legacy", token_count=1)
    redacted_receipt = append_mixtral_routing_shard(redacted_workspace, result)
    stored_receipt = append_mixtral_routing_shard(stored_workspace, result, store_token_text=True)
    redacted = _aggregate(redacted_workspace, inspection, redacted_receipt.run_key)
    stored = _aggregate(stored_workspace, inspection, stored_receipt.run_key)
    assert replace(redacted, shard_keys=stored.shard_keys) == stored


def test_packed_source_is_incompatible_with_legacy_inspection(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    result, _, packed_inspection = _run_result("packed", token_count=1)
    receipt = append_mixtral_routing_shard(workspace, result)
    legacy_inspection = _inspection("legacy")
    assert packed_inspection.report.model_key == legacy_inspection.report.model_key
    with pytest.raises(RoutingLoadError) as caught:
        _aggregate(workspace, legacy_inspection, receipt.run_key)
    assert caught.value.stage == "source"


@pytest.mark.parametrize("inspection_side", ["legacy", "packed"])
def test_same_run_mixed_layout_shards_with_distinct_tokens_are_incompatible(
    inspection_side: str, tmp_path: Path
) -> None:
    workspace = _workspace(tmp_path)
    legacy_result, _, legacy_inspection = _run_result("legacy", token_count=1)
    packed_result, _, packed_inspection = _run_result("packed", token_count=1)
    packed_token = TokenEvent(
        run_key="run-1",
        sequence_id="sequence-2",
        token_pos=0,
        token_id=99,
        token_text="other",
        phase=TokenPhase.DECODE,
    )
    packed_routes = tuple(
        event.model_copy(update={"token_key": packed_token.token_key})
        for event in packed_result.routing_events
    )
    packed_distinct = packed_result.__class__(object(), (packed_token,), packed_routes)
    legacy_receipt = append_mixtral_routing_shard(workspace, legacy_result)
    packed_receipt = append_mixtral_routing_shard(workspace, packed_distinct)
    assert legacy_receipt.run_key == packed_receipt.run_key == "run-1"
    assert legacy_receipt.shard_key != packed_receipt.shard_key
    inspection = legacy_inspection if inspection_side == "legacy" else packed_inspection
    before = _tree_snapshot(workspace)
    with pytest.raises(RoutingLoadError) as caught:
        _aggregate(workspace, inspection, "run-1")
    assert caught.value.stage == "source"
    assert _tree_snapshot(workspace) == before


@pytest.mark.parametrize("field", ["max_routing_rows", "max_source_bytes", "max_matrix_cells"])
@pytest.mark.parametrize("bad", [0, -1, True, 1.0, "1"])
def test_budget_arguments_are_strict_and_nonmutating(
    field: str, bad: object, tmp_path: Path
) -> None:
    workspace = _workspace(tmp_path)
    inspection = _inspection("legacy")
    before = _tree_snapshot(workspace)
    values: dict[str, object] = {
        "max_routing_rows": 100,
        "max_source_bytes": 100_000,
        "max_matrix_cells": 100,
    }
    values[field] = bad
    with pytest.raises((TypeError, ValueError)):
        aggregate_mixtral_routing_load(workspace, inspection, run_key="run-1", **values)
    assert _tree_snapshot(workspace) == before


def test_source_corruption_is_reopen_and_duplicate_is_conflict(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    result, _, inspection = _run_result("packed", token_count=1)
    receipt = append_mixtral_routing_shard(workspace, result)
    shard = workspace / receipt.relative_path
    (shard / "routing.parquet").write_bytes((shard / "routing.parquet").read_bytes() + b"tamper")
    with pytest.raises(RoutingShardError) as caught:
        _aggregate(workspace, inspection, receipt.run_key)
    assert caught.value.stage == "reopen"
    with pytest.raises(TypeError):
        aggregate_mixtral_routing_load(
            123,
            inspection,
            run_key=receipt.run_key,
            max_routing_rows=1,
            max_source_bytes=1,
            max_matrix_cells=1,
        )
    with pytest.raises(TypeError):
        aggregate_mixtral_routing_load(
            workspace,
            inspection,
            run_key=[receipt.run_key],
            max_routing_rows=1,
            max_source_bytes=1,
            max_matrix_cells=1,
        )


def test_valid_checksum_duplicate_expert_per_token_layer_is_rejected(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    result, _, inspection = _run_result("legacy", token_count=1)
    routes = list(result.routing_events)
    routes[1] = routes[1].model_copy(update={"expert_key": routes[0].expert_key})
    duplicate = result.__class__(object(), result.token_events, tuple(routes))
    receipt = append_mixtral_routing_shard(workspace, duplicate)
    with pytest.raises(RoutingLoadError) as caught:
        _aggregate(workspace, inspection, receipt.run_key)
    assert caught.value.stage == "source"


@pytest.mark.parametrize(
    "mutation",
    ["rank_gap", "rank_out_of_range", "missing_layer", "unknown_layer", "unknown_expert"],
)
def test_source_completeness_rejects_rank_layer_and_identity_mismatches(
    mutation: str, tmp_path: Path
) -> None:
    workspace = _workspace(tmp_path)
    result, _, inspection = _run_result("packed", token_count=1)
    routes = list(result.routing_events)
    if mutation == "rank_gap":
        routes = routes[1:]
    elif mutation == "rank_out_of_range":
        routes[1] = routes[1].model_copy(update={"rank": 2})
    elif mutation == "missing_layer":
        missing_key = routes[-1].layer_key
        routes = [event for event in routes if event.layer_key != missing_key]
    elif mutation == "unknown_layer":
        routes[0] = routes[0].model_copy(update={"layer_key": "component:" + "f" * 64})
    else:
        routes[0] = routes[0].model_copy(update={"expert_key": "component:" + "e" * 64})
    invalid = result.__class__(object(), result.token_events, tuple(routes))
    receipt = append_mixtral_routing_shard(workspace, invalid)
    with pytest.raises(RoutingLoadError) as caught:
        _aggregate(workspace, inspection, receipt.run_key)
    assert caught.value.stage == "source"


def test_valid_checksum_duplicate_rank_is_reopen(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    result, _, inspection = _run_result("packed", token_count=1)
    receipt = append_mixtral_routing_shard(workspace, result)
    routing_path = workspace / receipt.relative_path / "routing.parquet"
    connection = duckdb.connect(database=":memory:")
    try:
        rows = tuple(
            connection.execute(
                "SELECT * FROM read_parquet(?) ORDER BY event_index", [str(routing_path)]
            ).fetchall()
        )
    finally:
        connection.close()
    altered = list(rows[1])
    altered[7] = rows[0][7]
    _rewrite_parquet(routing_path, storage._ROUTING_COLUMNS, (rows[0], tuple(altered), *rows[2:]))
    _refresh_manifest_file(receipt, workspace, "routing.parquet")
    with pytest.raises(RoutingShardError) as caught:
        _aggregate(workspace, inspection, receipt.run_key)
    assert caught.value.stage == "reopen"


@pytest.mark.parametrize("corruption", ["malformed", "wrong_schema", "row_count", "semantic"])
def test_valid_checksum_source_corruption_is_reopen(corruption: str, tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    result, _, inspection = _run_result("legacy", token_count=1)
    receipt = append_mixtral_routing_shard(workspace, result)
    shard = workspace / receipt.relative_path
    routing_path = shard / "routing.parquet"
    if corruption == "malformed":
        routing_path.write_bytes(b"not parquet")
    else:
        connection = duckdb.connect(database=":memory:")
        try:
            rows = tuple(
                connection.execute(
                    "SELECT * FROM read_parquet(?) ORDER BY event_index", [str(routing_path)]
                ).fetchall()
            )
        finally:
            connection.close()
        if corruption == "wrong_schema":
            columns = list(storage._ROUTING_COLUMNS)
            columns[0] = ("wrong_store_schema_version", columns[0][1])
            _rewrite_parquet(routing_path, tuple(columns), rows)
        elif corruption == "row_count":
            _rewrite_parquet(routing_path, storage._ROUTING_COLUMNS, rows[:-1])
        else:
            altered = list(rows[0])
            altered[9] = float(altered[9]) + 1.0
            _rewrite_parquet(routing_path, storage._ROUTING_COLUMNS, (tuple(altered), *rows[1:]))
    _refresh_manifest_file(receipt, workspace, "routing.parquet")
    with pytest.raises(RoutingShardError) as caught:
        _aggregate(workspace, inspection, receipt.run_key)
    assert caught.value.stage == "reopen"


def test_query_failures_are_safe_and_connection_is_closed(tmp_path: Path, monkeypatch) -> None:
    workspace = _workspace(tmp_path)
    result, _, inspection = _run_result("legacy", token_count=1)
    receipt = append_mixtral_routing_shard(workspace, result)
    original = storage._validate_routing_load_source

    def fail(*args: object, **kwargs: object) -> object:
        raise OSError("query internals must not leak")

    monkeypatch.setattr(storage, "_validate_routing_load_source", fail)
    with pytest.raises(RoutingLoadError) as caught:
        _aggregate(workspace, inspection, receipt.run_key)
    assert caught.value.stage == "query"
    assert str(caught.value) == "routing load aggregation failed at query"
    monkeypatch.setattr(storage, "_validate_routing_load_source", original)


def test_network_cache_and_tmp_are_not_used(tmp_path: Path, monkeypatch) -> None:
    workspace = _workspace(tmp_path)
    result, _, inspection = _run_result("legacy", token_count=1)
    receipt = append_mixtral_routing_shard(workspace, result)
    cache = tmp_path / "cache"
    temp = tmp_path / "temp"
    cache.mkdir()
    temp.mkdir()
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
    monkeypatch.setenv("TMPDIR", str(temp))

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    before_cache = _tree_snapshot(cache)
    before_temp = _tree_snapshot(temp)
    _aggregate(workspace, inspection, receipt.run_key)
    assert _tree_snapshot(cache) == before_cache
    assert _tree_snapshot(temp) == before_temp


def test_matrix_does_not_retain_inspection_or_source_objects(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    result, _, inspection = _run_result("legacy", token_count=1)
    receipt = append_mixtral_routing_shard(workspace, result)
    inspection_ref = weakref.ref(inspection)
    matrix = _aggregate(workspace, inspection, receipt.run_key)
    del result
    del inspection
    gc.collect()
    assert inspection_ref() is None
    assert matrix.run_key == receipt.run_key
    assert all(isinstance(key, str) for key in matrix.shard_keys)


def test_ast_and_forbidden_import_guards() -> None:
    source_path = Path("src/moeatlas/analysis/routing_load.py")
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
        "importlib",
    }
    forbidden_filesystem_modules = {"os", "shutil", "tempfile"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] not in forbidden for alias in node.names)
            assert all(
                alias.name.split(".")[0] not in forbidden_filesystem_modules for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in forbidden
            assert (node.module or "").split(".")[0] not in forbidden_filesystem_modules
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "import_module":
                raise AssertionError("dynamic imports are forbidden")
            if isinstance(node.func, ast.Attribute) and node.func.attr == "format":
                raise AssertionError("SQL/path formatting is forbidden")
            if isinstance(node.func, ast.Attribute) and node.func.attr in {
                "execute",
                "executemany",
            }:
                assert node.args and isinstance(node.args[0], ast.Constant)
                assert isinstance(node.args[0].value, str)
                assert not isinstance(node.args[0], ast.JoinedStr)
                sql = node.args[0].value
                if "read_parquet" in sql or "GROUP BY" in sql:
                    assert "?" in sql
                    assert len(node.args) >= 2
            if isinstance(node.func, ast.Attribute) and node.func.attr in {
                "write",
                "write_text",
                "write_bytes",
                "mkdir",
                "unlink",
                "rename",
                "replace",
                "touch",
                "rmdir",
                "chmod",
            }:
                raise AssertionError("analysis must not mutate the filesystem")
            if isinstance(node.func, ast.Name | ast.Attribute) and (
                node.func.id == "open"
                if isinstance(node.func, ast.Name)
                else node.func.attr == "open"
            ):
                mode_nodes = []
                if len(node.args) > 1:
                    mode_nodes.append(node.args[1])
                mode_nodes.extend(
                    keyword.value for keyword in node.keywords if keyword.arg == "mode"
                )
                for mode_node in mode_nodes:
                    if isinstance(mode_node, ast.Constant) and isinstance(mode_node.value, str):
                        mode = mode_node.value.lower()
                        assert not any(
                            marker in mode
                            for marker in (
                                "w",
                                "a",
                                "x",
                                "+",
                                "write",
                                "append",
                                "create",
                                "update",
                            )
                        )
            if isinstance(node.func, ast.Attribute):
                root = node.func.value
                while isinstance(root, ast.Attribute):
                    root = root.value
                if isinstance(root, ast.Name):
                    assert root.id not in forbidden_filesystem_modules
        elif isinstance(node, ast.JoinedStr):
            rendered = "".join(part.value for part in node.values if isinstance(part, ast.Constant))
            assert not any(term in rendered.lower() for term in ("select", "parquet", "path"))
    assert "glob(" not in source
    assert "token_text" not in source
    assert "output" not in source
    assert 'database=":memory:"' in source
    assert source.count("database=") == 1
    assert "CREATE TABLE" not in source
    assert "INSERT INTO" not in source
    assert "write_parquet" not in source


def test_neutral_public_surface_and_historical_mixtral_names_are_identity_aliases() -> None:
    assert MixtralRoutingLoadMatrix is RoutingLoadMatrix
    assert aggregate_mixtral_routing_load is aggregate_routing_load


def test_qwen35_shared_expert_is_validated_but_excluded_from_neutral_matrix(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    inspection = _qwen35_inspection()
    result = _qwen_result()
    receipt = append_mixtral_routing_shard(workspace, result)
    matrix = aggregate_routing_load(
        workspace,
        inspection,
        run_key=receipt.run_key,
        max_routing_rows=1000,
        max_source_bytes=10_000_000,
        max_matrix_cells=1000,
    )
    assert type(matrix) is RoutingLoadMatrix
    assert matrix.adapter_name == inspection.descriptor.name
    assert len(matrix.expert_keys[0]) == inspection.report.facts.expert_count
    assert all(
        shared.component_key not in matrix.expert_keys[layer]
        for shared in inspection.report.components
        if shared.kind.value == "shared_expert"
        for layer in range(len(matrix.expert_keys))
    )
    assert matrix.assignment_count == (
        matrix.token_count * len(matrix.layer_keys) * matrix.routed_top_k
    )


@pytest.mark.parametrize("surface", ["conditional", "text"])
def test_qwen35_both_roots_cross_the_full_forward_store_inventory_and_analysis_path(
    surface: str, tmp_path: Path
) -> None:
    workspace = _workspace(tmp_path)
    result, _, inspection, _ = _run_qwen(surface)
    receipt = append_mixtral_routing_shard(workspace, result)
    inventory = list_mixtral_routing_runs(
        workspace,
        max_runs=10,
        max_shards=10,
        max_event_rows=1000,
        max_source_bytes=10_000_000,
    )
    matrix = aggregate_routing_load(
        workspace,
        inspection,
        run_key=receipt.run_key,
        max_routing_rows=1000,
        max_source_bytes=10_000_000,
        max_matrix_cells=1000,
    )
    assert inventory.run_count == 1
    assert inventory.routing_count == receipt.routing_count
    assert matrix.model_key == inspection.report.model_key
    assert matrix.assignment_count == receipt.routing_count


def test_future_descriptor_is_accepted_only_by_structural_routing_contract(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    original = _qwen35_inspection()
    descriptor = original.descriptor.model_copy(
        update={
            "name": "future-moe-static",
            "version": "2.0",
            "architecture_families": ("future_moe",),
        }
    )
    components = [
        component.model_copy(
            update={
                "capture": component.capture.model_copy(
                    update={"adapter": descriptor.name, "adapter_version": descriptor.version}
                )
                if component.capture is not None
                else None
            }
        )
        for component in original.report.components
    ]
    inspection = original.model_copy(
        update={
            "descriptor": descriptor,
            "report": original.report.model_copy(update={"components": components}),
        }
    )
    result = _qwen_result()
    receipt = append_mixtral_routing_shard(workspace, result)
    matrix = aggregate_routing_load(
        workspace,
        inspection,
        run_key=receipt.run_key,
        max_routing_rows=1000,
        max_source_bytes=10_000_000,
        max_matrix_cells=1000,
    )
    assert matrix.adapter_name == "future-moe-static"
    assert matrix.adapter_version == "2.0"
