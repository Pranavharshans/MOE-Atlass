from __future__ import annotations

import ast
import gc
import inspect
import socket
import sys
import urllib.request
from dataclasses import replace
from html.parser import HTMLParser
from pathlib import Path

import pytest

import moeatlas.analysis.routing_heatmap as heatmap
from moeatlas.analysis import (
    ROUTING_HEATMAP_SCHEMA_VERSION,
    RoutingLoadMatrix,
    render_compact_routing_load_heatmap,
    render_routing_load_heatmap,
)
from moeatlas.events import EVENT_SCHEMA_VERSION
from moeatlas.store import STORE_SCHEMA_VERSION


def _component(digit: str) -> str:
    return "component:" + digit * 64


def _shard(digit: str) -> str:
    return "shard:" + digit * 64


def _matrix(layout: str = "legacy_indexed") -> RoutingLoadMatrix:
    layer_keys = (_component("a"), _component("b"))
    expert_keys = (
        (_component("c"), _component("d"), _component("e"), _component("f")),
        (_component("1"), _component("2"), _component("3"), _component("4")),
    )
    counts = ((0, 2, 0, 2), (1, 1, 1, 1))
    shares = ((0.0, 0.5, 0.0, 0.5), (0.25, 0.25, 0.25, 0.25))
    ratios = ((0.0, 2.0, 0.0, 2.0), (1.0, 1.0, 1.0, 1.0))
    return RoutingLoadMatrix(
        schema_version="1.0",
        store_schema_version=STORE_SCHEMA_VERSION,
        event_schema_version=EVENT_SCHEMA_VERSION,
        run_key="run-heatmap",
        model_key="model:acme/mixtral@r1",
        adapter_name="huggingface-mixtral-static",
        adapter_version="1.0",
        inspection_digest="sha256:" + "1" * 64,
        layout=layout,
        shard_keys=(_shard("1"), _shard("2")),
        token_count=2,
        assignment_count=8,
        routed_top_k=2,
        layer_keys=layer_keys,
        layer_indices=(0, 1),
        expert_keys=expert_keys,
        assignment_counts=counts,
        assignment_shares=shares,
        load_ratios=ratios,
    )


def _render(
    metric: str = "assignment_counts", matrix: object | None = None, *, max_cells: int = 8
) -> str:
    return render_routing_load_heatmap(
        _matrix() if matrix is None else matrix,
        metric=metric,
        max_cells=max_cells,
    )


def _hex_component(value: int) -> str:
    return f"component:{value:064x}"


def _many_layer_matrix() -> RoutingLoadMatrix:
    layer_keys = tuple(_hex_component(100 + index) for index in range(11))
    expert_keys = tuple(
        tuple(_hex_component(1000 + layer * 4 + expert) for expert in range(4))
        for layer in range(11)
    )
    counts = tuple((0, 2, 0, 2) for _ in range(11))
    shares = tuple((0.0, 0.5, 0.0, 0.5) for _ in range(11))
    ratios = tuple((0.0, 2.0, 0.0, 2.0) for _ in range(11))
    return RoutingLoadMatrix(
        schema_version="1.0",
        store_schema_version=STORE_SCHEMA_VERSION,
        event_schema_version=EVENT_SCHEMA_VERSION,
        run_key="run-many-layers",
        model_key="model:acme/mixtral@r1",
        adapter_name="huggingface-mixtral-static",
        adapter_version="1.0",
        inspection_digest="sha256:" + "2" * 64,
        layout="legacy_indexed",
        shard_keys=(_shard("3"),),
        token_count=2,
        assignment_count=44,
        routed_top_k=2,
        layer_keys=layer_keys,
        layer_indices=tuple(range(11)),
        expert_keys=expert_keys,
        assignment_counts=counts,
        assignment_shares=shares,
        load_ratios=ratios,
    )


def _tiny_positive_matrix() -> RoutingLoadMatrix:
    return RoutingLoadMatrix(
        schema_version="1.0",
        store_schema_version=STORE_SCHEMA_VERSION,
        event_schema_version=EVENT_SCHEMA_VERSION,
        run_key="run-tiny-positive",
        model_key="model:acme/mixtral@r1",
        adapter_name="huggingface-mixtral-static",
        adapter_version="1.0",
        inspection_digest="sha256:" + "3" * 64,
        layout="packed",
        shard_keys=(_shard("4"),),
        token_count=100,
        assignment_count=200,
        routed_top_k=2,
        layer_keys=(_hex_component(200),),
        layer_indices=(0,),
        expert_keys=(
            (_hex_component(300), _hex_component(301), _hex_component(302), _hex_component(303)),
        ),
        assignment_counts=((1, 0, 0, 199),),
        assignment_shares=((0.005, 0.0, 0.0, 0.995),),
        load_ratios=((0.02, 0.0, 0.0, 3.98),),
    )


def _sized_matrix(layer_count: int, expert_count: int) -> RoutingLoadMatrix:
    counts = tuple((1, *(0 for _ in range(expert_count - 1))) for _ in range(layer_count))
    shares = tuple((1.0, *(0.0 for _ in range(expert_count - 1))) for _ in range(layer_count))
    ratios = tuple(
        (float(expert_count), *(0.0 for _ in range(expert_count - 1)))
        for _ in range(layer_count)
    )
    return RoutingLoadMatrix(
        schema_version="1.0",
        store_schema_version=STORE_SCHEMA_VERSION,
        event_schema_version=EVENT_SCHEMA_VERSION,
        run_key=f"run-{layer_count}x{expert_count}",
        model_key="model:acme/large-moe@r1",
        adapter_name="universal",
        adapter_version="1.0",
        inspection_digest="sha256:" + "4" * 64,
        layout="packed",
        shard_keys=(_shard("7"),),
        token_count=1,
        assignment_count=layer_count,
        routed_top_k=1,
        layer_keys=tuple(_hex_component(10_000 + index) for index in range(layer_count)),
        layer_indices=tuple(range(layer_count)),
        expert_keys=tuple(
            tuple(
                _hex_component(20_000 + layer * expert_count + expert)
                for expert in range(expert_count)
            )
            for layer in range(layer_count)
        ),
        assignment_counts=counts,
        assignment_shares=shares,
        load_ratios=ratios,
    )


def test_public_surface_signature_and_schema() -> None:
    assert ROUTING_HEATMAP_SCHEMA_VERSION == "1.0"
    signature = inspect.signature(render_routing_load_heatmap)
    assert tuple(signature.parameters) == ("matrix", "metric", "max_cells")
    assert signature.parameters["metric"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["max_cells"].kind is inspect.Parameter.KEYWORD_ONLY


def test_compact_renderer_contains_complete_adaptive_matrix_without_scroll() -> None:
    rendered = render_compact_routing_load_heatmap(
        _matrix(), metric="assignment_counts", max_cells=8
    )
    assert rendered.count("<td ") == 8
    assert rendered.count('scope="col"') == 5
    assert rendered.count('scope="row"') == 2
    assert 'aria-label="Complete routing matrix for 2 layers and 4 experts"' in rendered
    assert "table-layout: fixed" in rendered
    assert "overflow: hidden" in rendered
    assert 'tabindex="0"' in rendered
    assert 'data-target="layer:0/expert:0"' in rendered
    assert 'role="checkbox" aria-checked="false"' in rendered
    assert "td.is-selected" in rendered
    assert "Exact values and component identities are available on cell focus or hover." in rendered
    assert "overflow-x: auto" not in rendered
    assert '<main class="readable">' in rendered


@pytest.mark.parametrize(
    "layer_count,expert_count,density",
    [(17, 17, "dense"), (17, 241, "ultra")],
)
def test_compact_renderer_adapts_to_larger_model_dimensions(
    layer_count: int, expert_count: int, density: str
) -> None:
    cell_count = layer_count * expert_count
    rendered = render_compact_routing_load_heatmap(
        _sized_matrix(layer_count, expert_count),
        metric="assignment_counts",
        max_cells=cell_count,
    )
    assert rendered.count("<td ") == cell_count
    assert f'<main class="{density}">' in rendered
    assert f"{layer_count} layers × {expert_count} experts" in rendered


@pytest.mark.parametrize("layout", ["legacy_indexed", "packed"])
@pytest.mark.parametrize(
    "metric, expected",
    [
        ("assignment_counts", ("0", "2", "1")),
        ("assignment_shares", ("0.000000%", "50.000000%", "25.000000%")),
        ("load_ratios", ("0.000000×", "2.000000×", "1.000000×")),
    ],
)
def test_complete_deterministic_html_for_both_layouts(
    layout: str, metric: str, expected: tuple[str, ...]
) -> None:
    first = _render(metric, _matrix(layout))
    second = _render(metric, _matrix(layout))
    assert first == second
    assert first.endswith("\n") and not first.endswith("\n\n")
    assert first.startswith('<!doctype html>\n<html lang="en">')
    assert first.count("<style>") == first.count("</style>") == 1
    assert '<meta charset="utf-8">' in first
    assert '<meta name="viewport"' in first
    assert '<meta name="moeatlas-routing-heatmap-schema" content="1.0">' in first
    assert "Content-Security-Policy" in first
    assert '<meta name="referrer" content="no-referrer">' in first
    assert "Layer × Expert routing-load heatmap" in first
    assert (
        "Routing load only. Selection frequency is association evidence, not expert "
        "specialization or causal effect." in first
    )
    assert "MoEAtlas" in first and "run-heatmap" in first
    assert "EXPERIMENTAL" in first
    assert all(value in first for value in expected)
    assert "model:acme/mixtral@r1" in first
    assert "huggingface-mixtral-static" in first
    assert "sha256:" + "1" * 64 in first
    assert _shard("1") in first and _shard("2") in first
    assert layout in first


def test_table_is_accessible_complete_and_zero_inclusive() -> None:
    rendered = _render()
    assert rendered.count("<table") == 1
    assert rendered.count('scope="col"') == 5
    assert rendered.count('scope="row"') == 2
    assert rendered.count("<tr>") == 3
    assert rendered.count("<td ") == 8
    assert "<caption>Selected assignment count per layer × expert." in rendered
    for key in (*_matrix().layer_keys, *_matrix().expert_keys[0], *_matrix().expert_keys[1]):
        assert key in rendered
    assert 'class="heat-0"' in rendered
    assert 'class="heat-8"' in rendered
    assert 'data-heat="0"' in rendered
    assert 'data-heat="8"' in rendered
    assert 'aria-label="Layer 0' in rendered
    assert 'aria-label="Expert 0' in rendered
    assert 'scope="col"' in rendered and 'scope="row"' in rendered


def test_native_details_and_visible_shard_count() -> None:
    rendered = _render()
    assert rendered.count("<details") == rendered.count("</details>") == 1
    assert "<details open>" in rendered
    assert "<summary>Shard provenance (2 shards)</summary>" in rendered
    assert "Shard count: 2" in rendered
    assert rendered.index("<details") < rendered.index(_shard("1"))


@pytest.mark.parametrize("metric", ["assignment_counts", "assignment_shares", "load_ratios"])
def test_tiny_positive_values_use_heat_one(metric: str) -> None:
    rendered = _render(metric, _tiny_positive_matrix())
    assert 'data-heat="1"' in rendered
    assert 'data-heat="0"' in rendered
    assert 'data-heat="8"' in rendered


def test_numeric_layer_order_preserves_two_before_ten() -> None:
    rendered = _render("assignment_counts", _many_layer_matrix(), max_cells=44)
    table_body = rendered.split("<tbody>", 1)[1]
    assert rendered.count('scope="row"') == 11
    assert table_body.index("Layer 2") < table_body.index("Layer 10")
    assert "run-many-layers" in rendered


class _DocumentShape(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[str] = []
        self.attributes: list[tuple[str, list[tuple[str, str | None]]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append(tag)
        self.attributes.append((tag, attrs))


def test_html_parser_structure_and_security_surface() -> None:
    rendered = _render()
    parser = _DocumentShape()
    parser.feed(rendered)
    assert parser.tags.count("html") == 1
    assert parser.tags.count("head") == 1
    assert parser.tags.count("body") == 1
    assert parser.tags.count("table") == 1
    assert parser.tags.count("details") == 1
    assert parser.tags.count("style") == 1
    assert not any(
        tag in {"script", "form", "img", "iframe", "link", "video", "audio"} for tag in parser.tags
    )
    for _, attrs in parser.attributes:
        for name, value in attrs:
            assert not name.lower().startswith("on")
            assert not (value or "").lower().startswith(("http:", "https:", "javascript:"))
    assert rendered.lower().count("<style") == 1
    assert rendered.lower().count("<script") == 0


def test_metric_definitions_and_global_heat_formula_are_visible() -> None:
    for metric, text in (
        ("assignment_counts", "Selected assignment count per layer × expert."),
        ("assignment_shares", "Assignment share per layer: count ÷ (token count × routed top-k)."),
        (
            "load_ratios",
            "Load ratio per layer: share × expert count; 1.0× is ideal uniform "
            "load—not specialization.",
        ),
    ):
        rendered = _render(metric)
        assert metric in rendered
        assert text in rendered
    rendered = _render("load_ratios")
    assert "Global heat-0..8 bins use 1 + min(7, int((v / m) * 8))" in rendered
    assert "heat-0" in rendered and "heat-8" in rendered


@pytest.mark.parametrize("bad", [None, 1, True, "assignment_counts ", "counts"])
def test_metric_validation_is_first_and_exact(bad: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        render_routing_load_heatmap(_matrix(), metric=bad, max_cells=8)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [None, True, 1.0, "8", 0, -1])
def test_max_cells_validation_follows_metric(bad: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        render_routing_load_heatmap(_matrix(), metric="assignment_counts", max_cells=bad)  # type: ignore[arg-type]


def test_exact_matrix_type_and_fresh_revalidation() -> None:
    with pytest.raises(TypeError):
        render_routing_load_heatmap(object(), metric="assignment_counts", max_cells=8)
    subclass = type("MatrixSubclass", (RoutingLoadMatrix,), {})
    with pytest.raises(TypeError):
        render_routing_load_heatmap(
            subclass(
                **{
                    field: getattr(_matrix(), field)
                    for field in RoutingLoadMatrix.__dataclass_fields__
                }
            ),
            metric="assignment_counts",
            max_cells=8,
        )

    invalid = object.__new__(RoutingLoadMatrix)
    for field in RoutingLoadMatrix.__dataclass_fields__:
        object.__setattr__(invalid, field, getattr(_matrix(), field))
    object.__setattr__(invalid, "assignment_counts", ((0, 2, 0, 1), (1, 1, 1, 1)))
    with pytest.raises((TypeError, ValueError)):
        render_routing_load_heatmap(invalid, metric="assignment_counts", max_cells=8)


def test_cell_budget_is_after_fresh_matrix_validation() -> None:
    with pytest.raises(ValueError, match="matrix cells"):
        render_routing_load_heatmap(_matrix(), metric="assignment_counts", max_cells=7)
    invalid = object.__new__(RoutingLoadMatrix)
    for field in RoutingLoadMatrix.__dataclass_fields__:
        object.__setattr__(invalid, field, getattr(_matrix(), field))
    object.__setattr__(invalid, "assignment_counts", ((0, 0, 0, 0), (0, 0, 0, 0)))
    with pytest.raises((TypeError, ValueError)):
        render_routing_load_heatmap(invalid, metric="assignment_counts", max_cells=1)


def test_quote_aware_escaping_is_present_in_renderer_source() -> None:
    source = Path("src/moeatlas/analysis/routing_heatmap.py").read_text()
    assert "html.escape" in source
    assert "quote=True" in source


def test_fresh_string_and_input_field_identities_are_preserved() -> None:
    matrix = _matrix()
    field_ids = {
        field: id(getattr(matrix, field)) for field in RoutingLoadMatrix.__dataclass_fields__
    }
    original_fields = {
        field: getattr(matrix, field) for field in RoutingLoadMatrix.__dataclass_fields__
    }
    rendered = _render(matrix=matrix)
    second = _render(matrix=matrix)
    assert isinstance(rendered, str)
    assert rendered is not second
    for field, value in original_fields.items():
        assert getattr(matrix, field) == value
        assert id(getattr(matrix, field)) == field_ids[field]


def test_actual_renderer_input_is_not_retained_while_html_is_alive() -> None:
    matrix = _matrix()
    before = sys.getrefcount(matrix)
    rendered = _render(matrix=matrix)
    gc.collect()
    after = sys.getrefcount(matrix)
    assert after == before
    assert rendered.endswith("\n")


def test_equivalent_redacted_and_nonredacted_matrices_have_equal_table_values() -> None:
    redacted = _matrix()
    nonredacted = replace(redacted, shard_keys=(_shard("5"), _shard("6")))
    redacted_table = _render(matrix=redacted).split("<table", 1)[1]
    nonredacted_table = _render(matrix=nonredacted).split("<table", 1)[1]
    assert redacted_table == nonredacted_table


def test_hostile_dynamic_strings_are_quote_escaped(monkeypatch: pytest.MonkeyPatch) -> None:
    hostile = object.__new__(RoutingLoadMatrix)
    base = _matrix()
    for field in RoutingLoadMatrix.__dataclass_fields__:
        object.__setattr__(hostile, field, getattr(base, field))
    object.__setattr__(hostile, "run_key", "<run>&\"'")
    object.__setattr__(hostile, "layer_keys", ("<layer>&\"'", base.layer_keys[1]))
    object.__setattr__(
        hostile, "expert_keys", (("<expert>&\"'", *base.expert_keys[0][1:]), base.expert_keys[1])
    )
    monkeypatch.setattr(heatmap, "_fresh_matrix", lambda matrix: hostile)
    rendered = _render(matrix=base)
    assert "<script>" not in rendered
    assert "&lt;run&gt;&amp;&quot;&#x27;" in rendered
    assert "&lt;layer&gt;&amp;&quot;&#x27;" in rendered
    assert "&lt;expert&gt;&amp;&quot;&#x27;" in rendered


def test_offline_network_cache_and_temp_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    before_cache = tuple(cache.rglob("*"))
    before_temp = tuple(temp.rglob("*"))
    _render()
    assert tuple(cache.rglob("*")) == before_cache
    assert tuple(temp.rglob("*")) == before_temp


def test_ast_has_no_runtime_or_external_surface() -> None:
    source = Path("src/moeatlas/analysis/routing_heatmap.py").read_text()
    tree = ast.parse(source)
    forbidden_imports = {
        "torch",
        "transformers",
        "accelerate",
        "safetensors",
        "duckdb",
        "numpy",
        "pandas",
        "pyarrow",
        "polars",
        "os",
        "shutil",
        "tempfile",
        "pathlib",
        "urllib",
        "importlib",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] not in forbidden_imports for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in forbidden_imports
    lowered = source.lower()
    assert "<script" not in lowered
    assert "javascript:" not in lowered
    assert "url(" not in lowered
    assert "read_parquet" not in lowered
    assert "duckdb" not in lowered
    assert "write_text" not in lowered
    assert "write_bytes" not in lowered
    assert "mkdir" not in lowered
    assert "unlink" not in lowered
    assert "rename" not in lowered
    assert "replace" not in lowered
    assert "path.open" not in lowered
    assert "socket" not in lowered
    assert "urlopen" not in lowered
    assert "import_module" not in lowered
    assert "os." not in lowered
    assert "shutil." not in lowered
    assert "tempfile." not in lowered
    assert "pathlib" not in lowered
    assert "urllib" not in lowered
    assert "read_text" not in lowered
    assert "read_bytes" not in lowered
    assert "iterdir" not in lowered
    assert "rglob" not in lowered
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                assert node.func.id != "open"
            if isinstance(node.func, ast.Attribute):
                assert node.func.attr not in {
                    "write",
                    "write_text",
                    "write_bytes",
                    "mkdir",
                    "unlink",
                    "rename",
                    "replace",
                    "touch",
                    "open",
                    "urlopen",
                    "read_text",
                    "read_bytes",
                    "iterdir",
                    "rglob",
                    "glob",
                    "stat",
                }
                assert not node.func.attr.lower().startswith("on")


def test_neutral_renderer_public_surface_has_no_legacy_mixtral_aliases() -> None:
    assert not hasattr(heatmap, "MixtralRoutingLoadMatrix")
    assert not hasattr(heatmap, "render_mixtral_routing_load_heatmap")
    matrix = _matrix()
    assert "<!doctype html>" in render_routing_load_heatmap(
        matrix, metric="load_ratios", max_cells=100
    ).lower()
