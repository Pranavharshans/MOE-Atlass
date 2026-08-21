from __future__ import annotations

import ast
import inspect
from dataclasses import replace
from html.parser import HTMLParser
from pathlib import Path

import pytest

from moeatlas.analysis import (
    ROUTING_COMPARE_HEATMAP_SCHEMA_VERSION,
    RoutingLoadComparison,
    compare_routing_load,
    render_routing_load_comparison,
)
from moeatlas.events import EVENT_SCHEMA_VERSION
from moeatlas.store import STORE_SCHEMA_VERSION

from .test_analysis_routing_compare import _comparison_matrix, _matrix, _shard

_DIGEST = "sha256:" + "1" * 64

_METRIC_DEFINITIONS = {
    "count_deltas": (
        "Selected assignment count difference per layer × expert "
        "(comparison run − baseline run)."
    ),
    "share_deltas": (
        "Assignment share difference per layer: count ÷ (token count × routed "
        "top-k), comparison run − baseline run."
    ),
    "ratio_deltas": (
        "Load ratio difference per layer: share × expert count, comparison run "
        "− baseline run; 0.00× means identical load—not specialization."
    ),
}

_METRIC_VISIBLES = {
    "count_deltas": (">+1<", ">-1<", ">+0<"),
    "share_deltas": (">+25.000000%<", ">-25.000000%<", ">+0.000000%<"),
    "ratio_deltas": (">+1.000000×<", ">-1.000000×<", ">+0.000000×<"),
}

_METRIC_MAXIMUM_LINES = {
    "count_deltas": "Maximum absolute value: 1 count.",
    "share_deltas": "Maximum absolute value: 0.25 share.",
    "ratio_deltas": "Maximum absolute value: 1.0 ratio.",
}

_ZERO_VISIBLES = {
    "count_deltas": ">+0<",
    "share_deltas": ">+0.000000%<",
    "ratio_deltas": ">+0.000000×<",
}

_ZERO_MAXIMUM_LINES = {
    "count_deltas": "Maximum absolute value: 0 count.",
    "share_deltas": "Maximum absolute value: 0.0 share.",
    "ratio_deltas": "Maximum absolute value: 0.0 ratio.",
}

_BIN_BASE_COUNTS = ((0, 2, 0, 2), (1, 1, 1, 1))
_BIN_BASE_SHARES = ((0.0, 0.5, 0.0, 0.5), (0.25, 0.25, 0.25, 0.25))
_BIN_BASE_RATIOS = ((0.0, 2.0, 0.0, 2.0), (1.0, 1.0, 1.0, 1.0))
_BIN_OTHER_COUNTS = ((2, 0, 1, 1), (1, 1, 1, 1))
_BIN_OTHER_SHARES = ((0.5, 0.0, 0.25, 0.25), (0.25, 0.25, 0.25, 0.25))
_BIN_OTHER_RATIOS = ((2.0, 0.0, 1.0, 1.0), (1.0, 1.0, 1.0, 1.0))

_BIN_CLASSES = (
    ("heat-8", "cold-8", "heat-5", "cold-5"),
    ("delta-zero", "delta-zero", "delta-zero", "delta-zero"),
)
_BIN_LEVELS = (("8", "-8", "5", "-5"), ("0", "0", "0", "0"))
_BIN_VISIBLES = {
    "count_deltas": ("+2", "-2", "+1", "-1", "+0"),
    "share_deltas": ("+50.000000%", "-50.000000%", "+25.000000%", "-25.000000%", "+0.000000%"),
    "ratio_deltas": ("+2.000000×", "-2.000000×", "+1.000000×", "-1.000000×", "+0.000000×"),
}
_BIN_MAXIMUM_LINES = {
    "count_deltas": "Maximum absolute value: 2 count.",
    "share_deltas": "Maximum absolute value: 0.5 share.",
    "ratio_deltas": "Maximum absolute value: 2.0 ratio.",
}

_ALLOWED_CELL_CLASSES = frozenset(
    {
        "delta-zero",
        *(f"heat-{level}" for level in range(1, 9)),
        *(f"cold-{level}" for level in range(1, 9)),
    }
)

_PROVENANCE_ENTRIES = (
    "<dt>Compare-heatmap schema</dt><dd>1.0</dd>",
    "<dt>Routing-compare schema</dt><dd>1.0</dd>",
    f"<dt>Store schema</dt><dd>{STORE_SCHEMA_VERSION}</dd>",
    f"<dt>Event schema</dt><dd>{EVENT_SCHEMA_VERSION}</dd>",
    "<dt>Baseline run key</dt><dd>run-baseline</dd>",
    "<dt>Comparison run key</dt><dd>run-comparison</dd>",
    "<dt>Model key</dt><dd>model:acme/mixtral@r1</dd>",
    "<dt>Adapter</dt><dd>huggingface-mixtral-static / 1.0</dd>",
    f"<dt>Inspection digest</dt><dd>{_DIGEST}</dd>",
    "<dt>Layout</dt><dd>legacy_indexed</dd>",
    "<dt>Token count per run</dt><dd>2</dd>",
    "<dt>Routed top-k</dt><dd>2</dd>",
    "<dt>Layer count</dt><dd>2</dd>",
    "<dt>Expert count per layer</dt><dd>4</dd>",
    "<dt>Baseline assignment count</dt><dd>8</dd>",
    "<dt>Comparison assignment count</dt><dd>8</dd>",
)


def _compare() -> RoutingLoadComparison:
    return compare_routing_load(_matrix(), _comparison_matrix(), max_cells=8)


def _zero_comparison() -> RoutingLoadComparison:
    return compare_routing_load(_matrix(), _matrix(run_key="run-comparison"), max_cells=8)


def _bin_comparison() -> RoutingLoadComparison:
    return compare_routing_load(
        _matrix(
            "run-bin-baseline",
            counts=_BIN_BASE_COUNTS,
            shares=_BIN_BASE_SHARES,
            ratios=_BIN_BASE_RATIOS,
        ),
        _matrix(
            "run-bin-comparison",
            counts=_BIN_OTHER_COUNTS,
            shares=_BIN_OTHER_SHARES,
            ratios=_BIN_OTHER_RATIOS,
            shard_digits="37",
        ),
        max_cells=8,
    )


def _render(comparison: RoutingLoadComparison | None = None, metric: str = "count_deltas") -> str:
    return render_routing_load_comparison(
        _compare() if comparison is None else comparison,
        metric=metric,
        max_cells=8,
    )


def _forged(field_name: str, bad_value: object) -> RoutingLoadComparison:
    value = _compare()
    forged = object.__new__(RoutingLoadComparison)
    for field in RoutingLoadComparison.__dataclass_fields__:
        object.__setattr__(forged, field, getattr(value, field))
    object.__setattr__(forged, field_name, bad_value)
    return forged


class _MatrixShape(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[str] = []
        self.attributes: list[tuple[str, list[tuple[str, str | None]]]] = []
        self.table_count = 0
        self.caption_count = 0
        self.expert_headers: list[str | None] = []
        self.body_rows: list[list[tuple[str | None, str | None]]] = []
        self._in_body = False
        self._row: list[tuple[str | None, str | None]] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append(tag)
        self.attributes.append((tag, attrs))
        values = dict(attrs)
        if tag == "table":
            self.table_count += 1
        elif tag == "caption":
            self.caption_count += 1
        elif tag == "tbody":
            self._in_body = True
        elif tag == "thead":
            self._in_body = False
        elif tag == "tr" and self._in_body:
            self._row = []
        elif tag == "td" and self._row is not None:
            self._row.append((values.get("class"), values.get("data-delta")))
        elif tag == "th" and values.get("scope") == "col":
            self.expert_headers.append(values.get("aria-label"))

    def handle_endtag(self, tag: str) -> None:
        if tag == "tr" and self._row is not None:
            self.body_rows.append(self._row)
            self._row = None


def test_public_surface_signature_and_schema() -> None:
    assert ROUTING_COMPARE_HEATMAP_SCHEMA_VERSION == "1.0"
    signature = inspect.signature(render_routing_load_comparison)
    assert tuple(signature.parameters) == ("comparison", "metric", "max_cells")
    assert signature.parameters["comparison"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert signature.parameters["metric"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["max_cells"].kind is inspect.Parameter.KEYWORD_ONLY
    assert RoutingLoadComparison.__dataclass_params__.frozen is True


def test_export_is_reachable_from_the_package() -> None:
    import moeatlas.analysis as package

    assert package.render_routing_load_comparison is render_routing_load_comparison
    assert package.ROUTING_COMPARE_HEATMAP_SCHEMA_VERSION == "1.0"


@pytest.mark.parametrize("metric", ["count_deltas", "share_deltas", "ratio_deltas"])
def test_complete_document_per_metric(metric: str) -> None:
    definition = _METRIC_DEFINITIONS[metric]
    rendered = _render(metric=metric)
    assert rendered.startswith('<!doctype html>\n<html lang="en">')
    assert rendered.endswith("\n") and not rendered.endswith("\n\n")
    assert (
        "<title>MoEAtlas — Layer × Expert routing-load comparison — "
        "run-baseline vs run-comparison</title>"
    ) in rendered
    assert "<h1>Layer × Expert routing-load comparison</h1>" in rendered
    assert (
        "Routing-load deltas only. Differences in selection frequency are association "
        "evidence, not expert specialization or causal effect." in rendered
    )
    assert f"<strong>Metric:</strong> {metric} — {definition}" in rendered
    assert f"{definition} Visible values use {metric}.</caption>" in rendered
    assert '<meta name="moeatlas-routing-compare-heatmap-schema" content="1.0">' in rendered
    assert _METRIC_MAXIMUM_LINES[metric] in rendered
    assert all(visible in rendered for visible in _METRIC_VISIBLES[metric])
    assert "EXPERIMENTAL: this static view preserves the supplied inspection axes" in rendered
    assert _DIGEST in rendered
    assert "model:acme/mixtral@r1" in rendered
    assert "huggingface-mixtral-static / 1.0" in rendered
    assert "legacy_indexed" in rendered


def test_provenance_entries_are_complete() -> None:
    rendered = _render()
    assert rendered.count('<dl class="provenance">') == 1
    for entry in _PROVENANCE_ENTRIES:
        assert entry in rendered


def test_shard_details_list_both_run_universes() -> None:
    rendered = _render()
    assert rendered.count("<details open>") == 2
    assert "<summary>Baseline shard provenance (2 shards)</summary>" in rendered
    assert "<summary>Comparison shard provenance (2 shards)</summary>" in rendered
    assert rendered.count("<p>Shard count: 2</p>") == 2
    for key in (_shard("1"), _shard("9"), _shard("2"), _shard("3")):
        assert f"<li>{key}</li>" in rendered
    assert rendered.index("<details") < rendered.index(_shard("1"))
    assert rendered.index("Baseline shard provenance") < rendered.index(
        "Comparison shard provenance"
    )


def test_default_fixture_cells_use_extreme_or_neutral_bins() -> None:
    for metric in ("count_deltas", "share_deltas", "ratio_deltas"):
        rendered = _render(metric=metric)
        assert rendered.count('<td class="heat-8"') == 2
        assert rendered.count('<td class="cold-8"') == 2
        assert rendered.count('<td class="delta-zero"') == 4
        assert rendered.count('data-delta="8"') == 2
        assert rendered.count('data-delta="-8"') == 2
        assert rendered.count('data-delta="0"') == 4


def test_static_security_surface_is_closed() -> None:
    rendered = _render()
    lowered = rendered.lower()
    assert "Content-Security-Policy" in rendered
    assert "default-src 'none'" in rendered and "script-src 'none'" in rendered
    assert '<meta name="referrer" content="no-referrer">' in rendered
    assert lowered.count("<script") == 0
    assert "javascript:" not in lowered
    assert "http://" not in lowered and "https://" not in lowered
    assert "<iframe" not in lowered and "<img" not in lowered
    assert "<link" not in lowered and "<form" not in lowered


def test_rendering_twice_is_byte_identical() -> None:
    assert _render() == _render()


def test_equal_comparisons_render_identically() -> None:
    rebuilt = compare_routing_load(_matrix(), _comparison_matrix(), max_cells=8)
    assert _render(_compare()) == _render(rebuilt)


def test_exact_signed_bins_for_known_magnitudes() -> None:
    value = _bin_comparison()
    assert value.count_deltas == ((2, -2, 1, -1), (0, 0, 0, 0))
    for metric in ("count_deltas", "share_deltas", "ratio_deltas"):
        rendered = _render(value, metric=metric)
        assert _BIN_MAXIMUM_LINES[metric] in rendered
        for visible in _BIN_VISIBLES[metric]:
            assert f">{visible}<" in rendered
        parser = _MatrixShape()
        parser.feed(rendered)
        classes = tuple(tuple(cls or "" for cls, _ in row) for row in parser.body_rows)
        levels = tuple(tuple(level or "" for _, level in row) for row in parser.body_rows)
        assert classes == _BIN_CLASSES
        assert levels == _BIN_LEVELS


def test_identical_runs_render_every_cell_neutral() -> None:
    value = _zero_comparison()
    for metric in ("count_deltas", "share_deltas", "ratio_deltas"):
        rendered = _render(value, metric=metric)
        assert _ZERO_MAXIMUM_LINES[metric] in rendered
        assert _ZERO_VISIBLES[metric] in rendered
        assert rendered.count('<td class="delta-zero"') == 8
        assert rendered.count('data-delta="0"') == 8
        assert rendered.count("<td") == 8


def test_validation_order_is_metric_then_max_cells_then_type_then_budget() -> None:
    with pytest.raises(TypeError):
        render_routing_load_comparison(None, metric=42, max_cells=0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        render_routing_load_comparison(None, metric="unknown", max_cells=0)  # type: ignore[arg-type]
    with pytest.raises((TypeError, ValueError)):
        render_routing_load_comparison(None, metric="count_deltas", max_cells=0)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        render_routing_load_comparison(None, metric="count_deltas", max_cells=8)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [None, 42, b"count_deltas"])
def test_non_string_metric_is_rejected(bad: object) -> None:
    with pytest.raises(TypeError):
        render_routing_load_comparison(_compare(), metric=bad, max_cells=8)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", ["count_delta", "", "COUNT_DELTAS"])
def test_unknown_metric_string_is_rejected(bad: str) -> None:
    with pytest.raises(ValueError):
        render_routing_load_comparison(_compare(), metric=bad, max_cells=8)


@pytest.mark.parametrize("bad", [True, False, 0, -1, 1.5, "8", None])
def test_invalid_max_cells_is_rejected(bad: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        render_routing_load_comparison(_compare(), metric="count_deltas", max_cells=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [_matrix(), None, "run-baseline"])
def test_non_comparison_inputs_are_rejected(bad: object) -> None:
    with pytest.raises(TypeError):
        render_routing_load_comparison(bad, metric="count_deltas", max_cells=8)  # type: ignore[arg-type]


def test_exact_type_check_rejects_subclasses() -> None:
    subclass = type("ComparisonSubclass", (RoutingLoadComparison,), {})
    value = _compare()
    forged = subclass(
        **{field: getattr(value, field) for field in RoutingLoadComparison.__dataclass_fields__}
    )
    with pytest.raises(TypeError):
        render_routing_load_comparison(forged, metric="count_deltas", max_cells=8)


def test_cell_budget_is_enforced() -> None:
    with pytest.raises(ValueError, match="exceed max_cells"):
        render_routing_load_comparison(_compare(), metric="count_deltas", max_cells=7)
    render_routing_load_comparison(_compare(), metric="count_deltas", max_cells=8)


@pytest.mark.parametrize(
    "field_name, bad_value",
    [
        ("schema_version", "2.0"),
        ("comparison_run_key", "run-baseline"),
        ("comparison_shard_keys", (_shard("3"), _shard("2"))),
        ("share_deltas", ((float("nan"), -0.25, 0.0, 0.0), (0.25, -0.25, 0.0, 0.0))),
        ("count_deltas", ((1, -1, 0, 1), (1, -1, 0, 0))),
        ("layer_indices", (1, 0)),
    ],
)
def test_tampered_fields_fail_fresh_validation_at_render_time(
    field_name: str, bad_value: object
) -> None:
    forged = _forged(field_name, bad_value)
    with pytest.raises((TypeError, ValueError)):
        render_routing_load_comparison(forged, metric="count_deltas", max_cells=8)


def test_replace_based_tampering_is_rejected_eagerly() -> None:
    with pytest.raises(ValueError):
        replace(_compare(), schema_version="2.0")


def test_fresh_validation_precedes_cell_budget() -> None:
    forged = _forged("count_deltas", ((1, -1, 0, 1), (1, -1, 0, 0)))
    with pytest.raises(ValueError) as excinfo:
        render_routing_load_comparison(forged, metric="count_deltas", max_cells=1)
    assert "exceed max_cells" not in str(excinfo.value)


def test_html_parser_structure_and_allowed_cell_classes() -> None:
    rendered = _render()
    parser = _MatrixShape()
    parser.feed(rendered)
    assert parser.table_count == 1
    assert parser.caption_count == 1
    assert '<th scope="col">Layer</th>' in rendered
    labeled_headers = [label for label in parser.expert_headers if label is not None]
    assert len(parser.expert_headers) == 5
    assert len(labeled_headers) == 4
    assert all(label.startswith("Expert ") for label in labeled_headers)
    assert len(parser.body_rows) == 2
    for row in parser.body_rows:
        assert len(row) == 4
        for cell_class, level in row:
            assert cell_class in _ALLOWED_CELL_CLASSES
            assert level is not None
    assert not any(
        tag in {"script", "img", "iframe", "form", "link", "video", "audio"} for tag in parser.tags
    )
    for _, attrs in parser.attributes:
        for name, value in attrs:
            assert not name.lower().startswith("on")
            assert not (value or "").lower().startswith(("http:", "https:", "javascript:"))
    for key in (*_matrix().layer_keys, *_matrix().expert_keys[0], *_matrix().expert_keys[1]):
        assert key in rendered
    assert 'class="axis-key"' in rendered


def test_legend_covers_full_signed_scale() -> None:
    rendered = _render()
    legend = rendered.split('<div class="legend"', 1)[1].split("</div>", 1)[0]
    for level in range(1, 9):
        assert f'<span class="cold-{level}">cold-{level}</span>' in legend
        assert f'<span class="heat-{level}">heat-{level}</span>' in legend
    assert '<span class="delta-zero">delta-zero</span>' in legend
    assert legend.index("cold-8") < legend.index("delta-zero") < legend.index("heat-1")
    assert (
        "Signed delta-0..8 bins use 1 + min(7, int((|v| / m) * 8)) for nonzero values"
        in rendered
    )
    assert (
        "Zero deltas are neutral; warm colors mark experts the comparison run routes away "
        "from; cool-to-green colors mark experts it routes toward." in rendered
    )


def test_hostile_run_keys_are_quote_escaped() -> None:
    hostile_key = 'run<a>&"\''
    value = compare_routing_load(_matrix(run_key=hostile_key), _comparison_matrix(), max_cells=8)
    rendered = _render(value)
    assert "&lt;a&gt;&amp;&quot;&#x27;" in rendered
    assert hostile_key not in rendered
    assert "<script>" not in rendered


def test_module_source_has_no_runtime_or_external_surface() -> None:
    source = Path("src/moeatlas/analysis/compare_heatmap.py").read_text()
    tree = ast.parse(source)
    forbidden_imports = {
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
            assert all(alias.name.split(".")[0] not in forbidden_imports for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in forbidden_imports
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                assert node.func.id != "open"
            if isinstance(node.func, ast.Attribute):
                assert node.func.attr not in {
                    "write_text",
                    "write_bytes",
                    "mkdir",
                    "unlink",
                    "rename",
                }
    assert "token_text" not in source
    assert "connect(" not in source
