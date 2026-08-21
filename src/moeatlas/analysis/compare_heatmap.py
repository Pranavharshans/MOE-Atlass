"""Deterministic, dependency-free HTML for one routing-load comparison."""

from __future__ import annotations

import html

from .routing_compare import RoutingLoadComparison

ROUTING_COMPARE_HEATMAP_SCHEMA_VERSION = "1.0"

_METRICS = {
    "count_deltas": (
        "Selected assignment count difference per layer × expert "
        "(comparison run − baseline run).",
        "count",
    ),
    "share_deltas": (
        "Assignment share difference per layer: count ÷ (token count × routed "
        "top-k), comparison run − baseline run.",
        "share",
    ),
    "ratio_deltas": (
        "Load ratio difference per layer: share × expert count, comparison run "
        "− baseline run; 0.00× means identical load—not specialization.",
        "ratio",
    ),
}

_CSP = (
    "default-src 'none'; script-src 'none'; object-src 'none'; img-src 'none'; "
    "font-src 'none'; connect-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; "
    "form-action 'none'; frame-ancestors 'none'"
)


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _strict_positive_cells(value: object) -> int:
    if type(value) is not int or isinstance(value, bool):
        raise TypeError("max_cells must be an exact positive integer")
    if value <= 0:
        raise ValueError("max_cells must be positive")
    return value


def _fresh_comparison(comparison: RoutingLoadComparison) -> RoutingLoadComparison:
    return RoutingLoadComparison(
        schema_version=comparison.schema_version,
        store_schema_version=comparison.store_schema_version,
        event_schema_version=comparison.event_schema_version,
        baseline_run_key=comparison.baseline_run_key,
        comparison_run_key=comparison.comparison_run_key,
        model_key=comparison.model_key,
        adapter_name=comparison.adapter_name,
        adapter_version=comparison.adapter_version,
        inspection_digest=comparison.inspection_digest,
        layout=comparison.layout,
        token_count=comparison.token_count,
        routed_top_k=comparison.routed_top_k,
        baseline_shard_keys=comparison.baseline_shard_keys,
        comparison_shard_keys=comparison.comparison_shard_keys,
        baseline_assignment_count=comparison.baseline_assignment_count,
        comparison_assignment_count=comparison.comparison_assignment_count,
        layer_keys=comparison.layer_keys,
        layer_indices=comparison.layer_indices,
        expert_keys=comparison.expert_keys,
        count_deltas=comparison.count_deltas,
        share_deltas=comparison.share_deltas,
        ratio_deltas=comparison.ratio_deltas,
    )


def _signed_bin(value: float, maximum_absolute: float) -> tuple[str, int]:
    if maximum_absolute <= 0.0 or value == 0.0:
        return "delta-zero", 0
    if value > 0.0:
        level = 1 + min(7, int((value / maximum_absolute) * 8))
        return f"heat-{level}", level
    level = 1 + min(7, int((-value / maximum_absolute) * 8))
    return f"cold-{level}", -level


def _format_delta(metric: str, value: int | float) -> str:
    if metric == "count_deltas":
        return f"{value:+d}"
    if metric == "share_deltas":
        return f"{value:+.6%}"
    return f"{value:+.6f}×"


def render_routing_load_comparison(
    comparison: RoutingLoadComparison,
    *,
    metric: str,
    max_cells: int,
) -> str:
    """Render one complete routing-load comparison as standalone HTML5."""

    if type(metric) is not str:
        raise TypeError("metric must be an exact string")
    if metric not in _METRICS:
        raise ValueError("metric must be count_deltas, share_deltas, or ratio_deltas")
    _strict_positive_cells(max_cells)
    if type(comparison) is not RoutingLoadComparison:
        raise TypeError("comparison must be an exact RoutingLoadComparison")
    fresh = _fresh_comparison(comparison)
    expert_count = len(fresh.expert_keys[0])
    cells = len(fresh.layer_keys) * expert_count
    if cells > max_cells:
        raise ValueError("matrix cells exceed max_cells")

    metric_definition, value_kind = _METRICS[metric]
    deltas = getattr(fresh, metric)
    maximum_absolute = max(abs(value) for row in deltas for value in row)
    title = "Layer × Expert routing-load comparison"
    document_title = (
        f"MoEAtlas — {title} — {fresh.baseline_run_key} vs {fresh.comparison_run_key}"
    )

    lines = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '  <meta charset="utf-8">',
        '  <meta name="viewport" content="width=device-width, initial-scale=1">',
        '  <meta name="moeatlas-routing-compare-heatmap-schema" content="1.0">',
        f'  <meta http-equiv="Content-Security-Policy" content="{_CSP}">',
        '  <meta name="referrer" content="no-referrer">',
        f"  <title>{_escape(document_title)}</title>",
        "  <style>",
        '    :root { color-scheme: light; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }',  # noqa: E501
        "    * { box-sizing: border-box; }",
        "    body { margin: 0; min-width: 15rem; padding: clamp(1rem, 4vw, 3rem); color: #17212b; background: #f6f8fa; font-size: clamp(15px, 1vw + 0.25rem, 25px); line-height: 1.38; }",  # noqa: E501
        "    main { width: min(100%, 96rem); margin: 0 auto; }",
        "    h1, h2 { line-height: 1.2; letter-spacing: -0.02em; }",
        "    h1 { margin: 0 0 0.35em; font-size: clamp(1.6rem, 3vw, 2.45rem); }",
        "    h2 { margin: 1.5em 0 0.45em; font-size: 1.15em; }",
        "    p { max-width: 70ch; }",
        "    .warning { padding: 0.8em 1em; border-left: 0.3em solid #a15c00; background: #fff3dc; }",  # noqa: E501
        "    .provenance { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 18rem), 1fr)); gap: 0.3em 1.5em; margin: 1em 0; }",  # noqa: E501
        "    .provenance div { padding: 0.35em 0; border-bottom: 1px solid #d8dee4; }",
        "    dt { font-weight: 650; color: #43515d; }",
        "    dd { margin: 0.1em 0 0; overflow-wrap: anywhere; }",
        "    .table-wrap { overflow-x: auto; border: 1px solid #c9d1d9; background: #fff; }",
        "    table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums lining-nums; }",  # noqa: E501
        "    caption { padding: 0.8em; text-align: left; font-weight: 650; }",
        "    th, td { min-width: 8rem; padding: 0.55em 0.65em; border-right: 1px solid #e1e7ec; border-bottom: 1px solid #e1e7ec; text-align: right; vertical-align: middle; }",  # noqa: E501
        "    thead th { background: #eef2f5; color: #263541; font-weight: 650; }",
        "    tbody th { background: #f7f9fa; text-align: left; font-weight: 600; }",
        "    td:last-child, th:last-child { border-right: 0; }",
        "    tr:last-child th, tr:last-child td { border-bottom: 0; }",
        "    .axis-key { display: block; overflow-wrap: anywhere; font-size: 0.75em; font-weight: 500; color: #536471; }",  # noqa: E501
        "    .delta-zero { background: #ffffff; }",
        "    .heat-1 { background: hsl(210 55% 94%); } .heat-2 { background: hsl(205 65% 89%); } .heat-3 { background: hsl(200 70% 82%); } .heat-4 { background: hsl(190 72% 72%); } .heat-5 { background: hsl(175 68% 61%); } .heat-6 { background: hsl(150 62% 50%); } .heat-7 { background: hsl(104 58% 48%); } .heat-8 { background: hsl(70 70% 45%); color: #13210d; }",  # noqa: E501
        "    .cold-1 { background: hsl(50 70% 94%); } .cold-2 { background: hsl(45 75% 89%); } .cold-3 { background: hsl(40 80% 83%); } .cold-4 { background: hsl(35 85% 76%); } .cold-5 { background: hsl(28 88% 68%); } .cold-6 { background: hsl(20 85% 60%); } .cold-7 { background: hsl(12 78% 52%); color: #fff; } .cold-8 { background: hsl(4 72% 44%); color: #fff; }",  # noqa: E501
        "    .legend { display: flex; flex-wrap: wrap; gap: 0.35em 0.8em; font-size: 0.82em; }",
        "    .legend span { padding: 0.2em 0.45em; border: 1px solid #c9d1d9; }",
        "    footer { margin-top: 1.5em; color: #536471; font-size: 0.82em; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <main>",
        f"    <h1>{_escape(title)}</h1>",
        '    <p class="warning" role="note">Routing-load deltas only. Differences in selection frequency are association evidence, not expert specialization or causal effect.</p>',  # noqa: E501
        f"    <p><strong>Metric:</strong> {_escape(metric)} — {_escape(metric_definition)}</p>",
        '    <section aria-labelledby="provenance-heading">',
        '      <h2 id="provenance-heading">Frozen provenance and counts</h2>',
        '      <dl class="provenance">',
        f"        <div><dt>Compare-heatmap schema</dt><dd>{_escape(ROUTING_COMPARE_HEATMAP_SCHEMA_VERSION)}</dd></div>",  # noqa: E501
        f"        <div><dt>Routing-compare schema</dt><dd>{_escape(fresh.schema_version)}</dd></div>",  # noqa: E501
        f"        <div><dt>Store schema</dt><dd>{_escape(fresh.store_schema_version)}</dd></div>",
        f"        <div><dt>Event schema</dt><dd>{_escape(fresh.event_schema_version)}</dd></div>",
        f"        <div><dt>Baseline run key</dt><dd>{_escape(fresh.baseline_run_key)}</dd></div>",
        f"        <div><dt>Comparison run key</dt><dd>{_escape(fresh.comparison_run_key)}</dd></div>",  # noqa: E501
        f"        <div><dt>Model key</dt><dd>{_escape(fresh.model_key)}</dd></div>",
        f"        <div><dt>Adapter</dt><dd>{_escape(fresh.adapter_name)} / {_escape(fresh.adapter_version)}</dd></div>",  # noqa: E501
        f"        <div><dt>Inspection digest</dt><dd>{_escape(fresh.inspection_digest)}</dd></div>",
        f"        <div><dt>Layout</dt><dd>{_escape(fresh.layout)}</dd></div>",
        f"        <div><dt>Token count per run</dt><dd>{_escape(fresh.token_count)}</dd></div>",
        f"        <div><dt>Routed top-k</dt><dd>{_escape(fresh.routed_top_k)}</dd></div>",
        f"        <div><dt>Layer count</dt><dd>{_escape(len(fresh.layer_keys))}</dd></div>",
        f"        <div><dt>Expert count per layer</dt><dd>{_escape(expert_count)}</dd></div>",
        f"        <div><dt>Baseline assignment count</dt><dd>{_escape(fresh.baseline_assignment_count)}</dd></div>",  # noqa: E501
        f"        <div><dt>Comparison assignment count</dt><dd>{_escape(fresh.comparison_assignment_count)}</dd></div>",  # noqa: E501
        "      </dl>",
        "      <details open>",
        f"        <summary>Baseline shard provenance ({_escape(len(fresh.baseline_shard_keys))} shards)</summary>",  # noqa: E501
        f"        <p>Shard count: {_escape(len(fresh.baseline_shard_keys))}</p>",
        "        <ul>",
        *[f"          <li>{_escape(shard_key)}</li>" for shard_key in fresh.baseline_shard_keys],
        "        </ul>",
        "      </details>",
        "      <details open>",
        f"        <summary>Comparison shard provenance ({_escape(len(fresh.comparison_shard_keys))} shards)</summary>",  # noqa: E501
        f"        <p>Shard count: {_escape(len(fresh.comparison_shard_keys))}</p>",
        "        <ul>",
        *[
            f"          <li>{_escape(shard_key)}</li>"
            for shard_key in fresh.comparison_shard_keys
        ],
        "        </ul>",
        "      </details>",
        "    </section>",
        '    <section aria-labelledby="legend-heading">',
        '      <h2 id="legend-heading">Delta scale</h2>',
        f'      <p id="delta-formula">Signed delta-0..8 bins use 1 + min(7, int((|v| / m) * 8)) for nonzero values, where m is the maximum absolute cell in this artifact; zero values use delta-zero. Positive deltas use heat bins, negative deltas use cold bins. Maximum absolute value: {_escape(maximum_absolute)} {_escape(value_kind)}.</p>',  # noqa: E501
        '      <p id="delta-legend-text">Zero deltas are neutral; warm colors mark experts the comparison run routes away from; cool-to-green colors mark experts it routes toward. Intensity is relative to the largest absolute cell in this artifact.</p>',  # noqa: E501
        '      <div class="legend" aria-label="Delta scale from negative through zero to positive">',  # noqa: E501
        '        <span class="cold-8">cold-8</span><span class="cold-7">cold-7</span><span class="cold-6">cold-6</span><span class="cold-5">cold-5</span><span class="cold-4">cold-4</span><span class="cold-3">cold-3</span><span class="cold-2">cold-2</span><span class="cold-1">cold-1</span><span class="delta-zero">delta-zero</span><span class="heat-1">heat-1</span><span class="heat-2">heat-2</span><span class="heat-3">heat-3</span><span class="heat-4">heat-4</span><span class="heat-5">heat-5</span><span class="heat-6">heat-6</span><span class="heat-7">heat-7</span><span class="heat-8">heat-8</span>',  # noqa: E501
        "      </div>",
        "    </section>",
        '    <section aria-labelledby="matrix-heading">',
        '      <h2 id="matrix-heading">Layer × expert delta matrix</h2>',
        '      <div class="table-wrap">',
        '        <table aria-describedby="delta-formula">',
        f"          <caption>{_escape(metric_definition)} Visible values use {_escape(metric)}.</caption>",  # noqa: E501
        "          <thead>",
        "            <tr>",
        '              <th scope="col">Layer</th>',
    ]

    for expert_position, expert_key in enumerate(fresh.expert_keys[0]):
        escaped_key = _escape(expert_key)
        lines.append(
            f'              <th scope="col" title="{escaped_key}" aria-label="Expert {expert_position} {escaped_key}">Expert {expert_position}<span class="axis-key">{escaped_key}</span></th>'  # noqa: E501
        )
    lines.extend(
        (
            "            </tr>",
            "          </thead>",
            "          <tbody>",
        )
    )

    for row_position, layer_key in enumerate(fresh.layer_keys):
        escaped_layer = _escape(layer_key)
        lines.append("            <tr>")
        lines.append(
            f'              <th scope="row" title="{escaped_layer}" aria-label="Layer {fresh.layer_indices[row_position]} {escaped_layer}">Layer {fresh.layer_indices[row_position]}<span class="axis-key">{escaped_layer}</span></th>'  # noqa: E501
        )
        for expert_position, expert_key in enumerate(fresh.expert_keys[row_position]):
            value = deltas[row_position][expert_position]
            css_class, signed_level = _signed_bin(float(value), float(maximum_absolute))
            visible = _format_delta(metric, value)
            accessible = _escape(
                f"Layer {fresh.layer_indices[row_position]} {layer_key}; Expert "
                f"{expert_position} {expert_key}; {metric} {value}; signed delta level "
                f"{signed_level}"
            )
            lines.append(
                f'              <td class="{css_class}" data-delta="{signed_level}" '
                f'aria-label="{accessible}" title="{accessible}">{_escape(visible)}</td>'
            )
        lines.append("            </tr>")

    lines.extend(
        (
            "          </tbody>",
            "        </table>",
            "      </div>",
            "    </section>",
            "    <footer>EXPERIMENTAL: this static view preserves the supplied inspection axes and does not infer tokenization, generation, specialization, causality, or certification.</footer>",  # noqa: E501
            "  </main>",
            "</body>",
            "</html>",
        )
    )
    return "\n".join(lines) + "\n"


__all__ = [
    "ROUTING_COMPARE_HEATMAP_SCHEMA_VERSION",
    "render_routing_load_comparison",
]
