"""Deterministic, dependency-free HTML for one routing-load matrix."""

from __future__ import annotations

import html

from .routing_load import RoutingLoadMatrix

ROUTING_HEATMAP_SCHEMA_VERSION = "1.0"

_METRICS = {
    "assignment_counts": (
        "Selected assignment count per layer × expert.",
        "count",
    ),
    "assignment_shares": (
        "Assignment share per layer: count ÷ (token count × routed top-k).",
        "share",
    ),
    "load_ratios": (
        "Load ratio per layer: share × expert count; 1.0× is ideal uniform "
        "load—not specialization.",
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


def _fresh_matrix(matrix: RoutingLoadMatrix) -> RoutingLoadMatrix:
    return RoutingLoadMatrix(
        schema_version=matrix.schema_version,
        store_schema_version=matrix.store_schema_version,
        event_schema_version=matrix.event_schema_version,
        run_key=matrix.run_key,
        model_key=matrix.model_key,
        adapter_name=matrix.adapter_name,
        adapter_version=matrix.adapter_version,
        inspection_digest=matrix.inspection_digest,
        layout=matrix.layout,
        shard_keys=matrix.shard_keys,
        token_count=matrix.token_count,
        assignment_count=matrix.assignment_count,
        routed_top_k=matrix.routed_top_k,
        layer_keys=matrix.layer_keys,
        layer_indices=matrix.layer_indices,
        expert_keys=matrix.expert_keys,
        assignment_counts=matrix.assignment_counts,
        assignment_shares=matrix.assignment_shares,
        load_ratios=matrix.load_ratios,
    )


def _heat_bin(value: float, maximum: float) -> int:
    if maximum <= 0.0 or value <= 0.0:
        return 0
    return 1 + min(7, int((value / maximum) * 8))


def _format_value(metric: str, value: int | float) -> str:
    if metric == "assignment_counts":
        return str(value)
    if metric == "assignment_shares":
        return f"{value:.6%}"
    return f"{value:.6f}×"


def render_routing_load_heatmap(
    matrix: RoutingLoadMatrix,
    *,
    metric: str,
    max_cells: int,
) -> str:
    """Render one complete routing-load matrix as standalone HTML5."""

    if type(metric) is not str:
        raise TypeError("metric must be an exact string")
    if metric not in _METRICS:
        raise ValueError("metric must be assignment_counts, assignment_shares, or load_ratios")
    _strict_positive_cells(max_cells)
    if type(matrix) is not RoutingLoadMatrix:
        raise TypeError("matrix must be an exact RoutingLoadMatrix")
    fresh = _fresh_matrix(matrix)
    expert_count = len(fresh.expert_keys[0])
    cells = len(fresh.layer_keys) * expert_count
    if cells > max_cells:
        raise ValueError("matrix cells exceed max_cells")

    metric_definition, value_kind = _METRICS[metric]
    values = getattr(fresh, metric)
    maximum = max(max(row) for row in values)
    title = "Layer × Expert routing-load heatmap"
    document_title = f"MoEAtlas — {title} — {fresh.run_key}"

    lines = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '  <meta charset="utf-8">',
        '  <meta name="viewport" content="width=device-width, initial-scale=1">',
        '  <meta name="moeatlas-routing-heatmap-schema" content="1.0">',
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
        "    .heat-0 { background: hsl(210 45% 98%); } .heat-1 { background: hsl(210 55% 94%); } .heat-2 { background: hsl(205 65% 89%); } .heat-3 { background: hsl(200 70% 82%); } .heat-4 { background: hsl(190 72% 72%); } .heat-5 { background: hsl(175 68% 61%); } .heat-6 { background: hsl(150 62% 50%); } .heat-7 { background: hsl(104 58% 48%); } .heat-8 { background: hsl(70 70% 45%); color: #13210d; }",  # noqa: E501
        "    .legend { display: flex; flex-wrap: wrap; gap: 0.35em 0.8em; font-size: 0.82em; }",
        "    .legend span { padding: 0.2em 0.45em; border: 1px solid #c9d1d9; }",
        "    footer { margin-top: 1.5em; color: #536471; font-size: 0.82em; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <main>",
        f"    <h1>{_escape(title)}</h1>",
        '    <p class="warning" role="note">Routing load only. Selection frequency is association evidence, not expert specialization or causal effect.</p>',  # noqa: E501
        f"    <p><strong>Metric:</strong> {_escape(metric)} — {_escape(metric_definition)}</p>",
        '    <section aria-labelledby="provenance-heading">',
        '      <h2 id="provenance-heading">Frozen provenance and counts</h2>',
        '      <dl class="provenance">',
        f"        <div><dt>Heatmap schema</dt><dd>{_escape(ROUTING_HEATMAP_SCHEMA_VERSION)}</dd></div>",  # noqa: E501
        f"        <div><dt>Routing-load schema</dt><dd>{_escape(fresh.schema_version)}</dd></div>",
        f"        <div><dt>Store schema</dt><dd>{_escape(fresh.store_schema_version)}</dd></div>",
        f"        <div><dt>Event schema</dt><dd>{_escape(fresh.event_schema_version)}</dd></div>",
        f"        <div><dt>Run key</dt><dd>{_escape(fresh.run_key)}</dd></div>",
        f"        <div><dt>Model key</dt><dd>{_escape(fresh.model_key)}</dd></div>",
        f"        <div><dt>Adapter</dt><dd>{_escape(fresh.adapter_name)} / {_escape(fresh.adapter_version)}</dd></div>",  # noqa: E501
        f"        <div><dt>Inspection digest</dt><dd>{_escape(fresh.inspection_digest)}</dd></div>",
        f"        <div><dt>Layout</dt><dd>{_escape(fresh.layout)}</dd></div>",
        f"        <div><dt>Token count</dt><dd>{_escape(fresh.token_count)}</dd></div>",
        f"        <div><dt>Assignment count</dt><dd>{_escape(fresh.assignment_count)}</dd></div>",
        f"        <div><dt>Routed top-k</dt><dd>{_escape(fresh.routed_top_k)}</dd></div>",
        f"        <div><dt>Layer count</dt><dd>{_escape(len(fresh.layer_keys))}</dd></div>",
        f"        <div><dt>Expert count per layer</dt><dd>{_escape(expert_count)}</dd></div>",
        "      </dl>",
        "      <details open>",
        f"        <summary>Shard provenance ({_escape(len(fresh.shard_keys))} shards)</summary>",
        f"        <p>Shard count: {_escape(len(fresh.shard_keys))}</p>",
        "        <ul>",
        *[f"          <li>{_escape(shard_key)}</li>" for shard_key in fresh.shard_keys],
        "        </ul>",
        "      </details>",
        "    </section>",
        '    <section aria-labelledby="legend-heading">',
        '      <h2 id="legend-heading">Heat scale</h2>',
        f'      <p id="heat-formula">Global heat-0..8 bins use 1 + min(7, int((v / m) * 8)) for positive values, where m is the maximum cell in this artifact; zero values use heat-0. Global maximum: {_escape(maximum)} {_escape(value_kind)}.</p>',  # noqa: E501
        '      <p id="heat-legend-text">Zero values are heat-0; low values use low positive bins; high values use high bins. Intensity is relative to the maximum cell in this artifact.</p>',  # noqa: E501
        '      <div class="legend" aria-label="Heat scale from zero to eight">',
        *[f'        <span class="heat-{index}">heat-{index}</span>' for index in range(9)],
        "      </div>",
        "    </section>",
        '    <section aria-labelledby="matrix-heading">',
        '      <h2 id="matrix-heading">Layer × expert matrix</h2>',
        '      <div class="table-wrap">',
        '        <table aria-describedby="heat-formula">',
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
            value = values[row_position][expert_position]
            heat = _heat_bin(float(value), float(maximum))
            visible = _format_value(metric, value)
            accessible = _escape(
                f"Layer {fresh.layer_indices[row_position]} {layer_key}; Expert "
                f"{expert_position} {expert_key}; {metric} {value}"
            )
            lines.append(
                f'              <td class="heat-{heat}" data-heat="{heat}" '
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


# Preserve the historical spelling as an identity alias.  It is deliberately
# not a family-specific renderer, so all families share one byte-stable HTML
# contract.
render_mixtral_routing_load_heatmap = render_routing_load_heatmap

__all__ = [
    "ROUTING_HEATMAP_SCHEMA_VERSION",
    "render_routing_load_heatmap",
    "render_mixtral_routing_load_heatmap",
]
