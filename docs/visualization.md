# Routing-load visualization

Feature 28 provides one dependency-free, deterministic model-neutral
presentation boundary:
`render_routing_load_heatmap(matrix, *, metric, max_cells)`. It accepts only a
freshly reconstructed `RoutingLoadMatrix` and renders a complete
standalone HTML5 document. The exact metrics are `assignment_counts`,
`assignment_shares`, and `load_ratios`; the cell budget is strict and applies
before rendering.

The portable static HTML export is caller-owned and permanent:

```python
from pathlib import Path
import webbrowser

from moeatlas.analysis import render_routing_load_heatmap

# `inspection` is an existing AdapterInspection and `matrix` is an existing
# RoutingLoadMatrix from Feature 28.
inspection_path = Path("inspection.json")
inspection_path.write_text(inspection.to_json(), encoding="utf-8")
html = render_routing_load_heatmap(
    matrix,
    metric="load_ratios",
    max_cells=10_000,
)
artifact = Path("routing-load.html")
artifact.write_text(html, encoding="utf-8")
webbrowser.open(artifact.resolve().as_uri())
```

The historical `render_mixtral_routing_load_heatmap` name is an identity alias
of the neutral renderer. The renderer itself remains pure: it returns the string and does not save or
open anything. A future React UI is a separate presentation surface, not a replacement
for this portable static HTML export path.

The document preserves the matrix's layer and expert axes, numeric layer order,
layout, schema versions, run/model/adapter identity, inspection digest, token
and assignment counts, routed top-k, and shard keys inside native `<details>`
provenance with a visible shard count. Every cell is rendered, including
zero-count experts. Visible values use integer counts, six-decimal percent
shares, or six-decimal `×` ratios. Accessible table headers carry full
canonical layer and expert keys; cell `aria-label` and `title` values retain
unrounded canonical values with quote-aware escaping.

Heat classes are global and deterministic: zero values use `heat-0`, while
positive values use `1 + min(7, int((v / m) * 8))`, where `m` is the maximum
cell in this artifact, with the maximum reaching `heat-8`. The legend explicitly distinguishes zero, low, and high
intensity relative to that maximum. For load ratios, `1.0× is ideal uniform
load—not specialization`. The output has one embedded style block, system fonts,
responsive spacing, tabular numeric alignment, a strict Content-Security-Policy
(CSP) with no-resource defaults, and a
no-referrer policy. It contains no JavaScript, external resource, form, event,
storage, network, cache, model, or runner boundary.

The visible warning is intentional: `Routing load only. Selection frequency is association evidence, not expert specialization or causal effect.` This is an
`EXPERIMENTAL` static artifact over an accepted
Feature 20 value; it is not a UI, server, catalog, prompt, metric, or
model-validation claim. Tokenization, generation, checkpoint execution, and
MV-01 through MV-08 remain deferred. Feature 21 does not alter stored shard
bytes or the Feature 20 matrix contract.

## Comparison visualization

Feature 30 adds the dependency-free, deterministic presentation boundary for
Feature 29 values: `render_routing_load_comparison(comparison, *, metric,
max_cells)`. It accepts only a freshly reconstructed `RoutingLoadComparison`
and renders one complete standalone HTML5 document. The exact metrics are
`count_deltas`, `share_deltas`, and `ratio_deltas`; the cell budget is strict
and applies before rendering.

The document preserves both run keys, both sorted shard-key tuples inside
separate native `<details>` provenance sections, the shared model/adapter/
inspection/layout identity, token and top-k counts, and both assignment
counts. Visible values are signed: integer count deltas (`+1`, `-1`, `+0`),
six-decimal percent share deltas, or six-decimal `×` ratio deltas.

Delta classes are global, signed, and deterministic: zero cells use
`delta-zero`; positive deltas use heat bins and negative deltas use cold bins,
each via `1 + min(7, int((|v| / m) * 8))` where `m` is the maximum absolute
cell in this artifact. The legend renders the full `cold-8..delta-zero..heat-8`
scale and explains that warm colors mark experts the comparison run routes
away from while cool-to-green colors mark experts it routes toward. The
visible warning is `Routing-load deltas only. Differences in selection
frequency are association evidence, not expert specialization or causal
effect.`

The renderer remains pure: it returns the string and does not save or open
anything. The output has the same strict CSP, no-JavaScript, no-external-
resource posture as the single-run heatmap, with quote-aware escaping on every
interpolated key. It is an `EXPERIMENTAL` static artifact over an accepted
Feature 29 value; a delta is association evidence only and is never a
specialization, improvement, or causal claim. MV-01 through MV-08 remain
deferred; Feature 30 does not alter stored shard bytes, the Feature 28 matrix
contract, or Mixtral heatmap bytes.

## Serving in the local UI

The local server (`docs/server.md`) serves published heatmap documents at
`/api/runs/{run_key}/heatmap` and embeds them in the frontend's heatmap view
via a plain iframe. Because these documents are strict-CSP, no-JavaScript,
no-external-resource HTML5 files, embedding requires no scripts and adds no
surface: what the browser renders is byte-identical to the CLI-written
artifact. Runs without a published document show an honest empty state that
points to the `moeatlas heatmap` command above.

Feature 22 provides the bounded CLI composition for callers who already have
an inspection JSON document and Feature 19 workspace:

```bash
moeatlas heatmap /data/routing \
  --inspection inspection.json --run-key run-1 --metric load_ratios \
  --max-inspection-bytes 1000000 --max-routing-rows 1000000 \
  --max-source-bytes 100000000 --max-matrix-cells 100000 \
  --output routing-load.html
```

The CLI validates the output destination first, reads the bounded non-symlink
inspection, aggregates and renders exactly once, and publishes through the
existing `write_report_atomic()` atomic writer. Output must end in lowercase
`.html`; existing files require `--force`, and failed publication leaves no
partial artifact or temporary file. Install the optional DuckDB `store` extra
for a real Feature 19 workspace. It remains a thin, model-free composition boundary;
tokenization, checkpoint loading, browser automation, and generation are not
part of this command. Feature 22 is `EXPERIMENTAL` and does not change
MV-01 through MV-08.
