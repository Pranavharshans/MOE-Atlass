# Routing-load visualization

Feature 21 adds one dependency-free, deterministic presentation boundary:
`render_mixtral_routing_load_heatmap(matrix, *, metric, max_cells)`. It accepts
only a freshly reconstructed `MixtralRoutingLoadMatrix` and renders a complete
standalone HTML5 document. The exact metrics are `assignment_counts`,
`assignment_shares`, and `load_ratios`; the cell budget is strict and applies
before rendering.

The portable static HTML export is caller-owned and permanent:

```python
from pathlib import Path
import webbrowser

from moeatlas.analysis import render_mixtral_routing_load_heatmap

# `matrix` is an existing MixtralRoutingLoadMatrix from Feature 20.
html = render_mixtral_routing_load_heatmap(
    matrix,
    metric="load_ratios",
    max_cells=10_000,
)
artifact = Path("routing-load.html")
artifact.write_text(html, encoding="utf-8")
webbrowser.open(artifact.resolve().as_uri())
```

The renderer itself remains pure: it returns the string and does not save or
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
