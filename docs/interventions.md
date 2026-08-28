# Interventions

The `moeatlas.interventions` package is the only component allowed to mutate
a loaded module for causal observation, and it does so under one hard
contract: `run_intervention()` produces an outcome only after the module was
restored to its pre-intervention state. Every path — recipe-application
failure, observed-execution failure, cancellation, or restoration failure on
the success path — restores first and only then reports. A failed
restoration is surfaced as a distinct `restore` stage so a dirty module is
never mistaken for a clean one.

## Recipes

An `InterventionRecipe` is an immutable, content-addressed description of
exactly one bounded manipulation over family-neutral target labels. The
fixed operation vocabulary is `ablate`, `scale`, `reroute`, and
`alter_router`. Parameter exclusivity is enforced per operation: `scale`
requires exactly one finite `factor`, `alter_router` requires exactly one
finite `bias`, and `reroute` requires `alternates` covering exactly the
target set with each alternate distinct from its target; `ablate` takes no
parameters. Recipes serialize canonically as `moeatlas.intervention_recipe`
artifacts, and `recipe.fingerprint` returns the `sha256:<64 hex>` digest of
that canonical form — the same value recorded in `InterventionLineage` on
derived run specifications, so lineage binds to exact recipe content.

An `InterventionBudget` bounds one execution (`max_targets`) and serializes
as a `moeatlas.intervention_budget` artifact. Budgets are immutable inputs,
never ambient state: the same recipe under a smaller budget fails loudly at
the `contract` stage instead of silently narrowing the manipulation.

## Engine

The engine stays family-blind: native snapshot/apply/restore semantics live
behind an `InterventionCapability` supplied by an adapter, which must
provide callable `capture()`, `restore()`, and `apply()` primitives. The
engine sequences capture → apply → observe → restore, catches every failure
including cancellation, and guarantees restoration before anything is
reported. The returned `InterventionOutcome` records the manipulation —
schema version, recipe fingerprint, operation, targets — as a canonical
`moeatlas.intervention_outcome` artifact; callers own their observations.

Error stages are fixed: `contract`, `capture`, `apply`, `execute`,
`restore`. Error messages are safe and never echo input contents.

## Live baseline-derived workflow

The local server can execute real `ablate` and `scale` recipes against
independently exposed routed-expert modules. It can also execute `ablate`
against proven packed expert containers when the loaded model declares a
reversible Hugging Face expert backend. A completed baseline records its
resolved model and dataset revisions, chosen columns, row budget, generation
settings, and privacy policy. `POST /api/interventions/start` reconstructs
that exact request, adds immutable `InterventionLineage`, installs temporary
expert hooks, executes the same ordered rows, and restores every hook before
publishing evidence.

The paired `moeatlas.intervention_evidence` artifact records output-digest
changes, optional normalized exact-match score deltas when a reference column
was explicitly configured, row latency deltas, per-target invocation counts,
and restoration status. Routing deltas remain available through the ordinary
run-comparison heatmap. A target with zero invocations is explicitly marked as
not exercised and cannot support a causal claim.

The UI exposes only coordinates derived from the baseline discovery report.
Runs created before reconstruction metadata and output digests were added
must be repeated once as a fresh baseline.

During discovery, MoEAtlas also reads the loaded model's public Hugging Face
`get_experts_implementation()` snapshot before releasing model memory. Each
top-level or nested-model scope is retained separately. Built-in `eager`,
`batched_mm`, and `grouped_mm` declarations are distinguished from the fused
`sonicmoe` backend; accelerated or custom registrations remain unresolved
unless their semantics are known. The package uses duck typing and does not
import Transformers during ordinary model-free use.

On the first real run forward, a compatible Hugging Face backend receives a
temporary per-registry pass-through implementation. The delegate calls the
original backend without modifying arguments or its return value, records the
number of invocations, and restores the model's exact backend snapshot plus
every temporary registry entry before returning. The run records `verified`,
`not_exercised`, or `unavailable`; a restoration failure stops execution rather
than allowing later rows to use a dirty model. This handshake adds no separate
baseline or overhead-measurement run.

## Operation capability matrix

Discovery and the run-specific intervention endpoint publish one verdict per
operation instead of a single generic support flag:

- `capture_routing` is a run-validation candidate when static router targets
  exist; it becomes evidence only after a real forward.
- `zero_contribution` and `scale_contribution` are available for independently
  exposed expert hooks. `zero_contribution` is additionally available for a
  proven packed layout with a live expert backend: MoEAtlas preserves the
  selected top-k indices and masks only matching dispatch weights. Neither path
  changes routing or skips expert compute.
- `exclude_and_renormalize` is reported as not implemented when a router seam
  exists because safe execution still needs writable top-k weights and exact
  renormalization.
- `reroute_next_best` is reported as not implemented because capture of chosen
  experts does not prove access to next-best logits or writable dispatch.
- `skip_compute` is distinct from zeroing an already-computed output and remains
  not implemented until a pre-dispatch adapter proves that the selected kernel
  was bypassed.

Every row includes its status, bounded reason, structural/runtime evidence,
and whether the requested semantics change routing or skip compute. Static
packed targets may be selected, but the worker still requires and verifies a
reversible live backend before mutation.

## Honest scope

Local tests prove the mechanics without a model stack. Real checkpoint
certification remains a VM/GPU validation step. Models whose expert weights
are packed into tensors are reported separately from models whose execution
backend is fused. Static discovery reports the weight layout but leaves the
execution backend unresolved; it never treats packed storage as proof of
kernel fusion. Packed ablation is proven locally at the backend contract and
still requires real-checkpoint GPU certification. A backend name is declaration
evidence, not proof that a particular forward exercised it; that requires the
runtime handshake. Models that expose neither independent forward-hook modules
nor a proven packed expert axis are reported as unsupported rather than silently
approximated. One changed output is not enough to label an expert as
task-specialized: use scored task rows, negative-control datasets, repeated
runs, and multiple target controls before making that claim.
