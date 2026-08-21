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

## Honest scope

Synthetic modules prove the mechanics locally. Real-model causal claims —
effect sizes, regret, stability, and replication on actual checkpoints —
are deferred to the validation ledger until native-output certification;
unsupported, fused, or quantized paths surface explicit limitations rather
than silent degradation.
