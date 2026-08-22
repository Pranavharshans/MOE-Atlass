# Changelog

All notable changes to MoEAtlas are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project follows
Semantic Versioning: pre-1.0 minor versions may break APIs, and breaking
changes are always listed.

## [Unreleased]

### Added

- Neutral event-validation layer with exact-alias storage APIs (Slice 1).
- Content-addressed run specifications, lifecycle contracts, lineage
  (`moeatlas.runs`) (Slice 2).
- Shared application services, workspace catalog, storage ports, bounded
  queries, migration/rebuild (Slice 3).
- Capability-oriented adapter/runtime contracts, Qwen3.5-MoE adapter,
  unknown-family fixtures, non-rectangular top-k support (Slice 4).
- Bounded raw evidence, run-evidence export bundles, open-format
  readers/writers, assignment query seam (Slice 5).
- Prompt/dataset run engine: batching, checkpoints, resume, cancellation,
  per-row failures (Slice 6).
- Descriptive analysis set: routing load/summary/compare/heatmap, task
  association, agreement, stability, router margin, route churn,
  co-routing, expert similarity, Evidence Cards (Slice 7).
- Adapter plugin registry with trust policy and collisions; complete
  headless CLI: `scan`, `heatmap`, `routing-runs`, `compare`, `doctor`,
  `adapters list`, `run`, `export` (Slice 8).
- Local read-only FastAPI server (`moeatlas.server`) and loopback-default
  `moeatlas ui` launch (Slice 9).
- Failure-safe intervention engine, immutable recipes/budgets, paired
  causal-evidence summaries (Slice 10).
- Retention evaluation over the workspace run registry (Slice 11).
- Expert-event runtime reality (R3): structure-driven `EXPERT_ACTIVITY`
  capture producing per-invoked-expert norms, store schema 2.0 shards with
  an `experts.parquet` table, declared expert budgets, tamper/conflict
  semantics mirroring routing rows, legacy `1.0` shard compatibility, and
  bounded per-layer activation summaries in analysis.
