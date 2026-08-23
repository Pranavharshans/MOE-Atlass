# Archived PRD v1 acceptance audit

Status: frozen historical record. This document captures the model-free v1
acceptance review at the time it was written. It is not the current roadmap or
the current compatibility report. Use [roadmap.md](roadmap.md) for planned work
and [model-validation-ledger.md](model-validation-ledger.md) for current
model/GPU evidence.

This audit maps every MoEAtlas PRD v1 acceptance area to its implementation
surface, its local test evidence, and its honest status. It was produced
without any VM, GPU, or checkpoint access: every "model-free complete"
claim below rests only on the local serialized gate (pytest, ruff,
unittest discovery, wheel/sdist build) over synthetic fixtures, and every
row that needs real infrastructure points at its named deferred ledger row
instead of pretending. Nothing here certifies a model family; certification
is the final VM phase's job (MV-01 through MV-08, ST-01 through ST-04).

## Acceptance-area traceability

| PRD area | Implementation surface | Local evidence | Status |
| --- | --- | --- | --- |
| §§5–7 structure discovery and inspection | `moeatlas.discovery`, `moeatlas.runtime.scan`, static adapters | scanner/adapter contract suites over local fixtures | model-free complete |
| §§7–8 universal capability contracts | `moeatlas.runtime.capabilities`, `moeatlas.adapters.universe`, adapter decoders | capability/universe/decoder suites incl. unknown-family and non-rectangular fixtures | model-free complete; native equivalence stays MV-03/MV-04 |
| §9 run identity, lifecycle, engine | `moeatlas.runs`, `moeatlas.services.run_engine`, `run_service`, `run_inputs` | lifecycle/state-machine/checkpoint/resume suites over fake runtimes | model-free complete; real generation stays MV-01/MV-02 |
| §10 storage, export, interchange | `moeatlas.store` (shards, catalog, ports, run export/tables) | shard/manifest/tamper/atomicity/round-trip suites, v1 compatibility | model-free complete; target-filesystem durability stays ST-01–ST-03 |
| §11 descriptive analyses and Evidence Cards | `moeatlas.analysis` (load, summary, compare, heatmap, association, agreement, stability, margin, churn, corouting, similarity, causal evidence, evidence cards) | formula/golden/degenerate/budget/serialization suites over synthetic evidence | model-free complete; real per-token task evidence stays MV-deferred |
| §11.4 interventions and causal evidence | `moeatlas.interventions`, `analysis.causal_evidence` | recipe/engine/restoration/cancellation suites over synthetic modules | model-free complete; real causal claims stay MV-deferred |
| §§12–14 server and UI | `moeatlas.server` (DTOs, live job control plane, analysis/export routes), `moeatlas ui`, `frontend/` | TestClient endpoint/static-bundle suites, launch-policy suites, local browser smoke | model-free complete for the live local control plane; broad browser E2E and VM execution evidence remain deferred |
| §15 plugins | `moeatlas.adapters.registry`, `moeatlas.executors` seam | registry/policy/collision/isolation suites, fake plugins in subprocesses | model-free complete |
| §16 CLI and Python APIs | `moeatlas.cli` (scan, heatmap, routing-runs, compare, doctor, adapters, run, export, ui) | parser/handler/fixed-error suites over local fixtures | model-free complete |
| §17 privacy and retention | `PrivacyPolicy` defaults, redaction in events/storage/export, `services.retention` | redaction-fidelity suites, retention classification suites | model-free complete; scale enforcement stays ST-04 |
| §18 reliability and budgets | budgets across every entry point, checkpoints, resume, rebuild/reopen | corruption/failure-injection suites, budget-rejection suites | model-free complete; crash/fsync durability stays ST-03 |
| §19 benchmarks and release | `moeatlas.benchmarks`, governance files, CI workflow, examples | benchmark contract suites, anchor tests, clean-subprocess example suite | model-free complete; clean-install/Docker/screenshots stay deferred release evidence |
| §20 final certification | — | — | blocked: requires provisioned VM/GPU access that does not exist |

## Deferred and blocked infrastructure rows

The rows below are the complete list of claims this repository deliberately
does not make. They stay `deferred` (evidence pending infrastructure) or
`blocked` (infrastructure unavailable) in the
[validation ledger](model-validation-ledger.md); no local test promotes
them.

| ID | Claim requiring infrastructure |
| --- | --- |
| MV-01 | Real pinned MoE checkpoint loads through a certified adapter |
| MV-02 | Real static discovery/inspection output on that checkpoint |
| MV-03 | Routing capture matches native output or a golden reference |
| MV-04 | Passive output equivalence with hooks enabled and disabled |
| MV-05 | Routing-only overhead and memory behavior measured on hardware |
| MV-06 | CUDA validation on the provisioned VM |
| MV-07 | Fused/quantized limited-path validation with explicit downgrade |
| MV-08 | Complete model-dependent suite re-run after packaging |
| ST-01 | Legacy and packed shard validation on the target VM |
| ST-02 | Reopen/list and idempotent/conflict behavior on the target filesystem |
| ST-03 | Durability, permissions, and crash/rename-fsync recovery |
| ST-04 | Scale, workspace/catalog integration, and query-surface evidence |

## Compatibility tiers

Family compatibility tiers are generated only from evidence. Today no
family is certified: Mixtral and Qwen3.5-MoE carry the deepest synthetic
and contract coverage plus shipped adapters, DeepSeek/MiniMax and the
generic fallback carry contract-level support, and all of it remains
`EXPERIMENTAL` until MV-01 through MV-08 record real revisions on the final
VM. The release-time review of official immutable revisions happens after
that phase, per the multi-family release constitution in
[architecture](architecture.md).

## Audit verdict

Every acceptance area that can be satisfied without models,
networks, or accelerators is implemented, contract-tested, documented, and
pushed behind the serialized local gate. Every area that cannot is recorded
above with its named deferred row and is not claimed. The next concrete
step is provisioning the final VM to execute MV-01 through MV-08 and
ST-01 through ST-04; until then this project's honest state is
"model-free complete, certification blocked on infrastructure."
