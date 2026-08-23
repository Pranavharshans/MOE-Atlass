# MoEAtlas roadmap

Status: active

This is the only active product roadmap. The
[PRD v1 audit](prd-audit.md) is a frozen historical acceptance record, while
the [validation ledger](model-validation-ledger.md) is the authority for real
model, GPU, filesystem, and scale evidence.

## Product objective

A researcher can paste a Hugging Face MoE model and dataset, run bounded rows,
inspect validated routing and expert activity, choose candidate layer × expert
targets, apply reversible interventions, and compare task behavior against the
exact baseline.

## Delivered foundation

| Area | Current state |
| --- | --- |
| Model and dataset intake | Implemented with immutable revision evidence and explicit downloads |
| Generic MoE discovery | Implemented without a central family allowlist |
| Routing capture | Implemented for router payloads that pass runtime validation |
| Expert activity | Implemented when experts expose compatible module hooks |
| Storage and export | Versioned immutable shards, bundles, CSV, and Parquet |
| Research UI | Model intake, discovery, runs, heatmaps, comparisons, and interventions |
| Baseline interventions | Real zeroing and scaling for independently exposed experts |
| Causal comparisons | Matched input, score, output, latency, routing, invocation, and restoration evidence |
| Replication | Repeated studies and optional negative controls |
| Backend detection | Packed weights and fused execution reported separately |
| Hugging Face backend discovery | Public expert implementation declarations retained per model scope |
| Runtime handshake | Reversible pass-through verification on the first compatible forward |
| Capability reporting | One explicit verdict for every supported or planned operation |

Delivered does not mean universally certified. Every model-dependent claim
remains bounded to its recorded model revision and runtime.

## Next milestone 1: compatibility certification

Build and maintain a reproducible model matrix rather than treating individual
successful runs as universal support.

Required rows:

- LiquidAI/LFM2.5-8B-A1B
- ibm-granite/granite-4.0-h-tiny
- openai/gpt-oss-20b
- inclusionAI/Ling-3.0-tiny
- amd/Instella-MoE-16B-A3B-Think
- Qwen MoE checkpoints
- NVIDIA Nemotron MoE checkpoints
- GLM MoE checkpoints
- Gemma MoE checkpoints
- Kimi or Kimi-derived MoE checkpoints

Each row must record model revision, runtime profile, hardware, precision,
load, discovery, capture, heatmap, intervention, cleanup, and the exact failure
stage. A model is supported only when its full claimed path passes.

Exit criteria:

1. Failed model loads no longer terminate the HTTP server.
2. Model memory is released after completed, failed, and cancelled jobs.
3. Dependency and remote-code incompatibilities produce actionable diagnostics.
4. At least one model from each materially different backend class completes
   the supported journey.
5. The validation ledger contains commands and artifacts that another machine
   can reproduce.

## Next milestone 2: packed and fused expert interventions

Current detection is evidence, not intervention support.

Planned order:

1. Wrap a declared Hugging Face expert backend without changing its output.
2. Implement contribution zeroing and scaling at the backend boundary.
3. Prove restoration and passive equivalence.
4. Verify the requested layer × expert target was exercised.
5. Benchmark overhead and memory.
6. Add backend adapters only when the common Hugging Face seam cannot express
   the required operation.

Exit criteria:

- Packed and fused paths never display an executable action before runtime proof.
- The baseline and intervention use identical inputs and generation settings.
- Native and pass-through outputs match within declared tolerance.
- Failure or cancellation restores the exact original backend.

## Next milestone 3: router interventions

Implement operations that actually change routing:

1. Exclude selected experts and renormalize the remaining chosen weights.
2. Reroute to the next-best eligible expert when complete router scores exist.
3. Distinguish output suppression from real compute skipping.
4. Add compute skipping only where a pre-dispatch seam proves that the expert
   kernel was bypassed.

Exit criteria:

- Routing changes are visible in persisted assignments and comparison heatmaps.
- Renormalized weights satisfy the model's native invariants.
- Next-best rerouting is unavailable when full candidate scores are absent.
- Compute savings are claimed only with measured kernel or timing evidence.

## Next milestone 4: research validity and usability

- Dataset-specific evaluators beyond the bounded built-ins.
- Candidate ranking across tasks, prompts, and random negative controls.
- Confidence intervals and effect-size views in the UI.
- Exportable experiment reports tying every figure to immutable run evidence.
- Browser tests for the complete discovery → run → intervention → comparison flow.
- Large-matrix and million-event storage benchmarks.

## Delivery rules

- Add no family-specific code when a public backend or structural contract can
  express the same behavior.
- Add no model compatibility claim without a real checkpoint record.
- Keep model-free tests independent of Torch, Transformers, networks, and GPUs.
- A new abstraction needs at least two real consumers.
- A new dependency must remove more maintenance than it adds.
- One feature should make the next model easier to support; otherwise review it
  for deletion.
- Push coherent, passing changes rather than implementation diaries.

## Release gate

A production-ready 1.0 requires:

- A published compatibility matrix with representative separate, packed, fused,
  remote-code, and quantized models.
- Reliable process isolation and accelerator cleanup.
- At least one validated packed or fused intervention path.
- Reproducible baseline/intervention studies with controls.
- Clean wheel installation and restart testing on a fresh GPU VM.
- Documentation that describes the shipped application rather than historical
  implementation phases.
