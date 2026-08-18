# Model and GPU validation ledger

**Status: deferred** — this ledger is intentionally opened during the
foundation feature and will be completed in the final VM phase.

No model files are downloaded by the repository setup or model-free test
commands. A package being importable is not model validation. A passing CPU
unit test is not GPU compatibility evidence.

Feature 9 adds lazy HF/local execution with fake optional modules in the
model-free suite. Those tests verify call arguments, observation, and rollback
only; they do not change MV-01/MV-02 status and do not inspect caches or fetch
checkpoints.

Feature 10 adds `runtime.load_and_scan()` as a cleanup-safe composition of the
resolved loader and static discovery. Its model-free tests verify dispatch,
identity binding, report validation, and retryable cleanup only; it does not
certify a real checkpoint or change the deferred MV-01/MV-02 status.

Feature 11 adds the plan-file CLI entrypoint. Its model-free tests verify
strict plan parsing, source/resolution preflight, unchanged delegation, and
publication safety only; it does not resolve or certify a real checkpoint and
does not change the deferred MV-01/MV-02 status.

Feature 12 adds the caller-supplied static semantic-adapter protocol. Its
model-free tests verify strict manifests, identity binding, STRUCTURE-only
provenance, safe error boundaries, and no-runtime-action behavior only; it does
not certify any architecture or change the deferred MV-01/MV-02 status.

Feature 13 adds the explicitly caller-selected `MixtralStaticAdapter()` for
two model-free structural surfaces: official Transformers 4.50 indexed
experts and current packed direct `gate_up_proj`/`down_proj` tensors. Its tests use standard-library
fixtures only and verify exact family/config/topology/shape evidence,
STRUCTURE-only unverified provenance, and safe rejection of Qwen, dense,
fused, and malformed surfaces. It does not infer a Transformers version,
observe routing, certify Mixtral behavior, or change the deferred MV-01/MV-02
status; real checkpoints and VM/GPU evidence remain required.

## Deferred checks

| ID | Check | Current status | Required evidence before completion |
| --- | --- | --- | --- |
| MV-01 | Load a small, pinned real MoE checkpoint through the first certified adapter | deferred | model ID and immutable revision, license, download source, loader config, output |
| MV-02 | Run static discovery and inspect the semantic manifest | deferred | scan JSON, module paths, expert count/top-k, warnings, capability tier |
| MV-03 | Capture routing against native output or a golden reference | deferred | exact command, fixture/prompts, tolerances, comparison result |
| MV-04 | Verify passive output equivalence with hooks enabled and disabled | deferred | baseline/probed outputs, dtype/device, tolerance, result |
| MV-05 | Measure routing-only overhead and memory behavior | deferred | hardware, software versions, baseline/probed timings, peak memory |
| MV-06 | Run CUDA validation on the provisioned VM | deferred | GPU model/driver/CUDA, command, logs, artifact path |
| MV-07 | Validate a fused/quantized or otherwise limited execution path | deferred | backend/quantization settings, capability downgrade, trace evidence |
| MV-08 | Re-run the complete model-dependent suite after packaging | deferred | installed wheel/version, test report, model cache location, result |

## Final VM execution record

Fill this section only when the VM is provisioned. Do not replace the
`deferred` status with an assumption.

```text
VM/provider:
Date (UTC):
OS / architecture:
Python:
MoEAtlas commit/tag:
PyTorch / Transformers / safetensors:
GPU / driver / CUDA:
Model ID and immutable revision:
Tokenizer ID and revision:
Commands:
Artifacts and logs:
Results:
Known limitations:
```

## Completion rule

Each row becomes `passed`, `failed`, or `blocked` only with the evidence named
in its final column. If a model cannot expose a semantic signal, record the
capability tier and limitation instead of treating missing data as a test pass.
