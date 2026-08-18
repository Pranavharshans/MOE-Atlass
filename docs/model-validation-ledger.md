# Model and GPU validation ledger

**Status: deferred** — this ledger is intentionally opened during the
foundation feature and will be completed in the final VM phase.

No model files are downloaded by the repository setup or model-free test
commands. A package being importable is not model validation. A passing CPU
unit test is not GPU compatibility evidence.

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
