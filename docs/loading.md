# Model-source and loading contracts

`moeatlas.loading` remains a schema/planning boundary only. It never imports
PyTorch, Transformers, safetensors, a custom loader, or an already-instantiated
module; it does not resolve paths, inspect caches, contact Hugging Face, or
download checkpoints. The separate `moeatlas.runtime` module consumes a
validated plan only in its explicit lazy HF/local loading calls.

## Source requests

The discriminated `SourceRequest` union supports:

- `HuggingFaceSource`: canonical `model_id`, required requested revision,
  optional separate `TokenizerRequest`, and explicit offline/download policy;
- `LocalSource`: a lexically normalized runtime path plus portable
  `model_id`/revision. The absolute path remains in the machine-local plan
  JSON, but is excluded from portable intent and the plan ID; it is not read
  or resolved during validation;
- `InstanceSource`: portable identity for an already-instantiated module,
  with no runtime object field; and
- `CustomLoaderSource`: validated `module:function` syntax that is not
  imported by validation.

Requested revisions such as `main` or a tag remain requests, not immutable
resolution claims. A tokenizer revision is separate unless
`inherit_model_revision=True` is explicitly set. No source silently equates a
model and tokenizer revision.

## Load policy

`LoadConfig` defaults to CPU, `preserve` dtype, no quantization, offline mode,
no downloads, and `trust_remote_code=False`. CUDA device maps may describe
multiple CUDA devices; exact devices reject incompatible maps, MPS is accepted
as best-effort, and `auto` is a device selection rather than a device-map
target. Remote-code execution requires an explicit acknowledgement. Arbitrary
`loader_options` remain finite JSON data. Top-level option names controlled by
this contract cannot override audited fields such as revisions, device, dtype,
download policy, remote code, or quantization. Nested backend-specific objects
are preserved as data (even when they use non-security words such as `dtype`
or `revision`) and cannot override the audited top-level policy fields.
Credential-bearing keys and header/cookie containers are forbidden recursively
with case- and punctuation-insensitive matching. This covers variants such as
`Token`, `Authorization`, `api-key`, passwords, secrets, credentials, and
`headers`; benign backend names such as `secret_sauce_mode` remain data.

`DTypePolicy` describes requested loading intent. Its explicit values map to
the existing core `DType` only as a later manifest hint; `preserve` maps to
`DType.UNKNOWN` until runtime observation produces a manifest. It is not a
second observed-dtype contract.

`LoadingPlan.plan_id` is a full SHA-256 over portable request intent. Local
absolute paths and runtime objects are excluded, so equivalent logical local
sources on two machines share an ID. Frozen nested options prevent mutation
after ID calculation. Derived `security_warnings` report remote code,
downloads, quantization, MPS, and custom-loader user-code risks; callers cannot
inject warning fields.

## Resolution evidence

`ResolvedSource` records requested revisions separately from externally
resolved revisions and therefore always contains an immutable resolved model
revision. An entirely unresolved plan uses `LoadingPlan.resolution=None`; a
resolved tokenizer remains optional and is represented as a paired nullable
revision/evidence set. An immutable resolved revision must pair with
`ImmutableRevisionEvidence`: either a full 40-character Git commit or a full
SHA-256 content digest, with the resolved revision matching the evidence. A
branch, tag, short hash, or arbitrary “claimed” string cannot be labeled
immutable. Creating this contract performs no resolution. Runtime loading
requires the existing immutable evidence and keeps real checkpoint/network/GPU
validation deferred to MV-01/MV-02 and the final VM.

## CLI plan files

`moeatlas scan --loading-plan PLAN.json` accepts the same `LoadingPlan` JSON
contract for an explicit HF/local static scan. The CLI validates the whole
document, requires immutable model and tokenizer resolution, rejects
`InstanceSource`/`CustomLoaderSource`, and delegates the unchanged plan to
`moeatlas.runtime.load_and_scan()`. It does not infer identities, resolve
revisions, inspect local paths, or override policy. Direct positional CLI
sources remain limited to `fixture:synthetic`; real checkpoint certification
and VM evidence remain deferred in MV-01/MV-02.
