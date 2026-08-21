"""Model-neutral run identity and provenance contracts.

A :class:`RunSpecification` is the immutable, content-addressed intent for one
run: which resolved model and tokenizer revision, which probe plan, which
prompt or dataset fingerprint, which generation settings, which privacy
policy, and which intervention lineage. It binds existing artifacts by their
canonical identifiers and never loads a model, reads a dataset, contacts the
network, or executes anything.

The ``run_key`` is derived from the portable specification content so the same
intent always produces the same key and every stored event can name its run.
Workspace labels, tags, timestamps, and observed execution environments are
deliberately excluded from the identity: they describe where or when a
specification was used, not what it is.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, ClassVar, Literal, Self

from pydantic import (
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from ..core import (
    StrictManifestModel,
    VersionedManifest,
    canonical_identifier,
    stable_digest,
    validate_stable_identifier,
)
from ..loading import QuantizationPolicy, parse_loading_plan_id

RUN_SPEC_SCHEMA_VERSION = "1.0"

_RUN_KEY = re.compile(r"^run:([0-9a-f]{64})$")
_PLAN_ID = re.compile(r"^plan:([0-9a-f]{64})$")
_DIGEST = re.compile(r"^sha256:([0-9a-f]{64})$")

_MAX_TEXT = 1_000_000
_MAX_LOCATION = 500
_MAX_LABEL = 200
_MAX_MESSAGE = 500


class RunInputKind(str, Enum):
    """Kind of caller data a run consumes."""

    PROMPT = "prompt"
    DATASET = "dataset"

    def __str__(self) -> str:
        return self.value


class DatasetFormat(str, Enum):
    """Supported dataset descriptor formats; descriptors never fetch data."""

    JSONL = "jsonl"
    CSV = "csv"
    PARQUET = "parquet"
    TEXT = "text"
    ITERABLE = "iterable"
    HF_DATASETS = "hf_datasets"

    def __str__(self) -> str:
        return self.value


class RunMode(str, Enum):
    """Execution mode for dataset rows."""

    GENERATION = "generation"
    TEACHER_FORCED = "teacher_forced"

    def __str__(self) -> str:
        return self.value


class TokenTextPolicy(str, Enum):
    """Whether token text may be persisted with a run's events."""

    REDACTED = "redacted"
    STORED = "stored"

    def __str__(self) -> str:
        return self.value


class _FrozenMapping(dict[str, Any]):
    """Dict-compatible immutable JSON object for spec mappings."""

    @staticmethod
    def _immutable(*args: Any, **kwargs: Any) -> None:
        raise TypeError("run specification mapping is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable

    def __ior__(self, other: object) -> _FrozenMapping:
        self._immutable(other)
        return self


def _freeze_mapping(value: dict[str, Any]) -> dict[str, Any]:
    try:
        stable_digest(value)
    except TypeError as exc:
        raise ValueError("mapping must contain only finite, JSON-serializable values") from exc
    return _FrozenMapping({key: value[key] for key in sorted(value)})


def make_run_key(payload: Any) -> str:
    """Build the canonical ``run:<64 hex>`` key over portable specification content."""

    return f"run:{stable_digest(payload)}"


def parse_run_key(run_key: str) -> str:
    """Validate a canonical run key and return its digest."""

    if not isinstance(run_key, str):
        raise TypeError(f"run_key must be a string, got {type(run_key).__name__}")
    match = _RUN_KEY.fullmatch(run_key)
    if match is None:
        raise ValueError("run_key must use the canonical run:<64 lowercase hex> form")
    return match.group(1)


def parse_probe_plan_id(plan_id: str) -> str:
    """Validate a canonical probe-plan ID and return its digest."""

    if not isinstance(plan_id, str):
        raise TypeError(f"probe plan id must be a string, got {type(plan_id).__name__}")
    match = _PLAN_ID.fullmatch(plan_id)
    if match is None:
        raise ValueError("probe plan id must use canonical plan:<64 lowercase hex> form")
    return match.group(1)


def _optional_digest(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{field_name} must use the sha256:<64 lowercase hex> form")
    return value


def _bounded_label(value: str, *, field_name: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string, got {type(value).__name__}")
    if not value or len(value) > maximum:
        raise ValueError(f"{field_name} length must be between 1 and {maximum}")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{field_name} must not contain control characters")
    return value


def _sorted_unique(values: tuple[str, ...], *, field_name: str, maximum: int) -> tuple[str, ...]:
    for item in values:
        _bounded_label(item, field_name=field_name, maximum=maximum)
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must be unique")
    return tuple(sorted(values))


class ChatMessage(StrictManifestModel):
    """One chat-template message with an explicit role."""

    role: Literal["system", "user", "assistant"]
    content: StrictStr = Field(max_length=_MAX_TEXT)


class PromptInputSpec(StrictManifestModel):
    """Prompt-lab input: either chat messages or raw text, never both."""

    input_kind: Literal[RunInputKind.PROMPT] = RunInputKind.PROMPT
    messages: tuple[ChatMessage, ...] = ()
    text: StrictStr | None = Field(default=None, max_length=_MAX_TEXT)

    @model_validator(mode="after")
    def _exactly_one_form(self) -> Self:
        if bool(self.messages) == (self.text is not None):
            raise ValueError(
                "prompt input must specify exactly one of chat messages or raw text"
            )
        return self


class DatasetInputSpec(StrictManifestModel):
    """Dataset descriptor identity; reading the data is a later engine step."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_assignment=True)

    input_kind: Literal[RunInputKind.DATASET] = RunInputKind.DATASET
    format: DatasetFormat
    location: StrictStr = Field(max_length=_MAX_LOCATION)
    revision: StrictStr | None = None
    content_digest: StrictStr | None = None
    column_mapping: dict[str, str] = Field(default_factory=dict)
    row_count: StrictInt | None = Field(default=None, ge=0)
    sample_cap: StrictInt | None = Field(default=None, ge=1)
    batch_size: StrictInt | None = Field(default=None, ge=1)
    shuffle: StrictBool = False
    seed: StrictInt | None = Field(default=None, ge=0)
    mode: RunMode = RunMode.GENERATION

    @field_validator("location")
    @classmethod
    def _portable_location(cls, value: str) -> str:
        return _bounded_label(value, field_name="dataset location", maximum=_MAX_LOCATION)

    @field_validator("revision")
    @classmethod
    def _canonical_revision(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return canonical_identifier(value, field_name="dataset revision")

    @field_validator("content_digest")
    @classmethod
    def _digest_shape(cls, value: str | None) -> str | None:
        return _optional_digest(value, field_name="content_digest")

    @field_validator("column_mapping")
    @classmethod
    def _frozen_columns(cls, value: dict[str, str]) -> dict[str, str]:
        for role, column in value.items():
            _bounded_label(role, field_name="column mapping role", maximum=_MAX_LABEL)
            _bounded_label(column, field_name="column mapping column", maximum=_MAX_LABEL)
        return _freeze_mapping(value)


class DataProvenance(StrictManifestModel):
    """Immutable identity of the data a run consumes (PRD §9.3 Data group)."""

    input: PromptInputSpec | DatasetInputSpec
    row_count: StrictInt | None = Field(default=None, ge=0)
    task_labels: tuple[StrictStr, ...] = ()
    preprocessing: dict[str, Any] = Field(default_factory=dict)

    @field_validator("task_labels")
    @classmethod
    def _sorted_labels(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _sorted_unique(value, field_name="task label", maximum=_MAX_LABEL)

    @field_validator("preprocessing")
    @classmethod
    def _frozen_preprocessing(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _freeze_mapping(value)

    @property
    def fingerprint(self) -> str:
        """Content-addressed ``data:<64 hex>`` fingerprint of this provenance."""

        return f"data:{stable_digest(self.model_dump(mode='json'))}"


class GenerationConfig(StrictManifestModel):
    """Deterministic/sampling parameters recorded with a run (PRD §9.3)."""

    max_new_tokens: StrictInt | None = Field(default=None, ge=1, le=1_000_000)
    temperature: StrictFloat | None = Field(default=None, gt=0, le=100)
    top_p: StrictFloat | None = Field(default=None, gt=0, le=1)
    top_k: StrictInt | None = Field(default=None, ge=1)
    seed: StrictInt | None = Field(default=None, ge=0)
    repetition_penalty: StrictFloat | None = Field(default=None, gt=0, le=10)
    stop_sequences: tuple[StrictStr, ...] = ()
    do_sample: StrictBool | None = None

    @field_validator("stop_sequences")
    @classmethod
    def _bounded_stops(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for stop in value:
            _bounded_label(stop, field_name="stop sequence", maximum=_MAX_LABEL)
        if len(set(value)) != len(value):
            raise ValueError("stop sequences must be unique")
        return tuple(sorted(value))


class ModelProvenance(StrictManifestModel):
    """Resolved model/tokenizer identity bound to a loading plan (PRD §9.3)."""

    loading_plan_id: StrictStr
    model_id: StrictStr
    model_revision: StrictStr
    config_hash: StrictStr | None = None
    tokenizer_revision: StrictStr | None = None
    quantization: QuantizationPolicy = QuantizationPolicy.NONE

    @field_validator("loading_plan_id")
    @classmethod
    def _plan_shape(cls, value: str) -> str:
        parse_loading_plan_id(value)
        return value

    @field_validator("model_id")
    @classmethod
    def _canonical_model_id(cls, value: str) -> str:
        return canonical_identifier(value, field_name="model identifier")

    @field_validator("model_revision")
    @classmethod
    def _canonical_model_revision(cls, value: str) -> str:
        return canonical_identifier(value, field_name="model revision")

    @field_validator("config_hash")
    @classmethod
    def _hash_shape(cls, value: str | None) -> str | None:
        return _optional_digest(value, field_name="config_hash")


class ExecutionEnvironment(StrictManifestModel):
    """Observed execution metadata recorded by the engine, excluded from identity."""

    python_version: StrictStr | None = None
    pytorch_version: StrictStr | None = None
    transformers_version: StrictStr | None = None
    moeatlas_version: StrictStr | None = None
    device_map: StrictStr | None = None
    device_metadata: dict[str, Any] = Field(default_factory=dict)
    recorded_at: StrictStr | None = Field(default=None, max_length=_MAX_LABEL)

    @field_validator("device_metadata")
    @classmethod
    def _frozen_device_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _freeze_mapping(value)


class ProbeProvenance(StrictManifestModel):
    """Probe plan binding for a run (PRD §9.3 Probe plan group)."""

    probe_plan_id: StrictStr
    capture_level: StrictInt = Field(ge=0, le=5)
    intervention_opt_in: StrictBool = False

    @field_validator("probe_plan_id")
    @classmethod
    def _plan_shape(cls, value: str) -> str:
        parse_probe_plan_id(value)
        return value


class AdapterProvenance(StrictManifestModel):
    """Adapter/plugin provenance; fields are paired or both absent."""

    adapter: StrictStr | None = None
    adapter_version: StrictStr | None = None
    inspection_fingerprint: StrictStr | None = None

    @model_validator(mode="after")
    def _paired_identity(self) -> Self:
        if (self.adapter is None) != (self.adapter_version is None):
            raise ValueError("adapter and adapter_version must be provided together")
        return self

    @field_validator("inspection_fingerprint")
    @classmethod
    def _hash_shape(cls, value: str | None) -> str | None:
        return _optional_digest(value, field_name="inspection_fingerprint")


class PrivacyPolicy(StrictManifestModel):
    """Privacy choices carried by a run specification (defaults to redaction)."""

    token_text: TokenTextPolicy = TokenTextPolicy.REDACTED
    retain_raw_payloads: StrictBool = False
    allow_export: StrictBool = True


class InterventionLineage(StrictManifestModel):
    """Causal lineage binding a run to its baseline and recipe (PRD §9.3)."""

    baseline_run_key: StrictStr
    recipe_fingerprint: StrictStr
    operation: StrictStr = Field(max_length=_MAX_LABEL)
    targets: tuple[StrictStr, ...] = ()

    @field_validator("baseline_run_key")
    @classmethod
    def _stable_baseline(cls, value: str) -> str:
        return validate_stable_identifier(value, field_name="baseline_run_key")

    @field_validator("recipe_fingerprint")
    @classmethod
    def _hash_shape(cls, value: str) -> str:
        return _optional_digest(value, field_name="recipe_fingerprint")  # type: ignore[return-value]

    @field_validator("operation")
    @classmethod
    def _bounded_operation(cls, value: str) -> str:
        return _bounded_label(value, field_name="intervention operation", maximum=_MAX_LABEL)

    @field_validator("targets")
    @classmethod
    def _sorted_targets(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _sorted_unique(value, field_name="intervention target", maximum=_MAX_LABEL)


class RunSpecification(VersionedManifest):
    """Immutable, content-addressed intent for exactly one run.

    The ``run_key`` is derived from every identity-bearing group below.
    ``workspace``, ``tags``, ``created_at``, ``created_by``, and the observed
    ``execution`` environment are metadata and do not affect the key.
    """

    manifest_type: ClassVar[str] = "run_specification"

    run_key: StrictStr = ""
    workspace: StrictStr | None = None
    tags: tuple[StrictStr, ...] = ()
    created_at: StrictStr | None = Field(default=None, max_length=_MAX_LABEL)
    created_by: StrictStr | None = Field(default=None, max_length=_MAX_LABEL)
    replication: StrictInt = Field(default=0, ge=0)

    model: ModelProvenance
    data: DataProvenance
    generation: GenerationConfig | None = None
    probe: ProbeProvenance | None = None
    adapter: AdapterProvenance | None = None
    execution: ExecutionEnvironment | None = None
    privacy: PrivacyPolicy = Field(default_factory=PrivacyPolicy)
    intervention: InterventionLineage | None = None

    @field_validator("tags")
    @classmethod
    def _sorted_tags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _sorted_unique(value, field_name="tag", maximum=_MAX_LABEL)

    @field_validator("workspace")
    @classmethod
    def _bounded_workspace(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _bounded_label(value, field_name="workspace", maximum=_MAX_LABEL)

    def _identity_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "replication": self.replication,
            "schema": RUN_SPEC_SCHEMA_VERSION,
        }
        for group in ("model", "data", "generation", "probe", "adapter", "privacy", "intervention"):
            value = getattr(self, group)
            payload[group] = value.model_dump(mode="json") if value is not None else None
        return payload

    @property
    def run_digest(self) -> str:
        """Return the digest portion of the canonical run key."""

        return parse_run_key(self.run_key)

    @model_validator(mode="after")
    def _canonical_key_and_lineage(self) -> Self:
        expected = make_run_key(self._identity_payload())
        if "run_key" not in self.model_fields_set:
            object.__setattr__(self, "run_key", expected)
        elif self.run_key != expected:
            raise ValueError(
                f"run_key does not match the immutable run specification content; "
                f"expected {expected!r}"
            )

        if self.intervention is not None:
            if self.probe is None:
                raise ValueError(
                    "intervention lineage requires a probe plan on the specification"
                )
            if not self.probe.intervention_opt_in:
                raise ValueError(
                    "intervention lineage requires probe.intervention_opt_in to be true"
                )
        return self
