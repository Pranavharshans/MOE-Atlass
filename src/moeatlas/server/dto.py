"""Wire DTOs for the local server, layered over shared application services.

DTOs are plain strict Pydantic models with no FastAPI import: the wire
vocabulary stays testable and serializable without the optional server
dependency installed. Every response is derived from shared services —
the server layer owns no orchestration of its own.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class HealthResponse(_WireModel):
    package_name: str
    package_version: str
    python_version: str
    model_validation_status: str


class WorkspaceResponse(_WireModel):
    workspace: str
    run_count: int


class RunEntryResponse(_WireModel):
    run_key: str
    run_name: str | None = None
    state: str | None = None
    attempt: int = 1
    shard_count: int = 0
    token_event_count: int = 0
    routing_event_count: int = 0
    registered_at: str | None = None
    updated_at: str | None = None


class RunsResponse(_WireModel):
    workspace: str
    count: int
    entries: tuple[RunEntryResponse, ...] = Field(default_factory=tuple)


class RoutingShardEntryResponse(_WireModel):
    shard_key: str
    relative_path: str
    token_count: int
    routing_count: int
    token_text_stored: bool


class RunDetailResponse(_WireModel):
    run_key: str
    run_name: str | None = None
    state: str | None = None
    attempt: int = 1
    specification_fingerprint: str | None = None
    token_text_policy: str | None = None
    registered_at: str | None = None
    updated_at: str | None = None
    shards: tuple[RoutingShardEntryResponse, ...] = Field(default_factory=tuple)


class RunSummaryResponse(_WireModel):
    """Typed routing-load summary response for universal or certified runs."""

    run_key: str
    status: str
    reason: str | None = None
    adapter_name: str | None = None
    adapter_version: str | None = None
    token_count: int | None = None
    assignment_count: int | None = None
    layer_count: int | None = None
    expert_count: int | None = None
    routed_top_k: int | None = None
    inspection_digest: str | None = None


class AdapterEntryResponse(_WireModel):
    name: str
    version: str
    source: str
    distribution: str | None = None
    location: str
    architecture_families: tuple[str, ...] = ()
    status: str


class AdaptersResponse(_WireModel):
    entries: tuple[AdapterEntryResponse, ...] = Field(default_factory=tuple)
    collisions: tuple[tuple[str, str, str], ...] = Field(default_factory=tuple)
    failures: tuple[tuple[str, str], ...] = Field(default_factory=tuple)


class HubSearchEntryResponse(_WireModel):
    """Bounded public metadata for one model or dataset suggestion."""

    identifier: str
    kind: Literal["model", "dataset"]
    author: str | None = None
    downloads: int | None = None
    likes: int | None = None
    pipeline_tag: str | None = None
    library_name: str | None = None
    tags: tuple[str, ...] = Field(default_factory=tuple)
    last_modified: str | None = None


class HubSearchResponse(_WireModel):
    """Response for an explicit public Hugging Face search request."""

    schema_version: str
    kind: Literal["model", "dataset"]
    query: str
    count: int
    entries: tuple[HubSearchEntryResponse, ...] = Field(default_factory=tuple)


class DiscoveryRequest(_WireModel):
    """User intent for one live model discovery job."""

    model_id: str = Field(min_length=3, max_length=500)
    model_revision: str = Field(default="main", min_length=1, max_length=200)
    device: str = Field(default="auto", min_length=1, max_length=32)
    dtype: Literal["preserve", "float32", "float16", "bfloat16"] = "preserve"
    trust_remote_code: bool = False
    allow_downloads: bool = True


class RunStartRequest(_WireModel):
    """Bounded live run intent accepted by the local control plane."""

    run_name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
    model_id: str = Field(min_length=3, max_length=500)
    model_revision: str = Field(default="main", min_length=1, max_length=200)
    dataset_id: str = Field(min_length=3, max_length=500)
    dataset_revision: str = Field(default="main", min_length=1, max_length=200)
    dataset_config: str | None = Field(default=None, max_length=200)
    dataset_split: str = Field(default="train", min_length=1, max_length=200)
    prompt_column: str = Field(default="prompt", min_length=1, max_length=200)
    reference_column: str | None = Field(default=None, min_length=1, max_length=200)
    prompt_format: Literal["raw", "mmlu_multiple_choice"] = "raw"
    choices_column: str | None = Field(default=None, min_length=1, max_length=200)
    evaluation_method: Literal[
        "normalized_exact_match",
        "token_f1",
        "contains_reference",
        "multiple_choice_accuracy",
        "numeric_match",
    ] = "normalized_exact_match"
    sample_cap: int = Field(default=32, ge=1, le=10_000)
    dataset_seed: int | None = Field(default=None, ge=0)
    replication: int = Field(default=0, ge=0, le=1_000_000)
    batch_size: int = Field(default=1, ge=1, le=256)
    max_new_tokens: int = Field(default=128, ge=1, le=1_000_000)
    thinking_mode: Literal["model_default", "disabled", "enabled"] = "model_default"
    token_text_policy: Literal["redacted", "stored"] = "redacted"
    allow_export: bool = True
    retain_raw_payloads: bool = False
    mode: Literal["generation", "teacher_forced"] = "generation"
    device: str = Field(default="auto", min_length=1, max_length=32)
    dtype: Literal["preserve", "float32", "float16", "bfloat16"] = "preserve"
    trust_remote_code: bool = False
    allow_downloads: bool = True
    capture_expert_activity: bool = True
    measure_capture_overhead: bool = False
    resume_job_id: str | None = Field(default=None, max_length=100)


class DatasetRunRequest(_WireModel):
    """One dataset/config child in a parent run group."""

    dataset_id: str = Field(min_length=3, max_length=500)
    dataset_revision: str = Field(default="main", min_length=1, max_length=200)
    dataset_config: str | None = Field(default=None, max_length=200)
    dataset_split: str = Field(default="train", min_length=1, max_length=200)
    prompt_column: str = Field(default="prompt", min_length=1, max_length=200)
    reference_column: str | None = Field(default=None, min_length=1, max_length=200)
    prompt_format: Literal["raw", "mmlu_multiple_choice"] = "raw"
    choices_column: str | None = Field(default=None, min_length=1, max_length=200)


class RunGroupStartRequest(_WireModel):
    """One model/settings contract expanded across multiple datasets."""

    run_name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,49}$")
    model_id: str = Field(min_length=3, max_length=500)
    model_revision: str = Field(default="main", min_length=1, max_length=200)
    datasets: tuple[DatasetRunRequest, ...] = Field(min_length=2, max_length=16)
    evaluation_method: Literal[
        "normalized_exact_match",
        "token_f1",
        "contains_reference",
        "multiple_choice_accuracy",
        "numeric_match",
    ] = "normalized_exact_match"
    sample_cap: int = Field(default=32, ge=1, le=10_000)
    dataset_seed: int | None = Field(default=None, ge=0)
    replication: int = Field(default=0, ge=0, le=1_000_000)
    batch_size: int = Field(default=1, ge=1, le=256)
    max_new_tokens: int = Field(default=128, ge=1, le=1_000_000)
    thinking_mode: Literal["model_default", "disabled", "enabled"] = "model_default"
    token_text_policy: Literal["redacted", "stored"] = "redacted"
    allow_export: bool = True
    retain_raw_payloads: bool = False
    mode: Literal["generation", "teacher_forced"] = "generation"
    device: str = Field(default="auto", min_length=1, max_length=32)
    dtype: Literal["preserve", "float32", "float16", "bfloat16"] = "preserve"
    trust_remote_code: bool = False
    allow_downloads: bool = True
    capture_expert_activity: bool = True
    measure_capture_overhead: bool = False


class RunGroupsResponse(_WireModel):
    groups: tuple[dict[str, Any], ...] = Field(default_factory=tuple)


class JobProgressResponse(_WireModel):
    stage: str
    completed: int = 0
    total: int | None = None
    message: str = ""


class JobDiagnosticsReference(_WireModel):
    """Safe pointer to bounded diagnostics for one known server job."""

    endpoint: str
    available: bool = False
    entry_count: int = 0
    truncated: bool = False


class JobDiagnosticEntryResponse(_WireModel):
    """One sanitized structured job diagnostic record."""

    schema_version: str = "1.0"
    sequence: int
    at: str
    event: str
    kind: str | None = None
    stage: str | None = None
    completed: int | None = None
    total: int | None = None
    message: str | None = None
    exception_type: str | None = None
    exception_message: str | None = None
    traceback: str | None = None


class JobDiagnosticsResponse(_WireModel):
    """Bounded diagnostics read through the existing server control plane."""

    job_id: str
    kind: str
    state: Literal["queued", "running", "completed", "cancelled", "failed"]
    available: bool = False
    entry_count: int = 0
    truncated: bool = False
    entries: tuple[JobDiagnosticEntryResponse, ...] = Field(default_factory=tuple)


class JobResponse(_WireModel):
    job_id: str
    kind: str
    state: Literal["queued", "running", "completed", "cancelled", "failed"]
    progress: JobProgressResponse
    result: dict[str, Any] | None = None
    error: str | None = None
    diagnostics: JobDiagnosticsReference | None = None


class JobCreatedResponse(_WireModel):
    job_id: str
    kind: str
    state: Literal["queued", "running", "completed", "cancelled", "failed"]


class ArchitectureResponse(_WireModel):
    run_key: str
    status: Literal["available", "unavailable"]
    reason: str | None = None
    report: dict[str, Any] | None = None


class ActivityResponse(_WireModel):
    run_key: str
    status: Literal["available", "unavailable"]
    reason: str | None = None
    summary: dict[str, Any] | None = None


class RoutingSimilarityResponse(_WireModel):
    baseline_run_key: str
    comparison_run_key: str
    status: Literal["available", "unavailable"]
    reason: str | None = None
    report: dict[str, Any] | None = None


class InterventionRecipeRequest(_WireModel):
    operation: Literal["ablate", "scale", "reroute", "alter_router"]
    targets: tuple[str, ...] = Field(min_length=1, max_length=1024)
    factor: float | None = None
    bias: float | None = None
    alternates: tuple[tuple[str, str], ...] = ()


class InterventionRecipeResponse(_WireModel):
    status: Literal["prepared", "unsupported"]
    recipe: dict[str, Any]
    fingerprint: str
    reason: str | None = None


class InterventionStartRequest(_WireModel):
    """One real baseline-derived expert intervention request."""

    run_name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
    baseline_run_key: str = Field(min_length=5, max_length=80)
    operation: Literal["ablate", "scale"] = "ablate"
    targets: tuple[str, ...] = Field(min_length=1, max_length=64)
    factor: float | None = None


class InterventionTargetsResponse(_WireModel):
    """Discovered independently hookable experts for one baseline run."""

    run_key: str
    status: Literal["available", "unsupported"]
    reason: str | None = None
    targets: tuple[dict[str, Any], ...] = Field(default_factory=tuple)
    capability: dict[str, Any] | None = None


class InterventionEvidenceResponse(_WireModel):
    """Paired baseline/intervention output and performance evidence."""

    run_key: str
    status: Literal["available", "unavailable"]
    reason: str | None = None
    evidence: dict[str, Any] | None = None


class InterventionStudyRequest(_WireModel):
    """Persisted intervention runs to reduce as replications and controls."""

    intervention_run_keys: tuple[str, ...] = Field(min_length=2, max_length=100)
    control_run_keys: tuple[str, ...] = Field(default=(), max_length=100)


class InterventionStudyResponse(_WireModel):
    study_id: str
    study: dict[str, Any]


__all__ = [
    "AdapterEntryResponse",
    "AdaptersResponse",
    "ActivityResponse",
    "ArchitectureResponse",
    "DiscoveryRequest",
    "HealthResponse",
    "HubSearchEntryResponse",
    "HubSearchResponse",
    "InterventionRecipeRequest",
    "InterventionRecipeResponse",
    "InterventionStartRequest",
    "InterventionTargetsResponse",
    "InterventionEvidenceResponse",
    "InterventionStudyRequest",
    "InterventionStudyResponse",
    "JobCreatedResponse",
    "JobDiagnosticEntryResponse",
    "JobDiagnosticsReference",
    "JobDiagnosticsResponse",
    "JobProgressResponse",
    "JobResponse",
    "RoutingShardEntryResponse",
    "RoutingSimilarityResponse",
    "RunDetailResponse",
    "RunGroupStartRequest",
    "RunGroupsResponse",
    "RunEntryResponse",
    "RunSummaryResponse",
    "RunStartRequest",
    "DatasetRunRequest",
    "RunsResponse",
    "WorkspaceResponse",
]
