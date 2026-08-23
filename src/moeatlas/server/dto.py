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

    model_id: str = Field(min_length=3, max_length=500)
    model_revision: str = Field(default="main", min_length=1, max_length=200)
    dataset_id: str = Field(min_length=3, max_length=500)
    dataset_revision: str = Field(default="main", min_length=1, max_length=200)
    dataset_config: str | None = Field(default=None, max_length=200)
    dataset_split: str = Field(default="train", min_length=1, max_length=200)
    prompt_column: str = Field(default="prompt", min_length=1, max_length=200)
    sample_cap: int = Field(default=32, ge=1, le=10_000)
    batch_size: int = Field(default=1, ge=1, le=256)
    max_new_tokens: int = Field(default=128, ge=1, le=1_000_000)
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


class JobProgressResponse(_WireModel):
    stage: str
    completed: int = 0
    total: int | None = None
    message: str = ""


class JobResponse(_WireModel):
    job_id: str
    kind: str
    state: Literal["queued", "running", "completed", "cancelled", "failed"]
    progress: JobProgressResponse
    result: dict[str, Any] | None = None
    error: str | None = None


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
    "JobCreatedResponse",
    "JobProgressResponse",
    "JobResponse",
    "RoutingShardEntryResponse",
    "RunDetailResponse",
    "RunEntryResponse",
    "RunSummaryResponse",
    "RunStartRequest",
    "RunsResponse",
    "WorkspaceResponse",
]
