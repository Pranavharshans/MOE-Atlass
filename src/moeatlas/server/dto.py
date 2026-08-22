"""Wire DTOs for the local server, layered over shared application services.

DTOs are plain strict Pydantic models with no FastAPI import: the wire
vocabulary stays testable and serializable without the optional server
dependency installed. Every response is derived from shared services —
the server layer owns no orchestration of its own.
"""

from __future__ import annotations

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


__all__ = [
    "AdapterEntryResponse",
    "AdaptersResponse",
    "HealthResponse",
    "RoutingShardEntryResponse",
    "RunDetailResponse",
    "RunEntryResponse",
    "RunSummaryResponse",
    "RunsResponse",
    "WorkspaceResponse",
]
