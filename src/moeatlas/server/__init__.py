"""Local read-only server layer over shared application services."""

from __future__ import annotations

from .app import SERVER_SCHEMA_VERSION, ServerDependencyError, create_app
from .dto import (
    AdapterEntryResponse,
    AdaptersResponse,
    HealthResponse,
    RoutingShardEntryResponse,
    RunDetailResponse,
    RunEntryResponse,
    RunsResponse,
    RunSummaryResponse,
    WorkspaceResponse,
)

__all__ = [
    "SERVER_SCHEMA_VERSION",
    "AdapterEntryResponse",
    "AdaptersResponse",
    "HealthResponse",
    "RoutingShardEntryResponse",
    "RunDetailResponse",
    "RunEntryResponse",
    "RunSummaryResponse",
    "RunsResponse",
    "ServerDependencyError",
    "WorkspaceResponse",
    "create_app",
]
