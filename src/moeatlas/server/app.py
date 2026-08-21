"""Local read-only FastAPI application over shared application services.

The server is a thin wire layer: every endpoint delegates to the same
services the CLI uses (workspace snapshot, bounded run queries, adapter
registry) and owns no orchestration of its own. FastAPI is an optional
dependency imported only inside :func:`create_app`; the wire DTOs in
:mod:`moeatlas.server.dto` stay importable without it.

The app is read-only and local-first: no model loading, no downloads, no
network egress, no storage writes. Failures carry fixed safe details that
never echo input contents.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

SERVER_SCHEMA_VERSION = "1.0"
"""Schema version of the server wire contracts."""

_MAX_RESULTS_CEILING = 10_000


class ServerDependencyError(RuntimeError):
    """The optional server dependency is not installed."""

    def __init__(self) -> None:
        super().__init__("server dependency 'fastapi' is not installed")


def create_app(
    workspace: str | Path,
    *,
    max_results: int = 100,
) -> Any:
    """Build the local FastAPI application bound to one workspace.

    ``max_results`` bounds every run listing and must be a strict positive
    integer within the ceiling. The workspace need not exist yet: endpoints
    report the fixed ``workspace is not initialized`` failure until the
    catalog is initialized.
    """

    if type(max_results) is not int or isinstance(max_results, bool):
        raise TypeError("max_results must be an integer")
    if max_results <= 0 or max_results > _MAX_RESULTS_CEILING:
        raise ValueError(
            f"max_results must be between 1 and {_MAX_RESULTS_CEILING}"
        )
    if not isinstance(workspace, str | Path):
        raise TypeError("workspace must be a string or Path")
    bound_workspace = str(workspace)

    try:
        from fastapi import FastAPI, HTTPException, Query
    except ImportError as exc:
        raise ServerDependencyError() from exc

    from .dto import (
        AdapterEntryResponse,
        AdaptersResponse,
        HealthResponse,
        RunEntryResponse,
        RunsResponse,
        WorkspaceResponse,
    )

    app = FastAPI(
        title="MoEAtlas local server",
        version=SERVER_SCHEMA_VERSION,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    def _not_initialized() -> HTTPException:
        return HTTPException(status_code=404, detail="workspace is not initialized")

    @app.get("/healthz", response_model=HealthResponse)
    def healthz() -> HealthResponse:
        from .. import PRODUCT_NAME, __version__

        return HealthResponse(
            package_name=PRODUCT_NAME,
            package_version=__version__,
            python_version=sys.version.split()[0],
            model_validation_status="deferred",
        )

    @app.get("/api/workspace", response_model=WorkspaceResponse)
    def workspace_snapshot() -> WorkspaceResponse:
        from ..services import open_workspace

        try:
            snapshot = open_workspace(bound_workspace)
        except Exception as exc:
            raise _not_initialized() from exc
        return WorkspaceResponse(
            workspace=str(snapshot.path),
            run_count=len(snapshot.catalog.runs),
        )

    @app.get("/api/runs", response_model=RunsResponse)
    def runs(
        state: str | None = Query(default=None),
        limit: int = Query(default=max_results, ge=1, le=max_results),
    ) -> RunsResponse:
        from ..services import query_runs

        try:
            entries = query_runs(bound_workspace, state=state, max_results=limit)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid run state filter") from exc
        except Exception as exc:
            raise _not_initialized() from exc
        return RunsResponse(
            workspace=bound_workspace,
            count=len(entries),
            entries=tuple(
                RunEntryResponse(
                    run_key=entry.run_key,
                    state=entry.state,
                    attempt=entry.attempt,
                    shard_count=entry.shard_count,
                    token_event_count=entry.token_event_count,
                    routing_event_count=entry.routing_event_count,
                    registered_at=entry.registered_at,
                    updated_at=entry.updated_at,
                )
                for entry in entries
            ),
        )

    @app.get("/api/adapters", response_model=AdaptersResponse)
    def adapters() -> AdaptersResponse:
        from ..adapters import collect_adapter_registry

        report = collect_adapter_registry()
        return AdaptersResponse(
            entries=tuple(
                AdapterEntryResponse(
                    name=entry.record.name,
                    version=entry.record.version,
                    source=entry.record.source,
                    distribution=entry.record.distribution,
                    location=entry.record.location,
                    architecture_families=entry.record.architecture_families,
                    status=entry.status,
                )
                for entry in report.entries
            ),
            collisions=report.collisions,
            failures=report.failures,
        )

    return app


__all__ = ["SERVER_SCHEMA_VERSION", "ServerDependencyError", "create_app"]
