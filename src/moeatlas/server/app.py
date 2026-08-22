"""Local read-only FastAPI application over shared application services.

The server is a thin wire layer: every endpoint delegates to the same
services the CLI uses (workspace snapshot, bounded run queries, adapter
registry) and owns no orchestration of its own. FastAPI is an optional
dependency imported only inside :func:`create_app`; the wire DTOs in
:mod:`moeatlas.server.dto` stay importable without it.

The app is read-only and local-first: it performs no model loading, dataset
downloads, or storage writes. Public Hub metadata is fetched only for an
explicit ``/api/hub/search`` request, with a fixed HTTPS origin and bounded
results. Failures carry fixed safe details that never echo input contents.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

SERVER_SCHEMA_VERSION = "1.0"
"""Schema version of the server wire contracts."""

_MAX_RESULTS_CEILING = 10_000
_MAX_ARTIFACT_BYTES_CEILING = 100_000_000
_DEFAULT_ARTIFACT_BYTES = 10_000_000

_HEATMAP_DIRECTORY = "heatmaps"
_INSPECTION_DIRECTORY = "inspections"

_STATIC_DIRECTORY = Path(__file__).resolve().parent / "static"


class ServerDependencyError(RuntimeError):
    """The optional server dependency is not installed."""

    def __init__(self) -> None:
        super().__init__("server dependency 'fastapi' is not installed")


def create_app(
    workspace: str | Path,
    *,
    max_results: int = 100,
    max_artifact_bytes: int = _DEFAULT_ARTIFACT_BYTES,
) -> Any:
    """Build the local FastAPI application bound to one workspace.

    ``max_results`` bounds every run listing and must be a strict positive
    integer within the ceiling. ``max_artifact_bytes`` bounds every served
    artifact read the same way. The workspace need not exist yet: endpoints
    report the fixed ``workspace is not initialized`` failure until the
    catalog is initialized.
    """

    if type(max_results) is not int or isinstance(max_results, bool):
        raise TypeError("max_results must be an integer")
    if max_results <= 0 or max_results > _MAX_RESULTS_CEILING:
        raise ValueError(
            f"max_results must be between 1 and {_MAX_RESULTS_CEILING}"
        )
    if type(max_artifact_bytes) is not int or isinstance(max_artifact_bytes, bool):
        raise TypeError("max_artifact_bytes must be an integer")
    if max_artifact_bytes <= 0 or max_artifact_bytes > _MAX_ARTIFACT_BYTES_CEILING:
        raise ValueError(
            f"max_artifact_bytes must be between 1 and {_MAX_ARTIFACT_BYTES_CEILING}"
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
        HubSearchEntryResponse,
        HubSearchResponse,
        RoutingShardEntryResponse,
        RunDetailResponse,
        RunEntryResponse,
        RunsResponse,
        RunSummaryResponse,
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

    def _unknown_run() -> HTTPException:
        return HTTPException(status_code=404, detail="run is not registered")

    def _validated_run_key(run_key: object) -> str:
        from ..core.identity import validate_stable_identifier

        try:
            if type(run_key) is not str:
                raise TypeError("run_key must be an exact string")
            return validate_stable_identifier(run_key, field_name="run_key")
        except Exception as exc:
            raise _unknown_run() from exc

    def _catalog_entry(run_key: str) -> tuple[Path, Any]:
        from ..services import open_workspace

        try:
            snapshot = open_workspace(bound_workspace)
        except Exception as exc:
            raise _not_initialized() from exc
        entry = next(
            (e for e in snapshot.catalog.runs if e.run_key == run_key), None
        )
        if entry is None:
            raise _unknown_run()
        return snapshot.path, entry

    def _safe_heatmap_document(workspace_root: Path, run_key: str) -> Path | None:
        """Resolve one published heatmap document without following symlinks.

        The managed ``heatmaps`` directory and the candidate document must be
        real non-symlink entries whose canonical location stays inside the
        workspace; anything else reads as absent so traversal and symlink
        attacks never widen the served surface.
        """

        root = workspace_root / _HEATMAP_DIRECTORY
        candidate = root / f"{run_key}.html"
        try:
            if root.is_symlink() or not root.is_dir():
                return None
            if candidate.is_symlink() or not candidate.is_file():
                return None
            resolved_root = root.resolve()
            resolved_candidate = candidate.resolve()
        except OSError:
            return None
        if resolved_candidate.parent != resolved_root:
            return None
        return candidate

    def _safe_inspection_document(workspace_root: Path, run_key: str) -> Path | None:
        """Resolve one persisted topology document without following symlinks."""

        root = workspace_root / _INSPECTION_DIRECTORY
        candidate = root / f"{run_key}.json"
        try:
            if root.is_symlink() or not root.is_dir():
                return None
            if candidate.is_symlink() or not candidate.is_file():
                return None
            resolved_root = root.resolve()
            resolved_candidate = candidate.resolve()
        except OSError:
            return None
        if resolved_candidate.parent != resolved_root:
            return None
        return candidate

    @app.get("/healthz", response_model=HealthResponse)
    def healthz() -> HealthResponse:
        from .. import PRODUCT_NAME, __version__

        return HealthResponse(
            package_name=PRODUCT_NAME,
            package_version=__version__,
            python_version=sys.version.split()[0],
            model_validation_status="deferred",
        )

    @app.get("/api/hub/search", response_model=HubSearchResponse)
    def hub_search(
        kind: str = Query(default="model"),
        q: str = Query(default=""),
        limit: int = Query(default=6, ge=1, le=10),
    ) -> HubSearchResponse:
        """Return bounded public suggestions after an explicit UI query."""

        from ..services.hub import HubSearchError, search_hub

        try:
            entries = search_hub(kind, q, limit=limit)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400, detail="invalid Hugging Face search request"
            ) from exc
        except HubSearchError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        normalized_kind = kind if kind in {"model", "dataset"} else "model"
        return HubSearchResponse(
            schema_version="1.0",
            kind=normalized_kind,
            query=q.strip(),
            count=len(entries),
            entries=tuple(
                HubSearchEntryResponse(
                    identifier=entry.identifier,
                    kind=entry.kind,
                    author=entry.author,
                    downloads=entry.downloads,
                    likes=entry.likes,
                    pipeline_tag=entry.pipeline_tag,
                    library_name=entry.library_name,
                    tags=entry.tags,
                    last_modified=entry.last_modified,
                )
                for entry in entries
            ),
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

    @app.get("/api/runs/{run_key}", response_model=RunDetailResponse)
    def run_detail(run_key: str) -> RunDetailResponse:
        from ..store.ports import reader_from_workspace

        stable_run_key = _validated_run_key(run_key)
        _, entry = _catalog_entry(stable_run_key)
        try:
            receipts = reader_from_workspace(bound_workspace).list_shards(
                run_key=stable_run_key
            )
        except Exception as exc:
            raise HTTPException(
                status_code=404, detail="run shards are unavailable"
            ) from exc
        return RunDetailResponse(
            run_key=entry.run_key,
            state=entry.state,
            attempt=entry.attempt,
            specification_fingerprint=entry.specification_fingerprint,
            token_text_policy=entry.token_text_policy,
            registered_at=entry.registered_at,
            updated_at=entry.updated_at,
            shards=tuple(
                RoutingShardEntryResponse(
                    shard_key=receipt.shard_key,
                    relative_path=receipt.relative_path,
                    token_count=receipt.token_count,
                    routing_count=receipt.routing_count,
                    token_text_stored=receipt.token_text_stored,
                )
                for receipt in receipts
            ),
        )

    @app.get("/api/runs/{run_key}/summary", response_model=RunSummaryResponse)
    def run_summary(run_key: str) -> RunSummaryResponse:
        stable_run_key = _validated_run_key(run_key)
        workspace_path, _ = _catalog_entry(stable_run_key)
        inspection_path = _safe_inspection_document(workspace_path, stable_run_key)
        if inspection_path is None:
            return RunSummaryResponse(
                run_key=stable_run_key,
                status="unavailable",
                reason="published routing inspection is unavailable",
            )
        try:
            from ..adapters import AdapterInspection, UniversalRoutingInspection
            from ..analysis import aggregate_routing_load

            document = inspection_path.read_bytes()
            if len(document) > _DEFAULT_ARTIFACT_BYTES:
                raise ValueError("inspection exceeds the serving byte budget")
            try:
                inspection = UniversalRoutingInspection.model_validate_json(document)
            except Exception:
                inspection = AdapterInspection.model_validate_json(document)
            matrix = aggregate_routing_load(
                workspace_path,
                inspection,
                run_key=stable_run_key,
                max_routing_rows=1_000_000,
                max_source_bytes=1_000_000_000,
                max_matrix_cells=100_000,
            )
        except Exception:
            return RunSummaryResponse(
                run_key=stable_run_key,
                status="unavailable",
                reason="published routing inspection could not be analyzed",
            )
        return RunSummaryResponse(
            run_key=stable_run_key,
            status="available",
            adapter_name=matrix.adapter_name,
            adapter_version=matrix.adapter_version,
            token_count=matrix.token_count,
            assignment_count=matrix.assignment_count,
            layer_count=len(matrix.layer_keys),
            expert_count=len(matrix.expert_keys[0]),
            routed_top_k=matrix.routed_top_k,
            inspection_digest=matrix.inspection_digest,
        )

    @app.get("/api/runs/{run_key}/heatmap")
    def run_heatmap(run_key: str) -> Any:
        from fastapi import Response

        stable_run_key = _validated_run_key(run_key)
        workspace_path, _ = _catalog_entry(stable_run_key)
        candidate = _safe_heatmap_document(workspace_path, stable_run_key)
        if candidate is None:
            inspection_path = _safe_inspection_document(workspace_path, stable_run_key)
            if inspection_path is None:
                raise HTTPException(status_code=404, detail="run heatmap is not published")
            try:
                from ..adapters import AdapterInspection, UniversalRoutingInspection
                from ..analysis import aggregate_routing_load, render_routing_load_heatmap

                document = inspection_path.read_bytes()
                if len(document) > _DEFAULT_ARTIFACT_BYTES:
                    raise ValueError("inspection exceeds the serving byte budget")
                try:
                    inspection = UniversalRoutingInspection.model_validate_json(document)
                except Exception:
                    inspection = AdapterInspection.model_validate_json(document)
                matrix = aggregate_routing_load(
                    workspace_path,
                    inspection,
                    run_key=stable_run_key,
                    max_routing_rows=1_000_000,
                    max_source_bytes=1_000_000_000,
                    max_matrix_cells=100_000,
                )
                payload = render_routing_load_heatmap(
                    matrix, metric="assignment_counts", max_cells=100_000
                ).encode("utf-8")
                if len(payload) > max_artifact_bytes:
                    raise ValueError("run heatmap exceeds the serving byte budget")
                return Response(content=payload, media_type="text/html; charset=utf-8")
            except Exception as exc:
                raise HTTPException(
                    status_code=404, detail="run heatmap is not published"
                ) from exc
        try:
            size = candidate.stat().st_size
            if size > max_artifact_bytes:
                raise HTTPException(
                    status_code=404,
                    detail="run heatmap exceeds the serving byte budget",
                )
            with candidate.open("rb") as stream:
                payload = stream.read(max_artifact_bytes + 1)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=404, detail="run heatmap is not published"
            ) from exc
        if len(payload) > max_artifact_bytes:
            raise HTTPException(
                status_code=404,
                detail="run heatmap exceeds the serving byte budget",
            )
        return Response(content=payload, media_type="text/html; charset=utf-8")

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

    # Static frontend mount, registered strictly after every API route so
    # /healthz and /api/* always win. The packaged assets are dependency-free
    # vanilla HTML/CSS/JS; cache headers are explicitly disabled so local
    # development always observes freshly served bytes.
    if _STATIC_DIRECTORY.is_dir():
        from fastapi.staticfiles import StaticFiles

        @app.middleware("http")
        async def _disable_static_caching(request: Any, call_next: Any) -> Any:
            response = await call_next(request)
            if not (
                request.url.path.startswith("/api/") or request.url.path == "/healthz"
            ):
                response.headers["Cache-Control"] = "no-store"
            return response

        app.mount(
            "/", StaticFiles(directory=_STATIC_DIRECTORY, html=True), name="static"
        )

    return app


__all__ = ["SERVER_SCHEMA_VERSION", "ServerDependencyError", "create_app"]
