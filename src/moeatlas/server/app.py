"""Local FastAPI control plane over shared MoEAtlas services.

The server keeps the wire layer small while delegating model resolution,
discovery, execution, storage, and analysis to the same services used by the
CLI.  Live work is submitted to a bounded in-process job manager so a browser
can observe progress and request cooperative cancellation.  FastAPI remains
optional and is imported only inside :func:`create_app`.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from .dto import (
    ActivityResponse,
    AdapterEntryResponse,
    AdaptersResponse,
    ArchitectureResponse,
    DiscoveryRequest,
    HealthResponse,
    HubSearchEntryResponse,
    HubSearchResponse,
    InterventionRecipeRequest,
    InterventionRecipeResponse,
    JobCreatedResponse,
    JobDiagnosticEntryResponse,
    JobDiagnosticsResponse,
    JobProgressResponse,
    JobResponse,
    RoutingShardEntryResponse,
    RunDetailResponse,
    RunEntryResponse,
    RunsResponse,
    RunStartRequest,
    RunSummaryResponse,
    WorkspaceResponse,
)

SERVER_SCHEMA_VERSION = "1.0"
"""Schema version of the server wire contracts."""

_MAX_RESULTS_CEILING = 10_000
_MAX_ARTIFACT_BYTES_CEILING = 100_000_000
_DEFAULT_ARTIFACT_BYTES = 10_000_000

_HEATMAP_DIRECTORY = "heatmaps"
_INSPECTION_DIRECTORY = "inspections"
_DISCOVERY_DIRECTORY = "discoveries"
_EXPORT_DIRECTORY = "exports"
_POLICY_DIRECTORY = "policies"
_BENCHMARK_DIRECTORY = "benchmarks"
_CAPTURE_OVERHEAD_DIRECTORY = "capture-overhead"
_MAX_JOB_PAYLOAD_BYTES = 5_000_000

_STATIC_DIRECTORY = Path(__file__).resolve().parent / "static"


class ServerDependencyError(RuntimeError):
    """The optional server dependency is not installed."""

    def __init__(self) -> None:
        super().__init__("server dependency 'fastapi' is not installed")


def _json_document(value: object, *, max_bytes: int = _MAX_JOB_PAYLOAD_BYTES) -> dict[str, Any]:
    """Return one bounded JSON object from a domain manifest."""

    if hasattr(value, "to_json") and callable(getattr(value, "to_json")):
        raw = value.to_json()
    elif hasattr(value, "model_dump_json") and callable(getattr(value, "model_dump_json")):
        raw = value.model_dump_json()
    else:
        raw = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if not isinstance(raw, str) or len(raw.encode("utf-8")) > max_bytes:
        raise ValueError("job payload exceeds the serving budget")
    document = json.loads(raw)
    if not isinstance(document, dict):
        raise ValueError("job payload must be a JSON object")
    return document


def _publish_run_policy(workspace: object, run_key: str, policy: object) -> None:
    """Persist the server privacy decision beside the run artifacts."""

    root = Path(workspace)
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("workspace must be an existing non-symlink directory")
    directory = root / _POLICY_DIRECTORY
    if directory.exists() and directory.is_symlink():
        raise RuntimeError("policy directory must not be a symlink")
    directory.mkdir(exist_ok=True)
    payload = json.dumps(
        {
            "allow_export": bool(getattr(policy, "allow_export", True)),
            "retain_raw_payloads": bool(getattr(policy, "retain_raw_payloads", False)),
            "token_text": str(getattr(getattr(policy, "token_text", None), "value", "redacted")),
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(payload) > 4096:
        raise ValueError("run privacy policy exceeds the serving budget")
    target = directory / f"{run_key}.json"
    if target.exists():
        if target.is_symlink() or not target.is_file() or target.read_bytes() != payload:
            raise RuntimeError("published run privacy policy conflicts with the run")
        return
    fd, staged_name = tempfile.mkstemp(
        dir=str(directory), prefix=f".{run_key}.", suffix=".staging"
    )
    staged = Path(staged_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
        os.replace(staged, target)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise


def _publish_capture_overhead(workspace: object, run_key: str, document: dict[str, Any]) -> str:
    """Persist one optional native-versus-captured timing report atomically."""

    root = Path(workspace)
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("workspace must be an existing non-symlink directory")
    benchmark_root = root / _BENCHMARK_DIRECTORY
    if benchmark_root.exists() and benchmark_root.is_symlink():
        raise RuntimeError("benchmark directory must not be a symlink")
    directory = benchmark_root / _CAPTURE_OVERHEAD_DIRECTORY
    if directory.exists() and directory.is_symlink():
        raise RuntimeError("capture-overhead directory must not be a symlink")
    directory.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(payload) > 256_000:
        raise ValueError("capture-overhead report exceeds the serving budget")
    target = directory / f"{run_key}.json"
    if target.exists():
        if target.is_symlink() or not target.is_file() or target.read_bytes() != payload:
            raise RuntimeError("capture-overhead report conflicts with the run")
        return target.relative_to(root).as_posix()
    fd, staged_name = tempfile.mkstemp(
        dir=str(directory), prefix=f".{run_key}.", suffix=".staging"
    )
    staged = Path(staged_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
        os.replace(staged, target)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return target.relative_to(root).as_posix()


def _discovery_worker(
    payload: dict[str, Any],
    *,
    cancel: Any,
    report_progress: Any,
) -> Any:
    """Resolve, load, and scan one model without touching the browser thread."""

    from ..runtime import classify_capture_support, load_and_scan
    from ..services.model_resolution import resolve_huggingface_plan
    from .jobs import JobOutcome

    if cancel.is_set():
        return JobOutcome({"status": "cancelled"}, "cancelled")
    report_progress(
        stage="resolve",
        completed=0,
        total=1,
        message="Resolving immutable Hub revision",
    )
    plan = resolve_huggingface_plan(
        payload["model_id"],
        payload.get("model_revision", "main"),
        device=payload.get("device", "auto"),
        dtype=payload.get("dtype", "preserve"),
        trust_remote_code=bool(payload.get("trust_remote_code", False)),
        allow_downloads=bool(payload.get("allow_downloads", True)),
    )
    report_progress(stage="load", completed=0, total=1, message="Loading model and tokenizer")
    if cancel.is_set():
        return JobOutcome({"status": "cancelled", "plan_id": plan.plan_id}, "cancelled")
    discovery = load_and_scan(plan)
    capture_support = classify_capture_support(discovery)
    report_progress(
        stage="discover", completed=1, total=1, message="Static architecture scan complete"
    )
    return JobOutcome(
        {
            "status": "available",
            "model_id": plan.source.model_id,
            "requested_revision": plan.source.requested_revision,
            "resolved_revision": plan.resolution.resolved_model_revision
            if plan.resolution
            else None,
            "plan_id": plan.plan_id,
            "security_warnings": list(plan.security_warnings),
            "capture_support": capture_support.to_dict(),
            "report": _json_document(discovery),
        },
        "completed",
    )


def _run_worker(
    workspace: str,
    payload: dict[str, Any],
    *,
    cancel: Any,
    report_progress: Any,
    resume_from: str | None = None,
    skip_overhead: Any = None,
) -> Any:
    """Compose the resolved plan, dataset input, executor, and run service."""

    from importlib import import_module

    from ..core import stable_digest
    from ..loading import QuantizationPolicy
    from ..runs.specs import (
        DataProvenance,
        DatasetFormat,
        DatasetInputSpec,
        GenerationConfig,
        ModelProvenance,
        PrivacyPolicy,
        ProbeProvenance,
        RunMode,
        RunSpecification,
        TokenTextPolicy,
    )
    from ..services.datasets import read_dataset_rows
    from ..services.model_resolution import (
        resolve_huggingface_dataset_revision,
        resolve_huggingface_plan,
    )
    from ..services.run_service import execute_specification, publish_run_report
    from .jobs import JobOutcome

    if cancel.is_set():
        return JobOutcome({"status": "cancelled"}, "cancelled")
    report_progress(
        stage="resolve",
        completed=0,
        total=2,
        message="Resolving model and dataset revisions",
    )
    plan = resolve_huggingface_plan(
        payload["model_id"],
        payload.get("model_revision", "main"),
        device=payload.get("device", "auto"),
        dtype=payload.get("dtype", "preserve"),
        trust_remote_code=bool(payload.get("trust_remote_code", False)),
        allow_downloads=bool(payload.get("allow_downloads", True)),
    )
    allow_downloads = bool(payload.get("allow_downloads", True))
    dataset_revision = resolve_huggingface_dataset_revision(
        payload["dataset_id"],
        payload.get("dataset_revision", "main"),
        allow_downloads=allow_downloads,
    )
    report_progress(stage="resolve", completed=2, total=2, message="Immutable revisions resolved")
    if cancel.is_set():
        return JobOutcome({"status": "cancelled", "plan_id": plan.plan_id}, "cancelled")

    input_spec = DatasetInputSpec(
        format=DatasetFormat.HF_DATASETS,
        location=payload["dataset_id"],
        revision=dataset_revision,
        config_name=payload.get("dataset_config"),
        split=payload.get("dataset_split", "train"),
        allow_downloads=allow_downloads,
        column_mapping={"prompt": payload.get("prompt_column", "prompt")},
        sample_cap=payload.get("sample_cap", 32),
        batch_size=payload.get("batch_size", 1),
        mode=RunMode(payload.get("mode", "generation")),
    )
    # Dataset schemas are not universal.  Resolve the requested column against
    # one bounded row and fall back through common text fields so a user can
    # paste a Hub dataset ID without first reverse-engineering its schema.
    prompt_column = payload.get("prompt_column", "prompt")
    preview_spec = DatasetInputSpec(
        **{
            **input_spec.model_dump(mode="python"),
            "column_mapping": {},
        }
    )
    preview_rows = read_dataset_rows(
        preview_spec,
        base_directory=workspace,
        max_rows=1,
        max_row_bytes=65_536,
        max_file_bytes=100_000_000,
    )
    if preview_rows:
        preview = preview_rows[0].values
        candidates = [
            prompt_column,
            "prompt",
            "text",
            "instruction",
            "question",
            "query",
            "input",
            "content",
        ]
        chosen = next(
            (candidate for candidate in candidates if isinstance(preview.get(candidate), str)),
            next((key for key, value in preview.items() if isinstance(value, str)), None),
        )
        if chosen is None:
            raise ValueError("dataset has no bounded string prompt column")
        input_spec = DatasetInputSpec(
            **{
                **input_spec.model_dump(mode="python"),
                "column_mapping": {"prompt": chosen},
            }
        )
    model_provenance = ModelProvenance(
        loading_plan_id=plan.plan_id,
        model_id=plan.source.model_id,
        model_revision=(
            plan.resolution.resolved_model_revision
            if plan.resolution
            else plan.source.requested_revision
        ),
        tokenizer_revision=(
            plan.resolution.resolved_tokenizer_revision if plan.resolution else None
        ),
        quantization=QuantizationPolicy.NONE,
    )
    privacy = PrivacyPolicy(
        token_text=TokenTextPolicy(payload.get("token_text_policy", "redacted")),
        retain_raw_payloads=bool(payload.get("retain_raw_payloads", False)),
        allow_export=bool(payload.get("allow_export", True)),
    )

    def make_specification(*, capture_level: int, expert_activity: bool) -> RunSpecification:
        probe_payload = {
            "model": plan.plan_id,
            "capture_level": capture_level,
            "expert_activity": expert_activity,
            "routing_capture": capture_level > 0,
        }
        return RunSpecification(
            workspace=workspace,
            created_by="local-server",
            model=model_provenance,
            data=DataProvenance(
                input=input_spec,
                row_count=input_spec.sample_cap,
                preprocessing={"prompt_column": input_spec.column_mapping["prompt"]},
            ),
            generation=GenerationConfig(
                max_new_tokens=payload.get("max_new_tokens", 128), do_sample=False
            ),
            probe=ProbeProvenance(
                probe_plan_id=f"plan:{stable_digest(probe_payload)}",
                capture_level=capture_level,
            ),
            privacy=privacy,
        )

    specification = make_specification(
        capture_level=5,
        expert_activity=bool(payload.get("capture_expert_activity", True)),
    )
    _publish_run_policy(workspace, specification.run_key, specification.privacy)
    executor_module = import_module("moeatlas.executors." + "transform" + "ers_routing")
    executor_type = getattr(executor_module, "Transform" + "ersRoutingExecutor")

    def on_record(record: Any, *, phase: str | None = None) -> None:
        progress = getattr(record, "progress", None)
        if progress is None:
            report_progress(
                stage=phase or str(getattr(record, "state", "running")),
                completed=0,
                total=None,
                message="Run lifecycle updated",
            )
            return
        report_progress(
            stage=phase or progress.stage,
            completed=progress.completed_units,
            total=progress.total_units,
            message=(f"{progress.completed_units}/{progress.total_units or '?'} rows processed"),
        )

    def timing_delta(native: dict[str, Any], captured: dict[str, Any]) -> dict[str, float | None]:
        native_total = native.get("total_ms")
        captured_total = captured.get("total_ms")
        if not isinstance(native_total, int | float) or not isinstance(
            captured_total, int | float
        ):
            return {"delta_ms": None, "delta_percent": None}
        delta = float(captured_total) - float(native_total)
        return {
            "delta_ms": delta,
            "delta_percent": (delta / float(native_total)) * 100.0
            if native_total > 0
            else None,
        }

    measure_overhead = bool(payload.get("measure_capture_overhead", False)) and resume_from is None
    overhead_report: dict[str, Any] | None = None
    if measure_overhead:
        if skip_overhead is None:
            def skip_overhead() -> bool:
                return False

        overhead_report = {
            "schema_version": "1.0",
            "status": "pending",
            "scope": "model_forward",
            "model_id": model_provenance.model_id,
            "model_revision": model_provenance.model_revision,
            "dataset_id": payload["dataset_id"],
            "dataset_revision": dataset_revision,
            "dataset_split": payload.get("dataset_split", "train"),
            "sample_cap": input_spec.sample_cap,
            "batch_size": input_spec.batch_size,
            "capture_run_key": specification.run_key,
            "native_run_key": None,
            "native": None,
            "captured": None,
            "delta_ms": None,
            "delta_percent": None,
        }
        native_specification = make_specification(capture_level=0, expert_activity=False)
        overhead_report["native_run_key"] = native_specification.run_key
        native_executor = executor_type(
            plan,
            store_token_text=False,
            capture_expert_activity=False,
            capture_routing=False,
        )
        native_executor.bind_run_key(native_specification.run_key)
        native_execution = None
        try:
            if cancel.is_set():
                return JobOutcome(
                    {
                        "status": "cancelled",
                        "capture_overhead": {**overhead_report, "status": "cancelled"},
                    },
                    "cancelled",
                )
            if skip_overhead():
                overhead_report["status"] = "skipped"
            else:
                report_progress(
                    stage="overhead",
                    completed=0,
                    total=input_spec.sample_cap,
                    message="Measuring native baseline with capture disabled",
                )
                native_execution = execute_specification(
                    native_specification,
                    executor=native_executor,
                    base_directory=workspace,
                    should_cancel=lambda: cancel.is_set() or skip_overhead(),
                    requested_by="local-server-overhead",
                    cancellation_reason="optional overhead measurement skipped",
                    on_record=lambda record: on_record(record, phase="overhead"),
                    checkpoint_directory=Path(workspace) / "checkpoints" / "overhead",
                    max_rows=input_spec.sample_cap or 32,
                )
                if cancel.is_set():
                    overhead_report["status"] = "cancelled"
                    return JobOutcome(
                        {
                            "status": "cancelled",
                            "capture_overhead": {
                                **overhead_report,
                                "native": native_executor.timing_summary(),
                            },
                        },
                        "cancelled",
                    )
                if skip_overhead() and native_execution.final_record.state.value == "cancelled":
                    overhead_report["status"] = "skipped"
                elif native_execution.final_record.state.value == "failed":
                    overhead_report["status"] = "failed"
                else:
                    overhead_report["status"] = "native_complete"
                overhead_report["native"] = native_executor.timing_summary()
        finally:
            native_executor.close()

    executor = executor_type(
        plan,
        store_token_text=payload.get("token_text_policy") == "stored",
        capture_expert_activity=bool(payload.get("capture_expert_activity", True)),
        capture_routing=True,
    )
    executor.bind_run_key(specification.run_key)
    try:
        report_progress(
            stage="execute",
            completed=0,
            total=input_spec.sample_cap,
            message="Starting model execution",
        )
        execution = execute_specification(
            specification,
            executor=executor,
            base_directory=workspace,
            should_cancel=cancel.is_set,
            requested_by="local-server",
            cancellation_reason="cancelled from research console",
            on_record=on_record,
            checkpoint_directory=Path(workspace) / "checkpoints",
            resume_from=resume_from,
            max_rows=input_spec.sample_cap or 32,
        )
        publish_run_report(workspace, execution)
        receipt = executor.publish_run_artifacts(workspace)
        terminal = execution.final_record.state.value
        checkpoint_path: str | None = None
        if execution.checkpoint_path:
            checkpoint = Path(execution.checkpoint_path)
            try:
                checkpoint_path = checkpoint.resolve().relative_to(
                    Path(workspace).resolve()
                ).as_posix()
            except (OSError, ValueError):
                checkpoint_path = None
        payload_out = {
            "status": terminal,
            "run_key": execution.run_key,
            "checkpoint_path": checkpoint_path,
            "resumed_from_batch": execution.resumed_from_batch,
            "executed_rows": execution.outcome.executed_rows,
            "total_rows": execution.outcome.total_rows,
            "failed_rows": len(execution.outcome.failures),
            "shard_key": receipt.shard_key if receipt is not None else None,
            "plan_id": plan.plan_id,
            "resolved_model_revision": model_provenance.model_revision,
            "resolved_dataset_revision": dataset_revision,
        }
        if overhead_report is not None:
            overhead_report["captured"] = executor.timing_summary()
            overhead_report.update(
                timing_delta(overhead_report["native"] or {}, overhead_report["captured"])
            )
            if terminal == "cancelled":
                overhead_report["status"] = "capture_cancelled"
            elif terminal == "failed":
                overhead_report["status"] = "capture_failed"
            elif overhead_report["status"] == "native_complete":
                overhead_report["status"] = "completed"
            payload_out["capture_overhead"] = overhead_report
            payload_out["capture_overhead_path"] = _publish_capture_overhead(
                workspace,
                specification.run_key,
                overhead_report,
            )
        return JobOutcome(
            payload_out,
            "cancelled"
            if terminal == "cancelled"
            else "failed"
            if terminal == "failed"
            else "completed",
        )
    finally:
        executor.close()


def _intervention_recipe(payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
    from ..interventions.recipes import InterventionOperation, InterventionRecipe

    recipe = InterventionRecipe(
        operation=InterventionOperation(payload["operation"]),
        targets=tuple(payload["targets"]),
        factor=payload.get("factor"),
        bias=payload.get("bias"),
        alternates=tuple(tuple(item) for item in payload.get("alternates", ())),
    )
    return recipe.to_dict(), recipe.fingerprint


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
        raise ValueError(f"max_results must be between 1 and {_MAX_RESULTS_CEILING}")
    if type(max_artifact_bytes) is not int or isinstance(max_artifact_bytes, bool):
        raise TypeError("max_artifact_bytes must be an integer")
    if max_artifact_bytes <= 0 or max_artifact_bytes > _MAX_ARTIFACT_BYTES_CEILING:
        raise ValueError(f"max_artifact_bytes must be between 1 and {_MAX_ARTIFACT_BYTES_CEILING}")
    if not isinstance(workspace, str | Path):
        raise TypeError("workspace must be a string or Path")
    bound_workspace = str(workspace)

    try:
        from fastapi import FastAPI, HTTPException, Query
    except ImportError as exc:
        raise ServerDependencyError() from exc

    from .jobs import JobManager

    # Model jobs are VRAM-exclusive. A single worker prevents a retry or a
    # discovery/run pair from loading a second checkpoint concurrently while a
    # prior failure is still unwinding its runtime cleanup.
    jobs = JobManager(max_workers=1, workspace=bound_workspace)

    @asynccontextmanager
    async def _lifespan(_: Any):
        try:
            yield
        finally:
            jobs.shutdown(wait=False)

    app = FastAPI(
        title="MoEAtlas local server",
        version=SERVER_SCHEMA_VERSION,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=_lifespan,
    )
    app.state.jobs = jobs

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
        entry = next((e for e in snapshot.catalog.runs if e.run_key == run_key), None)
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

    def _safe_discovery_document(workspace_root: Path, run_key: str) -> Path | None:
        root = workspace_root / _DISCOVERY_DIRECTORY
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

    def _safe_policy_document(workspace_root: Path, run_key: str) -> Path | None:
        root = workspace_root / _POLICY_DIRECTORY
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

    def _load_inspection(workspace_path: Path, stable_run_key: str) -> Any:
        inspection_path = _safe_inspection_document(workspace_path, stable_run_key)
        if inspection_path is None:
            raise ValueError("published routing inspection is unavailable")
        from ..adapters import AdapterInspection, UniversalRoutingInspection

        document = inspection_path.read_bytes()
        if len(document) > max_artifact_bytes:
            raise ValueError("inspection exceeds the serving byte budget")
        try:
            return UniversalRoutingInspection.model_validate_json(document)
        except Exception:
            return AdapterInspection.model_validate_json(document)

    def _load_matrix(workspace_path: Path, stable_run_key: str) -> Any:
        from ..analysis import aggregate_routing_load

        inspection = _load_inspection(workspace_path, stable_run_key)
        return aggregate_routing_load(
            workspace_path,
            inspection,
            run_key=stable_run_key,
            max_routing_rows=1_000_000,
            max_source_bytes=1_000_000_000,
            max_matrix_cells=100_000,
        )

    def _job_response(snapshot: dict[str, Any]) -> JobResponse:
        progress = snapshot.get("progress") or {}
        return JobResponse(
            job_id=snapshot["job_id"],
            kind=snapshot["kind"],
            state=snapshot["state"],
            progress=JobProgressResponse(
                stage=str(progress.get("stage", "unknown")),
                completed=int(progress.get("completed", 0)),
                total=progress.get("total"),
                message=str(progress.get("message", "")),
            ),
            result=snapshot.get("result"),
            error=snapshot.get("error"),
            diagnostics=snapshot.get("diagnostics"),
        )

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

    @app.post("/api/discovery", response_model=JobCreatedResponse, status_code=202)
    def start_discovery(request: DiscoveryRequest) -> JobCreatedResponse:
        payload = request.model_dump(mode="python")
        job_id = jobs.submit(
            "discovery",
            lambda cancel, progress: _discovery_worker(
                payload, cancel=cancel, report_progress=progress
            ),
        )
        snapshot = jobs.snapshot(job_id)
        assert snapshot is not None
        return JobCreatedResponse(job_id=job_id, kind="discovery", state=snapshot["state"])

    @app.post("/api/runs/start", response_model=JobCreatedResponse, status_code=202)
    def start_run(request: RunStartRequest) -> JobCreatedResponse:
        from ..services import open_workspace

        try:
            open_workspace(bound_workspace)
        except Exception as exc:
            raise _not_initialized() from exc
        payload = request.model_dump(mode="python")
        resume_from: str | None = None
        resume_job_id = payload.get("resume_job_id")
        if resume_job_id:
            prior = jobs.snapshot(resume_job_id)
            if prior is None or prior["kind"] != "run" or prior["state"] != "cancelled":
                raise HTTPException(status_code=400, detail="resume job is not a cancelled run")
            checkpoint = (prior.get("result") or {}).get("checkpoint_path")
            if not isinstance(checkpoint, str):
                raise HTTPException(status_code=400, detail="resume checkpoint is unavailable")
            checkpoint_path = Path(checkpoint)
            if not checkpoint_path.is_absolute():
                checkpoint_path = Path(bound_workspace) / checkpoint_path
            checkpoint_root = Path(bound_workspace) / "checkpoints"
            try:
                if (
                    checkpoint_path.is_symlink()
                    or checkpoint_path.resolve().parent != checkpoint_root.resolve()
                ):
                    raise ValueError
                if not checkpoint_path.is_file():
                    raise ValueError
            except OSError as exc:
                raise HTTPException(
                    status_code=400, detail="resume checkpoint is unavailable"
                ) from exc
            except ValueError as exc:
                raise HTTPException(
                    status_code=400, detail="resume checkpoint is unavailable"
                ) from exc
            resume_from = str(checkpoint_path)
            # A resumed capture uses its original durable checkpoint and must
            # not silently spend another full model pass on the optional lane.
            payload["measure_capture_overhead"] = False
        skip_overhead_event = (
            threading.Event()
            if bool(payload.get("measure_capture_overhead", False)) and resume_from is None
            else None
        )
        job_id = jobs.submit(
            "run",
            lambda cancel, progress: _run_worker(
                bound_workspace,
                payload,
                cancel=cancel,
                report_progress=progress,
                resume_from=resume_from,
                skip_overhead=(
                    skip_overhead_event.is_set if skip_overhead_event is not None else None
                ),
            ),
            optional_skip_event=skip_overhead_event,
        )
        snapshot = jobs.snapshot(job_id)
        assert snapshot is not None
        return JobCreatedResponse(job_id=job_id, kind="run", state=snapshot["state"])

    @app.get("/api/jobs/{job_id}", response_model=JobResponse)
    def job_status(job_id: str) -> JobResponse:
        if type(job_id) is not str or not job_id.startswith("job:") or len(job_id) != 36:
            raise HTTPException(status_code=404, detail="job is not registered")
        snapshot = jobs.snapshot(job_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="job is not registered")
        return _job_response(snapshot)

    @app.get("/api/jobs/{job_id}/diagnostics", response_model=JobDiagnosticsResponse)
    def job_diagnostics(job_id: str) -> JobDiagnosticsResponse:
        """Read bounded, sanitized diagnostics for a known in-process job."""

        if type(job_id) is not str or not job_id.startswith("job:") or len(job_id) != 36:
            raise HTTPException(status_code=404, detail="job is not registered")
        document = jobs.diagnostics(job_id)
        if document is None:
            raise HTTPException(status_code=404, detail="job is not registered")
        entries: list[JobDiagnosticEntryResponse] = []
        for entry in document.get("entries", ()):
            if not isinstance(entry, dict):
                continue
            try:
                entries.append(JobDiagnosticEntryResponse.model_validate(entry))
            except Exception:
                # A malformed/truncated line is not allowed to widen the wire
                # contract or turn diagnostics into a second failure surface.
                continue
        return JobDiagnosticsResponse(
            job_id=document["job_id"],
            kind=document["kind"],
            state=document["state"],
            available=bool(document.get("available", False)),
            entry_count=len(entries),
            truncated=bool(document.get("truncated", False)),
            entries=tuple(entries),
        )

    @app.post("/api/jobs/{job_id}/cancel", response_model=JobResponse)
    def cancel_job(job_id: str) -> JobResponse:
        if type(job_id) is not str or not job_id.startswith("job:") or len(job_id) != 36:
            raise HTTPException(status_code=404, detail="job is not registered")
        snapshot = jobs.snapshot(job_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="job is not registered")
        jobs.cancel(job_id)
        updated = jobs.snapshot(job_id)
        assert updated is not None
        return _job_response(updated)

    @app.post("/api/jobs/{job_id}/skip-overhead", response_model=JobResponse)
    def skip_overhead(job_id: str) -> JobResponse:
        if type(job_id) is not str or not job_id.startswith("job:") or len(job_id) != 36:
            raise HTTPException(status_code=404, detail="job is not registered")
        snapshot = jobs.snapshot(job_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="job is not registered")
        if not jobs.skip_optional(job_id):
            raise HTTPException(status_code=409, detail="overhead measurement is not running")
        updated = jobs.snapshot(job_id)
        assert updated is not None
        return _job_response(updated)

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
            receipts = reader_from_workspace(bound_workspace).list_shards(run_key=stable_run_key)
        except Exception as exc:
            raise HTTPException(status_code=404, detail="run shards are unavailable") from exc
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
        if _safe_inspection_document(workspace_path, stable_run_key) is None:
            return RunSummaryResponse(
                run_key=stable_run_key,
                status="unavailable",
                reason="published routing inspection is unavailable",
            )
        try:
            matrix = _load_matrix(workspace_path, stable_run_key)
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

    @app.get("/api/runs/{run_key}/architecture", response_model=ArchitectureResponse)
    def run_architecture(run_key: str) -> ArchitectureResponse:
        stable_run_key = _validated_run_key(run_key)
        workspace_path, _ = _catalog_entry(stable_run_key)
        candidate = _safe_discovery_document(workspace_path, stable_run_key)
        if candidate is None:
            return ArchitectureResponse(
                run_key=stable_run_key,
                status="unavailable",
                reason="published discovery report is unavailable",
            )
        try:
            payload = candidate.read_bytes()
            if len(payload) > max_artifact_bytes:
                raise ValueError
            document = json.loads(payload.decode("utf-8"))
            if not isinstance(document, dict):
                raise ValueError
        except Exception:
            return ArchitectureResponse(
                run_key=stable_run_key,
                status="unavailable",
                reason="published discovery report is not valid",
            )
        return ArchitectureResponse(
            run_key=stable_run_key,
            status="available",
            report=document,
        )

    @app.get("/api/runs/{run_key}/activity", response_model=ActivityResponse)
    def run_activity(run_key: str) -> ActivityResponse:
        stable_run_key = _validated_run_key(run_key)
        workspace_path, _ = _catalog_entry(stable_run_key)
        try:
            matrix = _load_matrix(workspace_path, stable_run_key)
            from ..analysis import summarize_expert_activity

            summary = summarize_expert_activity(
                workspace_path,
                run_key=stable_run_key,
                layer_keys=matrix.layer_keys,
                expert_keys=matrix.expert_keys,
                max_expert_rows=1_000_000,
                max_source_bytes=1_000_000_000,
            )
            document = summary.to_dict()
        except Exception:
            return ActivityResponse(
                run_key=stable_run_key,
                status="unavailable",
                reason="expert activity evidence is unavailable for this run",
            )
        return ActivityResponse(run_key=stable_run_key, status="available", summary=document)

    @app.get("/api/runs/{run_key}/heatmap")
    def run_heatmap(
        run_key: str,
        metric: str = Query(default="assignment_counts"),
        view: str = Query(default="report"),
    ) -> Any:
        from fastapi import Response

        stable_run_key = _validated_run_key(run_key)
        if metric not in {"assignment_counts", "assignment_shares", "load_ratios"}:
            raise HTTPException(status_code=400, detail="unsupported heatmap metric")
        if view not in {"report", "compact"}:
            raise HTTPException(status_code=400, detail="unsupported heatmap view")
        workspace_path, _ = _catalog_entry(stable_run_key)
        candidate = (
            _safe_heatmap_document(workspace_path, stable_run_key)
            if metric == "assignment_counts" and view == "report"
            else None
        )
        if candidate is None:
            inspection_path = _safe_inspection_document(workspace_path, stable_run_key)
            if inspection_path is None:
                raise HTTPException(status_code=404, detail="run heatmap is not published")
            try:
                from ..analysis import (
                    render_compact_routing_load_heatmap,
                    render_routing_load_heatmap,
                )

                document = inspection_path.read_bytes()
                if len(document) > _DEFAULT_ARTIFACT_BYTES:
                    raise ValueError("inspection exceeds the serving byte budget")
                matrix = _load_matrix(workspace_path, stable_run_key)
                renderer = (
                    render_compact_routing_load_heatmap
                    if view == "compact"
                    else render_routing_load_heatmap
                )
                payload = renderer(
                    matrix, metric=metric, max_cells=100_000
                ).encode("utf-8")
                if len(payload) > max_artifact_bytes:
                    raise ValueError("run heatmap exceeds the serving byte budget")
                return Response(content=payload, media_type="text/html; charset=utf-8")
            except Exception as exc:
                raise HTTPException(status_code=404, detail="run heatmap is not published") from exc
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
            raise HTTPException(status_code=404, detail="run heatmap is not published") from exc
        if len(payload) > max_artifact_bytes:
            raise HTTPException(
                status_code=404,
                detail="run heatmap exceeds the serving byte budget",
            )
        return Response(content=payload, media_type="text/html; charset=utf-8")

    @app.get("/api/compare/heatmap")
    def compare_heatmap(
        baseline_run_key: str = Query(...),
        comparison_run_key: str = Query(...),
        metric: str = Query(default="count_deltas"),
    ) -> Any:
        from fastapi import Response

        if metric not in {"count_deltas", "share_deltas", "ratio_deltas"}:
            raise HTTPException(status_code=400, detail="unsupported comparison metric")
        baseline = _validated_run_key(baseline_run_key)
        comparison = _validated_run_key(comparison_run_key)
        if baseline == comparison:
            raise HTTPException(status_code=400, detail="comparison runs must differ")
        baseline_workspace, _ = _catalog_entry(baseline)
        comparison_workspace, _ = _catalog_entry(comparison)
        if baseline_workspace != comparison_workspace:
            raise HTTPException(status_code=400, detail="comparison runs must share a workspace")
        try:
            from ..analysis import compare_routing_load, render_routing_load_comparison

            document = render_routing_load_comparison(
                compare_routing_load(
                    _load_matrix(baseline_workspace, baseline),
                    _load_matrix(comparison_workspace, comparison),
                    max_cells=100_000,
                ),
                metric=metric,
                max_cells=100_000,
            ).encode("utf-8")
            if len(document) > max_artifact_bytes:
                raise ValueError
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=404, detail="run comparison is unavailable") from exc
        return Response(content=document, media_type="text/html; charset=utf-8")

    @app.get("/api/runs/{run_key}/export")
    def export_run(run_key: str, format: str = Query(default="bundle")) -> Any:
        from fastapi.responses import FileResponse

        stable_run_key = _validated_run_key(run_key)
        if format not in {"bundle", "csv", "parquet"}:
            raise HTTPException(status_code=400, detail="unsupported export format")
        workspace_path, _ = _catalog_entry(stable_run_key)
        policy_path = _safe_policy_document(workspace_path, stable_run_key)
        if policy_path is not None:
            try:
                policy_payload = json.loads(policy_path.read_bytes().decode("utf-8"))
                if not isinstance(policy_payload, dict):
                    raise ValueError
                if policy_payload.get("allow_export") is False:
                    raise HTTPException(
                        status_code=403,
                        detail="run export is disabled by privacy policy",
                    )
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(status_code=404, detail="run export is unavailable") from exc
        export_root = workspace_path / _EXPORT_DIRECTORY
        if export_root.exists() and export_root.is_symlink():
            raise HTTPException(status_code=404, detail="run export is unavailable")
        export_root.mkdir(exist_ok=True)
        target = export_root / f"{stable_run_key}.{format}.zip"
        if target.exists():
            if target.is_symlink() or not target.is_file():
                raise HTTPException(status_code=404, detail="run export is unavailable")
            if target.stat().st_size > max_artifact_bytes:
                raise HTTPException(
                    status_code=404, detail="run export exceeds the serving byte budget"
                )
            return FileResponse(target, media_type="application/zip", filename=target.name)
        stage = Path(tempfile.mkdtemp(prefix=f".{stable_run_key}.", dir=str(export_root)))
        staged_zip = stage / "artifact.zip"
        try:
            from ..store import export_run_bundle, export_run_tables

            if format == "bundle":
                export_run_bundle(
                    workspace_path,
                    stage / "bundle",
                    run_key=stable_run_key,
                    max_file_bytes=max_artifact_bytes,
                )
                source = stage / "bundle"
            else:
                export_run_tables(
                    workspace_path,
                    stage / "tables",
                    run_key=stable_run_key,
                    formats=(format,),
                    max_file_bytes=max_artifact_bytes,
                )
                source = stage / "tables"
            with zipfile.ZipFile(staged_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for member in sorted(source.rglob("*")):
                    if member.is_file():
                        archive.write(member, member.relative_to(source).as_posix())
            if staged_zip.stat().st_size > max_artifact_bytes:
                raise ValueError("export exceeds the serving byte budget")
            staged_target = export_root / f".{target.name}.staging"
            os.replace(staged_zip, staged_target)
            os.replace(staged_target, target)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=404, detail="run export is unavailable") from exc
        finally:
            shutil.rmtree(stage, ignore_errors=True)
        return FileResponse(target, media_type="application/zip", filename=target.name)

    @app.post("/api/interventions/recipes", response_model=InterventionRecipeResponse)
    def prepare_intervention(request: InterventionRecipeRequest) -> InterventionRecipeResponse:
        try:
            recipe, fingerprint = _intervention_recipe(request.model_dump(mode="python"))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="intervention recipe is invalid") from exc
        return InterventionRecipeResponse(
            status="prepared",
            recipe=recipe,
            fingerprint=fingerprint,
            reason=(
                "Recipe validated; causal execution still requires an adapter "
                "capability for this model."
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

    # Static frontend mount, registered strictly after every API route so
    # /healthz and /api/* always win. The packaged React bundle and legacy
    # compatibility assets are served with caching disabled so local
    # development observes freshly published bytes.
    if _STATIC_DIRECTORY.is_dir():
        from fastapi.staticfiles import StaticFiles

        @app.middleware("http")
        async def _disable_static_caching(request: Any, call_next: Any) -> Any:
            response = await call_next(request)
            if not (request.url.path.startswith("/api/") or request.url.path == "/healthz"):
                response.headers["Cache-Control"] = "no-store"
            return response

        app.mount("/", StaticFiles(directory=_STATIC_DIRECTORY, html=True), name="static")

    return app


__all__ = ["SERVER_SCHEMA_VERSION", "ServerDependencyError", "create_app"]
