"""Isolated discovery, execution, and intervention job workers.

The HTTP application imports these functions, while this module owns model loading,
dataset execution, process isolation, and atomic worker-side evidence publication.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

_POLICY_DIRECTORY = "policies"
_BENCHMARK_DIRECTORY = "benchmarks"
_CAPTURE_OVERHEAD_DIRECTORY = "capture-overhead"
_MAX_JOB_PAYLOAD_BYTES = 5_000_000


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
    fd, staged_name = tempfile.mkstemp(dir=str(directory), prefix=f".{run_key}.", suffix=".staging")
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
    fd, staged_name = tempfile.mkstemp(dir=str(directory), prefix=f".{run_key}.", suffix=".staging")
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

    from ..runtime import (
        RuntimeQualificationError,
        admit_huggingface_model,
        classify_capture_support,
        load_scan_and_observe,
        qualify_huggingface_runtime,
    )
    from ..services.model_resolution import resolve_huggingface_plan_with_metadata
    from .jobs import JobOutcome

    if cancel.is_set():
        return JobOutcome({"status": "cancelled"}, "cancelled")
    report_progress(
        stage="resolve",
        completed=0,
        total=1,
        message="Resolving immutable Hub revision",
    )
    plan, hub_metadata = resolve_huggingface_plan_with_metadata(
        payload["model_id"],
        payload.get("model_revision", "main"),
        device=payload.get("device", "auto"),
        dtype=payload.get("dtype", "preserve"),
        trust_remote_code=bool(payload.get("trust_remote_code", False)),
        allow_downloads=bool(payload.get("allow_downloads", True)),
    )
    report_progress(
        stage="qualify",
        completed=0,
        total=1,
        message="Checking model runtime requirements before weight loading",
    )
    qualification = qualify_huggingface_runtime(plan)
    if not qualification.ready:
        missing = list(qualification.missing_packages) + list(qualification.missing_imports)
        detail = ", ".join(missing) if missing else "remote-code permission"
        raise RuntimeQualificationError(f"model runtime is incompatible: {detail}")
    report_progress(
        stage="resources",
        completed=0,
        total=1,
        message="Checking cache disk and accelerator capacity",
    )
    resource_admission = admit_huggingface_model(
        plan.source.model_id,
        plan.resolution.resolved_model_revision
        if plan.resolution
        else plan.source.requested_revision,
        device=payload.get("device", "auto"),
        dtype=payload.get("dtype", "preserve"),
        allow_network=bool(payload.get("allow_downloads", True)),
    )
    report_progress(stage="load", completed=0, total=1, message="Loading model and tokenizer")
    if cancel.is_set():
        return JobOutcome({"status": "cancelled", "plan_id": plan.plan_id}, "cancelled")
    from ..interventions import inspect_intervention_capability

    discovery, intervention_capability = load_scan_and_observe(
        plan,
        lambda model, report: inspect_intervention_capability(report, model),
        load_progress=lambda stage, completed, total, message: report_progress(
            stage=stage,
            completed=completed,
            total=total,
            message=message,
        ),
    )

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
            "resource_admission": resource_admission.to_dict(),
            "runtime_qualification": qualification.to_dict(),
            "repository_size_bytes": hub_metadata.repository_size_bytes,
            "capture_support": capture_support.to_dict(),
            "intervention_capability": intervention_capability.to_dict(),
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
        InterventionLineage,
        ModelProvenance,
        PrivacyPolicy,
        ProbeProvenance,
        RunMode,
        RunSpecification,
        TokenTextPolicy,
    )
    from ..runtime import (
        RuntimeQualificationError,
        admit_huggingface_model,
        qualify_huggingface_runtime,
    )
    from ..services.datasets import read_dataset_rows
    from ..services.model_resolution import (
        resolve_huggingface_dataset_revision,
        resolve_huggingface_plan_with_metadata,
    )
    from ..services.run_metadata import publish_run_metadata
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
    plan, hub_metadata = resolve_huggingface_plan_with_metadata(
        payload["model_id"],
        payload.get("model_revision", "main"),
        device=payload.get("device", "auto"),
        dtype=payload.get("dtype", "preserve"),
        trust_remote_code=bool(payload.get("trust_remote_code", False)),
        allow_downloads=bool(payload.get("allow_downloads", True)),
    )
    allow_downloads = bool(payload.get("allow_downloads", True))
    report_progress(
        stage="qualify",
        completed=0,
        total=1,
        message="Checking model runtime requirements before weight loading",
    )
    qualification = qualify_huggingface_runtime(plan)
    if not qualification.ready:
        missing = list(qualification.missing_packages) + list(qualification.missing_imports)
        detail = ", ".join(missing) if missing else "remote-code permission"
        raise RuntimeQualificationError(f"model runtime is incompatible: {detail}")
    report_progress(
        stage="resources",
        completed=0,
        total=1,
        message="Checking cache disk and accelerator capacity",
    )
    resource_admission = admit_huggingface_model(
        plan.source.model_id,
        plan.resolution.resolved_model_revision
        if plan.resolution
        else plan.source.requested_revision,
        device=payload.get("device", "auto"),
        dtype=payload.get("dtype", "preserve"),
        allow_network=allow_downloads,
    )
    dataset_revision = resolve_huggingface_dataset_revision(
        payload["dataset_id"],
        payload.get("dataset_revision", "main"),
        allow_downloads=allow_downloads,
    )
    report_progress(stage="resolve", completed=2, total=2, message="Immutable revisions resolved")
    if cancel.is_set():
        return JobOutcome({"status": "cancelled", "plan_id": plan.plan_id}, "cancelled")

    requested_reference_column = payload.get("reference_column")
    initial_column_mapping = {"prompt": payload.get("prompt_column", "prompt")}
    if requested_reference_column is not None:
        initial_column_mapping["reference"] = requested_reference_column
    input_spec = DatasetInputSpec(
        format=DatasetFormat.HF_DATASETS,
        location=payload["dataset_id"],
        revision=dataset_revision,
        config_name=payload.get("dataset_config"),
        split=payload.get("dataset_split", "train"),
        allow_downloads=allow_downloads,
        column_mapping=initial_column_mapping,
        sample_cap=payload.get("sample_cap", 32),
        seed=payload.get("dataset_seed"),
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
        resolved_mapping = {"prompt": chosen}
        if requested_reference_column is not None:
            if requested_reference_column not in preview:
                raise ValueError("requested dataset reference column is unavailable")
            resolved_mapping["reference"] = requested_reference_column
        input_spec = DatasetInputSpec(
            **{
                **input_spec.model_dump(mode="python"),
                "column_mapping": resolved_mapping,
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

    intervention_payload = payload.get("intervention")
    recipe = (
        _intervention_recipe_object(intervention_payload)
        if isinstance(intervention_payload, dict)
        else None
    )
    baseline_run_key = payload.get("baseline_run_key")
    if (recipe is None) != (baseline_run_key is None):
        raise ValueError("intervention recipe and baseline run key must be provided together")

    def make_specification(
        *, capture_level: int, expert_activity: bool, run_name: str | None
    ) -> RunSpecification:
        probe_payload = {
            "model": plan.plan_id,
            "capture_level": capture_level,
            "expert_activity": expert_activity,
            "routing_capture": capture_level > 0,
            "intervention_recipe": recipe.fingerprint if recipe is not None else None,
        }
        return RunSpecification(
            run_name=run_name,
            workspace=workspace,
            created_by="local-server",
            model=model_provenance,
            data=DataProvenance(
                input=input_spec,
                row_count=input_spec.sample_cap,
                preprocessing={"column_mapping": dict(input_spec.column_mapping)},
            ),
            generation=GenerationConfig(
                max_new_tokens=payload.get("max_new_tokens", 128), do_sample=False
            ),
            probe=ProbeProvenance(
                probe_plan_id=f"plan:{stable_digest(probe_payload)}",
                capture_level=capture_level,
                intervention_opt_in=recipe is not None,
            ),
            privacy=privacy,
            intervention=(
                InterventionLineage(
                    baseline_run_key=baseline_run_key,
                    recipe_fingerprint=recipe.fingerprint,
                    operation=recipe.operation.value,
                    targets=recipe.targets,
                )
                if recipe is not None
                else None
            ),
        )

    specification = make_specification(
        capture_level=5,
        expert_activity=bool(payload.get("capture_expert_activity", True)),
        run_name=payload.get("run_name"),
    )
    _publish_run_policy(workspace, specification.run_key, specification.privacy)
    resolved_request = {
        **payload,
        "model_revision": model_provenance.model_revision,
        "dataset_revision": dataset_revision,
        "prompt_column": input_spec.column_mapping["prompt"],
        "reference_column": input_spec.column_mapping.get("reference"),
        "resume_job_id": None,
        "measure_capture_overhead": False
        if recipe is not None
        else bool(payload.get("measure_capture_overhead", False)),
    }
    publish_run_metadata(workspace, specification, resolved_request)
    executor_module = import_module("moeatlas.executors." + "transform" + "ers_routing")
    executor_type = getattr(executor_module, "Transform" + "ersRoutingExecutor")

    def on_load_progress(stage: str, completed: int, total: int, message: str) -> None:
        report_progress(
            stage=stage,
            completed=completed,
            total=total,
            message=message,
        )

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
        if not isinstance(native_total, int | float) or not isinstance(captured_total, int | float):
            return {"delta_ms": None, "delta_percent": None}
        delta = float(captured_total) - float(native_total)
        return {
            "delta_ms": delta,
            "delta_percent": (delta / float(native_total)) * 100.0 if native_total > 0 else None,
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
        native_specification = make_specification(
            capture_level=0, expert_activity=False, run_name=None
        )
        overhead_report["native_run_key"] = native_specification.run_key
        native_executor = executor_type(
            plan,
            store_token_text=False,
            capture_expert_activity=False,
            capture_routing=False,
            mode=payload.get("mode", "generation"),
            max_new_tokens=payload.get("max_new_tokens", 128),
            load_progress=on_load_progress,
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
        mode=payload.get("mode", "generation"),
        max_new_tokens=payload.get("max_new_tokens", 128),
        evaluation_method=payload.get("evaluation_method", "normalized_exact_match"),
        load_progress=on_load_progress,
    )
    executor.bind_run_key(specification.run_key)
    try:
        report_progress(
            stage="execute",
            completed=0,
            total=input_spec.sample_cap,
            message="Starting model execution",
        )

        def execute_current_specification() -> Any:
            return execute_specification(
                specification,
                executor=executor,
                base_directory=workspace,
                should_cancel=cancel.is_set,
                requested_by=(
                    "local-server-intervention" if recipe is not None else "local-server"
                ),
                cancellation_reason="cancelled from research console",
                on_record=on_record,
                checkpoint_directory=Path(workspace) / "checkpoints",
                resume_from=resume_from,
                max_rows=input_spec.sample_cap or 32,
            )

        intervention_outcome = None
        invocation_counts: dict[str, int] = {}
        if recipe is not None:
            execution, intervention_outcome, invocation_counts = executor.run_with_intervention(
                recipe,
                execute_current_specification,
            )
        else:
            execution = execute_current_specification()
        terminal = execution.final_record.state.value
        intervention_evidence_document: dict[str, Any] | None = None
        if recipe is not None and intervention_outcome is not None and terminal == "completed":
            from ..interventions import build_intervention_evidence
            from ..services.run_service import load_checkpoint

            assert isinstance(baseline_run_key, str)
            baseline_checkpoint = load_checkpoint(
                Path(workspace)
                / "checkpoints"
                / f"{baseline_run_key.removeprefix('run:')}.checkpoint.json"
            )
            intervention_evidence_document = build_intervention_evidence(
                baseline_run_key=baseline_run_key,
                intervention_run_key=execution.run_key,
                baseline_execution=baseline_checkpoint,
                intervention_execution=execution.outcome,
                recipe=recipe,
                outcome=intervention_outcome,
                invocation_counts=invocation_counts,
            )
        publish_run_report(workspace, execution)
        receipt = executor.publish_run_artifacts(workspace)
        checkpoint_path: str | None = None
        if execution.checkpoint_path:
            checkpoint = Path(execution.checkpoint_path)
            try:
                checkpoint_path = (
                    checkpoint.resolve().relative_to(Path(workspace).resolve()).as_posix()
                )
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
            # Keep row-level failure evidence on the job result as well as in
            # the checkpoint.  The view is bounded and sanitized by the run
            # service; it is safe for the control-plane response and lets the
            # diagnostics layer explain a failed run without an outer
            # exception object.
            "failure_summary": execution.failure_summary,
            "failure_evidence": execution.failure_evidence,
            "shard_key": receipt.shard_key if receipt is not None else None,
            "plan_id": plan.plan_id,
            "resolved_model_revision": model_provenance.model_revision,
            "resolved_dataset_revision": dataset_revision,
            "resource_admission": resource_admission.to_dict(),
            "runtime_qualification": qualification.to_dict(),
            "repository_size_bytes": hub_metadata.repository_size_bytes,
        }
        if recipe is not None and intervention_outcome is not None:
            payload_out["baseline_run_key"] = baseline_run_key
            payload_out["intervention_recipe_fingerprint"] = recipe.fingerprint
            payload_out["target_invocation_counts"] = invocation_counts
            if intervention_evidence_document is not None:
                from ..interventions import publish_intervention_evidence

                evidence_path = publish_intervention_evidence(
                    workspace, intervention_evidence_document
                )
                payload_out["intervention_evidence_path"] = evidence_path.relative_to(
                    Path(workspace)
                ).as_posix()
                payload_out["intervention_summary"] = {
                    key: intervention_evidence_document[key]
                    for key in (
                        "changed_output_fraction",
                        "task_score_delta",
                        "latency_delta_percent",
                        "all_targets_exercised",
                    )
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


class _ChildCancellation:
    """Cancellation view exposed inside an isolated model process."""

    def __init__(self, report_progress: Any) -> None:
        self._report_progress = report_progress

    def is_set(self) -> bool:
        control = getattr(self._report_progress, "is_set", None)
        return bool(callable(control) and control("cancel"))


def _child_progress(report_progress: Any, **fields: Any) -> None:
    """Translate the server worker callback onto the JSON process channel."""

    report_progress(fields)


def _discovery_process_entry(payload: Any, report_progress: Any) -> dict[str, Any]:
    """Importable child entry for one isolated discovery job."""

    if not isinstance(payload, dict):
        raise TypeError("discovery process payload must be an object")
    outcome = _discovery_worker(
        payload,
        cancel=_ChildCancellation(report_progress),
        report_progress=lambda **fields: _child_progress(report_progress, **fields),
    )
    return {"state": outcome.state, "payload": dict(outcome.payload)}


def _run_process_entry(payload: Any, report_progress: Any) -> dict[str, Any]:
    """Importable child entry for one isolated capture job."""

    if not isinstance(payload, dict):
        raise TypeError("run process payload must be an object")
    workspace = payload.get("workspace")
    request = payload.get("request")
    resume_from = payload.get("resume_from")
    if not isinstance(workspace, str) or not isinstance(request, dict):
        raise TypeError("run process payload is incomplete")
    if resume_from is not None and not isinstance(resume_from, str):
        raise TypeError("resume_from must be a string or null")
    control = getattr(report_progress, "is_set", None)
    outcome = _run_worker(
        workspace,
        request,
        cancel=_ChildCancellation(report_progress),
        report_progress=lambda **fields: _child_progress(report_progress, **fields),
        resume_from=resume_from,
        skip_overhead=((lambda: bool(control("skip_overhead"))) if callable(control) else None),
    )
    return {"state": outcome.state, "payload": dict(outcome.payload)}


def _isolated_job_worker(
    process_entry: Any,
    payload: dict[str, Any],
    *,
    cancel: Any,
    report_progress: Any,
    control_events: dict[str, Any] | None = None,
) -> Any:
    """Run one model operation outside the HTTP process and project its result."""

    from .jobs import JobOutcome
    from .process_worker import ProcessWorkerFailure, run_process_worker

    def relay(progress: Any) -> None:
        if not isinstance(progress, dict):
            return
        stage = progress.get("stage")
        completed = progress.get("completed")
        total = progress.get("total")
        message = progress.get("message")
        report_progress(
            stage=stage if isinstance(stage, str) and stage else "working",
            completed=(
                completed
                if type(completed) is int and not isinstance(completed, bool) and completed >= 0
                else 0
            ),
            total=(
                total if type(total) is int and not isinstance(total, bool) and total >= 0 else None
            ),
            message=message if isinstance(message, str) else "Model worker updated",
        )

    events = dict(control_events or {})
    events["cancel"] = cancel
    result = run_process_worker(
        process_entry,
        payload,
        cancel_event=cancel,
        control_events=events,
        on_progress=relay,
        max_payload_bytes=_MAX_JOB_PAYLOAD_BYTES,
        max_message_bytes=_MAX_JOB_PAYLOAD_BYTES,
    )
    if result.state == "cancelled":
        return JobOutcome({"status": "cancelled"}, "cancelled")
    if result.state != "completed":
        raise ProcessWorkerFailure(result)
    document = result.payload
    if not isinstance(document, dict):
        raise ValueError("model worker returned a non-object result")
    state = document.get("state")
    output = document.get("payload")
    if state not in {"completed", "cancelled", "failed"} or not isinstance(output, dict):
        raise ValueError("model worker returned an invalid job outcome")
    return JobOutcome(output, state)


def _intervention_recipe(payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
    recipe = _intervention_recipe_object(payload)
    return recipe.to_dict(), recipe.fingerprint


def _intervention_recipe_object(payload: dict[str, Any]) -> Any:
    from ..interventions.recipes import InterventionOperation, InterventionRecipe

    return InterventionRecipe(
        operation=InterventionOperation(payload["operation"]),
        targets=tuple(payload["targets"]),
        factor=payload.get("factor"),
        bias=payload.get("bias"),
        alternates=tuple(tuple(item) for item in payload.get("alternates", ())),
    )
