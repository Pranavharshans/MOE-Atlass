"""One-shot resolved runtime loading followed by static MoE discovery.

This module is intentionally a small orchestration boundary.  It does not
load optional dependencies itself, add a loader registry, run inference, or
capture tensors: the selected Feature 9 loader owns runtime policy and the
existing discovery scanner remains read-only and STRUCTURE-only.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from ..core import CapabilityLabel, ModelManifest
from ..discovery import DiscoveryReport
from ..discovery import scan as discovery_scan
from ..loading import HuggingFaceSource, LoadingPlan, LocalSource
from .contracts import (
    LoadedModel,
    PendingRuntimeCleanup,
    RuntimeCleanupError,
    RuntimeLoadError,
    RuntimeValidationError,
    _add_cleanup_note,
    _attach_pending_cleanup,
)
from .model_loader import load_huggingface, load_local

_Observation = TypeVar("_Observation")


def _select_loader(plan: LoadingPlan) -> Callable[[LoadingPlan], LoadedModel]:
    """Select the only two supported source loaders without a registry."""

    if not isinstance(plan, LoadingPlan):
        raise TypeError("plan must be a LoadingPlan")
    if isinstance(plan.source, HuggingFaceSource):
        return load_huggingface
    if isinstance(plan.source, LocalSource):
        return load_local
    raise RuntimeLoadError(
        "load_and_scan supports only HuggingFaceSource and LocalSource; "
        "call load_instance() or load_custom() explicitly, then discovery.scan() "
        "manually for already-instantiated or custom-loader sources"
    )


def _validate_report(report: object, manifest: ModelManifest) -> DiscoveryReport:
    """Validate the scanner result and bind it to the exact loaded manifest."""

    try:
        payload = report.model_dump(mode="json") if isinstance(report, DiscoveryReport) else report
        validated = DiscoveryReport.model_validate(payload)
    except Exception as exc:
        raise RuntimeValidationError(
            "static discovery did not return a valid DiscoveryReport"
        ) from exc
    if validated.model_key != manifest.model_key or validated.model_manifest != manifest:
        raise RuntimeValidationError(
            "static discovery report does not match the loaded model manifest"
        )
    if any(
        component.capabilities != [CapabilityLabel.STRUCTURE] for component in validated.components
    ):
        raise RuntimeValidationError("load_and_scan accepts only STRUCTURE discovery capabilities")
    try:
        validated.to_json()
    except Exception as exc:
        raise RuntimeValidationError("static discovery report is not JSON-serializable") from exc
    return validated


def _as_cleanup_error(error: BaseException) -> RuntimeCleanupError:
    if isinstance(error, RuntimeCleanupError):
        return error
    return RuntimeCleanupError((error,))


def _close_after_body_error(loaded: Any, original: BaseException) -> None:
    """Close while preserving a scan/control-flow error as the primary error."""

    try:
        loaded.close()
    except BaseException as cleanup_exception:
        cleanup_error = _as_cleanup_error(cleanup_exception)
        pending = PendingRuntimeCleanup(loaded.close)
        _attach_pending_cleanup(original, pending)
        _add_cleanup_note(original, cleanup_error)


def _close_after_success(loaded: Any) -> None:
    """Close before publishing a report, exposing retryable failures."""

    try:
        loaded.close()
    except BaseException as cleanup_exception:
        cleanup_error = _as_cleanup_error(cleanup_exception)
        pending = PendingRuntimeCleanup(loaded.close)
        _attach_pending_cleanup(cleanup_error, pending)
        raise cleanup_error


def load_scan_and_observe(
    plan: LoadingPlan,
    observer: Callable[[object, DiscoveryReport], _Observation],
) -> tuple[DiscoveryReport, _Observation]:
    """Load once, scan, and run one read-only observer before cleanup.

    The exact ``plan`` object is handed to the selected loader.  The loaded
    model and manifest are handed unchanged to ``discovery.scan`` and then to
    the supplied observer with the validated report. Cleanup is attempted for
    every outcome, including ``BaseException`` control-flow exits; a cleanup
    failure never replaces a scan/observer error or publishes partial evidence.
    """

    if not callable(observer):
        raise TypeError("observer must be callable")
    loader = _select_loader(plan)
    loaded = loader(plan)
    try:
        model = loaded.model
        manifest = loaded.manifest
        if not isinstance(manifest, ModelManifest):
            raise RuntimeValidationError("runtime loader did not return a ModelManifest")
        report = _validate_report(discovery_scan(model, manifest), manifest)
        observation = observer(model, report)
    except BaseException as body_error:
        _close_after_body_error(loaded, body_error)
        raise
    _close_after_success(loaded)
    return report, observation


def load_and_scan(plan: LoadingPlan) -> DiscoveryReport:
    """Load a resolved HF/local source once, scan it, then release the runtime."""

    report, _observation = load_scan_and_observe(plan, lambda _model, _report: None)
    return report


__all__ = ["load_and_scan", "load_scan_and_observe"]
