"""Strict static semantic-adapter contracts and one-shot inspection.

Adapters in this module are deliberately caller-supplied protocol objects.  No
adapter is selected, imported, registered, or executed outside the explicit
``inspect_static_adapter`` call.  The boundary only validates static
STRUCTURE evidence; runtime loading and semantic capture belong to later
features.
"""

from __future__ import annotations

import math
from inspect import getattr_static
from typing import Literal, Protocol

from pydantic import Field, StrictFloat, StrictStr, field_validator, model_validator

from ..core import (
    CapabilityLabel,
    CaptureSource,
    ModelManifest,
    VersionedManifest,
)
from ..discovery import DiscoveryReport

_ADAPTER_STAGES = frozenset({"descriptor", "detect", "discover"})


def _trimmed(value: str, *, field_name: str) -> str:
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be non-empty and trimmed")
    return value


def _sorted_unique(values: tuple[str, ...], *, field_name: str) -> tuple[str, ...]:
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must contain unique values")
    expected = tuple(sorted(values))
    if values != expected:
        raise ValueError(f"{field_name} must be sorted lexicographically")
    return values


def _messages(values: tuple[str, ...], *, field_name: str) -> tuple[str, ...]:
    return tuple(_trimmed(value, field_name=field_name) for value in values)


class AdapterDescriptor(VersionedManifest):
    """Portable identity and compatibility notes for one adapter."""

    manifest_type: Literal["adapter_descriptor"] = "adapter_descriptor"
    name: StrictStr
    version: StrictStr
    architecture_families: tuple[StrictStr, ...] = Field(min_length=1)
    compatibility_notes: tuple[StrictStr, ...] = ()

    @field_validator("name", "version")
    @classmethod
    def _identity_text(cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", "adapter identity")
        return _trimmed(value, field_name=field_name)

    @field_validator("architecture_families", "compatibility_notes")
    @classmethod
    def _canonical_lists(cls, value: tuple[str, ...], info: object) -> tuple[str, ...]:
        field_name = getattr(info, "field_name", "adapter values")
        trimmed = _messages(value, field_name=field_name)
        return _sorted_unique(trimmed, field_name=field_name)


class AdapterDetection(VersionedManifest):
    """Static evidence strength, not a probability or certification claim."""

    manifest_type: Literal["adapter_detection"] = "adapter_detection"
    score: StrictFloat = Field(ge=0.0, le=1.0)
    evidence: tuple[StrictStr, ...] = ()
    warnings: tuple[StrictStr, ...] = ()

    @field_validator("score", mode="before")
    @classmethod
    def _strict_finite_score(cls, value: object) -> float:
        if type(value) is not float:
            raise TypeError("score must be a finite float, not an integer or boolean")
        if not math.isfinite(value):
            raise ValueError("score must be finite")
        return value

    @field_validator("evidence", "warnings")
    @classmethod
    def _deterministic_messages(cls, value: tuple[str, ...], info: object) -> tuple[str, ...]:
        field_name = getattr(info, "field_name", "adapter messages")
        messages = _messages(value, field_name=field_name)
        if len(set(messages)) != len(messages):
            raise ValueError(f"{field_name} must contain unique values")
        return tuple(sorted(messages))

    @model_validator(mode="after")
    def _evidence_invariants(self) -> AdapterDetection:
        if self.score > 0.0 and not self.evidence:
            raise ValueError("positive adapter detection requires evidence")
        if self.score == 0.0 and not self.warnings:
            raise ValueError("zero adapter detection requires a warning")
        return self


class AdapterInspection(VersionedManifest):
    """The final validated static inspection publication."""

    manifest_type: Literal["adapter_inspection"] = "adapter_inspection"
    descriptor: AdapterDescriptor
    detection: AdapterDetection
    report: DiscoveryReport

    @model_validator(mode="after")
    def _positive_detection(self) -> AdapterInspection:
        if self.detection.score <= 0.0:
            raise ValueError("adapter inspection requires positive detection")
        _validate_report_components(self.report, self.descriptor)
        return self


class StaticSemanticAdapter(Protocol):
    """Caller-supplied protocol for static adapter inspection."""

    @property
    def descriptor(self) -> AdapterDescriptor: ...

    def detect(self, model: object, config: object) -> AdapterDetection: ...

    def discover(self, model: object, model_manifest: ModelManifest) -> DiscoveryReport: ...


class AdapterContractError(ValueError):
    """Safe contract failure tagged with one fixed inspection stage."""

    def __init__(self, stage: str) -> None:
        if stage not in _ADAPTER_STAGES:
            raise ValueError("adapter error stage must be descriptor, detect, or discover")
        self.stage = stage
        super().__init__(f"static adapter {stage} contract validation failed")


class AdapterExecutionError(RuntimeError):
    """Safe ordinary-exception wrapper tagged with one fixed stage."""

    def __init__(self, stage: str) -> None:
        if stage not in _ADAPTER_STAGES:
            raise ValueError("adapter error stage must be descriptor, detect, or discover")
        self.stage = stage
        super().__init__(f"static adapter {stage} execution failed")


def _contract(stage: str) -> AdapterContractError:
    return AdapterContractError(stage)


def _revalidate_descriptor(value: object) -> AdapterDescriptor:
    if type(value) is not AdapterDescriptor:
        raise _contract("descriptor")
    try:
        return AdapterDescriptor.model_validate(value.model_dump(mode="json"))
    except Exception as exc:
        raise _contract("descriptor") from exc


def _revalidate_detection(value: object) -> AdapterDetection:
    if type(value) is not AdapterDetection:
        raise _contract("detect")
    try:
        return AdapterDetection.model_validate(value.model_dump(mode="json"))
    except Exception as exc:
        raise _contract("detect") from exc


def _revalidate_report(value: object) -> DiscoveryReport:
    if type(value) is not DiscoveryReport:
        raise _contract("discover")
    try:
        return DiscoveryReport.model_validate(value.model_dump(mode="json"))
    except Exception as exc:
        raise _contract("discover") from exc


def _preflight_adapter_members(adapter: object) -> None:
    """Check protocol member presence without invoking descriptors."""

    for stage in ("descriptor", "detect", "discover"):
        try:
            getattr_static(adapter, stage)
        except Exception as exc:
            raise _contract(stage) from exc


def _validate_report_components(
    report: DiscoveryReport,
    descriptor: AdapterDescriptor,
) -> None:
    if report.model_key != report.model_manifest.model_key:
        raise _contract("discover")
    if not report.components:
        if not report.warnings:
            raise _contract("discover")
        return
    for component in report.components:
        if component.capabilities != [CapabilityLabel.STRUCTURE]:
            raise _contract("discover")
        capture = component.capture
        if (
            capture is None
            or capture.source is not CaptureSource.STATIC_STRUCTURE
            or capture.verified is not False
            or capture.adapter != descriptor.name
            or capture.adapter_version != descriptor.version
        ):
            raise _contract("discover")


def inspect_static_adapter(
    adapter: StaticSemanticAdapter,
    model: object,
    config: object,
    model_manifest: ModelManifest,
) -> AdapterInspection:
    """Run and validate one caller-supplied static adapter exactly once."""

    if not isinstance(model_manifest, ModelManifest):
        raise _contract("descriptor")
    try:
        ModelManifest.model_validate(model_manifest.model_dump(mode="json"))
    except Exception as exc:
        raise _contract("descriptor") from exc
    _preflight_adapter_members(adapter)

    try:
        descriptor_value = getattr(adapter, "descriptor")
    except Exception as exc:
        raise AdapterExecutionError("descriptor") from exc
    descriptor = _revalidate_descriptor(descriptor_value)

    try:
        detect = getattr(adapter, "detect")
    except Exception as exc:
        raise AdapterExecutionError("detect") from exc
    if not callable(detect):
        raise _contract("detect")
    try:
        detection_value = detect(model, config)
    except Exception as exc:
        raise AdapterExecutionError("detect") from exc
    detection = _revalidate_detection(detection_value)
    if detection.score <= 0.0:
        raise _contract("detect")

    try:
        discover = getattr(adapter, "discover")
    except Exception as exc:
        raise AdapterExecutionError("discover") from exc
    if not callable(discover):
        raise _contract("discover")
    try:
        report_value = discover(model, model_manifest)
    except Exception as exc:
        raise AdapterExecutionError("discover") from exc
    report = _revalidate_report(report_value)
    if report.model_key != model_manifest.model_key or report.model_manifest != model_manifest:
        raise _contract("discover")
    try:
        report.to_json()
    except Exception as exc:
        raise _contract("discover") from exc
    _validate_report_components(report, descriptor)

    try:
        return AdapterInspection(
            descriptor=descriptor,
            detection=detection,
            report=report,
        )
    except Exception as exc:
        raise _contract("discover") from exc


__all__ = [
    "AdapterContractError",
    "AdapterDescriptor",
    "AdapterDetection",
    "AdapterExecutionError",
    "AdapterInspection",
    "StaticSemanticAdapter",
    "inspect_static_adapter",
]
