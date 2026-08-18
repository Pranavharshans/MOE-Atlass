"""Strict, versioned contracts for static MoE discovery.

The discovery report is deliberately separate from runtime capture.  A scan
can describe a model-shaped object without claiming that routing was observed
or that any model library is installed.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Self

from pydantic import (
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from .. import __version__
from ..core import (
    CapabilityLabel,
    ComponentKind,
    ComponentManifest,
    ModelManifest,
    StrictManifestModel,
    VersionedManifest,
    make_component_key,
    parse_model_key,
    validate_stable_identifier,
)


class DiscoverySignal(str, Enum):
    """Named evidence sources contributing to a candidate score."""

    CONFIG_FIELD = "config_field"
    PATH_NAME = "path_name"
    CLASS_NAME = "class_name"
    CHILD_STRUCTURE = "child_structure"
    PARAMETER_SHAPE = "parameter_shape"
    INDEXED_EXPERT = "indexed_expert"
    SHARED_NAME = "shared_name"

    def __str__(self) -> str:
        return self.value


class DiscoveryEvidence(StrictManifestModel):
    """One explainable, bounded contribution to a candidate confidence."""

    signal: DiscoverySignal
    detail: StrictStr
    weight: StrictFloat = Field(ge=0, le=1)

    @field_validator("detail")
    @classmethod
    def _detail_is_nonempty(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("discovery evidence detail must be non-empty and trimmed")
        return value


class DiscoveryFacts(StrictManifestModel):
    """Normalized expert and routing facts found during a static scan."""

    expert_count: StrictInt | None = Field(default=None, ge=1)
    routed_top_k: StrictInt | None = Field(default=None, ge=1)
    shared_expert_count: StrictInt | None = Field(default=None, ge=1)
    expert_count_source: StrictStr | None = None
    routed_top_k_source: StrictStr | None = None
    shared_expert_count_source: StrictStr | None = None

    @field_validator(
        "expert_count_source",
        "routed_top_k_source",
        "shared_expert_count_source",
    )
    @classmethod
    def _source_is_nonempty(cls, value: str | None) -> str | None:
        if value is not None and (not value or value != value.strip()):
            raise ValueError("fact source must be non-empty and trimmed")
        return value

    @model_validator(mode="after")
    def _fact_source_pairs(self) -> Self:
        pairs = (
            ("expert_count", self.expert_count, self.expert_count_source),
            ("routed_top_k", self.routed_top_k, self.routed_top_k_source),
            (
                "shared_expert_count",
                self.shared_expert_count,
                self.shared_expert_count_source,
            ),
        )
        for name, value, source in pairs:
            if (value is None) != (source is None):
                raise ValueError(f"{name} and its source must be provided together")
        return self


def _validate_messages(value: list[str], *, field_name: str) -> list[str]:
    for message in value:
        if not message or message != message.strip():
            raise ValueError(f"{field_name} entries must be non-empty and trimmed")
    return value


class DiscoveryCandidate(StrictManifestModel):
    """A confidence-scored semantic candidate produced by static discovery."""

    component_key: StrictStr
    model_key: StrictStr
    kind: ComponentKind
    module_path: StrictStr
    layer_index: StrictInt | None = Field(default=None, ge=0)
    expert_index: StrictInt | None = Field(default=None, ge=0)
    confidence: StrictFloat = Field(ge=0, le=1)
    evidence: list[DiscoveryEvidence] = Field(min_length=1)
    routed: StrictBool | None = None
    shared: StrictBool | None = None
    warnings: list[StrictStr] = Field(default_factory=list)

    @field_validator("model_key")
    @classmethod
    def _model_key_is_canonical(cls, value: str) -> str:
        parse_model_key(value)
        return value

    @field_validator("component_key", "module_path")
    @classmethod
    def _stable_identity(cls, value: str, info: Any) -> str:
        return validate_stable_identifier(value, field_name=info.field_name)

    @field_validator("warnings")
    @classmethod
    def _warnings(cls, value: list[str]) -> list[str]:
        return _validate_messages(value, field_name="warnings")

    @model_validator(mode="after")
    def _identity_and_semantics(self) -> Self:
        evidence_keys = [(item.signal, item.detail, item.weight) for item in self.evidence]
        if len(set(evidence_keys)) != len(evidence_keys):
            raise ValueError("discovery evidence must not contain duplicate identical entries")
        expected_confidence = min(
            1.0, round(sum(evidence_keys_item[2] for evidence_keys_item in evidence_keys), 3)
        )
        if self.confidence != expected_confidence:
            raise ValueError(
                "confidence must equal min(1.0, round(sum(evidence weights), 3)); "
                f"expected {expected_confidence:.3f}"
            )
        if self.confidence < 0.60 and not any(
            "ambiguous" in warning.lower() for warning in self.warnings
        ):
            raise ValueError("confidence below 0.600 requires an ambiguity warning")
        expected_key = make_component_key(
            self.model_key,
            self.kind.value,
            self.module_path,
            layer_index=self.layer_index,
            expert_index=self.expert_index,
        )
        if self.component_key != expected_key:
            raise ValueError(
                "component_key does not match this discovery candidate identity; "
                f"expected {expected_key!r}"
            )
        if self.routed is True and self.shared is True:
            raise ValueError("a discovery candidate cannot be both routed and shared")
        if self.kind is ComponentKind.SHARED_EXPERT and self.shared is not True:
            raise ValueError("shared_expert candidates must set shared=True")
        if self.expert_index is not None and self.kind not in {
            ComponentKind.EXPERT,
            ComponentKind.SHARED_EXPERT,
        }:
            raise ValueError("expert_index is only valid for expert candidates")
        return self


class DiscoveryReport(VersionedManifest):
    """Portable static-discovery output linked to one model manifest."""

    manifest_type: Literal["discovery_report"] = "discovery_report"
    model_key: StrictStr
    model_manifest: ModelManifest
    scanner_version: StrictStr = __version__
    facts: DiscoveryFacts = Field(default_factory=DiscoveryFacts)
    candidates: list[DiscoveryCandidate] = Field(default_factory=list)
    components: list[ComponentManifest] = Field(default_factory=list)
    warnings: list[StrictStr] = Field(default_factory=list)

    @field_validator("model_key")
    @classmethod
    def _model_key_is_canonical(cls, value: str) -> str:
        parse_model_key(value)
        return value

    @field_validator("scanner_version")
    @classmethod
    def _scanner_version_is_nonempty(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("scanner_version must be non-empty and trimmed")
        return value

    @field_validator("warnings")
    @classmethod
    def _warnings(cls, value: list[str]) -> list[str]:
        return _validate_messages(value, field_name="warnings")

    @model_validator(mode="after")
    def _linked_components(self) -> Self:
        if self.model_manifest.model_key != self.model_key:
            raise ValueError("model_key must match model_manifest.model_key")

        candidate_by_key: dict[str, DiscoveryCandidate] = {}
        for candidate in self.candidates:
            if candidate.model_key != self.model_key:
                raise ValueError("every discovery candidate must link to report.model_key")
            if candidate.component_key in candidate_by_key:
                raise ValueError("discovery candidate component_key values must be unique")
            candidate_by_key[candidate.component_key] = candidate

        component_by_key: dict[str, ComponentManifest] = {}
        for component in self.components:
            if component.model_key != self.model_key:
                raise ValueError("every component manifest must link to report.model_key")
            if component.component_key in component_by_key:
                raise ValueError("component manifest component_key values must be unique")
            if component.capabilities != [CapabilityLabel.STRUCTURE]:
                raise ValueError(
                    "static discovery components must have exactly [STRUCTURE] capability"
                )
            component_by_key[component.component_key] = component

        if set(candidate_by_key) != set(component_by_key):
            raise ValueError("discovery candidates and component manifests must have matching keys")
        for key, candidate in candidate_by_key.items():
            component = component_by_key[key]
            if (
                candidate.kind is not component.kind
                or candidate.module_path != component.module_path
                or candidate.layer_index != component.layer_index
                or candidate.expert_index != component.expert_index
            ):
                raise ValueError("candidate and component identity fields must match")
            if candidate.routed != component.routed or candidate.shared != component.shared:
                raise ValueError("candidate and component routed/shared fields must agree")
            if candidate.warnings != component.warnings:
                raise ValueError("candidate and component warnings must agree")
        return self


__all__ = [
    "DiscoveryCandidate",
    "DiscoveryEvidence",
    "DiscoveryFacts",
    "DiscoveryReport",
    "DiscoverySignal",
]
