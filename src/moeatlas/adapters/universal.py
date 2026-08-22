"""Universal structure inspections for family-blind analysis.

A :class:`UniversalRoutingInspection` is a strict, canonical value document
derived from one static ``[STRUCTURE]`` discovery report without any certified
adapter.  It carries exactly what bounded routing-load analysis needs — model
identity, architecture-family provenance, the rectangular routed expert axes,
routed top-k, and a fixed ``packed``-equivalent layout tag — and marks its own
provenance as ``universal`` so downstream artifacts never present it as
certified adapter evidence.

The document performs no model loading, no network access, and imports no
model-runtime or storage dependency.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import Field, StrictInt, StrictStr, field_validator, model_validator

from ..core import (
    ComponentKind,
    StrictManifestModel,
    VersionedManifest,
    parse_component_key,
    parse_model_key,
    stable_digest,
)
from ..discovery import DiscoveryReport

UNIVERSAL_ROUTING_INSPECTION_SCHEMA_VERSION = "1.0"

_UNIVERSAL_LAYOUT = "packed"
_AXES_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_STRUCTURAL_KINDS = frozenset(
    {
        ComponentKind.MOE_LAYER,
        ComponentKind.ROUTER,
        ComponentKind.EXPERT,
        ComponentKind.SHARED_EXPERT,
    }
)


def _trimmed(value: str, *, field_name: str) -> str:
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be non-empty and trimmed")
    return value


def _axes_digest(
    layer_indices: tuple[int, ...],
    layer_keys: tuple[str, ...],
    expert_keys: tuple[tuple[str, ...], ...],
    routed_top_k: int,
) -> str:
    return "sha256:" + stable_digest(
        {
            "expert_keys": [list(row) for row in expert_keys],
            "layer_indices": list(layer_indices),
            "layer_keys": list(layer_keys),
            "routed_top_k": routed_top_k,
        }
    )


class UniversalLayerUniverse(StrictManifestModel):
    """The routed expert universe of exactly one layer."""

    layer_index: StrictInt = Field(ge=0)
    layer_key: StrictStr
    expert_keys: tuple[StrictStr, ...] = Field(min_length=1)

    @field_validator("layer_key")
    @classmethod
    def _canonical_layer_key(cls, value: str) -> str:
        parse_component_key(value)
        return value

    @field_validator("expert_keys")
    @classmethod
    def _canonical_expert_keys(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for key in value:
            parse_component_key(key)
        if len(set(value)) != len(value):
            raise ValueError("expert_keys must be unique within one layer")
        return value


class UniversalRoutingInspection(VersionedManifest):
    """Canonical analysis-ready topology derived from static discovery."""

    manifest_type: Literal["universal_routing_inspection"] = (
        "universal_routing_inspection"
    )
    provenance: Literal["universal"] = "universal"
    model_key: StrictStr
    architecture_families: tuple[StrictStr, ...] = Field(min_length=1)
    scanner_version: StrictStr
    layout: Literal["packed"] = _UNIVERSAL_LAYOUT
    routed_top_k: StrictInt = Field(ge=1)
    expert_count: StrictInt = Field(ge=1)
    layers: tuple[UniversalLayerUniverse, ...] = Field(min_length=1)
    axes_digest: StrictStr

    @field_validator("model_key")
    @classmethod
    def _canonical_model_key(cls, value: str) -> str:
        parse_model_key(value)
        return value

    @field_validator("architecture_families")
    @classmethod
    def _canonical_families(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        trimmed = tuple(
            _trimmed(family, field_name="architecture_families") for family in value
        )
        if len(set(trimmed)) != len(trimmed):
            raise ValueError("architecture_families must contain unique values")
        expected = tuple(sorted(trimmed))
        if trimmed != expected:
            raise ValueError("architecture_families must be sorted lexicographically")
        return expected

    @field_validator("scanner_version")
    @classmethod
    def _scanner_version_is_nonempty(cls, value: str) -> str:
        return _trimmed(value, field_name="scanner_version")

    @field_validator("layers")
    @classmethod
    def _canonical_layers(
        cls, value: tuple[UniversalLayerUniverse, ...]
    ) -> tuple[UniversalLayerUniverse, ...]:
        indices = [layer.layer_index for layer in value]
        if len(set(indices)) != len(indices):
            raise ValueError("layer_index values must be unique")
        if sorted(indices) != list(range(len(value))):
            raise ValueError("layer_index values must be contiguous from zero")
        keys = [layer.layer_key for layer in value]
        if len(set(keys)) != len(keys):
            raise ValueError("layer_key values must be unique across layers")
        flat_expert_keys = [key for layer in value for key in layer.expert_keys]
        if len(set(flat_expert_keys)) != len(flat_expert_keys):
            raise ValueError("expert_keys must be globally unique across layers")
        return value

    @model_validator(mode="after")
    def _invariants(self) -> UniversalRoutingInspection:
        counts = {len(layer.expert_keys) for layer in self.layers}
        if len(counts) != 1:
            raise ValueError("expert universes must be rectangular across layers")
        expert_count = next(iter(counts))
        if self.expert_count != expert_count:
            raise ValueError("expert_count does not match the layer expert axes")
        if self.routed_top_k > self.expert_count:
            raise ValueError("routed_top_k cannot exceed the expert axis")
        expected = _axes_digest(
            tuple(layer.layer_index for layer in self.layers),
            tuple(layer.layer_key for layer in self.layers),
            tuple(layer.expert_keys for layer in self.layers),
            self.routed_top_k,
        )
        if not isinstance(self.axes_digest, str) or _AXES_DIGEST.fullmatch(
            self.axes_digest
        ) is None:
            raise ValueError("axes_digest must be sha256:<64hex>")
        if self.axes_digest != expected:
            raise ValueError("axes_digest does not match the routing axes")
        return self


def build_universal_inspection(report: object) -> UniversalRoutingInspection:
    """Derive one universal inspection from an exact fresh discovery report.

    The walk consumes only model-neutral structural evidence — semantic kinds,
    routing flags, and integer layer/expert indices — and never inspects
    adapter names or module-path conventions.  The layout tag is the fixed
    ``packed`` equivalent because generic scans do not certify a native
    token-to-expert indexing layout.
    """

    if type(report) is not DiscoveryReport:
        raise TypeError("inspection source must be an exact DiscoveryReport")
    try:
        fresh = DiscoveryReport.model_validate(report.model_dump(mode="json"))
        if type(fresh) is not DiscoveryReport or fresh is report:
            raise TypeError("discovery report revalidation returned an unexpected type")
    except (TypeError, ValueError):
        raise
    except Exception as exc:
        raise ValueError("discovery report revalidation failed") from exc
    try:
        return _derive_universal_inspection(fresh)
    except (TypeError, ValueError):
        raise
    except Exception as exc:
        raise ValueError("universal inspection derivation failed") from exc


def _derive_universal_inspection(
    report: DiscoveryReport,
) -> UniversalRoutingInspection:
    facts = report.facts
    routed_top_k = facts.routed_top_k
    fact_expert_count = facts.expert_count
    if (
        type(routed_top_k) is not int
        or isinstance(routed_top_k, bool)
        or type(fact_expert_count) is not int
        or isinstance(fact_expert_count, bool)
    ):
        raise ValueError("discovery facts are not complete routing facts")

    components = tuple(report.components)
    routers = [
        component for component in components if component.kind is ComponentKind.ROUTER
    ]
    if not routers:
        raise ValueError("discovery report has no router universe")
    moe_layers = [
        component for component in components if component.kind is ComponentKind.MOE_LAYER
    ]
    if len(moe_layers) != len(routers):
        raise ValueError("discovery MoE layers do not exactly match routers")
    for component in components:
        if component.kind in _STRUCTURAL_KINDS:
            if component.layer_index is None or type(component.layer_index) is not int:
                raise ValueError("routing component layer index is not exact")
            parse_component_key(component.component_key)

    layer_records: list[tuple[int, str, tuple[str, ...]]] = []
    seen_layer_indices: set[int] = set()
    for router in routers:
        if router.layer_index is None or router.layer_index < 0:
            raise ValueError("router layer index is not exact")
        if router.layer_index in seen_layer_indices:
            raise ValueError("router layer indices are not unique")
        seen_layer_indices.add(router.layer_index)
        same_layers = [
            component
            for component in components
            if component.kind is ComponentKind.MOE_LAYER
            and component.layer_index == router.layer_index
        ]
        if len(same_layers) != 1:
            raise ValueError("router must bind one exact MoE layer")
        layer = same_layers[0]
        experts = [
            component
            for component in components
            if component.kind is ComponentKind.EXPERT
            and component.layer_index == router.layer_index
        ]
        indices = [component.expert_index for component in experts]
        if any(type(index) is not int for index in indices) or sorted(indices) != list(
            range(len(experts))
        ):
            raise ValueError("layer expert indices are not contiguous")
        if any(
            component.routed is not True or component.shared is True for component in experts
        ):
            raise ValueError("layer expert universe contains shared or unrouted experts")
        if len(experts) != fact_expert_count:
            raise ValueError("layer expert universe does not match discovery facts")
        experts.sort(key=lambda component: component.expert_index)
        expert_keys = tuple(component.component_key for component in experts)
        layer_records.append((router.layer_index, layer.component_key, expert_keys))
    layer_records.sort(key=lambda record: record[0])
    layer_indices = tuple(record[0] for record in layer_records)
    layer_keys = tuple(record[1] for record in layer_records)
    expert_keys = tuple(record[2] for record in layer_records)
    if layer_indices != tuple(range(len(layer_indices))):
        raise ValueError("discovery layer indices are not contiguous")
    flat_expert_keys = tuple(key for row in expert_keys for key in row)
    if len(set(flat_expert_keys)) != len(flat_expert_keys):
        raise ValueError("discovery expert keys are not globally unique")
    if routed_top_k > len(expert_keys[0]):
        raise ValueError("discovery routed top-k exceeds the layer expert count")
    return UniversalRoutingInspection(
        model_key=report.model_key,
        architecture_families=(report.model_manifest.architecture,),
        scanner_version=report.scanner_version,
        routed_top_k=routed_top_k,
        expert_count=len(expert_keys[0]),
        layers=tuple(
            UniversalLayerUniverse(
                layer_index=index,
                layer_key=key,
                expert_keys=row,
            )
            for index, key, row in zip(layer_indices, layer_keys, expert_keys, strict=True)
        ),
        axes_digest=_axes_digest(layer_indices, layer_keys, expert_keys, routed_top_k),
    )


__all__ = [
    "UNIVERSAL_ROUTING_INSPECTION_SCHEMA_VERSION",
    "UniversalLayerUniverse",
    "UniversalRoutingInspection",
    "build_universal_inspection",
]
