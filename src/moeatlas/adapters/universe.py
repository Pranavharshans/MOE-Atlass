"""Adapter-published routing universe contracts.

A :class:`RoutingUniverse` is the versioned, model-neutral description of one
model's routed expert topology, published by an adapter from its static
inspection.  It replaces the hidden rectangular assumptions of earlier
analysis (one global expert count and top-k, contiguous indices, one of two
hard-coded layouts) with explicit per-layer records: every layer names its own
expert set, native expert indices, routed top-k, and shared experts, and the
layout tag is declared by the adapter rather than drawn from a central
allowlist.  Non-rectangular MoE families are therefore first-class, and
rectangular consumers reduce a universe through
:func:`project_rectangular_universe` — an explicit projection, never a silent
invariant.

This module performs no model loading, no network access, and imports no
model-runtime dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, TypeAlias

from pydantic import Field, StrictInt, StrictStr, field_validator, model_validator

from ..core import (
    CaptureSource,
    ComponentKind,
    StrictManifestModel,
    VersionedManifest,
    parse_component_key,
    parse_model_key,
)
from .contracts import AdapterInspection

ROUTING_UNIVERSE_SCHEMA_VERSION = "1.0"

_UNIVERSE_STAGES = frozenset({"dependency", "publication", "projection"})
_LAYOUT_KEY = "layout"
_TOP_K_KEY = "routed_top_k"
_ROUTER_METADATA_KEYS = frozenset({_LAYOUT_KEY, _TOP_K_KEY})
_MAX_LAYOUT_LENGTH = 64
_STRUCTURAL_KINDS = frozenset(
    {
        ComponentKind.MOE_LAYER,
        ComponentKind.ROUTER,
        ComponentKind.EXPERT,
        ComponentKind.SHARED_EXPERT,
    }
)

# One derived layer during publication: (layer_index, moe_layer_key,
# router_key, expert_keys, expert_indices, routed_top_k, shared_expert_keys).
_LayerRecord: TypeAlias = tuple[
    int, str, str, tuple[str, ...], tuple[int, ...] | None, int, tuple[str, ...]
]


class RoutingUniverseError(ValueError):
    """Safe fixed-stage failure for routing universe publication/projection."""

    def __init__(
        self,
        stage: str,
        message: str | None = None,
        *,
        cause: BaseException | None = None,
    ) -> None:
        if type(stage) is not str or stage not in _UNIVERSE_STAGES:
            raise ValueError(f"unsupported routing universe stage: {stage!r}")
        self.stage = stage
        text = f"routing universe failed at {stage}"
        if message is not None:
            text = f"{text}: {message}"
        super().__init__(text)
        if cause is not None:
            self.__cause__ = cause


def _trimmed(value: str, *, field_name: str) -> str:
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be non-empty and trimmed")
    return value


class LayerRoutingUniverse(StrictManifestModel):
    """The routed expert universe of exactly one layer."""

    layer_index: StrictInt = Field(ge=0)
    moe_layer_key: StrictStr
    router_key: StrictStr
    expert_keys: tuple[StrictStr, ...] = Field(min_length=1)
    # Parallel to expert_keys: expert_indices[i] is the adapter's native
    # identifier for expert_keys[i].  Native identifiers may be sparse or
    # unordered; they carry no positional meaning of their own.
    expert_indices: tuple[StrictInt, ...] | None = None
    routed_top_k: StrictInt = Field(ge=1)
    shared_expert_keys: tuple[StrictStr, ...] = ()

    @field_validator("moe_layer_key", "router_key", "expert_keys", "shared_expert_keys")
    @classmethod
    def _canonical_keys(cls, value: object, info: object) -> object:
        if isinstance(value, str):
            parse_component_key(value)
            return value
        for key in value:  # type: ignore[union-attr]
            parse_component_key(key)
        return value

    @field_validator("expert_keys", "shared_expert_keys")
    @classmethod
    def _sorted_unique(cls, value: tuple[str, ...], info: object) -> tuple[str, ...]:
        field_name = getattr(info, "field_name", "keys")
        if list(value) != sorted(value) or len(set(value)) != len(value):
            raise ValueError(f"{field_name} must be sorted ascending and unique")
        return value

    @field_validator("expert_indices")
    @classmethod
    def _exact_indices(cls, value: tuple[int, ...] | None) -> tuple[int, ...] | None:
        if value is None:
            return value
        if any(type(index) is not int or index < 0 for index in value):
            raise ValueError("expert_indices must be non-negative exact integers")
        if len(set(value)) != len(value):
            raise ValueError("expert_indices must be unique")
        return value

    @model_validator(mode="after")
    def _invariants(self) -> LayerRoutingUniverse:
        if self.routed_top_k > len(self.expert_keys):
            raise ValueError("routed_top_k must not exceed the layer expert count")
        if self.expert_indices is not None and len(self.expert_indices) != len(
            self.expert_keys
        ):
            raise ValueError("expert_indices must be parallel to expert_keys")
        if set(self.shared_expert_keys) & set(self.expert_keys):
            raise ValueError("shared_expert_keys must be disjoint from expert_keys")
        return self


class RoutingUniverse(VersionedManifest):
    """Versioned per-layer routing topology published by one adapter."""

    manifest_type: ClassVar[str] = "routing_universe"

    model_key: StrictStr
    adapter_name: StrictStr
    adapter_version: StrictStr
    layout: StrictStr
    layers: tuple[LayerRoutingUniverse, ...] = Field(min_length=1)

    @field_validator("model_key")
    @classmethod
    def _canonical_model_key(cls, value: str) -> str:
        parse_model_key(value)
        return value

    @field_validator("adapter_name", "adapter_version")
    @classmethod
    def _identity_text(cls, value: str, info: object) -> str:
        return _trimmed(value, field_name=getattr(info, "field_name", "adapter identity"))

    @field_validator("layout")
    @classmethod
    def _layout_vocabulary(cls, value: str) -> str:
        _trimmed(value, field_name="layout")
        if len(value) > _MAX_LAYOUT_LENGTH:
            raise ValueError(f"layout must be at most {_MAX_LAYOUT_LENGTH} characters")
        if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
            raise ValueError("layout must not contain control characters")
        return value

    @property
    def layer_indices(self) -> tuple[int, ...]:
        """The sorted layer indices covered by this universe."""

        return tuple(layer.layer_index for layer in self.layers)

    @model_validator(mode="after")
    def _invariants(self) -> RoutingUniverse:
        indices = [layer.layer_index for layer in self.layers]
        if indices != sorted(indices):
            raise ValueError("layers must be sorted ascending by layer_index")
        if len(set(indices)) != len(indices):
            raise ValueError("layer_index values must be unique")
        router_keys = [layer.router_key for layer in self.layers]
        if len(set(router_keys)) != len(router_keys):
            raise ValueError("router_key values must be unique across layers")
        moe_keys = [layer.moe_layer_key for layer in self.layers]
        if len(set(moe_keys)) != len(moe_keys):
            raise ValueError("moe_layer_key values must be unique across layers")
        flat_expert_keys = [
            key for layer in self.layers for key in layer.expert_keys
        ]
        if len(set(flat_expert_keys)) != len(flat_expert_keys):
            raise ValueError("expert_keys must be globally unique across layers")
        return self


@dataclass(frozen=True, slots=True)
class RectangularProjection:
    """The explicit rectangular reduction of a routing universe."""

    expert_count: int
    routed_top_k: int
    layer_indices: tuple[int, ...]
    layer_keys: tuple[str, ...]
    expert_keys: tuple[tuple[str, ...], ...]


def publish_routing_universe(inspection: object) -> RoutingUniverse:
    """Derive the versioned routing universe from one static inspection.

    Publication is family-blind: it consumes only the model-neutral structural
    invariants of the inspection (semantic kinds, routing flags, capture
    provenance) and never inspects adapter names, module-path conventions, or
    configuration fields.  The layout tag and per-layer top-k provenance are
    taken from the router captures exactly as the adapter published them.
    """

    if type(inspection) is not AdapterInspection:
        raise RoutingUniverseError(
            "dependency", "inspection must be an exact AdapterInspection"
        )
    try:
        fresh = AdapterInspection.model_validate(inspection.model_dump(mode="json"))
        if type(fresh) is not AdapterInspection or fresh is inspection:
            raise ValueError("inspection revalidation returned an unexpected type")
    except RoutingUniverseError:
        raise
    except Exception as exc:
        raise RoutingUniverseError(
            "publication", "inspection revalidation failed", cause=exc
        ) from exc
    try:
        return _derive_universe(fresh)
    except RoutingUniverseError:
        raise
    except Exception as exc:
        raise RoutingUniverseError("publication", str(exc), cause=exc) from exc


def project_rectangular_universe(universe: object) -> RectangularProjection:
    """Explicitly reduce a universe to the rectangular analysis form.

    A universe is rectangular when every layer exposes the same expert count
    and routed top-k, layer indices are contiguous from zero, and every layer
    declares contiguous native expert indices from zero.  Anything else —
    variable expert universes, variable top-k schedules, non-contiguous
    indices — has no rectangular projection and fails here instead of being
    silently reshaped.
    """

    if type(universe) is not RoutingUniverse:
        raise TypeError(
            f"universe must be an exact RoutingUniverse, got {type(universe).__name__}"
        )
    layers = universe.layers
    expert_count = len(layers[0].expert_keys)
    routed_top_k = layers[0].routed_top_k
    for layer in layers:
        if len(layer.expert_keys) != expert_count:
            raise RoutingUniverseError(
                "projection",
                "layer expert counts vary; the universe is not rectangular",
            )
        if layer.routed_top_k != routed_top_k:
            raise RoutingUniverseError(
                "projection",
                "layer routed top-k schedules vary; the universe is not rectangular",
            )
    if universe.layer_indices != tuple(range(len(layers))):
        raise RoutingUniverseError(
            "projection", "layer indices are not contiguous from zero"
        )
    for layer in layers:
        if layer.expert_indices is None:
            raise RoutingUniverseError(
                "projection",
                f"layer {layer.layer_index} does not declare native expert indices",
            )
        if sorted(layer.expert_indices) != list(range(expert_count)):
            raise RoutingUniverseError(
                "projection",
                f"layer {layer.layer_index} native expert indices are not contiguous",
            )
    return RectangularProjection(
        expert_count=expert_count,
        routed_top_k=routed_top_k,
        layer_indices=universe.layer_indices,
        layer_keys=tuple(layer.moe_layer_key for layer in layers),
        expert_keys=tuple(layer.expert_keys for layer in layers),
    )


def _derive_universe(inspection: AdapterInspection) -> RoutingUniverse:
    descriptor = inspection.descriptor
    if (
        type(descriptor.name) is not str
        or type(descriptor.version) is not str
        or not descriptor.name
        or not descriptor.version
        or not descriptor.architecture_families
    ):
        raise ValueError("inspection descriptor identity is not complete")
    parse_model_key(inspection.report.model_key)
    model_key = inspection.report.model_key
    facts = inspection.report.facts
    fact_expert_count = _strict_fact(facts.expert_count, "expert_count")
    fact_top_k = _strict_fact(facts.routed_top_k, "routed_top_k")
    fact_shared_count = _strict_fact(facts.shared_expert_count, "shared_expert_count")

    components = tuple(inspection.report.components)
    routers = [
        component for component in components if component.kind is ComponentKind.ROUTER
    ]
    if not routers:
        raise ValueError("inspection has no router universe")
    moe_layers = [
        component for component in components if component.kind is ComponentKind.MOE_LAYER
    ]
    if len(moe_layers) != len(routers):
        raise ValueError("inspection MoE layers do not exactly match routers")
    for component in components:
        if component.kind in _STRUCTURAL_KINDS:
            if component.layer_index is None or type(component.layer_index) is not int:
                raise ValueError("routing component layer index is not exact")
            parse_component_key(component.component_key)

    layer_records: list[_LayerRecord] = []
    seen_layers: set[int] = set()
    layouts: set[str] = set()
    for router in routers:
        if router.layer_index is None or router.layer_index < 0:
            raise ValueError("router layer index is not exact")
        if router.layer_index in seen_layers:
            raise ValueError("router layer indices are not unique")
        seen_layers.add(router.layer_index)
        capture = router.capture
        if capture is None or not _ROUTER_METADATA_KEYS.issuperset(capture.metadata):
            raise ValueError("router layout provenance is not exact")
        if _LAYOUT_KEY not in capture.metadata:
            raise ValueError("router layout provenance is not exact")
        if (
            capture.source is not CaptureSource.STATIC_STRUCTURE
            or capture.adapter != descriptor.name
            or capture.adapter_version != descriptor.version
            or capture.verified is not False
        ):
            raise ValueError("router provenance is not exact static structure evidence")
        layout = capture.metadata[_LAYOUT_KEY]
        if type(layout) is not str:
            raise ValueError("router layout must be a string tag")
        _trimmed(layout, field_name="router layout")
        if len(layout) > _MAX_LAYOUT_LENGTH:
            raise ValueError("router layout exceeds the maximum tag length")
        layouts.add(layout)
        metadata_top_k = capture.metadata.get(_TOP_K_KEY)
        if metadata_top_k is not None and (
            type(metadata_top_k) is not int
            or isinstance(metadata_top_k, bool)
            or metadata_top_k < 1
        ):
            raise ValueError("router routed_top_k metadata must be a positive integer")
        same_layers = [
            component
            for component in components
            if component.kind is ComponentKind.MOE_LAYER
            and component.layer_index == router.layer_index
        ]
        if len(same_layers) != 1:
            raise ValueError("router must bind one exact MoE layer")
        layer = same_layers[0]
        if layer.routed is not None or layer.shared is not None:
            raise ValueError("MoE layer routing flags are not neutral")

        experts = [
            component
            for component in components
            if component.kind is ComponentKind.EXPERT
            and component.layer_index == router.layer_index
        ]
        if any(
            component.routed is not True or component.shared is True
            for component in experts
        ):
            raise ValueError("layer expert universe contains shared or unrouted experts")
        indices = [component.expert_index for component in experts]
        all_indexed = all(index is not None for index in indices)
        any_indexed = any(index is not None for index in indices)
        if all_indexed != any_indexed:
            raise ValueError("layer expert indices are partially declared")
        # expert_keys are canonically sorted; expert_indices stay parallel so
        # each entry maps a key to its adapter-native identifier.
        expert_keys = tuple(sorted(component.component_key for component in experts))
        if all_indexed:
            index_by_key = {
                component.component_key: component.expert_index
                for component in experts
            }
            expert_indices = tuple(index_by_key[key] for key in expert_keys)
        else:
            expert_indices = None
        if len(set(expert_keys)) != len(expert_keys):
            raise ValueError("layer expert keys are not unique")
        if len(expert_keys) == 0:
            raise ValueError("layer expert universe is empty")
        if fact_expert_count is not None and len(expert_keys) != fact_expert_count:
            raise ValueError("layer expert universe does not match inspection facts")

        shared = [
            component
            for component in components
            if component.kind is ComponentKind.SHARED_EXPERT
            and component.layer_index == router.layer_index
        ]
        if any(
            component.routed is not False
            or component.shared is not True
            or component.expert_index is not None
            for component in shared
        ):
            raise ValueError("shared expert structure is not exact")
        shared_keys = tuple(sorted(component.component_key for component in shared))
        if fact_shared_count is not None and len(shared_keys) != fact_shared_count:
            raise ValueError("shared expert universe does not match inspection facts")

        if metadata_top_k is not None:
            top_k = metadata_top_k
            if fact_top_k is not None and top_k != fact_top_k:
                raise ValueError("layer routed top-k does not match inspection facts")
        elif fact_top_k is not None:
            top_k = fact_top_k
        else:
            raise ValueError("router routed_top_k provenance is not available")
        if top_k > len(expert_keys):
            raise ValueError("layer routed top-k exceeds the layer expert count")

        layer_records.append(
            (
                router.layer_index,
                layer.component_key,
                router.component_key,
                expert_keys,
                expert_indices,
                top_k,
                shared_keys,
            )
        )
    if len(layouts) != 1:
        raise ValueError("inspection layouts are inconsistent")
    layer_records.sort(key=lambda record: record[0])

    routed_experts = {
        component.component_key
        for component in components
        if component.kind is ComponentKind.EXPERT
        and component.routed is True
        and component.shared is not True
    }
    if routed_experts != {key for record in layer_records for key in record[3]}:
        raise ValueError("inspection does not publish the full routed expert universe")
    # An EXPERT kind is a routed target in the canonical structural schema;
    # shared-expert components are validated above but are intentionally
    # absent from the routed universe.  Unlike the rectangular analysis path,
    # no contiguity or uniformity requirement is imposed here: per-layer
    # expert counts, native index sparsity, and top-k variation are all
    # first-class universe shapes.
    return RoutingUniverse(
        model_key=model_key,
        adapter_name=descriptor.name,
        adapter_version=descriptor.version,
        layout=next(iter(layouts)),
        layers=tuple(
            LayerRoutingUniverse(
                layer_index=record[0],
                moe_layer_key=record[1],
                router_key=record[2],
                expert_keys=record[3],
                expert_indices=record[4],
                routed_top_k=record[5],
                shared_expert_keys=record[6],
            )
            for record in layer_records
        ),
    )


def _strict_fact(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or isinstance(value, bool) or value < 1:
        raise ValueError(f"{field_name} fact must be a positive integer when present")
    return value
