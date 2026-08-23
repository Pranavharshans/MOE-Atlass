"""Bounded model-neutral routing-load aggregation over immutable shards."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..adapters import (
    AdapterInspection,
    RoutingUniverse,
    UniversalRoutingInspection,
    project_rectangular_universe,
    publish_routing_universe,
)
from ..core import (
    CaptureSource,
    ComponentKind,
    parse_component_key,
    parse_model_key,
    stable_digest,
    validate_stable_identifier,
)
from ..events import EVENT_SCHEMA_VERSION
from ..store import STORE_SCHEMA_VERSION
from ..store import routing_shards as _storage

ROUTING_LOAD_SCHEMA_VERSION = "1.0"

_LAYOUTS = frozenset({"legacy_indexed", "packed"})
_EXPERT_COUNT_SOURCES = frozenset({"config.num_local_experts", "config.num_experts"})
_TOP_K_SOURCE = "config.num_experts_per_tok"
_SHARED_EXPERT_SOURCE = "topology.shared_expert"
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_SHARD = re.compile(r"^shard:[0-9a-f]{64}$")
_ERROR_STAGES = frozenset({"inspection", "budget", "source", "query"})


class RoutingLoadError(RuntimeError):
    """Safe fixed-stage failure for bounded routing-load aggregation."""

    def __init__(self, stage: str) -> None:
        if stage not in _ERROR_STAGES:
            raise ValueError("routing load error stage is not supported")
        self.stage = stage
        super().__init__(f"routing load aggregation failed at {stage}")


def _error(stage: str, cause: BaseException | None = None) -> RoutingLoadError:
    error = RoutingLoadError(stage)
    if cause is not None:
        error.__cause__ = cause
    return error


def _strict_positive_budget(value: object, field_name: str) -> int:
    if type(value) is not int or isinstance(value, bool):
        raise TypeError(f"{field_name} must be a strict integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


@dataclass(frozen=True, slots=True)
class _InspectionUniverse:
    model_key: str
    adapter_name: str
    adapter_version: str
    inspection_digest: str
    layout: str
    layer_keys: tuple[str, ...]
    layer_indices: tuple[int, ...]
    expert_keys: tuple[tuple[str, ...], ...]
    expert_count: int
    routed_top_k: int


def _fresh_universal_universe(value: UniversalRoutingInspection) -> _InspectionUniverse:
    try:
        fresh = UniversalRoutingInspection.model_validate(value.model_dump(mode="json"))
        if type(fresh) is not UniversalRoutingInspection or fresh is value:
            raise TypeError("inspection revalidation returned an unexpected type")
        parse_model_key(fresh.model_key)
        inspection_digest = "sha256:" + stable_digest(fresh.model_dump(mode="json"))
        # The universal document marks its own provenance; adapter identity is
        # deliberately the fixed generic marker so downstream artifacts never
        # present a non-certified lane as certified adapter evidence.
        return _InspectionUniverse(
            model_key=fresh.model_key,
            adapter_name="universal",
            adapter_version=fresh.scanner_version,
            inspection_digest=inspection_digest,
            layout=fresh.layout,
            layer_keys=tuple(layer.layer_key for layer in fresh.layers),
            layer_indices=tuple(layer.layer_index for layer in fresh.layers),
            expert_keys=tuple(layer.expert_keys for layer in fresh.layers),
            expert_count=fresh.expert_count,
            routed_top_k=fresh.routed_top_k,
        )
    except (TypeError, ValueError):
        raise
    except Exception as exc:
        raise ValueError("inspection revalidation failed") from exc


def _fresh_universe(value: object) -> _InspectionUniverse:
    if type(value) is UniversalRoutingInspection:
        return _fresh_universal_universe(value)
    if type(value) is not AdapterInspection:
        raise TypeError("inspection must be an exact AdapterInspection")
    try:
        fresh = AdapterInspection.model_validate(value.model_dump(mode="json"))
        if type(fresh) is not AdapterInspection or fresh is value:
            raise TypeError("inspection revalidation returned an unexpected type")
        descriptor = fresh.descriptor
        # The analysis boundary deliberately does not maintain a family
        # allowlist.  Adapter identity is provenance carried by the inspection;
        # the routing universe below is accepted only after the complete,
        # model-neutral structural invariants have been checked.
        if (
            type(descriptor.name) is not str
            or type(descriptor.version) is not str
            or not descriptor.name
            or not descriptor.version
            or not descriptor.architecture_families
        ):
            raise ValueError("inspection descriptor identity is not complete")
        parse_model_key(fresh.report.model_key)
        model_key = fresh.report.model_key
        facts = fresh.report.facts
        expert_count = facts.expert_count
        routed_top_k = facts.routed_top_k
        if (
            type(expert_count) is not int
            or isinstance(expert_count, bool)
            or type(routed_top_k) is not int
            or isinstance(routed_top_k, bool)
            or expert_count <= 0
            or routed_top_k <= 0
            or routed_top_k > expert_count
            or facts.expert_count_source not in _EXPERT_COUNT_SOURCES
            or facts.routed_top_k_source != _TOP_K_SOURCE
        ):
            raise ValueError("inspection facts are not complete routing facts")
        components = tuple(fresh.report.components)
        routers = [component for component in components if component.kind is ComponentKind.ROUTER]
        if not routers:
            raise ValueError("inspection has no router universe")
        moe_layers = [
            component for component in components if component.kind is ComponentKind.MOE_LAYER
        ]
        if len(moe_layers) != len(routers):
            raise ValueError("inspection MoE layers do not exactly match routers")
        # Every structural routing component must be bound to one layer.  The
        # check is intentionally based on semantic kinds and flags rather than
        # adapter names or module-path conventions.
        structural_kinds = {
            ComponentKind.MOE_LAYER,
            ComponentKind.ROUTER,
            ComponentKind.EXPERT,
            ComponentKind.SHARED_EXPERT,
        }
        for component in components:
            if component.kind in structural_kinds:
                if component.layer_index is None or type(component.layer_index) is not int:
                    raise ValueError("routing component layer index is not exact")
                parse_component_key(component.component_key)

        layer_records: list[tuple[int, str, tuple[str, ...]]] = []
        seen_layer_indices: set[int] = set()
        layouts: set[str] = set()
        for router in routers:
            if (
                router.layer_index is None
                or type(router.layer_index) is not int
                or router.layer_index < 0
            ):
                raise ValueError("router layer index is not exact")
            if router.layer_index in seen_layer_indices:
                raise ValueError("router layer indices are not unique")
            seen_layer_indices.add(router.layer_index)
            capture = router.capture
            if capture is None or capture.metadata.keys() != {"layout"}:
                raise ValueError("router layout provenance is not exact")
            if (
                capture.source is not CaptureSource.STATIC_STRUCTURE
                or capture.adapter != descriptor.name
                or capture.adapter_version != descriptor.version
                or capture.verified is not False
            ):
                raise ValueError("router provenance is not exact static structure evidence")
            layout = capture.metadata.get("layout")
            if type(layout) is not str or layout not in _LAYOUTS:
                raise ValueError("router layout is not legacy_indexed or packed")
            layouts.add(layout)
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
            if len(experts) != expert_count:
                raise ValueError("layer expert universe does not match inspection facts")
            indices = [component.expert_index for component in experts]
            if any(type(index) is not int for index in indices) or sorted(indices) != list(
                range(expert_count)
            ):
                raise ValueError("layer expert indices are not contiguous")
            if any(
                component.routed is not True or component.shared is True for component in experts
            ):
                raise ValueError("layer expert universe contains shared or unrouted experts")
            experts.sort(key=lambda component: component.expert_index)
            expert_keys = tuple(component.component_key for component in experts)
            if len(set(expert_keys)) != len(expert_keys):
                raise ValueError("layer expert keys are not unique")
            for key in expert_keys:
                parse_component_key(key)

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
            layer_records.append((router.layer_index, layer.component_key, expert_keys))
        if len(layouts) != 1:
            raise ValueError("inspection layouts are inconsistent")
        layer_records.sort(key=lambda record: record[0])
        layer_indices = tuple(record[0] for record in layer_records)
        layer_keys = tuple(record[1] for record in layer_records)
        expert_keys = tuple(record[2] for record in layer_records)
        if layer_indices != tuple(range(len(layer_indices))):
            raise ValueError("inspection layer indices are not contiguous")
        if len(set(layer_keys)) != len(layer_keys):
            raise ValueError("inspection layer keys are not unique")
        flat_expert_keys = tuple(key for row in expert_keys for key in row)
        if len(set(flat_expert_keys)) != len(flat_expert_keys):
            raise ValueError("inspection expert keys are not globally unique")
        routed_experts = {
            component.component_key
            for component in components
            if component.kind is ComponentKind.EXPERT
            and component.routed is True
            and component.shared is not True
        }
        if routed_experts != {key for row in expert_keys for key in row}:
            raise ValueError("inspection does not publish the full routed expert universe")
        # An EXPERT kind is a routed target in the canonical structural schema;
        # accepting an unflagged or shared EXPERT would make the denominator
        # ambiguous.  Shared-expert components are validated above but are
        # intentionally absent from the routed universe and matrix axes.
        if any(
            component.kind is ComponentKind.EXPERT
            and (component.routed is not True or component.shared is True)
            for component in components
        ):
            raise ValueError("inspection contains a non-routed EXPERT component")
        shared_components = [
            component for component in components if component.kind is ComponentKind.SHARED_EXPERT
        ]
        known_layers = set(layer_indices)
        if any(component.layer_index not in known_layers for component in shared_components):
            raise ValueError("shared expert is bound outside the router universe")
        if facts.shared_expert_count is None:
            if shared_components:
                raise ValueError("shared expert components require shared-expert facts")
        elif (
            type(facts.shared_expert_count) is not int
            or isinstance(facts.shared_expert_count, bool)
            or facts.shared_expert_count <= 0
            or facts.shared_expert_count_source != _SHARED_EXPERT_SOURCE
            or len(shared_components) != facts.shared_expert_count * len(layer_indices)
            or any(
                sum(component.layer_index == layer for component in shared_components)
                != facts.shared_expert_count
                for layer in layer_indices
            )
        ):
            raise ValueError("shared-expert facts do not match the structure")
        inspection_digest = "sha256:" + stable_digest(fresh.model_dump(mode="json"))
        return _InspectionUniverse(
            model_key=model_key,
            adapter_name=descriptor.name,
            adapter_version=descriptor.version,
            inspection_digest=inspection_digest,
            layout=next(iter(layouts)),
            layer_keys=layer_keys,
            layer_indices=layer_indices,
            expert_keys=expert_keys,
            expert_count=expert_count,
            routed_top_k=routed_top_k,
        )
    except (TypeError, ValueError):
        raise
    except Exception as exc:
        raise ValueError("inspection revalidation failed") from exc


def _reconciled_universe(
    derived: _InspectionUniverse,
    inspection: AdapterInspection,
    declared: object,
) -> _InspectionUniverse:
    """Check a caller-declared routing universe against the inspection.

    The declared universe must be exactly what the inspection publishes and
    must pass the explicit rectangular projection.  Publication is
    family-blind, so this gates adapter-declared layouts and per-layer top-k
    metadata through the same named step instead of leaving rectangularity as
    a hidden invariant.  Axes themselves remain the validated legacy
    derivation so matrix construction stays byte-identical across both paths.
    """

    if type(declared) is not RoutingUniverse:
        raise TypeError(
            f"universe must be an exact RoutingUniverse, got {type(declared).__name__}"
        )
    published = publish_routing_universe(inspection)
    if published != declared:
        raise ValueError(
            "declared universe does not match the inspection-published routing universe"
        )
    project_rectangular_universe(declared)
    return derived


def _canonical_tuple(value: object, field_name: str) -> tuple[Any, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{field_name} must be an exact tuple")
    return value


def _canonical_component(value: object, field_name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must contain exact strings")
    parse_component_key(value)
    return value


_ROUTING_LOAD_ARTIFACT_TYPE = "moeatlas.routing_load_matrix"


def _strict_string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if type(value) is not list or any(type(item) is not str for item in value):
        raise ValueError(f"{field_name} must be a JSON array of strings")
    return tuple(value)


def _strict_index_tuple(value: object, field_name: str) -> tuple[int, ...]:
    if type(value) is not list or any(
        type(item) is not int or isinstance(item, bool) for item in value
    ):
        raise ValueError(f"{field_name} must be a JSON array of integers")
    return tuple(value)


def _strict_row_tuple(value: object, field_name: str) -> tuple[tuple[str, ...], ...]:
    if type(value) is not list:
        raise ValueError(f"{field_name} must be a JSON array of string arrays")
    rows: list[tuple[str, ...]] = []
    for row in value:
        rows.append(_strict_string_tuple(row, field_name))
    return tuple(rows)


def _strict_count_rows(value: object, field_name: str) -> tuple[tuple[int, ...], ...]:
    if type(value) is not list:
        raise ValueError(f"{field_name} must be a JSON array of integer arrays")
    rows: list[tuple[int, ...]] = []
    for row in value:
        if type(row) is not list or any(
            type(item) is not int or isinstance(item, bool) for item in row
        ):
            raise ValueError(f"{field_name} must be a JSON array of integer arrays")
        rows.append(tuple(row))
    return tuple(rows)


def _strict_float_rows(value: object, field_name: str) -> tuple[tuple[float, ...], ...]:
    if type(value) is not list:
        raise ValueError(f"{field_name} must be a JSON array of number arrays")
    rows: list[tuple[float, ...]] = []
    for row in value:
        if type(row) is not list or any(type(item) is not float for item in row):
            raise ValueError(f"{field_name} must be a JSON array of number arrays")
        rows.append(tuple(row))
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class RoutingLoadMatrix:
    schema_version: str
    store_schema_version: str
    event_schema_version: str
    run_key: str
    model_key: str
    adapter_name: str
    adapter_version: str
    inspection_digest: str
    layout: str
    shard_keys: tuple[str, ...]
    token_count: int
    assignment_count: int
    routed_top_k: int
    layer_keys: tuple[str, ...]
    layer_indices: tuple[int, ...]
    expert_keys: tuple[tuple[str, ...], ...]
    assignment_counts: tuple[tuple[int, ...], ...]
    assignment_shares: tuple[tuple[float, ...], ...]
    load_ratios: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not str
            or self.schema_version != ROUTING_LOAD_SCHEMA_VERSION
        ):
            raise ValueError("schema_version is not the exact routing-load version")
        if (
            type(self.store_schema_version) is not str
            or self.store_schema_version != STORE_SCHEMA_VERSION
        ):
            raise ValueError("store_schema_version is not the exact store version")
        if (
            type(self.event_schema_version) is not str
            or self.event_schema_version != EVENT_SCHEMA_VERSION
        ):
            raise ValueError("event_schema_version is not the exact event version")
        if (
            type(self.adapter_name) is not str
            or type(self.adapter_version) is not str
            or not self.adapter_name
            or not self.adapter_version
            or self.adapter_name != self.adapter_name.strip()
            or self.adapter_version != self.adapter_version.strip()
        ):
            raise ValueError("adapter identity is not complete")
        if type(self.run_key) is not str:
            raise TypeError("run_key must be an exact string")
        validate_stable_identifier(self.run_key, field_name="run_key")
        if type(self.model_key) is not str:
            raise TypeError("model_key must be an exact string")
        parse_model_key(self.model_key)
        if (
            type(self.inspection_digest) is not str
            or _DIGEST.fullmatch(self.inspection_digest) is None
        ):
            raise ValueError("inspection_digest must be sha256:<64hex>")
        if type(self.layout) is not str or self.layout not in _LAYOUTS:
            raise ValueError("layout must be legacy_indexed or packed")
        shard_keys = _canonical_tuple(self.shard_keys, "shard_keys")
        if not shard_keys or any(
            type(key) is not str or _SHARD.fullmatch(key) is None for key in shard_keys
        ):
            raise ValueError("shard_keys must be non-empty canonical shard keys")
        if tuple(sorted(shard_keys)) != shard_keys or len(set(shard_keys)) != len(shard_keys):
            raise ValueError("shard_keys must be sorted and unique")
        for field_name in ("token_count", "assignment_count", "routed_top_k"):
            value = getattr(self, field_name)
            if type(value) is not int or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field_name} must be a strict positive integer")
        layer_keys = _canonical_tuple(self.layer_keys, "layer_keys")
        layer_indices = _canonical_tuple(self.layer_indices, "layer_indices")
        expert_keys = _canonical_tuple(self.expert_keys, "expert_keys")
        count_rows = _canonical_tuple(self.assignment_counts, "assignment_counts")
        share_rows = _canonical_tuple(self.assignment_shares, "assignment_shares")
        ratio_rows = _canonical_tuple(self.load_ratios, "load_ratios")
        if (
            not layer_keys
            or len(layer_keys) != len(layer_indices)
            or len(layer_keys) != len(expert_keys)
        ):
            raise ValueError("layer axes must have one non-empty row per layer")
        if (
            len(layer_keys) != len(count_rows)
            or len(layer_keys) != len(share_rows)
            or len(layer_keys) != len(ratio_rows)
        ):
            raise ValueError("matrix row counts must match layer axes")
        if any(type(key) is not str for key in layer_keys):
            raise TypeError("layer_keys must contain exact strings")
        for key in layer_keys:
            parse_component_key(key)
        if len(set(layer_keys)) != len(layer_keys):
            raise ValueError("layer_keys must be unique")
        if any(
            type(index) is not int or isinstance(index, bool) or index < 0
            for index in layer_indices
        ):
            raise TypeError("layer_indices must contain strict nonnegative integers")
        if tuple(sorted(set(layer_indices))) != layer_indices:
            raise ValueError("layer_indices must be strictly ascending")
        expert_count: int | None = None
        all_expert_keys: set[str] = set()
        for row_index, row in enumerate(expert_keys):
            if type(row) is not tuple or not row:
                raise TypeError("expert_keys rows must be non-empty exact tuples")
            keys = tuple(_canonical_component(key, "expert_keys") for key in row)
            if len(set(keys)) != len(keys):
                raise ValueError("expert_keys rows must be unique")
            if all_expert_keys.intersection(keys):
                raise ValueError("expert_keys must be globally unique")
            all_expert_keys.update(keys)
            if expert_count is None:
                expert_count = len(keys)
            elif len(keys) != expert_count:
                raise ValueError("expert_keys must be rectangular")
            if keys != row:
                raise ValueError("expert_keys must contain canonical values")
            for matrix_name, matrix_row in (
                ("assignment_counts", count_rows[row_index]),
                ("assignment_shares", share_rows[row_index]),
                ("load_ratios", ratio_rows[row_index]),
            ):
                if type(matrix_row) is not tuple or len(matrix_row) != len(keys):
                    raise ValueError(f"{matrix_name} must match expert axes")
        assert expert_count is not None
        if self.routed_top_k > expert_count:
            raise ValueError("routed_top_k cannot exceed the expert axis")
        layer_total = self.token_count * self.routed_top_k
        total_assignments = 0
        for counts, shares, ratios in zip(count_rows, share_rows, ratio_rows, strict=True):
            for count in counts:
                if type(count) is not int or isinstance(count, bool) or count < 0:
                    raise TypeError("assignment_counts must contain strict nonnegative integers")
            if sum(counts) != layer_total:
                raise ValueError("each assignment-count row must sum to token_count*top_k")
            total_assignments += sum(counts)
            for count, share, ratio in zip(counts, shares, ratios, strict=True):
                if type(share) is not float or not math.isfinite(share) or share < 0:
                    raise TypeError("assignment_shares must contain finite nonnegative floats")
                if type(ratio) is not float or not math.isfinite(ratio) or ratio < 0:
                    raise TypeError("load_ratios must contain finite nonnegative floats")
                expected_share = count / layer_total
                if abs(share - expected_share) > 1e-12:
                    raise ValueError("assignment_shares do not match assignment counts")
                if ratio != share * expert_count:
                    raise ValueError("load_ratios do not match assignment shares")
            if abs(sum(shares) - 1.0) > 1e-12:
                raise ValueError("assignment shares must sum to one per layer")
            if abs(sum(ratios) / expert_count - 1.0) > 1e-12:
                raise ValueError("load ratios must have mean one per layer")
        if self.assignment_count != total_assignments:
            raise ValueError("assignment_count does not match assignment-count rows")
        expected_assignments = self.token_count * len(layer_keys) * self.routed_top_k
        if self.assignment_count != expected_assignments:
            raise ValueError("assignment_count does not match token/layer/top-k formula")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible data without runtime objects."""

        return {
            "artifact_type": _ROUTING_LOAD_ARTIFACT_TYPE,
            "schema_version": self.schema_version,
            "store_schema_version": self.store_schema_version,
            "event_schema_version": self.event_schema_version,
            "run_key": self.run_key,
            "model_key": self.model_key,
            "adapter_name": self.adapter_name,
            "adapter_version": self.adapter_version,
            "inspection_digest": self.inspection_digest,
            "layout": self.layout,
            "shard_keys": list(self.shard_keys),
            "token_count": self.token_count,
            "assignment_count": self.assignment_count,
            "routed_top_k": self.routed_top_k,
            "layer_keys": list(self.layer_keys),
            "layer_indices": list(self.layer_indices),
            "expert_keys": [list(row) for row in self.expert_keys],
            "assignment_counts": [list(row) for row in self.assignment_counts],
            "assignment_shares": [list(row) for row in self.assignment_shares],
            "load_ratios": [list(row) for row in self.load_ratios],
        }

    def to_json(self) -> str:
        """Serialize this matrix with deterministic key order and no whitespace."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> RoutingLoadMatrix:
        """Validate one canonical JSON document into an exact matrix value."""

        try:
            document = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("routing load matrix document is not valid JSON") from exc
        if type(document) is not dict:
            raise ValueError("routing load matrix document must be a JSON object")
        if (
            document.get("artifact_type") != _ROUTING_LOAD_ARTIFACT_TYPE
            or document.get("schema_version") != ROUTING_LOAD_SCHEMA_VERSION
        ):
            raise ValueError("document is not a routing load matrix artifact")
        try:
            return cls(
                schema_version=document["schema_version"],
                store_schema_version=document["store_schema_version"],
                event_schema_version=document["event_schema_version"],
                run_key=document["run_key"],
                model_key=document["model_key"],
                adapter_name=document["adapter_name"],
                adapter_version=document["adapter_version"],
                inspection_digest=document["inspection_digest"],
                layout=document["layout"],
                shard_keys=_strict_string_tuple(document["shard_keys"], "shard_keys"),
                token_count=document["token_count"],
                assignment_count=document["assignment_count"],
                routed_top_k=document["routed_top_k"],
                layer_keys=_strict_string_tuple(document["layer_keys"], "layer_keys"),
                layer_indices=_strict_index_tuple(document["layer_indices"], "layer_indices"),
                expert_keys=_strict_row_tuple(document["expert_keys"], "expert_keys"),
                assignment_counts=_strict_count_rows(
                    document["assignment_counts"], "assignment_counts"
                ),
                assignment_shares=_strict_float_rows(
                    document["assignment_shares"], "assignment_shares"
                ),
                load_ratios=_strict_float_rows(document["load_ratios"], "load_ratios"),
            )
        except KeyError as exc:
            raise ValueError("routing load matrix document is missing fields") from exc
        except (TypeError, ValueError):
            raise
        except Exception as exc:
            raise ValueError("routing load matrix document is not usable") from exc


def aggregate_routing_load(
    workspace: str | Path,
    inspection: AdapterInspection | UniversalRoutingInspection,
    *,
    run_key: str,
    max_routing_rows: int,
    max_source_bytes: int,
    max_matrix_cells: int,
    declared_universe: object | None = None,
) -> RoutingLoadMatrix:
    """Aggregate complete selected routing assignments over one run's shards.

    ``inspection`` accepts either a certified
    :class:`~moeatlas.adapters.AdapterInspection` or a universal structure
    inspection (:class:`~moeatlas.adapters.UniversalRoutingInspection`)
    derived from a static discovery report; both lanes produce the same
    matrix contract.  ``declared_universe`` optionally supplies the
    adapter-published :class:`~moeatlas.adapters.RoutingUniverse` for a
    certified inspection.  It must match the publication exactly and project
    to rectangular form explicitly; aggregation results are identical with or
    without it.
    """

    if not isinstance(workspace, str | Path):
        raise TypeError("workspace must be a string or pathlib.Path")
    if type(run_key) is not str:
        raise TypeError("run_key must be an exact string")
    validate_stable_identifier(run_key, field_name="run_key")
    for value, name in (
        (max_routing_rows, "max_routing_rows"),
        (max_source_bytes, "max_source_bytes"),
        (max_matrix_cells, "max_matrix_cells"),
    ):
        _strict_positive_budget(value, name)
    try:
        universe = _fresh_universe(inspection)
        if declared_universe is not None:
            universe = _reconciled_universe(universe, inspection, declared_universe)
    except (TypeError, ValueError) as exc:
        raise _error("inspection", exc)
    cells = len(universe.layer_keys) * universe.expert_count
    if cells > max_matrix_cells:
        raise _error("budget", ValueError("matrix cells exceed the matrix budget"))

    # Read the run through the public storage query seam instead of concrete
    # shard internals; the seam owns source discovery, budgets, validation,
    # conflict detection, and the bounded grouped-count queries.  Analysis
    # owns the bounded in-memory connection lifecycle: open once, close once.
    primary: BaseException | None = None
    records: tuple[_storage.RoutingShardAssignmentQuery, ...] | None = None
    connection: Any | None = None
    try:
        duckdb = _storage._load_duckdb()
        connection = duckdb.connect(database=":memory:")
        records = _storage.query_routing_run_assignments(
            workspace,
            run_key=run_key,
            layer_keys=universe.layer_keys,
            expert_keys=universe.expert_keys,
            routed_top_k=universe.routed_top_k,
            max_routing_rows=max_routing_rows,
            max_source_bytes=max_source_bytes,
            duckdb=duckdb,
            connection=connection,
        )
    except BaseException as exc:
        if isinstance(
            exc,
            _storage.RoutingShardError | RoutingLoadError | KeyboardInterrupt | SystemExit,
        ):
            primary = exc
        elif isinstance(exc, _storage.RoutingRunQueryError):
            primary = _error("query", exc)
        elif isinstance(exc, _storage.RoutingRunInventoryError):
            primary = _error("budget", exc)
        elif isinstance(exc, TypeError | ValueError | OSError):
            primary = _error("source", exc)
        else:
            primary = _error("query", exc)
    finally:
        if connection is not None:
            try:
                connection.close()
            except BaseException as close_error:
                if primary is None:
                    primary = (
                        close_error
                        if isinstance(
                            close_error,
                            _storage.RoutingShardError
                            | RoutingLoadError
                            | KeyboardInterrupt
                            | SystemExit,
                        )
                        else _error("query", close_error)
                    )
    if primary is not None:
        raise primary
    assert records is not None

    token_count = 0
    assignment_count = 0
    shard_keys: list[str] = []
    counts = [[0 for _ in range(universe.expert_count)] for _ in universe.layer_keys]
    for record in records:
        for layer_key, expert_key, count in record.assignment_counts:
            if layer_key not in universe.layer_keys:
                raise _error("source", ValueError("source layer is outside inspection"))
            layer_position = universe.layer_keys.index(layer_key)
            if expert_key not in universe.expert_keys[layer_position]:
                raise _error("source", ValueError("source expert is outside inspection"))
            expert_position = universe.expert_keys[layer_position].index(expert_key)
            counts[layer_position][expert_position] += int(count)
        token_count += record.token_count
        assignment_count += record.routing_count
        shard_keys.append(record.shard_key)

    expected_assignments = token_count * len(universe.layer_keys) * universe.routed_top_k
    if assignment_count != expected_assignments:
        raise _error("source", ValueError("run assignment count is incomplete"))
    layer_total = token_count * universe.routed_top_k
    shares = tuple(tuple(float(count) / layer_total for count in row) for row in counts)
    ratios = tuple(tuple(share * universe.expert_count for share in row) for row in shares)
    try:
        return RoutingLoadMatrix(
            schema_version=ROUTING_LOAD_SCHEMA_VERSION,
            store_schema_version=STORE_SCHEMA_VERSION,
            event_schema_version=EVENT_SCHEMA_VERSION,
            run_key=run_key,
            model_key=universe.model_key,
            adapter_name=universe.adapter_name,
            adapter_version=universe.adapter_version,
            inspection_digest=universe.inspection_digest,
            layout=universe.layout,
            shard_keys=tuple(sorted(shard_keys)),
            token_count=token_count,
            assignment_count=assignment_count,
            routed_top_k=universe.routed_top_k,
            layer_keys=universe.layer_keys,
            layer_indices=universe.layer_indices,
            expert_keys=universe.expert_keys,
            assignment_counts=tuple(tuple(row) for row in counts),
            assignment_shares=shares,
            load_ratios=ratios,
        )
    except (TypeError, ValueError) as exc:
        raise _error("source", exc)


__all__ = [
    "ROUTING_LOAD_SCHEMA_VERSION",
    "RoutingLoadMatrix",
    "RoutingLoadError",
    "aggregate_routing_load",
]
