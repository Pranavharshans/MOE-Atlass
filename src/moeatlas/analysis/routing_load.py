"""Bounded Mixtral routing-load aggregation over immutable Feature 19 shards."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..adapters import AdapterInspection
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

_ADAPTER_NAME = "huggingface-mixtral-static"
_ADAPTER_VERSION = "1.0"
_LAYOUTS = frozenset({"legacy_indexed", "packed"})
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_SHARD = re.compile(r"^shard:[0-9a-f]{64}$")
_ERROR_STAGES = frozenset({"inspection", "budget", "source", "query"})


class RoutingLoadError(RuntimeError):
    """Safe fixed-stage failure for bounded routing-load aggregation."""

    def __init__(self, stage: str) -> None:
        if stage not in _ERROR_STAGES:
            raise ValueError("routing load error stage is not supported")
        self.stage = stage
        super().__init__(f"mixtral routing load aggregation failed at {stage}")


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


def _fresh_universe(value: object) -> _InspectionUniverse:
    if type(value) is not AdapterInspection:
        raise TypeError("inspection must be an exact AdapterInspection")
    try:
        fresh = AdapterInspection.model_validate(value.model_dump(mode="json"))
        if type(fresh) is not AdapterInspection or fresh is value:
            raise TypeError("inspection revalidation returned an unexpected type")
        descriptor = fresh.descriptor
        if (
            descriptor.name != _ADAPTER_NAME
            or descriptor.version != _ADAPTER_VERSION
            or descriptor.architecture_families != ("mixtral",)
        ):
            raise ValueError("inspection descriptor is not the exact Mixtral identity")
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
            or facts.expert_count_source != "config.num_local_experts"
            or facts.routed_top_k_source != "config.num_experts_per_tok"
        ):
            raise ValueError("inspection facts are not exact Mixtral routing facts")
        components = tuple(fresh.report.components)
        routers = [component for component in components if component.kind is ComponentKind.ROUTER]
        if not routers:
            raise ValueError("inspection has no router universe")
        moe_layers = [
            component for component in components if component.kind is ComponentKind.MOE_LAYER
        ]
        if len(moe_layers) != len(routers):
            raise ValueError("inspection MoE layers do not exactly match routers")
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
                or capture.method != "mixtral-static-structure-v1"
                or capture.adapter != _ADAPTER_NAME
                or capture.adapter_version != _ADAPTER_VERSION
                or capture.verified is not False
            ):
                raise ValueError("router provenance is not exact Mixtral evidence")
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
            parse_component_key(layer.component_key)
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
        inspection_digest = "sha256:" + stable_digest(fresh.model_dump(mode="json"))
        return _InspectionUniverse(
            model_key=model_key,
            adapter_name=_ADAPTER_NAME,
            adapter_version=_ADAPTER_VERSION,
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


def _canonical_tuple(value: object, field_name: str) -> tuple[Any, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{field_name} must be an exact tuple")
    return value


def _canonical_component(value: object, field_name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must contain exact strings")
    parse_component_key(value)
    return value


@dataclass(frozen=True, slots=True)
class MixtralRoutingLoadMatrix:
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
            or self.adapter_name != _ADAPTER_NAME
            or self.adapter_version != _ADAPTER_VERSION
        ):
            raise ValueError("adapter identity is not the exact Mixtral identity")
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


def _validate_sources(
    workspace: str | Path,
    run_key: str,
    duckdb: Any,
    max_routing_rows: int,
    max_source_bytes: int,
) -> tuple[Path, tuple[tuple[Path, dict[str, object], str], ...]]:
    path = _storage._validate_workspace(workspace)
    stable_run_key = _storage._validate_run_key(run_key)
    run_parent = _storage._existing_run_parent(path, stable_run_key)
    if run_parent is None:
        raise _error("source", ValueError("run has no committed routing shards"))
    try:
        children = tuple(run_parent.iterdir())
    except Exception as exc:
        raise _error("source", exc)
    shard_paths: list[Path] = []
    for child in children:
        if _storage._STAGING_NAME.fullmatch(child.name):
            if child.is_symlink() or not child.is_dir():
                raise _storage.RoutingShardError("reopen")
            continue
        if child.name.startswith(".staging-"):
            raise _storage.RoutingShardError("reopen")
        if not child.name.startswith(_storage._SHARD_PREFIX):
            raise _storage.RoutingShardError("reopen")
        shard_paths.append(child)
    if not shard_paths:
        raise _error("source", ValueError("run has no committed routing shards"))
    sources: list[tuple[Path, dict[str, object], str]] = []
    source_bytes = 0
    declared_routing_rows = 0
    for shard in sorted(shard_paths, key=lambda item: item.name):
        manifest, shard_key = _storage._read_shard_manifest(
            shard, stable_run_key, duckdb, validate_files=False
        )
        declared_routing_rows += manifest["routing_count"]
        if declared_routing_rows > max_routing_rows:
            raise _error("budget", ValueError("routing rows exceed the source budget"))
        try:
            sizes = [
                (shard / _storage._MANIFEST_FILE).stat().st_size,
                (shard / _storage._TOKENS_FILE).stat().st_size,
                (shard / _storage._ROUTING_FILE).stat().st_size,
            ]
        except Exception as exc:
            raise _error("source", exc)
        source_bytes += sum(sizes)
        if source_bytes > max_source_bytes:
            raise _error("budget", ValueError("source bytes exceed the source budget"))
        sources.append((shard, manifest, shard_key))
    return path, tuple(sources)


def aggregate_mixtral_routing_load(
    workspace: str | Path,
    inspection: AdapterInspection,
    *,
    run_key: str,
    max_routing_rows: int,
    max_source_bytes: int,
    max_matrix_cells: int,
) -> MixtralRoutingLoadMatrix:
    """Aggregate complete selected routing assignments over one run's shards."""

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
    except (TypeError, ValueError) as exc:
        raise _error("inspection", exc)
    cells = len(universe.layer_keys) * universe.expert_count
    if cells > max_matrix_cells:
        raise _error("budget", ValueError("matrix cells exceed the matrix budget"))

    # Validate the managed workspace/run boundary before importing the optional
    # store dependency, so absent or malformed source is deterministic offline.
    workspace_path = _storage._validate_workspace(workspace)
    _storage._validate_run_key(run_key)
    if _storage._existing_run_parent(workspace_path, run_key) is None:
        raise _error("source", ValueError("run has no committed routing shards"))
    duckdb = _storage._load_duckdb()
    _, sources = _validate_sources(
        workspace_path,
        run_key,
        duckdb,
        max_routing_rows,
        max_source_bytes,
    )
    connection: Any | None = None
    primary: BaseException | None = None
    token_count = 0
    assignment_count = 0
    shard_keys: list[str] = []
    counts = [[0 for _ in range(universe.expert_count)] for _ in universe.layer_keys]
    seen_tokens: set[str] = set()
    seen_links: set[tuple[str, str, int]] = set()
    actual_counts: dict[Path, tuple[int, int]] = {}
    actual_routing_rows = 0
    try:
        connection = duckdb.connect(database=":memory:")
        for shard, manifest, shard_key in sources:
            try:
                _storage._validate_file_metadata(shard, manifest)
            except _storage.RoutingShardError:
                raise
            except Exception as exc:
                raise _storage.RoutingShardError("reopen") from exc
            token_path = shard / _storage._TOKENS_FILE
            routing_path = shard / _storage._ROUTING_FILE
            try:
                actual_token_count = connection.execute(
                    "SELECT COUNT(*) FROM read_parquet(?)", [str(token_path)]
                ).fetchone()[0]
                actual_routing_count = connection.execute(
                    "SELECT COUNT(*) FROM read_parquet(?)", [str(routing_path)]
                ).fetchone()[0]
            except _storage.RoutingShardError:
                raise
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as exc:
                raise _storage.RoutingShardError("reopen") from exc
            actual_routing_rows += int(actual_routing_count)
            if actual_routing_rows > max_routing_rows:
                raise _error("budget", ValueError("actual routing rows exceed the row budget"))
            if (
                actual_token_count != manifest["token_count"]
                or actual_routing_count != manifest["routing_count"]
            ):
                raise _storage.RoutingShardError("reopen") from ValueError(
                    "parquet row counts do not match manifest"
                )
            actual_counts[shard] = (int(actual_token_count), int(actual_routing_count))
        for shard, manifest, shard_key in sources:
            actual_token_count, actual_routing_count = actual_counts[shard]
            try:
                token_keys, routing_links = _storage._validate_routing_load_source(
                    shard,
                    run_key,
                    duckdb,
                    connection,
                    universe.layer_keys,
                    universe.expert_keys,
                    universe.routed_top_k,
                )
            except _storage.RoutingShardError:
                raise
            except OSError as exc:
                raise _error("query", exc)
            except Exception as exc:
                raise _error("source", exc)
            if seen_tokens.intersection(token_keys) or seen_links.intersection(routing_links):
                raise _storage.RoutingShardError("conflict")
            seen_tokens.update(token_keys)
            seen_links.update(routing_links)
            rows = connection.execute(
                "SELECT layer_key, expert_key, COUNT(*) AS assignment_count "
                "FROM read_parquet(?) GROUP BY layer_key, expert_key "
                "ORDER BY layer_key, expert_key",
                [str(routing_path)],
            ).fetchall()
            token_count += int(actual_token_count)
            assignment_count += int(actual_routing_count)
            shard_keys.append(shard_key)
            for layer_key, expert_key, count in rows:
                if layer_key not in universe.layer_keys:
                    raise _error("source", ValueError("source layer is outside inspection"))
                layer_position = universe.layer_keys.index(layer_key)
                if expert_key not in universe.expert_keys[layer_position]:
                    raise _error("source", ValueError("source expert is outside inspection"))
                expert_position = universe.expert_keys[layer_position].index(expert_key)
                counts[layer_position][expert_position] += int(count)
    except BaseException as exc:
        if isinstance(
            exc,
            _storage.RoutingShardError | RoutingLoadError | KeyboardInterrupt | SystemExit,
        ):
            primary = exc
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

    expected_assignments = token_count * len(universe.layer_keys) * universe.routed_top_k
    if assignment_count != expected_assignments:
        raise _error("source", ValueError("run assignment count is incomplete"))
    layer_total = token_count * universe.routed_top_k
    shares = tuple(tuple(float(count) / layer_total for count in row) for row in counts)
    ratios = tuple(tuple(share * universe.expert_count for share in row) for row in shares)
    try:
        return MixtralRoutingLoadMatrix(
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
    "MixtralRoutingLoadMatrix",
    "RoutingLoadError",
    "aggregate_mixtral_routing_load",
]
