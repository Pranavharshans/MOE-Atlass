"""Safe collection of the structural surfaces used by static discovery.

This module has no MoE heuristics. It only reads duck-typed module,
parameter, and configuration surfaces, preserving valid entries when an
iterator fails partway through and returning deterministic warnings.
"""

from __future__ import annotations

import operator
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

from ..core import validate_stable_identifier

_MISSING = object()

_CONFIG_ALIASES: dict[str, tuple[str, ...]] = {
    "expert_count": (
        "num_local_experts",
        "num_experts",
        "n_routed_experts",
        "num_expert",
    ),
    "routed_top_k": (
        "num_experts_per_tok",
        "num_experts_per_token",
        "num_selected_experts",
        "top_k",
        "top_k_experts",
    ),
    "shared_expert_count": (
        "n_shared_experts",
        "num_shared_experts",
        "shared_expert_count",
    ),
}


@dataclass(frozen=True)
class ModuleEntry:
    """One named module and its portable path."""

    path: str
    value: object


@dataclass(frozen=True)
class ConfigSnapshot:
    """Validated configuration facts and their original field sources."""

    expert_count: int | None
    expert_count_source: str | None
    routed_top_k: int | None
    routed_top_k_source: str | None
    shared_expert_count: int | None
    shared_expert_count_source: str | None

    @property
    def has_valid_field(self) -> bool:
        return any(
            value is not None
            for value in (self.expert_count, self.routed_top_k, self.shared_expert_count)
        )


def safe_getattr(value: object, name: str) -> object:
    """Read one attribute without allowing a broken descriptor to abort scan."""

    try:
        return getattr(value, name)
    except Exception:
        return _MISSING


def qualified_type_name(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _warning_for_exception(operation: str, exc: Exception, retained: int | None = None) -> str:
    suffix = f" after {retained} retained pair(s)" if retained is not None else ""
    return f"{operation} failed{suffix} with {type(exc).__name__}; later evidence was omitted"


def _pair_iterator(values: object) -> object:
    if isinstance(values, Mapping):
        return iter(values.items())
    return iter(values)  # type: ignore[arg-type]


def _collect_pairs(
    model: object,
    method_name: str,
    warnings: list[str],
    *,
    required: bool,
) -> list[tuple[str, object]]:
    method = safe_getattr(model, method_name)
    if method is _MISSING or not callable(method):
        if required:
            warnings.append(
                f"object does not expose callable {method_name}(); static traversal is empty"
            )
        return []

    try:
        iterator = _pair_iterator(method())
    except Exception as exc:
        warnings.append(_warning_for_exception(f"{method_name}() traversal", exc))
        return []

    pairs: list[tuple[str, object]] = []
    position = 0
    retained = 0
    while True:
        try:
            item = next(iterator)  # type: ignore[call-overload]
        except StopIteration:
            break
        except Exception as exc:
            warnings.append(_warning_for_exception(f"{method_name}() iteration", exc, retained))
            break
        if not isinstance(item, tuple) or len(item) != 2:
            warnings.append(f"{method_name}() item {position} is not a (name, value) pair")
            position += 1
            continue
        name, value = item
        if not isinstance(name, str):
            warnings.append(f"{method_name}() item {position} has a non-string name")
            position += 1
            continue
        pairs.append((name, value))
        retained += 1
        position += 1
    return pairs


def collect_named_modules(model: object, warnings: list[str]) -> dict[str, ModuleEntry]:
    """Collect and deterministically order named modules."""

    raw_entries = _collect_pairs(model, "named_modules", warnings, required=True)
    entries: list[tuple[str, object]] = []
    for position, (path, value) in enumerate(raw_entries):
        if path:
            try:
                validate_stable_identifier(path, field_name="module path")
            except (TypeError, ValueError):
                warnings.append(f"named_modules() item {position} has an invalid module path")
                continue
        entries.append((path, value))

    entries.sort(key=lambda pair: (pair[0], qualified_type_name(pair[1])))
    modules: dict[str, ModuleEntry] = {}
    for path, value in entries:
        if path in modules:
            warnings.append(f"named_modules() returned duplicate module path {path!r}; first kept")
            continue
        modules[path] = ModuleEntry(path=path, value=value)
    return modules


def _safe_shape(value: object) -> list[int] | None:
    """Extract a JSON-safe shape without importing or depending on tensor types."""

    shape = safe_getattr(value, "shape")
    if shape is _MISSING:
        return None
    try:
        dimensions = tuple(shape)  # type: ignore[arg-type]
    except TypeError:
        dimensions = (shape,)
    except Exception:
        return None

    normalized: list[int] = []
    for dimension in dimensions:
        try:
            normalized_dimension = operator.index(dimension)
        except (TypeError, ValueError, OverflowError):
            return None
        if normalized_dimension < 0:
            return None
        normalized.append(int(normalized_dimension))
    return normalized


def collect_parameter_shapes(model: object, warnings: list[str]) -> dict[str, dict[str, list[int]]]:
    """Collect readable parameter shapes keyed by every containing module path."""

    raw_parameters = _collect_pairs(model, "named_parameters", warnings, required=False)
    parameters: list[tuple[str, object]] = []
    for position, (name, parameter) in enumerate(raw_parameters):
        if not name:
            warnings.append(f"named_parameters() item {position} has an invalid parameter name")
            continue
        try:
            validate_stable_identifier(name, field_name="parameter name")
        except (TypeError, ValueError):
            warnings.append(f"named_parameters() item {position} has an invalid parameter name")
            continue
        parameters.append((name, parameter))

    parameters.sort(key=lambda pair: (pair[0], qualified_type_name(pair[1])))
    shapes_by_module: dict[str, dict[str, list[int]]] = defaultdict(dict)
    for name, parameter in parameters:
        shape = _safe_shape(parameter)
        if shape is None:
            continue
        path_parts = name.split(".")
        for index in range(len(path_parts)):
            module_path = ".".join(path_parts[:index])
            parameter_name = ".".join(path_parts[index:])
            shapes_by_module[module_path][parameter_name] = shape
    return {path: dict(values) for path, values in shapes_by_module.items()}


def _surface_value(surface: object, field_name: str) -> object:
    if isinstance(surface, Mapping):
        try:
            return surface.get(field_name, _MISSING)
        except Exception:
            return _MISSING
    return safe_getattr(surface, field_name)


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        normalized = operator.index(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return int(normalized) if normalized > 0 else None


def collect_config(model: object, warnings: list[str]) -> ConfigSnapshot:
    """Read known fact aliases from config first, then direct model fields."""

    config = safe_getattr(model, "config")
    surfaces: list[tuple[str, object]] = []
    if config is not _MISSING and config is not None:
        surfaces.append(("config", config))
    surfaces.append(("model", model))

    observations: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for semantic_name, aliases in _CONFIG_ALIASES.items():
        for surface_name, surface in surfaces:
            for alias in aliases:
                raw_value = _surface_value(surface, alias)
                if raw_value is _MISSING or raw_value is None:
                    continue
                normalized = _positive_int(raw_value)
                if normalized is None:
                    warnings.append(
                        f"{surface_name}.{alias} is not a positive integer; field ignored"
                    )
                    continue
                observations[semantic_name].append((f"{surface_name}.{alias}", normalized))

    selected: dict[str, tuple[int | None, str | None]] = {}
    for semantic_name in _CONFIG_ALIASES:
        values = observations.get(semantic_name, [])
        if not values:
            selected[semantic_name] = (None, None)
            continue
        unique_values = sorted({value for _, value in values})
        if len(unique_values) > 1:
            formatted = ", ".join(f"{source}={value}" for source, value in values)
            warnings.append(f"conflicting {semantic_name} configuration fields: {formatted}")
        selected[semantic_name] = (values[0][1], values[0][0])

    return ConfigSnapshot(
        expert_count=selected["expert_count"][0],
        expert_count_source=selected["expert_count"][1],
        routed_top_k=selected["routed_top_k"][0],
        routed_top_k_source=selected["routed_top_k"][1],
        shared_expert_count=selected["shared_expert_count"][0],
        shared_expert_count_source=selected["shared_expert_count"][1],
    )


def parent_path(path: str) -> str:
    return path.rpartition(".")[0]


def child_path(parent: str, leaf: str) -> str:
    return f"{parent}.{leaf}" if parent else leaf


__all__ = [
    "ConfigSnapshot",
    "ModuleEntry",
    "child_path",
    "collect_config",
    "collect_named_modules",
    "collect_parameter_shapes",
    "parent_path",
    "qualified_type_name",
    "safe_getattr",
]
