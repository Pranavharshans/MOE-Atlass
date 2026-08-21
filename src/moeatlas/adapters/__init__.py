"""Explicit static semantic-adapter contracts.

This package contains no adapter registry, plugin discovery, model loading, or
runtime imports.  Callers provide one protocol object to
``inspect_static_adapter`` when they want a validated STRUCTURE inspection.
"""

from .contracts import (
    AdapterContractError,
    AdapterDescriptor,
    AdapterDetection,
    AdapterExecutionError,
    AdapterInspection,
    StaticSemanticAdapter,
    inspect_static_adapter,
)
from .mixtral import MixtralStaticAdapter
from .planning import AdapterProbePlanError, build_routing_probe_plan
from .qwen3_5_moe import Qwen3_5MoeStaticAdapter
from .qwen3_moe import Qwen3MoeStaticAdapter
from .registry import (
    ADAPTER_REGISTRY_SCHEMA_VERSION,
    ENTRY_POINT_GROUP,
    AdapterPluginRecord,
    AdapterRegistryEntry,
    AdapterRegistryError,
    AdapterRegistryPolicy,
    AdapterRegistryReport,
    apply_registry_policy,
    builtin_adapter_records,
    collect_adapter_registry,
    discover_entry_point_records,
    match_adapters_for_family,
)
from .universe import (
    ROUTING_UNIVERSE_SCHEMA_VERSION,
    LayerRoutingUniverse,
    RectangularProjection,
    RoutingUniverse,
    RoutingUniverseError,
    project_rectangular_universe,
    publish_routing_universe,
)

__all__ = [
    "ADAPTER_REGISTRY_SCHEMA_VERSION",
    "ENTRY_POINT_GROUP",
    "AdapterContractError",
    "AdapterDescriptor",
    "AdapterDetection",
    "AdapterExecutionError",
    "AdapterInspection",
    "AdapterPluginRecord",
    "AdapterRegistryEntry",
    "AdapterRegistryError",
    "AdapterRegistryPolicy",
    "AdapterRegistryReport",
    "apply_registry_policy",
    "builtin_adapter_records",
    "collect_adapter_registry",
    "discover_entry_point_records",
    "match_adapters_for_family",
    "RoutingUniverse",
    "LayerRoutingUniverse",
    "RectangularProjection",
    "RoutingUniverseError",
    "publish_routing_universe",
    "project_rectangular_universe",
    "ROUTING_UNIVERSE_SCHEMA_VERSION",
    "StaticSemanticAdapter",
    "inspect_static_adapter",
    "MixtralStaticAdapter",
    "Qwen3MoeStaticAdapter",
    "Qwen3_5MoeStaticAdapter",
    "AdapterProbePlanError",
    "build_routing_probe_plan",
]
