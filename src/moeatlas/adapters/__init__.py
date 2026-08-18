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

__all__ = [
    "AdapterContractError",
    "AdapterDescriptor",
    "AdapterDetection",
    "AdapterExecutionError",
    "AdapterInspection",
    "StaticSemanticAdapter",
    "inspect_static_adapter",
    "MixtralStaticAdapter",
]
