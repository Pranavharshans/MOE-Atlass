"""Validated runtime integration for already-instantiated model sources.

Only :class:`InstanceSource` and :class:`CustomLoaderSource` are supported in
this phase. Hugging Face/local checkpoint loading remains deferred and this
package imports no model runtime.
"""

from .contracts import (
    CleanupCallback,
    CleanupError,
    CustomLoaderExecutionError,
    LoadedModel,
    LoadResult,
    PendingRuntimeCleanup,
    RuntimeArtifacts,
    RuntimeCleanupError,
    RuntimeLoadError,
    RuntimeObservation,
    RuntimeValidationError,
)
from .loader import load_custom, load_instance

__all__ = [
    "CleanupCallback",
    "CleanupError",
    "CustomLoaderExecutionError",
    "LoadResult",
    "LoadedModel",
    "PendingRuntimeCleanup",
    "RuntimeArtifacts",
    "RuntimeCleanupError",
    "RuntimeLoadError",
    "RuntimeObservation",
    "RuntimeValidationError",
    "load_custom",
    "load_instance",
]
