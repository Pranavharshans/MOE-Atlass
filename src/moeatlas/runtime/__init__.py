"""Validated instance/custom execution and lazy optional model loading.

HF/local calls import optional model packages only after explicit immutable
resolution and policy preflight. Real checkpoint, cache, GPU, and fidelity
validation remains deferred to the model-validation ledger and final VM.
"""

from .contracts import (
    CleanupCallback,
    CleanupError,
    CustomLoaderExecutionError,
    LoadedModel,
    LoadResult,
    ModelLoadError,
    ModelObservationError,
    ModelRuntimeDependencyError,
    PendingRuntimeCleanup,
    RuntimeArtifacts,
    RuntimeCleanupError,
    RuntimeLoadError,
    RuntimeObservation,
    RuntimeValidationError,
)
from .loader import load_custom, load_instance
from .model_loader import load_huggingface, load_local

__all__ = [
    "CleanupCallback",
    "CleanupError",
    "CustomLoaderExecutionError",
    "LoadResult",
    "LoadedModel",
    "ModelLoadError",
    "ModelObservationError",
    "ModelRuntimeDependencyError",
    "PendingRuntimeCleanup",
    "RuntimeArtifacts",
    "RuntimeCleanupError",
    "RuntimeLoadError",
    "RuntimeObservation",
    "RuntimeValidationError",
    "load_custom",
    "load_huggingface",
    "load_instance",
    "load_local",
]
