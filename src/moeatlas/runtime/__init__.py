"""Validated instance/custom execution and lazy optional model loading.

HF/local calls import optional model packages only after explicit immutable
resolution and policy preflight. Real checkpoint, cache, GPU, and fidelity
validation remains deferred to the model-validation ledger and final VM.
`load_and_scan()` composes those resolved HF/local loaders with static
discovery after which cleanup succeeds; it does not add inference or CLI
loading.
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
from .scan import load_and_scan

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
    "load_and_scan",
    "load_instance",
    "load_local",
]
