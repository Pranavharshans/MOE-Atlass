"""Runtime-only objects, result metadata, and retryable cleanup lifecycle."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Self

from ..core import DType, ModelManifest
from ..loading import LoadingPlan


class RuntimeLoadError(RuntimeError):
    """Base error for validated runtime execution failures."""


class RuntimeValidationError(RuntimeLoadError):
    """Raised when trusted runtime observations cannot form a manifest."""


class CustomLoaderExecutionError(RuntimeLoadError):
    """Raised when an explicitly opted-in custom loader cannot execute."""


class ModelRuntimeDependencyError(RuntimeLoadError):
    """Raised when the optional model-runtime extra is unavailable."""


class ModelLoadError(RuntimeLoadError):
    """Raised for a stage-specific HF/local loading failure."""

    def __init__(self, stage: str, message: str) -> None:
        self.stage = stage
        super().__init__(f"{stage}: {message}")


class ModelObservationError(RuntimeLoadError):
    """Raised when loaded runtime facts cannot be observed safely."""


class RuntimeCleanupError(RuntimeLoadError):
    """Raised when owned runtime cleanup fails and remains retryable."""

    def __init__(self, failures: tuple[BaseException, ...]) -> None:
        self.failures = failures
        summary = ", ".join(type(failure).__name__ for failure in failures)
        super().__init__(f"runtime cleanup failed for {len(failures)} callback(s): {summary}")


CleanupError = RuntimeCleanupError
CleanupCallback = Callable[[], None]


class PendingRuntimeCleanup:
    """Retryable cleanup handle attached after validation cleanup failure."""

    def __init__(self, callback: CleanupCallback) -> None:
        if not callable(callback):
            raise TypeError("PendingRuntimeCleanup callback must be callable")
        self._callback: CleanupCallback | None = callback
        self._last_error: BaseException | None = None

    @property
    def pending(self) -> bool:
        """Whether cleanup still needs a successful callback attempt."""

        return self._callback is not None

    @property
    def last_error(self) -> BaseException | None:
        """The most recent callback failure, if any."""

        return self._last_error

    def retry(self) -> None:
        """Retry cleanup; successful completion makes later retries no-ops."""

        if self._callback is None:
            return
        callback = self._callback
        try:
            callback()
        except RuntimeCleanupError as exc:
            self._last_error = exc
            setattr(exc, "pending_cleanup", self)
            raise
        except BaseException as exc:
            self._last_error = exc
            error = RuntimeCleanupError((exc,))
            setattr(error, "pending_cleanup", self)
            raise error from exc
        self._callback = None
        self._last_error = None


@dataclass(frozen=True, slots=True)
class RuntimeArtifacts:
    """Trusted runtime objects and observed facts supplied to the loader.

    This is deliberately a dataclass rather than a Pydantic model: ``model``,
    ``tokenizer``, and ``cleanup`` are process-local objects and must never be
    serialized or included in a loading-plan identity. ``owns_cleanup`` is an
    explicit ownership transfer. It defaults to ``False`` so instance objects
    remain caller-owned and are never closed by :class:`LoadedModel`.
    """

    model: object
    tokenizer: object
    config: object
    architecture: str
    dtype: DType
    device_map: Mapping[str, str]
    cleanup: CleanupCallback | None = None
    owns_cleanup: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.dtype, DType):
            raise TypeError("RuntimeArtifacts.dtype must be an observed core DType")
        if not isinstance(self.device_map, Mapping):
            raise TypeError("RuntimeArtifacts.device_map must be a mapping")
        if self.cleanup is not None and not callable(self.cleanup):
            raise TypeError("RuntimeArtifacts.cleanup must be callable or None")
        if not isinstance(self.owns_cleanup, bool):
            raise TypeError("RuntimeArtifacts.owns_cleanup must be a bool")
        if (self.cleanup is None) != (not self.owns_cleanup):
            raise ValueError("RuntimeArtifacts.cleanup and owns_cleanup must be provided together")


RuntimeObservation = RuntimeArtifacts


@dataclass(slots=True)
class LoadedModel:
    """A validated runtime object pair and its canonical observed manifest."""

    model: object | None
    tokenizer: object | None
    plan: LoadingPlan
    manifest: ModelManifest
    warnings: tuple[str, ...]
    _cleanup_callback: CleanupCallback | None = field(default=None, repr=False)
    _owns_cleanup: bool = field(default=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _context_active: bool = field(default=False, init=False, repr=False)

    @property
    def loading_plan(self) -> LoadingPlan:
        """Alias that makes the validated plan role explicit to callers."""

        return self.plan

    @property
    def model_manifest(self) -> ModelManifest:
        """Alias for callers that distinguish runtime and manifest objects."""

        return self.manifest

    @property
    def closed(self) -> bool:
        """Whether owned cleanup has completed or no cleanup was transferred."""

        return self._closed

    @property
    def owns_cleanup(self) -> bool:
        """Whether this result currently owns a retryable cleanup callback."""

        return self._owns_cleanup and self._cleanup_callback is not None

    def close(self) -> None:
        """Close owned runtime resources, retaining failed cleanup for retry."""

        if self._closed:
            return
        if not self._owns_cleanup or self._cleanup_callback is None:
            self._closed = True
            return

        callback = self._cleanup_callback
        try:
            callback()
        except RuntimeCleanupError:
            self._closed = False
            raise
        except BaseException as exc:
            self._closed = False
            raise RuntimeCleanupError((exc,)) from exc

        self._cleanup_callback = None
        self._owns_cleanup = False
        self._closed = True
        self.model = None
        self.tokenizer = None

    def __enter__(self) -> Self:
        if self._closed:
            raise RuntimeLoadError("a closed LoadedModel cannot be entered")
        if self._context_active:
            raise RuntimeLoadError("LoadedModel context is already active")
        self._context_active = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> bool:
        self._context_active = False
        if exc_value is not None:
            try:
                self.close()
            except RuntimeCleanupError as cleanup_error:
                _add_cleanup_note(exc_value, cleanup_error)
            return False
        self.close()
        return False


LoadResult = LoadedModel


def _add_cleanup_note(original: BaseException, cleanup_error: RuntimeCleanupError) -> None:
    """Annotate a body/validation error without replacing its root cause."""

    original.add_note(
        "runtime cleanup also failed; the owned cleanup callback remains retryable "
        f"({len(cleanup_error.failures)} failure(s))"
    )


def _attach_pending_cleanup(original: BaseException, pending: PendingRuntimeCleanup) -> None:
    """Expose a retry handle without changing the original raised exception."""

    setattr(original, "pending_cleanup", pending)
    setattr(original, "pending_runtime_cleanup", pending)


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
]
