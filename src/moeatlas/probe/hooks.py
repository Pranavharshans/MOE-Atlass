"""Transactional lifecycle-safe hook registration for duck-typed modules."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ..core import validate_stable_identifier
from .plan import HookPoint, ProbeLevel, ProbePlan
from .resolver import ProbeResolutionError, ResolvedProbePlan, resolve_probe_plan


class HookLifecycleError(RuntimeError):
    """Raised when a manager is entered or reused outside its lifecycle."""


class HookRegistrationError(RuntimeError):
    """Raised when callback or hook registration cannot be completed."""


@dataclass(frozen=True)
class HookBinding:
    """The callback lookup key for one module path and hook point."""

    module_path: str
    hook_point: HookPoint

    def __post_init__(self) -> None:
        validate_stable_identifier(self.module_path, field_name="hook binding module_path")
        if not isinstance(self.hook_point, HookPoint):
            try:
                object.__setattr__(self, "hook_point", HookPoint(self.hook_point))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"unknown hook point {self.hook_point!r}") from exc


@dataclass(frozen=True)
class HookRemovalFailure:
    """One handle-removal failure retained in aggregate cleanup reporting."""

    binding: HookBinding
    error: BaseException


class HookCleanupError(RuntimeError):
    """Raised after normal exit when one or more handles could not be removed."""

    def __init__(self, failures: tuple[HookRemovalFailure, ...]) -> None:
        self.failures = failures
        summary = "; ".join(
            f"{failure.binding.module_path}:{failure.binding.hook_point.value}"
            f" ({type(failure.error).__name__})"
            for failure in failures
        )
        super().__init__(f"hook cleanup failed for {len(failures)} handle(s): {summary}")


_HOOK_METHODS = {
    HookPoint.FORWARD_PRE: "register_forward_pre_hook",
    HookPoint.FORWARD: "register_forward_hook",
    HookPoint.FULL_BACKWARD: "register_full_backward_hook",
}

HookCallback = Callable[..., Any]


class HookManager:
    """Register callbacks transactionally and always attempt reverse cleanup.

    ``callbacks`` maps :class:`HookBinding` objects to caller-owned callback
    functions. The manager passes the runtime's callback arguments through
    unchanged to a caller callback, then discards that callback's return value
    and returns ``None`` to the runtime so observation cannot replace inputs,
    outputs, or gradients.
    """

    def __init__(
        self,
        model: object,
        plan: ProbePlan | ResolvedProbePlan,
        callbacks: Mapping[HookBinding, HookCallback],
    ) -> None:
        if not isinstance(plan, ProbePlan | ResolvedProbePlan):
            raise TypeError("plan must be a validated ProbePlan or ResolvedProbePlan")
        if not isinstance(callbacks, Mapping):
            raise TypeError("callbacks must map HookBinding objects to callables")
        self._model = model
        self._plan = plan
        self._callbacks = dict(callbacks)
        self._started = False
        self._active = False
        self._resolved: ResolvedProbePlan | None = None
        self._handles: list[tuple[HookBinding, object]] = []

    @property
    def active(self) -> bool:
        return self._active

    @property
    def resolved_plan(self) -> ResolvedProbePlan | None:
        return self._resolved

    @property
    def installed_count(self) -> int:
        return len(self._handles)

    def __enter__(self) -> HookManager:
        if self._started:
            raise HookLifecycleError(
                "HookManager instances are single-use and cannot be re-entered"
            )
        self._started = True
        try:
            if isinstance(self._plan, ResolvedProbePlan):
                try:
                    rebound = resolve_probe_plan(self._plan.plan, self._model)
                except ProbeResolutionError as exc:
                    raise HookRegistrationError(
                        f"resolved probe plan cannot bind to the provided model: {exc}"
                    ) from exc
                self._validate_resolved_model(self._plan, rebound)
                self._resolved = rebound
            else:
                self._resolved = resolve_probe_plan(self._plan, self._model)
            if self._resolved.plan.level is ProbeLevel.INTERVENTION:
                raise HookRegistrationError(
                    "INTERVENTION plans are defined but passive HookManager execution "
                    "is not implemented"
                )
            bindings = self._expected_bindings(self._resolved)
            self._validate_callbacks(bindings)
            for binding, module in bindings:
                method = getattr(module, _HOOK_METHODS[binding.hook_point])
                callback = self._callbacks[binding]
                handle = method(_passive_callback(callback))
                self._handles.append((binding, handle))
                if not callable(getattr(handle, "remove", None)):
                    raise HookRegistrationError(
                        f"registration for {binding.module_path!r} returned a handle "
                        "without remove()"
                    )
            self._active = True
            return self
        except BaseException as original:
            failures = self._remove_handles()
            self._active = False
            self._annotate(original, failures)
            raise

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> bool:
        if not self._active:
            return False
        failures = self._remove_handles()
        self._active = False
        if failures:
            if exc_value is not None:
                self._annotate(exc_value, failures)
                return False
            raise HookCleanupError(failures)
        return False

    def close(self) -> None:
        """Idempotently remove installed handles outside a ``with`` block."""

        if not self._handles:
            return
        failures = self._remove_handles()
        self._active = False
        if failures:
            raise HookCleanupError(failures)

    @staticmethod
    def _expected_bindings(
        resolved: ResolvedProbePlan,
    ) -> list[tuple[HookBinding, object]]:
        return [
            (HookBinding(path, hook_point), module)
            for path, hook_point, module in resolved.bindings
        ]

    @staticmethod
    def _validate_resolved_model(
        supplied: ResolvedProbePlan,
        rebound: ResolvedProbePlan,
    ) -> None:
        if len(supplied.targets) != len(rebound.targets):
            raise HookRegistrationError(
                "resolved probe plan target count does not match the provided model"
            )
        for supplied_target, rebound_target in zip(
            supplied.targets,
            rebound.targets,
            strict=True,
        ):
            if supplied_target.target != rebound_target.target:
                raise HookRegistrationError(
                    "resolved probe plan target metadata does not match the source plan"
                )
            if supplied_target.module is not rebound_target.module:
                raise HookRegistrationError(
                    "resolved probe plan module identity does not match the provided model"
                )

    def _validate_callbacks(self, bindings: list[tuple[HookBinding, object]]) -> None:
        expected = {binding for binding, _ in bindings}
        for binding, callback in self._callbacks.items():
            if not isinstance(binding, HookBinding):
                raise HookRegistrationError("callback keys must be HookBinding objects")
            if not callable(callback):
                raise HookRegistrationError(f"callback for {binding!r} is not callable")
        provided = set(self._callbacks)
        missing = sorted(expected - provided, key=_binding_sort_key)
        extra = sorted(provided - expected, key=_binding_sort_key)
        if missing or extra:
            details: list[str] = []
            if missing:
                details.append(f"missing callbacks: {missing!r}")
            if extra:
                details.append(f"callbacks have no selected target: {extra!r}")
            raise HookRegistrationError("; ".join(details))

    def _remove_handles(self) -> list[HookRemovalFailure]:
        handles = list(reversed(self._handles))
        failed_handles: list[tuple[HookBinding, object]] = []
        failures: list[HookRemovalFailure] = []
        for binding, handle in handles:
            try:
                handle.remove()
            except BaseException as exc:
                failures.append(HookRemovalFailure(binding=binding, error=exc))
                failed_handles.append((binding, handle))
        self._handles = list(reversed(failed_handles))
        return failures

    @staticmethod
    def _annotate(
        original: BaseException,
        failures: list[HookRemovalFailure],
    ) -> None:
        if not failures:
            return
        details = "; ".join(
            f"{failure.binding.module_path}:{failure.binding.hook_point.value} "
            f"({type(failure.error).__name__})"
            for failure in failures
        )
        original.add_note(
            f"hook cleanup failures were suppressed to preserve the original error: {details}"
        )


def _binding_sort_key(binding: HookBinding) -> tuple[str, int]:
    return (
        binding.module_path,
        {HookPoint.FORWARD_PRE: 0, HookPoint.FORWARD: 1, HookPoint.FULL_BACKWARD: 2}[  # noqa: E501
            binding.hook_point
        ],
    )


def _passive_callback(callback: HookCallback) -> HookCallback:
    """Forward exact callback arguments while suppressing callback returns."""

    def wrapper(*args: Any, **kwargs: Any) -> None:
        callback(*args, **kwargs)
        return None

    return wrapper


__all__ = [
    "HookBinding",
    "HookCallback",
    "HookCleanupError",
    "HookLifecycleError",
    "HookManager",
    "HookRegistrationError",
    "HookRemovalFailure",
]
