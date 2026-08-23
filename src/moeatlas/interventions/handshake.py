"""Reversible pass-through handshake for Hugging Face expert dispatch.

The handshake temporarily registers delegates around the model's declared
expert implementations, executes one caller-owned forward, and restores both
the model configuration and temporary registry entries before returning. It never
changes routing indices, weights, expert outputs, or the returned model value.
"""

from __future__ import annotations

import importlib
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

from .transformers import (
    ExpertBackendDiscoveryStatus,
    TransformersInterventionError,
    discover_huggingface_expert_backends,
)

_Result = TypeVar("_Result")


class ExpertBackendHandshakeStatus(str, Enum):
    """Outcome of one pass-through expert-backend handshake."""

    VERIFIED = "verified"
    NOT_EXERCISED = "not_exercised"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class ExpertBackendHandshakeReport:
    """Bounded evidence that a temporary delegate ran and was restored."""

    status: ExpertBackendHandshakeStatus
    invocation_counts: tuple[tuple[str, int], ...]
    restored: bool
    reason: str
    source: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "invocation_counts": dict(self.invocation_counts),
            "restored": self.restored,
            "reason": self.reason,
            "source": self.source,
        }


def _unavailable(reason: str) -> ExpertBackendHandshakeReport:
    return ExpertBackendHandshakeReport(
        status=ExpertBackendHandshakeStatus.UNAVAILABLE,
        invocation_counts=(),
        restored=True,
        reason=reason,
        source=None,
    )


def _transformers_expert_registry() -> object | None:
    try:
        module = importlib.import_module("transformers.integrations.moe")
        return getattr(module, "ALL_EXPERTS_FUNCTIONS", None)
    except Exception:
        return None


def _registry_function(registry: object, implementation: str) -> Callable[..., Any] | None:
    try:
        candidate = registry[implementation]  # type: ignore[index]
    except Exception:
        return None
    return candidate if callable(candidate) else None


def _eager_forward(module: object, *args: object, **kwargs: object) -> object:
    """Call the original function retained by ``functools.wraps``."""

    try:
        forward = getattr(type(module), "forward")
        original = getattr(forward, "__wrapped__")
    except Exception as exc:
        raise TransformersInterventionError(
            "eager expert backend does not expose its original forward"
        ) from exc
    if not callable(original):
        raise TransformersInterventionError(
            "eager expert backend does not expose its original forward"
        )
    return original(module, *args, **kwargs)


def run_huggingface_expert_handshake(
    model: object,
    execute: Callable[[], _Result],
    *,
    registry: object | None = None,
) -> tuple[_Result, ExpertBackendHandshakeReport]:
    """Run ``execute`` once through temporary, identity-preserving delegates.

    Unsupported interfaces take the ordinary execution path and return an
    explicit unavailable report. Once mutation begins, exact restoration is a
    hard boundary: inability to restore raises instead of allowing a dirty
    model to continue.
    """

    if not callable(execute):
        raise TypeError("execute must be callable")
    discovery = discover_huggingface_expert_backends(model)
    if discovery.status is not ExpertBackendDiscoveryStatus.OBSERVED:
        return execute(), _unavailable(discovery.reason)
    try:
        getter = getattr(model, "get_experts_implementation")
        setter = getattr(model, "set_experts_implementation")
    except Exception:
        return execute(), _unavailable("loaded model cannot switch expert backends safely")
    if not callable(getter) or not callable(setter):
        return execute(), _unavailable("loaded model cannot switch expert backends safely")
    active_registry = registry if registry is not None else _transformers_expert_registry()
    if active_registry is None:
        return execute(), _unavailable("Hugging Face expert backend registry is unavailable")
    set_item = getattr(type(active_registry), "__setitem__", None)
    delete_item = getattr(type(active_registry), "__delitem__", None)
    if not callable(set_item) or not callable(delete_item):
        return execute(), _unavailable("Hugging Face expert backend registry is not reversible")

    try:
        snapshot = getter()
    except Exception:
        return execute(), _unavailable("expert backend snapshot could not be repeated")
    if not isinstance(snapshot, Mapping):
        return execute(), _unavailable("expert backend snapshot changed shape")
    snapshot = dict(snapshot)
    implementations = sorted(
        {
            implementation
            for implementation in snapshot.values()
            if isinstance(implementation, str) and implementation
        }
    )
    if not implementations:
        return execute(), _unavailable("no active expert backend was declared")

    invocation_counts = {implementation: 0 for implementation in implementations}
    replacements: dict[str, str | None] = dict(snapshot)
    registered: list[str] = []
    switched = False

    def restore() -> None:
        failures: list[BaseException] = []
        if switched:
            try:
                setter(snapshot)
            except BaseException as exc:
                failures.append(exc)
        while registered:
            name = registered.pop()
            try:
                del active_registry[name]  # type: ignore[index]
            except BaseException as exc:
                failures.append(exc)
        if failures:
            raise TransformersInterventionError(
                f"expert backend handshake restoration failed ({len(failures)} error(s))"
            ) from failures[0]

    try:
        for implementation in implementations:
            original = (
                _eager_forward
                if implementation == "eager"
                else _registry_function(active_registry, implementation)
            )
            if original is None:
                restore()
                return execute(), _unavailable(
                    "an active expert backend is absent from the Hugging Face registry"
                )
            temporary_name = f"moeatlas_passthrough_{uuid.uuid4().hex}"

            def passthrough(
                module: object,
                *args: object,
                _implementation: str = implementation,
                _original: Callable[..., Any] = original,
                **kwargs: object,
            ) -> object:
                invocation_counts[_implementation] += 1
                return _original(module, *args, **kwargs)

            active_registry[temporary_name] = passthrough  # type: ignore[index]
            registered.append(temporary_name)
            for scope, active in tuple(replacements.items()):
                if active == implementation:
                    replacements[scope] = temporary_name
        switched = True
        setter(replacements)
    except Exception:
        restore()
        return execute(), _unavailable(
            "loaded model rejected the temporary pass-through expert backend"
        )
    except BaseException:
        restore()
        raise
    try:
        result = execute()
    except Exception:
        was_exercised = any(invocation_counts.values())
        restore()
        if not was_exercised:
            return execute(), _unavailable(
                "the model forward did not use the Hugging Face expert backend registry"
            )
        raise
    except BaseException:
        restore()
        raise
    restore()

    counts = tuple(sorted(invocation_counts.items()))
    total = sum(count for _implementation, count in counts)
    if total:
        return result, ExpertBackendHandshakeReport(
            status=ExpertBackendHandshakeStatus.VERIFIED,
            invocation_counts=counts,
            restored=True,
            reason="pass-through expert delegates were exercised and restored",
            source="transformers.integrations.moe.ALL_EXPERTS_FUNCTIONS",
        )
    return result, ExpertBackendHandshakeReport(
        status=ExpertBackendHandshakeStatus.NOT_EXERCISED,
        invocation_counts=counts,
        restored=True,
        reason="the forward completed but did not call the temporary expert delegates",
        source="transformers.integrations.moe.ALL_EXPERTS_FUNCTIONS",
    )


__all__ = [
    "ExpertBackendHandshakeReport",
    "ExpertBackendHandshakeStatus",
    "run_huggingface_expert_handshake",
]
