"""Built-in executor plugins for the ``moeatlas.executors`` group.

The CLI resolves ``--executor`` names through this registry first and falls
back to environment entry points, mirroring the adapter-registry precedent:
shipped built-ins win deterministically and third-party plugins publish under
the same group name. Construction is plan-bound but lazy — no module here
imports ``torch`` or ``transformers`` at import time.
"""

from __future__ import annotations

from typing import Any

from ..loading import LoadingPlan

EXECUTOR_ENTRY_POINT_GROUP = "moeatlas.executors"
"""Entry-point group third-party executor plugins publish under."""

EXECUTOR_NAME = "transformers-routing"
"""The built-in real-model executor shipped with this package."""


def builtin_executor_names() -> tuple[str, ...]:
    """Names of the executors shipped inside this package."""

    return (EXECUTOR_NAME,)


def build_builtin_executor(name: str, plan: LoadingPlan) -> Any | None:
    """Bind one built-in executor to a validated loading plan.

    Returns ``None`` when the name is not a built-in so callers can fall back
    to entry-point discovery. Unknown names never raise from here.
    """

    if type(name) is not str:
        raise TypeError("name must be a string")
    if not isinstance(plan, LoadingPlan):
        raise TypeError("plan must be a validated LoadingPlan")
    if name == EXECUTOR_NAME:
        from .transformers_routing import TransformersRoutingExecutor

        return TransformersRoutingExecutor(plan)
    return None


__all__ = [
    "EXECUTOR_ENTRY_POINT_GROUP",
    "EXECUTOR_NAME",
    "build_builtin_executor",
    "builtin_executor_names",
]
