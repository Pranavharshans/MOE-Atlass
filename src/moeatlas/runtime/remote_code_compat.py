"""Scoped compatibility bridges for explicitly trusted Hub remote code.

Bridges are capability based, never repository-name based.  They exist only
while Transformers imports caller-approved remote code, are removed again
after the load attempt, and are reported as provenance warnings.  This is for
small, semantics-preserving API removals; it is not a dependency installer or
an excuse to execute untrusted code.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

_BRIDGE_LOCK = threading.RLock()


@dataclass(frozen=True, slots=True)
class AppliedCompatibilityBridge:
    """One temporary, audited compatibility mutation."""

    name: str
    target: str
    reason: str

    def warning(self) -> str:
        return f"remote-code compatibility bridge applied: {self.name}"


def _torch_fx_available() -> bool:
    """Legacy Transformers v4 predicate under the modern torch baseline."""

    return True


@contextmanager
def remote_code_compatibility(
    transformers: Any,
    *,
    enabled: bool,
) -> Iterator[tuple[AppliedCompatibilityBridge, ...]]:
    """Temporarily install approved missing APIs for trusted remote code.

    The caller must pass ``enabled=True`` only after its existing remote-code
    acknowledgement checks.  Existing attributes are never replaced.
    """

    if type(enabled) is not bool:
        raise TypeError("enabled must be an exact bool")
    if not enabled:
        yield ()
        return
    try:
        utilities = getattr(transformers, "utils")
        import_utils = getattr(utilities, "import_utils")
    except Exception:
        yield ()
        return

    applied: list[tuple[object, str, AppliedCompatibilityBridge]] = []
    with _BRIDGE_LOCK:
        if not hasattr(import_utils, "is_torch_fx_available"):
            bridge = AppliedCompatibilityBridge(
                name="transformers.is_torch_fx_available",
                target="transformers.utils.import_utils.is_torch_fx_available",
                reason="legacy remote code imports a predicate removed in Transformers 5",
            )
            setattr(import_utils, "is_torch_fx_available", _torch_fx_available)
            applied.append((import_utils, "is_torch_fx_available", bridge))
        try:
            yield tuple(item[2] for item in applied)
        finally:
            for owner, attribute, _bridge in reversed(applied):
                try:
                    delattr(owner, attribute)
                except Exception:
                    # Do not replace the primary model-load outcome.  The
                    # process worker boundary remains the hard cleanup layer.
                    pass


__all__ = ["AppliedCompatibilityBridge", "remote_code_compatibility"]
