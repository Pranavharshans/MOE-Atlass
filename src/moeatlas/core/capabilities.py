"""Stable capability labels used by MoEAtlas manifests.

The labels intentionally describe *what was observed and validated*, not what
an adapter hopes to observe. A manifest may therefore report a lower tier or
``UNSUPPORTED`` without making a claim about the model's internal semantics.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from types import MappingProxyType
from typing import Final


class CapabilityLabel(str, Enum):
    """The capability tiers defined by the MoEAtlas PRD."""

    FULL = "FULL"
    ROUTING = "ROUTING"
    MODULE = "MODULE"
    STRUCTURE = "STRUCTURE"
    EXPERIMENTAL = "EXPERIMENTAL"
    UNSUPPORTED = "UNSUPPORTED"

    def __str__(self) -> str:
        return self.value


# A short alias is useful in adapter code while CapabilityLabel remains the
# explicit public name used in serialized manifests.
Capability = CapabilityLabel

CAPABILITY_LABELS: Final[tuple[CapabilityLabel, ...]] = tuple(CapabilityLabel)

CAPABILITY_SEMANTICS: Final[Mapping[CapabilityLabel, str]] = MappingProxyType(
    {
        CapabilityLabel.FULL: (
            "Validated router scores/top-k, expert activity, and supported "
            "interventions are captured with semantic fidelity."
        ),
        CapabilityLabel.ROUTING: (
            "Reliable router or top-k capture is available; expert internals "
            "may remain packed or fused."
        ),
        CapabilityLabel.MODULE: (
            "Module-level inputs/outputs are visible, but semantic MoE "
            "decoding is incomplete or unavailable."
        ),
        CapabilityLabel.STRUCTURE: (
            "Static module, configuration, or weight structure is available; "
            "inference capture has not been validated."
        ),
        CapabilityLabel.EXPERIMENTAL: (
            "Capture works but has not been certified against a golden or native reference."
        ),
        CapabilityLabel.UNSUPPORTED: (
            "The requested internal operation cannot currently be observed on the selected backend."
        ),
    }
)


class CaptureSource(str, Enum):
    """Where a captured semantic value came from."""

    NATIVE_OUTPUT = "native_output"
    MODULE_HOOK = "module_hook"
    ADAPTER_DECODER = "adapter_decoder"
    FUSED_DECODER = "fused_decoder"
    STATIC_STRUCTURE = "static_structure"
    UNKNOWN = "unknown"

    def __str__(self) -> str:
        return self.value


def capability_semantics(label: CapabilityLabel | str) -> str:
    """Return the documented meaning of a capability label.

    ``ValueError`` is used for unknown strings so adapter authors receive an
    actionable error before a non-standard label enters a manifest.
    """

    try:
        normalized = label if isinstance(label, CapabilityLabel) else CapabilityLabel(label)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in CapabilityLabel)
        raise ValueError(f"unknown capability label {label!r}; expected one of: {allowed}") from exc
    return CAPABILITY_SEMANTICS[normalized]
