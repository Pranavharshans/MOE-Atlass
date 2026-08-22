"""Static, model-runtime-independent MoE discovery contracts and scanner."""

from .models import (
    DiscoveryCandidate,
    DiscoveryEvidence,
    DiscoveryFacts,
    DiscoveryReport,
    DiscoverySignal,
)
from .routing_universe import bind_moe_layer_key, has_whole_word_moe_marker, trusted_routers
from .scanner import discover, scan

__all__ = [
    "DiscoveryCandidate",
    "DiscoveryEvidence",
    "DiscoveryFacts",
    "DiscoveryReport",
    "DiscoverySignal",
    "bind_moe_layer_key",
    "discover",
    "has_whole_word_moe_marker",
    "scan",
    "trusted_routers",
]
