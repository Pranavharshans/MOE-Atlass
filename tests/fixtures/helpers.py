"""Shared model-free event fixtures for storage contract tests."""

from __future__ import annotations

from moeatlas.events import RoutingEvent, TokenEvent, TokenPhase
from moeatlas.runtime import RoutingForwardResult


def plain_routing_result(
    *,
    run_key: str = "run-plain-1",
    token_count: int = 1,
    layer_count: int = 2,
    top_k: int = 2,
) -> RoutingForwardResult:
    """Build an exact, complete ``RoutingForwardResult`` without any model."""

    tokens = tuple(
        TokenEvent(
            run_key=run_key,
            sequence_id="sequence-1",
            token_pos=pos,
            token_id=100 + pos,
            token_text=str(pos),
            phase=TokenPhase.PREFILL,
        )
        for pos in range(token_count)
    )
    routing: list[RoutingEvent] = []
    for layer in range(layer_count):
        for token in tokens:
            for rank in range(top_k):
                routing.append(
                    RoutingEvent(
                        token_key=token.token_key,
                        layer_key="component:" + format(layer + 1, "064x"),
                        rank=rank,
                        expert_key="component:"
                        + format(1000 + layer * 16 + rank, "064x"),
                        router_logit=float(rank),
                        selected=True,
                    )
                )
    return RoutingForwardResult(
        output=object(),
        token_events=tokens,
        routing_events=tuple(routing),
    )


__all__ = ["plain_routing_result"]
