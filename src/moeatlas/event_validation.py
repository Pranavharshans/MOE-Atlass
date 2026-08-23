"""Runtime-independent validation for normalized event collections."""

from __future__ import annotations

from .events import ExpertEvent, RoutingEvent, TokenEvent


def fresh_token_events(
    value: object,
    *,
    strict_sequence: bool = False,
) -> tuple[TokenEvent, ...]:
    """Return fresh token events after validating collection invariants."""

    if type(value) is not tuple:
        raise TypeError("token_events must be an exact tuple")
    if not value:
        raise ValueError("token_events must be non-empty")
    fresh_events: list[TokenEvent] = []
    seen: set[str] = set()
    run_key: str | None = None
    sequence_id: str | None = None
    decode_started = False
    for event in value:
        if type(event) is not TokenEvent:
            raise TypeError("token_events must contain exact TokenEvent objects")
        fresh = TokenEvent.model_validate(event.model_dump(mode="json"))
        if type(fresh) is not TokenEvent or fresh is event:
            raise TypeError("token event revalidation returned an unexpected type")
        if fresh.token_key in seen:
            raise ValueError("token_events must have unique token keys")
        seen.add(fresh.token_key)
        if run_key is None:
            run_key = fresh.run_key
            sequence_id = fresh.sequence_id
        elif fresh.run_key != run_key:
            raise ValueError("token_events must share one run_key")
        elif strict_sequence and fresh.sequence_id != sequence_id:
            raise ValueError("token_events must share one run_key and sequence_id")
        if fresh.phase.value == "decode":
            decode_started = True
        elif decode_started and strict_sequence:
            raise ValueError("prefill tokens cannot follow decode tokens")
        if strict_sequence and fresh.token_pos != len(fresh_events):
            raise ValueError("token_events must contain one contiguous canonical sequence 0..N-1")
        fresh_events.append(fresh)
    return tuple(fresh_events)


def fresh_routing_events(value: object) -> tuple[RoutingEvent, ...]:
    """Return fresh routing events after validating collection invariants."""

    if type(value) is not tuple:
        raise TypeError("routing_events must be an exact tuple")
    if not value:
        raise ValueError("routing_events must be non-empty")
    fresh_events: list[RoutingEvent] = []
    for event in value:
        if type(event) is not RoutingEvent:
            raise TypeError("routing_events must contain exact RoutingEvent objects")
        fresh = RoutingEvent.model_validate(event.model_dump(mode="json"))
        if type(fresh) is not RoutingEvent or fresh is event:
            raise TypeError("routing event revalidation returned an unexpected type")
        fresh_events.append(fresh)
    return tuple(fresh_events)


def validate_routing_links(
    token_events: tuple[TokenEvent, ...],
    routing_events: tuple[RoutingEvent, ...],
) -> None:
    """Validate deterministic links between normalized token and routing events."""

    token_keys = tuple(event.token_key for event in token_events)
    token_key_set = set(token_keys)
    token_positions = {token_key: index for index, token_key in enumerate(token_keys)}
    if any(event.selected is not True for event in routing_events):
        raise ValueError("routing_events must contain only selected events")
    seen_links: set[tuple[str, str, int]] = set()
    represented: set[str] = set()
    layer_order: dict[str, int] = {}
    layer_last_token: dict[str, int] = {}
    layer_last_rank: dict[str, int] = {}
    current_layer = -1
    for event in routing_events:
        if event.token_key not in token_key_set:
            raise ValueError("every routing event must reference a supplied token")
        link = (event.token_key, event.layer_key, event.rank)
        if link in seen_links:
            raise ValueError("routing_events must have unique token-layer-rank links")
        seen_links.add(link)
        represented.add(event.token_key)
        if event.layer_key not in layer_order:
            current_layer += 1
            layer_order[event.layer_key] = current_layer
            layer_last_token[event.layer_key] = -1
            layer_last_rank[event.layer_key] = -1
        elif layer_order[event.layer_key] != current_layer:
            raise ValueError("routing_events must use deterministic layer order")
        token_position = token_positions[event.token_key]
        if token_position < layer_last_token[event.layer_key] or (
            token_position == layer_last_token[event.layer_key]
            and event.rank <= layer_last_rank[event.layer_key]
        ):
            raise ValueError("routing_events must use deterministic token/rank order")
        if token_position != layer_last_token[event.layer_key]:
            layer_last_rank[event.layer_key] = -1
        layer_last_token[event.layer_key] = token_position
        layer_last_rank[event.layer_key] = event.rank
    if represented != token_key_set:
        raise ValueError("every supplied token must be represented by routing events")


def fresh_expert_events(value: object) -> tuple[ExpertEvent, ...]:
    """Return fresh expert events after validating collection invariants."""

    if type(value) is not tuple:
        raise TypeError("expert_events must be an exact tuple")
    fresh_events: list[ExpertEvent] = []
    seen: set[tuple[str, str]] = set()
    for event in value:
        if type(event) is not ExpertEvent:
            raise TypeError("expert_events must contain exact ExpertEvent objects")
        fresh = ExpertEvent.model_validate(event.model_dump(mode="json"))
        if type(fresh) is not ExpertEvent or fresh is event:
            raise TypeError("expert event revalidation returned an unexpected type")
        link = (fresh.token_key, fresh.expert_key)
        if link in seen:
            raise ValueError("expert_events must have unique token-expert links")
        seen.add(link)
        fresh_events.append(fresh)
    return tuple(fresh_events)


def validate_expert_links(
    token_events: tuple[TokenEvent, ...],
    expert_events: tuple[ExpertEvent, ...],
) -> None:
    """Validate deterministic links between normalized token and expert events."""

    if not token_events:
        raise ValueError("expert links require at least one supplied token")
    token_key_set = {event.token_key for event in token_events}
    for event in expert_events:
        if event.token_key not in token_key_set:
            raise ValueError("every expert event must reference a supplied token")


__all__ = [
    "fresh_expert_events",
    "fresh_routing_events",
    "fresh_token_events",
    "validate_expert_links",
    "validate_routing_links",
]
