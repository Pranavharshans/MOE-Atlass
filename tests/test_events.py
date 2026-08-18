from __future__ import annotations

import math
import sys

import pytest
from pydantic import TypeAdapter, ValidationError

from moeatlas.core import (
    make_component_key,
    make_model_key,
    make_token_key,
    parse_token_key,
)
from moeatlas.events import Event, ExpertEvent, RoutingEvent, TokenEvent, TokenPhase

MODEL_KEY = make_model_key("org/demo-moe", "main")
LAYER_KEY = make_component_key(MODEL_KEY, "moe_layer", "layers.0", layer_index=0)
EXPERT_KEY = make_component_key(
    MODEL_KEY,
    "expert",
    "layers.0.experts.0",
    layer_index=0,
    expert_index=0,
)


def token_event(**overrides: object) -> TokenEvent:
    values: dict[str, object] = {
        "run_key": "run-1",
        "sequence_id": "sequence-1",
        "token_pos": 2,
        "token_id": 42,
        "token_text": " hello ",
        "phase": TokenPhase.PREFILL,
    }
    values.update(overrides)
    return TokenEvent(**values)


def routing_event(**overrides: object) -> RoutingEvent:
    values: dict[str, object] = {
        "token_key": token_event().token_key,
        "layer_key": LAYER_KEY,
        "rank": 0,
        "expert_key": EXPERT_KEY,
        "probability": 0.75,
        "selected": True,
    }
    values.update(overrides)
    return RoutingEvent(**values)


def expert_event(**overrides: object) -> ExpertEvent:
    values: dict[str, object] = {
        "token_key": token_event().token_key,
        "expert_key": EXPERT_KEY,
        "input_norm": 1.5,
        "latency_ms": 0.25,
    }
    values.update(overrides)
    return ExpertEvent(**values)


def test_token_identity_is_canonical_portable_and_roundtrippable() -> None:
    event = token_event()
    expected = make_token_key("run-1", "sequence-1", 2, 42, "prefill")

    assert event.token_key == expected
    assert parse_token_key(event.token_key) == event.token_key.removeprefix("token:")
    assert len(event.token_key.removeprefix("token:")) == 64
    assert event.token_text == " hello "
    assert TokenEvent.from_json(event.to_json(indent=2)) == event
    assert event.to_dict()["event_type"] == "token"
    assert event.to_dict()["schema_version"] == "1.0"

    # Presentation text is not identity: routing/expert events can continue to
    # refer to the same token if decoding text normalization changes.
    assert token_event(token_text="different").token_key == event.token_key
    assert token_event(token_pos=3).token_key != event.token_key
    assert token_event(phase=TokenPhase.DECODE).token_key != event.token_key


def test_token_identity_rejects_bad_or_mismatched_keys_and_coercion() -> None:
    with pytest.raises(ValidationError, match="does not match"):
        token_event(token_key="token:" + "0" * 64)
    with pytest.raises(ValidationError, match="canonical token"):
        token_event(token_key="")
    with pytest.raises(ValidationError, match="canonical token"):
        token_event(token_key="token:short")
    with pytest.raises(ValidationError, match="token_pos"):
        token_event(token_pos="2")
    with pytest.raises(ValidationError, match="token_id"):
        token_event(token_id=True)
    with pytest.raises(ValidationError, match="run_key"):
        token_event(run_key="/tmp/run")
    with pytest.raises(ValidationError, match="sequence_id"):
        token_event(sequence_id=" ")
    with pytest.raises(ValidationError, match="token_text"):
        token_event(token_text=42)
    with pytest.raises(ValidationError, match="token_pos"):
        token_event(token_pos=-1)


def test_routing_event_supports_partial_numeric_evidence_without_guessing() -> None:
    for evidence in (
        {"router_logit": -2.0, "probability": None},
        {"router_logit": None, "probability": 0.0},
        {"router_logit": None, "probability": None, "weight": 2.5},
    ):
        event = routing_event(**evidence)
        assert event.rank == 0
        assert event.selected is True

    with pytest.raises(ValidationError, match="requires.*evidence"):
        routing_event(router_logit=None, probability=None, weight=None)

    adapter = TypeAdapter(Event)
    restored = adapter.validate_json(routing_event().to_json())
    assert isinstance(restored, RoutingEvent)


def test_routing_event_enforces_canonical_keys_strict_numbers_and_bounds() -> None:
    with pytest.raises(ValidationError, match="different components"):
        routing_event(expert_key=LAYER_KEY)
    with pytest.raises(ValidationError, match="canonical token"):
        routing_event(token_key="token:bad")
    with pytest.raises(ValidationError, match="canonical component"):
        routing_event(layer_key="component:bad")
    with pytest.raises(ValidationError, match="rank"):
        routing_event(rank=True)
    with pytest.raises(ValidationError, match="rank"):
        routing_event(rank="0")
    with pytest.raises(ValidationError, match="rank"):
        routing_event(rank=-1)
    with pytest.raises(ValidationError, match="probability"):
        routing_event(probability=1.01)
    with pytest.raises(ValidationError, match="probability"):
        routing_event(probability=-0.01)
    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValidationError):
            routing_event(router_logit=value)
        with pytest.raises(ValidationError):
            routing_event(probability=value)
        with pytest.raises(ValidationError):
            routing_event(weight=value)
    with pytest.raises(ValidationError, match="selected"):
        routing_event(selected="true")
    with pytest.raises(ValidationError, match="extra_forbidden"):
        routing_event(unexpected=True)


def test_expert_event_requires_evidence_and_uses_explicit_latency_units() -> None:
    event = expert_event()
    assert event.latency_ms == 0.25
    assert ExpertEvent.from_json(event.to_json()) == event

    metadata_only = expert_event(input_norm=None, latency_ms=None, metadata={"source": "hook"})
    assert metadata_only.metadata == {"source": "hook"}
    with pytest.raises(ValidationError, match="requires a norm/latency"):
        expert_event(input_norm=None, latency_ms=None)

    for field in ("input_norm", "output_norm", "contribution_norm", "latency_ms"):
        with pytest.raises(ValidationError, match=field):
            expert_event(**{field: -0.1})
        with pytest.raises(ValidationError, match=field):
            expert_event(**{field: "0.1"})
        for value in (math.nan, math.inf, -math.inf):
            with pytest.raises(ValidationError):
                expert_event(**{field: value})


def test_expert_metadata_is_sorted_detached_and_deeply_immutable() -> None:
    source = {"z": {"values": [1, 2]}, "a": {"flag": True}}
    event = expert_event(input_norm=None, latency_ms=None, metadata=source)
    source["z"]["values"].append(99)

    assert event.metadata == {"a": {"flag": True}, "z": {"values": [1, 2]}}
    assert (
        event.to_json()
        == expert_event(
            input_norm=None,
            latency_ms=None,
            metadata={"a": {"flag": True}, "z": {"values": [1, 2]}},
        ).to_json()
    )
    with pytest.raises(TypeError, match="immutable"):
        event.metadata["new"] = True  # type: ignore[index]
    with pytest.raises(TypeError, match="immutable"):
        event.metadata["z"]["values"].append(3)  # type: ignore[attr-defined]

    with pytest.raises(ValidationError, match="JSON-serializable"):
        expert_event(metadata={"bad": object()})
    with pytest.raises(ValidationError, match="JSON-serializable"):
        expert_event(metadata={"bad": math.nan})
    with pytest.raises(ValidationError, match="JSON-serializable"):
        expert_event(metadata={"nested": {1: "non-string key"}})

    tuple_metadata = expert_event(
        input_norm=None,
        latency_ms=None,
        metadata={"tuple": (1, {"nested": [True, None]})},
    )
    assert tuple_metadata.to_dict()["metadata"] == {"tuple": [1, {"nested": [True, None]}]}
    assert ExpertEvent.from_json(tuple_metadata.to_json()) == tuple_metadata


def test_event_schema_is_strict_versioned_and_discriminated() -> None:
    adapter = TypeAdapter(Event)
    for original in (token_event(), routing_event(), expert_event()):
        restored = adapter.validate_json(original.to_json())
        assert type(restored) is type(original)
        assert restored == original

    with pytest.raises(ValidationError, match="schema_version"):
        TokenEvent.model_validate(token_event().to_dict() | {"schema_version": "2.0"})
    with pytest.raises(ValidationError, match="event_type"):
        TokenEvent.model_validate(token_event().to_dict() | {"event_type": "routing"})
    with pytest.raises(ValidationError, match="event_type"):
        TypeAdapter(Event).validate_json(
            token_event().to_json().replace('"event_type":"token"', '"event_type":"bad"')
        )


def test_event_import_does_not_load_model_runtime() -> None:
    assert not any(name in sys.modules for name in ("torch", "transformers", "safetensors"))
