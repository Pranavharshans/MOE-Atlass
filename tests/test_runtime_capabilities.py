"""Model-free tests for the routing decode capability contracts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from moeatlas.events import RoutingEvent
from moeatlas.runtime import (
    RouterPayloadShape,
    RoutingDecodeCapability,
    RoutingDecodeError,
    ScoreSemantics,
    native_id_map,
    validate_decoded_routing,
)
from moeatlas.runtime.capabilities import ROUTING_DECODE_CAPABILITY_SCHEMA_VERSION

from .test_adapters_universe import _layer, _universe

ROOT = Path(__file__).resolve().parents[1]

_TOKEN_KEY = "token:" + "5" * 64


def _key_of(layer_index: int, native_id: int) -> str:
    layer = _universe().layers[layer_index]
    return native_id_map(layer)[native_id]


class DictArraysCapability:
    """Unknown-family fake: mapping-keyed per-layer arrays with probabilities."""

    payload_shape = RouterPayloadShape.DICT_ARRAYS
    score_semantics = ScoreSemantics.PROBABILITIES

    def decode(
        self,
        payload: object,
        *,
        universe,  # noqa: ANN001 - fake mirrors the protocol loosely
        token_key: str,
    ) -> tuple[RoutingEvent, ...]:
        events: list[RoutingEvent] = []
        for layer in universe.layers:
            entry = payload[f"layer_{layer.layer_index}"]
            id_map = native_id_map(layer)
            for rank, (native_id, probability) in enumerate(
                zip(entry["ids"], entry["p"], strict=True)
            ):
                events.append(
                    RoutingEvent(
                        token_key=token_key,
                        layer_key=layer.moe_layer_key,
                        rank=rank,
                        expert_key=id_map[native_id],
                        probability=probability,
                        selected=True,
                    )
                )
        return tuple(events)


class AssignmentOnlyCapability:
    """Unknown-family fake: a 3-D nested index array and no scores at all."""

    payload_shape = RouterPayloadShape.ASSIGNMENT_INDICES
    score_semantics = ScoreSemantics.NONE

    def decode(
        self,
        payload: object,
        *,
        universe,  # noqa: ANN001
        token_key: str,
    ) -> tuple[RoutingEvent, ...]:
        events: list[RoutingEvent] = []
        for position, layer in enumerate(universe.layers):
            id_map = native_id_map(layer)
            for rank, native_id in enumerate(payload[position]):
                events.append(
                    RoutingEvent(
                        token_key=token_key,
                        layer_key=layer.moe_layer_key,
                        rank=rank,
                        expert_key=id_map[native_id],
                        weight=1.0,
                        selected=True,
                    )
                )
        return tuple(events)


class LogitsTupleCapability:
    """Legacy-shaped fake: per-layer logit vectors in a plain tuple."""

    payload_shape = RouterPayloadShape.LOGITS_TUPLE
    score_semantics = ScoreSemantics.LOGITS

    def decode(
        self,
        payload: object,
        *,
        universe,  # noqa: ANN001
        token_key: str,
    ) -> tuple[RoutingEvent, ...]:
        events: list[RoutingEvent] = []
        for layer, logits in zip(universe.layers, payload, strict=True):
            order = sorted(range(len(logits)), key=lambda i: (-logits[i], i))
            for rank, expert_position in enumerate(order[: layer.routed_top_k]):
                events.append(
                    RoutingEvent(
                        token_key=token_key,
                        layer_key=layer.moe_layer_key,
                        rank=rank,
                        expert_key=layer.expert_keys[expert_position],
                        router_logit=logits[expert_position],
                        selected=True,
                    )
                )
        return tuple(events)


def _dict_payload() -> dict[str, dict[str, list[float]]]:
    return {
        "layer_0": {"ids": [0, 3], "p": [0.7, 0.3]},
        "layer_1": {"ids": [9, 2, 5], "p": [0.5, 0.3, 0.2]},
    }


# ---------------------------------------------------------------------------
# Protocol conformance and happy paths across payload shapes
# ---------------------------------------------------------------------------


def test_fake_capabilities_satisfy_the_protocol() -> None:
    for fake in (DictArraysCapability(), AssignmentOnlyCapability(), LogitsTupleCapability()):
        assert isinstance(fake, RoutingDecodeCapability)


def test_dict_arrays_decode_with_sparse_native_ids_and_variable_top_k() -> None:
    universe = _universe()
    events = validate_decoded_routing(
        DictArraysCapability().decode(_dict_payload(), universe=universe, token_key=_TOKEN_KEY),
        universe=universe,
        token_key=_TOKEN_KEY,
        score_semantics=ScoreSemantics.PROBABILITIES,
    )
    assert len(events) == 5  # layer 0 selects 2, layer 1 selects 3
    by_layer = {event.layer_key: [event] for event in events}
    assert len(by_layer) == 2
    layer1_events = [
        event for event in events if event.layer_key == universe.layers[1].moe_layer_key
    ]
    # Native ids (9, 2, 5) are sparse and unordered; they must resolve through
    # the published parallel indices to canonical keys.
    expected = {9: _key_of(1, 9), 2: _key_of(1, 2), 5: _key_of(1, 5)}
    assert [event.rank for event in layer1_events] == [0, 1, 2]
    assert [event.expert_key for event in layer1_events] == [
        expected[9],
        expected[2],
        expected[5],
    ]
    assert all(event.probability is not None for event in events)
    assert all(event.router_logit is None for event in events)


def test_assignment_only_three_d_payload_decodes_without_scores() -> None:
    universe = _universe()
    payload = [[0, 2], [5, 0, 9]]
    events = validate_decoded_routing(
        AssignmentOnlyCapability().decode(payload, universe=universe, token_key=_TOKEN_KEY),
        universe=universe,
        token_key=_TOKEN_KEY,
        score_semantics=ScoreSemantics.NONE,
    )
    assert len(events) == 5
    assert all(
        event.router_logit is None and event.probability is None and event.weight == 1.0
        for event in events
    )
    layer0_events = [
        event for event in events if event.layer_key == universe.layers[0].moe_layer_key
    ]
    assert [event.expert_key for event in layer0_events] == [
        _key_of(0, 0),
        _key_of(0, 2),
    ]


def test_logits_tuple_decode_ranks_by_score() -> None:
    universe = _universe()
    capability = LogitsTupleCapability()
    # Layer 0 has width 4 and top_k 2: experts at positions 3 and 0 win.
    events = validate_decoded_routing(
        capability.decode(([0.1, 0.2, -0.5, 3.0], [9.0, 8.0, 7.0, 6.0, 5.0, 4.0]),
                          universe=universe, token_key=_TOKEN_KEY),
        universe=universe,
        token_key=_TOKEN_KEY,
        score_semantics=ScoreSemantics.LOGITS,
    )
    layer0 = [event for event in events if event.layer_key == universe.layers[0].moe_layer_key]
    assert [(event.rank, event.router_logit) for event in layer0] == [
        (0, 3.0),
        (1, 0.2),
    ]


def test_decoded_events_round_trip_through_the_event_contract() -> None:
    universe = _universe()
    events = validate_decoded_routing(
        DictArraysCapability().decode(_dict_payload(), universe=universe, token_key=_TOKEN_KEY),
        universe=universe,
        token_key=_TOKEN_KEY,
        score_semantics=ScoreSemantics.PROBABILITIES,
    )
    revived = tuple(RoutingEvent.model_validate(event.model_dump(mode="json")) for event in events)
    assert revived == events


# ---------------------------------------------------------------------------
# Shared postcondition validation
# ---------------------------------------------------------------------------


def _decoded(**overrides: object) -> tuple[RoutingEvent, ...]:
    universe = overrides.pop("universe", _universe())
    return DictArraysCapability().decode(
        overrides.pop("payload", _dict_payload()), universe=universe, token_key=_TOKEN_KEY
    )


def test_validate_rejects_wrong_token_and_foreign_layers() -> None:
    universe = _universe()
    with pytest.raises(RoutingDecodeError, match="does not match the requested token") as exc:
        validate_decoded_routing(
            _decoded(), universe=universe, token_key="token:" + "6" * 64,
            score_semantics=ScoreSemantics.PROBABILITIES,
        )
    assert exc.value.stage == "postcondition"

    foreign = _decoded()[0].model_copy(update={"layer_key": "component:" + "f" * 64})
    with pytest.raises(RoutingDecodeError, match="outside the universe"):
        validate_decoded_routing(
            (foreign,), universe=universe, token_key=_TOKEN_KEY,
            score_semantics=ScoreSemantics.PROBABILITIES,
        )


def test_validate_rejects_incomplete_ranks_duplicates_and_foreign_experts() -> None:
    universe = _universe()
    events = _decoded(universe=universe)
    truncated = tuple(event for event in events if not (event.rank == 1 and
                      event.layer_key == universe.layers[0].moe_layer_key))
    with pytest.raises(RoutingDecodeError, match=r"ranks are not exactly 0\.\.1"):
        validate_decoded_routing(
            truncated, universe=universe, token_key=_TOKEN_KEY,
            score_semantics=ScoreSemantics.PROBABILITIES,
        )

    duplicated = (events[0], events[0].model_copy(update={"rank": 1}))
    with pytest.raises(RoutingDecodeError, match="same expert twice"):
        validate_decoded_routing(
            duplicated, universe=universe, token_key=_TOKEN_KEY,
            score_semantics=ScoreSemantics.PROBABILITIES,
        )

    foreign_expert = events[0].model_copy(update={"expert_key": "component:" + "e" * 64})
    with pytest.raises(RoutingDecodeError, match="outside its published universe"):
        validate_decoded_routing(
            (foreign_expert, events[1]), universe=universe, token_key=_TOKEN_KEY,
            score_semantics=ScoreSemantics.PROBABILITIES,
        )


def test_validate_enforces_declared_score_semantics() -> None:
    universe = _universe()
    probability_rows = _decoded(universe=universe)
    with pytest.raises(RoutingDecodeError, match="forbid logit and probability claims"):
        validate_decoded_routing(
            probability_rows, universe=universe, token_key=_TOKEN_KEY,
            score_semantics=ScoreSemantics.NONE,
        )
    assignment_rows = AssignmentOnlyCapability().decode(
        [[0, 2], [5, 0, 9]], universe=universe, token_key=_TOKEN_KEY
    )
    claimed = assignment_rows[0].model_copy(update={"probability": 0.5})
    with pytest.raises(RoutingDecodeError, match="forbid logit and probability claims"):
        validate_decoded_routing(
            (claimed,) + assignment_rows[1:], universe=universe, token_key=_TOKEN_KEY,
            score_semantics=ScoreSemantics.NONE,
        )
    reweighted = assignment_rows[0].model_copy(update={"weight": 0.25})
    with pytest.raises(RoutingDecodeError, match="pin weight to the unweighted 1.0 marker"):
        validate_decoded_routing(
            (reweighted,) + assignment_rows[1:], universe=universe, token_key=_TOKEN_KEY,
            score_semantics=ScoreSemantics.NONE,
        )
    # Keep a weight column so the events stay contract-valid and the declared
    # PROBABILITIES semantics are what fails.
    stripped = tuple(
        event.model_copy(update={"probability": None, "weight": 1.0})
        for event in probability_rows
    )
    with pytest.raises(RoutingDecodeError, match="require a finite probability"):
        validate_decoded_routing(
            stripped, universe=universe, token_key=_TOKEN_KEY,
            score_semantics=ScoreSemantics.PROBABILITIES,
        )


def test_validate_rejects_unselected_rows_and_bad_containers() -> None:
    universe = _universe()
    events = _decoded(universe=universe)
    unselected = tuple(event.model_copy(update={"selected": False}) for event in events)
    with pytest.raises(RoutingDecodeError, match="must be selected"):
        validate_decoded_routing(
            unselected, universe=universe, token_key=_TOKEN_KEY,
            score_semantics=ScoreSemantics.PROBABILITIES,
        )
    with pytest.raises(RoutingDecodeError, match="tuple of exact RoutingEvent"):
        validate_decoded_routing(
            list(events), universe=universe, token_key=_TOKEN_KEY,
            score_semantics=ScoreSemantics.PROBABILITIES,
        )
    with pytest.raises(RoutingDecodeError, match="dependency") as exc:
        validate_decoded_routing(
            events, universe=universe, token_key=_TOKEN_KEY,
            score_semantics="probabilities",
        )
    assert exc.value.stage == "dependency"


def test_validate_layer_coverage_is_explicit_and_narrowable() -> None:
    universe = _universe()
    subset = tuple(
        event
        for event in _decoded(universe=universe)
        if event.layer_key == universe.layers[0].moe_layer_key
    )
    with pytest.raises(RoutingDecodeError, match="have no decoded rows"):
        validate_decoded_routing(
            subset, universe=universe, token_key=_TOKEN_KEY,
            score_semantics=ScoreSemantics.PROBABILITIES,
        )
    allowed = validate_decoded_routing(
        subset,
        universe=universe,
        token_key=_TOKEN_KEY,
        score_semantics=ScoreSemantics.PROBABILITIES,
        require_all_layers=False,
    )
    assert allowed == subset


# ---------------------------------------------------------------------------
# native identifier mapping and error contract
# ---------------------------------------------------------------------------


def test_native_id_map_resolves_sparse_identifiers() -> None:
    layer_one = _universe().layers[1]
    mapping = native_id_map(layer_one)
    assert mapping == dict(zip(layer_one.expert_indices, layer_one.expert_keys))
    assert set(mapping) == {7, 0, 2, 5, 9, 1}
    with pytest.raises(RoutingDecodeError, match="does not declare native expert indices"):
        native_id_map(_layer(3, 2))
    with pytest.raises(RoutingDecodeError, match="exact LayerRoutingUniverse") as exc:
        native_id_map({"not": "a layer"})
    assert exc.value.stage == "dependency"


def test_error_stage_contract() -> None:
    assert str(RoutingDecodeError("decode")) == "routing decode failed at decode"
    error = RoutingDecodeError("postcondition", "boom")
    assert error.stage == "postcondition"
    assert str(error) == "routing decode failed at postcondition: boom"
    with pytest.raises(ValueError, match="unsupported routing decode stage"):
        RoutingDecodeError("bogus")


def test_capability_schema_version_is_pinned() -> None:
    assert ROUTING_DECODE_CAPABILITY_SCHEMA_VERSION == "1.0"


# ---------------------------------------------------------------------------
# Isolation guards
# ---------------------------------------------------------------------------


def test_capabilities_import_without_model_stack() -> None:
    script = (
        "import sys\n"
        "for name in ('torch', 'transformers', 'duckdb', 'safetensors'):\n"
        "    sys.modules[name] = None\n"
        "import moeatlas.runtime.capabilities\n"
        "print('capabilities-import-ok')\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    ).rstrip(os.pathsep)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "capabilities-import-ok" in completed.stdout


def test_no_model_runtime_imported_by_capabilities() -> None:
    assert not any(
        name in sys.modules for name in ("torch", "transformers", "safetensors")
    )
