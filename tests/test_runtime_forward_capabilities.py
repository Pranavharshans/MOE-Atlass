"""Model-free tests for the universal capability-driven forward runner.

The neutral :func:`run_routing_forward` seam must execute any MoE family that
publishes a routing universe and supplies a declared hook decoder — with no
central family branching.  An unknown ``blocksparse_moe`` family with
mapping-keyed dict payloads and unordered native expert identifiers runs
through the exact same seam the Mixtral and Qwen3.5 wrappers compose.
"""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest

from moeatlas.adapters import (
    AdapterDescriptor,
    RoutingUniverseError,
    build_routing_probe_plan,
    publish_routing_universe,
)
from moeatlas.core import parse_token_key
from moeatlas.event_validation import fresh_token_events
from moeatlas.events import RoutingEvent, TokenEvent, TokenPhase
from moeatlas.runtime import (
    RouterPayloadShape,
    RoutingHookDecoder,
    ScoreSemantics,
    TokenSequencePolicy,
    native_id_map,
    run_mixtral_routing_forward,
    run_qwen3_5_routing_forward,
    run_routing_forward,
    validate_observed_routing,
)
from moeatlas.runtime.capabilities import RoutingDecodeError

from .test_adapters_universe import _rebuild_inspection
from .test_adapters_universe import _universe as _contract_universe
from .test_mixtral_routing_decoder import _inspection as _mixtral_inspection
from .test_mixtral_routing_decoder import _tokens
from .test_runtime_routing_forward import _run as _mixtral_forward_run

ROOT = Path(__file__).resolve().parents[1]

_TOKEN_KEY = "token:" + "7" * 64


def _unknown_descriptor() -> AdapterDescriptor:
    return AdapterDescriptor(
        name="acme-blocksparse-static",
        version="2.0",
        architecture_families=("blocksparse_moe",),
        compatibility_notes=("dict-array router payloads with unordered native ids",),
    )


def _unknown_inspection():
    base = _mixtral_inspection("legacy")
    return _rebuild_inspection(
        base,
        descriptor=_unknown_descriptor(),
        router_metadata={0: {"layout": "block_sparse"}, 1: {"layout": "block_sparse"}},
    )


def _dict_rows(count: int) -> dict[str, object]:
    """One mapping-keyed row per token with deliberately unordered native ids."""

    return {"rows": [{"ids": [2, 0], "p": [0.7, 0.3]} for _ in range(count)]}


class _BlockSparseDictDecoder:
    """Unknown-family decoder: dict payloads decoded through the published universe."""

    payload_shape = RouterPayloadShape.DICT_ARRAYS
    score_semantics = ScoreSemantics.PROBABILITIES

    def __init__(self, inspection: object, token_events: object) -> None:
        self._universe = publish_routing_universe(inspection)
        self._tokens = fresh_token_events(token_events)
        self._used_paths: set[str] = set()

    def __call__(
        self,
        context: object,
        module: object,
        inputs: object,
        payload: object,
    ) -> tuple[RoutingEvent, ...]:
        del module, inputs
        path = context.router.module_path  # type: ignore[attr-defined]
        if path in self._used_paths:
            raise RuntimeError("router decoder invocation is single-use per router")
        layer = next(
            candidate
            for candidate in self._universe.layers
            if candidate.moe_layer_key == context.layer_key  # type: ignore[attr-defined]
        )
        if (
            sorted(context.expert_keys) != list(layer.expert_keys)  # type: ignore[attr-defined]
            or context.routed_top_k != layer.routed_top_k  # type: ignore[attr-defined]
        ):
            raise ValueError("context does not match the published universe layer")
        id_map = native_id_map(layer)
        events: list[RoutingEvent] = []
        for token_event, entry in zip(self._tokens, payload["rows"], strict=True):
            for rank, (native_id, probability) in enumerate(
                zip(entry["ids"], entry["p"], strict=True)
            ):
                events.append(
                    RoutingEvent(
                        token_key=token_event.token_key,
                        layer_key=layer.moe_layer_key,
                        rank=rank,
                        expert_key=id_map[native_id],
                        probability=probability,
                        selected=True,
                    )
                )
        self._used_paths.add(path)
        return tuple(events)


class _ScorelessDictDecoder(_BlockSparseDictDecoder):
    """Declares PROBABILITIES but emits rows without the probability column."""

    def __call__(
        self,
        context: object,
        module: object,
        inputs: object,
        payload: object,
    ) -> tuple[RoutingEvent, ...]:
        events = super().__call__(context, module, inputs, payload)
        return tuple(
            event.model_copy(update={"probability": None, "weight": 1.0})
            for event in events
        )


class _PartialTokenDecoder(_BlockSparseDictDecoder):
    """Drops the second token and its rows so the capture is incomplete."""

    def __call__(
        self,
        context: object,
        module: object,
        inputs: object,
        payload: object,
    ) -> tuple[RoutingEvent, ...]:
        full_tokens = self._tokens
        self._tokens = full_tokens[:1]
        try:
            trimmed = dict(payload)
            trimmed["rows"] = payload["rows"][:1]
            return super().__call__(context, module, inputs, trimmed)
        finally:
            self._tokens = full_tokens


class _Handle:
    def __init__(self, owner: _HookedNode, callback: object) -> None:
        self.owner = owner
        self.callback = callback

    def remove(self) -> None:
        if self.callback in self.owner.callbacks:
            self.owner.callbacks.remove(self.callback)


class _HookedNode:
    def __init__(self, path: str) -> None:
        self.path = path
        self.callbacks: list[object] = []

    def register_forward_hook(self, callback: object) -> _Handle:
        self.callbacks.append(callback)
        return _Handle(self, callback)

    def fire(self, payload: object) -> None:
        for callback in tuple(self.callbacks):
            callback(self, (), payload)


class _UnknownFamilyModel:
    """Minimal callable model surface: named_modules plus firing hook nodes."""

    def __init__(
        self,
        inspection: object,
        payload: object,
        *,
        reverse_fire_order: bool = False,
    ) -> None:
        self.fire_paths = [
            target.module_path for target in build_routing_probe_plan(inspection).targets
        ]
        if reverse_fire_order:
            self.fire_paths = list(reversed(self.fire_paths))
        self.nodes = {path: _HookedNode(path) for path in self.fire_paths}
        self.payload = payload
        self.output = object()
        self.calls = 0
        self.named_modules_calls = 0
        self.received_kwargs: dict[str, object] | None = None

    def named_modules(self):
        self.named_modules_calls += 1
        return iter([(path, self.nodes[path]) for path in self.fire_paths])

    def __call__(self, **kwargs: object) -> object:
        self.calls += 1
        self.received_kwargs = kwargs
        for path in self.fire_paths:
            self.nodes[path].fire(self.payload)
        return self.output


def _run_unknown(
    model: _UnknownFamilyModel | None = None,
    *,
    token_count: int = 2,
    max_events: int = 32,
    decoder: object = _BlockSparseDictDecoder,
    token_sequence: TokenSequencePolicy = TokenSequencePolicy.SHARED_RUN,
    inspection: object | None = None,
):
    inspection = inspection if inspection is not None else _unknown_inspection()
    model = model or _UnknownFamilyModel(inspection, _dict_rows(token_count))
    result = run_routing_forward(
        model,
        inspection,
        build_routing_probe_plan(inspection),
        _tokens(token_count),
        {},
        decoder=decoder,  # type: ignore[arg-type]
        token_sequence=token_sequence,
        max_events=max_events,
    )
    return result, model, inspection


# ---------------------------------------------------------------------------
# Public contract and the unknown-family execution path
# ---------------------------------------------------------------------------


def test_public_api_and_policy_contract() -> None:
    signature = inspect.signature(run_routing_forward)
    assert tuple(signature.parameters) == (
        "model",
        "inspection",
        "plan",
        "token_events",
        "model_kwargs",
        "decoder",
        "token_sequence",
        "max_events",
    )
    assert signature.parameters["decoder"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["decoder"].default is inspect.Parameter.empty
    assert signature.parameters["token_sequence"].default is TokenSequencePolicy.SHARED_RUN
    assert signature.parameters["max_events"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["max_events"].default is inspect.Parameter.empty
    assert str(TokenSequencePolicy.SHARED_RUN) == "shared_run"
    assert str(TokenSequencePolicy.CANONICAL_SEQUENCE) == "canonical_sequence"
    for wrapper in (run_mixtral_routing_forward, run_qwen3_5_routing_forward):
        wrapper_signature = inspect.signature(wrapper)
        assert tuple(wrapper_signature.parameters) == (
            "model",
            "inspection",
            "plan",
            "token_events",
            "model_kwargs",
            "max_events",
        )


def test_unknown_family_executes_through_the_neutral_seam() -> None:
    marker = object()
    inspection = _unknown_inspection()
    model = _UnknownFamilyModel(inspection, _dict_rows(2))
    model.output = marker
    decoder_instance = _BlockSparseDictDecoder(inspection, _tokens(2))
    assert isinstance(decoder_instance, RoutingHookDecoder)
    result, _, _ = _run_unknown(model=model)
    assert result.output is marker
    assert model.calls == 1
    assert len(result.routing_events) == 2 * len(model.nodes) * 2
    assert all(node.callbacks == [] for node in model.nodes.values())
    universe = publish_routing_universe(inspection)
    layer_keys = [layer.moe_layer_key for layer in universe.layers]
    block_order: list[str] = []
    for event in result.routing_events:
        if not block_order or block_order[-1] != event.layer_key:
            block_order.append(event.layer_key)
    assert block_order == layer_keys
    for layer in universe.layers:
        id_map = native_id_map(layer)
        layer_events = [
            event for event in result.routing_events if event.layer_key == layer.moe_layer_key
        ]
        assert len(layer_events) == 4
        for position in range(2):
            token_rows = layer_events[position * 2 : (position + 1) * 2]
            assert [event.rank for event in token_rows] == [0, 1]
            assert [event.expert_key for event in token_rows] == [id_map[2], id_map[0]]
            assert [event.probability for event in token_rows] == [0.7, 0.3]
            assert all(
                event.router_logit is None and event.weight is None for event in token_rows
            )


def test_family_runners_reject_the_same_unknown_family_stack() -> None:
    inspection = _unknown_inspection()
    plan = build_routing_probe_plan(inspection)
    model = _UnknownFamilyModel(inspection, _dict_rows(1))
    with pytest.raises(ValueError, match="Mixtral static descriptor"):
        run_mixtral_routing_forward(model, inspection, plan, _tokens(1), {}, max_events=32)
    assert model.calls == 0
    with pytest.raises(ValueError, match="Qwen3.5 static descriptor"):
        run_qwen3_5_routing_forward(model, inspection, plan, _tokens(1), {}, max_events=32)
    assert model.calls == 0
    result = run_routing_forward(
        model, inspection, plan, _tokens(1), {},
        decoder=_BlockSparseDictDecoder, max_events=32,
    )
    assert model.calls == 1
    assert len(result.routing_events) == 4


def test_token_sequence_policies_divide_shared_run_and_canonical_order() -> None:
    result, _, _ = _run_unknown(token_count=2)
    assert len(result.routing_events) == 8
    gapped = (
        TokenEvent(
            run_key="run-1",
            sequence_id="sequence-1",
            token_pos=0,
            token_id=10,
            token_text="0",
            phase=TokenPhase.PREFILL,
        ),
        TokenEvent(
            run_key="run-1",
            sequence_id="sequence-1",
            token_pos=5,
            token_id=11,
            token_text="1",
            phase=TokenPhase.PREFILL,
        ),
    )
    inspection = _unknown_inspection()
    model = _UnknownFamilyModel(inspection, _dict_rows(2))
    with pytest.raises(ValueError, match="contiguous"):
        run_routing_forward(
            model, inspection, build_routing_probe_plan(inspection), gapped, {},
            decoder=_BlockSparseDictDecoder,
            token_sequence=TokenSequencePolicy.CANONICAL_SEQUENCE,
            max_events=32,
        )
    assert model.calls == 0
    with pytest.raises(TypeError, match="exact TokenSequencePolicy"):
        run_routing_forward(
            _UnknownFamilyModel(inspection, _dict_rows(1)),
            inspection,
            build_routing_probe_plan(inspection),
            _tokens(1),
            {},
            decoder=_BlockSparseDictDecoder,
            token_sequence="shared_run",
            max_events=32,
        )


def test_reverse_fire_order_is_rejected_after_cleanup_for_unknown_family() -> None:
    inspection = _unknown_inspection()
    model = _UnknownFamilyModel(inspection, _dict_rows(1), reverse_fire_order=True)
    with pytest.raises(ValueError, match="canonical layer block order"):
        run_routing_forward(
            model, inspection, build_routing_probe_plan(inspection), _tokens(1), {},
            decoder=_BlockSparseDictDecoder, max_events=32,
        )
    assert model.calls == 1
    assert all(node.callbacks == [] for node in model.nodes.values())


# ---------------------------------------------------------------------------
# Decoder factory and declared capability gates
# ---------------------------------------------------------------------------


def test_noncallable_decoder_factory_is_rejected_before_traversal() -> None:
    inspection = _unknown_inspection()
    model = _UnknownFamilyModel(inspection, _dict_rows(1))
    with pytest.raises(TypeError, match="decoder must be callable"):
        run_routing_forward(
            model, inspection, build_routing_probe_plan(inspection), _tokens(1), {},
            decoder=object(),  # type: ignore[arg-type]
            max_events=32,
        )
    assert model.named_modules_calls == 0
    assert model.calls == 0


def test_decoder_without_declared_capabilities_is_rejected() -> None:
    inspection = _unknown_inspection()
    model = _UnknownFamilyModel(inspection, _dict_rows(1))

    def bare_factory(inspection: object, token_events: object):
        del inspection, token_events
        return lambda *args: ()

    with pytest.raises(TypeError, match="declare payload_shape"):
        run_routing_forward(
            model, inspection, build_routing_probe_plan(inspection), _tokens(1), {},
            decoder=bare_factory, max_events=32,
        )
    assert model.calls == 0
    assert all(node.callbacks == [] for node in model.nodes.values())


def test_declared_score_semantics_are_enforced_on_captured_rows() -> None:
    inspection = _unknown_inspection()
    model = _UnknownFamilyModel(inspection, _dict_rows(1))
    with pytest.raises(ValueError, match="require a finite probability"):
        run_routing_forward(
            model, inspection, build_routing_probe_plan(inspection), _tokens(1), {},
            decoder=_ScorelessDictDecoder, max_events=32,
        )
    assert model.calls == 1
    assert all(node.callbacks == [] for node in model.nodes.values())


def test_incomplete_token_rows_fail_the_completeness_contract() -> None:
    inspection = _unknown_inspection()
    model = _UnknownFamilyModel(inspection, _dict_rows(2))
    with pytest.raises(ValueError, match="did not publish complete events"):
        run_routing_forward(
            model, inspection, build_routing_probe_plan(inspection), _tokens(2), {},
            decoder=_PartialTokenDecoder, max_events=32,
        )
    assert all(node.callbacks == [] for node in model.nodes.values())


def test_universe_gate_rejects_inconsistent_top_k_provenance_before_hooks() -> None:
    base = _mixtral_inspection("legacy")
    inspection = _rebuild_inspection(
        base,
        descriptor=_unknown_descriptor(),
        router_metadata={
            0: {"layout": "block_sparse", "routed_top_k": 3},
            1: {"layout": "block_sparse", "routed_top_k": 3},
        },
    )
    model = _UnknownFamilyModel(inspection, _dict_rows(1))
    with pytest.raises(RoutingUniverseError, match="does not match inspection facts"):
        run_routing_forward(
            model, inspection, build_routing_probe_plan(inspection), _tokens(1), {},
            decoder=_BlockSparseDictDecoder, max_events=32,
        )
    assert model.named_modules_calls == 0
    assert model.calls == 0


def test_neutral_seam_verifies_real_mixtral_capture_against_its_universe() -> None:
    result, _, inspection = _mixtral_forward_run("legacy", token_count=2)
    universe = publish_routing_universe(inspection)
    validated = validate_observed_routing(
        result.routing_events,
        universe=universe,
        token_keys=tuple(event.token_key for event in result.token_events),
        score_semantics=ScoreSemantics.LOGITS,
    )
    assert len(validated) == len(result.routing_events)


# ---------------------------------------------------------------------------
# validate_observed_routing shared postconditions
# ---------------------------------------------------------------------------


def _observed_token(index: int) -> str:
    return f"token:{index:064x}"


def _observed_events(
    token_key: str,
    *,
    layer0_ranks: tuple[int, ...] = (0, 1),
    layer1_ranks: tuple[int, ...] = (0, 1, 2),
    probability: float | None = 0.5,
    weight: float | None = None,
    selected: bool = True,
) -> tuple[RoutingEvent, ...]:
    universe = _contract_universe()
    events: list[RoutingEvent] = []
    for layer, ranks in ((universe.layers[0], layer0_ranks), (universe.layers[1], layer1_ranks)):
        for rank in ranks:
            events.append(
                RoutingEvent(
                    token_key=token_key,
                    layer_key=layer.moe_layer_key,
                    rank=rank,
                    expert_key=layer.expert_keys[rank],
                    probability=probability,
                    weight=weight,
                    selected=selected,
                )
            )
    return tuple(events)


def test_observed_postconditions_accept_complete_variable_top_k_captures() -> None:
    universe = _contract_universe()
    tokens = (_observed_token(1), _observed_token(2))
    events = _observed_events(tokens[0]) + _observed_events(tokens[1])
    validated = validate_observed_routing(
        events, universe=universe, token_keys=tokens, score_semantics=ScoreSemantics.PROBABILITIES
    )
    assert validated == events
    assert all(event is not source for event, source in zip(validated, events))


def test_observed_postconditions_reject_incomplete_groups_and_rows() -> None:
    universe = _contract_universe()
    tokens = (_observed_token(1), _observed_token(2))
    complete = _observed_events(tokens[0]) + _observed_events(tokens[1])
    dropped_token = tuple(event for event in complete if event.token_key != tokens[1])
    with pytest.raises(RoutingDecodeError, match="no observed rows") as exc:
        validate_observed_routing(
            dropped_token,
            universe=universe,
            token_keys=tokens,
            score_semantics=ScoreSemantics.PROBABILITIES,
        )
    assert exc.value.stage == "postcondition"

    truncated = tuple(
        event for event in complete if not (event.token_key == tokens[0] and event.rank == 1)
    )
    with pytest.raises(RoutingDecodeError, match=r"ranks are not exactly 0\.\.1"):
        validate_observed_routing(
            truncated,
            universe=universe,
            token_keys=tokens,
            score_semantics=ScoreSemantics.PROBABILITIES,
        )

    # Complete rank schedule but the same expert twice: ranks (0, 1) both
    # pointing at layer 0's first expert.
    layer0 = universe.layers[0]
    duplicated = (
        complete[0],
        complete[0].model_copy(update={"rank": 1}),
        *complete[2:],
    )
    assert duplicated[0].expert_key == layer0.expert_keys[0]
    with pytest.raises(RoutingDecodeError, match="same expert twice"):
        validate_observed_routing(
            duplicated,
            universe=universe,
            token_keys=tokens,
            score_semantics=ScoreSemantics.PROBABILITIES,
        )


def test_observed_postconditions_reject_foreign_rows_and_columns() -> None:
    universe = _contract_universe()
    tokens = (_observed_token(1),)
    complete = _observed_events(tokens[0])
    foreign_token = complete[0].model_copy(update={"token_key": _observed_token(9)})
    with pytest.raises(RoutingDecodeError, match="outside the capture"):
        validate_observed_routing(
            (foreign_token, *complete[1:]),
            universe=universe,
            token_keys=tokens,
            score_semantics=ScoreSemantics.PROBABILITIES,
        )

    foreign_expert = complete[0].model_copy(
        update={"expert_key": universe.layers[1].expert_keys[0]}
    )
    with pytest.raises(RoutingDecodeError, match="outside its published universe"):
        validate_observed_routing(
            (foreign_expert, *complete[1:]),
            universe=universe,
            token_keys=tokens,
            score_semantics=ScoreSemantics.PROBABILITIES,
        )

    unselected = complete[0].model_copy(update={"selected": False})
    with pytest.raises(RoutingDecodeError, match="must be selected"):
        validate_observed_routing(
            (unselected, *complete[1:]),
            universe=universe,
            token_keys=tokens,
            score_semantics=ScoreSemantics.PROBABILITIES,
        )

    scoreless = tuple(
        event.model_copy(update={"probability": None, "weight": 1.0}) for event in complete
    )
    with pytest.raises(RoutingDecodeError, match="require a finite probability"):
        validate_observed_routing(
            scoreless,
            universe=universe,
            token_keys=tokens,
            score_semantics=ScoreSemantics.PROBABILITIES,
        )

    claimed = complete[0].model_copy(update={"probability": 0.5})
    with pytest.raises(RoutingDecodeError, match="forbid logit and probability claims"):
        validate_observed_routing(
            (claimed, *scoreless[1:]),
            universe=universe,
            token_keys=tokens,
            score_semantics=ScoreSemantics.NONE,
        )


def test_observed_postcondition_dependency_arguments_are_strict() -> None:
    universe = _contract_universe()
    tokens = (_observed_token(1),)
    events = _observed_events(tokens[0])
    with pytest.raises(RoutingDecodeError, match="exact RoutingUniverse") as exc:
        validate_observed_routing(
            events, universe=object(), token_keys=tokens,
            score_semantics=ScoreSemantics.PROBABILITIES,
        )
    assert exc.value.stage == "dependency"
    with pytest.raises(RoutingDecodeError, match="tuple of exact strings"):
        validate_observed_routing(
            events, universe=universe, token_keys=list(tokens),
            score_semantics=ScoreSemantics.PROBABILITIES,
        )
    with pytest.raises(RoutingDecodeError, match="unique"):
        validate_observed_routing(
            events, universe=universe, token_keys=(tokens[0], tokens[0]),
            score_semantics=ScoreSemantics.PROBABILITIES,
        )
    parse_token_key(tokens[0])
    with pytest.raises(RoutingDecodeError, match="tuple of exact RoutingEvent"):
        validate_observed_routing(
            list(events), universe=universe, token_keys=tokens,
            score_semantics=ScoreSemantics.PROBABILITIES,
        )


# ---------------------------------------------------------------------------
# Isolation guards
# ---------------------------------------------------------------------------


def test_forward_runner_imports_without_model_stack() -> None:
    script = (
        "import sys\n"
        "for name in ('torch', 'transformers', 'duckdb', 'safetensors'):\n"
        "    sys.modules[name] = None\n"
        "import moeatlas.runtime.routing_forward\n"
        "print('forward-import-ok')\n"
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
    assert "forward-import-ok" in completed.stdout


def test_no_model_runtime_imported_by_forward_runner() -> None:
    import moeatlas.runtime.routing_forward as forward_module

    assert forward_module.__name__ == "moeatlas.runtime.routing_forward"
    assert not any(
        name in sys.modules for name in ("torch", "transformers", "safetensors")
    )
