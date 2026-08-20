from __future__ import annotations

import ast
import builtins
import gc
import inspect
import os
import socket
import subprocess
import tempfile
import urllib.request
import weakref
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from moeatlas.adapters import (
    AdapterInspection,
    Qwen3_5MoeStaticAdapter,
    build_routing_probe_plan,
    inspect_static_adapter,
)
from moeatlas.events import RoutingEvent, TokenEvent
from moeatlas.probe import ProbePlan
from moeatlas.runtime import (
    MixtralRoutingForwardResult,
    Qwen3_5RoutingDecoder,
    RoutingCaptureError,
    RoutingCaptureSession,
    RoutingForwardResult,
    run_qwen3_5_routing_forward,
)

from .fixtures.qwen3_5_moe import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeHookableForCausalLM,
    Qwen3_5MoeHookableForConditionalGeneration,
    Qwen3_5MoeTextConfig,
)
from .test_mixtral_routing_decoder import _inspection as _mixtral_inspection
from .test_qwen3_5_routing_decoder import (
    _inspection as _qwen35_inspection,
)
from .test_qwen3_5_routing_decoder import (
    _manifest as _qwen35_manifest,
)
from .test_qwen3_5_routing_decoder import (
    _packed_payload as _qwen35_packed_payload,
)
from .test_qwen3_5_routing_decoder import (
    _tokens as _qwen35_tokens,
)


class _RetentionObject:
    """Weak-referenceable holder used to prove caller ownership boundaries."""


class _QwenForwardModel:
    """Callable hookable model fixture for the shared Qwen forward seam."""

    def __init__(
        self,
        surface: str,
        *,
        rows: list[list[float]] | None = None,
        num_layers: int = 2,
        config: object | None = None,
        reverse: bool = False,
        duplicate: bool = False,
        trailing_duplicate: bool = False,
        fire_limit: int | None = None,
        failure: BaseException | None = None,
        removal_failures: int = 0,
        cleanup_failure: BaseException | None = None,
        registration_failure: bool = False,
    ) -> None:
        factory = (
            Qwen3_5MoeHookableForConditionalGeneration
            if surface == "conditional"
            else Qwen3_5MoeHookableForCausalLM
        )
        source = (
            factory(num_layers=num_layers, config=config)
            if config is not None
            else factory(num_layers=num_layers)
        )
        self.config = source.config
        self.nodes = source.nodes
        self._entries = tuple(source.named_modules())
        self._parameters = tuple(source.named_parameters())
        self.output = object()
        self.received_kwargs: dict[str, object] | None = None
        self.calls = 0
        self.reverse = reverse
        self.duplicate = duplicate
        self.trailing_duplicate = trailing_duplicate
        self.fire_limit = fire_limit
        self.failure = failure
        self.cleanup_failure = cleanup_failure
        self.rows = rows or [[1.0, 2.0, 0.0, 3.0]]
        self.input_value = _RetentionObject()
        self.payload_refs: list[weakref.ReferenceType[object]] = []
        self._removal_failures = removal_failures
        last_path = next(reversed(self.nodes)) if self.nodes else None
        for path, node in self.nodes.items():
            original_register = node.register_forward_hook

            def register(
                callback: object,
                *,
                _path: str = path,
                _original: object = original_register,
            ) -> object:
                if registration_failure and _path == last_path:
                    raise OSError("registration failure")
                handle = _original(callback)
                return _QwenHookHandle(handle, self)

            node.register_forward_hook = register  # type: ignore[method-assign]

    def named_modules(self):
        return iter(self._entries)

    def named_parameters(self):
        return iter(self._parameters)

    def __call__(self, **kwargs: object) -> object:
        self.calls += 1
        self.received_kwargs = kwargs
        nodes = list(self.nodes.values())
        if self.fire_limit is not None:
            nodes = nodes[: self.fire_limit]
        if self.reverse:
            nodes.reverse()
        payload = _qwen35_packed_payload(self.rows)
        self.payload_refs.extend(weakref.ref(item) for item in payload)
        for node in nodes:
            node.fire(payload, inputs=(self.input_value,))
            if self.duplicate and node is nodes[0]:
                node.fire(payload)
        if self.trailing_duplicate and nodes:
            nodes[-1].fire(payload)
        if self.failure is not None:
            raise self.failure
        return self.output


class _QwenHookHandle:
    def __init__(self, handle: object, owner: _QwenForwardModel) -> None:
        self._handle = handle
        self._owner = owner

    def remove(self) -> None:
        if self._owner.cleanup_failure is not None:
            failure = self._owner.cleanup_failure
            self._owner.cleanup_failure = None
            raise failure
        if self._owner._removal_failures:
            self._owner._removal_failures -= 1
            raise OSError("removal failure")
        self._handle.remove()


class _HardKillModel:
    """Model surface that fails if preflight traverses or calls it."""

    def __init__(self) -> None:
        self.calls = 0
        self.traversals = 0

    def named_modules(self):
        self.traversals += 1
        raise AssertionError("preflight must not traverse the model")

    def __call__(self, **kwargs: object) -> object:
        del kwargs
        self.calls += 1
        raise AssertionError("preflight must not call the model")


def _run_qwen(
    surface: str = "conditional",
    *,
    model: _QwenForwardModel | None = None,
    tokens: tuple[object, ...] | None = None,
    model_kwargs: dict[str, object] | None = None,
    max_events: int = 8,
):
    inspection = _qwen35_inspection(surface)
    plan = build_routing_probe_plan(inspection)
    actual_model = model or _QwenForwardModel(surface)
    result = run_qwen3_5_routing_forward(
        actual_model,
        inspection,
        plan,
        _qwen35_tokens(1) if tokens is None else tokens,
        {} if model_kwargs is None else model_kwargs,
        max_events=max_events,
    )
    return result, actual_model, inspection, plan


@pytest.mark.parametrize("surface", ["conditional", "text"])
def test_public_api_signature_alias_slots_repr_and_eq(surface: str) -> None:
    assert RoutingForwardResult is MixtralRoutingForwardResult
    assert RoutingForwardResult.__name__ == "RoutingForwardResult"
    assert RoutingForwardResult.__module__ == "moeatlas.runtime.routing_forward"
    assert RoutingForwardResult.__slots__ == ("output", "token_events", "routing_events")
    fields = RoutingForwardResult.__dataclass_fields__
    assert tuple(fields) == ("output", "token_events", "routing_events")
    assert fields["output"].repr is False
    assert RoutingForwardResult.__dataclass_params__.eq is False
    signature = inspect.signature(run_qwen3_5_routing_forward)
    assert tuple(signature.parameters) == (
        "model",
        "inspection",
        "plan",
        "token_events",
        "model_kwargs",
        "max_events",
    )
    assert signature.parameters["max_events"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["max_events"].default is inspect.Parameter.empty
    result, _, _, _ = _run_qwen(surface)
    assert "output=" not in repr(result)
    with pytest.raises(FrozenInstanceError):
        result.output = object()
    duplicate = RoutingForwardResult(result.output, result.token_events, result.routing_events)
    assert duplicate is not result
    assert duplicate != result


def test_result_alias_supports_exact_subclass_rejection() -> None:
    class ResultSubclass(RoutingForwardResult):
        pass

    result, _, _, _ = _run_qwen()
    with pytest.raises(TypeError, match="exact RoutingForwardResult"):
        from moeatlas.store.routing_shards import _fresh_events

        _fresh_events(ResultSubclass(result.output, result.token_events, result.routing_events))


@pytest.mark.parametrize("surface", ["conditional", "text"])
def test_both_roots_actual_decoder_session_multilayer_exact_events(surface: str) -> None:
    tokens = _qwen35_tokens(2)
    model = _QwenForwardModel(
        surface,
        rows=[[1.0, 2.0, 0.0, 3.0], [4.0, 0.0, 3.0, 1.0]],
    )
    result, _, inspection, plan = _run_qwen(
        surface, model=model, tokens=tokens, model_kwargs={"input_ids": object()}, max_events=16
    )
    assert type(result) is RoutingForwardResult
    assert result.output is model.output
    assert model.calls == 1
    assert len(result.routing_events) == len(tokens) * len(plan.targets) * 2
    assert tuple(dict.fromkeys(event.layer_key for event in result.routing_events)) == tuple(
        component.component_key
        for component in inspection.report.components
        if component.kind.value == "moe_layer"
    )
    assert model.received_kwargs is not None
    assert model.received_kwargs["input_ids"] is not None
    assert all(not node.callbacks for node in model.nodes.values())
    assert all(event.selected is True for event in result.routing_events)
    assert all(
        event.probability is None and event.weight is not None for event in result.routing_events
    )
    component_by_key = {
        component.component_key: component for component in inspection.report.components
    }
    assert all(
        component_by_key[event.expert_key].kind.value == "expert"
        and component_by_key[event.expert_key].shared is False
        for event in result.routing_events
    )
    first_layer_experts = sorted(
        (
            component
            for component in inspection.report.components
            if component.kind.value == "expert" and component.layer_index == 0
        ),
        key=lambda component: component.expert_index,
    )
    assert [event.expert_key for event in result.routing_events[:2]] == [
        first_layer_experts[3].component_key,
        first_layer_experts[1].component_key,
    ]
    assert [event.router_logit for event in result.routing_events[:2]] == [3.0, 2.0]
    assert result.token_events is not tokens
    assert all(left is not right for left, right in zip(result.token_events, tokens, strict=True))


def test_qwen_numeric_topk_order_is_not_lexical_and_kwargs_are_shallow() -> None:
    class NoCopy:
        def __deepcopy__(self, memo: object) -> object:
            raise AssertionError("values must not be deep-copied")

    value = NoCopy()
    result, model, _, _ = _run_qwen(model_kwargs={"value": value})
    assert model.received_kwargs is not None and model.received_kwargs["value"] is value
    # Scores select numeric expert indices 3 then 1 for the first row, rather
    # than lexical ordering of component-key strings.
    assert [event.router_logit for event in result.routing_events[:2]] == [3.0, 2.0]
    assert [event.rank for event in result.routing_events[:2]] == [0, 1]


def test_qwen_kwargs_are_shallow_copied_and_caller_mapping_is_unchanged() -> None:
    input_ids = object()
    nested = {"sentinel": object()}
    caller_kwargs: dict[str, object] = {"input_ids": input_ids, "nested": nested}
    before = dict(caller_kwargs)
    result, model, _, _ = _run_qwen(model_kwargs=caller_kwargs)
    assert result.output is model.output
    assert model.received_kwargs is not None
    assert model.received_kwargs is not caller_kwargs
    assert model.received_kwargs == before
    assert model.received_kwargs["input_ids"] is input_ids
    assert model.received_kwargs["nested"] is nested
    assert caller_kwargs == before


def test_qwen_multilayer_order_is_numeric_with_layer_2_before_layer_10() -> None:
    text_config = Qwen3_5MoeTextConfig(
        num_hidden_layers=11,
        layer_types=("full_attention", "linear_attention") * 5 + ("full_attention",),
    )
    model = _QwenForwardModel(
        "conditional",
        num_layers=11,
        config=Qwen3_5MoeConfig(text_config=text_config),
    )
    inspection = inspect_static_adapter(
        Qwen3_5MoeStaticAdapter(), model, model.config, _qwen35_manifest("conditional")
    )
    plan = build_routing_probe_plan(inspection)
    tokens = _qwen35_tokens(1)
    result = run_qwen3_5_routing_forward(
        model,
        inspection,
        plan,
        tokens,
        {},
        max_events=22,
    )
    assert model.calls == 1
    assert len(result.routing_events) == 22
    layer_by_key = {
        component.component_key: component
        for component in inspection.report.components
        if component.kind.value == "moe_layer"
    }
    layer_numbers = [
        layer_by_key[key].layer_index
        for key in dict.fromkeys(event.layer_key for event in result.routing_events)
    ]
    assert layer_numbers == list(range(11))
    assert layer_numbers.index(2) < layer_numbers.index(10)


def test_qwen_preflight_hostile_matrix_never_traverses_model() -> None:
    inspection = _qwen35_inspection()
    plan = build_routing_probe_plan(inspection)
    model = _QwenForwardModel("conditional")
    tokens = _qwen35_tokens(1)
    wrong_inspection = _mixtral_inspection("packed")
    wrong_plan = build_routing_probe_plan(wrong_inspection)
    cases: list[tuple[object, object, object, object, object, object]] = [
        (object(), inspection, plan, tokens, {}, 8),
        (model, wrong_inspection, wrong_plan, tokens, {}, 8),
        (model, object(), plan, tokens, {}, 8),
        (model, inspection, object(), tokens, {}, 8),
        (model, inspection, plan, (), {}, 8),
        (model, inspection, plan, tokens, [], 8),
        (model, inspection, plan, tokens, {" bad": 1}, 8),
        (model, inspection, plan, tokens, {}, 0),
        (model, inspection, plan, tokens, {}, True),
        (model, inspection, plan, tokens, {}, 1.0),
    ]
    for (
        candidate_model,
        candidate_inspection,
        candidate_plan,
        candidate_tokens,
        kwargs,
        budget,
    ) in cases:
        with pytest.raises((TypeError, ValueError)):
            run_qwen3_5_routing_forward(
                candidate_model,
                candidate_inspection,
                candidate_plan,
                candidate_tokens,
                kwargs,
                max_events=budget,  # type: ignore[arg-type]
            )
        assert model.calls == 0


def test_qwen_exact_subclass_report_provenance_and_topk_preflight() -> None:
    inspection = _qwen35_inspection()
    plan = build_routing_probe_plan(inspection)
    tokens = _qwen35_tokens(1)
    expected = len(tokens) * len(plan.targets) * inspection.report.facts.routed_top_k

    class InspectionSubclass(AdapterInspection):
        pass

    class TokenSubclass(TokenEvent):
        pass

    class TokenTupleSubclass(tuple):
        pass

    class KwargsSubclass(dict[str, object]):
        pass

    subclass_inspection = InspectionSubclass.model_validate(inspection.model_dump(mode="json"))
    subclass_token = TokenSubclass.model_validate(tokens[0].model_dump(mode="json"))
    bad_report = inspection.report.model_copy(update={"scanner_version": "tampered"})
    bad_provenance = inspection.report.components[0].provenance
    assert bad_provenance is not None
    bad_components = list(inspection.report.components)
    bad_components[0] = bad_components[0].model_copy(
        update={"provenance": bad_provenance.model_copy(update={"source": "tampered"})}
    )
    provenance_report = inspection.report.model_copy(update={"components": bad_components})
    bad_detection = inspection.detection.model_copy(update={"warnings": ("tampered",)})
    bad_descriptor = inspection.descriptor.model_copy(update={"version": "tampered"})
    bad_zero_topk = inspection.report.facts.model_copy(
        update={"routed_top_k": 0, "routed_top_k_source": "config.num_experts_per_tok"}
    )
    bad_excess_topk = inspection.report.facts.model_copy(
        update={"routed_top_k": 5, "routed_top_k_source": "config.num_experts_per_tok"}
    )

    cases = (
        (subclass_inspection, tokens, {}, expected),
        (inspection, TokenTupleSubclass(tokens), {}, expected),
        (inspection, (subclass_token,), {}, expected),
        (inspection, tokens, KwargsSubclass(), expected),
        (inspection.model_copy(update={"descriptor": bad_descriptor}), tokens, {}, expected),
        (inspection.model_copy(update={"detection": bad_detection}), tokens, {}, expected),
        (inspection.model_copy(update={"report": bad_report}), tokens, {}, expected),
        (inspection.model_copy(update={"report": provenance_report}), tokens, {}, expected),
        (
            inspection.model_copy(
                update={"report": inspection.report.model_copy(update={"facts": bad_zero_topk})}
            ),
            tokens,
            {},
            expected,
        ),
        (
            inspection.model_copy(
                update={"report": inspection.report.model_copy(update={"facts": bad_excess_topk})}
            ),
            tokens,
            {},
            expected,
        ),
        (inspection, tokens, {}, expected - 1),
        (inspection, tokens, {}, True),
        (inspection, tokens, {}, 1.0),
    )
    for case_index, (candidate_inspection, candidate_tokens, kwargs, budget) in enumerate(cases):
        model = _HardKillModel()
        with pytest.raises((TypeError, ValueError)):
            run_qwen3_5_routing_forward(
                model,
                candidate_inspection,
                plan,
                candidate_tokens,
                kwargs,
                max_events=budget,  # type: ignore[arg-type]
            )
        assert model.calls == 0, case_index
        assert model.traversals == 0, case_index


@pytest.mark.parametrize(
    "mutation",
    ["duplicate", "mixed_run", "mixed_phase", "mixed_sequence", "list", "reverse", "gap"],
)
def test_qwen_token_sequence_matrix_is_strict_before_model(mutation: str) -> None:
    inspection = _qwen35_inspection()
    plan = build_routing_probe_plan(inspection)
    tokens = _qwen35_tokens(2)

    def variant(event: TokenEvent, **updates: object) -> TokenEvent:
        payload = event.model_dump(mode="json")
        payload.pop("token_key", None)
        payload.update(updates)
        return TokenEvent.model_validate(payload)

    candidates: dict[str, object] = {
        "duplicate": (tokens[0], tokens[1], tokens[0]),
        "mixed_run": (tokens[0], variant(tokens[1], run_key="run-other")),
        "mixed_phase": (tokens[0], variant(tokens[1], phase="decode")),
        "mixed_sequence": (tokens[0], variant(tokens[1], sequence_id="sequence-other")),
        "list": [tokens[0], tokens[1]],
        "reverse": tuple(reversed(tokens)),
        "gap": (tokens[0], variant(tokens[1], token_pos=2)),
    }
    model = _HardKillModel()
    with pytest.raises((TypeError, ValueError)):
        run_qwen3_5_routing_forward(
            model,
            inspection,
            plan,
            candidates[mutation],  # type: ignore[arg-type]
            {},
            max_events=16,
        )
    assert model.calls == 0
    assert model.traversals == 0


def test_qwen_probe_plan_subclass_is_rejected_before_model() -> None:
    inspection = _qwen35_inspection()
    plan = build_routing_probe_plan(inspection)

    class PlanSubclass(ProbePlan):
        pass

    subclass_plan = PlanSubclass.model_validate(plan.model_dump(mode="json"))
    model = _HardKillModel()
    with pytest.raises(TypeError, match="exact ProbePlan"):
        run_qwen3_5_routing_forward(
            model,
            inspection,
            subclass_plan,
            _qwen35_tokens(1),
            {},
            max_events=8,
        )
    assert model.calls == 0
    assert model.traversals == 0


def test_qwen_sequence_plan_json_id_topk_and_budget_are_preflighted() -> None:
    inspection = _qwen35_inspection()
    plan = build_routing_probe_plan(inspection)
    model = _QwenForwardModel("conditional")
    tokens = _qwen35_tokens(2)
    expected = len(tokens) * len(plan.targets) * inspection.report.facts.routed_top_k
    with pytest.raises(ValueError, match="insufficient"):
        run_qwen3_5_routing_forward(model, inspection, plan, tokens, {}, max_events=expected - 1)
    bad_plan = plan.model_copy(update={"include": ("missing",)})
    with pytest.raises((TypeError, ValueError)):
        run_qwen3_5_routing_forward(model, inspection, bad_plan, tokens, {}, max_events=expected)
    bad_tokens = tuple(
        TokenEvent(
            run_key=event.run_key,
            sequence_id=event.sequence_id,
            token_pos=event.token_pos + 1,
            token_id=event.token_id,
            token_text=event.token_text,
            phase=event.phase,
        )
        for event in tokens
    )
    with pytest.raises(ValueError, match="contiguous"):
        run_qwen3_5_routing_forward(model, inspection, plan, bad_tokens, {}, max_events=expected)
    assert model.calls == 0


@pytest.mark.parametrize(
    "bad",
    [
        {"input_ids ": object()},
        {"": object()},
        {True: object()},
        {1: object()},
        {None: object()},
    ],
)
def test_qwen_kwargs_and_budget_are_strict_before_model(bad: dict[str, object]) -> None:
    inspection = _qwen35_inspection()
    plan = build_routing_probe_plan(inspection)
    model = _QwenForwardModel("conditional")
    with pytest.raises(TypeError, match="keys"):
        run_qwen3_5_routing_forward(model, inspection, plan, _qwen35_tokens(1), bad, max_events=8)
    assert model.calls == 0


@pytest.mark.parametrize("max_events", [-1, "8"])
def test_qwen_negative_and_string_budgets_are_strict_before_model(
    max_events: object,
) -> None:
    inspection = _qwen35_inspection()
    plan = build_routing_probe_plan(inspection)
    model = _HardKillModel()
    with pytest.raises((TypeError, ValueError)):
        run_qwen3_5_routing_forward(
            model,
            inspection,
            plan,
            _qwen35_tokens(1),
            {},
            max_events=max_events,  # type: ignore[arg-type]
        )
    assert model.calls == 0
    assert model.traversals == 0


@pytest.mark.parametrize("oversize", [0, 3])
def test_qwen_exact_and_oversize_budgets_publish_complete_results(oversize: int) -> None:
    inspection = _qwen35_inspection()
    plan = build_routing_probe_plan(inspection)
    expected = len(plan.targets) * inspection.report.facts.routed_top_k
    model = _QwenForwardModel("conditional")
    result, _, _, _ = _run_qwen(model=model, max_events=expected + oversize)
    assert model.calls == 1
    assert len(result.routing_events) == expected
    assert result.routing_events == tuple(result.routing_events)


@pytest.mark.parametrize(
    "model_options,expected",
    [
        ({"reverse": True}, "canonical layer block order"),
        ({"duplicate": True}, "decode"),
        ({"fire_limit": 1}, "routing capture"),
    ],
)
def test_qwen_incomplete_duplicate_reverse_and_truncation_cleanup(
    model_options: dict[str, object], expected: str
) -> None:
    inspection = _qwen35_inspection()
    plan = build_routing_probe_plan(inspection)
    model = _QwenForwardModel("conditional", **model_options)
    with pytest.raises((RoutingCaptureError, ValueError), match=expected):
        run_qwen3_5_routing_forward(model, inspection, plan, _qwen35_tokens(1), {}, max_events=8)
    assert model.calls == 1
    assert all(not node.callbacks for node in model.nodes.values())


def test_qwen_capture_truncation_is_rejected_after_complete_budget() -> None:
    inspection = _qwen35_inspection()
    plan = build_routing_probe_plan(inspection)
    model = _QwenForwardModel("conditional", trailing_duplicate=True)
    with pytest.raises(ValueError, match="complete events"):
        run_qwen3_5_routing_forward(model, inspection, plan, _qwen35_tokens(1), {}, max_events=4)
    assert model.calls == 1
    assert all(not node.callbacks for node in model.nodes.values())


@pytest.mark.parametrize(
    "failure", [ValueError("body"), KeyboardInterrupt("cancel"), SystemExit("exit")]
)
def test_qwen_body_control_flow_preserves_exact_primary(failure: BaseException) -> None:
    model = _QwenForwardModel("conditional", failure=failure)
    inspection = _qwen35_inspection()
    plan = build_routing_probe_plan(inspection)
    with pytest.raises(type(failure)) as caught:
        run_qwen3_5_routing_forward(model, inspection, plan, _qwen35_tokens(1), {}, max_events=8)
    assert caught.value is failure
    assert all(not node.callbacks for node in model.nodes.values())


@pytest.mark.parametrize(
    "failure",
    [
        ValueError("decoder ordinary"),
        RoutingCaptureError("decode"),
        KeyboardInterrupt("decoder cancel"),
        SystemExit("decoder exit"),
    ],
)
def test_qwen_decoder_origin_control_flow_and_cleanup(
    failure: BaseException, monkeypatch: pytest.MonkeyPatch
) -> None:
    inspection = _qwen35_inspection()
    plan = build_routing_probe_plan(inspection)

    def fail_decoder(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise failure

    monkeypatch.setattr(Qwen3_5RoutingDecoder, "__call__", fail_decoder)
    model = _QwenForwardModel("conditional")
    if isinstance(failure, KeyboardInterrupt | SystemExit):
        with pytest.raises(type(failure)) as caught:
            run_qwen3_5_routing_forward(
                model, inspection, plan, _qwen35_tokens(1), {}, max_events=8
            )
        assert caught.value is failure
    else:
        with pytest.raises(RoutingCaptureError) as caught:
            run_qwen3_5_routing_forward(
                model, inspection, plan, _qwen35_tokens(1), {}, max_events=8
            )
        assert caught.value.stage == "decode"
        assert caught.value.__cause__ is failure
        assert not hasattr(caught.value, "pending_cleanup")
    assert all(not node.callbacks for node in model.nodes.values())


def test_qwen_registration_failure_preserves_primary_and_cleans_partial_hooks() -> None:
    model = _QwenForwardModel("conditional", registration_failure=True)
    inspection = _qwen35_inspection()
    plan = build_routing_probe_plan(inspection)
    with pytest.raises(OSError, match="registration failure") as caught:
        run_qwen3_5_routing_forward(model, inspection, plan, _qwen35_tokens(1), {}, max_events=8)
    assert caught.value.__cause__ is None
    assert not hasattr(caught.value, "pending_cleanup")
    assert all(not node.callbacks for node in model.nodes.values())


def test_qwen_registration_rollback_with_cleanup_failure_has_no_pending_handle() -> None:
    model = _QwenForwardModel(
        "conditional",
        registration_failure=True,
        removal_failures=1,
    )
    inspection = _qwen35_inspection()
    plan = build_routing_probe_plan(inspection)
    with pytest.raises(OSError, match="registration failure") as caught:
        run_qwen3_5_routing_forward(model, inspection, plan, _qwen35_tokens(1), {}, max_events=8)
    assert not hasattr(caught.value, "pending_cleanup")
    assert all(not node.callbacks for node in model.nodes.values())


def test_qwen_normal_body_cleanup_failure_is_retryable_without_pending_after_retry() -> None:
    model = _QwenForwardModel("conditional", removal_failures=1)
    inspection = _qwen35_inspection()
    plan = build_routing_probe_plan(inspection)
    with pytest.raises(RoutingCaptureError) as caught:
        run_qwen3_5_routing_forward(model, inspection, plan, _qwen35_tokens(1), {}, max_events=8)
    assert caught.value.stage == "lifecycle"
    assert not hasattr(caught.value, "pending_cleanup")
    assert all(not node.callbacks for node in model.nodes.values())


@pytest.mark.parametrize(
    "cleanup_failure",
    [KeyboardInterrupt("cleanup cancel"), SystemExit("cleanup exit")],
)
def test_qwen_cleanup_control_flow_failures_are_not_published(
    cleanup_failure: BaseException,
) -> None:
    model = _QwenForwardModel("conditional", cleanup_failure=cleanup_failure)
    inspection = _qwen35_inspection()
    plan = build_routing_probe_plan(inspection)
    with pytest.raises(RoutingCaptureError) as caught:
        run_qwen3_5_routing_forward(model, inspection, plan, _qwen35_tokens(1), {}, max_events=8)
    assert caught.value.stage == "lifecycle"
    assert not hasattr(caught.value, "pending_cleanup")
    assert all(not node.callbacks for node in model.nodes.values())


@pytest.mark.parametrize(
    "primary",
    [
        ValueError("body and cleanup"),
        KeyboardInterrupt("body and cleanup"),
        SystemExit("body and cleanup"),
    ],
)
def test_qwen_dual_primary_and_cleanup_control_flow_preserves_primary(
    primary: BaseException,
) -> None:
    model = _QwenForwardModel("conditional", failure=primary, cleanup_failure=OSError("cleanup"))
    inspection = _qwen35_inspection()
    plan = build_routing_probe_plan(inspection)
    with pytest.raises(type(primary)) as caught:
        run_qwen3_5_routing_forward(model, inspection, plan, _qwen35_tokens(1), {}, max_events=8)
    assert caught.value is primary
    assert any("hook cleanup failures" in note for note in caught.value.__notes__)
    assert not hasattr(caught.value, "pending_cleanup")
    assert all(not node.callbacks for node in model.nodes.values())


def test_qwen_transient_cleanup_retry_has_no_pending_handle() -> None:
    model = _QwenForwardModel("conditional", failure=ValueError("body"), removal_failures=1)
    inspection = _qwen35_inspection()
    plan = build_routing_probe_plan(inspection)
    with pytest.raises(ValueError, match="body") as caught:
        run_qwen3_5_routing_forward(model, inspection, plan, _qwen35_tokens(1), {}, max_events=8)
    assert not hasattr(caught.value, "pending_cleanup")
    assert all(not node.callbacks for node in model.nodes.values())


def test_qwen_persistent_cleanup_exposes_retryable_pending_handle() -> None:
    model = _QwenForwardModel("conditional", failure=ValueError("body"), removal_failures=3)
    inspection = _qwen35_inspection()
    plan = build_routing_probe_plan(inspection)
    with pytest.raises(ValueError, match="body") as caught:
        run_qwen3_5_routing_forward(model, inspection, plan, _qwen35_tokens(1), {}, max_events=8)
    pending = caught.value.pending_cleanup
    assert pending is caught.value.pending_runtime_cleanup
    assert pending.pending is True
    assert any("runtime cleanup also failed" in note for note in caught.value.__notes__)
    pending.retry()
    assert pending.pending is False
    assert all(not node.callbacks for node in model.nodes.values())
    pending.retry()


def test_qwen_result_direct_construction_and_fresh_routing_events() -> None:
    result, _, _, _ = _run_qwen()
    duplicate = RoutingForwardResult(result.output, result.token_events, result.routing_events)
    assert all(
        fresh is not original
        for fresh, original in zip(duplicate.routing_events, result.routing_events, strict=True)
    )
    route = result.routing_events[0]
    with pytest.raises(ValueError, match="unique"):
        RoutingForwardResult(result.output, result.token_events, (route, route))
    with pytest.raises(ValueError, match="selected"):
        RoutingForwardResult(
            result.output,
            result.token_events,
            (route.model_copy(update={"selected": False}), *result.routing_events[1:]),
        )


def test_qwen_direct_result_exact_containers_order_rank_unknown_and_completeness() -> None:
    class TokenTupleSubclass(tuple):
        pass

    class RoutingTupleSubclass(tuple):
        pass

    class RoutingSubclass(RoutingEvent):
        pass

    model = _QwenForwardModel(
        "conditional",
        rows=[[1.0, 2.0, 0.0, 3.0], [4.0, 0.0, 3.0, 1.0]],
    )
    result, _, _, _ = _run_qwen(model=model, tokens=_qwen35_tokens(2), max_events=16)
    tokens = result.token_events
    routes = result.routing_events
    foreign_token = TokenEvent(
        run_key=tokens[0].run_key,
        sequence_id=tokens[0].sequence_id,
        token_pos=0,
        token_id=tokens[0].token_id + 1,
        token_text=tokens[0].token_text,
        phase=tokens[0].phase,
    )
    route_subclass = RoutingSubclass.model_validate(routes[0].model_dump(mode="json"))
    invalid_cases = (
        (list(tokens), routes),
        (TokenTupleSubclass(tokens), routes),
        (tokens, list(routes)),
        (tokens, RoutingTupleSubclass(routes)),
        ((tokens[0], object()), routes),
        (tokens, (routes[0], object())),
        ((), routes),
        (tokens, ()),
        (tokens, tuple(reversed(routes))),
        (tokens, (routes[0].model_copy(update={"rank": -1}), *routes[1:])),
        (tokens, (route_subclass, *routes[1:])),
        (
            tokens,
            (
                routes[0].model_copy(update={"token_key": foreign_token.token_key}),
                *routes[1:],
            ),
        ),
        (tokens, routes[:2]),
    )
    for candidate_tokens, candidate_routes in invalid_cases:
        with pytest.raises((TypeError, ValueError)):
            RoutingForwardResult(result.output, candidate_tokens, candidate_routes)  # type: ignore[arg-type]


def test_qwen_capture_rejects_unknown_token_layer_expert_rank_and_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspection = _qwen35_inspection()
    plan = build_routing_probe_plan(inspection)
    tokens = _qwen35_tokens(1)
    foreign_token = TokenEvent(
        run_key=tokens[0].run_key,
        sequence_id=tokens[0].sequence_id,
        token_pos=0,
        token_id=tokens[0].token_id + 1,
        token_text=tokens[0].token_text,
        phase=tokens[0].phase,
    )
    unknown_layer = next(
        component.component_key
        for component in inspection.report.components
        if component.kind.value == "expert_container"
    )
    unknown_expert = next(
        component.component_key
        for component in inspection.report.components
        if component.kind.value == "shared_expert"
    )
    cases = (
        ("token_key", foreign_token.token_key),
        ("layer_key", unknown_layer),
        ("expert_key", unknown_expert),
        ("rank", 2),
        ("selected", False),
    )
    original = Qwen3_5RoutingDecoder.__call__
    for field_name, value in cases:
        with monkeypatch.context() as patcher:

            def tampered(
                decoder: Qwen3_5RoutingDecoder,
                context: object,
                module: object,
                inputs: tuple[object, ...],
                output: object,
                *,
                _field_name: str = field_name,
                _value: object = value,
            ) -> tuple[object, ...]:
                events = original(decoder, context, module, inputs, output)
                first = events[0].model_copy(update={_field_name: _value})
                return (first, *events[1:])

            patcher.setattr(Qwen3_5RoutingDecoder, "__call__", tampered)
            with pytest.raises((RoutingCaptureError, ValueError)):
                run_qwen3_5_routing_forward(
                    _QwenForwardModel("conditional"),
                    inspection,
                    plan,
                    tokens,
                    {},
                    max_events=8,
                )


def test_qwen_direct_result_rejects_unknown_token_rank_selected_and_missing_tokens() -> None:
    model = _QwenForwardModel(
        "conditional",
        rows=[[1.0, 2.0, 0.0, 3.0], [4.0, 0.0, 3.0, 1.0]],
    )
    result, _, _, _ = _run_qwen(
        model=model,
        tokens=_qwen35_tokens(2),
        max_events=16,
    )
    route = result.routing_events[0]
    token = result.token_events[0]
    foreign_token = TokenEvent(
        run_key=token.run_key,
        sequence_id=token.sequence_id,
        token_pos=0,
        token_id=token.token_id + 1,
        token_text=token.token_text,
        phase=token.phase,
    )
    with pytest.raises(ValueError, match="supplied token"):
        RoutingForwardResult(
            result.output,
            result.token_events,
            (
                route.model_copy(update={"token_key": foreign_token.token_key}),
                *result.routing_events[1:],
            ),
        )
    with pytest.raises(ValueError, match="rank"):
        RoutingForwardResult(
            result.output,
            result.token_events,
            (route.model_copy(update={"rank": 2}), *result.routing_events[1:]),
        )
    with pytest.raises(ValueError, match="selected"):
        RoutingForwardResult(
            result.output,
            result.token_events,
            (route.model_copy(update={"selected": False}), *result.routing_events[1:]),
        )
    with pytest.raises(ValueError, match="represented"):
        RoutingForwardResult(
            result.output,
            result.token_events,
            result.routing_events[:2],
        )


def test_qwen_decoder_and_events_are_not_retained() -> None:
    result, model, _, _ = _run_qwen()
    model_ref = weakref.ref(model)
    payload_refs = tuple(model.payload_refs)
    del model
    gc.collect()
    assert model_ref() is None
    assert all(reference() is None for reference in payload_refs)
    assert result.output is not None


def test_qwen_result_keeps_output_identity_until_result_release() -> None:
    class Output:
        pass

    model = _QwenForwardModel("conditional")
    model.output = Output()
    result, _, _, _ = _run_qwen(model=model)
    output_ref = weakref.ref(result.output)
    assert output_ref() is result.output
    del model
    gc.collect()
    assert output_ref() is result.output
    del result
    gc.collect()
    assert output_ref() is None


@pytest.mark.parametrize("mode", ["success", "decoder", "body", "cleanup"])
def test_qwen_success_and_failure_release_inspection_plan_inputs_and_runtime_objects(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_options: dict[str, object] = {}
    if mode == "body":
        model_options["failure"] = ValueError("retention body")
    elif mode == "cleanup":
        model_options["removal_failures"] = 3
    model = _QwenForwardModel("conditional", **model_options)
    inspection = _qwen35_inspection()
    plan = build_routing_probe_plan(inspection)
    token_holder = _RetentionObject()
    token_holder.events = _qwen35_tokens(1)
    kwargs_holder = _RetentionObject()
    kwargs_holder.value = _RetentionObject()
    kwargs_holder.mapping = {"holder": kwargs_holder.value}

    inspection_ref = weakref.ref(inspection)
    plan_ref = weakref.ref(plan)
    token_refs = tuple(weakref.ref(token) for token in token_holder.events)
    kwargs_holder_ref = weakref.ref(kwargs_holder)
    kwargs_value_ref = weakref.ref(kwargs_holder.value)
    model_ref = weakref.ref(model)
    module_ref = weakref.ref(next(iter(model.nodes.values())))
    input_ref = weakref.ref(model.input_value)

    result: RoutingForwardResult | None = None
    caught: BaseException | None = None
    error: object | None = None
    pending: object | None = None
    if mode == "decoder":
        with monkeypatch.context() as patcher:

            def fail_decoder(*args: object, **kwargs: object) -> object:
                del args, kwargs
                raise ValueError("retention decoder")

            patcher.setattr(Qwen3_5RoutingDecoder, "__call__", fail_decoder)
            with pytest.raises(RoutingCaptureError) as error:
                run_qwen3_5_routing_forward(
                    model,
                    inspection,
                    plan,
                    token_holder.events,
                    kwargs_holder.mapping,
                    max_events=8,
                )
            caught = error.value
    elif mode == "success":
        result = run_qwen3_5_routing_forward(
            model,
            inspection,
            plan,
            token_holder.events,
            kwargs_holder.mapping,
            max_events=8,
        )
    else:
        with pytest.raises(BaseException) as error:
            run_qwen3_5_routing_forward(
                model,
                inspection,
                plan,
                token_holder.events,
                kwargs_holder.mapping,
                max_events=8,
            )
        caught = error.value
        if mode == "cleanup":
            pending = getattr(caught, "pending_cleanup", None)
            assert pending is not None
            pending.retry()

    payload_refs = tuple(model.payload_refs)
    if result is not None:
        assert result.output is model.output
    assert all(not node.callbacks for node in model.nodes.values())
    del caught, result, error, pending
    del model, model_options, inspection, plan, token_holder, kwargs_holder
    gc.collect()

    assert inspection_ref() is None
    assert plan_ref() is None
    assert all(reference() is None for reference in token_refs)
    assert kwargs_holder_ref() is None
    assert kwargs_value_ref() is None
    assert model_ref() is None
    assert module_ref() is None
    assert input_ref() is None
    assert all(reference() is None for reference in payload_refs)


def test_qwen_source_ast_is_offline_and_family_isolated() -> None:
    path = Path("src/moeatlas/runtime/routing_forward.py")
    tree = ast.parse(path.read_text())
    forbidden = {
        "accelerate",
        "builtins",
        "cachetools",
        "compileall",
        "torch",
        "transformers",
        "safetensors",
        "numpy",
        "np",
        "requests",
        "urllib",
        "socket",
        "os",
        "pathlib",
        "tempfile",
        "subprocess",
        "importlib",
        "shutil",
        "sys",
        "atexit",
        "signal",
        "store",
        "server",
        "ui",
    }
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert imported.isdisjoint(forbidden)
    forbidden_calls = {
        "__import__",
        "eval",
        "exec",
        "compile",
        "open",
        "import_module",
        "load_module",
        "run",
        "Popen",
        "system",
        "popen",
        "makedirs",
        "mkdir",
        "unlink",
        "remove",
        "replace",
        "rename",
        "getattr",
        "setattr",
        "globals",
        "locals",
        "vars",
    }
    assert not any(
        isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id in forbidden_calls)
            or (isinstance(node.func, ast.Attribute) and node.func.attr in forbidden_calls)
        )
        for node in ast.walk(tree)
    )


def test_qwen_forward_function_isolated_from_mixtral_decoder_and_branches() -> None:
    tree = ast.parse(Path("src/moeatlas/runtime/routing_forward.py").read_text())
    qwen_functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and node.name == "run_qwen3_5_routing_forward"
    ]
    assert len(qwen_functions) == 1
    function = qwen_functions[0]
    names = {node.id for node in ast.walk(function) if isinstance(node, ast.Name)}
    attributes = {node.attr for node in ast.walk(function) if isinstance(node, ast.Attribute)}
    assert not any("Mixtral" in name for name in names | attributes)
    assert "Qwen3_5RoutingDecoder" in names


def test_qwen_forward_is_offline_and_does_not_touch_tmp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("network access forbidden")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    before = tuple(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    _run_qwen()
    after = tuple(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    assert after == before


def test_qwen_forward_hardkills_filesystem_network_process_and_cache_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("runtime side effect forbidden")

    cache_env = {
        "XDG_CACHE_HOME": tmp_path / "xdg-cache",
        "HF_HOME": tmp_path / "hf-home",
        "HF_HUB_CACHE": tmp_path / "hf-hub-cache",
        "TRANSFORMERS_CACHE": tmp_path / "transformers-cache",
        "TORCH_HOME": tmp_path / "torch-home",
    }
    for cache_path in cache_env.values():
        cache_path.mkdir()
    for key, cache_path in cache_env.items():
        monkeypatch.setenv(key, str(cache_path))
    before_caches = {
        key: tuple(path.relative_to(cache_path).as_posix() for path in cache_path.rglob("*"))
        for key, cache_path in cache_env.items()
    }
    for module, name in (
        (builtins, "open"),
        (os, "open"),
        (os, "makedirs"),
        (os, "mkdir"),
        (os, "remove"),
        (os, "unlink"),
        (os, "replace"),
        (os, "rename"),
        (tempfile, "mkdtemp"),
        (subprocess, "Popen"),
        (subprocess, "run"),
    ):
        monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    before_env = dict(os.environ)
    before_tree = tuple(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    result, _, _, _ = _run_qwen()
    after_tree = tuple(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    assert result.output is not None
    assert after_tree == before_tree
    assert dict(os.environ) == before_env
    after_caches = {
        key: tuple(path.relative_to(cache_path).as_posix() for path in cache_path.rglob("*"))
        for key, cache_path in cache_env.items()
    }
    assert after_caches == before_caches
    source = Path("src/moeatlas/runtime/routing_forward.py").read_text()
    assert not any(
        marker in source
        for marker in (
            "HF_HOME",
            "TRANSFORMERS_CACHE",
            "TORCH_HOME",
            "XDG_CACHE_HOME",
            "MOEATLAS_CACHE",
        )
    )


def test_qwen_public_decoder_and_session_are_the_actual_components() -> None:
    assert Qwen3_5RoutingDecoder.__module__ == "moeatlas.runtime.qwen3_5_routing"
    assert RoutingCaptureSession.__module__ == "moeatlas.runtime.routing"
