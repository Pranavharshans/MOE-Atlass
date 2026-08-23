from __future__ import annotations

import ast
import gc
import inspect
import socket
import urllib.request
import weakref
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from moeatlas.adapters import build_routing_probe_plan
from moeatlas.probe import ProbeResolutionError
from moeatlas.runtime import (
    RoutingCaptureError,
    RoutingForwardResult,
    run_mixtral_routing_forward,
)

from .fixtures.mixtral import MixtralForCausalLM
from .test_mixtral_routing_decoder import (
    _inspection,
    _payload,
    _qwen_inspection,
    _tokens,
)


class _Handle:
    def __init__(self, owner: _Gate, callback: object) -> None:
        self.owner = owner
        self.callback = callback

    def remove(self) -> None:
        if self.owner.removal_failures:
            self.owner.removal_failures -= 1
            raise OSError("transient removal")
        if self.callback in self.owner.callbacks:
            self.owner.callbacks.remove(self.callback)


class _Gate:
    def __init__(
        self,
        path: str,
        *,
        removal_failures: int = 0,
        registration_failures: int = 0,
    ) -> None:
        self.path = path
        self.callbacks: list[object] = []
        self.removal_failures = removal_failures
        self.registration_failures = registration_failures

    def register_forward_hook(self, callback: object) -> _Handle:
        if self.registration_failures:
            self.registration_failures -= 1
            raise OSError("transient registration")
        self.callbacks.append(callback)
        return _Handle(self, callback)

    def fire(self, output: object) -> None:
        for callback in tuple(self.callbacks):
            callback(self, (), output)


class _ForwardModel:
    def __init__(
        self,
        layout: str,
        *,
        output: object | None = None,
        rows: list[list[float]] | None = None,
        fire_limit: int | None = None,
        removal_failures: int = 0,
        registration_failure: bool = False,
        duplicate_router: bool = False,
        reverse_router_order: bool = False,
    ) -> None:
        source = MixtralForCausalLM(layout=layout)
        self.config = source.config
        self.output = output if output is not None else object()
        self.rows = rows or [[1.0, 2.0, 0.0, 3.0]]
        self.fire_limit = fire_limit
        self.duplicate_router = duplicate_router
        self.reverse_router_order = reverse_router_order
        self.calls = 0
        self.named_modules_calls = 0
        self.received_kwargs: dict[str, object] | None = None
        self.nodes: dict[str, _Gate] = {}
        self._entries: list[tuple[str, object]] = []
        gate_paths = [path for path, _ in source.named_modules() if path and path.endswith(".gate")]
        last_gate_path = gate_paths[-1] if gate_paths else None
        for path, module in source.named_modules():
            if path and path.endswith(".gate"):
                node = _Gate(
                    path,
                    removal_failures=removal_failures,
                    registration_failures=(
                        1 if registration_failure and path == last_gate_path else 0
                    ),
                )
                self.nodes[path] = node
                self._entries.append((path, node))
            else:
                self._entries.append((path, module))

    def named_modules(self):
        self.named_modules_calls += 1
        return iter(self._entries)

    def __call__(self, **kwargs: object) -> object:
        self.calls += 1
        self.received_kwargs = kwargs
        payload = _payload("packed" if self.nodes and self._is_packed() else "legacy", self.rows)
        nodes = list(self.nodes.values())
        if self.fire_limit is not None:
            nodes = nodes[: self.fire_limit]
        if self.reverse_router_order:
            nodes.reverse()
        for node in nodes:
            node.fire(payload)
            if self.duplicate_router and node is nodes[0]:
                node.fire(payload)
        return self.output

    def _is_packed(self) -> bool:
        return any(path.endswith(".mlp.gate") for path in self.nodes)


def _plan(inspection):
    return build_routing_probe_plan(inspection)


def _run(
    layout: str = "legacy",
    *,
    token_count: int = 2,
    max_events: int = 32,
    model: _ForwardModel | None = None,
    model_kwargs: dict[str, object] | None = None,
):
    inspection = _inspection(layout)
    model = model or _ForwardModel(layout, rows=[[1.0, 2.0, 0.0, 3.0]] * token_count)
    result = run_mixtral_routing_forward(
        model,
        inspection,
        _plan(inspection),
        _tokens(token_count),
        {} if model_kwargs is None else model_kwargs,
        max_events=max_events,
    )
    return result, model, inspection


def test_public_api_and_result_dataclass_contract() -> None:
    import moeatlas.runtime.routing_forward as routing_forward

    assert not hasattr(routing_forward, "MixtralRoutingForwardResult")
    assert RoutingForwardResult.__name__ == "RoutingForwardResult"
    assert RoutingForwardResult.__slots__ == (
        "output",
        "token_events",
        "routing_events",
    )
    fields = RoutingForwardResult.__dataclass_fields__
    assert tuple(fields) == ("output", "token_events", "routing_events")
    assert fields["output"].repr is False
    assert RoutingForwardResult.__dataclass_params__.eq is False
    signature = inspect.signature(run_mixtral_routing_forward)
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


@pytest.mark.parametrize("layout", ["legacy", "packed"])
def test_one_forward_complete_capture_and_output_identity(layout: str) -> None:
    marker = object()
    result, model, inspection = _run(
        layout,
        token_count=2,
        model=_ForwardModel(
            layout, output=marker, rows=[[1.0, 2.0, 0.0, 3.0], [4.0, 0.0, 3.0, 1.0]]
        ),
    )
    assert result.output is marker
    assert model.calls == 1
    assert len(result.token_events) == 2
    assert len(result.routing_events) == 2 * len(model.nodes) * inspection.report.facts.routed_top_k
    assert all(node.callbacks == [] for node in model.nodes.values())
    assert "output=" not in repr(result)


def test_result_is_frozen_identity_equal_false_and_output_caller_owned() -> None:
    result, _, _ = _run("legacy", token_count=1)
    with pytest.raises(FrozenInstanceError):
        result.output = object()
    duplicate = RoutingForwardResult(
        result.output, result.token_events, result.routing_events
    )
    assert duplicate is not result
    assert duplicate != result
    assert duplicate.output is result.output


def test_preflight_rejects_noncallable_model_and_does_not_traverse() -> None:
    inspection = _inspection("legacy")
    model = _ForwardModel("legacy")
    with pytest.raises(TypeError, match="callable"):
        run_mixtral_routing_forward(
            object(), inspection, _plan(inspection), _tokens(1), {}, max_events=8
        )
    assert model.named_modules_calls == 0


@pytest.mark.parametrize("bad", [(), -1, True, 0, 1.0])
def test_preflight_rejects_invalid_budget_before_model_or_hooks(bad: object) -> None:
    inspection = _inspection("legacy")
    model = _ForwardModel("legacy")
    with pytest.raises((TypeError, ValueError)):
        run_mixtral_routing_forward(
            model, inspection, _plan(inspection), _tokens(1), {}, max_events=bad
        )
    assert model.calls == 0
    assert model.named_modules_calls == 0


def test_insufficient_budget_is_rejected_before_model_or_hooks() -> None:
    inspection = _inspection("legacy")
    model = _ForwardModel("legacy")
    expected = (
        len(_tokens(2)) * len(_plan(inspection).targets) * inspection.report.facts.routed_top_k
    )
    with pytest.raises(ValueError, match="insufficient"):
        run_mixtral_routing_forward(
            model, inspection, _plan(inspection), _tokens(2), {}, max_events=expected - 1
        )
    assert model.calls == 0
    assert model.named_modules_calls == 0
    assert all(node.callbacks == [] for node in model.nodes.values())


def test_kwargs_are_exact_dict_shallow_copied_without_value_inspection() -> None:
    class NoInspect:
        def __deepcopy__(self, memo: object) -> object:
            raise AssertionError("kwargs values must not be copied")

    value = NoInspect()
    model = _ForwardModel("legacy")
    result, _, _ = _run("legacy", token_count=1, model=model, model_kwargs={"value": value})
    assert result.output is model.output
    assert model.received_kwargs is not None
    assert model.received_kwargs["value"] is value

    class DictSubclass(dict[str, object]):
        pass

    inspection = _inspection("legacy")
    with pytest.raises(TypeError, match="exact dict"):
        run_mixtral_routing_forward(
            model, inspection, _plan(inspection), _tokens(1), DictSubclass(), max_events=8
        )


@pytest.mark.parametrize("key", ["", " value", "value ", "value\t"])
def test_kwargs_keys_are_nonempty_trimmed_exact_strings(key: str) -> None:
    inspection = _inspection("legacy")
    model = _ForwardModel("legacy")
    with pytest.raises(TypeError, match="keys"):
        run_mixtral_routing_forward(
            model, inspection, _plan(inspection), _tokens(1), {key: 1}, max_events=8
        )

    class StrSubclass(str):
        pass

    with pytest.raises(TypeError, match="keys"):
        run_mixtral_routing_forward(
            model,
            inspection,
            _plan(inspection),
            _tokens(1),
            {StrSubclass("value"): 1},
            max_events=8,
        )


def test_preflight_revalidates_family_and_canonical_plan() -> None:
    with pytest.raises(ValueError):
        run_mixtral_routing_forward(
            _ForwardModel("legacy"),
            _qwen_inspection(),
            _plan(_inspection("legacy")),
            _tokens(1),
            {},
            max_events=8,
        )
    inspection = _inspection("legacy")
    plan = _plan(inspection)
    bad_plan = plan.model_copy(update={"include": ("missing",)})
    with pytest.raises((TypeError, ValueError)):
        run_mixtral_routing_forward(
            _ForwardModel("legacy"), inspection, bad_plan, _tokens(1), {}, max_events=8
        )


@pytest.mark.parametrize("layout", ["legacy", "packed"])
def test_model_exception_keyboardinterrupt_systemexit_are_exact_and_cleanup(layout: str) -> None:
    inspection = _inspection(layout)
    plan = _plan(inspection)
    for failure in (ValueError("body"), KeyboardInterrupt("cancel"), SystemExit("exit")):

        class FailingModel(_ForwardModel):
            error = failure

            def __call__(self, **kwargs: object) -> object:
                super().__call__(**kwargs)
                raise self.error

        model = FailingModel(layout)
        with pytest.raises(type(failure)) as caught:
            run_mixtral_routing_forward(model, inspection, plan, _tokens(1), {}, max_events=8)
        assert caught.value is failure
        assert all(node.callbacks == [] for node in model.nodes.values())


def test_partial_rows_and_duplicate_router_capture_do_not_publish() -> None:
    model = _ForwardModel("legacy", fire_limit=1)
    inspection = _inspection("legacy")
    with pytest.raises((RoutingCaptureError, ValueError)):
        run_mixtral_routing_forward(
            model, inspection, _plan(inspection), _tokens(1), {}, max_events=8
        )
    assert all(node.callbacks == [] for node in model.nodes.values())


@pytest.mark.parametrize("layout", ["legacy", "packed"])
def test_cleanup_failure_prevents_publication(layout: str) -> None:
    inspection = _inspection(layout)
    model = _ForwardModel(layout, removal_failures=1)
    with pytest.raises(RoutingCaptureError, match="lifecycle"):
        run_mixtral_routing_forward(
            model, inspection, _plan(inspection), _tokens(1), {}, max_events=8
        )
    assert all(node.callbacks == [] for node in model.nodes.values())


@pytest.mark.parametrize("layout", ["legacy", "packed"])
def test_duplicate_same_router_fire_is_decode_failure_and_cleans_hooks(layout: str) -> None:
    inspection = _inspection(layout)
    model = _ForwardModel(layout, duplicate_router=True)
    with pytest.raises(RoutingCaptureError, match="decode"):
        run_mixtral_routing_forward(
            model, inspection, _plan(inspection), _tokens(1), {}, max_events=8
        )
    assert model.calls == 1
    assert all(node.callbacks == [] for node in model.nodes.values())


def test_enter_failure_and_partial_registration_retry_cleanup_preserve_primary() -> None:
    inspection = _inspection("legacy")

    enter_failure_exception = RuntimeError("enter failure")

    class EnterFailure(_ForwardModel):
        def named_modules(self):
            self.named_modules_calls += 1
            raise enter_failure_exception

    enter_failure = EnterFailure("legacy")
    with pytest.raises(ProbeResolutionError) as enter_error:
        run_mixtral_routing_forward(
            enter_failure,
            inspection,
            _plan(inspection),
            _tokens(1),
            {},
            max_events=8,
        )
    assert not hasattr(enter_error.value, "pending_cleanup")
    assert not hasattr(enter_error.value, "pending_runtime_cleanup")
    assert enter_error.value.__cause__ is enter_failure_exception
    assert all(node.callbacks == [] for node in enter_failure.nodes.values())

    partial = _ForwardModel("legacy", registration_failure=True, removal_failures=1)
    with pytest.raises(OSError) as partial_error:
        run_mixtral_routing_forward(
            partial, inspection, _plan(inspection), _tokens(1), {}, max_events=8
        )
    assert not hasattr(partial_error.value, "pending_cleanup")
    assert all(node.callbacks == [] for node in partial.nodes.values())


@pytest.mark.parametrize(
    "failure", [ValueError("body"), KeyboardInterrupt("cancel"), SystemExit("exit")]
)
def test_body_control_flow_with_transient_cleanup_preserves_exact_primary(
    failure: BaseException,
) -> None:
    class FailingModel(_ForwardModel):
        error = failure

        def __call__(self, **kwargs: object) -> object:
            super().__call__(**kwargs)
            raise self.error

    model = FailingModel("legacy", removal_failures=1)
    inspection = _inspection("legacy")
    with pytest.raises(type(failure)) as caught:
        run_mixtral_routing_forward(
            model, inspection, _plan(inspection), _tokens(1), {}, max_events=8
        )
    assert caught.value is failure
    assert not hasattr(caught.value, "pending_cleanup")
    assert all(node.callbacks == [] for node in model.nodes.values())


def test_persistent_cleanup_failure_attaches_retryable_pending_to_primary() -> None:
    class FailingModel(_ForwardModel):
        def __call__(self, **kwargs: object) -> object:
            super().__call__(**kwargs)
            raise ValueError("body")

    model = FailingModel("legacy", removal_failures=2)
    inspection = _inspection("legacy")
    with pytest.raises(ValueError, match="body") as caught:
        run_mixtral_routing_forward(
            model, inspection, _plan(inspection), _tokens(1), {}, max_events=8
        )
    pending = caught.value.pending_cleanup
    assert pending is caught.value.pending_runtime_cleanup
    assert pending.pending is True
    assert any("runtime cleanup also failed" in note for note in caught.value.__notes__)
    assert any(node.callbacks for node in model.nodes.values())
    pending.retry()
    assert pending.pending is False
    assert all(node.callbacks == [] for node in model.nodes.values())
    pending.retry()


@pytest.mark.parametrize("layout", ["legacy", "packed"])
def test_wrapper_rejects_reverse_model_router_order_after_cleanup(layout: str) -> None:
    inspection = _inspection(layout)
    model = _ForwardModel(layout, reverse_router_order=True)
    with pytest.raises(ValueError, match="canonical layer block order"):
        run_mixtral_routing_forward(
            model, inspection, _plan(inspection), _tokens(1), {}, max_events=8
        )
    assert all(node.callbacks == [] for node in model.nodes.values())


def test_result_event_inputs_are_fresh_copies() -> None:
    tokens = _tokens(1)
    inspection = _inspection("legacy")
    model = _ForwardModel("legacy")
    result = run_mixtral_routing_forward(
        model, inspection, _plan(inspection), tokens, {}, max_events=8
    )
    assert result.token_events is not tokens
    assert all(
        result_event is not source_event
        for result_event, source_event in zip(result.token_events, tokens, strict=True)
    )
    duplicate = RoutingForwardResult(
        result.output, result.token_events, result.routing_events
    )
    assert all(
        duplicate_event is not source_event
        for duplicate_event, source_event in zip(
            duplicate.routing_events, result.routing_events, strict=True
        )
    )


def test_payload_and_kwargs_values_are_not_retained() -> None:
    class NonRetainingModel(_ForwardModel):
        def __init__(self) -> None:
            super().__init__("legacy")
            self.payload_refs: list[weakref.ReferenceType[object]] = []

        def __call__(self, **kwargs: object) -> object:
            self.calls += 1
            payload = _payload("legacy", self.rows)
            self.payload_refs.append(weakref.ref(payload))
            for node in self.nodes.values():
                node.fire(payload)
            return object()

    model = NonRetainingModel()
    kwarg_value = _payload("legacy", [[1.0, 2.0, 0.0, 3.0]])
    kwarg_ref = weakref.ref(kwarg_value)
    kwargs = {"value": kwarg_value}
    result = run_mixtral_routing_forward(
        model, _inspection("legacy"), _plan(_inspection("legacy")), _tokens(1), kwargs, max_events=8
    )
    del kwargs
    del kwarg_value
    gc.collect()
    assert result.output is not None
    assert kwarg_ref() is None
    assert all(reference() is None for reference in model.payload_refs)


def test_result_rejects_invalid_links_missing_tokens_duplicates_and_unselected() -> None:
    valid, _, _ = _run("legacy", token_count=1)
    route = valid.routing_events[0]
    with pytest.raises(ValueError, match="supplied token"):
        RoutingForwardResult(
            valid.output,
            valid.token_events,
            (
                route.model_copy(update={"token_key": _tokens(2)[1].token_key}),
                *valid.routing_events[1:],
            ),
        )
    with pytest.raises(ValueError, match="represented"):
        RoutingForwardResult(valid.output, _tokens(2), valid.routing_events)
    with pytest.raises(ValueError, match="unique"):
        RoutingForwardResult(valid.output, valid.token_events, (route, route))
    with pytest.raises(ValueError, match="selected"):
        RoutingForwardResult(
            valid.output,
            valid.token_events,
            (route.model_copy(update={"selected": False}), *valid.routing_events[1:]),
        )


def test_result_rejects_nondeterministic_layer_order() -> None:
    valid, _, _ = _run("legacy", token_count=2)
    routes = list(valid.routing_events)
    routes[1], routes[2] = routes[2], routes[1]
    with pytest.raises(ValueError, match="deterministic"):
        RoutingForwardResult(valid.output, valid.token_events, tuple(routes))


def test_source_ast_has_no_optional_or_dynamic_imports() -> None:
    source_path = Path("src/moeatlas/runtime/routing_forward.py")
    module = ast.parse(source_path.read_text())
    forbidden = {"torch", "transformers", "accelerate", "safetensors", "numpy", "np", "importlib"}
    imports: set[str] = set()
    dynamic: list[ast.Call] = []
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add((node.module or "").split(".")[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in {"__import__", "import_module"}:
                dynamic.append(node)
            if isinstance(node.func, ast.Attribute) and node.func.attr == "import_module":
                dynamic.append(node)
    assert imports.isdisjoint(forbidden)
    assert not dynamic


def test_forward_is_offline_and_does_not_touch_tmp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("network access forbidden")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    cache = tmp_path / "cache"
    cache.mkdir()
    before = tuple(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    for layout in ("legacy", "packed"):
        _run(layout, token_count=1)
    after = tuple(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    assert after == before
