from __future__ import annotations

import ast
import gc
import inspect
import math
import socket
import urllib.request
import weakref
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from moeatlas.adapters import (
    MixtralStaticAdapter,
    Qwen3MoeStaticAdapter,
    build_routing_probe_plan,
    inspect_static_adapter,
)
from moeatlas.core import (
    ComponentKind,
    DType,
    ModelManifest,
    TokenizerIdentity,
    make_config_hash,
    make_model_key,
)
from moeatlas.events import TokenEvent, TokenPhase
from moeatlas.probe import ProbeTarget
from moeatlas.runtime import (
    MixtralRoutingDecoder,
    RoutingCaptureSession,
    RoutingCaptureTarget,
)

from .fixtures.mixtral import MixtralForCausalLM
from .fixtures.qwen3_moe import Qwen3MoeForCausalLM


def _inspection(layout: str):
    model = MixtralForCausalLM(layout=layout)
    manifest = ModelManifest(
        model_key=make_model_key("acme/mixtral", "r1"),
        architecture="mixtral",
        revision="r1",
        config_hash=make_config_hash({"architecture": "mixtral", "revision": "r1"}),
        tokenizer=TokenizerIdentity(identifier="acme/mixtral-tokenizer", revision="r1"),
        dtype=DType.FLOAT32,
        device_map={"": "cpu"},
    )
    return inspect_static_adapter(MixtralStaticAdapter(), model, model.config, manifest)


def _qwen_inspection():
    model = Qwen3MoeForCausalLM(layout="legacy_indexed")
    manifest = ModelManifest(
        model_key=make_model_key("acme/qwen3-moe", "r1"),
        architecture="qwen3_moe",
        revision="r1",
        config_hash=make_config_hash({"architecture": "qwen3_moe", "revision": "r1"}),
        tokenizer=TokenizerIdentity(identifier="acme/qwen3-tokenizer", revision="r1"),
        dtype=DType.FLOAT32,
        device_map={"": "cpu"},
    )
    return inspect_static_adapter(Qwen3MoeStaticAdapter(), model, model.config, manifest)


def _tokens(count: int = 2) -> tuple[TokenEvent, ...]:
    return tuple(
        TokenEvent(
            run_key="run-1",
            sequence_id="sequence-1",
            token_pos=index,
            token_id=index + 10,
            token_text=str(index),
            phase=TokenPhase.PREFILL,
        )
        for index in range(count)
    )


def _target(inspection, *, router_index: int = 0) -> RoutingCaptureTarget:
    routers = sorted(
        (
            component
            for component in inspection.report.components
            if component.kind.value == "router"
        ),
        key=lambda component: component.module_path,
    )
    router = routers[router_index]
    layer = next(
        component
        for component in inspection.report.components
        if component.kind.value == "moe_layer" and component.layer_index == router.layer_index
    )
    experts = sorted(
        (
            component
            for component in inspection.report.components
            if component.kind.value == "expert" and component.layer_index == router.layer_index
        ),
        key=lambda component: component.expert_index,
    )
    return RoutingCaptureTarget(
        router=ProbeTarget(
            module_path=router.module_path,
            component_key=router.component_key,
            component_kind=router.kind,
        ),
        layer_key=layer.component_key,
        expert_keys=tuple(component.component_key for component in experts),
        routed_top_k=inspection.report.facts.routed_top_k,
    )


class _TensorLike:
    def __init__(self, value: object, *, fail: BaseException | None = None) -> None:
        self.value = value
        self.fail = fail
        self.calls: list[str] = []

    def detach(self) -> _TensorLike:
        self.calls.append("detach")
        if self.fail is not None:
            raise self.fail
        return self

    def cpu(self) -> _TensorLike:
        self.calls.append("cpu")
        return self

    def float(self) -> _TensorLike:
        self.calls.append("float")
        return self

    def tolist(self) -> object:
        self.calls.append("tolist")
        return self.value


def _packed_scores(row: list[float], selected: list[int]) -> list[float]:
    maximum = max(row)
    values = [math.exp(value - maximum) for value in row]
    probabilities = [value / sum(values) for value in values]
    total = sum(probabilities[index] for index in selected)
    return [probabilities[index] / total for index in selected]


def _payload(layout: str, rows: list[list[float]], top_k: int = 2):
    if layout == "legacy":
        return _TensorLike(rows)
    selected_rows = []
    score_rows = []
    for row in rows:
        selected = sorted(range(len(row)), key=lambda index: (-row[index], index))[:top_k]
        selected_rows.append(selected)
        score_rows.append(_packed_scores(row, selected))
    return (
        _TensorLike(rows),
        _TensorLike(score_rows),
        _TensorLike(selected_rows),
    )


def test_public_surface_and_exact_constructor() -> None:
    assert MixtralRoutingDecoder.__module__ == "moeatlas.runtime.mixtral_routing"
    signature = inspect.signature(MixtralRoutingDecoder)
    assert tuple(signature.parameters) == ("inspection", "token_events")
    assert all(
        parameter.default is inspect.Parameter.empty for parameter in signature.parameters.values()
    )
    call_signature = inspect.signature(MixtralRoutingDecoder.__call__)
    assert tuple(call_signature.parameters) == ("self", "context", "module", "inputs", "output")
    assert MixtralRoutingDecoder.__slots__ == (
        "_bindings",
        "_inspection_layout",
        "_token_events",
        "_used_paths",
    )


@pytest.mark.parametrize("layout", ["legacy", "packed"])
def test_known_values_emit_observed_evidence_only(layout: str) -> None:
    inspection = _inspection(layout)
    decoder = MixtralRoutingDecoder(inspection, _tokens(2))
    output = _payload(layout, [[1.0, 2.0, 0.0, 3.0], [4.0, 0.0, 3.0, 1.0]])
    events = decoder(_target(inspection), object(), (), output)
    assert type(events) is tuple
    assert len(events) == 4
    assert [event.rank for event in events] == [0, 1, 0, 1]
    assert all(event.selected is True for event in events)
    assert all(event.probability is None for event in events)
    if layout == "legacy":
        assert all(event.weight is None for event in events)
        assert [event.router_logit for event in events] == [3.0, 2.0, 4.0, 3.0]
    else:
        assert all(event.weight is not None for event in events)
        assert [event.router_logit for event in events] == [3.0, 2.0, 4.0, 3.0]


@pytest.mark.parametrize("layout", ["legacy", "packed"])
def test_conversion_order_and_no_tensor_retention(layout: str) -> None:
    inspection = _inspection(layout)
    decoder = MixtralRoutingDecoder(inspection, _tokens(1))
    if layout == "legacy":
        tensor = _TensorLike([[1.0, 2.0, 0.0, 3.0]])
        decoder(_target(inspection), object(), (), tensor)
        assert tensor.calls == ["detach", "cpu", "float", "tolist"]
    else:
        logits = _TensorLike([[1.0, 2.0, 0.0, 3.0]])
        selected = [3, 1]
        scores = _TensorLike([_packed_scores([1.0, 2.0, 0.0, 3.0], selected)])
        indices = _TensorLike([selected])
        decoder(_target(inspection), object(), (), (logits, scores, indices))
        assert logits.calls == ["detach", "cpu", "float", "tolist"]
        assert scores.calls == ["detach", "cpu", "float", "tolist"]
        assert indices.calls == ["detach", "cpu", "tolist"]
    assert not any(name in decoder.__slots__ for name in ("_output", "_tensor", "_payload"))


@pytest.mark.parametrize("layout", ["legacy", "packed"])
def test_payload_objects_are_collectible_after_successful_call(layout: str) -> None:
    inspection = _inspection(layout)
    decoder = MixtralRoutingDecoder(inspection, _tokens(1))
    rows = [[1.0, 2.0, 0.0, 3.0]]
    if layout == "legacy":
        payload = _TensorLike(rows)
        references = (weakref.ref(payload),)
        decoder(_target(inspection), object(), (), payload)
        del payload
    else:
        selected = [3, 1]
        logits = _TensorLike(rows)
        scores = _TensorLike([_packed_scores(rows[0], selected)])
        indices = _TensorLike([selected])
        references = (weakref.ref(logits), weakref.ref(scores), weakref.ref(indices))
        decoder(_target(inspection), object(), (), (logits, scores, indices))
        del logits, scores, indices
    gc.collect()
    assert all(reference() is None for reference in references)


def test_fresh_revalidation_rejects_subclasses_and_duplicate_or_mixed_tokens() -> None:
    inspection = _inspection("legacy")

    class TokenSubclass(TokenEvent):
        pass

    with pytest.raises(TypeError):
        MixtralRoutingDecoder(
            inspection,
            (TokenSubclass.model_validate(_tokens(1)[0].model_dump()),),
        )
    with pytest.raises(ValueError, match="unique"):
        MixtralRoutingDecoder(inspection, (_tokens(1)[0], _tokens(1)[0]))
    decode = TokenEvent(
        run_key="run-1",
        sequence_id="sequence-1",
        token_pos=1,
        token_id=11,
        token_text="1",
        phase=TokenPhase.DECODE,
    )
    with pytest.raises(ValueError, match="one run_key and phase"):
        MixtralRoutingDecoder(inspection, (_tokens(1)[0], decode))


def test_descriptor_and_layout_tampering_rejected() -> None:
    inspection = _inspection("legacy")
    for update in (
        {"version": "2.0"},
        {"name": "other-static"},
        {"architecture_families": ("other",)},
    ):
        descriptor = inspection.descriptor.model_copy(update=update)
        with pytest.raises(ValueError):
            MixtralRoutingDecoder(
                inspection.model_copy(update={"descriptor": descriptor}), _tokens(1)
            )
    with pytest.raises(ValueError):
        MixtralRoutingDecoder(_qwen_inspection(), _tokens(1))
    router = next(
        component for component in inspection.report.components if component.kind.value == "router"
    )
    capture = router.capture.model_copy(update={"metadata": {"layout": "packed"}})
    components = [
        component.model_copy(update={"capture": capture})
        if component.component_key == router.component_key
        else component
        for component in inspection.report.components
    ]
    with pytest.raises(ValueError, match="consistent layout"):
        MixtralRoutingDecoder(
            inspection.model_copy(
                update={"report": inspection.report.model_copy(update={"components": components})}
            ),
            _tokens(1),
        )


def test_empty_token_tuple_and_non_tuple_inputs_rejected() -> None:
    inspection = _inspection("legacy")
    with pytest.raises(ValueError, match="non-empty"):
        MixtralRoutingDecoder(inspection, ())
    with pytest.raises(TypeError, match="exact tuple"):
        MixtralRoutingDecoder(inspection, list(_tokens(1)))
    decoder = MixtralRoutingDecoder(inspection, _tokens(1))
    with pytest.raises(TypeError, match="exact tuple"):
        decoder(_target(inspection), object(), [], _TensorLike([[1.0, 2.0, 0.0, 3.0]]))


def test_mixed_runs_and_invalid_token_values_rejected() -> None:
    inspection = _inspection("legacy")
    first = _tokens(1)[0]
    mixed_run = TokenEvent(
        run_key="run-2",
        sequence_id="sequence-1",
        token_pos=1,
        token_id=11,
        token_text="1",
        phase=TokenPhase.PREFILL,
    )
    with pytest.raises(ValueError, match="run_key and phase"):
        MixtralRoutingDecoder(inspection, (first, mixed_run))
    with pytest.raises((TypeError, ValueError)):
        MixtralRoutingDecoder(inspection, (first, object()))


@pytest.mark.parametrize("bad", [[[1.0, 2.0, 3.0]], [[1.0, 2.0, 3.0, math.inf]]])
def test_legacy_shape_and_finite_value_rejection(bad: list[list[float]]) -> None:
    inspection = _inspection("legacy")
    decoder = MixtralRoutingDecoder(inspection, _tokens(1))
    with pytest.raises((TypeError, ValueError)):
        decoder(_target(inspection), object(), (), _TensorLike(bad))


@pytest.mark.parametrize("layout", ["legacy", "packed"])
@pytest.mark.parametrize("row", [[3.0, 3.0, 1.0, 0.0], [3.0, 1.0, 1.0, 0.0]])
def test_selected_and_cutoff_ties_are_rejected(layout: str, row: list[float]) -> None:
    inspection = _inspection(layout)
    decoder = MixtralRoutingDecoder(inspection, _tokens(1))
    with pytest.raises(ValueError, match="tie"):
        output = _TensorLike([row]) if layout == "legacy" else _packed_tie_payload(row)
        decoder(_target(inspection), object(), (), output)


def _packed_tie_payload(row: list[float]) -> tuple[_TensorLike, _TensorLike, _TensorLike]:
    selected = sorted(range(len(row)), key=lambda index: (-row[index], index))[:2]
    return (
        _TensorLike([row]),
        _TensorLike([_packed_scores(row, selected)]),
        _TensorLike([selected]),
    )


@pytest.mark.parametrize(
    "output",
    [
        ("not", "a", "tuple"),
        (_TensorLike([[1.0, 2.0, 0.0, 3.0]]), _TensorLike([[0.5, 0.5]]), _TensorLike([[3, 3]])),
        (_TensorLike([[1.0, 2.0, 0.0, 3.0]]), _TensorLike([[0.5, 0.5]]), _TensorLike([[4, 1]])),
    ],
)
def test_packed_shape_index_and_native_score_rejection(output: object) -> None:
    inspection = _inspection("packed")
    decoder = MixtralRoutingDecoder(inspection, _tokens(1))
    with pytest.raises((TypeError, ValueError)):
        decoder(_target(inspection), object(), (), output)


@pytest.mark.parametrize(
    "output",
    [
        object(),
        (
            _TensorLike([[1.0, 2.0, 0.0, 3.0]]),
            _TensorLike([[0.7310586, 0.2689414]]),
        ),
        (
            _TensorLike([[1.0, 2.0, 0.0, 3.0]]),
            _TensorLike([[0.7310586, 0.2689414]]),
            _TensorLike([[3, 1]]),
            _TensorLike([[0.0]]),
        ),
    ],
)
def test_packed_output_requires_exact_three_item_tuple(output: object) -> None:
    inspection = _inspection("packed")
    decoder = MixtralRoutingDecoder(inspection, _tokens(1))
    with pytest.raises((TypeError, ValueError)):
        decoder(_target(inspection), object(), (), output)


@pytest.mark.parametrize("field", ["logits", "scores", "indices"])
def test_packed_shape_rejects_wrong_row_count(field: str) -> None:
    inspection = _inspection("packed")
    decoder = MixtralRoutingDecoder(inspection, _tokens(2))
    logits = [[1.0, 2.0, 0.0, 3.0], [4.0, 0.0, 3.0, 1.0]]
    indices = [[3, 1], [0, 2]]
    scores = [
        _packed_scores(logits[0], indices[0]),
        _packed_scores(logits[1], indices[1]),
    ]
    payload = {
        "logits": logits,
        "scores": scores,
        "indices": indices,
    }
    payload[field] = payload[field][:1]
    with pytest.raises(ValueError, match="exact shape"):
        decoder(
            _target(inspection),
            object(),
            (),
            (
                _TensorLike(payload["logits"]),
                _TensorLike(payload["scores"]),
                _TensorLike(payload["indices"]),
            ),
        )


@pytest.mark.parametrize("bad_indices", [[[3, -1]], [[3, True]], [[3, "1"]]])
def test_packed_indices_require_nonnegative_strict_integers(bad_indices: object) -> None:
    inspection = _inspection("packed")
    decoder = MixtralRoutingDecoder(inspection, _tokens(1))
    with pytest.raises(ValueError, match="indices"):
        decoder(
            _target(inspection),
            object(),
            (),
            (
                _TensorLike([[1.0, 2.0, 0.0, 3.0]]),
                _TensorLike([_packed_scores([1.0, 2.0, 0.0, 3.0], [3, 1])]),
                _TensorLike(bad_indices),
            ),
        )


@pytest.mark.parametrize(
    ("logits", "scores", "indices"),
    [
        ([[1.0, 2.0, 3.0]], [[0.5, 0.5]], [[2, 1]]),
        ([[1.0, 2.0, 0.0, 3.0]], [[0.5]], [[3, 1]]),
        ([[1.0, 2.0, 0.0, 3.0]], [[0.5, 0.5]], [[3]]),
        ([[1.0, 2.0, math.inf, 3.0]], [[0.5, 0.5]], [[3, 1]]),
        ([[1.0, 2.0, 0.0, 3.0]], [[math.inf, 0.0]], [[3, 1]]),
        ([[1.0, 2.0, 0.0, 3.0]], [[-0.1, 1.1]], [[3, 1]]),
        ([[1.0, 2.0, 0.0, 3.0]], [[1.1, -0.1]], [[3, 1]]),
        ([[3.0, 2.0, 0.0, 1.0]], _packed_scores([3.0, 2.0, 0.0, 1.0], [3, 0]), [[3, 0]]),
        ([[1.0, 2.0, 0.0, 3.0]], [[0.7, 0.4]], [[3, 1]]),
    ],
)
def test_packed_explicit_shape_value_score_and_topk_rejection(
    logits: list[list[float]], scores: list[list[float]], indices: list[list[int]]
) -> None:
    inspection = _inspection("packed")
    decoder = MixtralRoutingDecoder(inspection, _tokens(1))
    with pytest.raises((TypeError, ValueError)):
        decoder(
            _target(inspection),
            object(),
            (),
            (_TensorLike(logits), _TensorLike(scores), _TensorLike(indices)),
        )


def test_packed_score_cross_check_and_sum_tolerance() -> None:
    inspection = _inspection("packed")
    decoder = MixtralRoutingDecoder(inspection, _tokens(1))
    scores = _packed_scores([1.0, 2.0, 0.0, 3.0], [3, 1])
    scores[0] += 2e-5
    scores[1] -= 2e-5
    with pytest.raises(ValueError, match="softmax"):
        decoder(
            _target(inspection),
            object(),
            (),
            (_TensorLike([[1.0, 2.0, 0.0, 3.0]]), _TensorLike([scores]), _TensorLike([[3, 1]])),
        )


def test_context_binding_and_single_successful_invocation() -> None:
    inspection = _inspection("legacy")
    decoder = MixtralRoutingDecoder(inspection, _tokens(1))
    context = _target(inspection)
    decoder(context, object(), (), _TensorLike([[1.0, 2.0, 0.0, 3.0]]))
    with pytest.raises(RuntimeError, match="single-use"):
        decoder(context, object(), (), _TensorLike([[1.0, 2.0, 0.0, 3.0]]))
    tampered = RoutingCaptureTarget(
        router=ProbeTarget(
            module_path=context.router.module_path + "/wrong",
            component_key=context.router.component_key,
            component_kind=ComponentKind.ROUTER,
        ),
        layer_key=context.layer_key,
        expert_keys=context.expert_keys,
        routed_top_k=context.routed_top_k,
    )
    fresh_decoder = MixtralRoutingDecoder(inspection, _tokens(1))
    with pytest.raises(ValueError, match="bound"):
        fresh_decoder(tampered, object(), (), _TensorLike([[1.0, 2.0, 0.0, 3.0]]))


@pytest.mark.parametrize("field", ["router_key", "layer_key", "expert_order", "top_k"])
def test_every_context_binding_field_is_checked(field: str) -> None:
    inspection = _inspection("legacy")
    context = _target(inspection)
    if field == "router_key":
        tampered_router = ProbeTarget(
            module_path=context.router.module_path,
            component_key="component:" + "0" * 64,
            component_kind=ComponentKind.ROUTER,
        )
        tampered = replace(context, router=tampered_router)
    elif field == "layer_key":
        tampered = replace(context, layer_key="component:" + "1" * 64)
    elif field == "expert_order":
        tampered = replace(context, expert_keys=tuple(reversed(context.expert_keys)))
    else:
        tampered = replace(context, routed_top_k=context.routed_top_k + 1)
    decoder = MixtralRoutingDecoder(inspection, _tokens(1))
    with pytest.raises(ValueError, match="context"):
        decoder(tampered, object(), (), _TensorLike([[1.0, 2.0, 0.0, 3.0]]))


@pytest.mark.parametrize("error", [KeyboardInterrupt("cancel"), SystemExit("exit")])
def test_tensor_control_flow_errors_are_preserved(error: BaseException) -> None:
    inspection = _inspection("legacy")
    decoder = MixtralRoutingDecoder(inspection, _tokens(1))
    with pytest.raises(type(error)) as caught:
        decoder(_target(inspection), object(), (), _TensorLike(None, fail=error))
    assert caught.value is error


@dataclass
class _Handle:
    owner: _HookModule
    callback: object

    def remove(self) -> None:
        self.owner.callbacks.remove(self.callback)


class _HookModule:
    def __init__(self, path: str) -> None:
        self.path = path
        self.callbacks: list[object] = []

    def register_forward_hook(self, callback: object) -> _Handle:
        self.callbacks.append(callback)
        return _Handle(self, callback)

    def fire(self, output: object) -> None:
        for callback in tuple(self.callbacks):
            callback(self, (), output)


class _HookModel:
    def __init__(self, layout: str) -> None:
        source = MixtralForCausalLM(layout=layout)
        self.config = source.config
        self.nodes: dict[str, _HookModule] = {}
        self._entries = []
        for path, module in source.named_modules():
            if path and path.endswith(".gate"):
                node = _HookModule(path)
                self.nodes[path] = node
                self._entries.append((path, node))
            else:
                self._entries.append((path, module))

    def named_modules(self):
        return iter(self._entries)


@pytest.mark.parametrize("layout", ["legacy", "packed"])
def test_routing_capture_session_integration(layout: str) -> None:
    inspection = _inspection(layout)
    model = _HookModel(layout)
    decoder = MixtralRoutingDecoder(inspection, _tokens(1))
    plan = build_routing_probe_plan(inspection)
    with RoutingCaptureSession(model, inspection, plan, decoder, max_events=32) as session:
        for path, node in model.nodes.items():
            rows = [[1.0, 2.0, 0.0, 3.0]]
            node.fire(_payload(layout, rows))
    assert len(session.events) == len(model.nodes) * 2
    assert session.truncated is False
    assert all(node.callbacks == [] for node in model.nodes.values())


def test_source_has_no_optional_tensor_imports() -> None:
    source = ast.parse(inspect.getsource(MixtralRoutingDecoder))
    module = ast.parse(Path("src/moeatlas/runtime/mixtral_routing.py").read_text())
    forbidden = {"torch", "transformers", "accelerate", "safetensors", "numpy", "np"}
    imported: set[str] = set()
    dynamic_imports: list[ast.Call] = []
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in {"__import__", "import_module"}:
                dynamic_imports.append(node)
            elif isinstance(node.func, ast.Attribute) and node.func.attr == "import_module":
                dynamic_imports.append(node)
    assert source is not None
    assert imported.isdisjoint(forbidden | {"importlib"})
    assert not dynamic_imports


def test_decoder_invocation_is_offline_and_does_not_touch_cache_or_tmp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inspection = _inspection("legacy")
    decoder = MixtralRoutingDecoder(inspection, _tokens(1))
    cache = tmp_path / "empty-cache"
    cache.mkdir()
    before = sorted(
        (
            path.relative_to(tmp_path).as_posix(),
            path.is_dir(),
            path.read_bytes() if path.is_file() else None,
        )
        for path in tmp_path.rglob("*")
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("network access is forbidden in model-free decoder tests")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
    monkeypatch.setenv("HF_HOME", str(cache))
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(cache))
    decoder(_target(inspection), object(), (), _TensorLike([[1.0, 2.0, 0.0, 3.0]]))

    after = sorted(
        (
            path.relative_to(tmp_path).as_posix(),
            path.is_dir(),
            path.read_bytes() if path.is_file() else None,
        )
        for path in tmp_path.rglob("*")
    )
    assert after == before
