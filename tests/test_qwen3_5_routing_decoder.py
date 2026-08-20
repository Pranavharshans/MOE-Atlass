from __future__ import annotations

import ast
import gc
import inspect
import math
import socket
import urllib.request
import weakref
from dataclasses import replace
from pathlib import Path

import pytest

from moeatlas.adapters import (
    AdapterInspection,
    Qwen3_5MoeStaticAdapter,
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
from moeatlas.events import RoutingEvent, TokenEvent, TokenPhase
from moeatlas.probe import ProbeTarget
from moeatlas.runtime import Qwen3_5RoutingDecoder, RoutingCaptureSession, RoutingCaptureTarget

from .fixtures.qwen3_5_moe import (
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeForConditionalGeneration,
    Qwen3_5MoeHookableForCausalLM,
    Qwen3_5MoeHookableForConditionalGeneration,
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


def _manifest(surface: str = "conditional") -> ModelManifest:
    return ModelManifest(
        model_key=make_model_key("acme/qwen3.5", "r1"),
        architecture="qwen3_5_moe",
        revision="r1",
        config_hash=make_config_hash({"family": "qwen3.5", "surface": surface}),
        tokenizer=TokenizerIdentity(identifier="acme/qwen3.5-tokenizer", revision="r1"),
        dtype=DType.FLOAT32,
        device_map={"": "cpu"},
    )


def _inspection(surface: str = "conditional") -> AdapterInspection:
    model_type = (
        Qwen3_5MoeForConditionalGeneration if surface == "conditional" else Qwen3_5MoeForCausalLM
    )
    model = model_type()
    return inspect_static_adapter(
        Qwen3_5MoeStaticAdapter(), model, model.config, _manifest(surface)
    )


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


def _target(inspection: AdapterInspection, *, router_index: int = 0) -> RoutingCaptureTarget:
    routers = sorted(
        (
            component
            for component in inspection.report.components
            if component.kind is ComponentKind.ROUTER
        ),
        key=lambda component: component.module_path,
    )
    router = routers[router_index]
    layer = next(
        component
        for component in inspection.report.components
        if component.kind is ComponentKind.MOE_LAYER and component.layer_index == router.layer_index
    )
    experts = sorted(
        (
            component
            for component in inspection.report.components
            if component.kind is ComponentKind.EXPERT
            and component.layer_index == router.layer_index
        ),
        key=lambda component: component.expert_index,
    )
    return RoutingCaptureTarget(
        router=ProbeTarget(
            module_path=router.module_path,
            component_key=router.component_key,
            component_kind=ComponentKind.ROUTER,
        ),
        layer_key=layer.component_key,
        expert_keys=tuple(component.component_key for component in experts),
        routed_top_k=inspection.report.facts.routed_top_k,
    )


def _packed_scores(row: list[float], selected: list[int]) -> list[float]:
    maximum = max(row)
    values = [math.exp(value - maximum) for value in row]
    probabilities = [value / sum(values) for value in values]
    total = sum(probabilities[index] for index in selected)
    return [probabilities[index] / total for index in selected]


def _packed_payload(
    rows: list[list[float]], *, top_k: int = 2
) -> tuple[_TensorLike, _TensorLike, _TensorLike]:
    selected_rows = [
        sorted(range(len(row)), key=lambda index: (-row[index], index))[:top_k] for row in rows
    ]
    score_rows = [_packed_scores(row, selected) for row, selected in zip(rows, selected_rows)]
    return _TensorLike(rows), _TensorLike(score_rows), _TensorLike(selected_rows)


def test_public_surface_is_independent_and_exact() -> None:
    assert Qwen3_5RoutingDecoder.__module__ == "moeatlas.runtime.qwen3_5_routing"
    assert tuple(inspect.signature(Qwen3_5RoutingDecoder).parameters) == (
        "inspection",
        "token_events",
    )
    assert tuple(inspect.signature(Qwen3_5RoutingDecoder.__call__).parameters) == (
        "self",
        "context",
        "module",
        "inputs",
        "output",
    )
    assert Qwen3_5RoutingDecoder.__slots__ == ("_bindings", "_token_events", "_used_paths")
    source = ast.parse(Path("src/moeatlas/runtime/qwen3_5_routing.py").read_text())
    forbidden = {
        "torch",
        "transformers",
        "accelerate",
        "safetensors",
        "numpy",
        "np",
        "os",
        "pathlib",
        "socket",
        "urllib",
        "http",
        "requests",
        "importlib",
        "tempfile",
        "shutil",
        "subprocess",
        "multiprocessing",
        "threading",
        "asyncio",
        "webbrowser",
        "cache",
        "model",
        "models",
        "store",
        "server",
        "ui",
    }
    imported: set[str] = set()
    dynamic_imports: list[ast.Call] = []
    for node in ast.walk(source):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in {
                "__import__",
                "eval",
                "exec",
                "compile",
                "open",
            }:
                dynamic_imports.append(node)
            elif isinstance(node.func, ast.Attribute) and node.func.attr in {
                "import_module",
                "urlopen",
                "read_text",
                "write_text",
            }:
                dynamic_imports.append(node)
    assert imported.isdisjoint(forbidden)
    assert not dynamic_imports
    assert not any("mixtral" in name.lower() for name in imported)
    forbidden_calls = {
        "generate",
        "train",
        "eval",
        "save",
        "load",
        "download",
        "fetch",
        "request",
        "socket",
        "create_connection",
        "urlopen",
        "open",
        "read_text",
        "write_text",
        "unlink",
        "remove",
        "rename",
        "replace",
        "mkdir",
        "rmdir",
        "system",
        "popen",
        "spawn",
        "start",
        "serve",
        "launch",
        "connect",
    }
    assert {
        node.func.id
        for node in ast.walk(source)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }.isdisjoint(forbidden_calls)
    forbidden_attributes = {
        "MixtralRoutingDecoder",
        "mixtral",
        "model",
        "tokenizer",
        "named_modules",
        "named_parameters",
        "forward",
        "generate",
        "train",
        "eval",
        "save",
        "load",
        "download",
        "fetch",
        "cache",
        "tmp",
        "temp",
        "process",
        "server",
        "ui",
        "store",
        "storage",
        "filesystem",
        "read_text",
        "write_text",
        "unlink",
        "replace",
        "import_module",
        "urlopen",
    }
    assert {node.attr for node in ast.walk(source) if isinstance(node, ast.Attribute)}.isdisjoint(
        forbidden_attributes
    )
    assert not any(
        isinstance(node, ast.Name) and node.id == "MixtralRoutingDecoder"
        for node in ast.walk(source)
    )


@pytest.mark.parametrize("surface", ["conditional", "text"])
def test_known_qwen35_values_emit_only_routed_model_neutral_events(surface: str) -> None:
    inspection = _inspection(surface)
    decoder = Qwen3_5RoutingDecoder(inspection, _tokens(2))
    output = _packed_payload([[1.0, 2.0, 0.0, 3.0], [4.0, 0.0, 3.0, 1.0]])
    events = decoder(_target(inspection), object(), (), output)
    assert type(events) is tuple
    assert len(events) == 4
    assert [event.rank for event in events] == [0, 1, 0, 1]
    assert [event.router_logit for event in events] == [3.0, 2.0, 4.0, 3.0]
    assert all(type(event) is RoutingEvent for event in events)
    assert all(event.selected is True for event in events)
    assert all(event.probability is None and event.weight is not None for event in events)
    assert all(event.expert_key in _target(inspection).expert_keys for event in events)
    assert all(
        component.kind is not ComponentKind.SHARED_EXPERT
        for component in inspection.report.components
        if component.component_key in {event.expert_key for event in events}
    )


def test_conversion_is_exact_and_payload_is_not_retained() -> None:
    inspection = _inspection()
    decoder = Qwen3_5RoutingDecoder(inspection, _tokens(1))
    logits, scores, indices = _packed_payload([[1.0, 2.0, 0.0, 3.0]])
    decoder(_target(inspection), object(), (), (logits, scores, indices))
    assert logits.calls == ["detach", "cpu", "float", "tolist"]
    assert scores.calls == ["detach", "cpu", "float", "tolist"]
    assert indices.calls == ["detach", "cpu", "tolist"]
    assert not any(name in Qwen3_5RoutingDecoder.__slots__ for name in ("_output", "_tensor"))

    logits, scores, indices = _packed_payload([[1.0, 2.0, 0.0, 3.0]])
    references = tuple(weakref.ref(value) for value in (logits, scores, indices))
    Qwen3_5RoutingDecoder(inspection, _tokens(1))(
        _target(inspection), object(), (), (logits, scores, indices)
    )
    del logits, scores, indices
    gc.collect()
    assert all(reference() is None for reference in references)


def test_constructor_revalidates_tokens_and_rejects_wrong_family_or_shape() -> None:
    inspection = _inspection()

    class TokenSubclass(TokenEvent):
        pass

    with pytest.raises(TypeError):
        Qwen3_5RoutingDecoder(
            inspection,
            (TokenSubclass.model_validate(_tokens(1)[0].model_dump()),),
        )
    with pytest.raises(ValueError, match="unique"):
        Qwen3_5RoutingDecoder(inspection, (_tokens(1)[0], _tokens(1)[0]))
    with pytest.raises(ValueError, match="non-empty"):
        Qwen3_5RoutingDecoder(inspection, ())
    with pytest.raises(TypeError, match="exact tuple"):
        Qwen3_5RoutingDecoder(inspection, list(_tokens(1)))

    payload = inspection.model_dump(mode="json")
    payload["descriptor"]["name"] = "huggingface-mixtral-static"
    with pytest.raises(ValueError):
        Qwen3_5RoutingDecoder(AdapterInspection.model_validate(payload), _tokens(1))


def test_constructor_requires_a_full_contiguous_canonical_token_sequence() -> None:
    inspection = _inspection()
    first, second = _tokens(2)
    with pytest.raises(ValueError, match="contiguous"):
        Qwen3_5RoutingDecoder(inspection, (second, first))
    missing_position = TokenEvent(
        run_key=second.run_key,
        sequence_id=second.sequence_id,
        token_pos=2,
        token_id=second.token_id,
        token_text=second.token_text,
        phase=second.phase,
    )
    with pytest.raises(ValueError, match="contiguous"):
        Qwen3_5RoutingDecoder(inspection, (first, missing_position))
    offset = tuple(
        TokenEvent(
            run_key=event.run_key,
            sequence_id=event.sequence_id,
            token_pos=event.token_pos + 1,
            token_id=event.token_id,
            token_text=event.token_text,
            phase=event.phase,
        )
        for event in _tokens(2)
    )
    with pytest.raises(ValueError, match="0..N-1"):
        Qwen3_5RoutingDecoder(inspection, offset)
    different_sequence = TokenEvent(
        run_key=second.run_key,
        sequence_id="sequence-2",
        token_pos=second.token_pos,
        token_id=second.token_id,
        token_text=second.token_text,
        phase=second.phase,
    )
    with pytest.raises(ValueError, match="sequence_id"):
        Qwen3_5RoutingDecoder(inspection, (first, different_sequence))


def test_constructor_does_not_retain_original_inspection_or_token_objects() -> None:
    inspection = _inspection()
    tokens = _tokens(1)
    inspection_reference = weakref.ref(inspection)
    token_reference = weakref.ref(tokens[0])
    decoder = Qwen3_5RoutingDecoder(inspection, tokens)
    del inspection, tokens
    gc.collect()
    assert inspection_reference() is None
    assert token_reference() is None
    assert len(decoder._token_events) == 1


@pytest.mark.parametrize(
    "tampered",
    [
        lambda inspection: inspection.model_copy(
            update={
                "descriptor": inspection.descriptor.model_copy(
                    update={"compatibility_notes": ("tampered",)}
                )
            }
        ),
        lambda inspection: inspection.model_copy(
            update={
                "detection": inspection.detection.model_copy(
                    update={"evidence": ("tampered", "evidence")}
                )
            }
        ),
        lambda inspection: inspection.model_copy(
            update={"report": inspection.report.model_copy(update={"scanner_version": "9.9.9"})}
        ),
        lambda inspection: inspection.model_copy(
            update={"report": inspection.report.model_copy(update={"warnings": []})}
        ),
        lambda inspection: inspection.model_copy(
            update={
                "report": inspection.report.model_copy(
                    update={
                        "model_manifest": inspection.report.model_manifest.model_copy(
                            update={"architecture": "foreign_moe"}
                        )
                    }
                )
            }
        ),
    ],
)
def test_constructor_rejects_tampered_detection_report_and_manifest_provenance(tampered) -> None:
    with pytest.raises(ValueError):
        Qwen3_5RoutingDecoder(tampered(_inspection()), _tokens(1))


def test_constructor_rejects_tampered_capture_and_component_provenance() -> None:
    inspection = _inspection()
    first = inspection.report.components[0]
    capture = first.capture.model_copy(update={"metadata": {"layout": "legacy"}})
    provenance = first.provenance.model_copy(update={"source": "foreign-source"})
    component = first.model_copy(update={"capture": capture, "provenance": provenance})
    components = [component, *inspection.report.components[1:]]
    bad = inspection.model_copy(
        update={"report": inspection.report.model_copy(update={"components": components})}
    )
    with pytest.raises(ValueError):
        Qwen3_5RoutingDecoder(bad, _tokens(1))


@pytest.mark.parametrize("field", ["layer", "router", "container", "expert", "shared"])
def test_constructor_rejects_noncanonical_component_root_relationships(
    field: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = _inspection()
    components = list(base.report.components)
    foreign_root = "foreign.layers.0"
    positions = {
        "layer": next(
            index
            for index, component in enumerate(components)
            if component.kind is ComponentKind.MOE_LAYER and component.layer_index == 0
        ),
        "router": next(
            index
            for index, component in enumerate(components)
            if component.kind is ComponentKind.ROUTER and component.layer_index == 0
        ),
        "container": next(
            index
            for index, component in enumerate(components)
            if component.kind is ComponentKind.EXPERT_CONTAINER and component.layer_index == 0
        ),
        "expert": next(
            index
            for index, component in enumerate(components)
            if component.kind is ComponentKind.EXPERT and component.layer_index == 0
        ),
        "shared": next(
            index
            for index, component in enumerate(components)
            if component.kind is ComponentKind.SHARED_EXPERT and component.layer_index == 0
        ),
    }
    paths = {
        "layer": f"{foreign_root}.mlp",
        "router": f"{foreign_root}.mlp.gate",
        "container": f"{foreign_root}.mlp.experts",
        "expert": f"{foreign_root}.mlp.experts",
        "shared": f"{foreign_root}.mlp.shared_expert",
    }
    components[positions[field]] = components[positions[field]].model_copy(
        update={"module_path": paths[field]}
    )
    tampered = base.model_copy(
        update={"report": base.report.model_copy(update={"components": components})}
    )

    def return_tampered(cls, payload: object) -> AdapterInspection:
        del cls, payload
        return tampered

    monkeypatch.setattr(AdapterInspection, "model_validate", classmethod(return_tampered))
    with pytest.raises(ValueError, match="canonical"):
        Qwen3_5RoutingDecoder(base, _tokens(1))


@pytest.mark.parametrize(
    "output",
    [
        object(),
        ("not", "a", "tuple", "extra"),
        (_TensorLike([[1.0, 2.0, 0.0, 3.0]]), _TensorLike([[0.5, 0.5]])),
    ],
)
def test_payload_requires_exact_native_qwen_tuple(output: object) -> None:
    inspection = _inspection()
    decoder = Qwen3_5RoutingDecoder(inspection, _tokens(1))
    with pytest.raises((TypeError, ValueError)):
        decoder(_target(inspection), object(), (), output)


@pytest.mark.parametrize(
    ("logits", "scores", "indices"),
    [
        ([[1.0, 2.0, 3.0]], [[0.5, 0.5]], [[2, 1]]),
        ([[1.0, 2.0, 0.0, 3.0]], [[0.5]], [[3, 1]]),
        ([[1.0, 2.0, 0.0, 3.0]], [[0.5, 0.5]], [[3]]),
        ([[1.0, 2.0, math.inf, 3.0]], [[0.5, 0.5]], [[3, 1]]),
        ([[1.0, 2.0, 0.0, 3.0]], [[math.inf, 0.0]], [[3, 1]]),
        ([[1.0, 2.0, 0.0, 3.0]], [[-0.1, 1.1]], [[3, 1]]),
        ([[1.0, 2.0, 0.0, 3.0]], [[0.7, 0.4]], [[3, 1]]),
        ([[1.0, 2.0, 0.0, 3.0]], _packed_scores([1.0, 2.0, 0.0, 3.0], [3, 1]), [[3, 0]]),
    ],
)
def test_shape_value_index_and_native_score_contract(
    logits: list[list[float]], scores: list[list[float]], indices: list[list[int]]
) -> None:
    inspection = _inspection()
    decoder = Qwen3_5RoutingDecoder(inspection, _tokens(1))
    with pytest.raises((TypeError, ValueError)):
        decoder(
            _target(inspection),
            object(),
            (),
            (_TensorLike(logits), _TensorLike(scores), _TensorLike(indices)),
        )


@pytest.mark.parametrize("row", [[3.0, 3.0, 1.0, 0.0], [3.0, 1.0, 1.0, 0.0]])
def test_selected_and_cutoff_ties_are_rejected(row: list[float]) -> None:
    inspection = _inspection()
    decoder = Qwen3_5RoutingDecoder(inspection, _tokens(1))
    selected = sorted(range(len(row)), key=lambda index: (-row[index], index))[:2]
    with pytest.raises(ValueError, match="tie"):
        decoder(
            _target(inspection),
            object(),
            (),
            (
                _TensorLike([row]),
                _TensorLike([_packed_scores(row, selected)]),
                _TensorLike([selected]),
            ),
        )


def test_native_scores_must_be_independently_non_increasing() -> None:
    inspection = _inspection()
    decoder = Qwen3_5RoutingDecoder(inspection, _tokens(1))
    logits = [3.0, 2.0, 1.0, 0.0]
    expected = _packed_scores(logits, [0, 1])
    with pytest.raises(ValueError, match="non-increasing"):
        decoder(
            _target(inspection),
            object(),
            (),
            (_TensorLike([logits]), _TensorLike([list(reversed(expected))]), _TensorLike([[0, 1]])),
        )


@pytest.mark.parametrize("field", ["logits", "scores", "indices"])
def test_exact_shape_is_required_for_each_tuple_member(field: str) -> None:
    inspection = _inspection()
    decoder = Qwen3_5RoutingDecoder(inspection, _tokens(2))
    rows = [[1.0, 2.0, 0.0, 3.0], [4.0, 0.0, 3.0, 1.0]]
    selected = [[3, 1], [0, 2]]
    payload: dict[str, object] = {
        "logits": rows,
        "scores": [_packed_scores(row, indexes) for row, indexes in zip(rows, selected)],
        "indices": selected,
    }
    payload[field] = payload[field][:1]  # type: ignore[index]
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


@pytest.mark.parametrize("error", [KeyboardInterrupt("cancel"), SystemExit("exit")])
def test_tensor_control_flow_errors_are_preserved(error: BaseException) -> None:
    inspection = _inspection()
    decoder = Qwen3_5RoutingDecoder(inspection, _tokens(1))
    with pytest.raises(type(error)) as caught:
        decoder(
            _target(inspection),
            object(),
            (),
            (_TensorLike(None, fail=error), _TensorLike(None), _TensorLike(None)),
        )
    assert caught.value is error


def test_context_binding_inputs_and_successful_single_use_are_exact() -> None:
    inspection = _inspection()
    context = _target(inspection)
    decoder = Qwen3_5RoutingDecoder(inspection, _tokens(1))
    payload = _packed_payload([[1.0, 2.0, 0.0, 3.0]])
    with pytest.raises(TypeError, match="exact tuple"):
        decoder(context, object(), [], payload)
    decoder(context, object(), (), payload)
    with pytest.raises(RuntimeError, match="single-use"):
        decoder(context, object(), (), _packed_payload([[1.0, 2.0, 0.0, 3.0]]))

    tampered = replace(context, layer_key="component:" + "0" * 64)
    with pytest.raises(ValueError, match="context"):
        Qwen3_5RoutingDecoder(inspection, _tokens(1))(
            tampered, object(), (), _packed_payload([[1.0, 2.0, 0.0, 3.0]])
        )


@pytest.mark.parametrize("field", ["router", "layer_key", "expert_keys", "routed_top_k"])
def test_every_context_identity_field_is_checked(field: str) -> None:
    inspection = _inspection()
    context = _target(inspection)
    updates: dict[str, object]
    if field == "router":
        updates = {
            "router": ProbeTarget(
                module_path=context.router.module_path,
                component_key="component:" + "0" * 64,
                component_kind=ComponentKind.ROUTER,
            )
        }
    elif field == "layer_key":
        updates = {"layer_key": "component:" + "0" * 64}
    elif field == "expert_keys":
        updates = {"expert_keys": tuple(reversed(context.expert_keys))}
    else:
        updates = {"routed_top_k": context.routed_top_k + 1}
    with pytest.raises(ValueError, match="context"):
        Qwen3_5RoutingDecoder(inspection, _tokens(1))(
            replace(context, **updates),
            object(),
            (),
            _packed_payload([[1.0, 2.0, 0.0, 3.0]]),
        )


@pytest.mark.parametrize(
    "factory", [Qwen3_5MoeHookableForConditionalGeneration, Qwen3_5MoeHookableForCausalLM]
)
def test_routing_capture_session_uses_qwen_owned_conditional_and_text_hooks(factory) -> None:
    surface = "conditional" if "Conditional" in factory.__name__ else "text"
    source = factory()
    inspection = _inspection(surface)
    decoder = Qwen3_5RoutingDecoder(inspection, _tokens(1))
    plan = build_routing_probe_plan(inspection)
    with RoutingCaptureSession(source, inspection, plan, decoder, max_events=64) as session:
        for node in source.nodes.values():
            node.fire(_packed_payload([[1.0, 2.0, 0.0, 3.0]]))
    assert len(session.events) == len(source.nodes) * 2
    assert session.truncated is False
    assert all(node.callbacks == [] for node in source.nodes.values())


def test_failed_payload_does_not_consume_router_and_can_retry() -> None:
    inspection = _inspection()
    decoder = Qwen3_5RoutingDecoder(inspection, _tokens(1))
    context = _target(inspection)
    with pytest.raises(ValueError):
        decoder(
            context,
            object(),
            (),
            (
                _TensorLike([[3.0, 3.0, 1.0, 0.0]]),
                _TensorLike([[0.5, 0.5]]),
                _TensorLike([[0, 1]]),
            ),
        )
    events = decoder(context, object(), (), _packed_payload([[1.0, 2.0, 0.0, 3.0]]))
    assert len(events) == 2


def test_decoder_invocation_is_offline_and_does_not_touch_cache_or_tmp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inspection = _inspection()
    decoder = Qwen3_5RoutingDecoder(inspection, _tokens(1))
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
    decoder(_target(inspection), object(), (), _packed_payload([[1.0, 2.0, 0.0, 3.0]]))
    after = sorted(
        (
            path.relative_to(tmp_path).as_posix(),
            path.is_dir(),
            path.read_bytes() if path.is_file() else None,
        )
        for path in tmp_path.rglob("*")
    )
    assert after == before
