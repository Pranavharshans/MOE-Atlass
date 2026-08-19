from __future__ import annotations

import ast
import gc
import inspect
import weakref
from collections import UserDict
from pathlib import Path
from types import SimpleNamespace
from typing import get_type_hints

import pytest

import moeatlas.runtime.prompt_prefill as prefill
from moeatlas.adapters import build_routing_probe_plan
from moeatlas.core import ModelManifest, make_token_key
from moeatlas.events import TokenEvent, TokenPhase
from moeatlas.loading import InstanceSource, LoadingPlan, TokenizerRequest
from moeatlas.runtime import LoadedModel, MixtralPromptPrefillError, run_mixtral_prompt_prefill

from .test_mixtral_routing_decoder import _inspection, _payload
from .test_runtime_routing_forward import _ForwardModel


def test_public_surface_and_error_contract() -> None:
    assert MixtralPromptPrefillError("tokenize").stage == "tokenize"
    assert str(MixtralPromptPrefillError("encoding")) == "Mixtral prompt prefill failed at encoding"
    with pytest.raises(ValueError):
        MixtralPromptPrefillError("bad")  # type: ignore[arg-type]
    signature = inspect.signature(run_mixtral_prompt_prefill)
    assert tuple(signature.parameters) == (
        "loaded",
        "inspection",
        "plan",
        "prompt",
        "run_key",
        "sequence_id",
        "add_special_tokens",
        "max_prompt_chars",
        "max_tokens",
        "max_events",
    )
    assert all(
        signature.parameters[name].default is inspect.Parameter.empty
        for name in signature.parameters
    )


def test_encoding_materialization_order_and_budget_before_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Tensor:
        shape = (1, 2)

        def __init__(self, values: list[list[int]], calls: list[str]) -> None:
            self.values, self.calls = values, calls

        def detach(self):
            self.calls.append("detach")
            return self

        def cpu(self):
            self.calls.append("cpu")
            return self

        def tolist(self):
            self.calls.append("tolist")
            return self.values

    calls: list[str] = []
    ids = Tensor([[4, 5]], calls)
    mask = Tensor([[1, 1]], calls)

    def tokenizer(*_args, **_kwargs):
        return {"input_ids": ids, "attention_mask": mask}

    tokenizer.convert_ids_to_tokens = lambda values: [f"t{x}" for x in values]  # type: ignore[attr-defined]
    events, copied = prefill._encode(
        tokenizer,
        "hi",
        add_special_tokens=False,
        max_tokens=2,
        max_events=4,
        target_count=1,
        routed_top_k=2,
        converter=tokenizer.convert_ids_to_tokens,  # type: ignore[attr-defined]
        run_key="run-1",
        sequence_id="seq-1",
    )
    assert len(events) == 2 and copied["input_ids"] is ids
    assert calls == ["detach", "cpu", "tolist", "detach", "cpu", "tolist"]

    calls.clear()
    with pytest.raises(ValueError, match="max_events"):
        prefill._encode(
            tokenizer,
            "hi",
            add_special_tokens=False,
            max_tokens=2,
            max_events=3,
            target_count=1,
            routed_top_k=2,
            converter=tokenizer.convert_ids_to_tokens,  # type: ignore[attr-defined]
            run_key="run-1",
            sequence_id="seq-1",
        )
    assert calls == []


def test_delegate_propagates_exact_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = RuntimeError("delegate")
    fake_plan = SimpleNamespace(targets=(object(),))
    fake_inspection = SimpleNamespace(report=SimpleNamespace(facts=SimpleNamespace(routed_top_k=1)))
    monkeypatch.setattr(
        prefill,
        "_preflight",
        lambda *a, **k: (
            SimpleNamespace(model=object()),
            object(),
            fake_inspection,
            fake_plan,
            object(),
            1,
            1,
            1,
            1,
        ),
    )
    monkeypatch.setattr(prefill, "_encode", lambda *a, **k: ((), {}))
    monkeypatch.setattr(prefill, "_resolve_converter", lambda _tokenizer: object())

    def fail_delegate(*_args, **_kwargs):
        raise sentinel

    monkeypatch.setattr(prefill, "run_mixtral_routing_forward", fail_delegate)
    with pytest.raises(RuntimeError) as caught:
        run_mixtral_prompt_prefill(
            object(),
            object(),
            object(),
            "x",
            run_key="r",
            sequence_id="s",
            add_special_tokens=False,
            max_prompt_chars=1,
            max_tokens=1,
            max_events=1,
        )  # type: ignore[arg-type]
    assert caught.value is sentinel


class _Tensor:
    def __init__(self, values: list[list[int]]) -> None:
        self.values = values
        self.shape = (1, len(values[0]))
        self.calls: list[str] = []

    def detach(self):
        self.calls.append("detach")
        return self

    def cpu(self):
        self.calls.append("cpu")
        return self

    def tolist(self):
        self.calls.append("tolist")
        return self.values


class _Tokenizer:
    def __init__(self, n: int = 2) -> None:
        self.ids = _Tensor([list(range(10, 10 + n))])
        self.mask = _Tensor([[1] * n])
        self.calls: list[tuple[object, dict[str, object]]] = []
        self.convert_calls: list[list[int]] = []
        self.returned_mapping: object | None = None

    def __call__(self, prompt: str, **kwargs: object):
        self.calls.append((prompt, kwargs))
        self.returned_mapping = {"input_ids": self.ids, "attention_mask": self.mask}
        return self.returned_mapping

    def convert_ids_to_tokens(self, values: list[int]) -> list[str]:
        self.convert_calls.append(values)
        return [f"piece-{value}" for value in values]


def _loaded_fixture(layout: str, tokenizer: _Tokenizer, model: object | None = None):
    inspection = _inspection(layout)
    source = InstanceSource(
        model_id="acme/mixtral",
        requested_revision="r1",
        tokenizer=TokenizerRequest(identifier="acme/tok", requested_revision="r1"),
    )
    plan = LoadingPlan(source=source)
    return (
        LoadedModel(
            model=model or _ForwardModel(layout),
            tokenizer=tokenizer,
            plan=plan,
            manifest=inspection.report.model_manifest,
            warnings=(),
        ),
        inspection,
        build_routing_probe_plan(inspection),
    )


@pytest.mark.parametrize("layout", ["legacy", "packed"])
def test_real_feature18_legacy_and_packed_integration(layout: str) -> None:
    tokenizer = _Tokenizer(2)
    model = _ForwardModel(layout, rows=[[1.0, 2.0, 0.0, 3.0]] * 2)
    loaded, inspection, plan = _loaded_fixture(layout, tokenizer, model)
    expected = 2 * len(plan.targets) * inspection.report.facts.routed_top_k
    result = run_mixtral_prompt_prefill(
        loaded,
        inspection,
        plan,
        "hello",
        run_key="run-1",
        sequence_id="sequence-1",
        add_special_tokens=False,
        max_prompt_chars=5,
        max_tokens=2,
        max_events=expected,
    )
    assert result.output is model.output
    assert model.calls == 1
    assert len(result.token_events) == 2
    assert tokenizer.calls[0][0] == "hello"
    assert tokenizer.calls[0][1] == {
        "add_special_tokens": False,
        "padding": False,
        "truncation": False,
        "return_attention_mask": True,
        "return_token_type_ids": False,
        "return_tensors": "pt",
    }
    assert tokenizer.convert_calls == [[10, 11]]
    assert not any(node.callbacks for node in model.nodes.values())


def test_exact_preflight_boundaries_do_not_call_tokenizer_or_model() -> None:
    tokenizer = _Tokenizer()
    loaded, inspection, plan = _loaded_fixture("legacy", tokenizer)
    cases = [
        ("", {}),
        ("hello", {"max_prompt_chars": 0}),
        ("hello", {"max_tokens": 0}),
        ("hello", {"max_events": 0}),
        ("hello", {"add_special_tokens": 1}),
        ("hello", {"run_key": "bad id"}),
    ]
    for prompt, changes in cases:
        kwargs = dict(
            run_key="run-1",
            sequence_id="sequence-1",
            add_special_tokens=False,
            max_prompt_chars=5,
            max_tokens=2,
            max_events=32,
        )
        kwargs.update(changes)
        with pytest.raises((TypeError, ValueError)):
            run_mixtral_prompt_prefill(loaded, inspection, plan, prompt, **kwargs)
    assert tokenizer.calls == []
    assert loaded.model.calls == 0


def test_converter_lookup_is_encoding_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = _Tokenizer()
    loaded, inspection, plan = _loaded_fixture("legacy", tokenizer)

    class Broken(_Tokenizer):
        @property
        def convert_ids_to_tokens(self):
            raise OSError("lookup")

    broken = Broken()
    loaded.tokenizer = broken
    with pytest.raises(MixtralPromptPrefillError) as caught:
        run_mixtral_prompt_prefill(
            loaded,
            inspection,
            plan,
            "x",
            run_key="run-1",
            sequence_id="sequence-1",
            add_special_tokens=False,
            max_prompt_chars=1,
            max_tokens=2,
            max_events=32,
        )
    assert caught.value.stage == "encoding"
    assert isinstance(caught.value.__cause__, OSError)


def test_preflight_matrix_stops_all_side_effects() -> None:
    tokenizer = _Tokenizer()
    loaded, inspection, plan = _loaded_fixture("legacy", tokenizer)
    invalid = [
        ("prompt", {"max_prompt_chars": 0}),
        ("prompt", {"max_tokens": False}),
        ("prompt", {"max_events": -1}),
        ("prompt", {"add_special_tokens": 1}),
        ("", {}),
        ("prompt", {"run_key": "bad id"}),
    ]
    for prompt, changes in invalid:
        args = dict(
            run_key="run-1",
            sequence_id="sequence-1",
            add_special_tokens=False,
            max_prompt_chars=10,
            max_tokens=2,
            max_events=32,
        )
        args.update(changes)
        with pytest.raises((TypeError, ValueError)):
            run_mixtral_prompt_prefill(loaded, inspection, plan, prompt, **args)
    assert tokenizer.calls == [] and loaded.model.calls == 0


@pytest.mark.parametrize("shape", [(2, 1), (1, 0), (1, 3, 1)])
def test_encoding_shape_and_payload_matrix(shape: tuple[int, ...]) -> None:
    tokenizer = _Tokenizer()
    loaded, inspection, plan = _loaded_fixture("legacy", tokenizer)

    class Bad(_Tokenizer):
        def __init__(self):
            super().__init__(2)
            self.ids.shape = shape

    bad = Bad()
    loaded.tokenizer = bad
    with pytest.raises(MixtralPromptPrefillError) as caught:
        run_mixtral_prompt_prefill(
            loaded,
            inspection,
            plan,
            "x",
            run_key="run-1",
            sequence_id="sequence-1",
            add_special_tokens=False,
            max_prompt_chars=1,
            max_tokens=3,
            max_events=32,
        )
    assert caught.value.stage == "encoding"
    assert loaded.model.calls == 0


def test_converter_failure_and_control_flow_matrix() -> None:
    tokenizer = _Tokenizer()
    loaded, inspection, plan = _loaded_fixture("legacy", tokenizer)

    class Broken(_Tokenizer):
        def convert_ids_to_tokens(self, _values):
            raise KeyboardInterrupt()

    loaded.tokenizer = Broken()
    with pytest.raises(KeyboardInterrupt):
        run_mixtral_prompt_prefill(
            loaded,
            inspection,
            plan,
            "x",
            run_key="run-1",
            sequence_id="sequence-1",
            add_special_tokens=False,
            max_prompt_chars=1,
            max_tokens=3,
            max_events=32,
        )
    assert loaded.model.calls == 0


def test_feature18_failure_and_cleanup_matrix() -> None:
    class Failing(_ForwardModel):
        def __call__(self, **kwargs):
            self.calls += 1
            raise ValueError("body")

    tokenizer = _Tokenizer()
    model = Failing("legacy")
    loaded, inspection, plan = _loaded_fixture("legacy", tokenizer, model)
    with pytest.raises(ValueError, match="body"):
        run_mixtral_prompt_prefill(
            loaded,
            inspection,
            plan,
            "x",
            run_key="run-1",
            sequence_id="sequence-1",
            add_special_tokens=False,
            max_prompt_chars=1,
            max_tokens=2,
            max_events=32,
        )
    assert model.calls == 1 and not any(node.callbacks for node in model.nodes.values())


def test_prefill_does_not_retain_intermediates() -> None:
    tokenizer = _Tokenizer()
    model = _ForwardModel("legacy", rows=[[1.0, 2.0, 0.0, 3.0]] * 2)
    loaded, inspection, plan = _loaded_fixture("legacy", tokenizer, model)
    result = run_mixtral_prompt_prefill(
        loaded,
        inspection,
        plan,
        "x",
        run_key="run-1",
        sequence_id="sequence-1",
        add_special_tokens=False,
        max_prompt_chars=1,
        max_tokens=2,
        max_events=32,
    )
    token_ref = weakref.ref(tokenizer)
    del loaded, tokenizer
    gc.collect()
    assert token_ref() is None
    assert result.output is model.output


def test_prompt_prefill_source_hard_kills() -> None:
    tree = ast.parse(prefill.__file__ and open(prefill.__file__, encoding="utf-8").read())
    source = open(prefill.__file__, encoding="utf-8").read()
    forbidden = {
        "torch",
        "transformers",
        "accelerate",
        "safetensors",
        "subprocess",
        "socket",
        "urllib",
        "generate",
        "eval",
        "train",
    }
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert not names & forbidden
    assert "run_mixtral_routing_forward" in source
    assert 'return_tensors="pt"' in source


# Feature 24 evidence matrix.  The wrapper is deliberately a narrow
# composition boundary, so these tests are explicit about what happens before
# a caller-owned tokenizer or model can be touched.


def _prefill_kwargs(
    *,
    prompt: str = "hello",
    max_prompt_chars: int = 5,
    max_tokens: int = 2,
    max_events: int = 32,
    add_special_tokens: bool = False,
    run_key: str = "run-1",
    sequence_id: str = "sequence-1",
) -> dict[str, object]:
    return {
        "run_key": run_key,
        "sequence_id": sequence_id,
        "add_special_tokens": add_special_tokens,
        "max_prompt_chars": max_prompt_chars,
        "max_tokens": max_tokens,
        "max_events": max_events,
        "prompt": prompt,
    }


def _call_prefill(loaded, inspection, plan, **overrides: object):
    kwargs = _prefill_kwargs()
    kwargs.update(overrides)
    prompt = kwargs.pop("prompt")
    return run_mixtral_prompt_prefill(loaded, inspection, plan, prompt, **kwargs)


@pytest.mark.parametrize(
    "bad_loaded",
    [object(), None, SimpleNamespace(closed=False, model=object(), tokenizer=object())],
)
def test_preflight_rejects_every_wrong_loaded_shape_without_side_effects(
    bad_loaded: object,
) -> None:
    tokenizer = _Tokenizer()
    valid, inspection, plan = _loaded_fixture("legacy", tokenizer)
    with pytest.raises((TypeError, ValueError)):
        _call_prefill(bad_loaded, inspection, plan)  # type: ignore[arg-type]
    assert tokenizer.calls == []
    assert valid.model.calls == 0


def test_preflight_rejects_loaded_subclass_before_borrowing_runtime() -> None:
    class LoadedSubclass(LoadedModel):
        pass

    tokenizer = _Tokenizer()
    loaded, inspection, plan = _loaded_fixture("legacy", tokenizer)
    subclass = LoadedSubclass(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        plan=loaded.plan,
        manifest=loaded.manifest,
        warnings=loaded.warnings,
    )
    with pytest.raises(TypeError, match="exact LoadedModel"):
        _call_prefill(subclass, inspection, plan)
    assert tokenizer.calls == []
    assert loaded.model.calls == 0


@pytest.mark.parametrize(
    "field,value",
    [("model", None), ("model", object()), ("tokenizer", None), ("tokenizer", object())],
)
def test_preflight_rejects_missing_or_noncallable_runtime_members(
    field: str, value: object
) -> None:
    tokenizer = _Tokenizer()
    loaded, inspection, plan = _loaded_fixture("legacy", tokenizer)
    setattr(loaded, field, value)
    with pytest.raises(ValueError, match=field):
        _call_prefill(loaded, inspection, plan)
    assert tokenizer.calls == []
    assert getattr(loaded.model, "calls", 0) == 0


def test_preflight_rejects_closed_loaded_before_manifest_or_tokenizer() -> None:
    tokenizer = _Tokenizer()
    loaded, inspection, plan = _loaded_fixture("legacy", tokenizer)
    loaded._closed = True
    with pytest.raises(ValueError, match="open"):
        _call_prefill(loaded, inspection, plan)
    assert tokenizer.calls == []
    assert loaded.model.calls == 0


def test_preflight_rejects_wrong_manifest_types_without_execution() -> None:
    for value in (object(),):
        tokenizer = _Tokenizer()
        loaded, inspection, plan = _loaded_fixture("legacy", tokenizer)
        loaded.manifest = value  # type: ignore[assignment]
        with pytest.raises(TypeError, match="exact ModelManifest"):
            _call_prefill(loaded, inspection, plan)
        assert tokenizer.calls == []
        assert loaded.model.calls == 0

    class ManifestSubclass(ModelManifest):
        pass

    tokenizer = _Tokenizer()
    loaded, inspection, plan = _loaded_fixture("legacy", tokenizer)
    loaded.manifest = ManifestSubclass.model_validate(loaded.manifest.model_dump(mode="json"))
    with pytest.raises(TypeError, match="exact ModelManifest"):
        _call_prefill(loaded, inspection, plan)
    assert tokenizer.calls == []
    assert loaded.model.calls == 0


def test_preflight_rejects_tampered_manifest_and_does_not_call_runtime() -> None:
    tokenizer = _Tokenizer()
    loaded, inspection, plan = _loaded_fixture("legacy", tokenizer)
    loaded.manifest = loaded.manifest.model_copy(update={"revision": "tampered"})
    with pytest.raises((TypeError, ValueError)):
        _call_prefill(loaded, inspection, plan)
    assert tokenizer.calls == []
    assert loaded.model.calls == 0


@pytest.mark.parametrize("bad_inspection", [object(), SimpleNamespace(report=object())])
def test_preflight_rejects_wrong_inspection_types_without_execution(
    bad_inspection: object,
) -> None:
    tokenizer = _Tokenizer()
    loaded, inspection, plan = _loaded_fixture("legacy", tokenizer)
    with pytest.raises((TypeError, ValueError)):
        _call_prefill(loaded, bad_inspection, plan)
    assert tokenizer.calls == []
    assert loaded.model.calls == 0
    assert inspection is not bad_inspection


@pytest.mark.parametrize(
    "descriptor_update",
    [
        {"name": "different-adapter"},
        {"version": "9.9"},
        {"architecture_families": ("qwen3_moe",)},
    ],
)
def test_preflight_rejects_descriptor_identity_tampering(
    descriptor_update: dict[str, object],
) -> None:
    tokenizer = _Tokenizer()
    loaded, inspection, plan = _loaded_fixture("legacy", tokenizer)
    descriptor = inspection.descriptor.model_copy(update=descriptor_update)
    bad_inspection = inspection.model_copy(update={"descriptor": descriptor})
    with pytest.raises((TypeError, ValueError)):
        _call_prefill(loaded, bad_inspection, plan)
    assert tokenizer.calls == []
    assert loaded.model.calls == 0


def test_preflight_rejects_manifest_binding_and_plan_binding_tampering() -> None:
    tokenizer = _Tokenizer()
    loaded, inspection, plan = _loaded_fixture("legacy", tokenizer)
    report = inspection.report.model_copy(update={"model_key": "model:other@r1"})
    with pytest.raises((TypeError, ValueError)):
        _call_prefill(loaded, inspection.model_copy(update={"report": report}), plan)
    assert tokenizer.calls == []

    loaded, inspection, plan = _loaded_fixture("legacy", _Tokenizer())
    target = plan.targets[0].model_copy(update={"module_path": "layers.99.block_sparse_moe.gate"})
    bad_plan = plan.model_copy(update={"targets": (target, *plan.targets[1:])})
    with pytest.raises((TypeError, ValueError)):
        _call_prefill(loaded, inspection, bad_plan)
    assert loaded.model.calls == 0


@pytest.mark.parametrize("prompt", [None, b"hello", 1, ["hello"], ""])
def test_preflight_prompt_type_and_empty_matrix(prompt: object) -> None:
    tokenizer = _Tokenizer()
    loaded, inspection, plan = _loaded_fixture("legacy", tokenizer)
    with pytest.raises((TypeError, ValueError)):
        _call_prefill(loaded, inspection, plan, prompt=prompt)  # type: ignore[arg-type]
    assert tokenizer.calls == []
    assert loaded.model.calls == 0


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("run_key", ""),
        ("run_key", "bad id"),
        ("run_key", "/absolute/path"),
        ("run_key", "https://example.test/run"),
        ("run_key", "a/../b"),
        ("sequence_id", ""),
        ("sequence_id", "bad id"),
        ("sequence_id", "~/sequence"),
    ],
)
def test_preflight_identifier_matrix_is_side_effect_free(field: str, bad_value: object) -> None:
    tokenizer = _Tokenizer()
    loaded, inspection, plan = _loaded_fixture("legacy", tokenizer)
    with pytest.raises((TypeError, ValueError)):
        _call_prefill(loaded, inspection, plan, **{field: bad_value})
    assert tokenizer.calls == []
    assert loaded.model.calls == 0


@pytest.mark.parametrize("bad", [None, 0, -1, 1.0, "2", (), {}, True, False])
def test_preflight_all_budget_types_are_strict_and_side_effect_free(bad: object) -> None:
    for field in ("max_prompt_chars", "max_tokens", "max_events"):
        tokenizer = _Tokenizer()
        loaded, inspection, plan = _loaded_fixture("legacy", tokenizer)
        with pytest.raises((TypeError, ValueError)):
            _call_prefill(loaded, inspection, plan, **{field: bad})
        assert tokenizer.calls == [], field
        assert loaded.model.calls == 0, field


@pytest.mark.parametrize("flag", [1, 0, "false", None, [], object()])
def test_preflight_add_special_tokens_requires_exact_bool(flag: object) -> None:
    tokenizer = _Tokenizer()
    loaded, inspection, plan = _loaded_fixture("legacy", tokenizer)
    with pytest.raises(TypeError, match="add_special_tokens"):
        _call_prefill(loaded, inspection, plan, add_special_tokens=flag)  # type: ignore[arg-type]
    assert tokenizer.calls == []
    assert loaded.model.calls == 0


class _PromptError(Exception):
    """Caller-owned ordinary error used to prove stage wrapping."""


def test_user_prompt_error_from_tokenizer_is_rewrapped_with_exact_cause() -> None:
    class FailingTokenizer(_Tokenizer):
        def __call__(self, prompt: str, **kwargs: object):
            self.calls.append((prompt, kwargs))
            raise _PromptError("caller tokenizer failure")

    tokenizer = FailingTokenizer()
    loaded, inspection, plan = _loaded_fixture("legacy", tokenizer)
    with pytest.raises(MixtralPromptPrefillError) as caught:
        _call_prefill(loaded, inspection, plan)
    assert caught.value.stage == "tokenize"
    assert type(caught.value.__cause__) is _PromptError
    assert str(caught.value.__cause__) == "caller tokenizer failure"
    assert loaded.model.calls == 0


@pytest.mark.parametrize("control", [KeyboardInterrupt("cancel"), SystemExit("exit")])
def test_tokenizer_control_flow_is_never_rewrapped(control: BaseException) -> None:
    class FailingTokenizer(_Tokenizer):
        def __call__(self, prompt: str, **kwargs: object):
            self.calls.append((prompt, kwargs))
            raise control

    tokenizer = FailingTokenizer()
    loaded, inspection, plan = _loaded_fixture("legacy", tokenizer)
    with pytest.raises(type(control)) as caught:
        _call_prefill(loaded, inspection, plan)
    assert caught.value is control
    assert loaded.model.calls == 0


def test_converter_lookup_precedes_tokenizer_call_and_is_encoding_stage() -> None:
    order: list[str] = []

    class OrderedTokenizer(_Tokenizer):
        @property
        def convert_ids_to_tokens(self):
            order.append("lookup")

            def convert(values: list[int]) -> list[str]:
                order.append("convert")
                return [f"piece-{value}" for value in values]

            return convert

        def __call__(self, prompt: str, **kwargs: object):
            order.append("tokenize")
            return super().__call__(prompt, **kwargs)

    tokenizer = OrderedTokenizer(1)
    loaded, inspection, plan = _loaded_fixture("legacy", tokenizer)
    _call_prefill(loaded, inspection, plan, max_tokens=1, max_events=4)
    assert order == ["lookup", "tokenize", "convert"]


@pytest.mark.parametrize(
    "output",
    [
        (),
        [],
        {"input_ids": _Tensor([[1, 2]])},
        {
            "input_ids": _Tensor([[1, 2]]),
            "attention_mask": _Tensor([[1, 1]]),
            "extra": 1,
        },
        {1: _Tensor([[1, 2]]), "attention_mask": _Tensor([[1, 1]])},
    ],
)
def test_encoding_requires_mapping_with_exact_two_keys(output: object) -> None:
    with pytest.raises(MixtralPromptPrefillError) as caught:
        prefill._encode(
            lambda *_args, **_kwargs: output,
            "hi",
            add_special_tokens=False,
            max_tokens=2,
            max_events=4,
            target_count=1,
            routed_top_k=2,
            converter=lambda values: ["a", "b"],
            run_key="run-1",
            sequence_id="sequence-1",
        )
    assert caught.value.stage == "encoding"


def test_encoding_accepts_arbitrary_mapping_implementation_and_copies_only_keys() -> None:
    ids = _Tensor([[3, 4]])
    mask = _Tensor([[1, 1]])
    original = UserDict({"input_ids": ids, "attention_mask": mask})
    events, copied = prefill._encode(
        lambda *_args, **_kwargs: original,
        "hi",
        add_special_tokens=True,
        max_tokens=2,
        max_events=4,
        target_count=1,
        routed_top_k=2,
        converter=lambda values: ["a", "b"],
        run_key="run-1",
        sequence_id="sequence-1",
    )
    assert len(events) == 2
    assert type(copied) is dict
    assert tuple(copied) == ("input_ids", "attention_mask")
    assert copied["input_ids"] is ids and copied["attention_mask"] is mask
    assert copied is not original


class _LooseTensor:
    def __init__(self, shape: object, nested: object, calls: list[str] | None = None) -> None:
        self.shape = shape
        self.nested = nested
        self.calls = [] if calls is None else calls

    def detach(self):
        self.calls.append("detach")
        return self

    def cpu(self):
        self.calls.append("cpu")
        return self

    def tolist(self):
        self.calls.append("tolist")
        return self.nested


@pytest.mark.parametrize(
    "shape",
    [(2, 2), (1, 0), (1, 2, 1), (True, 2), (1, True), None],
)
def test_encoding_rejects_every_noncanonical_tensor_shape(shape: object) -> None:
    ids = _LooseTensor(shape, [[1, 2]])
    mask = _LooseTensor((1, 2), [[1, 1]])
    with pytest.raises(MixtralPromptPrefillError) as caught:
        prefill._encode(
            lambda *_args, **_kwargs: {"input_ids": ids, "attention_mask": mask},
            "hi",
            add_special_tokens=False,
            max_tokens=2,
            max_events=4,
            target_count=1,
            routed_top_k=2,
            converter=lambda values: ["a", "b"],
            run_key="run-1",
            sequence_id="sequence-1",
        )
    assert caught.value.stage == "encoding"
    assert ids.calls == []


def test_encoding_checks_shape_match_and_max_tokens_before_materialization() -> None:
    calls: list[str] = []
    ids = _LooseTensor((1, 3), [[1, 2, 3]], calls)
    mask = _LooseTensor((1, 2), [[1, 1]], calls)
    with pytest.raises(MixtralPromptPrefillError):
        prefill._encode(
            lambda *_args, **_kwargs: {"input_ids": ids, "attention_mask": mask},
            "hi",
            add_special_tokens=False,
            max_tokens=3,
            max_events=6,
            target_count=1,
            routed_top_k=2,
            converter=lambda values: ["a"] * len(values),
            run_key="run-1",
            sequence_id="sequence-1",
        )
    assert calls == []

    mask = _LooseTensor((1, 3), [[1, 1, 1]], calls)
    with pytest.raises(MixtralPromptPrefillError):
        prefill._encode(
            lambda *_args, **_kwargs: {"input_ids": ids, "attention_mask": mask},
            "hi",
            add_special_tokens=False,
            max_tokens=2,
            max_events=6,
            target_count=1,
            routed_top_k=2,
            converter=lambda values: ["a"] * len(values),
            run_key="run-1",
            sequence_id="sequence-1",
        )
    assert calls == []


@pytest.mark.parametrize(
    "ids_nested,mask_nested",
    [
        (([1, 2],), [[1, 1]]),
        ([[1], [2]], [[1, 1]]),
        ([1, 2], [[1, 1]]),
        ([[1, 2]], ([1, 1],)),
        ([[1, 2]], [[1]]),
        ([[1, 2]], [[1, 1, 1]]),
        ([[1, 2]], [[1, 1], [1, 1]]),
    ],
)
def test_encoding_requires_exact_nested_single_rows(
    ids_nested: object, mask_nested: object
) -> None:
    ids = _LooseTensor((1, 2), ids_nested)
    mask = _LooseTensor((1, 2), mask_nested)
    with pytest.raises(MixtralPromptPrefillError):
        prefill._encode(
            lambda *_args, **_kwargs: {"input_ids": ids, "attention_mask": mask},
            "hi",
            add_special_tokens=False,
            max_tokens=2,
            max_events=4,
            target_count=1,
            routed_top_k=2,
            converter=lambda values: ["a", "b"],
            run_key="run-1",
            sequence_id="sequence-1",
        )


@pytest.mark.parametrize("bad_ids", [[True, 2], [-1, 2], [1.0, 2], ["1", 2]])
def test_encoding_rejects_nonnegative_strict_integer_ids(bad_ids: list[object]) -> None:
    ids = _LooseTensor((1, 2), [bad_ids])
    mask = _LooseTensor((1, 2), [[1, 1]])
    with pytest.raises(MixtralPromptPrefillError):
        prefill._encode(
            lambda *_args, **_kwargs: {"input_ids": ids, "attention_mask": mask},
            "hi",
            add_special_tokens=False,
            max_tokens=2,
            max_events=4,
            target_count=1,
            routed_top_k=2,
            converter=lambda values: ["a", "b"],
            run_key="run-1",
            sequence_id="sequence-1",
        )


@pytest.mark.parametrize("bad_mask", [[0, 1], [True, 1], [1.0, 1], [1, -1], ["1", 1]])
def test_encoding_rejects_nonunit_attention_masks(bad_mask: list[object]) -> None:
    ids = _LooseTensor((1, 2), [[1, 2]])
    mask = _LooseTensor((1, 2), [bad_mask])
    with pytest.raises(MixtralPromptPrefillError):
        prefill._encode(
            lambda *_args, **_kwargs: {"input_ids": ids, "attention_mask": mask},
            "hi",
            add_special_tokens=False,
            max_tokens=2,
            max_events=4,
            target_count=1,
            routed_top_k=2,
            converter=lambda values: ["a", "b"],
            run_key="run-1",
            sequence_id="sequence-1",
        )


@pytest.mark.parametrize(
    "pieces",
    [("a", "b"), ["a"], ["a", 2], ["a", None], {"a": 1}],
)
def test_encoding_requires_exact_converter_string_list(pieces: object) -> None:
    ids = _LooseTensor((1, 2), [[1, 2]])
    mask = _LooseTensor((1, 2), [[1, 1]])
    with pytest.raises(MixtralPromptPrefillError):
        prefill._encode(
            lambda *_args, **_kwargs: {"input_ids": ids, "attention_mask": mask},
            "hi",
            add_special_tokens=False,
            max_tokens=2,
            max_events=4,
            target_count=1,
            routed_top_k=2,
            converter=lambda values: pieces,
            run_key="run-1",
            sequence_id="sequence-1",
        )


@pytest.mark.parametrize("control", [KeyboardInterrupt("cancel"), SystemExit("exit")])
def test_converter_control_flow_is_exact_after_materialization(control: BaseException) -> None:
    calls: list[str] = []
    ids = _LooseTensor((1, 2), [[1, 2]], calls)
    mask = _LooseTensor((1, 2), [[1, 1]], calls)

    def converter(_values: list[int]) -> list[str]:
        raise control

    with pytest.raises(type(control)) as caught:
        prefill._encode(
            lambda *_args, **_kwargs: {"input_ids": ids, "attention_mask": mask},
            "hi",
            add_special_tokens=False,
            max_tokens=2,
            max_events=4,
            target_count=1,
            routed_top_k=2,
            converter=converter,
            run_key="run-1",
            sequence_id="sequence-1",
        )
    assert caught.value is control
    assert calls == ["detach", "cpu", "tolist", "detach", "cpu", "tolist"]


def test_converter_prompt_error_is_rewrapped_as_encoding_with_exact_cause() -> None:
    tokenizer = _Tokenizer()
    loaded, inspection, plan = _loaded_fixture("legacy", tokenizer)

    def fail(_values: list[int]) -> list[str]:
        raise _PromptError("caller converter failure")

    loaded.tokenizer.convert_ids_to_tokens = fail  # type: ignore[attr-defined]
    with pytest.raises(MixtralPromptPrefillError) as caught:
        _call_prefill(loaded, inspection, plan)
    assert caught.value.stage == "encoding"
    assert type(caught.value.__cause__) is _PromptError
    assert loaded.model.calls == 0


@pytest.mark.parametrize("layout", ["legacy", "packed"])
@pytest.mark.parametrize("add_special_tokens", [False, True])
def test_real_feature18_integration_preserves_all_identity_and_ownership(
    layout: str, add_special_tokens: bool
) -> None:
    tokenizer = _Tokenizer(2)
    marker = object()
    model = _ForwardModel(layout, output=marker, rows=[[1.0, 2.0, 0.0, 3.0]] * 2)
    loaded, inspection, plan = _loaded_fixture(layout, tokenizer, model)
    expected = 2 * len(plan.targets) * inspection.report.facts.routed_top_k
    result = _call_prefill(
        loaded,
        inspection,
        plan,
        add_special_tokens=add_special_tokens,
        max_events=expected,
    )
    assert result.output is marker
    assert [event.token_id for event in result.token_events] == [10, 11]
    assert [event.token_pos for event in result.token_events] == [0, 1]
    assert [event.token_text for event in result.token_events] == ["piece-10", "piece-11"]
    assert all(event.phase is TokenPhase.PREFILL for event in result.token_events)
    assert [event.token_key for event in result.token_events] == [
        make_token_key("run-1", "sequence-1", position, token_id, "prefill")
        for position, token_id in enumerate((10, 11))
    ]
    assert all(type(event) is TokenEvent for event in result.token_events)
    assert all(
        event.run_key == "run-1" and event.sequence_id == "sequence-1"
        for event in result.token_events
    )
    assert len(result.routing_events) == expected
    assert {event.token_key for event in result.routing_events} == {
        event.token_key for event in result.token_events
    }
    assert model.calls == 1
    assert model.received_kwargs is not None
    assert set(model.received_kwargs) == {"input_ids", "attention_mask"}
    assert model.received_kwargs["input_ids"] is tokenizer.ids
    assert model.received_kwargs["attention_mask"] is tokenizer.mask
    assert tokenizer.calls[0][1]["add_special_tokens"] is add_special_tokens
    assert len(tokenizer.calls) == 1
    assert loaded.closed is False
    assert loaded.model is model and loaded.tokenizer is tokenizer
    assert all(not node.callbacks for node in model.nodes.values())


def test_unicode_prompt_pieces_and_prompt_character_budget_are_exact() -> None:
    class UnicodeTokenizer(_Tokenizer):
        def convert_ids_to_tokens(self, values: list[int]) -> list[str]:
            self.convert_calls.append(values)
            return ["മലയാളം", ""]

    prompt = "ഹായ്"
    tokenizer = UnicodeTokenizer(2)
    model = _ForwardModel("legacy", rows=[[1.0, 2.0, 0.0, 3.0]] * 2)
    loaded, inspection, plan = _loaded_fixture("legacy", tokenizer, model)
    result = _call_prefill(
        loaded,
        inspection,
        plan,
        prompt=prompt,
        max_prompt_chars=len(prompt),
        max_events=8,
    )
    assert tokenizer.calls[0][0] is prompt
    assert [event.token_text for event in result.token_events] == ["മലയാളം", ""]
    with pytest.raises(ValueError, match="max_prompt_chars"):
        _call_prefill(
            loaded,
            inspection,
            plan,
            prompt=prompt,
            max_prompt_chars=len(prompt) - 1,
            max_events=8,
        )
    assert len(tokenizer.calls) == 1


@pytest.mark.parametrize(
    "failure", [ValueError("body"), KeyboardInterrupt("cancel"), SystemExit("exit")]
)
def test_real_feature18_failure_preserves_exact_primary_and_removes_hooks(
    failure: BaseException,
) -> None:
    class FailingModel(_ForwardModel):
        error = failure

        def __call__(self, **kwargs: object) -> object:
            super().__call__(**kwargs)
            raise self.error

    model = FailingModel("legacy", rows=[[1.0, 2.0, 0.0, 3.0], [1.0, 2.0, 0.0, 3.0]])
    tokenizer = _Tokenizer()
    loaded, inspection, plan = _loaded_fixture("legacy", tokenizer, model)
    with pytest.raises(type(failure)) as caught:
        _call_prefill(loaded, inspection, plan)
    assert caught.value is failure
    assert model.calls == 1
    assert loaded.closed is False
    assert loaded.model is model and loaded.tokenizer is tokenizer
    assert all(not node.callbacks for node in model.nodes.values())


def test_real_feature18_enter_failure_preserves_enter_cause_and_cleanup_state() -> None:
    marker = RuntimeError("named modules failed")

    class EnterFailModel(_ForwardModel):
        def named_modules(self):
            self.named_modules_calls += 1
            raise marker

    model = EnterFailModel("legacy")
    loaded, inspection, plan = _loaded_fixture("legacy", _Tokenizer(), model)
    with pytest.raises(Exception) as caught:
        _call_prefill(loaded, inspection, plan)
    assert caught.value.__cause__ is marker
    assert loaded.closed is False
    assert loaded.model is model
    assert all(not node.callbacks for node in model.nodes.values())


def test_real_feature18_persistent_cleanup_failure_exposes_both_pending_aliases_and_retry() -> None:
    class FailingModel(_ForwardModel):
        def __call__(self, **kwargs: object) -> object:
            super().__call__(**kwargs)
            raise ValueError("body")

    model = FailingModel(
        "legacy",
        rows=[[1.0, 2.0, 0.0, 3.0], [1.0, 2.0, 0.0, 3.0]],
        removal_failures=2,
    )
    loaded, inspection, plan = _loaded_fixture("legacy", _Tokenizer(), model)
    with pytest.raises(ValueError, match="body") as caught:
        _call_prefill(loaded, inspection, plan)
    pending = caught.value.pending_cleanup
    assert pending is caught.value.pending_runtime_cleanup
    assert pending.pending is True
    assert loaded.closed is False
    assert loaded.model is model
    assert any("runtime cleanup also failed" in note for note in caught.value.__notes__)
    pending.retry()
    assert pending.pending is False
    assert all(not node.callbacks for node in model.nodes.values())
    pending.retry()


def test_prefill_nonretaining_fixture_releases_mapping_tensors_and_intermediates() -> None:
    class Encoding(dict[str, object]):
        pass

    detached_refs: list[weakref.ReferenceType[object]] = []
    cpu_refs: list[weakref.ReferenceType[object]] = []

    class CpuValue:
        def __init__(self, values: list[list[int]]) -> None:
            self.values = values

        def tolist(self) -> list[list[int]]:
            return self.values

    class DetachedValue:
        def __init__(self, values: list[list[int]]) -> None:
            self.values = values

        def cpu(self) -> CpuValue:
            value = CpuValue(self.values)
            cpu_refs.append(weakref.ref(value))
            return value

    class OriginalValue:
        def __init__(self, values: list[list[int]]) -> None:
            self.values = values
            self.shape = (1, len(values[0]))

        def detach(self) -> DetachedValue:
            value = DetachedValue(self.values)
            detached_refs.append(weakref.ref(value))
            return value

    class NonRetainingTokenizer:
        def __init__(self) -> None:
            self.ids = OriginalValue([[10, 11]])
            self.mask = OriginalValue([[1, 1]])
            self.mapping = Encoding({"input_ids": self.ids, "attention_mask": self.mask})

        def __call__(self, _prompt: str, **_kwargs: object) -> Encoding:
            return self.mapping

        def convert_ids_to_tokens(self, values: list[int]) -> list[str]:
            return [f"piece-{value}" for value in values]

    class NonRetainingModel(_ForwardModel):
        def __call__(self, **kwargs: object) -> object:
            del kwargs
            self.calls += 1
            # Deliberately do not retain the model kwargs or hook payload.
            routing_payload = _payload("legacy", [[1.0, 2.0, 0.0, 3.0], [1.0, 2.0, 0.0, 3.0]])
            for node in self.nodes.values():
                node.fire(routing_payload)
            return self.output

    tokenizer = NonRetainingTokenizer()
    mapping_ref = weakref.ref(tokenizer.mapping)
    ids_ref = weakref.ref(tokenizer.ids)
    mask_ref = weakref.ref(tokenizer.mask)
    model = NonRetainingModel("legacy")
    loaded, inspection, plan = _loaded_fixture("legacy", tokenizer, model)
    result = _call_prefill(loaded, inspection, plan, max_events=8)
    del loaded, tokenizer
    gc.collect()
    assert result.output is model.output
    assert mapping_ref() is None
    assert ids_ref() is None
    assert mask_ref() is None
    assert detached_refs and all(reference() is None for reference in detached_refs)
    assert cpu_refs and all(reference() is None for reference in cpu_refs)


def test_prefill_calls_converter_and_forward_exactly_once() -> None:
    tokenizer = _Tokenizer(1)
    loaded, inspection, plan = _loaded_fixture("legacy", tokenizer)
    _call_prefill(loaded, inspection, plan, max_tokens=1, max_events=4)
    assert len(tokenizer.calls) == 1
    assert len(tokenizer.convert_calls) == 1
    assert loaded.model.calls == 1
    assert tokenizer.ids.calls == ["detach", "cpu", "tolist"]
    assert tokenizer.mask.calls == ["detach", "cpu", "tolist"]


@pytest.mark.parametrize("stage", ["tokenize", "encoding"])
def test_existing_prompt_errors_are_rewrapped_by_call_provenance(stage: str) -> None:
    supplied = MixtralPromptPrefillError("encoding" if stage == "tokenize" else "tokenize")

    class Tokenizer(_Tokenizer):
        def __call__(self, prompt: str, **kwargs: object):
            if stage == "tokenize":
                raise supplied
            return super().__call__(prompt, **kwargs)

        def convert_ids_to_tokens(self, values: list[int]) -> list[str]:
            if stage == "encoding":
                raise supplied
            return super().convert_ids_to_tokens(values)

    loaded, inspection, plan = _loaded_fixture("legacy", Tokenizer())
    with pytest.raises(MixtralPromptPrefillError) as caught:
        _call_prefill(loaded, inspection, plan)
    assert caught.value is not supplied
    assert caught.value.stage == stage
    assert caught.value.__cause__ is supplied
    assert loaded.model.calls == 0


def test_noncallable_converter_fails_before_tokenizer_call() -> None:
    tokenizer = _Tokenizer()
    tokenizer.convert_ids_to_tokens = None  # type: ignore[assignment]
    loaded, inspection, plan = _loaded_fixture("legacy", tokenizer)
    with pytest.raises(MixtralPromptPrefillError) as caught:
        _call_prefill(loaded, inspection, plan)
    assert caught.value.stage == "encoding"
    assert isinstance(caught.value.__cause__, TypeError)
    assert tokenizer.calls == []
    assert loaded.model.calls == 0


def test_prefill_max_event_budget_is_structural_and_precedes_tensor_or_converter_calls() -> None:
    tokenizer = _Tokenizer(2)
    loaded, inspection, plan = _loaded_fixture("legacy", tokenizer)
    expected = 2 * len(plan.targets) * inspection.report.facts.routed_top_k
    with pytest.raises(ValueError, match="max_events"):
        _call_prefill(loaded, inspection, plan, max_events=expected - 1)
    assert tokenizer.calls == [
        (
            "hello",
            {
                "add_special_tokens": False,
                "padding": False,
                "truncation": False,
                "return_attention_mask": True,
                "return_token_type_ids": False,
                "return_tensors": "pt",
            },
        )
    ]
    assert tokenizer.ids.calls == [] and tokenizer.mask.calls == []
    assert tokenizer.convert_calls == []
    assert loaded.model.calls == 0


def test_public_signature_kinds_annotations_and_private_preflight_return_shape() -> None:
    signature = inspect.signature(run_mixtral_prompt_prefill)
    for name in ("loaded", "inspection", "plan", "prompt"):
        assert signature.parameters[name].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name in (
        "run_key",
        "sequence_id",
        "add_special_tokens",
        "max_prompt_chars",
        "max_tokens",
        "max_events",
    ):
        parameter = signature.parameters[name]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is inspect.Parameter.empty
    hints = get_type_hints(run_mixtral_prompt_prefill)
    assert hints["loaded"] is LoadedModel
    assert hints["inspection"].__name__ == "AdapterInspection"
    assert hints["plan"].__name__ == "ProbePlan"
    assert hints["return"].__name__ == "MixtralRoutingForwardResult"
    preflight_hints = get_type_hints(prefill._preflight)
    assert str(preflight_hints["return"]).count("int") >= 3


def test_prompt_prefill_source_has_no_filesystem_network_model_or_dynamic_import_paths() -> None:
    path = Path(prefill.__file__)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    imports: set[str] = set()
    calls: set[str] = set()
    dynamic_imports: list[ast.Call] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add((node.module or "").split(".")[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
                if node.func.id in {"__import__", "eval", "exec", "open", "compile"}:
                    dynamic_imports.append(node)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
                if node.func.attr in {
                    "import_module",
                    "urlopen",
                    "socket",
                    "read_text",
                    "write_text",
                }:
                    dynamic_imports.append(node)
    forbidden_imports = {
        "torch",
        "transformers",
        "accelerate",
        "safetensors",
        "numpy",
        "os",
        "pathlib",
        "socket",
        "urllib",
        "subprocess",
        "importlib",
        "tempfile",
        "shutil",
        "webbrowser",
        "http",
        "requests",
        "threading",
        "asyncio",
    }
    assert imports.isdisjoint(forbidden_imports)
    assert not dynamic_imports
    assert calls.isdisjoint(
        {
            "generate",
            "train",
            "eval",
            "save",
            "load",
            "download",
            "close",
            "__enter__",
            "__exit__",
            "mkdir",
            "write_bytes",
            "replace",
            "start",
            "run",
        }
    )
    assert "moeatlas.store" not in source
    assert "moeatlas.cli" not in source
    assert "FastAPI" not in source
    assert "run_mixtral_routing_forward" in source
    assert source.count("run_mixtral_routing_forward(") == 1
    assert (
        sum(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "tokenizer"
            for node in ast.walk(tree)
        )
        == 1
    )
    assert (
        sum(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "converter"
            for node in ast.walk(tree)
        )
        == 1
    )
    assert source.count('return_tensors="pt"') == 1
