"""Model-free tests for the structure-driven routing capture composition.

The generic seam must discover router modules from a static ``[STRUCTURE]``
report, attach passive hooks, and decode top-k generically — with no certified
adapter inspection anywhere. All doubles are torch-free; payloads are plain
lists or tiny tensor-like stand-ins implementing ``detach/cpu/float/tolist``.
"""

from __future__ import annotations

import inspect
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from moeatlas.discovery import DiscoveryFacts, scan
from moeatlas.events import TokenEvent, TokenPhase
from moeatlas.fixtures.synthetic_moe import (
    SyntheticConfig,
    SyntheticExpert,
    SyntheticExperts,
    SyntheticLayers,
    SyntheticMoEBlock,
    SyntheticRouter,
    SyntheticSharedExpert,
)
from moeatlas.runtime import (
    StructuredCaptureError,
    run_structured_routing_forward,
    structured_router_targets,
)

from .test_cli_scan import _loading_manifest, _loading_plan

ROOT = Path(__file__).resolve().parents[1]


class _Handle:
    def __init__(self, owner: _HookedNode, callback: object) -> None:
        self.owner = owner
        self.callback = callback

    def remove(self) -> None:
        if self.callback in self.owner.callbacks:
            self.owner.callbacks.remove(self.callback)


class _HookedNode:
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.callbacks: list[object] = []
        self.payload: object = None

    def register_forward_hook(self, callback: object) -> _Handle:
        self.callbacks.append(callback)
        return _Handle(self, callback)

    def fire(self, payload: object) -> None:
        self.payload = payload
        for callback in tuple(self.callbacks):
            callback(self, (), payload)


class _HookedRouter(_HookedNode, SyntheticRouter):
    pass


class _HookedModel:
    """Synthetic-MoE-shaped model whose routers are passive hook nodes."""

    def __init__(
        self,
        payload: object,
        *,
        num_layers: int = 2,
        reverse_fire_order: bool = False,
        skip_paths: tuple[str, ...] = (),
    ) -> None:
        config = SyntheticConfig(num_layers=num_layers)

        def make_block() -> SyntheticMoEBlock:
            experts = SyntheticExperts(
                children={
                    str(index): SyntheticExpert(
                        parameters={
                            "w1.weight": type("P", (), {"shape": (16, 8)})(),
                            "w2.weight": type("P", (), {"shape": (8, 16)})(),
                        }
                    )
                    for index in range(4)
                }
            )
            return SyntheticMoEBlock(
                children={
                    "router": _HookedRouter(
                        parameters={"weight": type("P", (), {"shape": (4, 8)})()}
                    ),
                    "experts": experts,
                    "shared_expert": SyntheticSharedExpert(
                        parameters={"weight": type("P", (), {"shape": (8, 8)})()}
                    ),
                }
            )

        self.config = config
        root_children = {
            "layers": SyntheticLayers(
                children={str(index): make_block() for index in range(config.num_layers)}
            )
        }
        self._children = root_children
        self.payload_by_path: dict[str, object] = {}
        self.fire_paths = [
            path for path, module in self.named_modules() if isinstance(module, _HookedRouter)
        ]
        if reverse_fire_order:
            self.fire_paths = list(reversed(self.fire_paths))
        self.skip_paths = set(skip_paths)
        self.calls = 0
        self.output = object()
        for path in self.fire_paths:
            self.payload_by_path[path] = payload

    def named_modules(self):
        yield "", self
        for child_name, child in self._children.items():
            for nested_name, nested_module in child.named_modules():
                full_name = child_name if not nested_name else f"{child_name}.{nested_name}"
                yield full_name, nested_module

    def __call__(self, **kwargs: object) -> object:
        self.calls += 1
        self.received_kwargs = kwargs
        for path in self.fire_paths:
            if path in self.skip_paths:
                continue
            node = dict(self.named_modules())[path]
            node.fire(self.payload_by_path[path])
        return self.output


def _tokens(count: int = 3) -> tuple[TokenEvent, ...]:
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


def _flat_logits(count: int = 3, num_experts: int = 4) -> list[list[float]]:
    return [[0.9 - 0.2 * index for index in range(num_experts)] for _ in range(count)]


def _report(model: _HookedModel):
    return scan(model, _loading_manifest(_loading_plan()))


def _targets(model: _HookedModel):
    return structured_router_targets(_report(model))


def test_targets_bind_layers_experts_and_top_k() -> None:
    model = _HookedModel(_flat_logits())
    targets = _targets(model)
    assert [target.layer_index for target in targets] == [0, 1]
    assert all(len(target.expert_keys) == 4 for target in targets)
    assert all(target.routed_top_k == 2 for target in targets)
    assert all(target.module_path.endswith(".router") for target in targets)


def test_missing_facts_are_rejected_before_hooks() -> None:
    model = _HookedModel(_flat_logits())
    report = _report(model)
    stripped = report.model_copy(update={"facts": DiscoveryFacts()})
    with pytest.raises(StructuredCaptureError, match="expert_count"):
        structured_router_targets(stripped)


def test_missing_expert_is_rejected() -> None:
    model = _HookedModel(_flat_logits())
    report = _report(model)
    stripped_candidates = [
        candidate
        for candidate in report.candidates
        if not (candidate.kind.value == "expert" and candidate.layer_index == 1)
    ]
    stripped_components = [
        component
        for component in report.components
        if not (component.kind.value == "expert" and component.layer_index == 1)
    ]
    stripped = report.model_copy(
        update={"candidates": stripped_candidates, "components": stripped_components}
    )
    with pytest.raises(StructuredCaptureError, match="routed experts"):
        structured_router_targets(stripped)


def test_flat_logits_capture_records_capability_note_and_logits() -> None:
    model = _HookedModel(_flat_logits())
    result = run_structured_routing_forward(
        model, _report(model), _tokens(), {"input_ids": [[10, 11, 12]]}, max_events=64
    )
    assert result.output is model.output
    assert len(result.routing_events) == 3 * 2 * 2
    assert any("normalization" in note for note in result.capability_notes)
    for event in result.routing_events:
        assert event.router_logit is not None
        assert event.probability is None and event.weight is None
        assert event.selected is True


def test_softmax_config_drives_probability_columns() -> None:
    model = _HookedModel(_flat_logits(1))
    config = {"score_function": "softmax"}
    result = run_structured_routing_forward(
        model, _report(model), _tokens(1), {"input_ids": [[10]]}, max_events=32, config=config
    )
    first_layer_rows = sorted(result.routing_events[:2], key=lambda event: event.rank)
    assert all(event.probability is not None for event in first_layer_rows)
    assert all(0.0 < event.probability < 1.0 for event in first_layer_rows)
    assert first_layer_rows[0].probability > first_layer_rows[1].probability
    assert result.capability_notes == ()


def test_sigmoid_config_drives_probability_columns() -> None:
    model = _HookedModel(_flat_logits(1))
    result = run_structured_routing_forward(
        model,
        _report(model),
        _tokens(1),
        {"input_ids": [[10]]},
        max_events=32,
        config=type("C", (), {"score_function": "sigmoid"})(),
    )
    events = [event for event in result.routing_events if event.rank == 0][:1]
    assert len(events) == 1
    expected = 1.0 / (1.0 + math.exp(-events[0].router_logit))
    assert events[0].probability == pytest.approx(expected)


def test_packed_payload_decodes_native_indices_and_scores() -> None:
    logits = [[1.0, 2.0, 3.0, 4.0]]
    scores = [[0.7, 0.3]]
    indices = [[3, 0]]
    model = _HookedModel((logits, scores, indices))
    result = run_structured_routing_forward(
        model, _report(model), _tokens(1), {"input_ids": [[10]]}, max_events=32
    )
    layer0 = sorted(result.routing_events[:2], key=lambda event: event.rank)
    assert layer0[0].expert_key != layer0[1].expert_key
    assert layer0[0].probability == pytest.approx(0.7)
    assert layer0[0].weight == pytest.approx(0.7)
    assert layer0[0].router_logit == pytest.approx(4.0)
    assert layer0[1].router_logit == pytest.approx(1.0)
    assert result.capability_notes == ()


def test_tied_cutoff_scores_are_rejected() -> None:
    tied = [[0.9, 0.9, 0.5, 0.1]]
    model = _HookedModel(tied)
    with pytest.raises(StructuredCaptureError, match="tied"):
        run_structured_routing_forward(
            model, _report(model), _tokens(1), {"input_ids": [[10]]}, max_events=32
        )
    assert model.calls == 1
    for path in model.fire_paths:
        assert dict(model.named_modules())[path].callbacks == []


def test_insufficient_budget_fails_before_any_hook_or_call() -> None:
    model = _HookedModel(_flat_logits())
    with pytest.raises(StructuredCaptureError, match="max_events is insufficient"):
        run_structured_routing_forward(
            model, _report(model), _tokens(3), {}, max_events=11
        )
    assert model.calls == 0
    for path in model.fire_paths:
        assert dict(model.named_modules())[path].callbacks == []


def test_unresolvable_router_module_never_reaches_the_model() -> None:
    class _Bare:
        def named_modules(self):
            return iter([("", self)])

        def __call__(self, **kwargs: object) -> object:  # pragma: no cover - never reached
            raise AssertionError("model must not be called")

    bare = _Bare()
    with pytest.raises(Exception, match="missing from named_modules"):
        run_structured_routing_forward(
            bare, _report(_HookedModel(_flat_logits())), _tokens(1), {}, max_events=32
        )


def test_silent_router_fails_the_capture_without_publishing() -> None:
    silent = _HookedModel(_flat_logits(1), skip_paths=("layers.1.router",))
    with pytest.raises(StructuredCaptureError, match="did not fire"):
        run_structured_routing_forward(
            silent, _report(silent), _tokens(1), {}, max_events=32
        )
    assert silent.calls == 1
    for path in silent.fire_paths:
        assert dict(silent.named_modules())[path].callbacks == []


def test_row_count_mismatch_fails_decode() -> None:
    model = _HookedModel([[0.5, 0.4, 0.3, 0.1]])
    with pytest.raises(StructuredCaptureError, match="one row per captured token"):
        run_structured_routing_forward(
            model, _report(model), _tokens(3), {}, max_events=64
        )


def test_reverse_fire_order_still_publishes_canonical_layer_blocks() -> None:
    model = _HookedModel(_flat_logits(1), reverse_fire_order=True)
    result = run_structured_routing_forward(
        model, _report(model), _tokens(1), {}, max_events=32
    )
    blocks: list[str] = []
    for event in result.routing_events:
        if not blocks or blocks[-1] != event.layer_key:
            blocks.append(event.layer_key)
    assert len(blocks) == 2
    targets = _targets(model)
    expected_order = [t.layer_key for t in sorted(targets, key=lambda t: t.layer_index)]
    assert blocks == expected_order


# ---------------------------------------------------------------------------
# Isolation guards
# ---------------------------------------------------------------------------


def test_generic_capture_imports_without_model_stack() -> None:
    script = (
        "import sys\n"
        "for name in ('torch', 'transformers', 'duckdb', 'safetensors'):\n"
        "    sys.modules[name] = None\n"
        "import moeatlas.runtime.generic_capture\n"
        "print('generic-capture-import-ok')\n"
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
    assert "generic-capture-import-ok" in completed.stdout


def test_no_model_runtime_imported_by_generic_capture() -> None:
    import moeatlas.runtime.generic_capture as module

    assert module.__name__ == "moeatlas.runtime.generic_capture"
    assert not any(name in sys.modules for name in ("torch", "transformers"))


def test_public_surface_contract() -> None:
    signature = inspect.signature(run_structured_routing_forward)
    assert tuple(signature.parameters)[:4] == (
        "model",
        "report",
        "token_events",
        "model_kwargs",
    )
    assert signature.parameters["max_events"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["config"].default is None
