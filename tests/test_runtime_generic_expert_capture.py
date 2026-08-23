"""Model-free tests for structure-driven expert-activity capture (R3.1).

The generic expert seam must resolve routed-expert modules from a static
``[STRUCTURE]`` report, attach passive ``EXPERT_ACTIVITY`` forward hooks, and
reduce each invocation to input/output/contribution L2 norms — with no model
stack import and no per-family code. All doubles are torch-free; payloads are
plain lists or tiny tensor-like stand-ins implementing
``detach/cpu/float/tolist``.
"""

from __future__ import annotations

import inspect
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from moeatlas.events import ExpertEvent, TokenPhase
from moeatlas.fixtures.synthetic_moe import (
    SyntheticConfig,
    SyntheticExpert,
    SyntheticExperts,
    SyntheticLayers,
    SyntheticMoEBlock,
    SyntheticRouter,
)
from moeatlas.runtime import (
    StructuredCaptureError,
    StructuredRoutingForwardResult,
    decode_expert_activity,
    run_structured_expert_forward,
    structured_expert_targets,
)

from .test_cli_scan import _loading_manifest, _loading_plan
from .test_runtime_generic_capture import _flat_logits

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

    def register_forward_hook(self, callback: object) -> _Handle:
        self.callbacks.append(callback)
        return _Handle(self, callback)

    def fire(self, *args: object) -> None:
        for callback in tuple(self.callbacks):
            callback(self, *args)


class _HookedRouter(_HookedNode, SyntheticRouter):
    pass


class _HookedExpert(_HookedNode, SyntheticExpert):
    pass


class _ExpertHookedModel:
    """Synthetic-MoE-shaped model with passive router AND expert hook nodes."""

    def __init__(
        self,
        router_payload: object,
        *,
        num_layers: int = 2,
        expert_inputs: dict[str, object] | None = None,
        expert_outputs: dict[str, object] | None = None,
        skip_paths: tuple[str, ...] = (),
    ) -> None:
        config = SyntheticConfig(num_layers=num_layers)
        self._expert_inputs = expert_inputs or {}
        self._expert_outputs = expert_outputs or {}
        self.skip_paths = set(skip_paths)
        self.token_rows = 1

        def make_block() -> SyntheticMoEBlock:
            experts = SyntheticExperts(
                children={
                    str(index): _HookedExpert(
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
                }
            )

        self.config = config
        root_children = {
            "layers": SyntheticLayers(
                children={str(index): make_block() for index in range(config.num_layers)}
            )
        }
        self._children = root_children
        self.calls = 0
        self.output = object()

    def named_modules(self):
        yield "", self
        for child_name, child in self._children.items():
            for nested_name, nested_module in child.named_modules():
                full_name = child_name if not nested_name else f"{child_name}.{nested_name}"
                yield full_name, nested_module

    def _node(self, path: str) -> object:
        return dict(self.named_modules())[path]

    def __call__(self, **kwargs: object) -> object:
        del kwargs
        self.calls += 1
        modules = dict(self.named_modules())
        rows = [[0.9 - 0.2 * index for index in range(4)] for _ in range(self.token_rows)]
        for path, module in modules.items():
            if isinstance(module, _HookedRouter) and path not in self.skip_paths:
                module.fire((), rows)
        for path, module in modules.items():
            if isinstance(module, _HookedExpert) and path not in self.skip_paths:
                inputs = self._expert_inputs.get(path, ([[1.0, 2.0]],))
                output = self._expert_outputs.get(path, [[3.0, 4.0, 5.0]])
                module.fire(inputs, output)
        return self.output


def _tokens(count: int = 1, *, sequence: str = "sequence-1") -> tuple:
    from moeatlas.events import TokenEvent

    return tuple(
        TokenEvent(
            run_key="run-1",
            sequence_id=sequence,
            token_pos=index,
            token_id=index + 10,
            token_text=str(index),
            phase=TokenPhase.PREFILL,
        )
        for index in range(count)
    )


def _report(model: _ExpertHookedModel):
    return scan_report(model)


def scan_report(model: object):
    from moeatlas.discovery import scan

    return scan(model, _loading_manifest(_loading_plan()))


def test_expert_targets_bind_layers_and_component_keys() -> None:
    model = _ExpertHookedModel(_flat_logits())
    targets = structured_expert_targets(scan_report(model))
    assert [target.layer_index for target in targets] == [0, 0, 0, 0, 1, 1, 1, 1]
    assert [target.module_path for target in targets[:4]] == [
        "layers.0.experts.0",
        "layers.0.experts.1",
        "layers.0.experts.2",
        "layers.0.experts.3",
    ]
    assert all(target.layer_key.startswith("component:") for target in targets)


def test_decode_expert_activity_closed_form_norms() -> None:
    input_norm, output_norm, contribution = decode_expert_activity(
        ([[3.0, 4.0]],), [[12.0, 5.0]]
    )
    assert input_norm == pytest.approx(5.0)
    assert output_norm == pytest.approx(13.0)
    assert contribution == pytest.approx(math.sqrt((12.0 - 3.0) ** 2 + (5.0 - 4.0) ** 2))


def test_decode_expert_activity_contribution_requires_matching_shape() -> None:
    _, _, contribution = decode_expert_activity(([[1.0, 2.0]],), [[1.0, 2.0, 3.0]])
    assert contribution is None


def test_decode_expert_activity_materializes_tensor_like_payloads() -> None:
    class _TensorLike:
        def __init__(self, rows: list[list[float]]) -> None:
            self._rows = rows

        def detach(self) -> _TensorLike:
            return self

        def cpu(self) -> _TensorLike:
            return self

        def float(self) -> _TensorLike:
            return self

        def tolist(self) -> list[list[float]]:
            return self._rows

    input_norm, output_norm, contribution = decode_expert_activity(
        (_TensorLike([[3.0, 4.0]]),), _TensorLike([[3.0, 4.0]])
    )
    assert input_norm == pytest.approx(5.0)
    assert output_norm == pytest.approx(5.0)
    assert contribution == pytest.approx(0.0)


def test_nonfinite_or_nonfloat_entries_are_rejected() -> None:
    with pytest.raises(StructuredCaptureError, match="finite"):
        decode_expert_activity(([[1.0, float("nan")]],), [[1.0]])
    with pytest.raises(StructuredCaptureError, match="finite"):
        decode_expert_activity(([[1.0, "x"]],), [[1.0]])


def test_full_forward_produces_linked_routing_and_expert_events() -> None:
    model = _ExpertHookedModel(_flat_logits())
    model.token_rows = 2
    tokens = _tokens(2)
    result = run_structured_expert_forward(
        model, scan_report(model), tokens, {}, max_events=64
    )
    assert type(result) is StructuredRoutingForwardResult
    # flat logits select experts 0 and 1 at every layer, top_k=2.
    assert len(result.routing_events) == 2 * 2 * 2
    assert len(result.expert_events) == 2 * 2 * 2
    selected_pairs = {(event.layer_key, event.expert_key) for event in result.routing_events}
    expert_pairs = {(event.expert_key) for event in result.expert_events}
    assert {pair[1] for pair in selected_pairs} == expert_pairs
    token_keys = {event.token_key for event in result.expert_events}
    assert token_keys == {event.token_key for event in tokens}
    for event in result.expert_events:
        assert type(event) is ExpertEvent
        assert event.input_norm is not None and event.input_norm > 0
        assert event.output_norm is not None and event.output_norm > 0
        assert event.contribution_norm is None
        assert event.metadata["invocation_token_count"] == 2


def test_inactive_sparse_experts_do_not_fail_activity_capture() -> None:
    """Real sparse MoEs invoke selected experts, not the entire universe."""

    inactive = tuple(
        f"layers.{layer}.experts.{expert}"
        for layer in range(2)
        for expert in (2, 3)
    )
    model = _ExpertHookedModel(_flat_logits(), skip_paths=inactive)

    result = run_structured_expert_forward(
        model, scan_report(model), _tokens(1), {}, max_events=32
    )

    # Flat logits select experts 0 and 1 at both layers. Experts 2 and 3 are
    # valid inactive cells and therefore produce neither hooks nor events.
    assert len(result.routing_events) == 1 * 2 * 2
    assert len(result.expert_events) == 1 * 2 * 2
    selected = {event.expert_key for event in result.routing_events}
    assert {event.expert_key for event in result.expert_events} == selected


def test_matching_shapes_record_contribution_norms() -> None:
    model = _ExpertHookedModel(
        _flat_logits(),
        expert_inputs={},
        expert_outputs={"layers.0.experts.0": [[1.5, 2.5]]},
    )
    result = run_structured_expert_forward(
        model, scan_report(model), _tokens(1), {}, max_events=32
    )
    matching = [
        event
        for event in result.expert_events
        if event.expert_key
        == next(
            target.component_key
            for target in structured_expert_targets(scan_report(model))
            if target.module_path == "layers.0.experts.0"
        )
    ]
    assert matching
    expected = math.sqrt(0.5 ** 2 + 0.5 ** 2)
    assert all(event.contribution_norm == pytest.approx(expected) for event in matching)
    others = [event for event in result.expert_events if event not in matching]
    assert all(event.contribution_norm is None for event in others)


def test_selected_expert_that_does_not_fire_fails_without_publishing() -> None:
    model = _ExpertHookedModel(
        _flat_logits(), skip_paths=("layers.0.experts.0", "layers.1.experts.0")
    )
    with pytest.raises(StructuredCaptureError, match="did not fire"):
        run_structured_expert_forward(model, scan_report(model), _tokens(1), {}, max_events=32)
    assert model.calls == 1
    for path, module in model.named_modules():
        if isinstance(module, _HookedNode):
            assert module.callbacks == []


def test_silent_router_still_fails_the_expert_capture() -> None:
    model = _ExpertHookedModel(_flat_logits(), skip_paths=("layers.0.router",))
    with pytest.raises(StructuredCaptureError, match="routers did not fire"):
        run_structured_expert_forward(model, scan_report(model), _tokens(1), {}, max_events=32)
    assert model.calls == 1
    for path, module in model.named_modules():
        if isinstance(module, _HookedNode):
            assert module.callbacks == []


def test_insufficient_expert_budget_fails_before_any_hook_or_call() -> None:
    model = _ExpertHookedModel(_flat_logits())
    with pytest.raises(StructuredCaptureError, match="max_expert_events is insufficient"):
        run_structured_expert_forward(
            model, scan_report(model), _tokens(3), {}, max_events=64, max_expert_events=11
        )
    assert model.calls == 0
    for path, module in model.named_modules():
        if isinstance(module, _HookedNode):
            assert module.callbacks == []


def test_invalid_argument_types_are_rejected_before_resolution() -> None:
    model = _ExpertHookedModel(_flat_logits())
    report = scan_report(model)
    with pytest.raises(TypeError, match="max_expert_events"):
        run_structured_expert_forward(
            model, report, _tokens(1), {}, max_events=32, max_expert_events=True
        )
    with pytest.raises(ValueError, match="max_expert_events"):
        run_structured_expert_forward(
            model, report, _tokens(1), {}, max_events=32, max_expert_events=0
        )
    assert model.calls == 0


def test_result_revalidates_expert_events() -> None:
    model = _ExpertHookedModel(_flat_logits())
    base = run_structured_expert_forward(
        model, scan_report(model), _tokens(1), {}, max_events=32
    )
    with pytest.raises(TypeError, match="exact ExpertEvent"):
        StructuredRoutingForwardResult(
            output=base.output,
            token_events=base.token_events,
            routing_events=base.routing_events,
            expert_events=(object(),),  # type: ignore[list-item]
        )
    with pytest.raises(ValueError, match="expert_events must have unique"):
        StructuredRoutingForwardResult(
            output=base.output,
            token_events=base.token_events,
            routing_events=base.routing_events,
            expert_events=base.expert_events + base.expert_events,
        )


def test_generic_expert_capture_imports_without_model_stack() -> None:
    script = (
        "import sys\n"
        "for name in ('torch', 'transformers', 'duckdb', 'safetensors'):\n"
        "    sys.modules[name] = None\n"
        "import moeatlas.runtime.generic_capture\n"
        "print('expert-capture-import-ok')\n"
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
    assert "expert-capture-import-ok" in completed.stdout


def test_public_surface_contract() -> None:
    signature = inspect.signature(run_structured_expert_forward)
    assert tuple(signature.parameters)[:4] == (
        "model",
        "report",
        "token_events",
        "model_kwargs",
    )
    assert signature.parameters["max_events"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["config"].default is None
    assert signature.parameters["max_expert_events"].kind is inspect.Parameter.KEYWORD_ONLY
    assert inspect.signature(structured_expert_targets).parameters.keys() == {"report"}
