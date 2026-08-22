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


def test_ling_style_payload_preserves_scaled_weights_without_probability_claims() -> None:
    # Ling-3.0-tiny emits (indices, weights, logits).  Its routed weights can
    # include a model-specific scaling factor and therefore are not guaranteed
    # to lie in the probability interval.
    indices = [[3, 0]]
    weights = [[1.4, 0.6]]
    logits = [[1.0, 2.0, 3.0, 4.0]]
    model = _HookedModel((indices, weights, logits))
    result = run_structured_routing_forward(
        model, _report(model), _tokens(1), {"input_ids": [[10]]}, max_events=32
    )
    layer0 = sorted(result.routing_events[:2], key=lambda event: event.rank)
    assert layer0[0].router_logit == pytest.approx(4.0)
    assert layer0[0].weight == pytest.approx(1.4)
    assert layer0[1].router_logit == pytest.approx(1.0)
    assert layer0[1].weight == pytest.approx(0.6)
    assert all(event.probability is None for event in layer0)
    assert any("retained as weights" in note for note in result.capability_notes)


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
# Ling/BailingMoeV3-style foreign-family regression coverage
# ---------------------------------------------------------------------------


class _LingHooks(_HookedNode):
    """Passive hook node with fixture-tree children and parameters."""

    def __init__(
        self,
        *,
        children: dict[str, object] | None = None,
        parameters: dict[str, object] | None = None,
    ) -> None:
        super().__init__()
        self._children = dict(children or {})
        self._parameters = dict(parameters or {})

    def named_modules(self):
        yield "", self
        for child_name, child in self._children.items():
            for nested_name, nested_module in child.named_modules():
                full_name = child_name if not nested_name else f"{child_name}.{nested_name}"
                yield full_name, nested_module

    def named_parameters(self):
        for parameter_name, parameter in self._parameters.items():
            yield parameter_name, parameter
        for child_name, child in self._children.items():
            for nested_name, parameter in child.named_parameters():
                yield f"{child_name}.{nested_name}", parameter


def _ling_param(shape: tuple[int, ...]) -> object:
    return type("P", (), {"shape": shape})()


class BailingMoeV3RMSNorm(_LingHooks):
    """Class name tokenizes to contain 'moe' (observed VM noise source)."""


class BailingMoeV3Linear(_LingHooks):
    pass


class BailingMoeV3MLP(_LingHooks):
    """One routed expert's SwiGLU block; its gate_proj tokenizes like a router."""


class BailingMoeV3SharedExpertMLP(_LingHooks):
    pass


class BailingMoeV3Experts(_LingHooks):
    pass


class BailingMoeV3MoEGate(_LingHooks):
    """The real router module producing routing logits."""


class BailingMoeV3SparseMoeBlock(_LingHooks):
    pass


class BailingMoeV3Attention(_LingHooks):
    pass


class BailingMoeV3DecoderLayer(_LingHooks):
    pass


class LingNamedModel:
    """Torch-free BailingMoeV3-shaped double mimicking observed Ling naming.

    Every noise source from the live VM scan is present: 130 ROUTER
    candidates per layer (one real ``mlp.gate`` plus every SwiGLU expert and
    shared-expert ``gate_proj``) and five MOE_LAYER candidates per layer from
    ``...Moe...`` class names — while only ``model.layers.<n>.mlp.gate``
    produces router logits.
    """

    def __init__(
        self,
        payload: object,
        *,
        num_layers: int = 2,
        num_experts: int = 8,
        routed_top_k: int = 2,
    ) -> None:
        hidden, intermediate = 32, 64

        def swiglu(cls: type[_LingHooks]) -> _LingHooks:
            return cls(
                children={
                    "gate_proj": BailingMoeV3Linear(
                        parameters={"weight": _ling_param((intermediate, hidden))}
                    ),
                    "up_proj": BailingMoeV3Linear(
                        parameters={"weight": _ling_param((intermediate, hidden))}
                    ),
                    "down_proj": BailingMoeV3Linear(
                        parameters={"weight": _ling_param((hidden, intermediate))}
                    ),
                }
            )

        def make_layer() -> BailingMoeV3DecoderLayer:
            return BailingMoeV3DecoderLayer(
                children={
                    "input_layernorm": BailingMoeV3RMSNorm(
                        parameters={"weight": _ling_param((hidden,))}
                    ),
                    "self_attn": BailingMoeV3Attention(
                        children={
                            projection: BailingMoeV3Linear(
                                parameters={"weight": _ling_param((hidden, hidden))}
                            )
                            for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
                        }
                    ),
                    "post_attention_layernorm": BailingMoeV3RMSNorm(
                        parameters={"weight": _ling_param((hidden,))}
                    ),
                    "mlp": BailingMoeV3SparseMoeBlock(
                        children={
                            "gate": BailingMoeV3MoEGate(
                                parameters={"weight": _ling_param((num_experts, hidden))}
                            ),
                            "experts": BailingMoeV3Experts(
                                children={
                                    str(index): swiglu(BailingMoeV3MLP)
                                    for index in range(num_experts)
                                }
                            ),
                            "shared_expert": swiglu(BailingMoeV3SharedExpertMLP),
                        }
                    ),
                }
            )

        self.config = {
            "num_experts": num_experts,
            "num_experts_per_tok": routed_top_k,
            "n_shared_experts": 1,
        }
        self._children = {
            "model": _LingHooks(
                children={
                    "embed_tokens": BailingMoeV3Linear(
                        parameters={"weight": _ling_param((hidden, hidden))}
                    ),
                    "layers": _LingHooks(
                        children={str(index): make_layer() for index in range(num_layers)}
                    ),
                    "norm": BailingMoeV3RMSNorm(parameters={"weight": _ling_param((hidden,))}),
                }
            )
        }
        self.payload_by_path: dict[str, object] = {}
        self.fire_paths = [
            path
            for path, module in self.named_modules()
            if isinstance(module, BailingMoeV3MoEGate)
        ]
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

    def named_parameters(self):
        for child_name, child in self._children.items():
            for nested_name, parameter in child.named_parameters():
                yield f"{child_name}.{nested_name}", parameter

    def __call__(self, **kwargs: object) -> object:
        del kwargs
        self.calls += 1
        modules = dict(self.named_modules())
        for path in self.fire_paths:
            modules[path].fire(self.payload_by_path[path])
        return self.output


def test_ling_style_report_pollution_never_becomes_hook_targets() -> None:
    model = LingNamedModel(_flat_logits(1, 8))
    report = _report(model)
    routers = [c for c in report.components if c.kind.value == "router"]
    moe_layers = [c for c in report.components if c.kind.value == "moe_layer"]
    assert len(routers) > len(model.fire_paths) == 2
    assert len(moe_layers) > 2
    targets = structured_router_targets(report)
    assert [target.module_path for target in targets] == [
        "model.layers.0.mlp.gate",
        "model.layers.1.mlp.gate",
    ]
    assert [target.layer_index for target in targets] == [0, 1]
    assert all(target.routed_top_k == 2 for target in targets)
    assert all(len(target.expert_keys) == 8 for target in targets)
    published_blocks = {
        component.component_key
        for component in report.components
        if component.kind.value == "moe_layer" and component.module_path.endswith(".mlp")
    }
    assert {target.layer_key for target in targets} == published_blocks


def test_ling_style_capture_decodes_end_to_end_through_the_real_routers() -> None:
    logits = [[1.0 - 0.05 * index for index in range(8)] for _ in range(2)]
    model = LingNamedModel(logits)
    result = run_structured_routing_forward(
        model, _report(model), _tokens(2), {"input_ids": [[10, 11]]}, max_events=64
    )
    assert result.output is model.output
    assert len(result.routing_events) == 2 * 2 * 2
    assert any("normalization" in note for note in result.capability_notes)
    assert model.calls == 1
    modules = dict(model.named_modules())
    for path in model.fire_paths:
        assert modules[path].callbacks == []


class DeepseekV2MoE(_LingHooks):
    pass


class DeepseekV2MLP(_LingHooks):
    pass


class DeepseekV2Gate(_LingHooks):
    pass


class DeepseekV2Experts(_LingHooks):
    pass


class DeepseekV2ExpertFFN(_LingHooks):
    pass


class _NestedRouterModel(_LingHooks):
    """Callable tree root whose gate leaves fire one passive payload."""

    def __init__(
        self,
        *,
        children: dict[str, object],
        config: dict[str, object],
        fire_class: type[_LingHooks],
        payload: object,
    ) -> None:
        super().__init__(children=children)
        self.config = config
        self.payload = payload
        self.calls = 0
        self.output = object()
        self.fire_paths = [
            path for path, module in self.named_modules() if isinstance(module, fire_class)
        ]

    def __call__(self, **kwargs: object) -> object:
        del kwargs
        self.calls += 1
        modules = dict(self.named_modules())
        for path in self.fire_paths:
            modules[path].fire(self.payload)
        return self.output


def _deepseek_style_model(payload: object, *, gate_leaf: str = "gate") -> _NestedRouterModel:
    """Router published beside the block while experts nest under ``mlp``."""

    def make_block() -> DeepseekV2MoE:
        return DeepseekV2MoE(
            children={
                gate_leaf: DeepseekV2Gate(parameters={"weight": _ling_param((4, 16))}),
                "mlp": DeepseekV2MLP(
                    children={
                        "experts": DeepseekV2Experts(
                            children={
                                str(index): DeepseekV2ExpertFFN(
                                    parameters={
                                        "w1": _ling_param((32, 16)),
                                        "w2": _ling_param((16, 32)),
                                    }
                                )
                                for index in range(4)
                            }
                        )
                    }
                ),
            }
        )

    return _NestedRouterModel(
        children={
            "layers": _LingHooks(
                children={"0": make_block(), "1": make_block()}
            )
        },
        config={"num_experts": 4, "num_experts_per_tok": 2},
        fire_class=DeepseekV2Gate,
        payload=payload,
    )


def test_fallback_binds_gate_routers_nested_away_from_containers() -> None:
    model = _deepseek_style_model(_flat_logits(1, 4))
    report = _report(model)
    targets = structured_router_targets(report)
    assert [target.module_path for target in targets] == ["layers.0.gate", "layers.1.gate"]
    published_blocks = {
        component.component_key
        for component in report.components
        if component.kind.value == "moe_layer"
    }
    assert {target.layer_key for target in targets} == published_blocks
    result = run_structured_routing_forward(
        model, report, _tokens(1), {"input_ids": [[10]]}, max_events=32
    )
    assert len(result.routing_events) == 1 * 2 * 2


def test_fallback_rejects_router_names_without_an_exact_gate_leaf() -> None:
    model = _deepseek_style_model(_flat_logits(1, 4), gate_leaf="gating_unit")
    with pytest.raises(StructuredCaptureError, match="does not publish any routed router"):
        structured_router_targets(_report(model))
    assert model.calls == 0


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
