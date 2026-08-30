"""Model-free generic capture checks for the Qwen3.8 Flash Next surface."""

from __future__ import annotations

from moeatlas.core import (
    DType,
    ModelManifest,
    TokenizerIdentity,
    make_config_hash,
    make_model_key,
)
from moeatlas.discovery import scan
from moeatlas.runtime import (
    classify_capture_support,
    run_structured_routing_forward,
    structured_router_targets,
)

from .fixtures.qwen4_exp import Qwen4ExpHookableForConditionalGeneration


def _manifest() -> ModelManifest:
    return ModelManifest(
        model_key=make_model_key("Qwen/Qwen3.8-Flash-Next-FP8", "fixture"),
        architecture="qwen4_exp",
        revision="fixture",
        config_hash=make_config_hash({"fixture": "qwen4_exp"}),
        tokenizer=TokenizerIdentity(identifier="fixture/tokenizer", revision="fixture"),
        dtype=DType.FLOAT32,
        device_map={"": "cpu"},
    )


def _tokens(count: int = 1):
    from moeatlas.events import TokenEvent, TokenPhase

    return tuple(
        TokenEvent(
            run_key="qwen4-exp-generic",
            sequence_id="sequence-1",
            token_pos=index,
            token_id=index + 10,
            token_text=str(index),
            phase=TokenPhase.PREFILL,
        )
        for index in range(count)
    )


def test_generic_scan_binds_one_shared_component_and_one_router_per_layer() -> None:
    model = Qwen4ExpHookableForConditionalGeneration(
        ([[1.0, 2.0, 3.0, 4.0]], [[0.7, 0.3]], [[3, 0]])
    )
    report = scan(model, _manifest())

    assert report.facts.expert_count == 4
    assert report.facts.routed_top_k == 2
    assert report.facts.shared_expert_count == 1
    shared_paths = [
        component.module_path
        for component in report.components
        if component.kind.value == "shared_expert"
    ]
    assert shared_paths == [
        "model.language_model.layers.0.mlp.shared_expert",
        "model.language_model.layers.1.mlp.shared_expert",
    ]
    assert all("shared_expert_gate" not in path for path in shared_paths)
    assert all("shared_expert.gate_proj" not in path for path in shared_paths)

    targets = structured_router_targets(report)
    assert [target.module_path for target in targets] == [
        "model.language_model.layers.0.mlp.gate",
        "model.language_model.layers.1.mlp.gate",
    ]


def test_packed_qwen4_tuple_decodes_routed_events_and_cleans_up_hooks() -> None:
    model = Qwen4ExpHookableForConditionalGeneration(
        ([[1.0, 2.0, 3.0, 4.0]], [[0.7, 0.3]], [[3, 0]])
    )
    report = scan(model, _manifest())
    result = run_structured_routing_forward(
        model,
        report,
        _tokens(),
        {"input_ids": [[10]]},
        max_events=16,
    )

    assert result.output is model.output
    assert model.calls == 1
    assert len(result.routing_events) == 4
    assert all(event.selected is True for event in result.routing_events)
    assert all(event.probability is not None for event in result.routing_events)
    assert not any("shared_expert" in event.expert_key for event in result.routing_events)
    assert all(not module._hooks for module in model._modules.values())  # type: ignore[attr-defined]


def test_packed_qwen4_is_routing_candidate_without_logical_expert_hooks() -> None:
    model = Qwen4ExpHookableForConditionalGeneration(
        ([[1.0, 2.0, 3.0, 4.0]], [[0.7, 0.3]], [[3, 0]])
    )
    report = scan(model, _manifest())

    support = classify_capture_support(report)
    assert support.grade == "routing_candidate"
    assert support.routing_capture == "candidate"
    assert support.expert_activity_capture == "unavailable"
    assert support.router_target_count == 2
    assert support.expert_target_count == 0
    assert any(
        "packed" in limitation or "expert component" in limitation
        for limitation in support.limitations
    )
