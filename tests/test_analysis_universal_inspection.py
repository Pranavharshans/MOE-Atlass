from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from moeatlas.adapters import (
    UniversalLayerUniverse,
    UniversalRoutingInspection,
    build_universal_inspection,
)
from moeatlas.analysis import (
    RoutingLoadError,
    RoutingLoadMatrix,
    aggregate_routing_load,
    render_routing_load_heatmap,
    summarize_routing_load,
)
from moeatlas.core import (
    CapabilityLabel,
    ComponentKind,
    ComponentManifest,
    DType,
    ModelManifest,
    TokenizerIdentity,
    make_component_key,
    make_config_hash,
    make_model_key,
)
from moeatlas.discovery import (
    DiscoveryCandidate,
    DiscoveryEvidence,
    DiscoveryFacts,
    DiscoveryReport,
    DiscoverySignal,
    scan,
)
from moeatlas.events import RoutingEvent, TokenEvent, TokenPhase
from moeatlas.runtime import RoutingForwardResult
from moeatlas.store import append_routing_shard

from .test_cli_scan import _loading_manifest, _loading_plan
from .test_runtime_generic_capture import LingNamedModel

try:
    import duckdb  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised without the store extra
    duckdb = None

ROOT = Path(__file__).resolve().parents[1]


def _component(
    model_key: str,
    kind: ComponentKind,
    module_path: str,
    *,
    layer_index: int | None,
    expert_index: int | None = None,
    routed: bool | None = None,
    shared: bool | None = None,
) -> ComponentManifest:
    return ComponentManifest(
        component_key=make_component_key(
            model_key,
            kind.value,
            module_path,
            layer_index=layer_index,
            expert_index=expert_index,
        ),
        model_key=model_key,
        kind=kind,
        module_path=module_path,
        layer_index=layer_index,
        expert_index=expert_index,
        capabilities=[CapabilityLabel.STRUCTURE],
        routed=routed,
        shared=shared,
    )


def _generic_report(
    *,
    model_id: str = "acme/bailing-hybrid",
    architecture: str = "bailing_hybrid",
    layers: int = 2,
    experts: int = 4,
    top_k: int = 2,
) -> DiscoveryReport:
    """A model-free foreign-family STRUCTURE report (no certified adapter)."""

    model_key = make_model_key(model_id, "r1")
    components = [
        _component(
            model_key,
            ComponentKind.MOE_LAYER,
            f"model.layers.{index}",
            layer_index=index,
        )
        for index in range(layers)
    ]
    components.extend(
        _component(
            model_key,
            ComponentKind.ROUTER,
            f"model.layers.{index}.gate",
            layer_index=index,
        )
        for index in range(layers)
    )
    components.extend(
        _component(
            model_key,
            ComponentKind.EXPERT,
            f"model.layers.{layer}.experts.{expert}",
            layer_index=layer,
            expert_index=expert,
            routed=True,
            shared=False,
        )
        for layer in range(layers)
        for expert in range(experts)
    )

    def candidate(component: ComponentManifest) -> DiscoveryCandidate:
        return DiscoveryCandidate(
            component_key=component.component_key,
            model_key=model_key,
            kind=component.kind,
            module_path=component.module_path,
            layer_index=component.layer_index,
            expert_index=component.expert_index,
            confidence=0.8,
            evidence=[
                DiscoveryEvidence(
                    signal=DiscoverySignal.PATH_NAME,
                    detail=f"module path marks {component.kind.value}",
                    weight=0.8,
                )
            ],
            routed=component.routed,
            shared=component.shared,
        )

    return DiscoveryReport(
        model_key=model_key,
        model_manifest=ModelManifest(
            model_key=model_key,
            architecture=architecture,
            revision="r1",
            config_hash=make_config_hash({"architecture": architecture}),
            tokenizer=TokenizerIdentity(identifier="acme/generic-tokenizer", revision="r1"),
            dtype=DType.FLOAT32,
            device_map={"": "cpu"},
        ),
        scanner_version="0.0.0",
        facts=DiscoveryFacts(
            expert_count=experts,
            expert_count_source="config.num_local_experts",
            routed_top_k=top_k,
            routed_top_k_source="config.num_experts_per_tok",
        ),
        candidates=[candidate(component) for component in components],
        components=components,
    )


def _universal(report: DiscoveryReport | None = None) -> UniversalRoutingInspection:
    return build_universal_inspection(report or _generic_report())


def _events(document: UniversalRoutingInspection, run_key: str) -> RoutingForwardResult:
    layer_keys = [layer.layer_key for layer in document.layers]
    expert_keys = [list(layer.expert_keys) for layer in document.layers]
    token_events = tuple(
        TokenEvent(
            run_key=run_key,
            sequence_id="sequence-1",
            token_pos=pos,
            token_id=100 + pos,
            token_text=str(pos),
            phase=TokenPhase.PREFILL,
        )
        for pos in range(3)
    )
    routing_events: list[RoutingEvent] = []
    for layer_position, layer_key in enumerate(layer_keys):
        for token in token_events:
            offset = token.token_pos % len(expert_keys[layer_position])
            for rank in range(document.routed_top_k):
                expert_key = expert_keys[layer_position][(offset + rank) % len(expert_keys)]
                routing_events.append(
                    RoutingEvent(
                        token_key=token.token_key,
                        layer_key=layer_key,
                        rank=rank,
                        expert_key=expert_key,
                        probability=0.5,
                        selected=True,
                    )
                )
    return RoutingForwardResult(object(), token_events, tuple(routing_events))


def _ling_report() -> DiscoveryReport:
    model = LingNamedModel([[1.0 - 0.05 * index for index in range(8)]])
    return scan(model, _loading_manifest(_loading_plan()))


def test_universal_builder_accepts_a_noisy_self_consistent_foreign_scan() -> None:
    report = _ling_report()
    routers = [c for c in report.components if c.kind.value == "router"]
    assert len(routers) > 2
    moe_layers = [c for c in report.components if c.kind.value == "moe_layer"]
    assert len(moe_layers) > 2
    document = build_universal_inspection(report)
    assert type(document) is UniversalRoutingInspection
    assert document.expert_count == 8
    assert document.routed_top_k == 2
    assert [layer.layer_index for layer in document.layers] == [0, 1]
    expected_keys = {
        component.component_key
        for component in report.components
        if component.kind.value == "moe_layer" and component.module_path.endswith(".mlp")
    }
    assert {layer.layer_key for layer in document.layers} == expected_keys


def test_universal_builder_numbers_nonzero_origin_stacks_ordinally() -> None:
    report = _generic_report()
    model_key = report.model_key

    def shifted(item: DiscoveryCandidate | object):
        index = item.layer_index
        if index is None:
            return item
        new_index = index + 1
        module_path = item.module_path.replace(
            f".layers.{index}", f".layers.{new_index}", 1
        )
        key = make_component_key(
            model_key,
            item.kind.value,
            module_path,
            layer_index=new_index,
            expert_index=item.expert_index,
        )
        return item.model_copy(
            update={"component_key": key, "module_path": module_path, "layer_index": new_index}
        )

    moved = report.model_copy(
        update={
            "components": [shifted(component) for component in report.components],
            "candidates": [shifted(candidate) for candidate in report.candidates],
        }
    )
    document = build_universal_inspection(moved)
    assert [layer.layer_index for layer in document.layers] == [0, 1]
    assert document.expert_count == 4


def test_universal_builder_tolerates_name_token_noise_on_consistent_reports() -> None:
    report = _generic_report()
    twin = _component(
        report.model_key,
        ComponentKind.ROUTER,
        "model.layers.0.gate-twin",
        layer_index=0,
    )
    twin_candidate = DiscoveryCandidate(
        component_key=twin.component_key,
        model_key=report.model_key,
        kind=twin.kind,
        module_path=twin.module_path,
        layer_index=twin.layer_index,
        expert_index=twin.expert_index,
        confidence=0.8,
        evidence=[
            DiscoveryEvidence(
                signal=DiscoverySignal.PATH_NAME,
                detail="module path marks router",
                weight=0.8,
            )
        ],
    )
    noisy = report.model_copy(
        update={
            "components": [*report.components, twin],
            "candidates": [*report.candidates, twin_candidate],
        }
    )
    document = build_universal_inspection(noisy)
    assert tuple(layer.layer_index for layer in document.layers) == (0, 1)
    assert document.expert_count == 4


def test_universal_document_derives_from_a_foreign_family_report() -> None:
    report = _generic_report()
    document = build_universal_inspection(report)
    assert type(document) is UniversalRoutingInspection
    assert document.manifest_type == "universal_routing_inspection"
    assert document.provenance == "universal"
    assert document.model_key == report.model_key
    assert document.architecture_families == ("bailing_hybrid",)
    assert document.layout == "packed"
    assert document.routed_top_k == 2
    assert document.expert_count == 4
    assert tuple(layer.layer_index for layer in document.layers) == (0, 1)
    assert document.axes_digest.startswith("sha256:")
    assert document.scanner_version == report.scanner_version


def test_universal_document_accepts_proven_packed_expert_axes() -> None:
    from .test_runtime_generic_capture import _HookedRouter

    class PackedMoE:
        pass

    class PackedExperts:
        pass

    class PackedModel:
        config = {"num_experts": 4, "num_experts_per_tok": 2}

        def __init__(self) -> None:
            self.gate = _HookedRouter(
                parameters={"weight": type("P", (), {"shape": (4, 8)})()}
            )

        def named_modules(self):
            yield "", self
            yield "layers.0.moe", PackedMoE()
            yield "layers.0.moe.gate", self.gate
            yield "layers.0.moe.experts", PackedExperts()

        def named_parameters(self):
            yield "layers.0.moe.gate.weight", type("P", (), {"shape": (4, 8)})()
            yield "layers.0.moe.experts.w1", type("P", (), {"shape": (4, 16, 8)})()

    report = scan(PackedModel(), _loading_manifest(_loading_plan()))
    document = build_universal_inspection(report)
    assert document.layout == "packed"
    assert document.expert_count == 4
    assert len(document.layers) == 1
    assert len(document.layers[0].expert_keys) == 4


def test_universal_construction_rejects_wrong_types_and_incomplete_facts() -> None:
    with pytest.raises(TypeError, match="exact DiscoveryReport"):
        build_universal_inspection(object())
    report = _generic_report()
    empty_facts = report.model_copy(update={"facts": DiscoveryFacts()})
    with pytest.raises(ValueError, match="not complete routing facts"):
        build_universal_inspection(empty_facts)


def test_universal_builder_rejects_broken_structures() -> None:
    report = _generic_report()

    def without_components(kind: ComponentKind) -> DiscoveryReport:
        survivors = [
            item.model_copy()
            for item in report.components
            if item.kind is not kind
        ]
        candidates = [
            item.model_copy()
            for item in report.candidates
            if item.component_key in {item.component_key for item in survivors}
        ]
        return report.model_copy(update={"components": survivors, "candidates": candidates})

    with pytest.raises(ValueError, match="no router universe"):
        build_universal_inspection(without_components(ComponentKind.ROUTER))
    unrouted = report.model_copy(
        update={
            "components": [
                (
                    item.model_copy(update={"routed": False})
                    if item.kind is ComponentKind.EXPERT
                    else item.model_copy()
                )
                for item in report.components
            ]
        }
    )
    candidates = {
        item.component_key: item.model_copy(update={"routed": False})
        if item.kind is ComponentKind.EXPERT
        else item.model_copy()
        for item in report.candidates
    }
    unrouted = unrouted.model_copy(update={"candidates": list(candidates.values())})
    with pytest.raises(ValueError, match="shared or unrouted experts"):
        build_universal_inspection(unrouted)

    duplicate_layer = _generic_report(layers=1)
    # A second trusted router claiming the same routed layer is genuinely
    # contradictory; name-token noise (e.g. a "gate-twin" leaf) is not.
    twin = _component(
        duplicate_layer.model_key,
        ComponentKind.ROUTER,
        "model.layers.0.extra.gate",
        layer_index=0,
    )
    twin_candidate = DiscoveryCandidate(
        component_key=twin.component_key,
        model_key=duplicate_layer.model_key,
        kind=twin.kind,
        module_path=twin.module_path,
        layer_index=twin.layer_index,
        expert_index=twin.expert_index,
        confidence=0.8,
        evidence=[
            DiscoveryEvidence(
                signal=DiscoverySignal.PATH_NAME,
                detail="module path marks router",
                weight=0.8,
            )
        ],
    )
    with pytest.raises(ValueError, match="router layer indices are not unique"):
        build_universal_inspection(
            duplicate_layer.model_copy(
                update={
                    "components": [*duplicate_layer.components, twin],
                    "candidates": [*duplicate_layer.candidates, twin_candidate],
                }
            )
        )


def test_universal_document_round_trips_canonically() -> None:
    document = _universal()
    payload = document.to_json()
    assert payload == document.to_json()
    restored = UniversalRoutingInspection.from_json(payload)
    assert type(restored) is UniversalRoutingInspection
    assert restored == document
    assert restored.to_json() == payload
    parsed = json.loads(payload)
    assert parsed["manifest_type"] == "universal_routing_inspection"
    assert parsed["provenance"] == "universal"
    assert parsed["layout"] == "packed"


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"axes_digest": "sha256:" + "0" * 64}, "axes_digest does not match"),
        ({"manifest_type": "adapter_inspection"}, "manifest_type"),
        ({"provenance": "certified"}, "provenance"),
        ({"layout": "legacy_indexed"}, "layout"),
        ({"unexpected": True}, "extra_forbidden"),
        ({"schema_version": "9.9"}, "schema_version"),
    ],
)
def test_universal_document_tampering_is_rejected(
    mutation: dict[str, object], match: str
) -> None:
    document = _universal()
    payload = json.loads(document.to_json())
    payload.update(mutation)
    with pytest.raises(Exception, match=match):
        UniversalRoutingInspection.from_json(json.dumps(payload))


def test_axis_tampering_is_rejected_by_the_axes_digest() -> None:
    document = _universal()
    payload = json.loads(document.to_json())
    payload["layers"][0]["expert_keys"] = list(reversed(payload["layers"][0]["expert_keys"]))
    with pytest.raises(ValueError, match="axes_digest does not match"):
        UniversalRoutingInspection.from_json(json.dumps(payload))
    payload = json.loads(document.to_json())
    payload["layers"][1]["layer_index"] = 5
    with pytest.raises(ValueError, match="contiguous"):
        UniversalRoutingInspection.from_json(json.dumps(payload))
    payload = json.loads(document.to_json())
    payload["routed_top_k"] = 99
    with pytest.raises(ValueError, match="cannot exceed"):
        UniversalRoutingInspection.from_json(json.dumps(payload))


def test_aggregate_rejects_unknown_and_tampered_universal_documents(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(RoutingLoadError) as wrong_type:
        aggregate_routing_load(
            workspace,
            object(),
            run_key="run-1",
            max_routing_rows=10,
            max_source_bytes=10_000,
            max_matrix_cells=100,
        )
    assert wrong_type.value.stage == "inspection"
    assert wrong_type.value.__cause__ is not None

    document = _universal()
    payload = json.loads(document.to_json())
    tampered = UniversalRoutingInspection.model_construct(
        **{
            **payload,
            "architecture_families": tuple(payload["architecture_families"]),
            "axes_digest": "sha256:" + "0" * 64,
            "layers": tuple(
                UniversalLayerUniverse.model_validate(layer)
                for layer in payload["layers"]
            ),
        }
    )
    with pytest.raises(RoutingLoadError) as tampered_error:
        aggregate_routing_load(
            workspace,
            tampered,
            run_key="run-1",
            max_routing_rows=10,
            max_source_bytes=10_000,
            max_matrix_cells=100,
        )
    assert tampered_error.value.stage == "inspection"


def test_aggregate_enforces_the_matrix_cell_budget_for_universal_documents(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    document = _universal()
    with pytest.raises(RoutingLoadError) as budget:
        aggregate_routing_load(
            workspace,
            document,
            run_key="run-1",
            max_routing_rows=10,
            max_source_bytes=10_000,
            max_matrix_cells=7,
        )
    assert budget.value.stage == "budget"


def test_declared_universes_require_a_certified_inspection(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    document = _universal()
    with pytest.raises(RoutingLoadError) as declared:
        aggregate_routing_load(
            workspace,
            document,
            run_key="run-1",
            max_routing_rows=10,
            max_source_bytes=10_000,
            max_matrix_cells=100,
            declared_universe=object(),
        )
    assert declared.value.stage == "inspection"


@pytest.mark.skipif(duckdb is None, reason="duckdb store extra is unavailable")
def test_generic_family_shards_aggregate_and_render_all_three_metrics(
    tmp_path: Path,
) -> None:
    document = _universal()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    receipt = append_routing_shard(workspace, _events(document, receipt_run := "run-1"))
    assert receipt.run_key == receipt_run

    matrix = aggregate_routing_load(
        workspace,
        document,
        run_key=receipt.run_key,
        max_routing_rows=10_000,
        max_source_bytes=10_000_000,
        max_matrix_cells=10_000,
    )
    assert type(matrix) is RoutingLoadMatrix
    assert matrix.run_key == receipt.run_key
    assert matrix.model_key == document.model_key
    assert matrix.adapter_name == "universal"
    assert matrix.layout == "packed"
    assert matrix.routed_top_k == document.routed_top_k
    assert matrix.token_count == 3
    assert matrix.assignment_count == 3 * 2 * document.routed_top_k
    assert matrix.layer_keys == tuple(layer.layer_key for layer in document.layers)
    assert matrix.expert_keys == tuple(
        tuple(layer.expert_keys) for layer in document.layers
    )
    summary = summarize_routing_load(matrix, max_cells=10_000)
    assert summary.dead_expert_fraction >= 0.0

    for metric in ("assignment_counts", "assignment_shares", "load_ratios"):
        html = render_routing_load_heatmap(matrix, metric=metric, max_cells=10_000)
        assert html.startswith("<!doctype html>")
        assert html.count("<table") == 1
        assert html.count("<td") == 8
        assert "Layer × Expert routing-load heatmap" in html
        assert document.model_key in html


def _heatmap_command(
    workspace: Path,
    inspection: Path,
    output: Path,
    *,
    run_key: str,
    metric: str = "load_ratios",
) -> list[str]:
    return [
        "heatmap",
        str(workspace),
        "--inspection",
        str(inspection),
        "--run-key",
        run_key,
        "--metric",
        metric,
        "--max-inspection-bytes",
        "1000000",
        "--max-routing-rows",
        "1000000",
        "--max-source-bytes",
        "100000000",
        "--max-matrix-cells",
        "100000",
        "--output",
        str(output),
    ]


def _compare_command(
    workspace: Path,
    inspection: Path,
    output: Path,
    *,
    baseline: str,
    comparison: str,
) -> list[str]:
    return [
        "compare",
        str(workspace),
        "--inspection",
        str(inspection),
        "--baseline-run-key",
        baseline,
        "--comparison-run-key",
        comparison,
        "--metric",
        "ratio_deltas",
        "--max-inspection-bytes",
        "1000000",
        "--max-routing-rows",
        "1000000",
        "--max-source-bytes",
        "100000000",
        "--max-matrix-cells",
        "100000",
        "--output",
        str(output),
    ]


@pytest.mark.skipif(duckdb is None, reason="duckdb store extra is unavailable")
def test_heatmap_cli_renders_a_generic_family_workspace(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from moeatlas.cli import main

    document = _universal()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    receipt = append_routing_shard(workspace, _events(document, "run-1"))
    inspection = tmp_path / "universal-inspection.json"
    inspection.write_text(document.to_json(), encoding="utf-8")
    output = tmp_path / "routing-load.html"

    assert main(_heatmap_command(workspace, inspection, output, run_key=receipt.run_key)) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"saved routing heatmap to {output}\n"
    html = output.read_text(encoding="utf-8")
    assert html.count("<td") == 8
    assert "universal" in html


@pytest.mark.skipif(duckdb is None, reason="duckdb store extra is unavailable")
def test_compare_cli_accepts_one_universal_document_over_two_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from moeatlas.cli import main

    document = _universal()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    baseline_receipt = append_routing_shard(workspace, _events(document, "run-1"))
    comparison_receipt = append_routing_shard(workspace, _events(document, "run-2"))
    assert baseline_receipt.run_key != comparison_receipt.run_key
    inspection = tmp_path / "universal-inspection.json"
    inspection.write_text(document.to_json(), encoding="utf-8")
    output = tmp_path / "routing-comparison.html"

    assert (
        main(
            _compare_command(
                workspace,
                inspection,
                output,
                baseline=baseline_receipt.run_key,
                comparison=comparison_receipt.run_key,
            )
        )
        == 0
    )
    capsys.readouterr()
    html = output.read_text(encoding="utf-8")
    assert "<!doctype html>" in html
    assert baseline_receipt.run_key in html


def test_cli_rejects_an_invalid_universal_document_without_publication(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from moeatlas.cli import main

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    document = _universal()
    payload = json.loads(document.to_json())
    payload["layers"][0]["expert_keys"] = list(reversed(payload["layers"][0]["expert_keys"]))
    inspection = tmp_path / "tampered.json"
    inspection.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "output.html"

    assert main(_heatmap_command(workspace, inspection, output, run_key="run-1")) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "valid AdapterInspection or universal inspection document" in captured.err
    assert not output.exists()


def test_cli_help_names_both_accepted_inspection_documents(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from moeatlas.cli import main

    with pytest.raises(SystemExit) as caught:
        main(["heatmap", "--help"])
    assert caught.value.code == 0
    help_text = capsys.readouterr().out
    assert "AdapterInspection.to_json()" in help_text
    assert "universal_routing_inspection" in help_text

    with pytest.raises(SystemExit) as caught:
        main(["compare", "--help"])
    assert caught.value.code == 0
    help_text = capsys.readouterr().out
    assert "AdapterInspection.to_json()" in help_text
    assert "universal_routing_inspection" in help_text


def test_universal_module_stays_pure_and_non_networked() -> None:
    source = (ROOT / "src" / "moeatlas" / "adapters" / "universal.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    forbidden = {
        "torch",
        "transformers",
        "accelerate",
        "safetensors",
        "duckdb",
        "socket",
        "urllib",
        "requests",
        "webbrowser",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".", 1)[0] not in forbidden for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".", 1)[0] not in forbidden
    lowered = source.lower()
    for term in ("urlopen", "create_connection", "torch", "transformers", "duckdb"):
        assert term not in lowered
