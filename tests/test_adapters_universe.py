"""Model-free tests for adapter-published routing universe contracts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from moeatlas.adapters import (
    ROUTING_UNIVERSE_SCHEMA_VERSION,
    AdapterDescriptor,
    AdapterInspection,
    LayerRoutingUniverse,
    RectangularProjection,
    RoutingUniverse,
    RoutingUniverseError,
    project_rectangular_universe,
    publish_routing_universe,
)
from moeatlas.core import (
    ComponentKind,
    make_component_key,
    make_model_key,
)
from moeatlas.discovery import DiscoveryFacts, DiscoveryReport

from .test_mixtral_routing_decoder import _inspection as _mixtral_inspection
from .test_qwen3_5_routing_decoder import _inspection as _qwen35_inspection

ROOT = Path(__file__).resolve().parents[1]

_MODEL_KEY = make_model_key("acme/nonrect", "r1")


def _expert_key(layer: int, index: int) -> str:
    return make_component_key(
        _MODEL_KEY,
        "expert",
        f"blocks.l{layer}.experts.{index}",
        layer_index=layer,
        expert_index=index,
    )


def _moe_layer_key(layer: int) -> str:
    return make_component_key(
        _MODEL_KEY, "moe_layer", f"blocks.l{layer}", layer_index=layer
    )


def _router_key(layer: int) -> str:
    return make_component_key(
        _MODEL_KEY, "router", f"blocks.l{layer}.gate", layer_index=layer
    )


def _shared_key(layer: int, index: int) -> str:
    return make_component_key(
        _MODEL_KEY,
        "shared_expert",
        f"blocks.l{layer}.shared.{index}",
        layer_index=layer,
    )


def _layer(
    index: int,
    width: int,
    *,
    top_k: int = 1,
    shared: int = 0,
    indices: tuple[int, ...] | None = None,
) -> LayerRoutingUniverse:
    return LayerRoutingUniverse(
        layer_index=index,
        moe_layer_key=_moe_layer_key(index),
        router_key=_router_key(index),
        expert_keys=tuple(sorted(_expert_key(index, i) for i in range(width))),
        expert_indices=None if indices is None else tuple(indices),
        routed_top_k=top_k,
        shared_expert_keys=tuple(sorted(_shared_key(index, i) for i in range(shared))),
    )


def _universe(**overrides: object) -> RoutingUniverse:
    """A deliberately non-rectangular two-layer universe.

    Widths (4/6), top-k schedules (2/3), sparse native indices on the second
    layer, and a layout tag outside the historical allowlist are all legal
    shapes for the contract.
    """

    values: dict[str, object] = {
        "model_key": _MODEL_KEY,
        "adapter_name": "fake-universe-adapter",
        "adapter_version": "1.2.3",
        "layout": "custom_sparse",
        "layers": (
            _layer(0, 4, top_k=2, indices=(0, 1, 2, 3)),
            _layer(1, 6, top_k=3, indices=(7, 0, 2, 5, 9, 1)),
        ),
    }
    values.update(overrides)
    return RoutingUniverse(**values)  # type: ignore[arg-type]


def _rebuild_inspection(
    inspection: AdapterInspection,
    *,
    descriptor: AdapterDescriptor | None = None,
    router_metadata: dict[int, dict[str, object]] | None = None,
    facts: DiscoveryFacts | None = None,
    extra_expert_index: int | None = None,
    extra_expert_layer: int = 0,
) -> AdapterInspection:
    """Rebuild an inspection with patched descriptor/captures/facts/components."""

    desc = descriptor if descriptor is not None else inspection.descriptor
    report = inspection.report
    component_data = [component.to_dict() for component in report.components]
    for data in component_data:
        capture = data.get("capture")
        if capture is None:
            continue
        capture = dict(capture)
        capture["adapter"] = desc.name
        capture["adapter_version"] = desc.version
        if (
            ComponentKind(data["kind"]) is ComponentKind.ROUTER
            and router_metadata
            and data["layer_index"] in router_metadata
        ):
            capture["metadata"] = {
                **capture["metadata"],
                **router_metadata[data["layer_index"]],
            }
        data["capture"] = capture
    if extra_expert_index is not None:
        template = next(
            data
            for data in component_data
            if ComponentKind(data["kind"]) is ComponentKind.EXPERT
            and data["layer_index"] == 0
        )
        clone = dict(template)
        clone["expert_index"] = extra_expert_index
        clone["layer_index"] = extra_expert_layer
        clone["module_path"] = (
            f"blocks.l{extra_expert_layer}.experts.{extra_expert_index}"
        )
        clone["component_key"] = make_component_key(
            report.model_key,
            "expert",
            clone["module_path"],
            layer_index=extra_expert_layer,
            expert_index=extra_expert_index,
        )
        candidate_template = next(
            candidate.model_dump(mode="json")
            for candidate in report.candidates
            if candidate.component_key == template["component_key"]
        )
        candidate = dict(candidate_template)
        candidate["component_key"] = clone["component_key"]
        candidate["module_path"] = clone["module_path"]
        candidate["layer_index"] = extra_expert_layer
        candidate["expert_index"] = extra_expert_index
        component_data.append(clone)
        candidate_data = [
            item.model_dump(mode="json") for item in report.candidates
        ] + [candidate]
    else:
        candidate_data = [item.model_dump(mode="json") for item in report.candidates]
    report_data = report.to_dict()
    report_data["components"] = component_data
    report_data["candidates"] = candidate_data
    if facts is not None:
        report_data["facts"] = facts.model_dump(mode="json")
    rebuilt_report = DiscoveryReport.model_validate(report_data)
    return AdapterInspection(descriptor=desc, detection=inspection.detection, report=rebuilt_report)


# ---------------------------------------------------------------------------
# Publication from real adapter inspections
# ---------------------------------------------------------------------------


def test_publish_mixtral_universes_and_round_trip() -> None:
    for layout, tag in (("legacy", "legacy_indexed"), ("packed", "packed")):
        universe = publish_routing_universe(_mixtral_inspection(layout))
        assert universe.manifest_type == "routing_universe"
        assert universe.schema_version == ROUTING_UNIVERSE_SCHEMA_VERSION
        assert universe.layout == tag
        assert universe.layer_indices == (0, 1)
        assert all(len(layer.expert_keys) == 4 for layer in universe.layers)
        assert all(layer.routed_top_k == 2 for layer in universe.layers)
        assert all(layer.shared_expert_keys == () for layer in universe.layers)
        assert sorted(universe.layers[0].expert_indices or ()) == [0, 1, 2, 3]
        assert RoutingUniverse.from_json(universe.to_json()) == universe


def test_publish_qwen35_universe_includes_shared_experts() -> None:
    universe = publish_routing_universe(_qwen35_inspection())
    assert universe.adapter_name == "huggingface-qwen3.5-moe-static"
    assert universe.layer_indices == (0, 1)
    for layer in universe.layers:
        assert len(layer.shared_expert_keys) == 1
        assert set(layer.shared_expert_keys).isdisjoint(layer.expert_keys)


def test_unknown_family_descriptor_publishes() -> None:
    base = _mixtral_inspection("legacy")
    unknown = _rebuild_inspection(
        base,
        descriptor=AdapterDescriptor(
            name="acme-unknown-adapter",
            version="9.9.9",
            architecture_families=("acme-unknown-moe-v3",),
        ),
    )
    universe = publish_routing_universe(unknown)
    assert universe.adapter_name == "acme-unknown-adapter"
    assert universe.adapter_version == "9.9.9"
    assert universe.model_key == base.report.model_key


def test_per_layer_top_k_metadata_publishes_variable_schedule() -> None:
    base = _mixtral_inspection("legacy")
    # With the global fact removed, each router must declare its own top-k;
    # declaring different values publishes a genuinely variable schedule.
    variable = _rebuild_inspection(
        base,
        facts=DiscoveryFacts(),
        router_metadata={0: {"routed_top_k": 2}, 1: {"routed_top_k": 1}},
    )
    universe = publish_routing_universe(variable)
    assert [layer.routed_top_k for layer in universe.layers] == [2, 1]


def test_metadata_top_k_conflicting_with_facts_is_rejected() -> None:
    base = _mixtral_inspection("legacy")
    conflicting = _rebuild_inspection(
        base, router_metadata={0: {"routed_top_k": 3}}
    )
    with pytest.raises(RoutingUniverseError, match="does not match inspection facts") as exc:
        publish_routing_universe(conflicting)
    assert exc.value.stage == "publication"


def test_missing_top_k_provenance_is_rejected() -> None:
    base = _mixtral_inspection("legacy")
    stripped = _rebuild_inspection(base, facts=DiscoveryFacts())
    with pytest.raises(RoutingUniverseError, match="routed_top_k provenance") as exc:
        publish_routing_universe(stripped)
    assert exc.value.stage == "publication"


def test_inconsistent_router_layouts_are_rejected() -> None:
    base = _mixtral_inspection("legacy")
    mixed = _rebuild_inspection(base, router_metadata={1: {"layout": "packed"}})
    with pytest.raises(RoutingUniverseError, match="layouts are inconsistent") as exc:
        publish_routing_universe(mixed)
    assert exc.value.stage == "publication"


def test_router_metadata_extra_keys_are_rejected() -> None:
    base = _mixtral_inspection("legacy")
    polluted = _rebuild_inspection(base, router_metadata={0: {"secret": 1}})
    with pytest.raises(RoutingUniverseError, match="layout provenance") as exc:
        publish_routing_universe(polluted)
    assert exc.value.stage == "publication"


def test_uncovered_routed_expert_is_rejected() -> None:
    base = _mixtral_inspection("legacy")
    # Facts are stripped and top-k provenance moves onto the routers so the
    # coverage invariant is the one that fails, not top-k availability.
    padded = _rebuild_inspection(
        base,
        facts=DiscoveryFacts(),
        router_metadata={0: {"routed_top_k": 2}, 1: {"routed_top_k": 2}},
        extra_expert_index=9,
        extra_expert_layer=7,
    )
    with pytest.raises(RoutingUniverseError, match="full routed expert universe") as exc:
        publish_routing_universe(padded)
    assert exc.value.stage == "publication"


def test_publication_requires_exact_inspection_type() -> None:
    with pytest.raises(RoutingUniverseError, match="exact AdapterInspection") as exc:
        publish_routing_universe("not an inspection")
    assert exc.value.stage == "dependency"
    assert str(exc.value).startswith("routing universe failed at dependency")


# ---------------------------------------------------------------------------
# Contract-level non-rectangular shapes
# ---------------------------------------------------------------------------


def test_nonrectangular_universe_constructs_and_serializes() -> None:
    universe = _universe()
    widths = [len(layer.expert_keys) for layer in universe.layers]
    top_ks = [layer.routed_top_k for layer in universe.layers]
    assert widths == [4, 6]
    assert top_ks == [2, 3]
    assert universe.layer_indices == (0, 1)
    assert universe.layout == "custom_sparse"
    assert RoutingUniverse.from_json(universe.to_json()) == universe


def test_gap_layers_and_sparse_native_indices_are_allowed() -> None:
    universe = _universe(
        layers=(
            _layer(0, 3, indices=(0, 2, 4)),
            _layer(5, 3, indices=(11, 4, 0)),
        )
    )
    assert universe.layer_indices == (0, 5)
    assert universe.layers[1].expert_indices == (11, 4, 0)
    assert RoutingUniverse.from_json(universe.to_json()) == universe


def test_layer_field_validation() -> None:
    with pytest.raises(ValidationError, match="must not exceed"):
        _layer(0, 2, top_k=3)
    with pytest.raises(ValidationError, match="sorted ascending"):
        LayerRoutingUniverse(
            layer_index=0,
            moe_layer_key=_moe_layer_key(0),
            router_key=_router_key(0),
            expert_keys=tuple(reversed(_layer(0, 2).expert_keys)),
            routed_top_k=1,
        )
    with pytest.raises(ValidationError, match="disjoint"):
        LayerRoutingUniverse(
            layer_index=0,
            moe_layer_key=_moe_layer_key(0),
            router_key=_router_key(0),
            expert_keys=_layer(0, 2).expert_keys,
            routed_top_k=1,
            shared_expert_keys=(_layer(0, 2).expert_keys[0],),
        )
    with pytest.raises(ValidationError, match="parallel"):
        _layer(0, 3, indices=(0, 1))
    with pytest.raises(ValidationError, match="must be unique"):
        _layer(0, 3, indices=(0, 0, 1))
    with pytest.raises(ValidationError, match="non-negative exact integers"):
        _layer(0, 2, indices=(-1, 0))


def test_universe_field_validation() -> None:
    layers_two = (_layer(0, 2), _layer(1, 2))
    with pytest.raises(ValidationError, match="sorted ascending by layer_index"):
        _universe(layers=(layers_two[1], layers_two[0]))
    with pytest.raises(ValidationError, match="layer_index values must be unique"):
        _universe(layers=(_layer(0, 2), _layer(0, 3)))
    with pytest.raises(ValidationError, match="globally unique"):
        _universe(
            layers=(
                _layer(0, 2),
                LayerRoutingUniverse(
                    layer_index=1,
                    moe_layer_key=_moe_layer_key(1),
                    router_key=_router_key(1),
                    expert_keys=_layer(0, 2).expert_keys,
                    routed_top_k=1,
                ),
            )
        )
    with pytest.raises(ValidationError, match="at least 1"):
        _universe(layers=())
    with pytest.raises(ValidationError, match="non-empty and trimmed"):
        _universe(layout="  ")
    with pytest.raises(ValidationError, match="at most 64 characters"):
        _universe(layout="x" * 65)
    with pytest.raises(ValidationError, match="control characters"):
        _universe(layout="bad\x00layout")
    with pytest.raises(ValidationError, match="model_key"):
        _universe(model_key="not a model key")


# ---------------------------------------------------------------------------
# Rectangular projection
# ---------------------------------------------------------------------------


def test_rectangular_projection_of_published_universe() -> None:
    universe = publish_routing_universe(_mixtral_inspection("legacy"))
    projection = project_rectangular_universe(universe)
    assert isinstance(projection, RectangularProjection)
    assert projection.expert_count == 4
    assert projection.routed_top_k == 2
    assert projection.layer_indices == (0, 1)
    assert projection.layer_keys == tuple(layer.moe_layer_key for layer in universe.layers)
    assert projection.expert_keys == tuple(layer.expert_keys for layer in universe.layers)


def test_projection_rejects_variable_widths_and_top_k() -> None:
    with pytest.raises(RoutingUniverseError, match="expert counts vary"):
        project_rectangular_universe(_universe())
    uniform_width = _universe(
        layers=(_layer(0, 4, top_k=2), _layer(1, 4, top_k=3))
    )
    with pytest.raises(RoutingUniverseError, match="top-k schedules vary"):
        project_rectangular_universe(uniform_width)


def test_projection_rejects_non_contiguous_layers() -> None:
    gapped = _universe(layers=(_layer(0, 4), _layer(5, 4)))
    with pytest.raises(RoutingUniverseError, match="not contiguous from zero"):
        project_rectangular_universe(gapped)


def test_projection_requires_native_expert_indices() -> None:
    unindexed = _universe(layers=(_layer(0, 4), _layer(1, 4)))
    with pytest.raises(RoutingUniverseError, match="does not declare native expert indices"):
        project_rectangular_universe(unindexed)


def test_projection_rejects_non_contiguous_native_indices() -> None:
    sparse = _universe(layers=(_layer(0, 3, indices=(0, 2, 4)), _layer(1, 3, indices=(0, 1, 2))))
    with pytest.raises(RoutingUniverseError, match="not contiguous"):
        project_rectangular_universe(sparse)


def test_projection_requires_exact_universe_type() -> None:
    with pytest.raises(TypeError, match="exact RoutingUniverse"):
        project_rectangular_universe({"not": "a universe"})


# ---------------------------------------------------------------------------
# Error contract and isolation guards
# ---------------------------------------------------------------------------


def test_error_stage_contract() -> None:
    assert str(RoutingUniverseError("projection")) == (
        "routing universe failed at projection"
    )
    error = RoutingUniverseError("publication", "boom")
    assert error.stage == "publication"
    assert str(error) == "routing universe failed at publication: boom"
    with pytest.raises(ValueError, match="unsupported routing universe stage"):
        RoutingUniverseError("bogus")


def test_universe_import_without_model_stack() -> None:
    script = (
        "import sys\n"
        "for name in ('torch', 'transformers', 'duckdb', 'safetensors'):\n"
        "    sys.modules[name] = None\n"
        "import moeatlas.adapters.universe\n"
        "print('universe-import-ok')\n"
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
    assert "universe-import-ok" in completed.stdout


def test_no_model_runtime_imported_by_universe() -> None:
    assert not any(
        name in sys.modules for name in ("torch", "transformers", "safetensors")
    )
