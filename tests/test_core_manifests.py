from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from moeatlas.core import (
    CAPABILITY_LABELS,
    CAPABILITY_SEMANTICS,
    CapabilityLabel,
    CaptureProvenance,
    CaptureSource,
    ComponentKind,
    ComponentManifest,
    DType,
    ModelManifest,
    Provenance,
    TokenizerIdentity,
    capability_semantics,
    make_component_key,
    make_config_hash,
    make_model_key,
    parse_model_key,
)

ROOT = Path(__file__).resolve().parents[1]


def model_manifest() -> ModelManifest:
    model_key = make_model_key("acme/demo-moe", "main")
    return ModelManifest(
        model_key=model_key,
        architecture="demo_moe",
        revision="main",
        config_hash=make_config_hash({"experts": 4, "top_k": 2}),
        tokenizer=TokenizerIdentity(identifier="acme/demo-tokenizer", revision="main"),
        dtype=DType.BFLOAT16,
        device_map={"": "cpu", "layers.0": "cpu"},
        provenance=Provenance(
            source="synthetic-fixture",
            tool_version="0.1.0",
            metadata={"fixture": "small-moe", "seed": 7},
        ),
    )


def component_manifest(capability: CapabilityLabel = CapabilityLabel.ROUTING) -> ComponentManifest:
    model_key = model_manifest().model_key
    capture = None
    if capability is CapabilityLabel.FULL:
        capture = CaptureProvenance(
            source=CaptureSource.MODULE_HOOK,
            method="router-hook",
            adapter="synthetic",
            adapter_version="0.1",
            verified=True,
        )
    return ComponentManifest(
        component_key=make_component_key(
            model_key,
            ComponentKind.EXPERT.value,
            "layers.0.experts.1",
            layer_index=0,
            expert_index=1,
        ),
        model_key=model_key,
        kind=ComponentKind.EXPERT,
        module_path="layers.0.experts.1",
        layer_index=0,
        expert_index=1,
        tensor_shapes={"w1": [8, 16], "w2": [16, 8]},
        capabilities=[capability],
        routed=True,
        capture=capture,
    )


def test_capability_labels_are_stable_and_documented() -> None:
    assert tuple(CAPABILITY_LABELS) == tuple(CapabilityLabel)
    assert {label.value for label in CAPABILITY_LABELS} == {
        "FULL",
        "ROUTING",
        "MODULE",
        "STRUCTURE",
        "EXPERIMENTAL",
        "UNSUPPORTED",
    }
    assert set(CAPABILITY_SEMANTICS) == set(CapabilityLabel)
    assert capability_semantics("ROUTING").startswith("Reliable router")
    with pytest.raises(ValueError, match="unknown capability label"):
        capability_semantics("semantic")


def test_model_manifest_json_round_trip_is_json_serializable() -> None:
    manifest = model_manifest()
    encoded = manifest.to_json()
    decoded = ModelManifest.from_json(encoded)

    assert decoded == manifest
    assert json.loads(encoded) == manifest.to_dict()
    assert manifest.to_dict()["schema_version"] == "1.0"
    assert manifest.to_dict()["dtype"] == "bfloat16"
    assert manifest.to_dict()["tokenizer"]["revision"] == "main"


def test_component_manifest_json_round_trip_preserves_capture_provenance() -> None:
    manifest = component_manifest(CapabilityLabel.FULL)
    decoded = ComponentManifest.from_json(manifest.to_json(indent=2))

    assert decoded == manifest
    assert decoded.capture is not None
    assert decoded.capture.source is CaptureSource.MODULE_HOOK
    assert decoded.to_dict()["capabilities"] == ["FULL"]


def test_schema_is_strict_and_rejects_unknown_fields_and_versions() -> None:
    values = model_manifest().to_dict()
    values["model_key"] = 42
    with pytest.raises(ValidationError, match="model_key"):
        ModelManifest.model_validate(values)

    values = model_manifest().to_dict()
    values["unexpected"] = True
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ModelManifest.model_validate(values)

    values = model_manifest().to_dict()
    values["schema_version"] = "2.0"
    with pytest.raises(ValidationError, match="schema_version"):
        ModelManifest.model_validate(values)


def test_component_invariants_produce_actionable_errors() -> None:
    values = component_manifest().to_dict()
    values["layer_index"] = "0"
    with pytest.raises(ValidationError, match="layer_index"):
        ComponentManifest.model_validate(values)

    values = component_manifest().to_dict()
    values["routed"] = True
    values["shared"] = True
    with pytest.raises(ValidationError, match="both routed and shared"):
        ComponentManifest.model_validate(values)

    values = component_manifest().to_dict()
    values["capabilities"] = ["UNSUPPORTED"]
    with pytest.raises(ValidationError, match="requires at least one warning"):
        ComponentManifest.model_validate(values)

    values = component_manifest().to_dict()
    values["capabilities"] = ["FULL"]
    with pytest.raises(ValidationError, match="verified capture provenance"):
        ComponentManifest.model_validate(values)


def test_model_key_format_and_revision_binding_are_enforced() -> None:
    values = model_manifest().to_dict()
    values["revision"] = "refs/main"
    with pytest.raises(ValidationError, match="does not match the revision encoded"):
        ModelManifest.model_validate(values)

    values = model_manifest().to_dict()
    values["model_key"] = "model:acme/demo-moe"
    with pytest.raises(ValidationError, match="model:<portable-id>@<revision>"):
        ModelManifest.model_validate(values)

    values = model_manifest().to_dict()
    values["model_key"] = "model:acme/demo-moe@main@extra"
    with pytest.raises(ValidationError, match="exactly one '@' separator"):
        ModelManifest.model_validate(values)

    with pytest.raises(ValueError, match="must not contain '@'"):
        make_model_key("acme/demo@moe", "main")
    with pytest.raises(ValueError, match="must not contain '@'"):
        make_model_key("acme/demo-moe", "main@extra")
    assert parse_model_key("model:acme/demo-moe@refs/main") == (
        "acme/demo-moe",
        "refs/main",
    )


def test_component_key_is_bound_to_component_identity() -> None:
    values = component_manifest().to_dict()
    values["component_key"] = "component:" + ("0" * 64)
    with pytest.raises(ValidationError, match="component_key does not match"):
        ComponentManifest.model_validate(values)


def test_full_and_experimental_capabilities_have_distinct_evidence_rules() -> None:
    values = component_manifest(CapabilityLabel.FULL).to_dict()
    values["capture"]["verified"] = False
    with pytest.raises(ValidationError, match="verified capture provenance"):
        ComponentManifest.model_validate(values)

    values = component_manifest().to_dict()
    values["capabilities"] = ["EXPERIMENTAL"]
    values["capture"] = {
        "source": "module_hook",
        "method": "router-hook",
        "verified": False,
    }
    experimental = ComponentManifest.model_validate(values)
    assert experimental.capabilities == [CapabilityLabel.EXPERIMENTAL]

    values["capture"]["verified"] = True
    with pytest.raises(ValidationError, match="unverified capture provenance"):
        ComponentManifest.model_validate(values)


def test_capture_adapter_and_version_must_be_provided_together() -> None:
    with pytest.raises(ValidationError, match="adapter and adapter_version"):
        CaptureProvenance(
            source=CaptureSource.ADAPTER_DECODER,
            method="router-decoder",
            adapter="synthetic",
        )

    with pytest.raises(ValidationError, match="adapter and adapter_version"):
        CaptureProvenance(
            source=CaptureSource.ADAPTER_DECODER,
            method="router-decoder",
            adapter_version="0.1",
        )


def test_metadata_must_be_json_serializable() -> None:
    with pytest.raises(ValidationError, match="JSON-serializable"):
        Provenance(source="fixture", tool_version="0.1.0", metadata={"bad": {1, 2}})


def test_identity_helpers_are_order_independent_and_portable() -> None:
    assert make_config_hash({"a": 1, "b": [2, 3]}) == make_config_hash({"b": [2, 3], "a": 1})
    model_key = make_model_key("acme/demo-moe", "refs/main")
    assert model_key == "model:acme/demo-moe@refs/main"
    assert make_component_key(
        model_key,
        "expert",
        "layers.0.experts.1",
        layer_index=0,
        expert_index=1,
    )

    with pytest.raises(ValueError, match="absolute path"):
        make_model_key("/Users/example/model", "main")
    with pytest.raises(ValueError, match="absolute path"):
        make_component_key(model_key, "expert", "/layers/0")
    with pytest.raises(ValueError, match="model:<portable-id>@<revision>"):
        make_component_key("demo-model", "expert", "layers.0.experts.1")

    values = component_manifest().to_dict()
    values["model_key"] = "demo-model"
    with pytest.raises(ValidationError, match="model:<portable-id>@<revision>"):
        ComponentManifest.model_validate(values)


def test_identity_helpers_do_not_depend_on_python_hash_seed() -> None:
    source = """
from moeatlas.core import make_component_key, make_config_hash, make_model_key
key = make_model_key('acme/demo-moe', 'main')
print(make_config_hash({'b': 2, 'a': 1}))
print(make_component_key(key, 'expert', 'layers.0.experts.1', layer_index=0, expert_index=1))
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    outputs = []
    for seed in ("1", "17", "random"):
        environment["PYTHONHASHSEED"] = seed
        result = subprocess.run(
            [sys.executable, "-c", source],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1] == outputs[2]


def test_core_import_does_not_load_model_runtime() -> None:
    assert not any(name in sys.modules for name in ("torch", "transformers", "safetensors"))
