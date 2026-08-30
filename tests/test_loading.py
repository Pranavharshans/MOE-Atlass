from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from moeatlas.core import DType, make_config_hash
from moeatlas.loading import (
    CustomLoaderSource,
    DeviceKind,
    DownloadPolicy,
    DTypePolicy,
    HuggingFaceSource,
    ImmutableRevisionEvidence,
    InstanceSource,
    LoadConfig,
    LoadingPlan,
    LocalSource,
    QuantizationPolicy,
    ResolvedSource,
    RevisionEvidenceKind,
    SourceKind,
    SourceRequest,
    TokenizerRequest,
    make_loading_plan_id,
    parse_loading_plan_id,
    portable_loading_intent,
)

MODEL_ID = "org/demo-moe"
MODEL_REVISION = "main"
TOKENIZER_ID = "org/demo-tokenizer"


def hf_source(**overrides: object) -> HuggingFaceSource:
    values: dict[str, object] = {
        "model_id": MODEL_ID,
        "requested_revision": MODEL_REVISION,
    }
    values.update(overrides)
    return HuggingFaceSource(**values)


def load_config(**overrides: object) -> LoadConfig:
    values: dict[str, object] = {}
    values.update(overrides)
    return LoadConfig(**values)


def test_discriminated_sources_round_trip_without_runtime_objects() -> None:
    sources: tuple[SourceRequest, ...] = (
        hf_source(
            tokenizer=TokenizerRequest(
                identifier=TOKENIZER_ID,
                requested_revision="tok-v1",
            )
        ),
        LocalSource(
            path="/nonexistent/cache/../demo",
            model_id=MODEL_ID,
            requested_revision="content-v1",
        ),
        InstanceSource(model_id=MODEL_ID, requested_revision="instance-v1"),
        CustomLoaderSource(
            model_id=MODEL_ID,
            requested_revision="custom-v1",
            loader_reference="package.loader:build_model",
        ),
    )
    adapter = TypeAdapter(SourceRequest)

    for source in sources:
        restored = adapter.validate_json(source.to_json())
        assert type(restored) is type(source)
        assert restored == source
        assert restored.to_dict()["source_type"] in {
            "huggingface",
            "local",
            "instance",
            "custom",
        }

    with pytest.raises(ValidationError, match="extra_forbidden"):
        InstanceSource(
            model_id=MODEL_ID,
            requested_revision="v1",
            runtime_object=object(),
        )


def test_huggingface_defaults_are_offline_and_revision_is_requested_not_immutable() -> None:
    source = hf_source()
    assert source.download_policy is DownloadPolicy.OFFLINE
    assert source.allow_downloads is False
    assert source.requested_revision == "main"

    with pytest.raises(ValidationError, match="requested_revision"):
        HuggingFaceSource(model_id=MODEL_ID)
    with pytest.raises(ValidationError, match="requested_revision"):
        hf_source(requested_revision=123)

    downloadable = hf_source(
        download_policy=DownloadPolicy.ALLOW_DOWNLOADS,
        allow_downloads=True,
    )
    assert downloadable.allow_downloads is True
    with pytest.raises(ValidationError, match="offline.*allow_downloads"):
        hf_source(allow_downloads=True)
    with pytest.raises(ValidationError, match="requires explicit"):
        hf_source(download_policy=DownloadPolicy.ALLOW_DOWNLOADS)


def test_local_path_is_lexical_runtime_input_and_never_filesystem_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(os.path, "exists", lambda value: pytest.fail("path was inspected"))
    source = LocalSource(
        path=str(tmp_path / "missing" / ".." / "model"),
        model_id=MODEL_ID,
        requested_revision="content-v1",
    )
    assert source.path == str(tmp_path / "model")

    with pytest.raises(ValidationError, match="must not be empty"):
        LocalSource(path="", model_id=MODEL_ID, requested_revision="v1")
    with pytest.raises(ValidationError, match="NUL"):
        LocalSource(path="model\x00dir", model_id=MODEL_ID, requested_revision="v1")
    with pytest.raises(ValidationError, match="model_id"):
        LocalSource(path="relative/model", model_id="/absolute/local", requested_revision="v1")


def test_custom_loader_reference_is_syntax_only_and_not_imported() -> None:
    valid = CustomLoaderSource(
        model_id=MODEL_ID,
        requested_revision="v1",
        loader_reference="some.module:factory",
    )
    assert valid.loader_reference == "some.module:factory"
    for reference in ("some.module", "some.module:", ":factory", "some.module:factory.bad"):
        with pytest.raises(ValidationError, match="module:function"):
            CustomLoaderSource(
                model_id=MODEL_ID,
                requested_revision="v1",
                loader_reference=reference,
            )


def test_tokenizer_revision_is_separate_and_inheritance_is_explicit() -> None:
    explicit = TokenizerRequest(identifier=TOKENIZER_ID, requested_revision="tok-v1")
    inherited = TokenizerRequest(identifier=TOKENIZER_ID, inherit_model_revision=True)
    assert explicit.requested_revision == "tok-v1"
    assert inherited.inherit_model_revision is True

    with pytest.raises(ValidationError, match="required"):
        TokenizerRequest(identifier=TOKENIZER_ID)
    with pytest.raises(ValidationError, match="must be omitted"):
        TokenizerRequest(
            identifier=TOKENIZER_ID,
            requested_revision="tok-v1",
            inherit_model_revision=True,
        )
    with pytest.raises(ValidationError, match="identifier"):
        TokenizerRequest(identifier=" ", requested_revision="v1")

    inherited_plan = LoadingPlan(
        source=hf_source(tokenizer=inherited),
    )
    tokenizer_intent = portable_loading_intent(inherited_plan.source, inherited_plan.config)[
        "source"
    ]["tokenizer"]
    assert tokenizer_intent["requested_revision"] == MODEL_REVISION
    assert tokenizer_intent["inherit_model_revision"] is True


def test_load_config_defaults_and_device_map_invariants() -> None:
    config = load_config()
    assert config.device == DeviceKind.CPU.value
    assert config.dtype is DTypePolicy.PRESERVE
    assert config.quantization is QuantizationPolicy.NONE
    assert config.download_policy is DownloadPolicy.OFFLINE
    assert config.allow_downloads is False
    assert config.trust_remote_code is False
    assert config.ram_offload is False
    assert DTypePolicy.PRESERVE.manifest_dtype_hint() is DType.UNKNOWN
    assert DTypePolicy.FLOAT32.manifest_dtype_hint() is DType.FLOAT32
    assert DTypePolicy.FLOAT16.manifest_dtype_hint() is DType.FLOAT16
    assert DTypePolicy.BFLOAT16.manifest_dtype_hint() is DType.BFLOAT16

    multi_gpu = load_config(
        device=DeviceKind.CUDA.value,
        device_map={"layers.0": "cuda:0", "layers.1": "cuda:1"},
    )
    assert multi_gpu.device_map == {"layers.0": "cuda:0", "layers.1": "cuda:1"}

    with pytest.raises(ValidationError, match="incompatible"):
        load_config(device="cuda:0", device_map={"": "cuda:1"})
    with pytest.raises(ValidationError, match="incompatible"):
        load_config(device=DeviceKind.CPU.value, device_map={"": "cuda:0"})
    with pytest.raises(ValidationError, match="device"):
        load_config(device="tpu")
    with pytest.raises(ValidationError, match="device_map"):
        load_config(device_map={"layers.0": 0})
    with pytest.raises(ValidationError, match="device_map"):
        load_config(device_map={"layers.0": "auto"})
    with pytest.raises(ValidationError, match="incompatible"):
        load_config(device=DeviceKind.CUDA.value, device_map={"": "mps"})


def test_ram_offload_is_strictly_bound_to_auto_and_plan_identity() -> None:
    config = load_config(device=DeviceKind.AUTO.value, ram_offload=True)
    assert config.ram_offload is True
    assert any("CPU RAM offload" in warning for warning in config.security_warnings)
    assert portable_loading_intent(hf_source(), config)["config"]["ram_offload"] is True
    assert LoadingPlan(source=hf_source(), config=config).plan_id != LoadingPlan(
        source=hf_source(), config=load_config(device=DeviceKind.AUTO.value)
    ).plan_id

    with pytest.raises(ValidationError, match="requires device='auto'"):
        load_config(device=DeviceKind.CPU.value, ram_offload=True)
    with pytest.raises(ValidationError, match="explicit device_map"):
        load_config(device=DeviceKind.AUTO.value, ram_offload=True, device_map={"": "cpu"})
    with pytest.raises(ValidationError):
        load_config(device=DeviceKind.AUTO.value, ram_offload=1)


def test_load_config_security_acknowledgement_download_and_derived_warnings() -> None:
    with pytest.raises(ValidationError, match="trust_remote_code=True"):
        load_config(trust_remote_code=True)
    with pytest.raises(ValidationError, match="requires trust_remote_code"):
        load_config(remote_code_acknowledged=True)

    config = load_config(
        device=DeviceKind.MPS.value,
        dtype=DTypePolicy.FLOAT16,
        quantization=QuantizationPolicy.INT4,
        trust_remote_code=True,
        remote_code_acknowledged=True,
        download_policy=DownloadPolicy.ALLOW_DOWNLOADS,
        allow_downloads=True,
    )
    source = hf_source(
        download_policy=DownloadPolicy.ALLOW_DOWNLOADS,
        allow_downloads=True,
    )
    plan = LoadingPlan(source=source, config=config)
    warnings = plan.security_warnings
    assert warnings == tuple(sorted(warnings))
    assert any("trust_remote_code" in warning for warning in warnings)
    assert any("quantization=int4" in warning for warning in warnings)
    assert any("MPS" in warning for warning in warnings)
    assert any("downloads" in warning for warning in warnings)

    custom_plan = LoadingPlan(
        source=CustomLoaderSource(
            model_id=MODEL_ID,
            requested_revision="v1",
            loader_reference="package.loader:build_model",
        ),
        config=load_config(),
    )
    assert any("user code" in warning for warning in custom_plan.security_warnings)

    with pytest.raises(ValidationError, match="policies must agree"):
        LoadingPlan(source=source)


def test_load_options_are_finite_json_safe_and_deeply_immutable() -> None:
    original = {"z": {"values": [1, 2]}, "a": (True, None)}
    config = load_config(loader_options=original)
    original["z"]["values"].append(99)  # type: ignore[index]
    assert config.loader_options == {"a": [True, None], "z": {"values": [1, 2]}}
    with pytest.raises(TypeError, match="immutable"):
        config.loader_options["new"] = True  # type: ignore[index]
    with pytest.raises(TypeError, match="immutable"):
        config.loader_options["z"]["values"].append(3)  # type: ignore[attr-defined]
    assert LoadConfig.from_json(config.to_json()) == config

    with pytest.raises(ValidationError, match="JSON-compatible"):
        load_config(loader_options={"bad": float("nan")})
    with pytest.raises(ValidationError, match="JSON-compatible"):
        load_config(loader_options={"nested": {1: "bad"}})
    with pytest.raises(ValidationError, match="JSON-compatible"):
        load_config(loader_options={"bad": object()})
    for reserved in (
        "trust_remote_code",
        "local_files_only",
        "revision",
        "device",
        "device_map",
        "dtype",
        "torch_dtype",
        "quantization_config",
        "ram_offload",
        "load_in_4bit",
        "load_in_8bit",
    ):
        with pytest.raises(ValidationError, match="reserved"):
            load_config(loader_options={reserved: True})
    nested_backend_options = load_config(
        device="cpu",
        dtype=DTypePolicy.FLOAT32,
        loader_options={
            "nested": {
                "requested_revision": "other",
                "dtype": "backend-specific",
            }
        },
    )
    assert nested_backend_options.device == "cpu"
    assert nested_backend_options.dtype is DTypePolicy.FLOAT32
    assert nested_backend_options.loader_options["nested"] == {
        "dtype": "backend-specific",
        "requested_revision": "other",
    }
    with pytest.raises(TypeError, match="immutable"):
        nested_backend_options.loader_options["nested"]["dtype"] = "changed"  # type: ignore[index]


def test_loading_plan_id_is_full_sha_path_independent_and_tamper_evident() -> None:
    first = LoadingPlan(
        source=LocalSource(
            path="/machine-a/checkpoints/model",
            model_id=MODEL_ID,
            requested_revision="v1",
        ),
        config=load_config(loader_options={"b": 2, "a": 1}),
    )
    second = LoadingPlan(
        source=LocalSource(
            path="/machine-b/checkpoints/model",
            model_id=MODEL_ID,
            requested_revision="v1",
        ),
        config=load_config(loader_options={"a": 1, "b": 2}),
    )
    assert first.plan_id == second.plan_id
    assert parse_loading_plan_id(first.plan_id) == first.plan_id.removeprefix("loadplan:")
    assert len(parse_loading_plan_id(first.plan_id)) == 64
    assert "machine-a" not in json.dumps(portable_loading_intent(first.source, first.config))
    assert "machine-b" not in json.dumps(portable_loading_intent(second.source, second.config))
    assert first.to_json() != second.to_json()

    changed_revision = LoadingPlan(
        source=LocalSource(
            path="/machine-a/checkpoints/model",
            model_id=MODEL_ID,
            requested_revision="v2",
        ),
        config=first.config,
    )
    changed_config = LoadingPlan(
        source=first.source,
        config=load_config(dtype=DTypePolicy.FLOAT32),
    )
    assert changed_revision.plan_id != first.plan_id
    assert changed_config.plan_id != first.plan_id
    assert make_loading_plan_id(first.source, first.config) == first.plan_id

    with pytest.raises(ValidationError, match="plan_id"):
        LoadingPlan(source=first.source, config=first.config, plan_id="loadplan:" + "0" * 64)
    with pytest.raises(ValidationError, match="canonical loadplan"):
        LoadingPlan(source=first.source, plan_id="not-a-plan")


def test_huggingface_source_and_config_policies_must_match_in_a_plan() -> None:
    source = hf_source(
        download_policy=DownloadPolicy.ALLOW_DOWNLOADS,
        allow_downloads=True,
    )
    with pytest.raises(ValidationError, match="policies must agree"):
        LoadingPlan(source=source)
    plan = LoadingPlan(
        source=source,
        config=load_config(
            download_policy=DownloadPolicy.ALLOW_DOWNLOADS,
            allow_downloads=True,
        ),
    )
    assert plan.config.allow_downloads is True


def test_resolved_source_records_requested_and_external_immutable_evidence() -> None:
    source = hf_source(
        tokenizer=TokenizerRequest(identifier=TOKENIZER_ID, requested_revision="tok-main")
    )
    resolution = ResolvedSource(
        source_type=SourceKind.HUGGINGFACE,
        model_id=MODEL_ID,
        requested_model_revision=MODEL_REVISION,
        resolved_model_revision="sha256:" + "a" * 64,
        resolved_model_revision_evidence=ImmutableRevisionEvidence(
            kind=RevisionEvidenceKind.SHA256,
            digest="a" * 64,
            evidence_source="external Hub content digest",
        ),
        requested_tokenizer_revision="tok-main",
        resolved_tokenizer_revision="b" * 40,
        resolved_tokenizer_revision_evidence=ImmutableRevisionEvidence(
            kind=RevisionEvidenceKind.GIT_COMMIT,
            digest="b" * 40,
            evidence_source="external tokenizer commit response",
        ),
    )
    plan = LoadingPlan(source=source, resolution=resolution)
    restored = LoadingPlan.from_json(plan.to_json())
    assert restored == plan
    assert restored.resolution is not None
    assert restored.resolution.requested_model_revision == MODEL_REVISION

    tokenizer_unresolved = ResolvedSource(
        source_type=SourceKind.HUGGINGFACE,
        model_id=MODEL_ID,
        requested_model_revision=MODEL_REVISION,
        requested_tokenizer_revision=None,
        resolved_model_revision="sha256:" + "c" * 64,
        resolved_model_revision_evidence=ImmutableRevisionEvidence(
            kind=RevisionEvidenceKind.SHA256,
            digest="c" * 64,
            evidence_source="external model content digest",
        ),
        resolved_tokenizer_revision=None,
        resolved_tokenizer_revision_evidence=None,
    )
    assert ResolvedSource.from_json(tokenizer_unresolved.to_json()) == tokenizer_unresolved

    with pytest.raises(ValidationError, match="resolved_model_revision"):
        ResolvedSource(
            source_type=SourceKind.HUGGINGFACE,
            model_id=MODEL_ID,
            requested_model_revision=MODEL_REVISION,
            resolved_model_revision=None,
            resolved_model_revision_evidence=None,
        )
    with pytest.raises(ValidationError, match="resolved_model_revision_evidence"):
        ResolvedSource(
            source_type=SourceKind.HUGGINGFACE,
            model_id=MODEL_ID,
            requested_model_revision=MODEL_REVISION,
            resolved_model_revision="sha256:" + "a" * 64,
        )
    with pytest.raises(ValidationError, match="full 40-character"):
        ImmutableRevisionEvidence(
            kind=RevisionEvidenceKind.GIT_COMMIT,
            digest="a" * 39,
            evidence_source="claimed",
        )
    with pytest.raises(ValidationError, match="full 64-character"):
        ImmutableRevisionEvidence(
            kind=RevisionEvidenceKind.SHA256,
            digest="b" * 40,
            evidence_source="claimed",
        )
    with pytest.raises(ValidationError, match="does not match"):
        ResolvedSource(
            source_type=SourceKind.HUGGINGFACE,
            model_id=MODEL_ID,
            requested_model_revision=MODEL_REVISION,
            resolved_model_revision="main",
            resolved_model_revision_evidence=ImmutableRevisionEvidence(
                kind=RevisionEvidenceKind.GIT_COMMIT,
                digest="c" * 40,
                evidence_source="claimed branch",
            ),
        )
    with pytest.raises(ValidationError, match="resolved_tokenizer_revision"):
        ResolvedSource(
            source_type=SourceKind.HUGGINGFACE,
            model_id=MODEL_ID,
            requested_model_revision=MODEL_REVISION,
            resolved_tokenizer_revision="b" * 64,
            resolved_tokenizer_revision_evidence="external evidence",
        )
    with pytest.raises(ValidationError, match="requested_model_revision"):
        LoadingPlan(
            source=source,
            resolution=resolution.model_copy(update={"requested_model_revision": "other"}),
        )


def test_source_and_loading_contracts_are_strict_and_model_free() -> None:
    with pytest.raises(ValidationError, match="model_id"):
        hf_source(model_id=123)
    with pytest.raises(ValidationError, match="trust_remote_code"):
        load_config(trust_remote_code="false")
    with pytest.raises(ValidationError, match="extra_forbidden"):
        load_config(unexpected=True)
    with pytest.raises(ValidationError, match="extra_forbidden"):
        hf_source(unexpected=True)
    with pytest.raises(ValidationError, match="@"):
        hf_source(model_id="org/model@ambiguous")
    with pytest.raises(ValidationError, match="@"):
        hf_source(requested_revision="ref@ambiguous")
    with pytest.raises(ValueError, match="canonical loadplan"):
        parse_loading_plan_id("loadplan:" + "A" * 64)
    assert not any(name in sys.modules for name in ("torch", "transformers", "safetensors"))


def test_loading_hash_uses_existing_config_hash_convention() -> None:
    config = {"dtype": "float32", "device": "cpu"}
    assert make_config_hash(config).startswith("sha256:")
