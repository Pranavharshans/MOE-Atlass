from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib import request

import pytest
from pydantic import ValidationError

from moeatlas.adapters import (
    AdapterContractError,
    AdapterDescriptor,
    AdapterDetection,
    AdapterExecutionError,
    AdapterInspection,
    inspect_static_adapter,
)
from moeatlas.core import (
    CapabilityLabel,
    CaptureProvenance,
    CaptureSource,
    DType,
    ModelManifest,
    TokenizerIdentity,
    make_config_hash,
    make_model_key,
)
from moeatlas.discovery import DiscoveryReport, scan

from .fixtures import SyntheticMoE


def _manifest(*, suffix: str = "") -> ModelManifest:
    revision = "main" if not suffix else f"main-{suffix}"
    return ModelManifest(
        model_key=make_model_key("acme/demo-moe", revision),
        architecture="demo_moe",
        revision=revision,
        config_hash=make_config_hash({"experts": 4, "top_k": 2}),
        tokenizer=TokenizerIdentity(identifier="acme/demo-tokenizer", revision=revision),
        dtype=DType.BFLOAT16,
        device_map={"": "cpu"},
    )


def _descriptor(name: str = "synthetic-static") -> AdapterDescriptor:
    return AdapterDescriptor(
        name=name,
        version="1.0",
        architecture_families=("demo_moe", "synthetic"),
        compatibility_notes=("static structure only",),
    )


def _static_report(
    model_manifest: ModelManifest,
    descriptor: AdapterDescriptor,
    *,
    empty: bool = False,
) -> DiscoveryReport:
    base = scan(SyntheticMoE(), model_manifest)
    if empty:
        return DiscoveryReport(
            model_key=model_manifest.model_key,
            model_manifest=model_manifest,
            warnings=["adapter found no static components"],
        )
    components = [
        component.model_copy(
            update={
                "capture": CaptureProvenance(
                    source=CaptureSource.STATIC_STRUCTURE,
                    method="static-adapter",
                    adapter=descriptor.name,
                    adapter_version=descriptor.version,
                    verified=False,
                )
            }
        )
        for component in base.components
    ]
    return base.model_copy(update={"components": components})


class GoodAdapter:
    def __init__(self, model_manifest: ModelManifest, *, empty: bool = False) -> None:
        self.model_manifest = model_manifest
        self.empty = empty
        self.calls: list[tuple[str, object, object]] = []
        self._descriptor = _descriptor()

    @property
    def descriptor(self) -> AdapterDescriptor:
        self.calls.append(("descriptor", self, None))
        return self._descriptor

    def detect(self, model: object, config: object) -> AdapterDetection:
        self.calls.append(("detect", model, config))
        return AdapterDetection(score=0.8, evidence=("module surface",))

    def discover(self, model: object, model_manifest: ModelManifest) -> DiscoveryReport:
        self.calls.append(("discover", model, model_manifest))
        return _static_report(model_manifest, self._descriptor, empty=self.empty)


def test_contracts_are_strict_frozen_deterministic_and_json_roundtrip() -> None:
    descriptor = _descriptor()
    detection = AdapterDetection(
        score=0.75,
        evidence=("config", "module surface"),
        warnings=("z-warning", "a-warning"),
    )
    assert descriptor.architecture_families == ("demo_moe", "synthetic")
    assert detection.evidence == ("config", "module surface")
    assert detection.warnings == ("a-warning", "z-warning")
    with pytest.raises(ValidationError):
        descriptor.name = "changed"  # type: ignore[misc]

    decoded_descriptor = AdapterDescriptor.from_json(descriptor.to_json())
    decoded_detection = AdapterDetection.from_json(detection.to_json())
    assert decoded_descriptor == descriptor
    assert decoded_detection == detection
    assert descriptor.to_json() == decoded_descriptor.to_json()
    assert json.loads(descriptor.to_json())["manifest_type"] == "adapter_descriptor"
    assert json.loads(detection.to_json())["manifest_type"] == "adapter_detection"


@pytest.mark.parametrize(
    "factory",
    [
        lambda: AdapterDescriptor(name=" x", version="1", architecture_families=("a",)),
        lambda: AdapterDescriptor(name="x", version="1", architecture_families=("b", "a")),
        lambda: AdapterDescriptor(name="x", version="1", architecture_families=("a", "a")),
        lambda: AdapterDescriptor(
            name="x", version="1", architecture_families=("a",), compatibility_notes=("",)
        ),
        lambda: AdapterDetection(score=1, evidence=("e",)),
        lambda: AdapterDetection(score=True, evidence=("e",)),
        lambda: AdapterDetection(score=float("nan"), evidence=("e",)),
        lambda: AdapterDetection(score=0.5),
        lambda: AdapterDetection(score=0.0),
        lambda: AdapterDetection(score=0.5, evidence=("e", "e")),
        lambda: AdapterDetection(score=0.5, evidence=("e",), warnings=("w", "w")),
        lambda: AdapterDetection(score=0.5, evidence=("",)),
        lambda: AdapterDetection(score=0.5, evidence=(" ",)),
        lambda: AdapterDetection(score=0.5, evidence=("e",), warnings=("",)),
        lambda: AdapterDetection(score=0.5, evidence=("e",), warnings=(" ",)),
    ],
)
def test_invalid_adapter_schema_values_are_rejected(factory: Any) -> None:
    with pytest.raises((ValidationError, TypeError, ValueError)):
        factory()


def test_descriptor_forbids_registry_package_and_source_fields() -> None:
    with pytest.raises(ValidationError):
        AdapterDescriptor(
            name="x",
            version="1",
            architecture_families=("a",),
            source="module:function",  # type: ignore[call-arg]
        )


def test_inspection_calls_descriptor_detect_discover_in_order_and_preserves_identity() -> None:
    model = SyntheticMoE()
    config = object()
    manifest = _manifest()
    adapter = GoodAdapter(manifest)

    inspection = inspect_static_adapter(adapter, model, config, manifest)

    assert isinstance(inspection, AdapterInspection)
    assert [call[0] for call in adapter.calls] == ["descriptor", "detect", "discover"]
    assert adapter.calls[1][1] is model
    assert adapter.calls[1][2] is config
    assert adapter.calls[2][1] is model
    assert adapter.calls[2][2] is manifest
    assert inspection.report.model_manifest == manifest
    assert inspection.detection.score > 0
    assert json.loads(inspection.to_json())["manifest_type"] == "adapter_inspection"
    assert "SyntheticMoE" not in inspection.to_json()


def test_two_caller_adapters_are_explicit_and_no_registry_is_used() -> None:
    manifest = _manifest()
    model = SyntheticMoE()
    first = GoodAdapter(manifest)
    second = GoodAdapter(manifest)
    second._descriptor = _descriptor("other-static")

    assert inspect_static_adapter(first, model, {}, manifest).descriptor.name == "synthetic-static"
    assert inspect_static_adapter(second, model, {}, manifest).descriptor.name == "other-static"
    assert [call[0] for call in first.calls] == ["descriptor", "detect", "discover"]
    assert [call[0] for call in second.calls] == ["descriptor", "detect", "discover"]


def test_zero_detection_blocks_discover() -> None:
    class ZeroAdapter(GoodAdapter):
        def detect(self, model: object, config: object) -> AdapterDetection:
            self.calls.append(("detect", model, config))
            return AdapterDetection(score=0.0, warnings=("no supported structure",))

    adapter = ZeroAdapter(_manifest())
    with pytest.raises(AdapterContractError) as exc_info:
        inspect_static_adapter(adapter, SyntheticMoE(), {}, adapter.model_manifest)
    assert exc_info.value.stage == "detect"
    assert [call[0] for call in adapter.calls] == ["descriptor", "detect"]


@pytest.mark.parametrize(
    ("attribute", "value"),
    [("descriptor", None), ("detect", None), ("discover", None)],
)
def test_missing_or_noncallable_adapter_members_are_rejected(attribute: str, value: object) -> None:
    class Incomplete:
        descriptor = _descriptor()
        detect = lambda self, model, config: AdapterDetection(  # noqa: E731
            score=0.8, evidence=("x",)
        )
        discover = lambda self, model, manifest: _static_report(manifest, self.descriptor)  # noqa: E731

    adapter = Incomplete()
    setattr(adapter, attribute, value)
    with pytest.raises(AdapterContractError):
        inspect_static_adapter(adapter, SyntheticMoE(), {}, _manifest())


@pytest.mark.parametrize("missing", ["descriptor", "detect", "discover"])
def test_truly_missing_members_are_static_contract_failures_without_descriptor_execution(
    missing: str,
) -> None:
    accessed: list[str] = []

    def descriptor(_adapter: object) -> AdapterDescriptor:
        accessed.append("descriptor")
        return _descriptor()

    def detect(_adapter: object, model: object, config: object) -> AdapterDetection:
        return AdapterDetection(score=0.8, evidence=("x",))

    def discover(_adapter: object, model: object, model_manifest: ModelManifest) -> DiscoveryReport:
        return _static_report(model_manifest, _descriptor())

    members: dict[str, object] = {
        "descriptor": property(descriptor),
        "detect": detect,
        "discover": discover,
    }
    members.pop(missing)
    adapter = type("MissingAdapter", (), members)()

    with pytest.raises(AdapterContractError) as exc_info:
        inspect_static_adapter(adapter, object(), {}, _manifest())
    assert exc_info.value.stage == missing
    assert accessed == []


def test_wrong_return_types_and_model_copy_tampering_are_rejected() -> None:
    manifest = _manifest()

    class WrongDetect(GoodAdapter):
        def detect(self, model: object, config: object) -> object:
            return {"score": 1.0}

    with pytest.raises(AdapterContractError) as detect_error:
        inspect_static_adapter(WrongDetect(manifest), SyntheticMoE(), {}, manifest)
    assert detect_error.value.stage == "detect"

    class TamperedDetect(GoodAdapter):
        def detect(self, model: object, config: object) -> AdapterDetection:
            return AdapterDetection(score=0.8, evidence=("x",)).model_copy(update={"score": 0.0})

    with pytest.raises(AdapterContractError) as tampered_error:
        inspect_static_adapter(TamperedDetect(manifest), SyntheticMoE(), {}, manifest)
    assert tampered_error.value.stage == "detect"

    class WrongDiscover(GoodAdapter):
        def discover(self, model: object, model_manifest: ModelManifest) -> object:
            return {"model_key": model_manifest.model_key}

    with pytest.raises(AdapterContractError) as discover_error:
        inspect_static_adapter(WrongDiscover(manifest), SyntheticMoE(), {}, manifest)
    assert discover_error.value.stage == "discover"


def test_report_must_match_manifest_and_have_static_unverified_components() -> None:
    manifest = _manifest()

    with pytest.raises(ValidationError):
        AdapterInspection(
            descriptor=_descriptor(),
            detection=AdapterDetection(score=0.8, evidence=("x",)),
            report=scan(SyntheticMoE(), manifest),
        )

    class DifferentManifest(GoodAdapter):
        def discover(self, model: object, model_manifest: ModelManifest) -> DiscoveryReport:
            return _static_report(_manifest(suffix="other"), self._descriptor)

    with pytest.raises(AdapterContractError):
        inspect_static_adapter(DifferentManifest(manifest), SyntheticMoE(), {}, manifest)

    class Elevated(GoodAdapter):
        def discover(self, model: object, model_manifest: ModelManifest) -> DiscoveryReport:
            report = _static_report(model_manifest, self._descriptor)
            components = [
                item.model_copy(update={"capabilities": [CapabilityLabel.ROUTING]})
                for item in report.components
            ]
            return report.model_copy(update={"components": components})

    with pytest.raises(AdapterContractError):
        inspect_static_adapter(Elevated(manifest), SyntheticMoE(), {}, manifest)

    class BadCapture(GoodAdapter):
        def discover(self, model: object, model_manifest: ModelManifest) -> DiscoveryReport:
            report = _static_report(model_manifest, self._descriptor)
            components = [
                item.model_copy(
                    update={
                        "capture": CaptureProvenance(
                            source=CaptureSource.MODULE_HOOK,
                            method="hook",
                            adapter=self._descriptor.name,
                            adapter_version=self._descriptor.version,
                            verified=False,
                        )
                    }
                )
                for item in report.components
            ]
            return report.model_copy(update={"components": components})

    with pytest.raises(AdapterContractError):
        inspect_static_adapter(BadCapture(manifest), SyntheticMoE(), {}, manifest)


def test_empty_report_requires_warning() -> None:
    manifest = _manifest()

    class SilentEmpty(GoodAdapter):
        def discover(self, model: object, model_manifest: ModelManifest) -> DiscoveryReport:
            return DiscoveryReport(
                model_key=model_manifest.model_key,
                model_manifest=model_manifest,
            )

    with pytest.raises(AdapterContractError):
        inspect_static_adapter(SilentEmpty(manifest), SyntheticMoE(), {}, manifest)

    inspection = inspect_static_adapter(
        GoodAdapter(manifest, empty=True), SyntheticMoE(), {}, manifest
    )
    assert inspection.report.components == []
    assert inspection.report.warnings


@pytest.mark.parametrize("stage", ["descriptor", "detect", "discover"])
def test_ordinary_adapter_errors_are_wrapped_without_secret_leak(stage: str) -> None:
    manifest = _manifest()
    secret = "TOP_SECRET_ADAPTER_PAYLOAD"

    class Failing:
        @property
        def descriptor(self) -> AdapterDescriptor:
            if stage == "descriptor":
                raise ValueError(secret)
            return _descriptor()

        def detect(self, model: object, config: object) -> AdapterDetection:
            if stage == "detect":
                raise ValueError(secret)
            return AdapterDetection(score=0.8, evidence=("x",))

        def discover(self, model: object, model_manifest: ModelManifest) -> DiscoveryReport:
            if stage == "discover":
                raise ValueError(secret)
            return _static_report(model_manifest, _descriptor())

    with pytest.raises(AdapterExecutionError) as exc_info:
        inspect_static_adapter(Failing(), SyntheticMoE(), {"secret": secret}, manifest)
    assert exc_info.value.stage == stage
    assert secret not in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert str(exc_info.value.__cause__) == secret


@pytest.mark.parametrize("control_flow", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("stage", ["descriptor", "detect", "discover"])
def test_control_flow_exceptions_propagate_unchanged(
    stage: str, control_flow: type[BaseException]
) -> None:
    manifest = _manifest()

    class Failing:
        @property
        def descriptor(self) -> AdapterDescriptor:
            if stage == "descriptor":
                raise control_flow()
            return _descriptor()

        def detect(self, model: object, config: object) -> AdapterDetection:
            if stage == "detect":
                raise control_flow()
            return AdapterDetection(score=0.8, evidence=("x",))

        def discover(self, model: object, model_manifest: ModelManifest) -> DiscoveryReport:
            if stage == "discover":
                raise control_flow()
            return _static_report(model_manifest, _descriptor())

    with pytest.raises(control_flow):
        inspect_static_adapter(Failing(), SyntheticMoE(), {}, manifest)


def test_model_is_an_untouched_sentinel_and_no_surface_is_preflighted() -> None:
    class UntouchableModel:
        @property
        def named_modules(self) -> object:
            raise AssertionError("model surface must not be accessed")

        def forward(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("forward must not be called")

        def generate(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("generate must not be called")

    model = UntouchableModel()
    manifest = _manifest()
    inspection = inspect_static_adapter(GoodAdapter(manifest), model, object(), manifest)
    assert inspection.report.model_key == manifest.model_key


def test_inspection_does_not_use_network_cache_or_filesystem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("network must not be used")

    monkeypatch.setattr(socket, "create_connection", fail_network)
    monkeypatch.setattr(request, "urlopen", fail_network)
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    monkeypatch.setenv("HF_HOME", str(cache_root / "hf"))
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(cache_root / "transformers"))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(cache_root / "hub"))
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    manifest = _manifest()
    inspection = inspect_static_adapter(GoodAdapter(manifest), SyntheticMoE(), object(), manifest)

    assert inspection.detection.score > 0
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == before


def test_import_is_lazy_and_does_not_query_entry_points() -> None:
    root = Path(__file__).resolve().parents[1]
    child_code = """
import importlib.metadata
import importlib.util
import sys

def fail_entry_points(*args, **kwargs):
    raise AssertionError("entry-point discovery must not run")

importlib.metadata.entry_points = fail_entry_points
import moeatlas.adapters
assert all(
    name not in sys.modules
    for name in ("torch", "transformers", "safetensors", "accelerate")
)
assert importlib.util.find_spec("moeatlas.adapters") is not None
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    subprocess.run(
        [sys.executable, "-c", child_code],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def test_descriptor_and_detection_are_versioned_and_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        AdapterDescriptor(
            name="x",
            version="1",
            architecture_families=("a",),
            manifest_type="not-adapter",  # type: ignore[arg-type]
        )
    with pytest.raises(ValidationError):
        AdapterDetection(score=0.5, evidence=("x",), future_field="nope")  # type: ignore[call-arg]
