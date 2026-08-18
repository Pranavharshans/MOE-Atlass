from __future__ import annotations

import socket
import sys
import urllib.request
from pathlib import Path
from typing import Any

import pytest

import moeatlas.runtime.scan as runtime_scan
from moeatlas.core import (
    CapabilityLabel,
    DType,
    ModelManifest,
    Provenance,
    TokenizerIdentity,
    make_config_hash,
    make_model_key,
)
from moeatlas.discovery import DiscoveryReport
from moeatlas.discovery import scan as real_scan
from moeatlas.fixtures import SyntheticMoE
from moeatlas.loading import (
    CustomLoaderSource,
    HuggingFaceSource,
    InstanceSource,
    LoadingPlan,
    LocalSource,
    SourceKind,
    TokenizerRequest,
)
from moeatlas.runtime import (
    LoadedModel,
    RuntimeCleanupError,
    RuntimeLoadError,
    RuntimeValidationError,
    load_and_scan,
)
from moeatlas.runtime.model_loader import _CleanupStack

MODEL_ID = "fixture/bridge-moe"
TOKENIZER_ID = "fixture/bridge-tokenizer"
MODEL_REVISION = "bridge-revision"
OPTIONAL_MODULES = ("torch", "transformers", "accelerate", "safetensors")


def _loaded_optional_modules() -> set[str]:
    return {name for name in OPTIONAL_MODULES if name in sys.modules}


def _plan(source_type: SourceKind, *, path: str = "/private/tmp/bridge-model") -> LoadingPlan:
    tokenizer = TokenizerRequest(identifier=TOKENIZER_ID, requested_revision="tokenizer-revision")
    common: dict[str, Any] = {
        "model_id": MODEL_ID,
        "requested_revision": MODEL_REVISION,
        "tokenizer": tokenizer,
    }
    if source_type is SourceKind.HUGGINGFACE:
        source = HuggingFaceSource(**common)
    elif source_type is SourceKind.LOCAL:
        source = LocalSource(path=path, **common)
    elif source_type is SourceKind.INSTANCE:
        source = InstanceSource(**common)
    else:
        source = CustomLoaderSource(
            loader_reference="tests.test_runtime_scan:fixture_loader", **common
        )
    return LoadingPlan(source=source)


def _manifest() -> ModelManifest:
    return ModelManifest(
        model_key=make_model_key(MODEL_ID, MODEL_REVISION),
        architecture="bridge_moe",
        revision=MODEL_REVISION,
        config_hash=make_config_hash({"fixture": "bridge", "experts": 4}),
        tokenizer=TokenizerIdentity(identifier=TOKENIZER_ID, revision="tokenizer-revision"),
        dtype=DType.FLOAT32,
        device_map={"": "cpu"},
        provenance=Provenance(source="bridge-test", tool_version="test"),
    )


def _loaded(
    plan: LoadingPlan,
    *,
    model: object | None = None,
    cleanup: Any | None = None,
) -> LoadedModel:
    return LoadedModel(
        model=model if model is not None else SyntheticMoE(),
        tokenizer=object(),
        plan=plan,
        manifest=_manifest(),
        warnings=(),
        _cleanup_callback=cleanup,
        _owns_cleanup=cleanup is not None,
    )


@pytest.mark.parametrize(
    ("source_type", "loader_name"),
    [
        (SourceKind.HUGGINGFACE, "load_huggingface"),
        (SourceKind.LOCAL, "load_local"),
    ],
)
def test_load_and_scan_dispatches_exact_plan_and_loaded_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source_type: SourceKind,
    loader_name: str,
) -> None:
    path = str(tmp_path / "not-read")
    plan = _plan(source_type, path=path)
    model = SyntheticMoE()
    loaded = _loaded(plan, model=model, cleanup=lambda: None)
    seen: dict[str, object] = {}

    def fake_loader(received_plan: LoadingPlan) -> LoadedModel:
        seen["plan"] = received_plan
        return loaded

    def fake_scan(received_model: object, received_manifest: ModelManifest) -> DiscoveryReport:
        seen["model"] = received_model
        seen["manifest"] = received_manifest
        return real_scan(received_model, received_manifest)

    monkeypatch.setattr(runtime_scan, loader_name, fake_loader)
    monkeypatch.setattr(runtime_scan, "discovery_scan", fake_scan)

    report = load_and_scan(plan)

    assert seen["plan"] is plan
    assert seen["model"] is model
    assert seen["manifest"] is loaded.manifest
    assert report.model_manifest == loaded.manifest
    assert report.model_key == loaded.manifest.model_key
    assert all(
        component.capabilities == [CapabilityLabel.STRUCTURE] for component in report.components
    )
    assert loaded.closed is True
    assert loaded.model is None
    assert loaded.tokenizer is None


@pytest.mark.parametrize("source_type", [SourceKind.INSTANCE, SourceKind.CUSTOM])
def test_unsupported_sources_fail_before_loader_or_optional_import(
    monkeypatch: pytest.MonkeyPatch,
    source_type: SourceKind,
) -> None:
    plan = _plan(source_type)
    called = False

    def forbidden(_plan: LoadingPlan) -> LoadedModel:
        nonlocal called
        called = True
        raise AssertionError("unsupported source was dispatched")

    monkeypatch.setattr(runtime_scan, "load_huggingface", forbidden)
    monkeypatch.setattr(runtime_scan, "load_local", forbidden)
    before = _loaded_optional_modules()

    with pytest.raises(RuntimeLoadError, match=r"load_instance\(\)|load_custom\(\)"):
        load_and_scan(plan)

    after = _loaded_optional_modules()
    assert called is False
    assert after == before


def test_load_failure_is_unchanged_and_never_scans_or_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(SourceKind.HUGGINGFACE)
    load_error = ValueError("loader failure")
    scanned = False

    def fail_loader(received_plan: LoadingPlan) -> LoadedModel:
        assert received_plan is plan
        raise load_error

    def forbidden_scan(_model: object, _manifest: ModelManifest) -> DiscoveryReport:
        nonlocal scanned
        scanned = True
        raise AssertionError("scan should not run")

    monkeypatch.setattr(runtime_scan, "load_huggingface", fail_loader)
    monkeypatch.setattr(runtime_scan, "discovery_scan", forbidden_scan)

    with pytest.raises(ValueError) as caught:
        load_and_scan(plan)
    assert caught.value is load_error
    assert scanned is False


def test_malformed_scanner_return_is_rejected_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(SourceKind.HUGGINGFACE)
    loaded = _loaded(plan, cleanup=lambda: None)
    scanned = False

    def malformed_scan(_model: object, _manifest: ModelManifest) -> object:
        nonlocal scanned
        scanned = True
        return {"manifest_type": "not-a-discovery-report"}

    monkeypatch.setattr(runtime_scan, "load_huggingface", lambda received: loaded)
    monkeypatch.setattr(runtime_scan, "discovery_scan", malformed_scan)

    with pytest.raises(RuntimeValidationError, match="valid DiscoveryReport"):
        load_and_scan(plan)
    assert scanned is True
    assert loaded.closed is True
    assert loaded.model is None
    assert loaded.tokenizer is None


def test_report_bound_to_different_manifest_is_rejected_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(SourceKind.HUGGINGFACE)
    loaded = _loaded(plan, cleanup=lambda: None)
    other_manifest = _manifest().model_copy(
        update={"model_key": make_model_key("fixture/other-moe", MODEL_REVISION)}
    )
    other_report = real_scan(SyntheticMoE(), other_manifest)

    monkeypatch.setattr(runtime_scan, "load_huggingface", lambda received: loaded)
    monkeypatch.setattr(runtime_scan, "discovery_scan", lambda *_args: other_report)

    with pytest.raises(RuntimeValidationError, match="does not match"):
        load_and_scan(plan)
    assert loaded.closed is True
    assert loaded.model is None
    assert loaded.tokenizer is None


def test_elevated_capability_report_is_rejected_at_bridge_boundary_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(SourceKind.HUGGINGFACE)
    loaded = _loaded(plan, cleanup=lambda: None)
    valid_report = real_scan(SyntheticMoE(), loaded.manifest)
    assert valid_report.components
    elevated_component = valid_report.components[0].model_copy(
        update={"capabilities": [CapabilityLabel.ROUTING]}
    )
    elevated_report = valid_report.model_copy(update={"components": [elevated_component]})

    monkeypatch.setattr(runtime_scan, "load_huggingface", lambda received: loaded)
    monkeypatch.setattr(runtime_scan, "discovery_scan", lambda *_args: elevated_report)

    with pytest.raises(RuntimeValidationError, match="valid DiscoveryReport"):
        load_and_scan(plan)
    assert loaded.closed is True
    assert loaded.model is None
    assert loaded.tokenizer is None


def test_bridge_is_network_cache_and_file_output_free(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_names = (
        "HF_HOME",
        "TRANSFORMERS_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "HF_DATASETS_CACHE",
        "TORCH_HOME",
    )
    cache_dirs: list[Path] = []
    for index, name in enumerate(cache_names):
        cache_dir = tmp_path / f"cache-{index}"
        cache_dir.mkdir()
        cache_dirs.append(cache_dir)
        monkeypatch.setenv(name, str(cache_dir))

    def network_forbidden(*_args: Any, **_kwargs: Any) -> object:
        raise AssertionError("bridge attempted network access")

    monkeypatch.setattr(socket, "create_connection", network_forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", network_forbidden)
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    plan = _plan(SourceKind.HUGGINGFACE)
    loaded = _loaded(plan, cleanup=lambda: None)
    report = real_scan(loaded.model, loaded.manifest)
    monkeypatch.setattr(runtime_scan, "load_huggingface", lambda received: loaded)
    monkeypatch.setattr(runtime_scan, "discovery_scan", lambda *_args: report)

    assert load_and_scan(plan) == report
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert after == before
    assert all(not any(cache_dir.iterdir()) for cache_dir in cache_dirs)


@pytest.mark.parametrize("error_kind", ["value", "keyboard", "system"])
def test_loader_control_flow_failure_is_unchanged_and_never_scans(
    monkeypatch: pytest.MonkeyPatch,
    error_kind: str,
) -> None:
    plan = _plan(SourceKind.HUGGINGFACE)
    if error_kind == "value":
        load_error: BaseException = ValueError("loader failure")
    elif error_kind == "keyboard":
        load_error = KeyboardInterrupt()
    else:
        load_error = SystemExit("loader exit")
    scanned = False

    def fail_loader(received_plan: LoadingPlan) -> LoadedModel:
        assert received_plan is plan
        raise load_error

    def forbidden_scan(_model: object, _manifest: ModelManifest) -> DiscoveryReport:
        nonlocal scanned
        scanned = True
        raise AssertionError("scan should not run")

    monkeypatch.setattr(runtime_scan, "load_huggingface", fail_loader)
    monkeypatch.setattr(runtime_scan, "discovery_scan", forbidden_scan)

    with pytest.raises(type(load_error)) as caught:
        load_and_scan(plan)
    assert caught.value is load_error
    assert scanned is False


@pytest.mark.parametrize("error_kind", ["value", "keyboard", "system"])
def test_scan_failure_preserves_control_flow_and_closes(
    monkeypatch: pytest.MonkeyPatch,
    error_kind: str,
) -> None:
    plan = _plan(SourceKind.HUGGINGFACE)
    close_calls = 0

    def cleanup() -> None:
        nonlocal close_calls
        close_calls += 1

    loaded = _loaded(plan, cleanup=cleanup)
    if error_kind == "value":
        scan_error: BaseException = ValueError("scan body")
    elif error_kind == "keyboard":
        scan_error = KeyboardInterrupt()
    else:
        scan_error = SystemExit("scan exit")

    def fail_scan(_model: object, _manifest: ModelManifest) -> DiscoveryReport:
        raise scan_error

    monkeypatch.setattr(runtime_scan, "load_huggingface", lambda received: loaded)
    monkeypatch.setattr(runtime_scan, "discovery_scan", fail_scan)

    with pytest.raises(type(scan_error)) as caught:
        load_and_scan(plan)
    assert caught.value is scan_error
    assert close_calls == 1
    assert loaded.closed is True
    assert not hasattr(scan_error, "pending_cleanup")


def test_scan_and_cleanup_failure_preserve_body_and_retry_hidden_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(SourceKind.HUGGINGFACE)
    close_calls = 0

    def cleanup() -> None:
        nonlocal close_calls
        close_calls += 1
        if close_calls == 1:
            raise OSError("cleanup detail must not leak")

    loaded = _loaded(plan, cleanup=cleanup)
    scan_error = ValueError("scan body")
    monkeypatch.setattr(runtime_scan, "load_huggingface", lambda received: loaded)

    def fail_scan(_model: object, _manifest: ModelManifest) -> DiscoveryReport:
        raise scan_error

    monkeypatch.setattr(runtime_scan, "discovery_scan", fail_scan)

    with pytest.raises(ValueError) as caught:
        load_and_scan(plan)
    assert caught.value is scan_error
    pending = scan_error.pending_cleanup
    assert pending.pending is True
    assert scan_error.__notes__ == [
        "runtime cleanup also failed; the owned cleanup callback remains retryable (1 failure(s))"
    ]
    assert "cleanup detail" not in str(scan_error)

    pending.retry()
    pending.retry()
    assert pending.pending is False
    assert close_calls == 2
    assert loaded.closed is True
    assert loaded.model is None
    assert loaded.tokenizer is None


def test_success_publishes_only_after_cleanup_and_retry_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(SourceKind.HUGGINGFACE)
    close_calls = 0
    report = real_scan(SyntheticMoE(), _manifest())

    def cleanup() -> None:
        nonlocal close_calls
        close_calls += 1
        if close_calls == 1:
            raise OSError("cleanup detail")

    loaded = _loaded(plan, cleanup=cleanup)
    monkeypatch.setattr(runtime_scan, "load_huggingface", lambda received: loaded)
    monkeypatch.setattr(runtime_scan, "discovery_scan", lambda *_args: report)

    with pytest.raises(RuntimeCleanupError) as caught:
        load_and_scan(plan)
    assert not isinstance(caught.value, DiscoveryReport)
    pending = caught.value.pending_cleanup
    assert pending.pending is True
    assert caught.value.failures and type(caught.value.failures[0]) is OSError

    pending.retry()
    pending.retry()
    assert pending.pending is False
    assert close_calls == 2
    assert loaded.closed is True
    assert loaded.model is None
    assert loaded.tokenizer is None


def test_cleanup_stack_order_and_report_serialization_have_no_runtime_leaks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = str(tmp_path / "bridge-secret-local-path")
    plan = _plan(SourceKind.LOCAL, path=path)
    model = SyntheticMoE()
    cleanup_order: list[str] = []

    class Closeable:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            cleanup_order.append(self.name)

    model.close = lambda: cleanup_order.append("model")  # type: ignore[attr-defined]
    tokenizer = Closeable("tokenizer")
    config = Closeable("config")
    stack = _CleanupStack()
    stack.add_object(model)
    stack.add_object(tokenizer)
    stack.add_object(config)
    loaded = LoadedModel(
        model=model,
        tokenizer=tokenizer,
        plan=plan,
        manifest=_manifest(),
        warnings=(),
        _cleanup_callback=stack,
        _owns_cleanup=True,
    )
    monkeypatch.setattr(runtime_scan, "load_local", lambda received: loaded)

    report = load_and_scan(plan)
    encoded = report.to_json()
    assert DiscoveryReport.from_json(encoded) == report
    assert path not in encoded
    assert "bridge-secret" not in encoded
    assert "Closeable" not in encoded
    assert cleanup_order == ["config", "tokenizer", "model"]


def test_importing_bridge_keeps_optional_model_modules_lazy() -> None:
    before = _loaded_optional_modules()
    __import__("moeatlas.runtime.scan")
    after = _loaded_optional_modules()
    assert after == before
