from __future__ import annotations

import hashlib
import json
import socket
import sys
import types
import urllib.request
from pathlib import Path
from typing import Any

import pytest

import moeatlas.runtime.model_loader as model_loader
from moeatlas.core import DType
from moeatlas.loading import (
    DownloadPolicy,
    HuggingFaceSource,
    ImmutableRevisionEvidence,
    LoadConfig,
    LoadingPlan,
    LocalSource,
    ResolvedSource,
    RevisionEvidenceKind,
    SourceKind,
    TokenizerRequest,
)
from moeatlas.runtime import (
    ModelLoadError,
    ModelObservationError,
    ModelRuntimeDependencyError,
    RuntimeCleanupError,
    RuntimeValidationError,
    load_huggingface,
    load_local,
)

MODEL_ID = "org/test-moe"
TOKENIZER_ID = "org/test-tokenizer"
MODEL_REQUEST = "main"
TOKENIZER_REQUEST = "tok-main"
MODEL_COMMIT = "a" * 40
TOKENIZER_COMMIT = "b" * 40


@pytest.fixture(autouse=True)
def fake_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("torch")
    module.float32 = object()
    module.float16 = object()
    module.bfloat16 = object()
    monkeypatch.setitem(sys.modules, "torch", module)


def _plan(
    source_type: SourceKind = SourceKind.HUGGINGFACE,
    *,
    path: str = "/tmp/moeatlas-test-model",
    config: LoadConfig | None = None,
    model_evidence_source: str = "fake hub resolver",
) -> LoadingPlan:
    tokenizer = TokenizerRequest(
        identifier=TOKENIZER_ID,
        requested_revision=TOKENIZER_REQUEST,
    )
    if source_type is SourceKind.HUGGINGFACE:
        source = HuggingFaceSource(
            model_id=MODEL_ID,
            requested_revision=MODEL_REQUEST,
            tokenizer=tokenizer,
            download_policy=(config.download_policy if config is not None else "offline"),
            allow_downloads=(config.allow_downloads if config is not None else False),
        )
    else:
        source = LocalSource(
            path=path,
            model_id=MODEL_ID,
            requested_revision=MODEL_REQUEST,
            tokenizer=tokenizer,
        )
    resolution = ResolvedSource(
        source_type=source_type,
        model_id=MODEL_ID,
        requested_model_revision=MODEL_REQUEST,
        resolved_model_revision=MODEL_COMMIT,
        resolved_model_revision_evidence=ImmutableRevisionEvidence(
            kind=RevisionEvidenceKind.GIT_COMMIT,
            digest=MODEL_COMMIT,
            evidence_source=model_evidence_source,
        ),
        requested_tokenizer_revision=TOKENIZER_REQUEST,
        resolved_tokenizer_revision=TOKENIZER_COMMIT,
        resolved_tokenizer_revision_evidence=ImmutableRevisionEvidence(
            kind=RevisionEvidenceKind.GIT_COMMIT,
            digest=TOKENIZER_COMMIT,
            evidence_source="fake tokenizer resolver",
        ),
    )
    return LoadingPlan(source=source, config=config or LoadConfig(), resolution=resolution)


class _FakeConfig:
    model_type = "fake_moe"
    _commit_hash = MODEL_COMMIT

    def __init__(self, *, close_log: list[str]) -> None:
        self._close_log = close_log

    def to_dict(self) -> dict[str, Any]:
        return {"model_type": self.model_type, "hidden_size": 8, "layers": 2}

    def close(self) -> None:
        self._close_log.append("config")


class _FakeTokenizer:
    _commit_hash = TOKENIZER_COMMIT

    def __init__(self, close_log: list[str]) -> None:
        self._close_log = close_log

    def close(self) -> None:
        self._close_log.append("tokenizer")


class _FakeModel:
    dtype = "float32"
    device = "cpu"

    def __init__(self, config: _FakeConfig, close_log: list[str]) -> None:
        self.config = config
        self.hf_device_map = {"": "cpu"}
        self._close_log = close_log
        self.to_calls: list[str] = []

    def named_modules(self):
        yield "", self

    def named_parameters(self):
        return iter(())

    def to(self, device: str) -> _FakeModel:
        self.to_calls.append(device)
        self.device = device
        self.hf_device_map = {"": device}
        return self

    def close(self) -> None:
        self._close_log.append("model")


def _fake_transformers(
    calls: list[tuple[str, str, dict[str, Any]]],
    close_log: list[str],
    *,
    model_factory: Any | None = None,
) -> types.ModuleType:
    module = types.ModuleType("transformers")

    class AutoConfig:
        @staticmethod
        def from_pretrained(target: str, **kwargs: Any) -> _FakeConfig:
            calls.append(("config", target, kwargs))
            return _FakeConfig(close_log=close_log)

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(target: str, **kwargs: Any) -> _FakeTokenizer:
            calls.append(("tokenizer", target, kwargs))
            return _FakeTokenizer(close_log)

    class AutoModel:
        @staticmethod
        def from_pretrained(target: str, **kwargs: Any) -> _FakeModel:
            calls.append(("model", target, kwargs))
            if model_factory is not None:
                return model_factory(target, kwargs)
            return _FakeModel(kwargs["config"], close_log)

    module.AutoConfig = AutoConfig
    module.AutoTokenizer = AutoTokenizer
    module.AutoModel = AutoModel
    return module


def _install_fake(monkeypatch: pytest.MonkeyPatch, module: types.ModuleType) -> None:
    monkeypatch.setitem(sys.modules, "transformers", module)


def test_invalid_resolution_quantization_and_local_path_fail_before_optional_imports(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    imported: list[str] = []
    original_import = model_loader.importlib.import_module

    def tracked_import(name: str, package: str | None = None) -> Any:
        imported.append(name)
        return original_import(name, package)

    monkeypatch.setattr(model_loader.importlib, "import_module", tracked_import)
    unresolved = _plan().model_copy(update={"resolution": None})
    with pytest.raises(RuntimeValidationError, match="immutable model revision"):
        load_huggingface(unresolved)
    assert imported == []

    quantized = _plan(config=LoadConfig(quantization="int4"))
    with pytest.raises(ModelLoadError, match="quantization"):
        load_huggingface(quantized)
    assert imported == []

    missing_local = _plan(SourceKind.LOCAL, path=str(tmp_path / "missing"))
    with pytest.raises(ModelLoadError, match="existing directory"):
        load_local(missing_local)
    assert imported == []


def test_model_factory_selection_preserves_declared_task_heads() -> None:
    assert (
        model_loader._model_factory_name(types.SimpleNamespace(architectures=["LingForCausalLM"]))
        == "AutoModelForCausalLM"
    )
    assert (
        model_loader._model_factory_name(
            types.SimpleNamespace(architectures=["T5ForConditionalGeneration"])
        )
        == "AutoModelForSeq2SeqLM"
    )
    assert (
        model_loader._model_factory_name(types.SimpleNamespace(is_encoder_decoder=True))
        == "AutoModelForSeq2SeqLM"
    )
    assert model_loader._model_factory_name(types.SimpleNamespace()) == "AutoModel"


def test_multimodal_conditional_factory_uses_vision_surface_before_seq2seq() -> None:
    assert (
        model_loader._model_factory_name(
            types.SimpleNamespace(
                architectures=["VisionForConditionalGeneration"],
                vision_config=types.SimpleNamespace(),
            )
        )
        == "AutoModelForMultimodalLM"
    )
    # An absent marker must preserve the existing encoder-decoder behavior.
    assert (
        model_loader._model_factory_name(
            types.SimpleNamespace(
                architectures=["T5ForConditionalGeneration"],
                vision_config=None,
            )
        )
        == "AutoModelForSeq2SeqLM"
    )


def test_declared_transformers_class_precedes_ambiguous_conditional_auto_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, dict[str, Any]]] = []
    close_log: list[str] = []
    module = _fake_transformers(calls, close_log)

    class Qwen3_5MoeForConditionalGeneration:
        @staticmethod
        def from_pretrained(target: str, **kwargs: Any) -> _FakeModel:
            calls.append(("declared-model", target, kwargs))
            return _FakeModel(kwargs["config"], close_log)

    module.Qwen3_5MoeForConditionalGeneration = Qwen3_5MoeForConditionalGeneration
    _install_fake(monkeypatch, module)
    monkeypatch.setattr(
        _FakeConfig,
        "architectures",
        ["Qwen3_5MoeForConditionalGeneration"],
        raising=False,
    )

    result = load_huggingface(_plan())
    assert calls[2][0] == "declared-model"
    result.close()


def test_qwen4_declared_transformers_class_wins_over_conditional_auto_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, dict[str, Any]]] = []
    close_log: list[str] = []
    module = _fake_transformers(calls, close_log)

    class Qwen4ExpForConditionalGeneration:
        @staticmethod
        def from_pretrained(target: str, **kwargs: Any) -> _FakeModel:
            calls.append(("qwen4-declared-model", target, kwargs))
            return _FakeModel(kwargs["config"], close_log)

    module.Qwen4ExpForConditionalGeneration = Qwen4ExpForConditionalGeneration
    _install_fake(monkeypatch, module)
    monkeypatch.setattr(
        _FakeConfig,
        "architectures",
        ["Qwen4ExpForConditionalGeneration"],
        raising=False,
    )

    result = load_huggingface(_plan())
    assert calls[2][0] == "qwen4-declared-model"
    result.close()


def test_huggingface_loader_reports_configuration_tokenizer_and_weight_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, dict[str, Any]]] = []
    close_log: list[str] = []
    _install_fake(monkeypatch, _fake_transformers(calls, close_log))
    progress: list[tuple[str, int, int, str]] = []

    result = load_huggingface(
        _plan(),
        progress_callback=lambda stage, completed, total, message: progress.append(
            (stage, completed, total, message)
        ),
    )

    assert [(stage, completed, total) for stage, completed, total, _ in progress] == [
        ("model_cache", 0, 3),
        ("model_cache", 1, 3),
        ("model_cache", 2, 3),
        ("model_cache", 3, 3),
    ]
    assert "weight shards" in progress[2][3]
    result.close()


def test_download_enabled_loader_uses_complete_immutable_cache_offline_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, dict[str, Any]]] = []
    close_log: list[str] = []
    _install_fake(monkeypatch, _fake_transformers(calls, close_log))
    config = LoadConfig(download_policy="allow_downloads", allow_downloads=True)

    result = load_huggingface(_plan(config=config))

    assert all(call[2]["local_files_only"] is True for call in calls)
    result.close()


def test_download_enabled_loader_falls_back_only_when_cache_is_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, dict[str, Any]]] = []
    close_log: list[str] = []
    module = _fake_transformers(calls, close_log)
    original = module.AutoConfig.from_pretrained

    def cache_sensitive_config(target: str, **kwargs: Any) -> _FakeConfig:
        calls.append(("config-attempt", target, kwargs))
        if kwargs["local_files_only"]:
            raise OSError("not found in cache while local_files_only=True")
        return original(target, **kwargs)

    module.AutoConfig.from_pretrained = staticmethod(cache_sensitive_config)
    _install_fake(monkeypatch, module)
    config = LoadConfig(download_policy="allow_downloads", allow_downloads=True)
    progress: list[tuple[str, int, int, str]] = []

    result = load_huggingface(
        _plan(config=config),
        progress_callback=lambda stage, completed, total, message: progress.append(
            (stage, completed, total, message)
        ),
    )

    attempts = [call for call in calls if call[0] == "config-attempt"]
    assert [call[2]["local_files_only"] for call in attempts] == [True, False]
    assert any(stage == "model_download" for stage, _, _, _ in progress)
    result.close()


@pytest.mark.parametrize(
    "architectures",
    [["_PrivateModel"], ["not-a-python-identifier"], [1], None],
)
def test_invalid_or_unavailable_declared_classes_fall_back_to_safe_auto_model(
    monkeypatch: pytest.MonkeyPatch,
    architectures: object,
) -> None:
    calls: list[tuple[str, str, dict[str, Any]]] = []
    close_log: list[str] = []
    module = _fake_transformers(calls, close_log)
    _install_fake(monkeypatch, module)
    monkeypatch.setattr(_FakeConfig, "architectures", architectures, raising=False)

    result = load_huggingface(_plan())
    assert calls[2][0] == "model"
    result.close()


def test_huggingface_uses_resolved_revisions_audited_kwargs_and_observed_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, dict[str, Any]]] = []
    close_log: list[str] = []
    _install_fake(monkeypatch, _fake_transformers(calls, close_log))
    fake_torch = types.ModuleType("torch")
    fake_torch.float16 = object()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    config = LoadConfig(dtype="float16", loader_options={"attn_implementation": "eager"})
    result = load_huggingface(_plan(config=config))

    assert calls[0][0:2] == ("config", MODEL_ID)
    assert calls[0][2] == {
        "local_files_only": True,
        "revision": MODEL_COMMIT,
        "trust_remote_code": False,
    }
    assert calls[1][0:2] == ("tokenizer", TOKENIZER_ID)
    assert calls[1][2]["revision"] == TOKENIZER_COMMIT
    assert calls[1][2]["local_files_only"] is True
    assert calls[2][0:2] == ("model", MODEL_ID)
    assert calls[2][2]["revision"] == MODEL_COMMIT
    assert calls[2][2]["torch_dtype"] is fake_torch.float16
    assert calls[2][2]["attn_implementation"] == "eager"
    assert calls[2][2]["config"] is not None
    assert any("requested dtype" in warning for warning in result.warnings)
    assert result.manifest.model_key == f"model:{MODEL_ID}@{MODEL_COMMIT}"
    assert result.manifest.tokenizer.revision == TOKENIZER_COMMIT
    assert result.manifest.dtype is DType.FLOAT32
    assert result.manifest.provenance is not None
    assert (
        result.manifest.provenance.metadata["model_revision_evidence_source"]
        == "sha256:" + hashlib.sha256(b"fake hub resolver").hexdigest()
    )
    assert result.manifest.provenance.metadata["parameter_dtype_inventory"] == {
        "audit_version": "1.0",
        "status": "unavailable",
        "reason": "model exposes no named parameters",
        "tensor_count": 0,
        "unsized_tensor_count": 0,
        "element_count": 0,
        "logical_bytes": 0,
        "mixed_dtype": False,
        "dtype_rows": [],
        "inventory_digest": None,
    }
    result.close()
    result.close()
    assert close_log == ["model", "tokenizer", "config"]


def test_local_uses_exact_directory_and_is_always_offline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str, dict[str, Any]]] = []
    close_log: list[str] = []
    _install_fake(monkeypatch, _fake_transformers(calls, close_log))
    source_path = tmp_path / "model"
    source_path.mkdir()
    result = load_local(_plan(SourceKind.LOCAL, path=str(source_path)))

    assert [target for _, target, _ in calls] == [str(source_path)] * 3
    assert all(kwargs["local_files_only"] is True for _, _, kwargs in calls)
    assert calls[0][2]["revision"] == MODEL_COMMIT
    assert calls[1][2]["revision"] == TOKENIZER_COMMIT
    encoded = result.manifest.to_json()
    assert str(source_path) not in encoded
    assert "FakeTokenizer" not in encoded
    result.close()


def test_online_and_remote_code_policy_is_forwarded_and_warned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, dict[str, Any]]] = []
    close_log: list[str] = []
    _install_fake(monkeypatch, _fake_transformers(calls, close_log))
    config = LoadConfig(
        download_policy=DownloadPolicy.ALLOW_DOWNLOADS,
        allow_downloads=True,
        trust_remote_code=True,
        remote_code_acknowledged=True,
    )
    result = load_huggingface(_plan(config=config))
    assert all(kwargs["local_files_only"] is True for _, _, kwargs in calls)
    assert all(kwargs["trust_remote_code"] is True for _, _, kwargs in calls)
    assert any("trust_remote_code" in warning for warning in result.warnings)
    assert any("downloads" in warning for warning in result.warnings)
    assert result.manifest.provenance is not None
    assert result.manifest.provenance.metadata["security_warnings"] == list(
        _plan(config=config).security_warnings
    )
    result.close()


def test_device_map_and_auto_require_accelerate_and_forward_copies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, dict[str, Any]]] = []
    close_log: list[str] = []
    _install_fake(monkeypatch, _fake_transformers(calls, close_log))
    monkeypatch.setitem(sys.modules, "accelerate", types.ModuleType("accelerate"))
    requested_map = {"layers.0": "cuda:0"}
    mapped = load_huggingface(_plan(config=LoadConfig(device="cuda", device_map=requested_map)))
    assert calls[2][2]["device_map"] == requested_map
    requested_map["layers.0"] = "cpu"
    assert calls[2][2]["device_map"] == {"layers.0": "cuda:0"}
    assert any("requested device_map" in warning for warning in mapped.warnings)
    mapped.close()

    calls.clear()
    automatic = load_huggingface(_plan(config=LoadConfig(device="auto")))
    assert calls[2][2]["device_map"] == "auto"
    automatic.close()


@pytest.mark.parametrize("device", ["cuda", "cuda:1"])
def test_cuda_request_warns_when_single_device_observation_remains_cpu(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    calls: list[tuple[str, str, dict[str, Any]]] = []
    close_log: list[str] = []

    class CpuAfterToModel(_FakeModel):
        def to(self, requested: str) -> _FakeModel:
            self.to_calls.append(requested)
            return self

    def model_factory(_target: str, kwargs: dict[str, Any]) -> CpuAfterToModel:
        return CpuAfterToModel(kwargs["config"], close_log)

    _install_fake(
        monkeypatch,
        _fake_transformers(calls, close_log, model_factory=model_factory),
    )
    result = load_huggingface(_plan(config=LoadConfig(device=device)))
    assert result.manifest.device_map == {"": "cpu"}
    assert any(f"requested device {device!r} differs" in warning for warning in result.warnings)
    result.close()


def test_missing_torch_and_accelerate_fail_before_factory_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, dict[str, Any]]] = []
    close_log: list[str] = []
    _install_fake(monkeypatch, _fake_transformers(calls, close_log))
    original_import = model_loader.importlib.import_module

    def no_torch(name: str, package: str | None = None) -> Any:
        if name == "torch":
            raise ModuleNotFoundError("torch missing")
        return original_import(name, package)

    monkeypatch.setattr(model_loader.importlib, "import_module", no_torch)
    with pytest.raises(ModelRuntimeDependencyError, match="torch"):
        load_huggingface(_plan())
    assert calls == []

    monkeypatch.setattr(model_loader.importlib, "import_module", original_import)
    requested = LoadConfig(device="cuda", device_map={"layers.0": "cuda:0"})

    def no_accelerate(name: str, package: str | None = None) -> Any:
        if name == "accelerate":
            raise ModuleNotFoundError("accelerate missing")
        return original_import(name, package)

    monkeypatch.setattr(model_loader.importlib, "import_module", no_accelerate)
    with pytest.raises(ModelRuntimeDependencyError, match="accelerate"):
        load_huggingface(_plan(config=requested))
    assert calls == []


def test_objects_without_close_are_still_owned_and_references_clear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoCloseConfig:
        model_type = "no_close"

        def to_dict(self) -> dict[str, str]:
            return {"model_type": "no_close"}

    class NoCloseModel:
        dtype = "float64"
        config = NoCloseConfig()
        hf_device_map = {"": "cpu"}

        def named_modules(self):
            yield "", self

        def to(self, _device: str) -> NoCloseModel:
            return self

    class NoCloseTokenizer:
        pass

    class AutoConfig:
        @staticmethod
        def from_pretrained(_target: str, **_kwargs: Any) -> NoCloseConfig:
            return NoCloseConfig()

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(_target: str, **_kwargs: Any) -> NoCloseTokenizer:
            return NoCloseTokenizer()

    class AutoModel:
        @staticmethod
        def from_pretrained(_target: str, **_kwargs: Any) -> NoCloseModel:
            return NoCloseModel()

    module = types.ModuleType("transformers")
    module.AutoConfig = AutoConfig
    module.AutoTokenizer = AutoTokenizer
    module.AutoModel = AutoModel
    _install_fake(monkeypatch, module)
    result = load_huggingface(_plan())
    model = result.model
    tokenizer = result.tokenizer
    result.close()
    assert result.closed is True
    assert result.model is None
    assert result.tokenizer is None
    assert model is not None and tokenizer is not None
    assert result.manifest.dtype is DType.FLOAT64


def test_cleanup_retries_only_failed_callbacks_and_system_exit_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, dict[str, Any]]] = []
    close_log: list[str] = []
    attempts = 0

    def model_factory(_target: str, kwargs: dict[str, Any]) -> _FakeModel:
        model = _FakeModel(kwargs["config"], close_log)

        def close() -> None:
            nonlocal attempts
            attempts += 1
            close_log.append("model")
            if attempts == 1:
                raise OSError("transient")

        model.close = close  # type: ignore[method-assign]
        return model

    _install_fake(
        monkeypatch,
        _fake_transformers(calls, close_log, model_factory=model_factory),
    )
    result = load_huggingface(_plan())
    with pytest.raises(RuntimeCleanupError, match="cleanup failed") as cleanup_error:
        result.close()
    assert isinstance(cleanup_error.value.failures[0], OSError)
    assert close_log == ["model", "tokenizer", "config"]
    result.close()
    assert close_log == ["model", "tokenizer", "config", "model"]
    assert result.closed is True

    close_log.clear()

    def exiting_model(_target: str, _kwargs: dict[str, Any]) -> Any:
        raise SystemExit("exit")

    _install_fake(
        monkeypatch,
        _fake_transformers(calls, close_log, model_factory=exiting_model),
    )
    with pytest.raises(SystemExit, match="exit"):
        load_huggingface(_plan())
    assert close_log == ["tokenizer", "config"]


def test_huggingface_rejects_sha_evidence_but_local_accepts_external_attestation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_calls: list[tuple[str, str, dict[str, Any]]] = []
    close_log: list[str] = []
    _install_fake(monkeypatch, _fake_transformers(fake_calls, close_log))
    source_plan = _plan()
    model_digest = "c" * 64
    tokenizer_digest = "d" * 64
    sha_resolution = source_plan.resolution.model_copy(
        update={
            "resolved_model_revision": "sha256:" + model_digest,
            "resolved_model_revision_evidence": ImmutableRevisionEvidence(
                kind=RevisionEvidenceKind.SHA256,
                digest=model_digest,
                evidence_source="content digest",
            ),
            "resolved_tokenizer_revision": "sha256:" + tokenizer_digest,
            "resolved_tokenizer_revision_evidence": ImmutableRevisionEvidence(
                kind=RevisionEvidenceKind.SHA256,
                digest=tokenizer_digest,
                evidence_source="tokenizer content digest",
            ),
        }
    )
    with pytest.raises(ModelLoadError, match="git-commit"):
        load_huggingface(
            LoadingPlan(
                source=source_plan.source,
                config=source_plan.config,
                resolution=sha_resolution,
            )
        )

    local_path = tmp_path / "model"
    local_path.mkdir()
    local_source_plan = _plan(SourceKind.LOCAL, path=str(local_path))
    local_result = load_local(
        LoadingPlan(
            source=local_source_plan.source,
            config=local_source_plan.config,
            resolution=sha_resolution.model_copy(
                update={"source_type": SourceKind.LOCAL, "model_id": MODEL_ID}
            ),
        )
    )
    assert local_result.manifest.revision == "sha256:" + model_digest
    local_result.close()


def test_exposed_huggingface_commit_mismatch_is_typed_and_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, dict[str, Any]]] = []
    close_log: list[str] = []

    class BadCommitConfig(_FakeConfig):
        _commit_hash = "c" * 40

    def bad_commit_model(_target: str, _kwargs: dict[str, Any]) -> _FakeModel:
        return _FakeModel(BadCommitConfig(close_log=close_log), close_log)

    _install_fake(
        monkeypatch,
        _fake_transformers(calls, close_log, model_factory=bad_commit_model),
    )
    with pytest.raises(ModelObservationError, match="does not match immutable resolution"):
        load_huggingface(_plan())
    assert close_log == ["model", "tokenizer", "config"]


@pytest.mark.parametrize(
    "options",
    [
        {"Token": "super-secret"},
        {"accessToken": "super-secret"},
        {"authToken": "super-secret"},
        {"Authentication": "Bearer super-secret"},
        {"nested": {"Authorization": "Bearer super-secret"}},
        {"nested": {"api-key": "super-secret"}},
        {"nested": {"apiKey": "super-secret"}},
        {"nested": {"xApiKey": "super-secret"}},
        {"nested": {"APIKEY": "super-secret"}},
        {"nested": {"password": "super-secret"}},
        {"nested": {"clientSecret": "super-secret"}},
        {"nested": {"client_secret": "super-secret"}},
        {"headers": {"Authorization": "Bearer super-secret"}},
        {"request_headers": {"x-custom": "super-secret"}},
        {"HTTPHeaders": {"Authorization": "Bearer super-secret"}},
    ],
)
def test_credential_shaped_loader_options_are_case_insensitive_and_never_leak(
    options: dict[str, Any],
) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="credential-bearing|reserved") as caught:
        LoadConfig(loader_options=options)
    assert "super-secret" not in str(caught.value)


def test_benign_nested_backend_options_round_trip_in_plan_json() -> None:
    config = LoadConfig(
        loader_options={
            "backend": {
                "dtype": "float16",
                "revision": "backend-specific",
                "secret_sauce_mode": "fast",
            }
        }
    )
    plan = _plan(config=config)
    restored = LoadingPlan.from_json(plan.to_json())
    assert restored.config.loader_options == config.loader_options


def test_plan_json_rejects_nested_mixed_case_credentials_without_echoing_values() -> None:
    from pydantic import ValidationError

    plan = _plan(config=LoadConfig(loader_options={"backend": {"dtype": "float16"}}))
    payload = plan.to_dict()
    payload["config"]["loader_options"] = {
        "backend": {"headers": {"Authorization": "Bearer plan-secret"}}
    }
    with pytest.raises(ValidationError, match="credential-bearing|reserved") as caught:
        LoadingPlan.from_json(json.dumps(payload))
    assert "plan-secret" not in str(caught.value)


def test_mutated_loader_policy_and_options_fail_before_optional_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []
    calls: list[tuple[str, str, dict[str, Any]]] = []
    original_import = model_loader.importlib.import_module

    def tracked_import(name: str, package: str | None = None) -> Any:
        imported.append(name)
        return original_import(name, package)

    monkeypatch.setattr(model_loader.importlib, "import_module", tracked_import)
    plan = _plan()

    for loader_options in (
        {"Revision": "attacker-branch"},
        {"nested": {"Authorization": "Bearer attacker"}},
    ):
        mutated_config = plan.config.model_copy(update={"loader_options": loader_options})
        mutated_plan = plan.model_copy(update={"config": mutated_config})
        with pytest.raises(ModelLoadError, match="loader_options"):
            load_huggingface(mutated_plan)

    mutated_config = plan.config.model_copy(
        update={"allow_downloads": True, "download_policy": DownloadPolicy.ALLOW_DOWNLOADS}
    )
    mutated_plan = plan.model_copy(update={"config": mutated_config})
    with pytest.raises(ModelLoadError, match="download policies"):
        load_huggingface(mutated_plan)
    assert imported == []
    assert calls == []


def test_reserved_credentials_and_observation_failure_never_leak_or_skip_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pydantic import ValidationError

    for options in (
        {"token": "secret"},
        {"nested": {"access_token": "secret"}},
        {"nested": {"secret_sauce_mode": "fast"}},
    ):
        if "secret_sauce_mode" in options.get("nested", {}):
            assert LoadConfig(loader_options=options).loader_options == options
        else:
            with pytest.raises(ValidationError, match="credential-bearing|reserved"):
                LoadConfig(loader_options=options)

    close_log: list[str] = []
    calls: list[tuple[str, str, dict[str, Any]]] = []

    class BadModel(_FakeModel):
        @property
        def config(self) -> Any:
            raise ValueError("bad config observation")

        @config.setter
        def config(self, _value: Any) -> None:
            pass

    def bad_model(_target: str, kwargs: dict[str, Any]) -> BadModel:
        return BadModel(kwargs["config"], close_log)

    _install_fake(
        monkeypatch,
        _fake_transformers(calls, close_log, model_factory=bad_model),
    )
    with pytest.raises(ModelObservationError, match="model.config"):
        load_huggingface(_plan())
    assert close_log == ["model", "tokenizer", "config"]


def test_observation_failure_cleanup_failure_attaches_retryable_pending_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    close_log: list[str] = []
    attempts = 0
    calls: list[tuple[str, str, dict[str, Any]]] = []

    class BadModel(_FakeModel):
        @property
        def config(self) -> Any:
            raise ValueError("bad config observation")

        @config.setter
        def config(self, _value: Any) -> None:
            pass

        def close(self) -> None:
            nonlocal attempts
            attempts += 1
            close_log.append("model")
            if attempts == 1:
                raise OSError("model cleanup temporarily unavailable")

    def bad_model(_target: str, kwargs: dict[str, Any]) -> BadModel:
        return BadModel(kwargs["config"], close_log)

    _install_fake(
        monkeypatch,
        _fake_transformers(calls, close_log, model_factory=bad_model),
    )
    with pytest.raises(ModelObservationError, match="model.config") as caught:
        load_huggingface(_plan())
    assert close_log == ["model", "tokenizer", "config"]
    pending = caught.value.pending_cleanup
    assert pending.pending is True
    pending.retry()
    pending.retry()
    assert close_log == ["model", "tokenizer", "config", "model"]
    assert pending.pending is False


def test_missing_optional_dependency_is_typed_and_chained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(name: str, package: str | None = None) -> Any:
        if name == "transformers":
            raise ModuleNotFoundError("no transformers")
        return __import__(name)

    monkeypatch.setattr(model_loader.importlib, "import_module", missing)
    with pytest.raises(
        ModelRuntimeDependencyError, match=r"pip install 'moeatlas\[model\]'"
    ) as caught:
        load_huggingface(_plan())
    assert isinstance(caught.value.__cause__, ImportError)


def test_observation_falls_back_to_parameter_dtype_and_warns_on_unknown_architecture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, dict[str, Any]]] = []
    close_log: list[str] = []
    fake = _fake_transformers(calls, close_log)

    class ModelWithoutDtype(_FakeModel):
        def __init__(self, config: _FakeConfig, log: list[str]) -> None:
            super().__init__(config, log)
            self.__dict__.pop("hf_device_map", None)
            self.__dict__.pop("device", None)

        @property
        def dtype(self) -> None:
            return None

        def named_parameters(self):
            return iter([("weight", types.SimpleNamespace(dtype="float32"))])

        def to(self, _device: str) -> ModelWithoutDtype:
            return self

    class NoModelTypeConfig(_FakeConfig):
        model_type = None

    def model_factory(_target: str, kwargs: dict[str, Any]) -> ModelWithoutDtype:
        config = NoModelTypeConfig(close_log=close_log)
        model = ModelWithoutDtype(config, close_log)
        model.device = "cuda:0"
        return model

    # Replace only the model factory while retaining the ordinary call recorders.
    fake = _fake_transformers(calls, close_log, model_factory=model_factory)
    _install_fake(monkeypatch, fake)
    result = load_huggingface(_plan())
    assert result.manifest.dtype is DType.FLOAT32
    assert any("model class" in warning for warning in result.warnings)
    assert any("requested device" in warning for warning in result.warnings)
    result.close()


def test_partial_model_failure_rolls_back_reverse_order_and_preserves_control_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    close_log: list[str] = []
    calls: list[tuple[str, str, dict[str, Any]]] = []

    def failing_model(_target: str, _kwargs: dict[str, Any]) -> Any:
        raise ValueError("checkpoint failure")

    _install_fake(
        monkeypatch,
        _fake_transformers(calls, close_log, model_factory=failing_model),
    )
    with pytest.raises(ModelLoadError, match="model") as caught:
        load_huggingface(_plan())
    assert isinstance(caught.value.__cause__, ValueError)
    assert close_log == ["tokenizer", "config"]

    def interrupting_model(_target: str, _kwargs: dict[str, Any]) -> Any:
        raise KeyboardInterrupt("cancelled")

    close_log.clear()
    calls.clear()
    _install_fake(
        monkeypatch,
        _fake_transformers(calls, close_log, model_factory=interrupting_model),
    )
    with pytest.raises(KeyboardInterrupt, match="cancelled"):
        load_huggingface(_plan())
    assert close_log == ["tokenizer", "config"]


def test_fake_loader_performs_no_network_or_cache_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str, dict[str, Any]]] = []
    close_log: list[str] = []
    _install_fake(monkeypatch, _fake_transformers(calls, close_log))
    cache_home = tmp_path / "hf-home"
    cache_transformers = tmp_path / "transformers-cache"
    cache_hub = tmp_path / "hub-cache"
    for name, value in (
        ("HF_HOME", cache_home),
        ("TRANSFORMERS_CACHE", cache_transformers),
        ("HUGGINGFACE_HUB_CACHE", cache_hub),
    ):
        monkeypatch.setenv(name, str(value))

    def fail_network(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("network access is forbidden in model-free tests")

    monkeypatch.setattr(socket, "create_connection", fail_network)
    monkeypatch.setattr(urllib.request, "urlopen", fail_network)
    result = load_huggingface(_plan())
    result.close()
    assert not list(tmp_path.rglob("*"))
