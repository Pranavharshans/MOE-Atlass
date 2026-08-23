from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Any

import pytest

import moeatlas.runtime.loader as loader_module
from moeatlas.core import DType, make_config_hash, make_model_key
from moeatlas.fixtures import SyntheticMoE
from moeatlas.loading import (
    CustomLoaderSource,
    ImmutableRevisionEvidence,
    InstanceSource,
    LoadingPlan,
    ResolvedSource,
    RevisionEvidenceKind,
    SourceKind,
    TokenizerRequest,
)
from moeatlas.runtime import (
    CustomLoaderExecutionError,
    LoadedModel,
    PendingRuntimeCleanup,
    RuntimeArtifacts,
    RuntimeCleanupError,
    RuntimeLoadError,
    RuntimeValidationError,
    load_custom,
    load_instance,
)

MODEL_ID = "fixture/runtime-moe"
TOKENIZER_ID = "fixture/runtime-tokenizer"
REQUESTED_MODEL_REVISION = "requested-ref"
REQUESTED_TOKENIZER_REVISION = "tokenizer-ref"
RESOLVED_MODEL_REVISION = "a" * 40
RESOLVED_TOKENIZER_REVISION = "b" * 40


@dataclass
class TokenizerStub:
    name: str = "runtime-tokenizer"


@dataclass
class ConfigStub:
    scale: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {"nested": {"scale": self.scale}, "layers": 2}


def _evidence(kind: RevisionEvidenceKind, digest: str, source: str) -> ImmutableRevisionEvidence:
    return ImmutableRevisionEvidence(kind=kind, digest=digest, evidence_source=source)


def _source(source_type: SourceKind, *, tokenizer: bool = True):
    tokenizer_request = (
        TokenizerRequest(
            identifier=TOKENIZER_ID,
            requested_revision=REQUESTED_TOKENIZER_REVISION,
        )
        if tokenizer
        else None
    )
    common = {
        "model_id": MODEL_ID,
        "requested_revision": REQUESTED_MODEL_REVISION,
        "tokenizer": tokenizer_request,
    }
    if source_type is SourceKind.INSTANCE:
        return InstanceSource(**common)
    return CustomLoaderSource(loader_reference="tests.runtime_fixture:build", **common)


def _plan(
    source_type: SourceKind = SourceKind.INSTANCE,
    *,
    tokenizer: bool = True,
    resolved_tokenizer: bool = True,
) -> LoadingPlan:
    source = _source(source_type, tokenizer=tokenizer)
    resolution = ResolvedSource(
        source_type=source_type,
        model_id=MODEL_ID,
        requested_model_revision=REQUESTED_MODEL_REVISION,
        resolved_model_revision=RESOLVED_MODEL_REVISION,
        resolved_model_revision_evidence=_evidence(
            RevisionEvidenceKind.GIT_COMMIT,
            RESOLVED_MODEL_REVISION,
            "runtime test resolver",
        ),
        requested_tokenizer_revision=REQUESTED_TOKENIZER_REVISION if tokenizer else None,
        resolved_tokenizer_revision=(
            RESOLVED_TOKENIZER_REVISION if tokenizer and resolved_tokenizer else None
        ),
        resolved_tokenizer_revision_evidence=(
            _evidence(
                RevisionEvidenceKind.GIT_COMMIT,
                RESOLVED_TOKENIZER_REVISION,
                "runtime tokenizer resolver",
            )
            if tokenizer and resolved_tokenizer
            else None
        ),
    )
    return LoadingPlan(source=source, resolution=resolution)


def _artifacts(
    *,
    model: object | None = None,
    tokenizer: object | None = None,
    config: object | None = None,
    architecture: str = "synthetic_moe",
    dtype: DType = DType.FLOAT32,
    device_map: dict[str, str] | None = None,
    cleanup=None,
    owns_cleanup: bool = False,
) -> RuntimeArtifacts:
    return RuntimeArtifacts(
        model=SyntheticMoE() if model is None else model,
        tokenizer=TokenizerStub() if tokenizer is None else tokenizer,
        config=ConfigStub() if config is None else config,
        architecture=architecture,
        dtype=dtype,
        device_map={"": "cpu"} if device_map is None else device_map,
        cleanup=cleanup,
        owns_cleanup=owns_cleanup,
    )


def test_instance_load_builds_observed_manifest_without_serializing_runtime_objects() -> None:
    model = SyntheticMoE()
    tokenizer = TokenizerStub()
    config = {"layers": 2, "nested": {"scale": 1.0}}
    plan = _plan()
    result = load_instance(
        plan,
        model,
        tokenizer,
        observation=RuntimeArtifacts(
            model=model,
            tokenizer=tokenizer,
            config=config,
            architecture="synthetic_moe",
            dtype=DType.FLOAT32,
            device_map={"": "cpu"},
        ),
    )

    assert isinstance(result, LoadedModel)
    assert result.manifest.model_key == make_model_key(MODEL_ID, RESOLVED_MODEL_REVISION)
    assert result.manifest.revision == RESOLVED_MODEL_REVISION
    assert result.manifest.config_hash == make_config_hash(config)
    assert result.manifest.tokenizer.identifier == TOKENIZER_ID
    assert result.manifest.tokenizer.revision == RESOLVED_TOKENIZER_REVISION
    assert result.manifest.dtype is DType.FLOAT32
    assert result.manifest.device_map == {"": "cpu"}
    assert result.manifest.provenance is not None
    assert result.manifest.provenance.source == "runtime:instance"
    assert result.manifest.provenance.metadata == {
        "loading_plan_id": plan.plan_id,
        "model_revision_evidence": "git_commit",
        "resolution_method": "external_resolver",
        "security_warnings": [],
        "source_type": "instance",
        "tokenizer_revision_evidence": "git_commit",
    }
    encoded = result.manifest.to_json()
    assert "SyntheticMoE" not in encoded
    assert "TokenizerStub" not in encoded
    result.close()


def test_instance_objects_are_caller_owned_and_close_is_idempotent() -> None:
    model = SyntheticMoE()
    tokenizer = TokenizerStub()
    result = load_instance(
        _plan(),
        model,
        tokenizer,
        observation=_artifacts(model=model, tokenizer=tokenizer),
    )

    result.close()
    result.close()
    assert result.closed is True
    assert result.model is model
    assert result.tokenizer is tokenizer


def test_runtime_artifacts_require_unambiguous_cleanup_ownership() -> None:
    with pytest.raises(ValueError, match="provided together"):
        _artifacts(cleanup=lambda: None)
    with pytest.raises(ValueError, match="provided together"):
        _artifacts(owns_cleanup=True)


def test_explicit_cleanup_ownership_retries_failures_and_drops_owned_references() -> None:
    calls = 0

    def cleanup() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("transient cleanup")

    model = SyntheticMoE()
    tokenizer = TokenizerStub()
    result = load_instance(
        _plan(),
        model,
        tokenizer,
        observation=_artifacts(
            model=model,
            tokenizer=tokenizer,
            cleanup=cleanup,
            owns_cleanup=True,
        ),
    )
    with pytest.raises(RuntimeCleanupError, match="cleanup failed"):
        result.close()
    assert calls == 1
    assert result.closed is False
    assert result.owns_cleanup is True
    result.close()
    result.close()
    assert calls == 2
    assert result.closed is True
    assert result.model is None
    assert result.tokenizer is None


def test_pending_runtime_cleanup_keeps_permanent_failures_retryable() -> None:
    calls = 0

    def permanent_failure() -> None:
        nonlocal calls
        calls += 1
        raise OSError("permanent cleanup")

    pending = PendingRuntimeCleanup(permanent_failure)
    with pytest.raises(RuntimeCleanupError) as first:
        pending.retry()
    assert pending.pending is True
    assert first.value.pending_cleanup is pending
    with pytest.raises(RuntimeCleanupError):
        pending.retry()
    assert pending.pending is True
    assert calls == 2


def test_context_cleanup_failure_does_not_hide_body_error_and_remains_retryable() -> None:
    calls = 0

    def cleanup() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("temporary cleanup")

    model = SyntheticMoE()
    tokenizer = TokenizerStub()
    result = load_instance(
        _plan(),
        model,
        tokenizer,
        observation=_artifacts(
            model=model,
            tokenizer=tokenizer,
            cleanup=cleanup,
            owns_cleanup=True,
        ),
    )
    with pytest.raises(ValueError, match="body failure") as caught:
        with result:
            raise ValueError("body failure")
    assert any("cleanup also failed" in note for note in caught.value.__notes__)
    assert result.closed is False
    result.close()
    assert calls == 2


def test_runtime_source_and_resolution_boundaries_are_explicit() -> None:
    with pytest.raises(RuntimeLoadError, match="InstanceSource"):
        load_instance(
            _plan(SourceKind.CUSTOM),
            SyntheticMoE(),
            TokenizerStub(),
            observation=_artifacts(),
        )
    with pytest.raises(RuntimeLoadError, match="CustomLoaderSource"):
        load_custom(_plan(SourceKind.INSTANCE), execute_user_code=True)

    unresolved = _plan().model_copy(update={"resolution": None})
    unresolved_model = SyntheticMoE()
    unresolved_tokenizer = TokenizerStub()
    with pytest.raises(RuntimeValidationError, match="immutable model revision"):
        load_instance(
            unresolved,
            unresolved_model,
            unresolved_tokenizer,
            observation=_artifacts(model=unresolved_model, tokenizer=unresolved_tokenizer),
        )

    no_tokenizer = _plan(tokenizer=False)
    no_tokenizer_model = SyntheticMoE()
    no_tokenizer_tokenizer = TokenizerStub()
    with pytest.raises(RuntimeValidationError, match="tokenizer request"):
        load_instance(
            no_tokenizer,
            no_tokenizer_model,
            no_tokenizer_tokenizer,
            observation=_artifacts(
                model=no_tokenizer_model,
                tokenizer=no_tokenizer_tokenizer,
            ),
        )

    no_resolved_tokenizer = _plan(resolved_tokenizer=False)
    no_resolved_model = SyntheticMoE()
    no_resolved_tokenizer_obj = TokenizerStub()
    with pytest.raises(RuntimeValidationError, match="resolved tokenizer evidence"):
        load_instance(
            no_resolved_tokenizer,
            no_resolved_model,
            no_resolved_tokenizer_obj,
            observation=_artifacts(
                model=no_resolved_model,
                tokenizer=no_resolved_tokenizer_obj,
            ),
        )


def test_runtime_observation_preflight_does_not_iterate_or_forward() -> None:
    class Surface:
        def __init__(self) -> None:
            self.lookups = 0

        def named_modules(self):
            self.lookups += 1
            raise AssertionError("named_modules must not be iterated in Feature 8")

    model = Surface()
    tokenizer = TokenizerStub()
    result = load_instance(
        _plan(),
        model,
        tokenizer,
        observation=_artifacts(model=model, tokenizer=tokenizer),
    )
    assert model.lookups == 0
    result.close()

    invalid_model = object()
    invalid_tokenizer = TokenizerStub()
    with pytest.raises(RuntimeValidationError, match="callable named_modules"):
        load_instance(
            _plan(),
            invalid_model,
            invalid_tokenizer,
            observation=_artifacts(model=invalid_model, tokenizer=invalid_tokenizer),
        )


def test_runtime_observation_and_config_validation_is_strict() -> None:
    model = SyntheticMoE()
    tokenizer = TokenizerStub()
    nonfinite = load_instance(
        _plan(),
        model,
        tokenizer,
        observation=_artifacts(
            model=model,
            tokenizer=tokenizer,
            config={
                "limits": [float("inf"), float("-inf"), float("nan")],
            },
        ),
    )
    assert nonfinite.manifest.config_hash == make_config_hash(
        {
            "limits": [
                {"__moeatlas_non_finite_float__": "positive_infinity"},
                {"__moeatlas_non_finite_float__": "negative_infinity"},
                {"__moeatlas_non_finite_float__": "nan"},
            ]
        }
    )
    nonfinite.close()
    model = SyntheticMoE()
    tokenizer = TokenizerStub()
    result = load_instance(
        _plan(),
        model,
        tokenizer,
        observation=_artifacts(
            model=model,
            tokenizer=tokenizer,
            config={"nested": {1: "canonicalized", 2: "also-canonicalized"}},
        ),
    )
    assert result.manifest.config_hash == make_config_hash(
        {"nested": {"1": "canonicalized", "2": "also-canonicalized"}}
    )
    result.close()

    with pytest.raises(RuntimeValidationError, match="collide after JSON string normalization"):
        model = SyntheticMoE()
        tokenizer = TokenizerStub()
        load_instance(
            _plan(),
            model,
            tokenizer,
            observation=_artifacts(
                model=model,
                tokenizer=tokenizer,
                config={"nested": {1: "integer", "1": "string"}},
            ),
        )
    with pytest.raises(RuntimeValidationError, match="config"):
        model = SyntheticMoE()
        tokenizer = TokenizerStub()
        load_instance(
            _plan(),
            model,
            tokenizer,
            observation=_artifacts(model=model, tokenizer=tokenizer, config=object()),
        )
    with pytest.raises(RuntimeValidationError, match="device_map"):
        model = SyntheticMoE()
        tokenizer = TokenizerStub()
        load_instance(
            _plan(),
            model,
            tokenizer,
            observation=_artifacts(
                model=model,
                tokenizer=tokenizer,
                device_map={"": "cpu 0"},
            ),
        )
    with pytest.raises(RuntimeValidationError, match="architecture"):
        model = SyntheticMoE()
        tokenizer = TokenizerStub()
        load_instance(
            _plan(),
            model,
            tokenizer,
            observation=_artifacts(
                model=model,
                tokenizer=tokenizer,
                architecture="bad architecture",
            ),
        )

    with pytest.raises(TypeError, match="core DType"):
        RuntimeArtifacts(
            model=SyntheticMoE(),
            tokenizer=TokenizerStub(),
            config={},
            architecture="synthetic_moe",
            dtype="float32",  # type: ignore[arg-type]
            device_map={"": "cpu"},
        )


def test_custom_loader_requires_opt_in_and_executes_exact_reference(monkeypatch) -> None:
    module_name = "moeatlas_test_custom_loader"
    module = types.ModuleType(module_name)
    calls: list[LoadingPlan] = []
    expected_model = SyntheticMoE()
    expected_tokenizer = TokenizerStub()

    def build(plan: LoadingPlan) -> RuntimeArtifacts:
        calls.append(plan)
        return _artifacts(model=expected_model, tokenizer=expected_tokenizer)

    module.build = build  # type: ignore[attr-defined]
    import_calls: list[str] = []
    original_import = loader_module.importlib.import_module

    def tracked_import(name: str, package: str | None = None):
        import_calls.append(name)
        return original_import(name, package)

    monkeypatch.setattr(loader_module.importlib, "import_module", tracked_import)
    plan = _plan(SourceKind.CUSTOM)
    with pytest.raises(CustomLoaderExecutionError, match="disabled"):
        load_custom(plan)
    assert import_calls == []

    sys.modules[module_name] = module
    try:
        plan = LoadingPlan(
            source=CustomLoaderSource(
                model_id=MODEL_ID,
                requested_revision=REQUESTED_MODEL_REVISION,
                tokenizer=TokenizerRequest(
                    identifier=TOKENIZER_ID,
                    requested_revision=REQUESTED_TOKENIZER_REVISION,
                ),
                loader_reference=f"{module_name}:build",
            ),
            resolution=_plan(SourceKind.CUSTOM).resolution,
        )
        result = load_custom(plan, execute_user_code=True)
        assert import_calls == [module_name]
        assert calls == [plan]
        assert result.model is expected_model
        assert result.warnings == (
            "custom loader reference execution runs user code and is not imported by validation",
        )
        result.close()
    finally:
        sys.modules.pop(module_name, None)


def test_custom_loader_errors_are_typed_and_preserve_causes(monkeypatch) -> None:
    missing_plan = LoadingPlan(
        source=CustomLoaderSource(
            model_id=MODEL_ID,
            requested_revision=REQUESTED_MODEL_REVISION,
            tokenizer=TokenizerRequest(
                identifier=TOKENIZER_ID,
                requested_revision=REQUESTED_TOKENIZER_REVISION,
            ),
            loader_reference="module_that_does_not_exist_987:build",
        ),
        resolution=_plan(SourceKind.CUSTOM).resolution,
    )
    with pytest.raises(CustomLoaderExecutionError) as missing:
        load_custom(missing_plan, execute_user_code=True)
    assert isinstance(missing.value.__cause__, ModuleNotFoundError)

    module_name = "moeatlas_test_loader_errors"
    module = types.ModuleType(module_name)
    module.not_callable = "nope"

    def raises(_plan: LoadingPlan) -> RuntimeArtifacts:
        raise ValueError("loader boom")

    def wrong_return(_plan: LoadingPlan):
        return {"model": "not artifacts"}

    def interrupts(_plan: LoadingPlan) -> RuntimeArtifacts:
        raise KeyboardInterrupt("stop")

    def exits(_plan: LoadingPlan) -> RuntimeArtifacts:
        raise SystemExit("exit")

    module.raises = raises  # type: ignore[attr-defined]
    module.wrong_return = wrong_return  # type: ignore[attr-defined]
    module.interrupts = interrupts  # type: ignore[attr-defined]
    module.exits = exits  # type: ignore[attr-defined]
    sys.modules[module_name] = module
    try:

        def custom_plan(function: str) -> LoadingPlan:
            return LoadingPlan(
                source=CustomLoaderSource(
                    model_id=MODEL_ID,
                    requested_revision=REQUESTED_MODEL_REVISION,
                    tokenizer=TokenizerRequest(
                        identifier=TOKENIZER_ID,
                        requested_revision=REQUESTED_TOKENIZER_REVISION,
                    ),
                    loader_reference=f"{module_name}:{function}",
                ),
                resolution=_plan(SourceKind.CUSTOM).resolution,
            )

        with pytest.raises(CustomLoaderExecutionError, match="not callable"):
            load_custom(custom_plan("not_callable"), execute_user_code=True)
        with pytest.raises(CustomLoaderExecutionError, match="execution failed") as failed:
            load_custom(custom_plan("raises"), execute_user_code=True)
        assert isinstance(failed.value.__cause__, ValueError)
        with pytest.raises(CustomLoaderExecutionError, match="exactly RuntimeArtifacts"):
            load_custom(custom_plan("wrong_return"), execute_user_code=True)
        with pytest.raises(KeyboardInterrupt, match="stop"):
            load_custom(custom_plan("interrupts"), execute_user_code=True)
        with pytest.raises(SystemExit, match="exit"):
            load_custom(custom_plan("exits"), execute_user_code=True)
    finally:
        sys.modules.pop(module_name, None)


def test_custom_post_validation_failure_attempts_owned_cleanup_and_notes_failure() -> None:
    module_name = "moeatlas_test_loader_cleanup"
    module = types.ModuleType(module_name)
    calls: list[str] = []

    def cleanup() -> None:
        calls.append("cleanup")
        if len(calls) == 1:
            raise OSError("cleanup failure")

    def build(_plan: LoadingPlan) -> RuntimeArtifacts:
        return _artifacts(
            model=object(),
            cleanup=cleanup,
            owns_cleanup=True,
        )

    module.build = build  # type: ignore[attr-defined]
    sys.modules[module_name] = module
    try:
        source_plan = LoadingPlan(
            source=CustomLoaderSource(
                model_id=MODEL_ID,
                requested_revision=REQUESTED_MODEL_REVISION,
                tokenizer=TokenizerRequest(
                    identifier=TOKENIZER_ID,
                    requested_revision=REQUESTED_TOKENIZER_REVISION,
                ),
                loader_reference=f"{module_name}:build",
            ),
            resolution=_plan(SourceKind.CUSTOM).resolution,
        )
        with pytest.raises(RuntimeValidationError, match="named_modules") as failed:
            load_custom(source_plan, execute_user_code=True)
        assert calls == ["cleanup"]
        assert any("cleanup also failed" in note for note in failed.value.__notes__)
        pending = failed.value.pending_cleanup
        assert pending.pending is True
        pending.retry()
        assert pending.pending is False
        pending.retry()
        assert calls == ["cleanup", "cleanup"]
    finally:
        sys.modules.pop(module_name, None)


def test_validation_failure_cleanup_success_runs_once_without_pending_handle() -> None:
    calls: list[str] = []
    module_name = "moeatlas_test_loader_cleanup_success"
    module = types.ModuleType(module_name)

    def build(_plan: LoadingPlan) -> RuntimeArtifacts:
        return _artifacts(
            model=object(),
            cleanup=lambda: calls.append("cleanup"),
            owns_cleanup=True,
        )

    module.build = build  # type: ignore[attr-defined]
    sys.modules[module_name] = module
    try:
        source_plan = LoadingPlan(
            source=CustomLoaderSource(
                model_id=MODEL_ID,
                requested_revision=REQUESTED_MODEL_REVISION,
                tokenizer=TokenizerRequest(
                    identifier=TOKENIZER_ID,
                    requested_revision=REQUESTED_TOKENIZER_REVISION,
                ),
                loader_reference=f"{module_name}:build",
            ),
            resolution=_plan(SourceKind.CUSTOM).resolution,
        )
        with pytest.raises(RuntimeValidationError, match="named_modules") as failed:
            load_custom(source_plan, execute_user_code=True)
        assert calls == ["cleanup"]
        assert not hasattr(failed.value, "pending_cleanup")
    finally:
        sys.modules.pop(module_name, None)


def test_runtime_plan_and_manifest_remain_runtime_object_free() -> None:
    plan = _plan()
    artifacts = _artifacts()
    result = load_instance(plan, artifacts.model, artifacts.tokenizer, observation=artifacts)
    assert "RuntimeArtifacts" not in plan.to_json()
    assert "SyntheticMoE" not in result.manifest.to_json()
    result.close()


def test_runtime_module_does_not_import_model_dependencies() -> None:
    assert not any(name in sys.modules for name in ("torch", "transformers", "safetensors"))
