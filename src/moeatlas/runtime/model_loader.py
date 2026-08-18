"""Lazy Hugging Face and local Transformers loading.

The optional runtime is imported only after source, immutable-resolution, and
loading-policy preflight. Observation helpers live in ``observation.py`` so
this module remains focused on staged calls and transactional ownership.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Mapping
from typing import Any

from ..loading import (
    DeviceKind,
    DTypePolicy,
    HuggingFaceSource,
    LoadingPlan,
    LocalSource,
    QuantizationPolicy,
    ResolvedSource,
    RevisionEvidenceKind,
    _is_credential_loader_key,
    _is_reserved_loader_option_key,
)
from .contracts import (
    LoadedModel,
    ModelLoadError,
    ModelRuntimeDependencyError,
    PendingRuntimeCleanup,
    RuntimeArtifacts,
    RuntimeCleanupError,
    RuntimeLoadError,
    _add_cleanup_note,
    _attach_pending_cleanup,
)
from .loader import _load_artifacts, _resolution_for
from .observation import (
    model_config,
    observed_architecture,
    observed_device_map,
    observed_dtype,
    requested_device_warnings,
    requested_dtype_warnings,
    safe_evidence_source,
    validate_exposed_commits,
)


class _CleanupStack:
    """Reverse-order cleanup retaining only callbacks that failed."""

    def __init__(self) -> None:
        self._callbacks: list[Callable[[], None]] = []
        self._object_ids: set[int] = set()

    @property
    def pending(self) -> bool:
        return bool(self._callbacks)

    def add_object(self, value: object) -> None:
        """Own every acquired object; objects without close use a no-op."""

        identity = id(value)
        if identity in self._object_ids:
            return
        try:
            close = getattr(value, "close", None)
        except Exception:
            close = None
        self._object_ids.add(identity)
        self._callbacks.append(close if callable(close) else lambda: None)

    def __call__(self) -> None:
        failures: list[BaseException] = []
        failed_in_reverse: list[Callable[[], None]] = []
        for callback in reversed(self._callbacks):
            try:
                callback()
            except BaseException as exc:
                failures.append(exc)
                failed_in_reverse.append(callback)
        self._callbacks = list(reversed(failed_in_reverse))
        if failures:
            raise RuntimeCleanupError(tuple(failures))


def _cleanup_stack_on_failure(stack: _CleanupStack, original: BaseException) -> None:
    if not stack.pending:
        return
    pending = PendingRuntimeCleanup(stack)
    try:
        pending.retry()
    except RuntimeCleanupError as cleanup_error:
        _attach_pending_cleanup(original, pending)
        _add_cleanup_note(original, cleanup_error)


def _import_optional(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except Exception as exc:
        raise ModelRuntimeDependencyError(
            f"optional dependency {name!r} is required for model loading; "
            "install it with pip install 'moeatlas[model]'"
        ) from exc


def _require_factory(transformers: Any, name: str) -> Any:
    try:
        factory = getattr(transformers, name)
        loader = getattr(factory, "from_pretrained")
    except Exception as exc:
        raise ModelLoadError(
            "preflight",
            f"optional transformers module does not provide {name}.from_pretrained",
        ) from exc
    if not callable(loader):
        raise ModelLoadError("preflight", f"{name}.from_pretrained is not callable")
    return loader


def _validate_plan(
    plan: LoadingPlan, source_type: type[HuggingFaceSource | LocalSource]
) -> tuple[ResolvedSource, str, str, str]:
    if not isinstance(plan, LoadingPlan):
        raise TypeError("plan must be a LoadingPlan")
    if not isinstance(plan.source, source_type):
        raise RuntimeLoadError(f"model loader requires a plan with {source_type.__name__}")
    # ``model_copy(update=...)`` can intentionally bypass Pydantic validators;
    # repeat the policy checks before resolution or any optional import/I/O.
    _loader_options(plan)
    if isinstance(plan.source, HuggingFaceSource) and (
        plan.source.download_policy != plan.config.download_policy
        or plan.source.allow_downloads != plan.config.allow_downloads
    ):
        raise ModelLoadError(
            "preflight",
            "Hugging Face source and load config download policies must agree explicitly",
        )
    resolution_data = _resolution_for(plan)
    if plan.config.quantization is not QuantizationPolicy.NONE:
        raise ModelLoadError(
            "preflight",
            "quantization is not supported by this non-quantized Feature 9 loader",
        )
    resolution = resolution_data[0]
    if isinstance(plan.source, HuggingFaceSource) and (
        resolution.resolved_model_revision_evidence.kind is not RevisionEvidenceKind.GIT_COMMIT
        or resolution.resolved_tokenizer_revision_evidence.kind
        is not RevisionEvidenceKind.GIT_COMMIT
    ):
        raise ModelLoadError(
            "preflight",
            "Hugging Face loading requires immutable git-commit evidence for model and tokenizer",
        )
    if isinstance(plan.source, LocalSource) and plan.config.allow_downloads:
        raise ModelLoadError(
            "preflight",
            "local loading is always offline and cannot allow downloads",
        )
    return resolution_data


def _copy_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise ModelLoadError("preflight", "loader_options keys must be strings")
            if _is_credential_loader_key(key):
                raise ModelLoadError(
                    "preflight",
                    "loader_options cannot contain credential-bearing keys or headers",
                )
            copied[key] = _copy_json(nested)
        return copied
    if isinstance(value, list | tuple):
        return [_copy_json(item) for item in value]
    return value


def _loader_options(plan: LoadingPlan) -> dict[str, Any]:
    options = _copy_json(plan.config.loader_options)
    if not isinstance(options, dict):  # pragma: no cover - schema guards this
        raise ModelLoadError("preflight", "loader_options must be a JSON object")
    reserved = sorted(
        key for key in options if isinstance(key, str) and _is_reserved_loader_option_key(key)
    )
    if reserved:
        raise ModelLoadError(
            "preflight",
            "loader_options cannot override audited fields: " + ", ".join(reserved),
        )
    return options


def _dtype_kwarg(plan: LoadingPlan) -> Any | None:
    torch = _import_optional("torch")
    if plan.config.dtype is DTypePolicy.PRESERVE:
        return None
    dtype_name = {
        DTypePolicy.FLOAT32: "float32",
        DTypePolicy.FLOAT16: "float16",
        DTypePolicy.BFLOAT16: "bfloat16",
    }[plan.config.dtype]
    try:
        return getattr(torch, dtype_name)
    except Exception as exc:
        raise ModelRuntimeDependencyError(
            f"torch does not expose requested dtype {dtype_name!r}; "
            "install a compatible version with pip install 'moeatlas[model]'"
        ) from exc


def _device_map_kwarg(plan: LoadingPlan) -> Any | None:
    if plan.config.device_map:
        _import_optional("accelerate")
        return dict(plan.config.device_map)
    if plan.config.device == DeviceKind.AUTO.value:
        _import_optional("accelerate")
        return "auto"
    return None


def _common_kwargs(
    *,
    revision: str,
    local_files_only: bool,
    trust_remote_code: bool,
) -> dict[str, Any]:
    return {
        "local_files_only": local_files_only,
        "revision": revision,
        "trust_remote_code": trust_remote_code,
    }


def _call_stage(stage: str, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        return function(*args, **kwargs)
    except Exception as exc:
        raise ModelLoadError(stage, "loader call failed") from exc


def _place_model(model: object, plan: LoadingPlan, stack: _CleanupStack) -> object:
    if plan.config.device == DeviceKind.AUTO.value or plan.config.device_map:
        return model
    try:
        to = getattr(model, "to", None)
    except Exception as exc:
        raise ModelLoadError("placement", "could not inspect model.to") from exc
    if not callable(to):
        raise ModelLoadError(
            "placement",
            f"loaded model does not expose callable to({plan.config.device!r})",
        )
    try:
        placed = to(plan.config.device)
    except Exception as exc:
        raise ModelLoadError("placement", "model.to(device) failed") from exc
    if placed is not None and placed is not model:
        stack.add_object(placed)
        return placed
    return model


def _model_kwargs(
    plan: LoadingPlan,
    *,
    revision: str,
    local_files_only: bool,
    trust_remote_code: bool,
    config: object,
    dtype: Any | None,
    device_map: Any | None,
) -> dict[str, Any]:
    kwargs = _loader_options(plan)
    kwargs.update(
        _common_kwargs(
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
    )
    kwargs["config"] = config
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    if device_map is not None:
        kwargs["device_map"] = device_map
    return kwargs


def _load_transformers(
    plan: LoadingPlan,
    *,
    model_target: str,
    tokenizer_target: str,
    local_files_only: bool,
    resolution_data: tuple[ResolvedSource, str, str, str],
) -> LoadedModel:
    resolution, model_revision, _tokenizer_identifier, tokenizer_revision = resolution_data
    transformers = _import_optional("transformers")
    config_loader = _require_factory(transformers, "AutoConfig")
    tokenizer_loader = _require_factory(transformers, "AutoTokenizer")
    model_loader = _require_factory(transformers, "AutoModel")
    dtype = _dtype_kwarg(plan)
    device_map = _device_map_kwarg(plan)
    common_model = _common_kwargs(
        revision=model_revision,
        local_files_only=local_files_only,
        trust_remote_code=plan.config.trust_remote_code,
    )
    common_tokenizer = _common_kwargs(
        revision=tokenizer_revision,
        local_files_only=local_files_only,
        trust_remote_code=plan.config.trust_remote_code,
    )
    stack = _CleanupStack()
    submitted = False
    try:
        loaded_config = _call_stage("config", config_loader, model_target, **common_model)
        stack.add_object(loaded_config)
        tokenizer = _call_stage("tokenizer", tokenizer_loader, tokenizer_target, **common_tokenizer)
        stack.add_object(tokenizer)
        model = _call_stage(
            "model",
            model_loader,
            model_target,
            **_model_kwargs(
                plan,
                revision=model_revision,
                local_files_only=local_files_only,
                trust_remote_code=plan.config.trust_remote_code,
                config=loaded_config,
                dtype=dtype,
                device_map=device_map,
            ),
        )
        stack.add_object(model)
        model = _place_model(model, plan, stack)
        model_config_object, config_warnings = model_config(model, loaded_config)
        architecture, architecture_warnings = observed_architecture(model, model_config_object)
        observed_model_dtype, dtype_warnings = observed_dtype(model)
        observed_map, device_warnings = observed_device_map(model)
        if isinstance(plan.source, HuggingFaceSource):
            validate_exposed_commits(model, model_config_object, tokenizer, resolution)
        warnings = tuple(
            sorted(
                set(
                    (
                        *config_warnings,
                        *architecture_warnings,
                        *dtype_warnings,
                        *requested_dtype_warnings(plan, observed_model_dtype),
                        *device_warnings,
                        *requested_device_warnings(plan, observed_map),
                    )
                )
            )
        )
        evidence = {
            "model_revision_evidence_source": safe_evidence_source(
                resolution.resolved_model_revision_evidence.evidence_source
            ),
            "tokenizer_revision_evidence_source": safe_evidence_source(
                resolution.resolved_tokenizer_revision_evidence.evidence_source
            ),
        }
        artifacts = RuntimeArtifacts(
            model=model,
            tokenizer=tokenizer,
            config=model_config_object,
            architecture=architecture,
            dtype=observed_model_dtype,
            device_map=observed_map,
            cleanup=stack if stack.pending else None,
            owns_cleanup=stack.pending,
        )
        submitted = True
        return _load_artifacts(
            plan,
            artifacts,
            additional_warnings=warnings,
            additional_provenance=evidence,
        )
    except BaseException as exc:
        if not submitted:
            _cleanup_stack_on_failure(stack, exc)
        raise


def load_huggingface(plan: LoadingPlan) -> LoadedModel:
    """Load a resolved Hugging Face source through lazy Transformers APIs."""

    resolution_data = _validate_plan(plan, HuggingFaceSource)
    source = plan.source
    assert isinstance(source, HuggingFaceSource)
    return _load_transformers(
        plan,
        model_target=source.model_id,
        tokenizer_target=resolution_data[2],
        local_files_only=not source.allow_downloads,
        resolution_data=resolution_data,
    )


def load_local(plan: LoadingPlan) -> LoadedModel:
    """Load a resolved local source from exactly its declared directory."""

    resolution_data = _validate_plan(plan, LocalSource)
    source = plan.source
    assert isinstance(source, LocalSource)
    if not os.path.isdir(source.path):
        raise ModelLoadError("preflight", "local model path must be an existing directory")
    return _load_transformers(
        plan,
        model_target=source.path,
        tokenizer_target=source.path,
        local_files_only=True,
        resolution_data=resolution_data,
    )


__all__ = ["load_huggingface", "load_local"]
