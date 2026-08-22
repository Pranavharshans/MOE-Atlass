"""Manifest validation and explicit instance/custom-loader execution."""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from typing import Any

from .. import __version__
from ..core import (
    DType,
    ModelManifest,
    Provenance,
    TokenizerIdentity,
    make_config_hash,
    make_model_key,
    validate_stable_identifier,
)
from ..loading import (
    CustomLoaderSource,
    InstanceSource,
    LoadingPlan,
    ResolvedSource,
)
from .contracts import (
    CleanupCallback,
    CustomLoaderExecutionError,
    LoadedModel,
    PendingRuntimeCleanup,
    RuntimeArtifacts,
    RuntimeLoadError,
    RuntimeValidationError,
    _add_cleanup_note,
    _attach_pending_cleanup,
)


def _normalize_json(value: Any, *, path: str = "config") -> Any:
    """Defensively copy finite JSON data with deterministic object ordering.

    Transformers config objects may expose JSON-object keys as integers after
    ``PretrainedConfig.to_dict()`` (for example ``id2label``).  JSON itself
    represents those keys as strings, so canonical observation converts exact
    integer keys to their JSON spelling while rejecting unsupported key types
    and collisions instead of silently dropping data.
    """

    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, nested in value.items():
            if type(key) is str:
                normalized_key = key
            elif type(key) is int:
                normalized_key = str(key)
            else:
                raise RuntimeValidationError(
                    f"{path} object keys must be strings or exact integers"
                )
            if normalized_key in normalized:
                raise RuntimeValidationError(
                    f"{path} object keys collide after JSON string normalization: "
                    f"{normalized_key!r}"
                )
            normalized[normalized_key] = _normalize_json(
                nested, path=f"{path}.{normalized_key}"
            )
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, list | tuple):
        return [_normalize_json(item, path=f"{path}[]") for item in value]
    if value is None or isinstance(value, str | int | bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RuntimeValidationError(f"{path} values must be finite")
        return value
    raise RuntimeValidationError(f"{path} contains unsupported value type {type(value).__name__}")


def _observed_config(config: object) -> dict[str, Any]:
    if isinstance(config, Mapping):
        raw_config: object = config
    else:
        try:
            to_dict = getattr(config, "to_dict", None)
        except Exception as exc:
            raise RuntimeValidationError(
                "runtime config must be a mapping or expose to_dict()"
            ) from exc
        if not callable(to_dict):
            raise RuntimeValidationError("runtime config must be a mapping or expose to_dict()")
        try:
            raw_config = to_dict()
        except Exception as exc:
            raise RuntimeValidationError("runtime config to_dict() failed") from exc
    if not isinstance(raw_config, Mapping):
        raise RuntimeValidationError("runtime config to_dict() must return a mapping")
    normalized = _normalize_json(raw_config)
    if not isinstance(normalized, dict):  # pragma: no cover - guarded by mapping input
        raise RuntimeValidationError("runtime config must normalize to an object")
    return normalized


def _observed_device_map(device_map: Mapping[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for module_path, device in device_map.items():
        if not isinstance(module_path, str) or not isinstance(device, str):
            raise RuntimeValidationError("observed device_map keys and values must be strings")
        if module_path:
            try:
                validate_stable_identifier(module_path, field_name="device_map module path")
            except (TypeError, ValueError) as exc:
                raise RuntimeValidationError(
                    "observed device_map module paths must be canonical identifiers"
                ) from exc
        if (
            not device
            or device != device.strip()
            or any(character.isspace() for character in device)
        ):
            raise RuntimeValidationError(
                "observed device_map values must be non-empty names without whitespace"
            )
        normalized[module_path] = device
    return {key: normalized[key] for key in sorted(normalized)}


def _preflight_model(model: object) -> None:
    if model is None:
        raise RuntimeValidationError("runtime model must not be None")
    try:
        named_modules = getattr(model, "named_modules")
    except Exception as exc:
        raise RuntimeValidationError("runtime model must expose callable named_modules()") from exc
    if not callable(named_modules):
        raise RuntimeValidationError("runtime model must expose callable named_modules()")


def _resolution_for(plan: LoadingPlan) -> tuple[ResolvedSource, str, str, str]:
    resolution = plan.resolution
    if resolution is None:
        raise RuntimeValidationError(
            "a resolved immutable model revision is required; set LoadingPlan.resolution"
        )
    if resolution.source_type.value != plan.source.source_type:
        raise RuntimeValidationError("resolved source type does not match the loading plan")
    if resolution.model_id != plan.source.model_id:
        raise RuntimeValidationError("resolved model ID does not match the loading plan")
    if resolution.requested_model_revision != plan.source.requested_revision:
        raise RuntimeValidationError(
            "resolved requested model revision does not match the loading plan"
        )
    model_evidence = resolution.resolved_model_revision_evidence
    model_revision = resolution.resolved_model_revision
    if model_evidence is None or not model_revision:
        raise RuntimeValidationError(
            "resolved model revision and immutable evidence are required before runtime loading"
        )
    if model_revision != model_evidence.canonical_revision:
        raise RuntimeValidationError("resolved model revision does not match immutable evidence")

    tokenizer_request = plan.source.tokenizer
    if tokenizer_request is None:
        raise RuntimeValidationError(
            "InstanceSource/CustomLoaderSource requires a tokenizer request for "
            "the v1 text-model boundary"
        )
    expected_tokenizer_revision = (
        plan.source.requested_revision
        if tokenizer_request.inherit_model_revision
        else tokenizer_request.requested_revision
    )
    if resolution.requested_tokenizer_revision != expected_tokenizer_revision:
        raise RuntimeValidationError(
            "resolved tokenizer request does not match the source tokenizer request"
        )
    tokenizer_revision = resolution.resolved_tokenizer_revision
    tokenizer_evidence = resolution.resolved_tokenizer_revision_evidence
    if tokenizer_revision is None or tokenizer_evidence is None:
        raise RuntimeValidationError(
            "a tokenizer request requires matching immutable resolved tokenizer evidence"
        )
    if tokenizer_revision != tokenizer_evidence.canonical_revision:
        raise RuntimeValidationError(
            "resolved tokenizer revision does not match immutable evidence"
        )
    return (
        resolution,
        model_revision,
        tokenizer_request.identifier,
        tokenizer_revision,
    )


def _build_manifest(
    plan: LoadingPlan,
    artifacts: RuntimeArtifacts,
    *,
    additional_warnings: tuple[str, ...] = (),
    additional_provenance: Mapping[str, Any] | None = None,
) -> tuple[ModelManifest, tuple[str, ...]]:
    _preflight_model(artifacts.model)
    if artifacts.tokenizer is None:
        raise RuntimeValidationError("runtime tokenizer must not be None")
    if not isinstance(artifacts.architecture, str):
        raise RuntimeValidationError("observed architecture must be a string")
    if (
        not artifacts.architecture
        or artifacts.architecture != artifacts.architecture.strip()
        or any(character.isspace() for character in artifacts.architecture)
    ):
        raise RuntimeValidationError(
            "observed architecture must be a non-empty name without whitespace"
        )
    if not isinstance(artifacts.dtype, DType):
        raise RuntimeValidationError("observed dtype must be a core DType")

    config = _observed_config(artifacts.config)
    device_map = _observed_device_map(artifacts.device_map)
    resolution, model_revision, tokenizer_identifier, tokenizer_revision = _resolution_for(plan)
    try:
        model_key = make_model_key(plan.source.model_id, model_revision)
    except (TypeError, ValueError) as exc:
        raise RuntimeValidationError("runtime model identity is not canonical") from exc
    security_warnings = tuple(sorted(set(plan.security_warnings)))
    warnings = tuple(sorted(set((*security_warnings, *additional_warnings))))
    model_evidence = resolution.resolved_model_revision_evidence
    tokenizer_evidence = resolution.resolved_tokenizer_revision_evidence
    if model_evidence is None or tokenizer_evidence is None:  # pragma: no cover
        raise RuntimeValidationError("immutable resolution evidence is incomplete")
    provenance_metadata: dict[str, Any] = {
        "loading_plan_id": plan.plan_id,
        "model_revision_evidence": model_evidence.kind.value,
        "resolution_method": resolution.resolution_method,
        "security_warnings": list(security_warnings),
        "source_type": plan.source.source_type,
        "tokenizer_revision_evidence": tokenizer_evidence.kind.value,
    }
    if additional_provenance:
        normalized_provenance = _normalize_json(
            additional_provenance,
            path="runtime provenance",
        )
        if not isinstance(normalized_provenance, dict):  # pragma: no cover
            raise RuntimeValidationError("runtime provenance must normalize to an object")
        protected = set(provenance_metadata).intersection(normalized_provenance)
        if protected:
            raise RuntimeValidationError(
                "additional runtime provenance cannot override core metadata: "
                + ", ".join(sorted(protected))
            )
        provenance_metadata.update(normalized_provenance)
    if additional_warnings:
        provenance_metadata["runtime_warnings"] = list(sorted(set(additional_warnings)))
    try:
        manifest = ModelManifest(
            model_key=model_key,
            architecture=artifacts.architecture,
            revision=model_revision,
            config_hash=make_config_hash(config),
            tokenizer=TokenizerIdentity(
                identifier=tokenizer_identifier,
                revision=tokenizer_revision,
            ),
            dtype=artifacts.dtype,
            device_map=device_map,
            provenance=Provenance(
                source=f"runtime:{plan.source.source_type}",
                tool_version=__version__,
                metadata=provenance_metadata,
            ),
            warnings=list(warnings),
        )
    except Exception as exc:
        raise RuntimeValidationError(
            "observed runtime facts could not form a ModelManifest"
        ) from exc
    return manifest, warnings


def _cleanup_after_validation_failure(
    artifacts: RuntimeArtifacts,
    original: BaseException,
) -> None:
    if not artifacts.owns_cleanup or artifacts.cleanup is None:
        return
    pending = PendingRuntimeCleanup(artifacts.cleanup)
    try:
        pending.retry()
    except RuntimeLoadError as cleanup_error:
        _attach_pending_cleanup(original, pending)
        _add_cleanup_note(original, cleanup_error)


def _load_artifacts(
    plan: LoadingPlan,
    artifacts: RuntimeArtifacts,
    *,
    additional_warnings: tuple[str, ...] = (),
    additional_provenance: Mapping[str, Any] | None = None,
) -> LoadedModel:
    try:
        manifest, warnings = _build_manifest(
            plan,
            artifacts,
            additional_warnings=additional_warnings,
            additional_provenance=additional_provenance,
        )
    except BaseException as exc:
        _cleanup_after_validation_failure(artifacts, exc)
        raise
    cleanup = artifacts.cleanup if artifacts.owns_cleanup else None
    return LoadedModel(
        model=artifacts.model,
        tokenizer=artifacts.tokenizer,
        plan=plan,
        manifest=manifest,
        warnings=warnings,
        _cleanup_callback=cleanup,
        _owns_cleanup=artifacts.owns_cleanup,
    )


def load_instance(
    plan: LoadingPlan,
    model: object,
    tokenizer: object,
    observation: RuntimeArtifacts | None = None,
    *,
    config: object | None = None,
    architecture: str | None = None,
    dtype: DType | None = None,
    device_map: Mapping[str, str] | None = None,
    cleanup: CleanupCallback | None = None,
    owns_cleanup: bool = False,
) -> LoadedModel:
    """Validate a caller-owned instance against an ``InstanceSource`` plan."""

    if not isinstance(plan, LoadingPlan):
        raise TypeError("plan must be a LoadingPlan")
    if not isinstance(plan.source, InstanceSource):
        raise RuntimeLoadError("load_instance requires a plan with InstanceSource")
    if observation is not None:
        if not isinstance(observation, RuntimeArtifacts):
            raise TypeError("observation must be RuntimeArtifacts")
        if observation.model is not model or observation.tokenizer is not tokenizer:
            raise RuntimeValidationError(
                "observation model/tokenizer must be the exact objects passed to load_instance"
            )
        if (
            any(value is not None for value in (config, architecture, dtype, device_map, cleanup))
            or owns_cleanup
        ):
            raise TypeError("observation cannot be combined with separate runtime facts")
        artifacts = observation
    else:
        if any(value is None for value in (config, architecture, dtype, device_map)):
            raise TypeError(
                "load_instance requires observation or config, architecture, dtype, and device_map"
            )
        artifacts = RuntimeArtifacts(
            model=model,
            tokenizer=tokenizer,
            config=config,
            architecture=architecture,
            dtype=dtype,
            device_map=device_map,
            cleanup=cleanup,
            owns_cleanup=owns_cleanup,
        )
    return _load_artifacts(plan, artifacts)


def load_custom(plan: LoadingPlan, *, execute_user_code: bool = False) -> LoadedModel:
    """Execute one explicit ``module:function`` custom loader after opt-in."""

    if not isinstance(plan, LoadingPlan):
        raise TypeError("plan must be a LoadingPlan")
    if not isinstance(plan.source, CustomLoaderSource):
        raise RuntimeLoadError("load_custom requires a plan with CustomLoaderSource")
    if execute_user_code is not True:
        raise CustomLoaderExecutionError(
            "custom loader execution is disabled; pass execute_user_code=True explicitly"
        )
    # Reject an impossible runtime plan before importing user code.
    _resolution_for(plan)

    module_name, separator, function_name = plan.source.loader_reference.partition(":")
    if not separator:  # pragma: no cover - CustomLoaderSource validates this
        raise CustomLoaderExecutionError("custom loader reference must use module:function syntax")
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise CustomLoaderExecutionError(
            f"could not import custom loader module {module_name!r}"
        ) from exc
    try:
        loader = getattr(module, function_name)
    except Exception as exc:
        raise CustomLoaderExecutionError(
            f"custom loader attribute {function_name!r} was not found"
        ) from exc
    if not callable(loader):
        raise CustomLoaderExecutionError(
            f"custom loader attribute {function_name!r} is not callable"
        )
    try:
        artifacts = loader(plan)
    except Exception as exc:
        raise CustomLoaderExecutionError("custom loader execution failed") from exc
    if type(artifacts) is not RuntimeArtifacts:
        raise CustomLoaderExecutionError(
            "custom loader must return exactly RuntimeArtifacts, not a tuple or mapping"
        )
    return _load_artifacts(plan, artifacts)


__all__ = ["load_custom", "load_instance"]
