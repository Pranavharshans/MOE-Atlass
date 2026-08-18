"""Pure observation helpers for loaded Transformers objects."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any

from ..core import DType
from ..loading import (
    DeviceKind,
    DTypePolicy,
    LoadingPlan,
    ResolvedSource,
    RevisionEvidenceKind,
)
from .contracts import ModelObservationError

_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_MISSING = object()


def _device_string(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return f"cuda:{value}"
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    try:
        device_type = getattr(value, "type", None)
        device_index = getattr(value, "index", None)
    except Exception:
        device_type = None
        device_index = None
    if isinstance(device_type, str) and device_type:
        if isinstance(device_index, int) and not isinstance(device_index, bool):
            return f"{device_type}:{device_index}"
        return device_type
    return str(value).strip() or None


def observed_device_map(model: object) -> tuple[dict[str, str], tuple[str, ...]]:
    """Read ``hf_device_map`` or fall back to the model's device."""

    try:
        exposed_map = getattr(model, "hf_device_map", _MISSING)
    except Exception as exc:
        raise ModelObservationError("could not inspect model hf_device_map") from exc
    if isinstance(exposed_map, Mapping):
        normalized: dict[str, str] = {}
        for module_path, value in exposed_map.items():
            if not isinstance(module_path, str):
                raise ModelObservationError("model hf_device_map keys must be strings")
            device = _device_string(value)
            if device is None:
                raise ModelObservationError("model hf_device_map contains an invalid device")
            normalized[module_path] = device
        if normalized:
            return normalized, ()

    try:
        device = _device_string(getattr(model, "device", None))
    except Exception as exc:
        raise ModelObservationError("could not inspect model device") from exc
    if device is not None:
        return {"": device}, ()
    return {}, ("loaded model did not expose hf_device_map or device",)


def _dtype_name(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        name = value
    else:
        try:
            name = getattr(value, "name", None) or str(value)
        except Exception:
            return None
    name = name.lower().replace("torch.", "")
    return {
        "float": "float32",
        "float32": "float32",
        "float64": "float64",
        "double": "float64",
        "float16": "float16",
        "half": "float16",
        "bfloat16": "bfloat16",
        "int8": "int8",
        "uint8": "uint8",
        "int4": "int4",
    }.get(name)


def observed_dtype(model: object) -> tuple[DType, tuple[str, ...]]:
    """Observe the declared model dtype or only the first parameter dtype."""

    try:
        declared = getattr(model, "dtype", _MISSING)
    except Exception as exc:
        raise ModelObservationError("could not inspect model dtype") from exc
    value = None if declared is _MISSING else declared
    if value is None:
        try:
            named_parameters = getattr(model, "named_parameters", None)
            if callable(named_parameters):
                first = next(iter(named_parameters()), None)
                if first is not None:
                    value = getattr(first[1], "dtype", None)
        except Exception as exc:
            raise ModelObservationError(
                "could not inspect the first model parameter dtype"
            ) from exc
    name = _dtype_name(value)
    mapping = {
        "float64": DType.FLOAT64,
        "float32": DType.FLOAT32,
        "float16": DType.FLOAT16,
        "bfloat16": DType.BFLOAT16,
        "int8": DType.INT8,
        "uint8": DType.UINT8,
        "int4": DType.INT4,
    }
    if name in mapping:
        return mapping[name], ()
    return DType.UNKNOWN, ("loaded model dtype was unavailable or not a known core DType",)


def _config_field(config: object, name: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(name)
    try:
        return getattr(config, name, None)
    except Exception as exc:
        raise ModelObservationError(f"could not inspect loaded config field {name!r}") from exc


def observed_architecture(model: object, config: object) -> tuple[str, tuple[str, ...]]:
    model_type = _config_field(config, "model_type")
    if (
        isinstance(model_type, str)
        and model_type.strip()
        and not any(character.isspace() for character in model_type)
    ):
        return model_type, ()
    model_class = type(model)
    architecture = f"{model_class.__module__}.{model_class.__qualname__}"
    return architecture, (
        "config.model_type was unavailable; architecture was inferred from the model class",
    )


def model_config(model: object, loaded_config: object) -> tuple[object, tuple[str, ...]]:
    try:
        actual = getattr(model, "config", _MISSING)
    except Exception as exc:
        raise ModelObservationError("could not inspect model.config") from exc
    if actual is _MISSING or actual is None:
        return loaded_config, ("model.config was unavailable; using the loaded config",)
    return actual, ()


def safe_evidence_source(value: str) -> str:
    """Store only a deterministic digest of an arbitrary evidence label."""

    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _exposed_commit_hashes(value: object) -> list[str]:
    hashes: list[str] = []
    for attribute in ("_commit_hash", "commit_hash"):
        try:
            candidate = getattr(value, attribute, None)
        except Exception as exc:
            raise ModelObservationError(
                f"could not inspect exposed {attribute} resolution evidence"
            ) from exc
        if candidate is not None:
            if not isinstance(candidate, str):
                raise ModelObservationError("exposed commit hash must be a string")
            hashes.append(candidate)
    try:
        init_kwargs = getattr(value, "init_kwargs", None)
    except Exception as exc:
        raise ModelObservationError("could not inspect tokenizer resolution evidence") from exc
    if isinstance(init_kwargs, Mapping):
        candidate = init_kwargs.get("_commit_hash")
        if candidate is not None:
            if not isinstance(candidate, str):
                raise ModelObservationError("exposed commit hash must be a string")
            hashes.append(candidate)
    return hashes


def validate_exposed_commits(
    model: object,
    model_config_object: object,
    tokenizer: object,
    resolution: ResolvedSource,
) -> None:
    expected = (
        resolution.resolved_model_revision_evidence.kind,
        resolution.resolved_model_revision_evidence.digest,
    )
    for label, value, expected_pair in (
        ("model", model, expected),
        ("model config", model_config_object, expected),
        (
            "tokenizer",
            tokenizer,
            (
                resolution.resolved_tokenizer_revision_evidence.kind,
                resolution.resolved_tokenizer_revision_evidence.digest,
            ),
        ),
    ):
        for exposed in _exposed_commit_hashes(value):
            if not _GIT_COMMIT.fullmatch(exposed):
                raise ModelObservationError(
                    f"exposed {label} commit hash must be a full lowercase 40-character digest"
                )
            expected_kind, expected_digest = expected_pair
            if expected_kind is RevisionEvidenceKind.GIT_COMMIT and exposed != expected_digest:
                raise ModelObservationError(
                    f"exposed {label} commit hash does not match immutable resolution"
                )


def requested_device_warnings(plan: LoadingPlan, device_map: Mapping[str, str]) -> tuple[str, ...]:
    requested = plan.config.device
    warnings: list[str] = []
    if plan.config.device_map and dict(plan.config.device_map) != dict(device_map):
        warnings.append(
            f"requested device_map differs from observed device map {dict(device_map)!r}"
        )
    if not device_map or requested == DeviceKind.AUTO.value:
        return tuple(warnings)
    values = tuple(device_map.values())
    requested_map_values = tuple(plan.config.device_map.values())
    allows_cpu_offload = bool(requested_map_values) and "cpu" in requested_map_values

    def matches(value: str) -> bool:
        if requested == DeviceKind.CPU.value:
            return value == "cpu"
        if requested == DeviceKind.MPS.value:
            return value == "mps"
        if requested == DeviceKind.CUDA.value:
            return value.startswith("cuda") or (value == "cpu" and allows_cpu_offload)
        return value == requested or (value == "cpu" and allows_cpu_offload)

    if not all(matches(value) for value in values):
        warnings.append(
            f"requested device {requested!r} differs from observed device map {dict(device_map)!r}"
        )
    return tuple(warnings)


def requested_dtype_warnings(plan: LoadingPlan, observed: DType) -> tuple[str, ...]:
    """Report explicit requested-vs-observed dtype disagreement."""

    if plan.config.dtype is DTypePolicy.PRESERVE:
        return ()
    requested = plan.config.dtype.manifest_dtype_hint()
    if requested is observed:
        return ()
    return (
        f"requested dtype {plan.config.dtype.value!r} differs from observed "
        f"dtype {observed.value!r}",
    )


__all__ = [
    "model_config",
    "observed_architecture",
    "observed_device_map",
    "observed_dtype",
    "requested_device_warnings",
    "requested_dtype_warnings",
    "safe_evidence_source",
    "validate_exposed_commits",
]
