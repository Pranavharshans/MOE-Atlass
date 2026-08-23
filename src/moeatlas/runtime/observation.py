"""Pure observation helpers for loaded Transformers objects."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
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
_PARAMETER_AUDIT_VERSION = "1.0"
_DEFAULT_PARAMETER_TENSOR_BUDGET = 1_000_000


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


def observed_parameter_dtype_inventory(
    model: object,
    *,
    max_parameter_tensors: int = _DEFAULT_PARAMETER_TENSOR_BUDGET,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Audit every named parameter without retaining names or tensor objects.

    The returned inventory is deliberately aggregate-only.  A digest binds it
    to the exact ordered ``(name, dtype, numel, element_size)`` observations,
    while the public rows expose enough evidence to detect mixed or silently
    unconverted parameter pools without publishing model internals.

    ``logical_bytes`` is the tensor-level ``numel * element_size`` total.  It
    is not advertised as allocated GPU memory because wrapper parameters and
    compressed storage formats can have different physical representations.
    """

    if type(max_parameter_tensors) is not int or isinstance(
        max_parameter_tensors, bool
    ):
        raise TypeError("max_parameter_tensors must be an exact integer")
    if max_parameter_tensors <= 0:
        raise ValueError("max_parameter_tensors must be positive")
    try:
        named_parameters = getattr(model, "named_parameters", None)
    except Exception as exc:
        raise ModelObservationError("could not inspect model named_parameters") from exc
    if not callable(named_parameters):
        return (
            {
                "audit_version": _PARAMETER_AUDIT_VERSION,
                "status": "unavailable",
                "reason": "model does not expose callable named_parameters",
                "tensor_count": 0,
                "unsized_tensor_count": 0,
                "element_count": 0,
                "logical_bytes": 0,
                "mixed_dtype": False,
                "dtype_rows": [],
                "inventory_digest": None,
            },
            ("parameter dtype inventory is unavailable",),
        )

    buckets: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    digest_rows: list[tuple[str, str, int, int]] = []
    unsized_tensor_count = 0
    try:
        parameters = named_parameters()
        for index, entry in enumerate(parameters):
            if index >= max_parameter_tensors:
                raise ModelObservationError(
                    "parameter dtype inventory exceeded the tensor budget"
                )
            if type(entry) is not tuple or len(entry) != 2:
                raise ModelObservationError(
                    "model named_parameters yielded an invalid entry"
                )
            name, parameter = entry
            if type(name) is not str or not name:
                raise ModelObservationError(
                    "model named_parameters yielded an invalid name"
                )
            dtype = _dtype_name(getattr(parameter, "dtype", None)) or "unknown"
            numel_function = getattr(parameter, "numel", None)
            element_size_function = getattr(parameter, "element_size", None)
            if not callable(numel_function) or not callable(element_size_function):
                numel = -1
                element_size = -1
            else:
                numel = numel_function()
                element_size = element_size_function()
                if (
                    type(numel) is not int
                    or isinstance(numel, bool)
                    or numel < 0
                    or type(element_size) is not int
                    or isinstance(element_size, bool)
                    or element_size <= 0
                ):
                    numel = -1
                    element_size = -1
            bucket = buckets[dtype]
            bucket[0] += 1
            if numel >= 0 and element_size > 0:
                bucket[1] += numel
                bucket[2] += numel * element_size
                bucket[3] += 1
            else:
                unsized_tensor_count += 1
            digest_rows.append((name, dtype, numel, element_size))
    except ModelObservationError:
        raise
    except Exception as exc:
        raise ModelObservationError("could not audit model parameter dtypes") from exc

    if not digest_rows:
        return (
            {
                "audit_version": _PARAMETER_AUDIT_VERSION,
                "status": "unavailable",
                "reason": "model exposes no named parameters",
                "tensor_count": 0,
                "unsized_tensor_count": 0,
                "element_count": 0,
                "logical_bytes": 0,
                "mixed_dtype": False,
                "dtype_rows": [],
                "inventory_digest": None,
            },
            ("parameter dtype inventory is unavailable",),
        )

    dtype_rows = [
        {
            "dtype": dtype,
            "tensor_count": values[0],
            "sized_tensor_count": values[3],
            "element_count": values[1],
            "logical_bytes": values[2],
        }
        for dtype, values in sorted(buckets.items())
    ]
    digest_payload = json.dumps(
        digest_rows,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    inventory = {
        "audit_version": _PARAMETER_AUDIT_VERSION,
        "status": "partial" if unsized_tensor_count else "available",
        "reason": (
            "one or more parameters did not expose exact size metadata"
            if unsized_tensor_count
            else None
        ),
        "tensor_count": len(digest_rows),
        "unsized_tensor_count": unsized_tensor_count,
        "element_count": sum(row["element_count"] for row in dtype_rows),
        "logical_bytes": sum(row["logical_bytes"] for row in dtype_rows),
        "mixed_dtype": len(dtype_rows) > 1,
        "dtype_rows": dtype_rows,
        "inventory_digest": f"sha256:{hashlib.sha256(digest_payload).hexdigest()}",
    }
    warning_rows = []
    if inventory["mixed_dtype"]:
        warning_rows.append("loaded model parameters use multiple dtypes")
    if unsized_tensor_count:
        warning_rows.append("parameter dtype inventory has incomplete size metadata")
    warnings = tuple(warning_rows)
    return inventory, warnings


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
    "observed_parameter_dtype_inventory",
    "requested_device_warnings",
    "requested_dtype_warnings",
    "safe_evidence_source",
    "validate_exposed_commits",
]
