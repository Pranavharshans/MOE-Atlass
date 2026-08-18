"""Strict, runtime-independent model-source and loading-plan contracts.

This module describes intent only. It never imports a model runtime, resolves a
local path, inspects a cache, imports a custom loader, or downloads a model.
Later runtime integration will receive an already-instantiated object
separately from these serializable contracts.
"""

from __future__ import annotations

import json
import math
import os
import re
from enum import Enum
from typing import Annotated, Any, Literal, Self, TypeAlias

from pydantic import ConfigDict, Field, StrictBool, StrictStr, field_validator, model_validator

from .core import (
    DType,
    StrictManifestModel,
    make_model_key,
    stable_digest,
    validate_stable_identifier,
)

LOADING_SCHEMA_VERSION = "1.0"
_PLAN_ID = re.compile(r"^loadplan:([0-9a-f]{64})$")
_LOADER_REFERENCE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*:[A-Za-z_][A-Za-z0-9_]*$"
)
_CUDA_DEVICE = re.compile(r"^cuda(?::[0-9]+)?$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_RESERVED_LOADER_OPTION_KEYS = frozenset(
    {
        "allow_downloads",
        "access_token",
        "auth_token",
        "cache_dir",
        "config",
        "device",
        "device_map",
        "dtype",
        "download_policy",
        "load_in_4bit",
        "load_in_8bit",
        "local_files_only",
        "model_id",
        "model_revision",
        "offline",
        "offload_folder",
        "path",
        "proxies",
        "quantization",
        "quantization_config",
        "remote_code_acknowledged",
        "requested_revision",
        "revision",
        "source_type",
        "token",
        "tokenizer_id",
        "tokenizer_revision",
        "torch_dtype",
        "trust_remote_code",
        "use_auth_token",
        "force_download",
        "loader_reference",
    }
)
_SECRET_LOADER_OPTION_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "api_secret",
        "api_token",
        "apikey",
        "auth",
        "auth_header",
        "auth_token",
        "authorization",
        "authentication",
        "bearer",
        "client_secret",
        "cookie",
        "cookies",
        "cookie_header",
        "credential",
        "credentials",
        "password",
        "passwd",
        "passphrase",
        "proxies",
        "secret",
        "secrets",
        "secret_key",
        "token",
        "use_auth_token",
    }
)
_CREDENTIAL_CONTAINER_KEYS = frozenset(
    {
        "cookie_header",
        "cookie_headers",
        "cookies",
        "extra_header",
        "extra_headers",
        "header",
        "headers",
        "http_header",
        "http_headers",
        "request_header",
        "request_headers",
        "response_header",
        "response_headers",
    }
)
_SECRET_SUFFIXES = ("_credential", "_credentials", "_password", "_passwd", "_passphrase")
_SECRET_PREFIXES = ("client_", "server_", "service_", "app_", "api_", "oauth_")
_CREDENTIAL_WORD = re.compile(
    r"(?:^|_)(?:token|password|passwd|passphrase|credential|credentials|cookie|cookies)(?:_|$)"
)
_AUTH_WORD = re.compile(r"(?:^|_)(?:auth|authorization|bearer)(?:_|$)")
_HEADER_WORD = re.compile(r"(?:^|_)(?:header|headers)(?:_|$)")
_API_KEY_WORD = re.compile(r"(?:^|_)api_?key(?:_|$)")
_SECRET_WORD = re.compile(
    r"(?:^|_)(?:client|server|service|app|api|oauth|jwt|session)_secret(?:s)?(?:_|$)"
)


def _normalize_loader_option_key(key: str) -> str:
    """Normalize option keys for deterministic, case-insensitive policy checks."""

    # Split both lower-to-upper and acronym-to-word transitions before
    # casefolding so ``accessToken``, ``xApiKey``, and ``HTTPHeaders`` cannot
    # bypass the same policy as snake/kebab case.
    camel = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", key)
    camel = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", camel)
    return re.sub(r"[^a-z0-9]+", "_", camel.casefold()).strip("_")


def _is_credential_loader_key(key: str) -> bool:
    """Return whether a key could carry a credential or authorization header.

    This deliberately targets credential-shaped names rather than every key
    containing words such as ``secret``.  For example, a backend-specific
    ``secret_sauce_mode`` remains valid while ``client_secret`` and
    ``x-api-key`` are rejected.
    """

    normalized = _normalize_loader_option_key(key)
    if normalized in _SECRET_LOADER_OPTION_KEYS or normalized in _CREDENTIAL_CONTAINER_KEYS:
        return True
    if (
        _CREDENTIAL_WORD.search(normalized)
        or _AUTH_WORD.search(normalized)
        or _HEADER_WORD.search(normalized)
        or _API_KEY_WORD.search(normalized)
        or _SECRET_WORD.search(normalized)
    ):
        return True
    if normalized.endswith(_SECRET_SUFFIXES):
        return True
    if normalized.endswith("_token") or normalized.endswith("_api_key"):
        return True
    if normalized in {"private_key", "signing_key", "encryption_key"}:
        return True
    if normalized in {"secret_token", "secret_value", "secret_key"}:
        return True
    if normalized.endswith(("_secret", "_secrets")) and normalized.startswith(_SECRET_PREFIXES):
        return True
    if normalized.startswith("api_") and normalized.endswith(("_key", "_token", "_secret")):
        return True
    return False


_NORMALIZED_RESERVED_LOADER_OPTION_KEYS = frozenset(
    _normalize_loader_option_key(key) for key in _RESERVED_LOADER_OPTION_KEYS
)


def _is_reserved_loader_option_key(key: str) -> bool:
    """Return whether a top-level option would override audited loader policy."""

    return _normalize_loader_option_key(key) in _NORMALIZED_RESERVED_LOADER_OPTION_KEYS


class SourceKind(str, Enum):
    """Supported source request discriminators."""

    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    INSTANCE = "instance"
    CUSTOM = "custom"

    def __str__(self) -> str:
        return self.value


class DownloadPolicy(str, Enum):
    """Explicit network/cache policy for a future loader."""

    OFFLINE = "offline"
    ALLOW_DOWNLOADS = "allow_downloads"

    def __str__(self) -> str:
        return self.value


class DeviceKind(str, Enum):
    """Common device selections accepted by the future runtime boundary."""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    AUTO = "auto"

    def __str__(self) -> str:
        return self.value


class DTypePolicy(str, Enum):
    """Requested load dtype policy, distinct from observed manifest dtype."""

    PRESERVE = "preserve"
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"

    def __str__(self) -> str:
        return self.value

    def manifest_dtype_hint(self) -> DType:
        """Map an explicit request to the existing observed-dtype enum.

        ``PRESERVE`` intentionally maps to ``UNKNOWN`` because the loaded
        dtype is not known until a later runtime has produced a manifest.
        """

        return {
            DTypePolicy.PRESERVE: DType.UNKNOWN,
            DTypePolicy.FLOAT32: DType.FLOAT32,
            DTypePolicy.FLOAT16: DType.FLOAT16,
            DTypePolicy.BFLOAT16: DType.BFLOAT16,
        }[self]


class QuantizationPolicy(str, Enum):
    """Minimal truthful quantization policy surface for future loaders."""

    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"

    def __str__(self) -> str:
        return self.value


class VersionedLoadingModel(StrictManifestModel):
    """Shared strict/frozen schema and JSON behavior for loading contracts."""

    # Loading options may contain credentials supplied by an API caller.  Do
    # not let Pydantic echo those values in a ValidationError's input context.
    model_config = ConfigDict(hide_input_in_errors=True)
    schema_version: Literal["1.0"] = Field(default=LOADING_SCHEMA_VERSION, frozen=True)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible data without runtime objects."""

        return self.model_dump(mode="json")

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize this contract using deterministic field/container order."""

        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> Self:
        """Validate a JSON document against this concrete contract."""

        return cls.model_validate_json(payload)


class _FrozenDict(dict[str, Any]):
    """Dict-compatible immutable JSON object for nested load options."""

    @staticmethod
    def _immutable(*args: Any, **kwargs: Any) -> None:
        raise TypeError("loading configuration metadata is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable

    def __ior__(self, other: object) -> _FrozenDict:
        self._immutable(other)
        return self


class _FrozenList(list[Any]):
    """List-compatible immutable JSON array for nested load options."""

    @staticmethod
    def _immutable(*args: Any, **kwargs: Any) -> None:
        raise TypeError("loading configuration metadata is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __iadd__ = _immutable
    __imul__ = _immutable
    append = _immutable
    clear = _immutable
    extend = _immutable
    insert = _immutable
    pop = _immutable
    remove = _immutable
    reverse = _immutable
    sort = _immutable


class _ReservedLoaderOptionError(ValueError):
    """Raised when arbitrary loader kwargs try to override audited policy."""


def _validate_json_node(value: Any) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError("loader options object keys must be strings")
            if _is_credential_loader_key(key):
                raise _ReservedLoaderOptionError(
                    "loader options cannot contain credential-bearing keys or headers"
                )
            _validate_json_node(nested)
    elif isinstance(value, list | tuple):
        for nested in value:
            _validate_json_node(nested)
    elif value is None or isinstance(value, str | int | bool):
        return
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("loader options values must be finite")
    else:
        raise TypeError(f"loader options contain unsupported type {type(value).__name__}")


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return _FrozenDict({key: _freeze_json(value[key]) for key in sorted(value)})
    if isinstance(value, list | tuple):
        return _FrozenList(_freeze_json(item) for item in value)
    return value


def _validate_json_options(value: dict[str, Any]) -> dict[str, Any]:
    try:
        reserved = sorted(
            key for key in value if isinstance(key, str) and _is_reserved_loader_option_key(key)
        )
        if reserved:
            raise _ReservedLoaderOptionError(
                "loader_options top-level keys are reserved; set the audited policy "
                f"fields instead: {', '.join(reserved)}"
            )
        _validate_json_node(value)
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
    except _ReservedLoaderOptionError as exc:
        raise ValueError(str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise ValueError("loader_options must contain only finite JSON-compatible values") from exc
    return value


def _stable_token(value: str, *, field_name: str) -> str:
    return validate_stable_identifier(value, field_name=field_name)


def _revision_token(value: str, *, field_name: str) -> str:
    return _stable_token(value, field_name=field_name)


def _validate_download_policy(policy: DownloadPolicy, allow_downloads: bool) -> None:
    if policy is DownloadPolicy.OFFLINE and allow_downloads:
        raise ValueError("offline download_policy cannot be combined with allow_downloads=True")
    if policy is DownloadPolicy.ALLOW_DOWNLOADS and not allow_downloads:
        raise ValueError("allow_downloads download_policy requires explicit allow_downloads=True")


def _normalize_local_path(value: str) -> str:
    if not value or not value.strip():
        raise ValueError("local path must not be empty")
    if "\x00" in value:
        raise ValueError("local path must not contain NUL")
    if any(ord(character) < 32 for character in value):
        raise ValueError("local path must not contain control characters")
    # normpath is lexical only. It does not resolve symlinks, read the path, or
    # consult the filesystem; the value remains runtime input, not identity.
    normalized = os.path.normpath(value)
    if not normalized or normalized == os.path.sep and not value:
        raise ValueError("local path must not be empty")
    return normalized


def _validate_device(value: str, *, field_name: str, allow_auto: bool = True) -> str:
    allowed_named = {DeviceKind.CPU.value, DeviceKind.CUDA.value, DeviceKind.MPS.value}
    if allow_auto:
        allowed_named.add(DeviceKind.AUTO.value)
    if value in allowed_named or _CUDA_DEVICE.fullmatch(value):
        return value
    raise ValueError(f"{field_name} must be cpu, cuda, cuda:<index>, mps, or auto; got {value!r}")


class TokenizerRequest(VersionedLoadingModel):
    """Separate tokenizer identity with an explicit revision inheritance rule."""

    identifier: StrictStr
    requested_revision: StrictStr | None = None
    inherit_model_revision: StrictBool = False

    @field_validator("identifier")
    @classmethod
    def _identifier(cls, value: str) -> str:
        return _stable_token(value, field_name="tokenizer identifier")

    @field_validator("requested_revision")
    @classmethod
    def _requested_revision(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _revision_token(value, field_name="tokenizer requested_revision")

    @model_validator(mode="after")
    def _revision_rule(self) -> Self:
        if self.inherit_model_revision and self.requested_revision is not None:
            raise ValueError(
                "tokenizer requested_revision must be omitted when inherit_model_revision=True"
            )
        if not self.inherit_model_revision and self.requested_revision is None:
            raise ValueError(
                "tokenizer requested_revision is required unless inherit_model_revision=True"
            )
        return self


class _IdentitySource(VersionedLoadingModel):
    """Shared portable identity fields for all source discriminators."""

    model_id: StrictStr
    requested_revision: StrictStr
    tokenizer: TokenizerRequest | None = None

    @field_validator("model_id")
    @classmethod
    def _model_id(cls, value: str) -> str:
        return _stable_token(value, field_name="model_id")

    @field_validator("requested_revision")
    @classmethod
    def _revision(cls, value: str) -> str:
        return _revision_token(value, field_name="requested_revision")

    @model_validator(mode="after")
    def _model_key_compatible(self) -> Self:
        # Source requests eventually bind to ModelManifest.model_key. Validate
        # that identity boundary now so an otherwise-valid source cannot carry
        # the ambiguous ``@`` separator into a later runtime integration.
        make_model_key(self.model_id, self.requested_revision)
        return self


class HuggingFaceSource(_IdentitySource):
    """A canonical Hub repository request; no Hub access occurs here."""

    source_type: Literal["huggingface"] = "huggingface"
    download_policy: DownloadPolicy = DownloadPolicy.OFFLINE
    allow_downloads: StrictBool = False

    @model_validator(mode="after")
    def _download_policy(self) -> Self:
        _validate_download_policy(self.download_policy, self.allow_downloads)
        return self


class LocalSource(_IdentitySource):
    """A lexical local directory request whose path is never portable identity."""

    source_type: Literal["local"] = "local"
    path: StrictStr

    @field_validator("path")
    @classmethod
    def _path(cls, value: str) -> str:
        return _normalize_local_path(value)


class InstanceSource(_IdentitySource):
    """Identity for an already-instantiated module supplied separately by API."""

    source_type: Literal["instance"] = "instance"


class CustomLoaderSource(_IdentitySource):
    """A validated ``module:function`` reference that is never imported here."""

    source_type: Literal["custom"] = "custom"
    loader_reference: StrictStr

    @field_validator("loader_reference")
    @classmethod
    def _loader_reference(cls, value: str) -> str:
        if not _LOADER_REFERENCE.fullmatch(value):
            raise ValueError(
                "loader_reference must use module:function syntax and is not imported "
                "during validation"
            )
        return value


SourceRequest: TypeAlias = Annotated[
    HuggingFaceSource | LocalSource | InstanceSource | CustomLoaderSource,
    Field(discriminator="source_type"),
]


class LoadConfig(VersionedLoadingModel):
    """Bounded future-loader policy with no runtime side effects."""

    device: StrictStr = DeviceKind.CPU.value
    device_map: dict[StrictStr, StrictStr] = Field(default_factory=dict)
    dtype: DTypePolicy = DTypePolicy.PRESERVE
    quantization: QuantizationPolicy = QuantizationPolicy.NONE
    trust_remote_code: StrictBool = False
    remote_code_acknowledged: StrictBool = False
    download_policy: DownloadPolicy = DownloadPolicy.OFFLINE
    allow_downloads: StrictBool = False
    loader_options: dict[StrictStr, Any] = Field(default_factory=dict)

    @field_validator("device")
    @classmethod
    def _device(cls, value: str) -> str:
        return _validate_device(value, field_name="device")

    @field_validator("device_map")
    @classmethod
    def _device_map(cls, value: dict[str, str]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for module_path, device in value.items():
            if module_path:
                _stable_token(module_path, field_name="device_map module path")
            normalized[module_path] = _validate_device(
                device,
                field_name=f"device_map[{module_path!r}]",
                allow_auto=False,
            )
        return dict(sorted(normalized.items()))

    @field_validator("loader_options")
    @classmethod
    def _loader_options(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_options(value)

    @model_validator(mode="after")
    def _normalize_and_validate(self) -> Self:
        object.__setattr__(self, "device_map", _FrozenDict(self.device_map))
        object.__setattr__(self, "loader_options", _freeze_json(self.loader_options))
        if self.trust_remote_code != self.remote_code_acknowledged:
            if self.trust_remote_code:
                raise ValueError("trust_remote_code=True requires remote_code_acknowledged=True")
            raise ValueError("remote_code_acknowledged=True requires trust_remote_code=True")
        _validate_download_policy(self.download_policy, self.allow_downloads)
        self._validate_device_map_compatibility()
        return self

    def _validate_device_map_compatibility(self) -> None:
        mapped_devices = set(self.device_map.values())
        if not mapped_devices or self.device == DeviceKind.AUTO.value:
            return
        if self.device == DeviceKind.CPU.value:
            incompatible = mapped_devices - {DeviceKind.CPU.value}
        elif self.device == DeviceKind.MPS.value:
            incompatible = mapped_devices - {DeviceKind.MPS.value}
        elif self.device == DeviceKind.CUDA.value:
            incompatible = {
                item
                for item in mapped_devices
                if item != DeviceKind.CPU.value and not _CUDA_DEVICE.fullmatch(item)
            }
        elif self.device.startswith("cuda:"):
            incompatible = mapped_devices - {self.device, DeviceKind.CPU.value}
        else:
            incompatible = set()
        if incompatible:
            raise ValueError(
                f"device={self.device!r} is incompatible with device_map targets "
                f"{sorted(incompatible)!r}"
            )

    @property
    def security_warnings(self) -> tuple[str, ...]:
        """Return deterministic warnings derived from policy, never caller fields."""

        warnings: list[str] = []
        if self.trust_remote_code:
            warnings.append("trust_remote_code is enabled; execute repository code only if trusted")
        if self.quantization is not QuantizationPolicy.NONE:
            warnings.append(
                f"quantization={self.quantization.value} may limit semantic capture fidelity"
            )
        if self.device == DeviceKind.MPS.value or DeviceKind.MPS.value in self.device_map.values():
            warnings.append("MPS support is best-effort and remains runtime-dependent")
        if self.allow_downloads:
            warnings.append("model downloads are explicitly allowed by this loading policy")
        return tuple(sorted(warnings))


class RevisionEvidenceKind(str, Enum):
    """Canonical immutable-reference forms accepted from an external resolver."""

    GIT_COMMIT = "git_commit"
    SHA256 = "sha256"


class ImmutableRevisionEvidence(VersionedLoadingModel):
    """Structured evidence that cannot label a branch/tag as immutable."""

    kind: RevisionEvidenceKind
    digest: StrictStr
    evidence_source: StrictStr

    @field_validator("digest")
    @classmethod
    def _digest(cls, value: str) -> str:
        if not value or value != value.strip() or value != value.lower():
            raise ValueError("immutable revision digest must be lowercase hexadecimal")
        return value

    @field_validator("evidence_source")
    @classmethod
    def _evidence_source(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("evidence_source must be non-empty and trimmed")
        return value

    @model_validator(mode="after")
    def _canonical_digest(self) -> Self:
        if self.kind is RevisionEvidenceKind.GIT_COMMIT and not _GIT_COMMIT.fullmatch(self.digest):
            raise ValueError("git_commit evidence requires a full 40-character commit digest")
        if self.kind is RevisionEvidenceKind.SHA256 and not _SHA256_DIGEST.fullmatch(self.digest):
            raise ValueError("sha256 evidence requires a full 64-character digest")
        return self

    @property
    def canonical_revision(self) -> str:
        """Return the only resolved revision string permitted by this evidence."""

        if self.kind is RevisionEvidenceKind.GIT_COMMIT:
            return self.digest
        return f"sha256:{self.digest}"


class ResolvedSource(VersionedLoadingModel):
    """External resolution evidence kept separate from requested load intent."""

    source_type: SourceKind
    model_id: StrictStr
    requested_model_revision: StrictStr
    resolved_model_revision: StrictStr
    resolved_model_revision_evidence: ImmutableRevisionEvidence
    requested_tokenizer_revision: StrictStr | None = None
    resolved_tokenizer_revision: StrictStr | None = None
    resolved_tokenizer_revision_evidence: ImmutableRevisionEvidence | None = None
    resolution_method: StrictStr = "external_resolver"

    @field_validator("model_id")
    @classmethod
    def _model_id(cls, value: str) -> str:
        return _stable_token(value, field_name="model_id")

    @field_validator(
        "requested_model_revision",
        "resolved_model_revision",
        "requested_tokenizer_revision",
        "resolved_tokenizer_revision",
    )
    @classmethod
    def _revision_tokens(cls, value: str | None, info: Any) -> str | None:
        if value is None:
            return None
        return _revision_token(value, field_name=info.field_name)

    @field_validator("resolution_method")
    @classmethod
    def _evidence_tokens(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value or value != value.strip():
            raise ValueError("resolution method must be non-empty and trimmed")
        return value

    @model_validator(mode="after")
    def _model_key_compatible(self) -> Self:
        make_model_key(self.model_id, self.requested_model_revision)
        return self

    @model_validator(mode="after")
    def _evidence_pairs(self) -> Self:
        if (self.resolved_tokenizer_revision is None) != (
            self.resolved_tokenizer_revision_evidence is None
        ):
            raise ValueError(
                "resolved_tokenizer_revision and its immutable resolution evidence must be paired"
            )
        if self.resolved_model_revision != self.resolved_model_revision_evidence.canonical_revision:
            raise ValueError(
                "resolved_model_revision does not match its immutable evidence kind/digest"
            )
        if self.resolved_tokenizer_revision is not None and (
            self.resolved_tokenizer_revision
            != self.resolved_tokenizer_revision_evidence.canonical_revision
        ):
            raise ValueError(
                "resolved_tokenizer_revision does not match its immutable evidence kind/digest"
            )
        if (
            self.resolved_tokenizer_revision is not None
            and self.requested_tokenizer_revision is None
        ):
            raise ValueError("resolved_tokenizer_revision requires requested_tokenizer_revision")
        return self


def _portable_tokenizer_intent(
    tokenizer: TokenizerRequest | None,
    model_revision: str,
) -> dict[str, Any] | None:
    if tokenizer is None:
        return None
    requested_revision = (
        model_revision if tokenizer.inherit_model_revision else tokenizer.requested_revision
    )
    return {
        "identifier": tokenizer.identifier,
        "inherit_model_revision": tokenizer.inherit_model_revision,
        "requested_revision": requested_revision,
    }


def _portable_source_intent(source: SourceRequest) -> dict[str, Any]:
    intent: dict[str, Any] = {
        "source_type": source.source_type,
        "model_id": source.model_id,
        "requested_revision": source.requested_revision,
        "tokenizer": _portable_tokenizer_intent(source.tokenizer, source.requested_revision),
    }
    if isinstance(source, HuggingFaceSource):
        intent["download_policy"] = source.download_policy.value
        intent["allow_downloads"] = source.allow_downloads
    if isinstance(source, CustomLoaderSource):
        intent["loader_reference"] = source.loader_reference
    # LocalSource.path and an instance runtime object are deliberately absent.
    return intent


def _portable_config_intent(config: LoadConfig) -> dict[str, Any]:
    return {
        "allow_downloads": config.allow_downloads,
        "device": config.device,
        "device_map": dict(config.device_map),
        "download_policy": config.download_policy.value,
        "dtype": config.dtype.value,
        "loader_options": config.loader_options,
        "quantization": config.quantization.value,
        "remote_code_acknowledged": config.remote_code_acknowledged,
        "trust_remote_code": config.trust_remote_code,
    }


def portable_loading_intent(source: SourceRequest, config: LoadConfig) -> dict[str, Any]:
    """Return only portable, serializable request intent for auditing/hashing."""

    return {"config": _portable_config_intent(config), "source": _portable_source_intent(source)}


def make_loading_plan_id(source: SourceRequest, config: LoadConfig) -> str:
    """Build a full-SHA ID over portable intent, excluding local path/runtime objects."""

    return f"loadplan:{stable_digest(portable_loading_intent(source, config))}"


def parse_loading_plan_id(plan_id: str) -> str:
    """Validate and return the digest from a canonical loading-plan ID."""

    if not isinstance(plan_id, str):
        raise TypeError(f"plan_id must be a string, got {type(plan_id).__name__}")
    match = _PLAN_ID.fullmatch(plan_id)
    if match is None:
        raise ValueError("plan_id must use canonical loadplan:<64 lowercase hex> form")
    return match.group(1)


class LoadingPlan(VersionedLoadingModel):
    """Frozen request plan for a future loader; no loading occurs here."""

    plan_id: StrictStr = ""
    source: SourceRequest
    config: LoadConfig = Field(default_factory=LoadConfig)
    resolution: ResolvedSource | None = None

    @field_validator("plan_id")
    @classmethod
    def _plan_id_shape(cls, value: str) -> str:
        if value:
            parse_loading_plan_id(value)
        return value

    @model_validator(mode="after")
    def _canonical_id_and_resolution(self) -> Self:
        expected = make_loading_plan_id(self.source, self.config)
        if "plan_id" not in self.model_fields_set:
            object.__setattr__(self, "plan_id", expected)
        elif self.plan_id != expected:
            raise ValueError(
                f"plan_id does not match portable loading intent; expected {expected!r}"
            )

        if isinstance(self.source, HuggingFaceSource) and (
            self.source.download_policy != self.config.download_policy
            or self.source.allow_downloads != self.config.allow_downloads
        ):
            raise ValueError(
                "Hugging Face source and load config download policies must agree explicitly"
            )

        if self.resolution is not None:
            if self.resolution.source_type.value != self.source.source_type:
                raise ValueError("resolution source_type must match the requested source")
            if self.resolution.model_id != self.source.model_id:
                raise ValueError("resolution model_id must match the requested source")
            if self.resolution.requested_model_revision != self.source.requested_revision:
                raise ValueError("resolution requested_model_revision must match the source")
            expected_tokenizer_revision = None
            if self.source.tokenizer is not None:
                expected_tokenizer_revision = _portable_tokenizer_intent(
                    self.source.tokenizer,
                    self.source.requested_revision,
                )["requested_revision"]
            if self.resolution.requested_tokenizer_revision != expected_tokenizer_revision:
                raise ValueError(
                    "resolution requested_tokenizer_revision must match the tokenizer request"
                )
        return self

    @property
    def security_warnings(self) -> tuple[str, ...]:
        """Return config-derived and source-derived warnings, never caller input."""

        warnings = list(self.config.security_warnings)
        if isinstance(self.source, HuggingFaceSource) and self.source.allow_downloads:
            warnings.append("Hugging Face source permits network downloads by explicit opt-in")
        if isinstance(self.source, CustomLoaderSource):
            warnings.append(
                "custom loader reference execution runs user code and is not imported by validation"
            )
        return tuple(sorted(set(warnings)))


__all__ = [
    "CustomLoaderSource",
    "DTypePolicy",
    "DeviceKind",
    "DownloadPolicy",
    "HuggingFaceSource",
    "ImmutableRevisionEvidence",
    "InstanceSource",
    "LOADING_SCHEMA_VERSION",
    "LoadConfig",
    "LoadingPlan",
    "LocalSource",
    "QuantizationPolicy",
    "ResolvedSource",
    "RevisionEvidenceKind",
    "SourceKind",
    "SourceRequest",
    "TokenizerRequest",
    "VersionedLoadingModel",
    "make_loading_plan_id",
    "parse_loading_plan_id",
    "portable_loading_intent",
]
