"""Versioned, strict Pydantic contracts for canonical MoE manifests.

These models describe observed model structure and capture provenance only.
They never import a model runtime and can be created from synthetic or static
data in CPU-only environments.
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any, ClassVar, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from .capabilities import CapabilityLabel, CaptureSource
from .identity import make_component_key, parse_model_key, validate_stable_identifier

SCHEMA_VERSION = "1.0"
_CONFIG_HASH_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+-]*$")


class DType(str, Enum):
    """Dtype values recorded by a model manifest."""

    FLOAT64 = "float64"
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    INT8 = "int8"
    UINT8 = "uint8"
    INT4 = "int4"
    UNKNOWN = "unknown"

    def __str__(self) -> str:
        return self.value


class ComponentKind(str, Enum):
    """Semantic component kinds understood by the manifest contract."""

    MOE_LAYER = "moe_layer"
    ROUTER = "router"
    EXPERT_CONTAINER = "expert_container"
    EXPERT = "expert"
    SHARED_EXPERT = "shared_expert"
    MODULE = "module"

    def __str__(self) -> str:
        return self.value


class StrictManifestModel(BaseModel):
    """Shared Pydantic v2 policy for all public manifest models."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_assignment=True,
    )


class VersionedManifest(StrictManifestModel):
    """Base class adding the explicit schema version and JSON helpers."""

    schema_version: Literal["1.0"] = Field(default=SCHEMA_VERSION, frozen=True)
    manifest_type: ClassVar[str]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary, including enum values."""

        return self.model_dump(mode="json")

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize this manifest to JSON using the versioned schema."""

        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> Self:
        """Validate a JSON document against this manifest's strict schema."""

        return cls.model_validate_json(payload)


class TokenizerIdentity(StrictManifestModel):
    """Tokenizer identity and immutable revision stored with a model run."""

    identifier: StrictStr
    revision: StrictStr

    @field_validator("identifier", "revision")
    @classmethod
    def _stable_token(cls, value: str, info: Any) -> str:
        return validate_stable_identifier(value, field_name=f"tokenizer {info.field_name}")


class Provenance(StrictManifestModel):
    """Portable provenance attached to a manifest or component."""

    source: StrictStr
    tool_version: StrictStr
    metadata: dict[StrictStr, Any] = Field(default_factory=dict)

    @field_validator("source", "tool_version")
    @classmethod
    def _nonempty_token(cls, value: str, info: Any) -> str:
        if not value or value != value.strip():
            raise ValueError(f"{info.field_name} must be a non-empty canonical string")
        return value

    @field_validator("metadata")
    @classmethod
    def _json_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must contain only JSON-serializable values") from exc
        return value


class CaptureProvenance(StrictManifestModel):
    """Evidence about how a component value was captured or decoded."""

    source: CaptureSource
    method: StrictStr
    adapter: StrictStr | None = None
    adapter_version: StrictStr | None = None
    verified: StrictBool = False
    metadata: dict[StrictStr, Any] = Field(default_factory=dict)

    @field_validator("method", "adapter", "adapter_version")
    @classmethod
    def _optional_nonempty_token(cls, value: str | None, info: Any) -> str | None:
        if value is not None and (not value or value != value.strip()):
            raise ValueError(f"{info.field_name} must be a non-empty canonical string")
        return value

    @field_validator("metadata")
    @classmethod
    def _json_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must contain only JSON-serializable values") from exc
        return value

    @model_validator(mode="after")
    def _adapter_pair(self) -> Self:
        if (self.adapter is None) != (self.adapter_version is None):
            raise ValueError("adapter and adapter_version must be provided together")
        return self


def _validate_messages(value: list[str], *, field_name: str) -> list[str]:
    for message in value:
        if not message or message != message.strip():
            raise ValueError(f"{field_name} entries must be non-empty and trimmed")
    return value


class ModelManifest(VersionedManifest):
    """Portable identity and runtime metadata for one executable MoE model."""

    manifest_type: Literal["model_manifest"] = "model_manifest"
    model_key: StrictStr
    architecture: StrictStr
    revision: StrictStr
    config_hash: StrictStr
    tokenizer: TokenizerIdentity
    dtype: DType
    device_map: dict[StrictStr, StrictStr] = Field(default_factory=dict)
    provenance: Provenance | None = None
    warnings: list[StrictStr] = Field(default_factory=list)

    @field_validator("model_key", "revision")
    @classmethod
    def _stable_identity(cls, value: str, info: Any) -> str:
        if info.field_name == "model_key":
            parse_model_key(value)
        return validate_stable_identifier(value, field_name=info.field_name)

    @field_validator("architecture")
    @classmethod
    def _architecture_name(cls, value: str) -> str:
        if not value or value != value.strip() or any(character.isspace() for character in value):
            raise ValueError("architecture must be a non-empty string without whitespace")
        return value

    @field_validator("config_hash")
    @classmethod
    def _config_hash_token(cls, value: str) -> str:
        if not value or value != value.strip() or not _CONFIG_HASH_TOKEN.fullmatch(value):
            raise ValueError(
                "config_hash must be a non-empty canonical digest token "
                "(make_config_hash(...) returns sha256:<hex>)"
            )
        return value

    @field_validator("device_map")
    @classmethod
    def _device_assignments(cls, value: dict[str, str]) -> dict[str, str]:
        for module_path, device in value.items():
            if module_path:
                validate_stable_identifier(module_path, field_name="device_map module path")
            if (
                not device
                or device != device.strip()
                or any(character.isspace() for character in device)
            ):
                raise ValueError(
                    "device_map values must be non-empty device names without whitespace"
                )
        return value

    @field_validator("warnings")
    @classmethod
    def _warnings(cls, value: list[str]) -> list[str]:
        return _validate_messages(value, field_name="warnings")

    @model_validator(mode="after")
    def _revision_matches_model_key(self) -> Self:
        _, encoded_revision = parse_model_key(self.model_key)
        if encoded_revision != self.revision:
            raise ValueError(
                f"revision does not match the revision encoded in model_key ({encoded_revision!r})"
            )
        return self


class ComponentManifest(VersionedManifest):
    """Semantic component metadata linked to a model manifest."""

    manifest_type: Literal["component_manifest"] = "component_manifest"
    component_key: StrictStr
    model_key: StrictStr
    kind: ComponentKind
    module_path: StrictStr
    layer_index: StrictInt | None = Field(default=None, ge=0)
    expert_index: StrictInt | None = Field(default=None, ge=0)
    tensor_shapes: dict[StrictStr, list[StrictInt]] = Field(default_factory=dict)
    capabilities: list[CapabilityLabel] = Field(min_length=1)
    routed: StrictBool | None = None
    shared: StrictBool | None = None
    capture: CaptureProvenance | None = None
    provenance: Provenance | None = None
    warnings: list[StrictStr] = Field(default_factory=list)

    @field_validator("component_key", "model_key")
    @classmethod
    def _stable_keys(cls, value: str, info: Any) -> str:
        if info.field_name == "model_key":
            parse_model_key(value)
        return validate_stable_identifier(value, field_name=info.field_name)

    @field_validator("module_path")
    @classmethod
    def _module_path(cls, value: str) -> str:
        return validate_stable_identifier(value, field_name="module_path")

    @field_validator("tensor_shapes")
    @classmethod
    def _tensor_shapes(cls, value: dict[str, list[int]]) -> dict[str, list[int]]:
        for name, shape in value.items():
            if not name or name != name.strip():
                raise ValueError("tensor_shapes keys must be non-empty and trimmed")
            if any(dimension < 0 for dimension in shape):
                raise ValueError("tensor_shapes dimensions must be non-negative")
        return value

    @field_validator("capabilities")
    @classmethod
    def _capabilities(cls, value: list[CapabilityLabel]) -> list[CapabilityLabel]:
        if len(set(value)) != len(value):
            raise ValueError("capabilities must not contain duplicate labels")
        if CapabilityLabel.UNSUPPORTED in value and len(value) != 1:
            raise ValueError("UNSUPPORTED must be the only capability label")
        return value

    @field_validator("warnings")
    @classmethod
    def _warnings(cls, value: list[str]) -> list[str]:
        return _validate_messages(value, field_name="warnings")

    @model_validator(mode="after")
    def _semantic_invariants(self) -> Self:
        expected_component_key = make_component_key(
            self.model_key,
            self.kind.value,
            self.module_path,
            layer_index=self.layer_index,
            expert_index=self.expert_index,
        )
        if self.component_key != expected_component_key:
            raise ValueError(
                "component_key does not match this component identity; "
                f"expected {expected_component_key!r}"
            )
        if self.routed is True and self.shared is True:
            raise ValueError("a component cannot be both routed and shared")
        if self.kind is ComponentKind.SHARED_EXPERT and self.shared is not True:
            raise ValueError("shared_expert components must set shared=True")
        if self.expert_index is not None and self.kind not in {
            ComponentKind.EXPERT,
            ComponentKind.SHARED_EXPERT,
        }:
            raise ValueError("expert_index is only valid for expert components")
        if CapabilityLabel.FULL in self.capabilities and (
            self.capture is None or self.capture.verified is not True
        ):
            raise ValueError("FULL capability requires verified capture provenance (verified=True)")
        if CapabilityLabel.EXPERIMENTAL in self.capabilities and (
            self.capture is None or self.capture.verified is not False
        ):
            raise ValueError(
                "EXPERIMENTAL capability requires unverified capture provenance (verified=False)"
            )
        if CapabilityLabel.UNSUPPORTED in self.capabilities and not self.warnings:
            raise ValueError("UNSUPPORTED capability requires at least one warning")
        return self
