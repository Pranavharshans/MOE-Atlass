"""Strict, normalized, model-runtime-independent MoE event contracts.

This module defines event identity and evidence only. It does not capture
tensors, write storage, install hooks, load models, or infer missing values.
"""

from __future__ import annotations

import json
import math
from enum import Enum
from typing import Annotated, Any, Literal, Self, TypeAlias

from pydantic import (
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from .core import (
    StrictManifestModel,
    make_token_key,
    parse_component_key,
    parse_token_key,
    validate_stable_identifier,
)

EVENT_SCHEMA_VERSION = "1.0"


class TokenPhase(str, Enum):
    """Generation phase attached to a token identity."""

    PREFILL = "prefill"
    DECODE = "decode"

    def __str__(self) -> str:
        return self.value


class _FrozenMetadata(dict[str, Any]):
    """Dict-compatible immutable JSON object used for event metadata."""

    @staticmethod
    def _immutable(*args: Any, **kwargs: Any) -> None:
        raise TypeError("event metadata is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable

    def __ior__(self, other: object) -> _FrozenMetadata:
        self._immutable(other)
        return self


class _FrozenList(list[Any]):
    """List-compatible immutable JSON array used inside event metadata."""

    @staticmethod
    def _immutable(*args: Any, **kwargs: Any) -> None:
        raise TypeError("event metadata is immutable")

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


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return _FrozenMetadata({key: _freeze_json(value[key]) for key in sorted(value)})
    if isinstance(value, list | tuple):
        return _FrozenList(_freeze_json(item) for item in value)
    return value


def _json_metadata(value: dict[str, Any]) -> dict[str, Any]:
    try:
        _validate_json_node(value)
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must contain only finite, JSON-serializable values") from exc
    return value


def _validate_json_node(value: Any) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError("metadata object keys must be strings")
            _validate_json_node(nested)
    elif isinstance(value, list | tuple):
        for nested in value:
            _validate_json_node(nested)
    elif value is None or isinstance(value, str | int | bool):
        return
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("metadata values must be finite")
    else:
        raise TypeError(f"metadata contains unsupported value type {type(value).__name__}")


def _finite(value: float | None, *, field_name: str) -> float | None:
    if value is not None and not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    return value


class EventModel(StrictManifestModel):
    """Shared strict schema and JSON helpers for normalized events."""

    schema_version: Literal["1.0"] = Field(default=EVENT_SCHEMA_VERSION, frozen=True)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible data with deterministic nested metadata."""

        return self.model_dump(mode="json")

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize the event without NaN/Infinity or process-dependent ordering."""

        separators = (",", ":") if indent is None else None
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=False,
            separators=separators,
            indent=indent,
        )

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> Self:
        """Validate a JSON document against the concrete event schema."""

        return cls.model_validate_json(payload)


class TokenEvent(EventModel):
    """One generated or prompt token with a portable canonical identity."""

    event_type: Literal["token"] = "token"
    token_key: StrictStr = ""
    run_key: StrictStr
    sequence_id: StrictStr
    token_pos: StrictInt = Field(ge=0)
    token_id: StrictInt = Field(ge=0)
    token_text: StrictStr
    phase: TokenPhase

    @field_validator("run_key", "sequence_id")
    @classmethod
    def _identifier(cls, value: str, info: Any) -> str:
        return validate_stable_identifier(value, field_name=info.field_name)

    @field_validator("token_key")
    @classmethod
    def _token_key_shape(cls, value: str) -> str:
        parse_token_key(value)
        return value

    @model_validator(mode="after")
    def _canonical_identity(self) -> Self:
        expected = make_token_key(
            self.run_key,
            self.sequence_id,
            self.token_pos,
            self.token_id,
            self.phase.value,
        )
        if "token_key" not in self.model_fields_set:
            object.__setattr__(self, "token_key", expected)
        elif self.token_key != expected:
            raise ValueError(f"token_key does not match this token; expected {expected!r}")
        return self


class RoutingEvent(EventModel):
    """One router/expert assignment with capability-driven partial evidence."""

    event_type: Literal["routing"] = "routing"
    token_key: StrictStr
    layer_key: StrictStr
    rank: StrictInt = Field(ge=0)
    expert_key: StrictStr
    router_logit: StrictFloat | None = None
    probability: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    weight: StrictFloat | None = None
    selected: StrictBool

    @field_validator("token_key")
    @classmethod
    def _token_key_shape(cls, value: str) -> str:
        parse_token_key(value)
        return value

    @field_validator("layer_key", "expert_key")
    @classmethod
    def _component_key_shape(cls, value: str) -> str:
        parse_component_key(value)
        return value

    @field_validator("router_logit", "probability", "weight")
    @classmethod
    def _finite_numeric(cls, value: float | None, info: Any) -> float | None:
        return _finite(value, field_name=info.field_name)

    @model_validator(mode="after")
    def _has_numeric_evidence(self) -> Self:
        if self.layer_key == self.expert_key:
            raise ValueError(
                "routing event layer_key and expert_key must identify different components"
            )
        if self.router_logit is None and self.probability is None and self.weight is None:
            raise ValueError("routing event requires router_logit, probability, or weight evidence")
        return self


class ExpertEvent(EventModel):
    """Expert measurement evidence linked to a canonical token and expert."""

    event_type: Literal["expert"] = "expert"
    token_key: StrictStr
    expert_key: StrictStr
    input_norm: StrictFloat | None = Field(default=None, ge=0.0)
    output_norm: StrictFloat | None = Field(default=None, ge=0.0)
    contribution_norm: StrictFloat | None = Field(default=None, ge=0.0)
    latency_ms: StrictFloat | None = Field(default=None, ge=0.0)
    metadata: dict[StrictStr, Any] = Field(default_factory=dict)

    @field_validator("token_key")
    @classmethod
    def _token_key_shape(cls, value: str) -> str:
        parse_token_key(value)
        return value

    @field_validator("expert_key")
    @classmethod
    def _expert_key_shape(cls, value: str) -> str:
        parse_component_key(value)
        return value

    @field_validator("input_norm", "output_norm", "contribution_norm", "latency_ms")
    @classmethod
    def _finite_nonnegative(cls, value: float | None, info: Any) -> float | None:
        return _finite(value, field_name=info.field_name)

    @field_validator("metadata")
    @classmethod
    def _metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _json_metadata(value)

    @model_validator(mode="after")
    def _freeze_and_require_evidence(self) -> Self:
        frozen_metadata = _freeze_json(self.metadata)
        object.__setattr__(self, "metadata", frozen_metadata)
        has_measurement = any(
            value is not None
            for value in (
                self.input_norm,
                self.output_norm,
                self.contribution_norm,
                self.latency_ms,
            )
        )
        if not has_measurement and not self.metadata:
            raise ValueError(
                "expert event requires a norm/latency measurement or non-empty metadata"
            )
        return self


Event: TypeAlias = Annotated[
    TokenEvent | RoutingEvent | ExpertEvent,
    Field(discriminator="event_type"),
]


__all__ = [
    "EVENT_SCHEMA_VERSION",
    "Event",
    "EventModel",
    "ExpertEvent",
    "RoutingEvent",
    "TokenEvent",
    "TokenPhase",
]
