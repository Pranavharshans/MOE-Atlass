"""Strict, serializable probe-plan contracts.

Plans describe passive observation intent only. They do not contain runtime
objects or callbacks, so a plan can be hashed, reviewed, and resolved in a
torch-free process before a hook manager is constructed.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum, IntEnum
from typing import Any, Literal, Self

from pydantic import (
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from ..core import (
    ComponentKind,
    StrictManifestModel,
    VersionedManifest,
    parse_component_key,
    stable_digest,
    validate_stable_identifier,
)


class ProbeLevel(IntEnum):
    """Progressive probe capabilities defined by the MoEAtlas PRD."""

    STRUCTURE = 0
    ROUTING = 1
    EXPERT_ACTIVITY = 2
    FULL_ACTIVATIONS = 3
    GRADIENTS = 4
    INTERVENTION = 5


class HookPoint(str, Enum):
    """Duck-typed module hook surfaces understood by the passive manager."""

    FORWARD_PRE = "forward_pre"
    FORWARD = "forward"
    FULL_BACKWARD = "full_backward"

    def __str__(self) -> str:
        return self.value


class CaptureMode(str, Enum):
    """How a callback is permitted to describe captured values."""

    STATS = "stats"
    REDUCED = "reduced"
    RAW = "raw"


class ReductionPolicy(str, Enum):
    """Serializable reduction intent; no tensor reduction is performed here."""

    COUNTS = "counts"
    MEAN = "mean"
    MIN_MAX = "min_max"
    TOP_K = "top_k"
    NONE = "none"


def _validate_messages(value: tuple[str, ...], *, field_name: str) -> tuple[str, ...]:
    for message in value:
        if not message or message != message.strip():
            raise ValueError(f"{field_name} entries must be non-empty and trimmed")
    return value


class ProbeTarget(StrictManifestModel):
    """Portable target identity resolved against one named-module path."""

    module_path: StrictStr
    component_key: StrictStr | None = None
    component_kind: ComponentKind | None = None

    @field_validator("module_path")
    @classmethod
    def _module_path(cls, value: str) -> str:
        return validate_stable_identifier(value, field_name="probe target module_path")

    @field_validator("component_key")
    @classmethod
    def _component_key(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parse_component_key(value)
        return value

    @model_validator(mode="after")
    def _component_identity_pair(self) -> Self:
        if (self.component_key is None) != (self.component_kind is None):
            raise ValueError(
                "probe target component_key and component_kind must be provided together"
            )
        return self


class CapturePolicy(StrictManifestModel):
    """Bounded capture, reduction, and sampling intent."""

    mode: CaptureMode = CaptureMode.STATS
    reduction: ReductionPolicy = ReductionPolicy.COUNTS
    include_inputs: StrictBool = False
    include_outputs: StrictBool = True
    include_gradients: StrictBool = False
    raw_opt_in: StrictBool = False
    max_items: StrictInt | None = Field(default=None, ge=1)
    max_bytes: StrictInt | None = Field(default=None, ge=1)
    sample_rate: StrictFloat = Field(default=1.0, gt=0, le=1)
    sample_seed: StrictInt | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _bounded_raw_capture(self) -> Self:
        if (self.mode is CaptureMode.RAW) != (self.reduction is ReductionPolicy.NONE):
            raise ValueError("CaptureMode.RAW and ReductionPolicy.NONE must be selected together")
        if self.sample_rate < 1 and self.sample_seed is None:
            raise ValueError("sample_rate < 1 requires sample_seed for reproducibility")
        raw_mode = self.mode is CaptureMode.RAW
        has_budget = self.max_items is not None or self.max_bytes is not None
        if raw_mode and not self.raw_opt_in:
            raise ValueError("raw capture requires raw_opt_in=True")
        if self.raw_opt_in and not has_budget:
            raise ValueError("raw_opt_in=True requires a positive max_items or max_bytes budget")
        return self


_HOOK_ORDER = {
    HookPoint.FORWARD_PRE: 0,
    HookPoint.FORWARD: 1,
    HookPoint.FULL_BACKWARD: 2,
}

_ALLOWED_HOOKS: dict[ProbeLevel, frozenset[HookPoint]] = {
    ProbeLevel.STRUCTURE: frozenset(),
    ProbeLevel.ROUTING: frozenset({HookPoint.FORWARD_PRE, HookPoint.FORWARD}),
    ProbeLevel.EXPERT_ACTIVITY: frozenset({HookPoint.FORWARD}),
    ProbeLevel.FULL_ACTIVATIONS: frozenset({HookPoint.FORWARD_PRE, HookPoint.FORWARD}),
    ProbeLevel.GRADIENTS: frozenset({HookPoint.FORWARD, HookPoint.FULL_BACKWARD}),
    ProbeLevel.INTERVENTION: frozenset({HookPoint.FORWARD_PRE, HookPoint.FORWARD}),
}


def make_probe_plan_id(payload: Mapping[str, Any]) -> str:
    """Return a stable plan identifier from JSON-compatible plan data."""

    return f"plan:{stable_digest(dict(payload))}"


class ProbePlan(VersionedManifest):
    """Versioned intent for one passive or explicitly opted-in probe."""

    manifest_type: Literal["probe_plan"] = "probe_plan"
    plan_id: StrictStr = ""
    level: ProbeLevel
    hook_points: tuple[HookPoint, ...] = ()
    targets: tuple[ProbeTarget, ...] = Field(min_length=1)
    include: tuple[StrictStr, ...] = ()
    exclude: tuple[StrictStr, ...] = ()
    capture: CapturePolicy = Field(default_factory=CapturePolicy)
    intervention_opt_in: StrictBool = False

    @field_validator("plan_id")
    @classmethod
    def _plan_id_token(cls, value: str) -> str:
        if value and (not value.startswith("plan:") or len(value) != len("plan:") + 64):
            raise ValueError("plan_id must be plan:<64-character sha256 digest>")
        return value

    @field_validator("include", "exclude")
    @classmethod
    def _selector_paths(cls, value: tuple[str, ...], info: Any) -> tuple[str, ...]:
        for path in value:
            validate_stable_identifier(path, field_name=f"probe {info.field_name} path")
        return _validate_messages(value, field_name=f"probe {info.field_name}")

    @model_validator(mode="after")
    def _invariants_and_identity(self) -> Self:
        object.__setattr__(
            self,
            "hook_points",
            tuple(sorted(self.hook_points, key=lambda point: _HOOK_ORDER[point])),
        )
        object.__setattr__(
            self,
            "targets",
            tuple(
                sorted(
                    self.targets,
                    key=lambda target: (target.module_path, target.component_key or ""),
                )
            ),
        )
        object.__setattr__(self, "include", tuple(sorted(self.include)))
        object.__setattr__(self, "exclude", tuple(sorted(self.exclude)))

        if len(set(self.hook_points)) != len(self.hook_points):
            raise ValueError("hook_points must not contain duplicates")
        target_paths = [target.module_path for target in self.targets]
        if len(set(target_paths)) != len(target_paths):
            raise ValueError("probe targets must not share a module_path")
        if len(set(self.include)) != len(self.include):
            raise ValueError("probe include paths must not contain duplicates")
        if len(set(self.exclude)) != len(self.exclude):
            raise ValueError("probe exclude paths must not contain duplicates")
        overlap = set(self.include) & set(self.exclude)
        if overlap:
            raise ValueError(f"probe include and exclude paths overlap: {sorted(overlap)!r}")

        allowed = _ALLOWED_HOOKS[self.level]
        incompatible = set(self.hook_points) - allowed
        if incompatible:
            names = sorted(point.value for point in incompatible)
            raise ValueError(
                f"hook points {names!r} are incompatible with probe level {self.level.name}"
            )
        if self.level is ProbeLevel.STRUCTURE and self.hook_points:
            raise ValueError("STRUCTURE level is static and cannot register hook points")
        if self.level > ProbeLevel.STRUCTURE and not self.hook_points:
            raise ValueError(f"probe level {self.level.name} requires at least one hook point")
        if self.level is ProbeLevel.FULL_ACTIVATIONS:
            if HookPoint.FORWARD not in self.hook_points:
                raise ValueError("FULL_ACTIVATIONS requires the forward hook point")
            if not self.capture.include_outputs:
                raise ValueError("FULL_ACTIVATIONS requires capture.include_outputs=True")
            if (
                self.capture.mode is not CaptureMode.RAW
                or self.capture.reduction is not ReductionPolicy.NONE
            ):
                raise ValueError(
                    "FULL_ACTIVATIONS requires CaptureMode.RAW and ReductionPolicy.NONE"
                )
        if self.level is ProbeLevel.GRADIENTS:
            if HookPoint.FULL_BACKWARD not in self.hook_points:
                raise ValueError("GRADIENTS requires the full_backward hook point")
            if not self.capture.include_gradients:
                raise ValueError("GRADIENTS requires capture.include_gradients=True")
        if self.capture.include_gradients and self.level is not ProbeLevel.GRADIENTS:
            raise ValueError("capture.include_gradients requires exactly the GRADIENTS probe level")
        if self.level >= ProbeLevel.FULL_ACTIVATIONS:
            if not self.capture.raw_opt_in:
                raise ValueError(
                    f"{self.level.name} requires capture.raw_opt_in=True for full values"
                )
            if self.capture.max_items is None and self.capture.max_bytes is None:
                raise ValueError(
                    f"{self.level.name} requires a positive max_items or max_bytes budget"
                )
        if self.level is ProbeLevel.INTERVENTION and not self.intervention_opt_in:
            raise ValueError("INTERVENTION requires intervention_opt_in=True")

        expected_id = make_probe_plan_id(self._identity_payload())
        if not self.plan_id:
            object.__setattr__(self, "plan_id", expected_id)
        elif self.plan_id != expected_id:
            raise ValueError(f"plan_id does not match this plan; expected {expected_id!r}")
        return self

    def _identity_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json", exclude={"plan_id"})
        return payload

    @property
    def plan_hash(self) -> str:
        """Return the digest portion of ``plan_id`` for cache keys."""

        return self.plan_id.removeprefix("plan:")

    def resolve(self, model: object):
        """Resolve this plan against a duck-typed named-module surface."""

        from .resolver import resolve_probe_plan

        return resolve_probe_plan(self, model)


__all__ = [
    "CaptureMode",
    "CapturePolicy",
    "HookPoint",
    "ProbeLevel",
    "ProbePlan",
    "ProbeTarget",
    "ReductionPolicy",
    "make_probe_plan_id",
]
