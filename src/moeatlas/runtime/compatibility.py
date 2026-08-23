"""Model-neutral runtime compatibility preflight contracts.

This module is deliberately declarative.  It does not import Transformers,
execute Hub code, inspect a model object, or contact the network.  A caller
supplies the small metadata document it has already inspected and a set of
available runtime profiles.  The resolver then chooses the best compatible
profile, records any *approved* compatibility bridges, and produces a cache
identity bound to the immutable model revision and hardware fingerprint.

The important boundary is that static metadata proposes a runtime; a worker
that actually imports or loads a model remains responsible for validating the
selection.  No model-family or model-name branches belong here.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar

from ..core import canonical_identifier, stable_digest, validate_stable_identifier

COMPATIBILITY_SCHEMA_VERSION = "1.0"
_IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}$")
_PACKAGE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*")
_IMPORT_PATH = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*(?::[A-Za-z_][A-Za-z0-9_.]*)?$")
_VERSION = re.compile(r"^[0-9]+(?:\.[0-9]+)*(?:[-+][A-Za-z0-9.-]+)?$")
_VERSION_COMPARISON = re.compile(
    r"^(==|!=|>=|<=|>|<|~=)?\s*"
    r"([0-9]+(?:\.[0-9]+)*(?:[-+][A-Za-z0-9.-]+)?)$"
)
_PACKAGE_REQUIREMENT = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9_.-]*)"
    r"\s*(?P<constraint>(?:==|!=|>=|<=|>|<|~=)\s*"
    r"[0-9]+(?:\.[0-9]+)*(?:[-+][A-Za-z0-9.-]+)?)?$"
)


class RuntimeCompatibilityError(ValueError):
    """Raised when declarative compatibility input is malformed."""


class RuntimeSelectionStatus(str, Enum):
    """Bounded outcome of runtime preflight."""

    READY = "ready"
    INCOMPATIBLE = "incompatible"

    def __str__(self) -> str:
        return self.value


def _text(value: object, *, field_name: str, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    if type(value) is not str or not value or value != value.strip():
        raise RuntimeCompatibilityError(f"{field_name} must be a non-empty trimmed string")
    if any(ord(char) < 32 for char in value):
        raise RuntimeCompatibilityError(f"{field_name} must not contain control characters")
    return value


def _token(value: object, *, field_name: str) -> str:
    text = _text(value, field_name=field_name)
    assert text is not None
    return text


def _immutable_revision(value: object) -> str:
    revision = _token(value, field_name="immutable_revision")
    if _IMMUTABLE_REVISION.fullmatch(revision) is None:
        raise RuntimeCompatibilityError(
            "immutable_revision must be a lowercase 40-character commit SHA"
        )
    return revision


def _unique_tokens(values: object, *, field_name: str, validator: Any = _token) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise RuntimeCompatibilityError(f"{field_name} must be a sequence of strings")
    normalized = tuple(validator(value, field_name=f"{field_name}[]") for value in values)
    if len(set(normalized)) != len(normalized):
        raise RuntimeCompatibilityError(f"{field_name} must not contain duplicate entries")
    return tuple(sorted(normalized))


def _import_token(value: object, *, field_name: str) -> str:
    token = _token(value, field_name=field_name)
    if _IMPORT_PATH.fullmatch(token) is None:
        raise RuntimeCompatibilityError(
            f"{field_name} must use module or module:attribute import notation"
        )
    return token


def _package_token(value: object, *, field_name: str) -> str:
    token = _token(value, field_name=field_name)
    match = _PACKAGE_REQUIREMENT.fullmatch(token)
    if match is None:
        raise RuntimeCompatibilityError(
            f"{field_name} must be a package name with an optional version"
        )
    # Preserve a version constraint while exposing the normalized package name
    # through ``package_name``.  Requirement strings are intentionally not
    # evaluated here; runtime profiles carry the installed package versions.
    return token


def package_name(requirement: str) -> str:
    """Return the normalized package name from a requirement token."""

    token = _token(requirement, field_name="package requirement")
    match = _PACKAGE_REQUIREMENT.fullmatch(token)
    if match is None:  # pragma: no cover - guarded by RuntimeRequirements
        raise RuntimeCompatibilityError("package requirement has no package name")
    return match.group("name").casefold().replace("_", "-")


def _package_constraint(requirement: str) -> tuple[str, str | None]:
    token = _package_token(requirement, field_name="package requirement")
    match = _PACKAGE_REQUIREMENT.fullmatch(token)
    assert match is not None  # validated by _package_token
    name = match.group("name").casefold().replace("_", "-")
    constraint = match.group("constraint")
    return name, constraint.strip() if constraint else None


def _normalized_package_entry(requirement: str) -> str:
    name, constraint = _package_constraint(requirement)
    return name + (constraint.replace(" ", "") if constraint else "")


def _version_tuple(value: str) -> tuple[int, ...]:
    if _VERSION.fullmatch(value) is None:
        raise RuntimeCompatibilityError(f"unsupported version token: {value!r}")
    numeric = re.split(r"[-+]", value, maxsplit=1)[0]
    return tuple(int(part) for part in numeric.split("."))


def _compare_versions(left: str, right: str) -> int:
    left_parts = _version_tuple(left)
    right_parts = _version_tuple(right)
    width = max(len(left_parts), len(right_parts))
    left_padded = left_parts + (0,) * (width - len(left_parts))
    right_padded = right_parts + (0,) * (width - len(right_parts))
    return (left_padded > right_padded) - (left_padded < right_padded)


def version_satisfies(version: str, requirement: str | None) -> bool:
    """Evaluate the small, deterministic version subset used by preflight.

    Requirement clauses use the common comparison operators and comma for
    conjunction (for example ``>=4.45,<5``).  This intentionally avoids a
    dependency on ``packaging`` in the core application.
    """

    stable_version = _token(version, field_name="version")
    if requirement is None or requirement == "":
        _version_tuple(stable_version)
        return True
    clauses = [part.strip() for part in requirement.split(",")]
    if not clauses or any(not clause for clause in clauses):
        raise RuntimeCompatibilityError("version requirement contains an empty clause")
    for clause in clauses:
        match = _VERSION_COMPARISON.fullmatch(clause)
        if match is None:
            raise RuntimeCompatibilityError(f"unsupported version requirement: {requirement!r}")
        operator = match.group(1) or "=="
        target = match.group(2)
        comparison = _compare_versions(stable_version, target)
        if operator == "==" and comparison != 0:
            return False
        if operator == "!=" and comparison == 0:
            return False
        if operator == ">=" and comparison < 0:
            return False
        if operator == "<=" and comparison > 0:
            return False
        if operator == ">" and comparison <= 0:
            return False
        if operator == "<" and comparison >= 0:
            return False
        if operator == "~=" and (
            comparison < 0
            or _version_tuple(stable_version)[0] != _version_tuple(target)[0]
        ):
            return False
    return True


@dataclass(frozen=True, slots=True)
class RuntimeRequirements:
    """Requirements extracted from config and remote-code metadata.

    ``imports`` contains module paths or ``module:attribute`` symbols that a
    remote module needs.  They are represented as data; this class never
    imports them.  ``packages`` may carry ordinary version constraints.
    """

    transformers: str | None = None
    python: str | None = None
    packages: tuple[str, ...] = ()
    imports: tuple[str, ...] = ()
    remote_code: bool = False
    cuda: bool = False
    dtype: str | None = None

    def __post_init__(self) -> None:
        if self.transformers is not None:
            _token(self.transformers, field_name="transformers requirement")
        if self.python is not None:
            _token(self.python, field_name="python requirement")
        if type(self.remote_code) is not bool or type(self.cuda) is not bool:
            raise RuntimeCompatibilityError("remote_code and cuda must be exact booleans")
        if self.dtype is not None:
            _token(self.dtype, field_name="dtype")
        object.__setattr__(
            self,
            "packages",
            _unique_tokens(self.packages, field_name="packages", validator=_package_token),
        )
        object.__setattr__(
            self,
            "imports",
            _unique_tokens(self.imports, field_name="imports", validator=_import_token),
        )


@dataclass(frozen=True, slots=True)
class ModelRuntimeMetadata:
    """Static model metadata used to select a worker profile.

    The revision must already be resolved to an immutable Hub commit.  The
    optional :meth:`from_config` constructor accepts a JSON-like config and
    only reads declarative fields; it does not execute ``auto_map`` code.
    """

    model_id: str
    immutable_revision: str
    requirements: RuntimeRequirements = RuntimeRequirements()
    model_type: str | None = None
    architectures: tuple[str, ...] = ()
    auto_map: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        identifier = validate_stable_identifier(self.model_id, field_name="model_id")
        object.__setattr__(self, "model_id", identifier)
        object.__setattr__(self, "immutable_revision", _immutable_revision(self.immutable_revision))
        if not isinstance(self.requirements, RuntimeRequirements):
            raise RuntimeCompatibilityError("requirements must be RuntimeRequirements")
        for field_name in ("architectures", "auto_map"):
            object.__setattr__(
                self,
                field_name,
                _unique_tokens(getattr(self, field_name), field_name=field_name),
            )
        if self.model_type is not None:
            _token(self.model_type, field_name="model_type")

    @classmethod
    def from_config(
        cls,
        model_id: str,
        immutable_revision: str,
        config: Mapping[str, object],
        *,
        imports: Iterable[str] = (),
        packages: Iterable[str] = (),
        remote_code: bool | None = None,
        python: str | None = None,
        cuda: bool = False,
        dtype: str | None = None,
    ) -> ModelRuntimeMetadata:
        """Build metadata by reading known JSON config fields only.

        ``auto_map`` presence marks a model as requiring remote code unless
        callers explicitly override it.  The value is retained only as an
        explainable fingerprint; no referenced module is loaded here.
        """

        if not isinstance(config, Mapping):
            raise RuntimeCompatibilityError("model config must be a mapping")
        architectures_value = config.get("architectures", ())
        auto_map_value = config.get("auto_map", {})
        if isinstance(architectures_value, str) or not isinstance(architectures_value, Sequence):
            raise RuntimeCompatibilityError("config.architectures must be a sequence")
        if not isinstance(auto_map_value, Mapping):
            raise RuntimeCompatibilityError("config.auto_map must be an object")
        auto_map = tuple(
            sorted(
                f"{_token(key, field_name='auto_map key')}="
                f"{_token(value, field_name='auto_map value')}"
                for key, value in auto_map_value.items()
            )
        )
        declared_remote = bool(auto_map)
        if remote_code is None:
            remote_code = declared_remote
        if type(remote_code) is not bool:
            raise RuntimeCompatibilityError("remote_code must be an exact bool")
        declared_transformers = config.get("transformers_version")
        if declared_transformers is not None and type(declared_transformers) is not str:
            raise RuntimeCompatibilityError("config.transformers_version must be a string")
        requirements = RuntimeRequirements(
            transformers=declared_transformers,
            python=python,
            packages=tuple(packages),
            imports=tuple(imports),
            remote_code=remote_code,
            cuda=cuda,
            dtype=dtype,
        )
        return cls(
            model_id=model_id,
            immutable_revision=immutable_revision,
            requirements=requirements,
            model_type=(
                config.get("model_type") if isinstance(config.get("model_type"), str) else None
            ),
            architectures=tuple(architectures_value),
            auto_map=auto_map,
        )


@dataclass(frozen=True, slots=True)
class HardwareFingerprint:
    """Portable hardware facts used in runtime cache identity and matching."""

    accelerator: str = "cpu"
    compute_capability: str | None = None
    driver: str | None = None
    runtime: str | None = None
    memory_bytes: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "accelerator", _token(self.accelerator, field_name="accelerator"))
        for field_name in ("compute_capability", "driver", "runtime"):
            value = getattr(self, field_name)
            if value is not None:
                _token(value, field_name=field_name)
        if self.memory_bytes is not None and (
            type(self.memory_bytes) is not int or self.memory_bytes < 0
        ):
            raise RuntimeCompatibilityError("memory_bytes must be a non-negative integer or None")

    def to_dict(self) -> dict[str, object]:
        return {
            "accelerator": self.accelerator,
            "compute_capability": self.compute_capability,
            "driver": self.driver,
            "memory_bytes": self.memory_bytes,
            "runtime": self.runtime,
        }


@dataclass(frozen=True, slots=True)
class CompatibilityBridge:
    """An approved, model-neutral import/API compatibility bridge."""

    bridge_id: str
    target_import: str
    description: str
    transformers: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "bridge_id",
            canonical_identifier(self.bridge_id, field_name="bridge_id"),
        )
        object.__setattr__(
            self,
            "target_import",
            _import_token(self.target_import, field_name="target_import"),
        )
        _token(self.description, field_name="description")
        if self.transformers is not None:
            _token(self.transformers, field_name="transformers bridge requirement")


# This bridge is intentionally keyed by a missing API symbol, not by a model
# name.  A worker may apply it only after the isolated import probe confirms
# that the symbol is absent and that the scoped import is otherwise safe.
TORCH_FX_AVAILABLE_BRIDGE = CompatibilityBridge(
    bridge_id="transformers/import_utils/is_torch_fx_available",
    target_import="transformers.utils.import_utils.is_torch_fx_available",
    description="Provide the legacy torch.fx availability predicate to remote code.",
    transformers=">=5",
)
APPROVED_COMPATIBILITY_BRIDGES: tuple[CompatibilityBridge, ...] = (
    TORCH_FX_AVAILABLE_BRIDGE,
)


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    """An available worker environment and its declared capabilities."""

    profile_id: str
    python_version: str
    transformers_version: str
    packages: tuple[str, ...] = ()
    imports: tuple[str, ...] = ()
    bridge_ids: tuple[str, ...] = ()
    accelerators: tuple[str, ...] = ("cpu",)
    dtypes: tuple[str, ...] = ("float32", "float16", "bfloat16", "int8", "int4")
    allows_remote_code: bool = True
    priority: int = 100

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "profile_id",
            canonical_identifier(self.profile_id, field_name="profile_id"),
        )
        _version_tuple(_token(self.python_version, field_name="python_version"))
        _version_tuple(_token(self.transformers_version, field_name="transformers_version"))
        if type(self.allows_remote_code) is not bool:
            raise RuntimeCompatibilityError("allows_remote_code must be an exact bool")
        if type(self.priority) is not int or isinstance(self.priority, bool):
            raise RuntimeCompatibilityError("priority must be an integer")
        object.__setattr__(
            self,
            "packages",
            tuple(
                sorted(
                    {
                        _normalized_package_entry(
                            _package_token(item, field_name="packages[]")
                        )
                        for item in self.packages
                    }
                )
            ),
        )
        object.__setattr__(
            self,
            "imports",
            _unique_tokens(self.imports, field_name="imports", validator=_import_token),
        )
        object.__setattr__(
            self, "bridge_ids", _unique_tokens(self.bridge_ids, field_name="bridge_ids")
        )
        object.__setattr__(
            self,
            "accelerators",
            _unique_tokens(self.accelerators, field_name="accelerators"),
        )
        object.__setattr__(self, "dtypes", _unique_tokens(self.dtypes, field_name="dtypes"))


@dataclass(frozen=True, slots=True)
class RuntimeProfileCacheKey:
    """Stable cache identity bound to model commit, runtime, and hardware."""

    model_id: str
    immutable_revision: str
    runtime_profile_id: str
    hardware: HardwareFingerprint
    schema_version: ClassVar[str] = COMPATIBILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_id",
            validate_stable_identifier(self.model_id, field_name="model_id"),
        )
        object.__setattr__(self, "immutable_revision", _immutable_revision(self.immutable_revision))
        object.__setattr__(
            self,
            "runtime_profile_id",
            canonical_identifier(self.runtime_profile_id, field_name="runtime_profile_id"),
        )
        if not isinstance(self.hardware, HardwareFingerprint):
            raise RuntimeCompatibilityError("hardware must be HardwareFingerprint")

    def to_dict(self) -> dict[str, object]:
        return {
            "hardware": self.hardware.to_dict(),
            "immutable_revision": self.immutable_revision,
            "model_id": self.model_id,
            "runtime_profile_id": self.runtime_profile_id,
            "schema_version": self.schema_version,
        }

    @property
    def digest(self) -> str:
        return f"runtime:{stable_digest(self.to_dict())}"


@dataclass(frozen=True, slots=True)
class RuntimeProfileSelection:
    """Explainable result of deterministic runtime profile resolution."""

    status: RuntimeSelectionStatus
    metadata: ModelRuntimeMetadata
    profile: RuntimeProfile | None
    bridges: tuple[CompatibilityBridge, ...] = ()
    reasons: tuple[str, ...] = ()
    missing_packages: tuple[str, ...] = ()
    missing_imports: tuple[str, ...] = ()
    cache_key: RuntimeProfileCacheKey | None = None

    @property
    def ready(self) -> bool:
        return self.status is RuntimeSelectionStatus.READY

    def to_dict(self) -> dict[str, object]:
        return {
            "cache_key": self.cache_key.to_dict() if self.cache_key is not None else None,
            "cache_digest": self.cache_key.digest if self.cache_key is not None else None,
            "missing_imports": list(self.missing_imports),
            "missing_packages": list(self.missing_packages),
            "profile_id": self.profile.profile_id if self.profile is not None else None,
            "reasons": list(self.reasons),
            "status": self.status.value,
            "bridges": [bridge.bridge_id for bridge in self.bridges],
        }


def _bridge_for_import(
    target: str,
    *,
    profile: RuntimeProfile,
    requirements: RuntimeRequirements,
    bridges: Mapping[str, CompatibilityBridge],
) -> CompatibilityBridge | None:
    candidate = next(
        (
            bridge
            for bridge in bridges.values()
            if bridge.target_import == target and bridge.bridge_id in profile.bridge_ids
        ),
        None,
    )
    if candidate is None:
        return None
    if candidate.transformers is not None and not version_satisfies(
        profile.transformers_version, candidate.transformers
    ):
        return None
    # A bridge is only selected for an explicitly declared missing import.
    # ``requirements`` is kept as an argument to make that invariant obvious
    # at the call site and to leave room for future package-scoped bridges.
    if target not in requirements.imports:
        return None
    return candidate


def resolve_runtime_profile(
    metadata: ModelRuntimeMetadata,
    profiles: Iterable[RuntimeProfile],
    *,
    hardware: HardwareFingerprint | None = None,
    bridges: Iterable[CompatibilityBridge] = APPROVED_COMPATIBILITY_BRIDGES,
) -> RuntimeProfileSelection:
    """Choose the best compatible profile without executing model code.

    Candidates are ordered by fewest compatibility bridges, explicit profile
    priority, and profile ID.  This makes selection reproducible even when the
    caller supplies profiles in a different order.  Incompatible results
    retain bounded reasons and never produce a cache key.
    """

    if not isinstance(metadata, ModelRuntimeMetadata):
        raise RuntimeCompatibilityError("metadata must be ModelRuntimeMetadata")
    stable_hardware = hardware if hardware is not None else HardwareFingerprint()
    if not isinstance(stable_hardware, HardwareFingerprint):
        raise RuntimeCompatibilityError("hardware must be HardwareFingerprint")
    profile_list = tuple(profiles)
    if any(not isinstance(profile, RuntimeProfile) for profile in profile_list):
        raise RuntimeCompatibilityError("profiles must contain RuntimeProfile values")
    if len({profile.profile_id for profile in profile_list}) != len(profile_list):
        raise RuntimeCompatibilityError("duplicate profile_id values are not allowed")
    bridge_map: dict[str, CompatibilityBridge] = {}
    for bridge in bridges:
        if not isinstance(bridge, CompatibilityBridge):
            raise RuntimeCompatibilityError("bridges must contain CompatibilityBridge values")
        if bridge.bridge_id in bridge_map:
            raise RuntimeCompatibilityError("bridges must have unique bridge_id values")
        bridge_map[bridge.bridge_id] = bridge

    candidates: list[tuple[int, int, str, RuntimeProfile, tuple[CompatibilityBridge, ...]]] = []
    rejected: list[str] = []
    for profile in profile_list:
        requirements = metadata.requirements
        profile_reasons: list[str] = []
        if requirements.remote_code and not profile.allows_remote_code:
            profile_reasons.append("remote code is not enabled")
        if requirements.python is not None and not version_satisfies(
            profile.python_version, requirements.python
        ):
            profile_reasons.append("Python version is outside the declared requirement")
        installed_packages: dict[str, str | None] = {}
        for installed in profile.packages:
            name, constraint = _package_constraint(installed)
            # A profile describes one installed version.  A bare package token
            # remains useful for presence-only requirements but cannot satisfy
            # a versioned model requirement.
            installed_packages[name] = (
                constraint.removeprefix("==")
                if constraint and constraint.startswith("==")
                else None
            )
        missing_packages: list[str] = []
        for requirement in requirements.packages:
            name, constraint = _package_constraint(requirement)
            installed_version = installed_packages.get(name)
            if name not in installed_packages:
                missing_packages.append(name)
            elif constraint is not None:
                if installed_version is None or not version_satisfies(
                    installed_version, constraint
                ):
                    missing_packages.append(_normalized_package_entry(requirement))
        if missing_packages:
            profile_reasons.append(
                "required package is unavailable: " + ", ".join(missing_packages)
            )
        if requirements.cuda and stable_hardware.accelerator.casefold() not in {
            accelerator.casefold() for accelerator in profile.accelerators
        }:
            profile_reasons.append("required accelerator is unavailable")
        if requirements.dtype is not None and requirements.dtype not in profile.dtypes:
            profile_reasons.append("requested dtype is unavailable")
        missing_imports = sorted(set(requirements.imports).difference(profile.imports))
        selected_bridges = tuple(
            sorted(
                (
                    bridge
                    for target in missing_imports
                    if (bridge := _bridge_for_import(
                        target, profile=profile, requirements=requirements, bridges=bridge_map
                    ))
                    is not None
                ),
                key=lambda bridge: bridge.bridge_id,
            )
        )
        bridged_imports = {bridge.target_import for bridge in selected_bridges}
        unresolved_imports = sorted(set(missing_imports).difference(bridged_imports))
        # A declared Transformers version is evidence, not an immutable model
        # family rule.  An approved bridge may explicitly make a newer runtime
        # viable when the isolated import probe identifies the bridge target.
        transformers_compatible = requirements.transformers is None or version_satisfies(
            profile.transformers_version, requirements.transformers
        )
        if not transformers_compatible and not selected_bridges:
            profile_reasons.append("Transformers version is outside the declared requirement")
        if unresolved_imports:
            profile_reasons.append(
                "required import is unavailable: " + ", ".join(unresolved_imports)
            )
        if profile_reasons:
            rejected.append(f"{profile.profile_id}: " + "; ".join(profile_reasons))
            continue
        candidates.append(
            (len(selected_bridges), profile.priority, profile.profile_id, profile, selected_bridges)
        )

    if not candidates:
        return RuntimeProfileSelection(
            status=RuntimeSelectionStatus.INCOMPATIBLE,
            metadata=metadata,
            profile=None,
            reasons=tuple(sorted(rejected)) or ("no runtime profiles were provided",),
            missing_packages=tuple(
                sorted(
                    {
                        package_name(requirement)
                        for requirement in metadata.requirements.packages
                    }
                )
            ),
            missing_imports=tuple(metadata.requirements.imports),
        )

    _, _, _, selected, selected_bridges = min(candidates)
    key = RuntimeProfileCacheKey(
        model_id=metadata.model_id,
        immutable_revision=metadata.immutable_revision,
        runtime_profile_id=selected.profile_id,
        hardware=stable_hardware,
    )
    reasons = (
        "selected by bridge count, profile priority, and profile ID",
    )
    if selected_bridges:
        reasons += ("approved compatibility bridge required: " + ", ".join(
            bridge.bridge_id for bridge in selected_bridges
        ),)
    return RuntimeProfileSelection(
        status=RuntimeSelectionStatus.READY,
        metadata=metadata,
        profile=selected,
        bridges=selected_bridges,
        reasons=reasons,
        cache_key=key,
    )


# Short aliases keep callers expressive without hiding the canonical API.
preflight_runtime = resolve_runtime_profile
select_runtime_profile = resolve_runtime_profile


__all__ = [
    "APPROVED_COMPATIBILITY_BRIDGES",
    "COMPATIBILITY_SCHEMA_VERSION",
    "CompatibilityBridge",
    "HardwareFingerprint",
    "ModelRuntimeMetadata",
    "RuntimeCompatibilityError",
    "RuntimeProfile",
    "RuntimeProfileCacheKey",
    "RuntimeProfileSelection",
    "RuntimeRequirements",
    "RuntimeSelectionStatus",
    "TORCH_FX_AVAILABLE_BRIDGE",
    "package_name",
    "preflight_runtime",
    "resolve_runtime_profile",
    "select_runtime_profile",
    "version_satisfies",
]
