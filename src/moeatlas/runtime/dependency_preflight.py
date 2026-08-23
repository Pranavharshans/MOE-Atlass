"""Model-neutral dependency checks performed before model weights are loaded.

The loader may need optional packages (for example ``tiktoken``) or import
symbols supplied by trusted remote code.  This module keeps that information
as data and compares it with a local runtime inventory.  It deliberately does
not contact the Hub, import model code, or install packages.  A later worker
may use the result to select an isolated environment or present an actionable
failure before allocating model weights.

``RuntimeRequirements`` and ``resolve_runtime_profile`` remain the canonical
compatibility contracts.  The helpers here add a small local-observation seam:
the profile resolver answers *which declared worker profile could satisfy the
requirements*, while this module answers *whether the currently observed
inventory actually contains them*.
"""

from __future__ import annotations

import importlib.metadata
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .compatibility import (
    APPROVED_COMPATIBILITY_BRIDGES,
    CompatibilityBridge,
    HardwareFingerprint,
    ModelRuntimeMetadata,
    RuntimeCompatibilityError,
    RuntimeProfile,
    RuntimeProfileSelection,
    RuntimeRequirements,
    resolve_runtime_profile,
    version_satisfies,
)

DEPENDENCY_PREFLIGHT_SCHEMA_VERSION = "1.0"
_PACKAGE_REQUIREMENT = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9_.-]*)"
    r"\s*(?P<constraint>(?:==|!=|>=|<=|>|<|~=)\s*"
    r"[0-9]+(?:\.[0-9]+)*(?:[-+][A-Za-z0-9.-]+)?)?$"
)
_IMPORT_PATH = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*(?::[A-Za-z_][A-Za-z0-9_.]*)?$")


class DependencyPreflightError(ValueError):
    """Raised when a dependency inventory or requirement is malformed."""


class DependencyStatus(str, Enum):
    """Bounded result of local dependency inspection."""

    READY = "ready"
    INCOMPATIBLE = "incompatible"

    def __str__(self) -> str:
        return self.value


class DependencyRequirementState(str, Enum):
    """State for one package or import requirement."""

    AVAILABLE = "available"
    MISSING = "missing"
    VERSION_MISMATCH = "version_mismatch"

    def __str__(self) -> str:
        return self.value


def _text(value: object, *, field_name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise DependencyPreflightError(f"{field_name} must be a non-empty trimmed string")
    if any(ord(char) < 32 for char in value):
        raise DependencyPreflightError(f"{field_name} must not contain control characters")
    return value


def _package_requirement(value: object, *, field_name: str) -> str:
    token = _text(value, field_name=field_name)
    if _PACKAGE_REQUIREMENT.fullmatch(token) is None:
        raise DependencyPreflightError(
            f"{field_name} must be a package name with an optional version constraint"
        )
    return token


def _import_requirement(value: object, *, field_name: str) -> str:
    token = _text(value, field_name=field_name)
    if _IMPORT_PATH.fullmatch(token) is None:
        raise DependencyPreflightError(
            f"{field_name} must use module or module:attribute import notation"
        )
    return token


def _unique_sorted(values: object, *, field_name: str, validator: Any) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise DependencyPreflightError(f"{field_name} must be a sequence of strings")
    normalized = tuple(validator(value, field_name=f"{field_name}[]") for value in values)
    if len(set(normalized)) != len(normalized):
        raise DependencyPreflightError(f"{field_name} must not contain duplicate entries")
    return tuple(sorted(normalized))


def _split_package_requirement(requirement: str) -> tuple[str, str | None]:
    match = _PACKAGE_REQUIREMENT.fullmatch(requirement)
    if match is None:  # pragma: no cover - validated by callers
        raise DependencyPreflightError("invalid package requirement")
    name = match.group("name").casefold().replace("_", "-")
    constraint = match.group("constraint")
    return name, constraint.strip() if constraint else None


def _normalized_package_requirement(requirement: str) -> str:
    name, constraint = _split_package_requirement(requirement)
    return name + (constraint.replace(" ", "") if constraint else "")


def _module_name(import_requirement: str) -> str:
    return import_requirement.split(":", 1)[0]


def _installed_package_versions(packages: Sequence[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for token in packages:
        name, constraint = _split_package_requirement(token)
        # Inventories describe installed distributions.  A constraint other
        # than == is retained as presence evidence but does not assert a
        # concrete installed version.
        installed = (
            constraint.removeprefix("==")
            if constraint and constraint.startswith("==")
            else None
        )
        versions[name] = installed
    return versions


@dataclass(frozen=True, slots=True)
class RuntimeDependencyInventory:
    """A bounded observation of packages/imports available to a worker.

    Package entries are distribution names or ``name==version`` values.  The
    import entries are module paths or ``module:attribute`` symbols that have
    already been validated by the worker.  No entry is imported by this class.
    """

    packages: tuple[str, ...] = ()
    imports: tuple[str, ...] = ()
    python_version: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.packages, str) or not isinstance(self.packages, Sequence):
            raise DependencyPreflightError("packages must be a sequence of strings")
        normalized_packages = tuple(
            _normalized_package_requirement(
                _package_requirement(item, field_name="packages[]")
            )
            for item in self.packages
        )
        if len(set(normalized_packages)) != len(normalized_packages):
            raise DependencyPreflightError("packages must not contain duplicate entries")
        object.__setattr__(
            self,
            "packages",
            tuple(sorted(normalized_packages)),
        )
        object.__setattr__(
            self,
            "imports",
            _unique_sorted(self.imports, field_name="imports", validator=_import_requirement),
        )
        if self.python_version is not None:
            _text(self.python_version, field_name="python_version")

    @classmethod
    def from_profile(cls, profile: RuntimeProfile) -> RuntimeDependencyInventory:
        """Convert a declarative worker profile into an inventory."""

        if not isinstance(profile, RuntimeProfile):
            raise DependencyPreflightError("profile must be RuntimeProfile")
        return cls(
            packages=profile.packages,
            imports=profile.imports,
            python_version=profile.python_version,
        )

    @classmethod
    def from_environment(cls) -> RuntimeDependencyInventory:
        """Observe installed distributions without importing model packages.

        ``importlib.metadata`` reads local package metadata only.  The result
        is intentionally presence-oriented: versions that cannot be expressed
        in the bounded compatibility grammar remain as bare package entries,
        so discovery can still report package presence without making a false
        version claim.
        """

        packages: set[str] = set()
        for distribution in importlib.metadata.distributions():
            name = distribution.metadata.get("Name")
            version = distribution.version
            if not isinstance(name, str) or not name.strip():
                continue
            try:
                normalized_name = _normalized_package_requirement(name)
            except DependencyPreflightError:
                continue
            if isinstance(version, str) and _PACKAGE_REQUIREMENT.fullmatch(
                f"{name}=={version}"
            ):
                packages.add(_normalized_package_requirement(f"{name}=={version}"))
            else:
                packages.add(normalized_name)

        imports: set[str] = set()
        try:
            package_map = importlib.metadata.packages_distributions()
        except Exception:
            package_map = {}
        for module in package_map:
            if isinstance(module, str) and _IMPORT_PATH.fullmatch(module):
                imports.add(module)
        return cls(
            packages=tuple(sorted(packages)),
            imports=tuple(sorted(imports)),
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "imports": list(self.imports),
            "packages": list(self.packages),
            "python_version": self.python_version,
            "schema_version": DEPENDENCY_PREFLIGHT_SCHEMA_VERSION,
        }


@dataclass(frozen=True, slots=True)
class DependencyCheck:
    """Per-inventory dependency evidence for one model requirement set."""

    status: DependencyStatus
    requirements: RuntimeRequirements
    inventory: RuntimeDependencyInventory
    package_states: tuple[tuple[str, DependencyRequirementState], ...] = ()
    import_states: tuple[tuple[str, DependencyRequirementState], ...] = ()
    missing_packages: tuple[str, ...] = ()
    missing_imports: tuple[str, ...] = ()
    version_mismatches: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return self.status is DependencyStatus.READY

    def to_dict(self) -> dict[str, object]:
        return {
            "import_states": {
                name: state.value for name, state in self.import_states
            },
            "missing_imports": list(self.missing_imports),
            "missing_packages": list(self.missing_packages),
            "package_states": {
                name: state.value for name, state in self.package_states
            },
            "reasons": list(self.reasons),
            "status": self.status.value,
            "version_mismatches": list(self.version_mismatches),
        }


@dataclass(frozen=True, slots=True)
class RuntimeDependencyPreflight:
    """Combined local dependency evidence and runtime-profile selection."""

    metadata: ModelRuntimeMetadata
    selection: RuntimeProfileSelection
    dependency_check: DependencyCheck
    inventory: RuntimeDependencyInventory

    @property
    def ready(self) -> bool:
        return self.selection.ready and self.dependency_check.ready

    @property
    def status(self) -> DependencyStatus:
        return DependencyStatus.READY if self.ready else DependencyStatus.INCOMPATIBLE

    def to_dict(self) -> dict[str, object]:
        return {
            "dependency_check": self.dependency_check.to_dict(),
            "inventory": self.inventory.to_dict(),
            "metadata": {
                "immutable_revision": self.metadata.immutable_revision,
                "model_id": self.metadata.model_id,
            },
            "runtime_selection": self.selection.to_dict(),
            "status": self.status.value,
            "schema_version": DEPENDENCY_PREFLIGHT_SCHEMA_VERSION,
        }


def check_declared_dependencies(
    requirements: RuntimeRequirements,
    inventory: RuntimeDependencyInventory,
    *,
    satisfied_imports: Iterable[str] = (),
) -> DependencyCheck:
    """Check package/import declarations against a local inventory.

    This function is side-effect free.  Import requirements are matched using
    exact module/symbol evidence or an exact module entry; the function never
    imports a module to test it.
    """

    if not isinstance(requirements, RuntimeRequirements):
        raise DependencyPreflightError("requirements must be RuntimeRequirements")
    if not isinstance(inventory, RuntimeDependencyInventory):
        raise DependencyPreflightError("inventory must be RuntimeDependencyInventory")
    if isinstance(satisfied_imports, str):
        raise DependencyPreflightError("satisfied_imports must be an iterable of strings")
    try:
        virtual_imports = {
            _import_requirement(item, field_name="satisfied_imports[]")
            for item in satisfied_imports
        }
    except TypeError as exc:
        raise DependencyPreflightError(
            "satisfied_imports must be an iterable of strings"
        ) from exc

    installed = _installed_package_versions(inventory.packages)
    package_states: list[tuple[str, DependencyRequirementState]] = []
    missing_packages: list[str] = []
    version_mismatches: list[str] = []
    for requirement in requirements.packages:
        normalized = _normalized_package_requirement(requirement)
        name, constraint = _split_package_requirement(normalized)
        installed_version = installed.get(name)
        if installed_version is None and name not in installed:
            package_states.append((normalized, DependencyRequirementState.MISSING))
            missing_packages.append(normalized)
            continue
        if constraint is not None:
            if installed_version is None:
                state = DependencyRequirementState.VERSION_MISMATCH
            else:
                try:
                    state = (
                        DependencyRequirementState.AVAILABLE
                        if version_satisfies(installed_version, constraint)
                        else DependencyRequirementState.VERSION_MISMATCH
                    )
                except RuntimeCompatibilityError:
                    state = DependencyRequirementState.VERSION_MISMATCH
            package_states.append((normalized, state))
            if state is DependencyRequirementState.VERSION_MISMATCH:
                version_mismatches.append(normalized)
        else:
            package_states.append((normalized, DependencyRequirementState.AVAILABLE))

    available_imports = set(inventory.imports)
    import_states: list[tuple[str, DependencyRequirementState]] = []
    missing_imports: list[str] = []
    for requirement in requirements.imports:
        module = _module_name(requirement)
        state = (
            DependencyRequirementState.AVAILABLE
            if (
                requirement in available_imports
                or module in available_imports
                or requirement in virtual_imports
                or module in virtual_imports
            )
            else DependencyRequirementState.MISSING
        )
        import_states.append((requirement, state))
        if state is DependencyRequirementState.MISSING:
            missing_imports.append(requirement)

    reasons: list[str] = []
    if missing_packages:
        reasons.append("required package is unavailable: " + ", ".join(missing_packages))
    if version_mismatches:
        reasons.append("required package version is incompatible: " + ", ".join(version_mismatches))
    if missing_imports:
        reasons.append("required import is unavailable: " + ", ".join(missing_imports))
    return DependencyCheck(
        status=DependencyStatus.INCOMPATIBLE if reasons else DependencyStatus.READY,
        requirements=requirements,
        inventory=inventory,
        package_states=tuple(package_states),
        import_states=tuple(import_states),
        missing_packages=tuple(missing_packages),
        missing_imports=tuple(missing_imports),
        version_mismatches=tuple(version_mismatches),
        reasons=tuple(reasons),
    )


def preflight_runtime_dependencies(
    metadata: ModelRuntimeMetadata,
    profiles: Iterable[RuntimeProfile],
    *,
    inventory: RuntimeDependencyInventory | None = None,
    hardware: HardwareFingerprint | None = None,
    bridges: Iterable[CompatibilityBridge] = APPROVED_COMPATIBILITY_BRIDGES,
) -> RuntimeDependencyPreflight:
    """Compose local dependency checks with deterministic profile selection.

    When no inventory is provided, the selected declarative profile is used as
    the inventory.  Callers that are about to load a model should pass an
    inventory observed from the actual worker environment to detect stale
    profile declarations before weights are allocated.
    """

    if not isinstance(metadata, ModelRuntimeMetadata):
        raise DependencyPreflightError("metadata must be ModelRuntimeMetadata")
    profile_list = tuple(profiles)
    selection = resolve_runtime_profile(
        metadata,
        profile_list,
        hardware=hardware,
        bridges=bridges,
    )
    if inventory is None:
        if selection.profile is not None:
            inventory = RuntimeDependencyInventory.from_profile(selection.profile)
        else:
            inventory = RuntimeDependencyInventory()
    dependency_check = check_declared_dependencies(metadata.requirements, inventory)
    # A bridge can satisfy a declared import at runtime even when the local
    # inventory intentionally excludes the temporary symbol.  Preserve the
    # original requirements and inventory while recording that approved
    # virtual evidence in the per-import state.
    if selection.bridges:
        dependency_check = check_declared_dependencies(
            metadata.requirements,
            inventory,
            satisfied_imports=(bridge.target_import for bridge in selection.bridges),
        )
    return RuntimeDependencyPreflight(
        metadata=metadata,
        selection=selection,
        dependency_check=dependency_check,
        inventory=inventory,
    )


__all__ = [
    "DEPENDENCY_PREFLIGHT_SCHEMA_VERSION",
    "DependencyCheck",
    "DependencyPreflightError",
    "DependencyRequirementState",
    "DependencyStatus",
    "RuntimeDependencyInventory",
    "RuntimeDependencyPreflight",
    "check_declared_dependencies",
    "preflight_runtime_dependencies",
]
