"""Model-free tests for dependency evidence collected before weight loading."""

from __future__ import annotations

import pytest

from moeatlas.runtime import (
    TORCH_FX_AVAILABLE_BRIDGE,
    DependencyPreflightError,
    DependencyRequirementState,
    DependencyStatus,
    HardwareFingerprint,
    ModelRuntimeMetadata,
    RuntimeDependencyInventory,
    RuntimeProfile,
    RuntimeRequirements,
    check_declared_dependencies,
    preflight_runtime_dependencies,
)

REVISION = "a" * 40
TORCH_FX_IMPORT = "transformers.utils.import_utils.is_torch_fx_available"


def _profile(*, bridge: bool = True, imports: tuple[str, ...] = ()) -> RuntimeProfile:
    return RuntimeProfile(
        profile_id="transformers-current",
        python_version="3.12.4",
        transformers_version="5.15.1",
        packages=("torch==2.5.1", "transformers==5.15.1", "tiktoken==0.9.0"),
        imports=imports,
        bridge_ids=(TORCH_FX_AVAILABLE_BRIDGE.bridge_id,) if bridge else (),
        accelerators=("cpu", "cuda"),
    )


def test_inventory_normalizes_packages_without_importing_them() -> None:
    inventory = RuntimeDependencyInventory(
        packages=("TIKTOKEN==0.9.0", "torch"),
        imports=("tiktoken", "torch"),
        python_version="3.12.4",
    )
    assert inventory.packages == ("tiktoken==0.9.0", "torch")
    assert inventory.imports == ("tiktoken", "torch")
    assert inventory.to_dict()["schema_version"] == "1.0"


def test_missing_optional_package_is_reported_before_execution() -> None:
    requirements = RuntimeRequirements(
        packages=("tiktoken",),
        imports=("tiktoken",),
    )
    result = check_declared_dependencies(
        requirements,
        RuntimeDependencyInventory(packages=("torch==2.5.1",), imports=("torch",)),
    )
    assert result.status is DependencyStatus.INCOMPATIBLE
    assert result.missing_packages == ("tiktoken",)
    assert result.missing_imports == ("tiktoken",)
    assert result.package_states == (("tiktoken", DependencyRequirementState.MISSING),)
    assert any("tiktoken" in reason for reason in result.reasons)


def test_versioned_package_requirement_does_not_accept_unknown_version() -> None:
    result = check_declared_dependencies(
        RuntimeRequirements(packages=("tiktoken>=0.9",)),
        RuntimeDependencyInventory(packages=("tiktoken",)),
    )
    assert result.status is DependencyStatus.INCOMPATIBLE
    assert result.version_mismatches == ("tiktoken>=0.9",)
    assert result.missing_packages == ()


def test_versioned_package_requirement_checks_installed_version() -> None:
    available = check_declared_dependencies(
        RuntimeRequirements(packages=("tiktoken>=0.9",)),
        RuntimeDependencyInventory(packages=("tiktoken==0.9.0",)),
    )
    rejected = check_declared_dependencies(
        RuntimeRequirements(packages=("tiktoken>=0.9",)),
        RuntimeDependencyInventory(packages=("tiktoken==0.8.0",)),
    )
    assert available.ready
    assert rejected.version_mismatches == ("tiktoken>=0.9",)


def test_bridge_satisfies_only_the_declared_missing_import() -> None:
    metadata = ModelRuntimeMetadata(
        model_id="inclusionAI/example",
        immutable_revision=REVISION,
        requirements=RuntimeRequirements(
            transformers="4.45.2",
            imports=(TORCH_FX_IMPORT,),
            remote_code=True,
        ),
    )
    preflight = preflight_runtime_dependencies(
        metadata,
        (_profile(),),
        inventory=RuntimeDependencyInventory(
            packages=("torch==2.5.1", "transformers==5.15.1"),
            imports=(),
        ),
        hardware=HardwareFingerprint(accelerator="cuda"),
    )
    assert preflight.ready
    assert preflight.selection.bridges == (TORCH_FX_AVAILABLE_BRIDGE,)
    assert preflight.dependency_check.requirements == metadata.requirements
    assert preflight.dependency_check.missing_imports == ()


def test_preflight_rejects_stale_profile_inventory() -> None:
    metadata = ModelRuntimeMetadata(
        model_id="acme/moe",
        immutable_revision=REVISION,
        requirements=RuntimeRequirements(packages=("tiktoken",)),
    )
    preflight = preflight_runtime_dependencies(
        metadata,
        (_profile(),),
        inventory=RuntimeDependencyInventory(packages=("torch==2.5.1",)),
    )
    assert not preflight.ready
    assert preflight.status is DependencyStatus.INCOMPATIBLE
    assert preflight.selection.ready
    assert preflight.dependency_check.missing_packages == ("tiktoken",)


def test_inventory_from_profile_is_declared_only() -> None:
    profile = _profile(imports=("tiktoken",))
    inventory = RuntimeDependencyInventory.from_profile(profile)
    assert inventory.python_version == profile.python_version
    assert "tiktoken==0.9.0" in inventory.packages
    assert inventory.imports == ("tiktoken",)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (("tiktoken", "tiktoken"), "duplicate"),
        (("bad package",), "package"),
    ],
)
def test_inventory_validation_is_strict(value: tuple[str, ...], message: str) -> None:
    with pytest.raises(DependencyPreflightError, match=message):
        RuntimeDependencyInventory(packages=value)


def test_satisfied_imports_is_validated_and_cannot_be_a_string() -> None:
    requirements = RuntimeRequirements(imports=("tiktoken",))
    inventory = RuntimeDependencyInventory()
    with pytest.raises(DependencyPreflightError, match="iterable"):
        check_declared_dependencies(requirements, inventory, satisfied_imports="tiktoken")


def test_preflight_without_profile_has_bounded_incompatible_result() -> None:
    metadata = ModelRuntimeMetadata("acme/moe", REVISION)
    result = preflight_runtime_dependencies(metadata, ())
    assert not result.ready
    assert result.selection.profile is None
    assert result.to_dict()["status"] == "incompatible"
