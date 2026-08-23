"""Model-free tests for declarative runtime compatibility preflight."""

from __future__ import annotations

import pytest

from moeatlas.runtime import (
    APPROVED_COMPATIBILITY_BRIDGES,
    TORCH_FX_AVAILABLE_BRIDGE,
    CompatibilityBridge,
    HardwareFingerprint,
    ModelRuntimeMetadata,
    RuntimeCompatibilityError,
    RuntimeProfile,
    RuntimeRequirements,
    RuntimeSelectionStatus,
    resolve_runtime_profile,
    version_satisfies,
)

REVISION = "a" * 40
TORCH_FX_IMPORT = "transformers.utils.import_utils.is_torch_fx_available"


def _current(*, bridge: bool = True, imports: tuple[str, ...] = ()) -> RuntimeProfile:
    return RuntimeProfile(
        profile_id="transformers-current",
        python_version="3.12.4",
        transformers_version="5.15.1",
        packages=("torch", "transformers", "accelerate", "einops==0.8.2", "fla-core==0.5.2"),
        imports=imports,
        bridge_ids=(TORCH_FX_AVAILABLE_BRIDGE.bridge_id,) if bridge else (),
        accelerators=("cpu", "cuda"),
        priority=20,
    )


def _legacy() -> RuntimeProfile:
    return RuntimeProfile(
        profile_id="transformers-4x-remote",
        python_version="3.11.9",
        transformers_version="4.45.2",
        packages=("torch", "transformers", "einops==0.8.2", "fla-core==0.5.2"),
        imports=(TORCH_FX_IMPORT,),
        accelerators=("cpu", "cuda"),
        priority=40,
    )


def test_version_subset_is_deterministic_and_bounded() -> None:
    assert version_satisfies("5.15.1", ">=5,<6")
    assert version_satisfies("4.45.2", "~=4.45")
    assert not version_satisfies("5.15.1", ">=4.50,<5")
    with pytest.raises(RuntimeCompatibilityError, match="unsupported version requirement"):
        version_satisfies("5.15.1", "^5")


def test_config_constructor_reads_declarative_auto_map_without_importing_it() -> None:
    metadata = ModelRuntimeMetadata.from_config(
        "inclusionAI/example",
        REVISION,
        {
            "model_type": "bailing_hybrid",
            "architectures": ["BailingMoeForCausalLM"],
            "auto_map": {"AutoConfig": "configuration_example.ExampleConfig"},
            "transformers_version": "4.45.2",
        },
        imports=(TORCH_FX_IMPORT,),
        packages=("einops>=0.8",),
    )
    assert metadata.requirements.remote_code is True
    assert metadata.requirements.transformers == "4.45.2"
    assert metadata.architectures == ("BailingMoeForCausalLM",)
    assert metadata.auto_map == ("AutoConfig=configuration_example.ExampleConfig",)
    assert metadata.immutable_revision == REVISION


def test_legacy_remote_code_can_use_approved_bridge_on_current_runtime() -> None:
    metadata = ModelRuntimeMetadata(
        model_id="inclusionAI/example",
        immutable_revision=REVISION,
        requirements=RuntimeRequirements(
            transformers="4.45.2",
            packages=("einops>=0.8",),
            imports=(TORCH_FX_IMPORT,),
            remote_code=True,
            cuda=True,
        ),
    )
    selection = resolve_runtime_profile(
        metadata,
        (_current(),),
        hardware=HardwareFingerprint(accelerator="cuda", compute_capability="10.0"),
    )
    assert selection.status is RuntimeSelectionStatus.READY
    assert selection.profile is not None
    assert selection.profile.profile_id == "transformers-current"
    assert selection.bridges == (TORCH_FX_AVAILABLE_BRIDGE,)
    assert selection.cache_key is not None
    assert selection.cache_key.digest.startswith("runtime:")


def test_selection_is_order_independent_and_cache_key_tracks_hardware_revision() -> None:
    metadata = ModelRuntimeMetadata(
        model_id="acme/moe",
        immutable_revision=REVISION,
        requirements=RuntimeRequirements(packages=("torch",)),
    )
    hardware = HardwareFingerprint(accelerator="cuda", memory_bytes=80_000_000_000)
    first = resolve_runtime_profile(metadata, (_current(), _legacy()), hardware=hardware)
    second = resolve_runtime_profile(metadata, (_legacy(), _current()), hardware=hardware)
    assert first.to_dict() == second.to_dict()
    changed_hardware = resolve_runtime_profile(
        metadata,
        (_current(), _legacy()),
        hardware=HardwareFingerprint(accelerator="cuda", memory_bytes=40_000_000_000),
    )
    assert first.cache_key is not None and changed_hardware.cache_key is not None
    assert first.cache_key.digest != changed_hardware.cache_key.digest
    changed_revision = resolve_runtime_profile(
        metadata.__class__(
            model_id=metadata.model_id,
            immutable_revision="b" * 40,
            requirements=metadata.requirements,
        ),
        (_current(),),
        hardware=hardware,
    )
    assert changed_revision.cache_key is not None
    assert first.cache_key.digest != changed_revision.cache_key.digest


def test_incompatible_selection_has_exact_bounded_reasons_and_no_cache_key() -> None:
    metadata = ModelRuntimeMetadata(
        model_id="acme/moe",
        immutable_revision=REVISION,
        requirements=RuntimeRequirements(
            packages=("missing-package",),
            imports=("missing.module.Symbol",),
            remote_code=True,
            cuda=True,
        ),
    )
    selection = resolve_runtime_profile(
        metadata,
        (_current(bridge=False),),
        hardware=HardwareFingerprint(accelerator="cpu"),
    )
    assert selection.status is RuntimeSelectionStatus.INCOMPATIBLE
    assert selection.profile is None
    assert selection.cache_key is None
    assert any("required package is unavailable" in reason for reason in selection.reasons)
    assert any("required import is unavailable" in reason for reason in selection.reasons)


def test_bridge_registry_is_explicit_and_duplicate_inputs_are_rejected() -> None:
    assert TORCH_FX_AVAILABLE_BRIDGE in APPROVED_COMPATIBILITY_BRIDGES
    with pytest.raises(RuntimeCompatibilityError, match="duplicate"):
        RuntimeRequirements(imports=(TORCH_FX_IMPORT, TORCH_FX_IMPORT))
    with pytest.raises(RuntimeCompatibilityError, match="duplicate"):
        resolve_runtime_profile(
            ModelRuntimeMetadata("acme/moe", REVISION),
            (_current(), _current()),
        )
    with pytest.raises(RuntimeCompatibilityError, match="target_import"):
        CompatibilityBridge("bad", "not an import", "reason")
