"""Model-free contracts for pre-download resource admission."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import moeatlas.runtime.resource_admission as admission_module
from moeatlas.runtime import (
    ResourceAdmissionError,
    ResourceAdmissionStatus,
    ResourceSnapshot,
    cached_snapshot_bytes,
    estimate_hub_checkpoint_bytes,
    evaluate_resource_admission,
)

GIB = 1024**3


def test_disk_shortfall_is_rejected_before_download() -> None:
    admission = evaluate_resource_admission(
        20 * GIB,
        cached_bytes=0,
        snapshot=ResourceSnapshot(disk_free_bytes=10 * GIB),
        device="auto",
        dtype="bfloat16",
    )

    assert admission.status is ResourceAdmissionStatus.REJECTED
    assert admission.download_bytes == 20 * GIB
    with pytest.raises(ResourceAdmissionError, match="cache disk"):
        admission.require()


def test_auto_placement_warns_while_explicit_cuda_rejects_memory_shortfall() -> None:
    snapshot = ResourceSnapshot(
        disk_free_bytes=100 * GIB,
        accelerator_free_bytes=10 * GIB,
        accelerator_total_bytes=24 * GIB,
        accelerator_name="Synthetic GPU",
    )
    automatic = evaluate_resource_admission(
        20 * GIB,
        cached_bytes=20 * GIB,
        snapshot=snapshot,
        device="auto",
        dtype="float16",
    )
    explicit = evaluate_resource_admission(
        20 * GIB,
        cached_bytes=20 * GIB,
        snapshot=snapshot,
        device="cuda",
        dtype="float16",
    )

    assert automatic.status is ResourceAdmissionStatus.CAUTION
    assert automatic.accepted is True
    assert explicit.status is ResourceAdmissionStatus.REJECTED
    assert explicit.accepted is False


def test_unknown_checkpoint_size_is_caution_not_false_rejection() -> None:
    admission = evaluate_resource_admission(
        None,
        cached_bytes=0,
        snapshot=ResourceSnapshot(disk_free_bytes=1),
        device="cuda",
        dtype="preserve",
    )

    assert admission.status is ResourceAdmissionStatus.CAUTION
    assert admission.accepted is True


def test_cached_snapshot_counts_unique_hub_blobs(tmp_path: Path) -> None:
    root = tmp_path / "hub"
    blob = root / "models--org--model" / "blobs" / "abc"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"weights")
    config_blob = blob.parent / "config"
    config_blob.write_bytes(b"configuration")
    snapshot = root / "models--org--model" / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "one.safetensors").symlink_to(blob)
    (snapshot / "duplicate.safetensors").symlink_to(blob)
    (snapshot / "config.json").symlink_to(config_blob)

    assert cached_snapshot_bytes("org/model", "a" * 40, cache_root=root) == 7


def test_hub_estimate_prefers_root_safetensors_and_ignores_alternates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    info = SimpleNamespace(
        siblings=[
            SimpleNamespace(rfilename="model-00001-of-00002.safetensors", size=11),
            SimpleNamespace(rfilename="model-00002-of-00002.safetensors", size=13),
            SimpleNamespace(rfilename="quantized/model.safetensors", size=5_000),
            SimpleNamespace(rfilename="pytorch_model.bin", size=100),
        ]
    )

    class Api:
        def model_info(self, **kwargs):
            assert kwargs["files_metadata"] is True
            return info

    real_import = admission_module.importlib.import_module

    def fake_import(name: str):
        if name == "huggingface_hub":
            return SimpleNamespace(HfApi=Api)
        return real_import(name)

    monkeypatch.setattr(admission_module.importlib, "import_module", fake_import)

    assert estimate_hub_checkpoint_bytes("org/model", "a" * 40) == 24
