"""Bounded resource checks before a model allocates checkpoint weights."""

from __future__ import annotations

import importlib
import os
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

RESOURCE_ADMISSION_SCHEMA_VERSION = "1.0"
_GIB = 1024**3
_MAX_CACHE_FILES = 100_000


class ResourceAdmissionStatus(str, Enum):
    READY = "ready"
    CAUTION = "caution"
    REJECTED = "rejected"


class ResourceAdmissionError(RuntimeError):
    """A fixed, actionable resource rejection before model loading."""

    def __init__(self, message: str) -> None:
        super().__init__(message[:500])


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    disk_free_bytes: int | None
    accelerator_free_bytes: int | None = None
    accelerator_total_bytes: int | None = None
    accelerator_name: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "disk_free_bytes",
            "accelerator_free_bytes",
            "accelerator_total_bytes",
        ):
            value = getattr(self, field_name)
            if value is not None and (
                type(value) is not int or isinstance(value, bool) or value < 0
            ):
                raise ValueError(f"{field_name} must be non-negative or null")


@dataclass(frozen=True, slots=True)
class ResourceAdmission:
    status: ResourceAdmissionStatus
    checkpoint_bytes: int | None
    cached_bytes: int
    download_bytes: int | None
    disk_required_bytes: int | None
    accelerator_required_bytes: int | None
    snapshot: ResourceSnapshot
    reasons: tuple[str, ...] = ()

    @property
    def accepted(self) -> bool:
        return self.status is not ResourceAdmissionStatus.REJECTED

    def require(self) -> None:
        if not self.accepted:
            raise ResourceAdmissionError(self.reasons[0] if self.reasons else "model rejected")

    def to_dict(self) -> dict[str, Any]:
        return {
            "accelerator_free_bytes": self.snapshot.accelerator_free_bytes,
            "accelerator_name": self.snapshot.accelerator_name,
            "accelerator_required_bytes": self.accelerator_required_bytes,
            "accelerator_total_bytes": self.snapshot.accelerator_total_bytes,
            "cached_bytes": self.cached_bytes,
            "checkpoint_bytes": self.checkpoint_bytes,
            "disk_free_bytes": self.snapshot.disk_free_bytes,
            "disk_required_bytes": self.disk_required_bytes,
            "download_bytes": self.download_bytes,
            "reasons": list(self.reasons),
            "schema_version": RESOURCE_ADMISSION_SCHEMA_VERSION,
            "status": self.status.value,
        }


def evaluate_resource_admission(
    checkpoint_bytes: int | None,
    *,
    cached_bytes: int,
    snapshot: ResourceSnapshot,
    device: str,
    dtype: str,
) -> ResourceAdmission:
    """Evaluate disk strictly and accelerator capacity conservatively.

    Explicit CUDA placement is rejected when a known checkpoint estimate does
    not fit. ``auto`` remains a caution because Accelerate may legally place
    layers on CPU or unified memory instead of the discrete accelerator.
    """

    if checkpoint_bytes is not None and (
        type(checkpoint_bytes) is not int
        or isinstance(checkpoint_bytes, bool)
        or checkpoint_bytes < 0
    ):
        raise ValueError("checkpoint_bytes must be non-negative or null")
    if type(cached_bytes) is not int or isinstance(cached_bytes, bool) or cached_bytes < 0:
        raise ValueError("cached_bytes must be non-negative")
    if not isinstance(snapshot, ResourceSnapshot):
        raise TypeError("snapshot must be a ResourceSnapshot")
    if not isinstance(device, str) or not isinstance(dtype, str):
        raise TypeError("device and dtype must be strings")

    reasons: list[str] = []
    rejected = False
    download_bytes = (
        max(0, checkpoint_bytes - cached_bytes) if checkpoint_bytes is not None else None
    )
    disk_required = (
        download_bytes + max(_GIB, checkpoint_bytes // 20)
        if checkpoint_bytes is not None and download_bytes is not None
        else None
    )
    if (
        disk_required is not None
        and snapshot.disk_free_bytes is not None
        and disk_required > snapshot.disk_free_bytes
    ):
        rejected = True
        reasons.append("not enough free cache disk for this checkpoint")

    dtype_factor = 2.1 if dtype == "float32" else 1.2
    accelerator_required = (
        int(checkpoint_bytes * dtype_factor) + _GIB if checkpoint_bytes is not None else None
    )
    accelerator_shortfall = (
        accelerator_required is not None
        and snapshot.accelerator_free_bytes is not None
        and accelerator_required > snapshot.accelerator_free_bytes
    )
    if accelerator_shortfall and device.startswith("cuda"):
        rejected = True
        reasons.append("checkpoint is larger than the available accelerator memory budget")
    elif accelerator_shortfall and device == "auto":
        reasons.append("checkpoint may require CPU or unified-memory placement")

    if checkpoint_bytes is None:
        reasons.append("checkpoint size could not be established before loading")
    status = (
        ResourceAdmissionStatus.REJECTED
        if rejected
        else ResourceAdmissionStatus.CAUTION
        if reasons
        else ResourceAdmissionStatus.READY
    )
    return ResourceAdmission(
        status=status,
        checkpoint_bytes=checkpoint_bytes,
        cached_bytes=cached_bytes,
        download_bytes=download_bytes,
        disk_required_bytes=disk_required,
        accelerator_required_bytes=accelerator_required,
        snapshot=snapshot,
        reasons=tuple(reasons),
    )


def _hub_cache_root() -> Path:
    explicit = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _existing_disk_anchor(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def cached_snapshot_bytes(model_id: str, revision: str, *, cache_root: Path | None = None) -> int:
    """Return bounded unique file bytes already present for one snapshot."""

    if not isinstance(model_id, str) or model_id.count("/") != 1:
        raise ValueError("model_id must use owner/name form")
    if not isinstance(revision, str) or not revision:
        raise ValueError("revision must be a non-empty string")
    root = cache_root or _hub_cache_root()
    owner, name = model_id.split("/", 1)
    snapshot = root / f"models--{owner}--{name}" / "snapshots" / revision
    try:
        resolved_root = root.resolve()
        if snapshot.is_symlink() or not snapshot.is_dir():
            return 0
        seen: set[tuple[int, int]] = set()
        total = 0
        for index, candidate in enumerate(snapshot.rglob("*")):
            if index >= _MAX_CACHE_FILES:
                break
            if not candidate.is_file():
                continue
            relative_name = candidate.relative_to(snapshot).as_posix().casefold()
            if "/" in relative_name or not (
                relative_name.endswith(".safetensors")
                or (
                    relative_name.endswith(".bin")
                    and ("pytorch_model" in relative_name or relative_name.startswith("model"))
                )
            ):
                continue
            resolved = candidate.resolve()
            if not resolved.is_relative_to(resolved_root):
                continue
            stat = resolved.stat()
            identity = (stat.st_dev, stat.st_ino)
            if identity not in seen:
                seen.add(identity)
                total += stat.st_size
        return total
    except OSError:
        return 0


def observe_resource_snapshot(*, cache_root: Path | None = None) -> ResourceSnapshot:
    """Observe local cache disk and CUDA memory without making either mandatory."""

    root = cache_root or _hub_cache_root()
    try:
        disk_free = shutil.disk_usage(_existing_disk_anchor(root)).free
    except OSError:
        disk_free = None

    accelerator_free: int | None = None
    accelerator_total: int | None = None
    accelerator_name: str | None = None
    try:
        torch = importlib.import_module("torch")
        cuda = getattr(torch, "cuda")
        if cuda.is_available():
            free, total = cuda.mem_get_info()
            if type(free) is int and type(total) is int:
                accelerator_free = free
                accelerator_total = total
            name = cuda.get_device_name(0)
            if isinstance(name, str):
                accelerator_name = name[:160]
    except Exception:
        pass
    return ResourceSnapshot(
        disk_free_bytes=disk_free,
        accelerator_free_bytes=accelerator_free,
        accelerator_total_bytes=accelerator_total,
        accelerator_name=accelerator_name,
    )


def estimate_hub_checkpoint_bytes(
    model_id: str,
    revision: str,
    *,
    fallback_repository_bytes: int | None = None,
) -> int | None:
    """Ask the Hub for per-file metadata and estimate selected weight bytes."""

    try:
        hub = importlib.import_module("huggingface_hub")
        info = hub.HfApi().model_info(
            repo_id=model_id,
            revision=revision,
            files_metadata=True,
            timeout=20,
        )
        siblings = getattr(info, "siblings", ())
        groups: dict[str, list[int]] = {"safetensors": [], "pytorch": []}
        for sibling in siblings or ():
            filename = getattr(sibling, "rfilename", None)
            size = getattr(sibling, "size", None)
            if not isinstance(filename, str) or type(size) is not int or size < 0:
                continue
            lower = filename.casefold()
            # ``from_pretrained(repo_id)`` selects the repository root.  Do
            # not count alternate quantizations or conversions in subfolders.
            if "/" in lower:
                continue
            if lower.endswith(".safetensors") and "optimizer" not in lower:
                groups["safetensors"].append(size)
            elif lower.endswith(".bin") and (
                "pytorch_model" in lower or lower.rsplit("/", 1)[-1].startswith("model")
            ):
                groups["pytorch"].append(size)
        if groups["safetensors"]:
            return sum(groups["safetensors"])
        if groups["pytorch"]:
            return sum(groups["pytorch"])
    except Exception:
        pass
    return fallback_repository_bytes


def admit_huggingface_model(
    model_id: str,
    revision: str,
    *,
    device: str,
    dtype: str,
    allow_network: bool = True,
) -> ResourceAdmission:
    """Observe and enforce one pre-download model resource decision."""

    root = _hub_cache_root()
    if type(allow_network) is not bool:
        raise TypeError("allow_network must be a boolean")
    checkpoint_bytes = (
        estimate_hub_checkpoint_bytes(
            model_id,
            revision,
            # ``usedStorage`` can include alternate formats and subfolders.  It is
            # retained as evidence by the resolver but is not safe for a hard
            # admission decision when exact per-file metadata is unavailable.
            fallback_repository_bytes=None,
        )
        if allow_network
        else None
    )
    admission = evaluate_resource_admission(
        checkpoint_bytes,
        cached_bytes=cached_snapshot_bytes(model_id, revision, cache_root=root),
        snapshot=observe_resource_snapshot(cache_root=root),
        device=device,
        dtype=dtype,
    )
    admission.require()
    return admission


__all__ = [
    "RESOURCE_ADMISSION_SCHEMA_VERSION",
    "ResourceAdmission",
    "ResourceAdmissionError",
    "ResourceAdmissionStatus",
    "ResourceSnapshot",
    "admit_huggingface_model",
    "cached_snapshot_bytes",
    "estimate_hub_checkpoint_bytes",
    "evaluate_resource_admission",
    "observe_resource_snapshot",
]
