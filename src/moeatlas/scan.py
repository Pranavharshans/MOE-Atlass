"""Phase 0 scan source resolution and deterministic report output."""

from __future__ import annotations

import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from . import __version__
from .core import (
    DType,
    ModelManifest,
    Provenance,
    TokenizerIdentity,
    make_config_hash,
    make_model_key,
)
from .discovery import DiscoveryReport
from .discovery import scan as discover_scan
from .fixtures import SyntheticConfig, SyntheticMoE

PHASE0_FIXTURE_SOURCE = "fixture:synthetic"
SYNTHETIC_MODEL_KEY = make_model_key("fixture/synthetic-moe", "v1")
SYNTHETIC_TOKENIZER_ID = "fixture/synthetic-tokenizer"
SYNTHETIC_REVISION = "v1"


class ScanSourceError(ValueError):
    """Raised when a requested Phase 0 scan source is unavailable."""


class ScanOutputError(OSError):
    """Raised when a report cannot be written without violating output policy."""


def synthetic_model_manifest() -> ModelManifest:
    """Return the fixed, explicitly non-certified manifest for the fixture."""

    config = asdict(SyntheticConfig())
    return ModelManifest(
        model_key=SYNTHETIC_MODEL_KEY,
        architecture="synthetic_moe",
        revision=SYNTHETIC_REVISION,
        config_hash=make_config_hash(config),
        tokenizer=TokenizerIdentity(
            identifier=SYNTHETIC_TOKENIZER_ID,
            revision=SYNTHETIC_REVISION,
        ),
        dtype=DType.FLOAT32,
        device_map={"": "cpu"},
        provenance=Provenance(
            source=PHASE0_FIXTURE_SOURCE,
            tool_version=__version__,
            metadata={"fixture": "synthetic", "certified": False},
        ),
        warnings=[
            "fixture:synthetic is a deterministic torch-free surface; "
            "real-model support is deferred"
        ],
    )


def resolve_scan_source(source: str) -> tuple[SyntheticMoE, ModelManifest]:
    """Resolve only the explicit Phase 0 fixture namespace.

    No other source is inspected, loaded, looked up in a cache, or sent to a
    network. Real HF/local loading belongs to the deferred Phase 1 workflow.
    """

    if source != PHASE0_FIXTURE_SOURCE:
        raise ScanSourceError(
            f"source {source!r} is not available in Phase 0; use "
            f"{PHASE0_FIXTURE_SOURCE!r}. HF/local model loading is deferred to Phase 1 "
            "(MV-01/MV-02)."
        )
    return SyntheticMoE(), synthetic_model_manifest()


def scan_source(source: str) -> DiscoveryReport:
    """Build the complete semantic report for one supported Phase 0 source."""

    model, manifest = resolve_scan_source(source)
    return discover_scan(model, manifest)


def report_payload(report: DiscoveryReport) -> str:
    """Return one deterministic JSON document, terminated by one newline."""

    return f"{report.to_json()}\n"


def write_report_atomic(payload: str, output: str | Path, *, force: bool = False) -> Path:
    """Write a complete report through a same-directory temporary file.

    Parent directories are never created implicitly. Without ``force``, an
    existing output is refused and left untouched; publication uses an atomic
    hard-link create so a concurrent destination cannot be clobbered. With
    ``force``, the completed temporary file is atomically replaced.
    """

    target = Path(output)
    target_exists = os.path.lexists(target)
    if target_exists:
        if target.is_dir():
            raise ScanOutputError(f"output path is a directory: {target}")
        if not force:
            raise ScanOutputError(f"output already exists: {target}; pass --force to replace it")

    parent = target.parent
    if not parent.exists():
        raise ScanOutputError(f"output parent does not exist: {parent}")
    if not parent.is_dir():
        raise ScanOutputError(f"output parent is not a directory: {parent}")

    temporary_path: Path | None = None
    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=parent,
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

        if force:
            os.replace(temporary_path, target)
        else:
            try:
                os.link(temporary_path, target)
            except FileExistsError as exc:
                raise ScanOutputError(
                    f"output already exists: {target}; pass --force to replace it"
                ) from exc
            os.unlink(temporary_path)
        temporary_path = None
    except ScanOutputError:
        raise
    except OSError as exc:
        raise ScanOutputError(f"could not write output {target}: {exc}") from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
    return target


__all__ = [
    "PHASE0_FIXTURE_SOURCE",
    "SYNTHETIC_MODEL_KEY",
    "SYNTHETIC_REVISION",
    "SYNTHETIC_TOKENIZER_ID",
    "ScanOutputError",
    "ScanSourceError",
    "report_payload",
    "resolve_scan_source",
    "scan_source",
    "synthetic_model_manifest",
    "write_report_atomic",
]
