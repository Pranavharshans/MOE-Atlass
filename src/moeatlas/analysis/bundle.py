"""Bounded, atomic export bundles for canonical analysis artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .routing_compare import RoutingLoadComparison
from .routing_load import RoutingLoadMatrix
from .routing_summary import RoutingLoadSummary

ANALYSIS_BUNDLE_SCHEMA_VERSION = "1.0"

_BUNDLE_ARTIFACT_TYPE = "moeatlas.analysis_bundle"
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_MATRIX_NAME = "routing_load_matrix.json"
_COMPARISON_NAME = "routing_load_comparison.json"
_SUMMARY_NAME = "routing_load_summary.json"
_MANIFEST_NAME = "manifest.json"
_TEMP_PREFIX = ".bundle-temp-"


@dataclass(frozen=True, slots=True)
class AnalysisBundleEntry:
    name: str
    sha256: str
    bytes: int


@dataclass(frozen=True, slots=True)
class AnalysisBundleReceipt:
    schema_version: str
    entries: tuple[AnalysisBundleEntry, ...]
    total_bytes: int

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not str
            or self.schema_version != ANALYSIS_BUNDLE_SCHEMA_VERSION
        ):
            raise ValueError("schema_version is not the exact analysis-bundle version")
        if type(self.entries) is not tuple or not self.entries:
            raise ValueError("entries must be a non-empty tuple")
        names: list[str] = []
        total = 0
        for entry in self.entries:
            if type(entry) is not AnalysisBundleEntry:
                raise TypeError("entries must contain exact AnalysisBundleEntry values")
            if type(entry.name) is not str or not entry.name.endswith(".json"):
                raise ValueError("entry names must be canonical .json names")
            if entry.name in names:
                raise ValueError("entry names must be unique")
            names.append(entry.name)
            if type(entry.sha256) is not str or _DIGEST.fullmatch(entry.sha256) is None:
                raise ValueError("entry digests must be sha256:<64hex>")
            if type(entry.bytes) is not int or isinstance(entry.bytes, bool) or entry.bytes <= 0:
                raise ValueError("entry byte counts must be strict positive integers")
            total += entry.bytes
        if names != sorted(names):
            raise ValueError("entries must be sorted by name")
        if type(self.total_bytes) is not int or isinstance(self.total_bytes, bool):
            raise TypeError("total_bytes must be an exact integer")
        if self.total_bytes != total:
            raise ValueError("total_bytes does not match the entry byte counts")


def _document(value: object) -> str:
    return value.to_json()  # type: ignore[attr-defined]


def _identity(value: object) -> tuple[str, str, str, str, str]:
    return (
        value.model_key,  # type: ignore[attr-defined]
        value.adapter_name,  # type: ignore[attr-defined]
        value.adapter_version,  # type: ignore[attr-defined]
        value.inspection_digest,  # type: ignore[attr-defined]
        value.layout,  # type: ignore[attr-defined]
    )


def _digest_document(document: str) -> str:
    return "sha256:" + hashlib.sha256(document.encode("utf-8")).hexdigest()


def _atomic_write(path: Path, payload: str) -> None:
    temporary = path.parent / (_TEMP_PREFIX + path.name)
    try:
        with open(temporary, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _validate_destination(raw: object) -> Path:
    if not isinstance(raw, str | Path):
        raise TypeError("destination must be a string or pathlib.Path")
    destination = Path(raw)
    if destination.exists():
        if not destination.is_dir() or any(destination.iterdir()):
            raise ValueError("destination must be a nonexistent or empty directory")
    parent = destination.parent
    if not parent.is_dir():
        raise ValueError("destination parent is not a directory")
    return destination


def write_analysis_bundle(
    destination: str | Path,
    *,
    matrix: RoutingLoadMatrix | None = None,
    comparison: RoutingLoadComparison | None = None,
    summary: RoutingLoadSummary | None = None,
) -> AnalysisBundleReceipt:
    """Write one coherent, atomically published analysis bundle directory."""

    selected: list[tuple[str, object]] = []
    if matrix is not None:
        if type(matrix) is not RoutingLoadMatrix:
            raise TypeError("matrix must be an exact RoutingLoadMatrix or None")
        selected.append((_MATRIX_NAME, matrix))
    if comparison is not None:
        if type(comparison) is not RoutingLoadComparison:
            raise TypeError("comparison must be an exact RoutingLoadComparison or None")
        selected.append((_COMPARISON_NAME, comparison))
    if summary is not None:
        if type(summary) is not RoutingLoadSummary:
            raise TypeError("summary must be an exact RoutingLoadSummary or None")
        selected.append((_SUMMARY_NAME, summary))
    if not selected:
        raise ValueError("a bundle requires at least one analysis artifact")

    identities = {_identity(value) for _, value in selected}
    if len(identities) != 1:
        raise ValueError("bundle artifacts do not share one model/adapter/inspection identity")

    destination_path = _validate_destination(destination)
    documents = [(name, _document(value)) for name, value in selected]
    entries = [
        {
            "name": name,
            "sha256": _digest_document(document),
            "bytes": len(document.encode("utf-8")),
        }
        for name, document in sorted(documents, key=lambda item: item[0])
    ]
    manifest_document = json.dumps(
        {
            "artifact_type": _BUNDLE_ARTIFACT_TYPE,
            "schema_version": ANALYSIS_BUNDLE_SCHEMA_VERSION,
            "entry_count": len(entries),
            "entries": entries,
            "total_bytes": sum(entry["bytes"] for entry in entries),
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    created_directory = False
    written: list[Path] = []
    try:
        if not destination_path.exists():
            destination_path.mkdir()
            created_directory = True
        for name, document in documents:
            target = destination_path / name
            _atomic_write(target, document)
            written.append(target)
        manifest_target = destination_path / _MANIFEST_NAME
        _atomic_write(manifest_target, manifest_document)
        written.append(manifest_target)
    except BaseException:
        for path in written:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        if created_directory:
            try:
                destination_path.rmdir()
            except OSError:
                pass
        raise

    receipt_entries = tuple(
        AnalysisBundleEntry(name=entry["name"], sha256=entry["sha256"], bytes=entry["bytes"])
        for entry in entries
    )
    return AnalysisBundleReceipt(
        schema_version=ANALYSIS_BUNDLE_SCHEMA_VERSION,
        entries=receipt_entries,
        total_bytes=sum(entry.bytes for entry in receipt_entries),
    )


__all__ = [
    "ANALYSIS_BUNDLE_SCHEMA_VERSION",
    "AnalysisBundleEntry",
    "AnalysisBundleReceipt",
    "write_analysis_bundle",
]
