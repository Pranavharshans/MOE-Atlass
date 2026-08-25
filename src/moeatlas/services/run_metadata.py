"""Durable reconstruction metadata for exact baseline-derived runs."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..core import validate_stable_identifier
from ..runs import RunSpecification
from ..store.catalog import read_catalog

RUN_METADATA_SCHEMA_VERSION = "1.0"
RUN_METADATA_ARTIFACT_TYPE = "moeatlas.run_metadata"

_DIRECTORY = "run-metadata"
_NAMED_DIRECTORY = "runs"
_NAMED_FILE = "run.json"
_MAX_BYTES = 1_000_000
_MAX_NAMED_RUNS = 10_000


class RunMetadataError(RuntimeError):
    """Safe failure for baseline reconstruction metadata."""


def publish_run_metadata(
    workspace: str | Path,
    specification: RunSpecification,
    request: Mapping[str, Any],
) -> Path:
    """Publish the exact resolved request and immutable run specification."""

    if type(specification) is not RunSpecification:
        raise TypeError("specification must be an exact RunSpecification")
    if not isinstance(request, Mapping) or not all(type(key) is str for key in request):
        raise TypeError("request must be a mapping with string keys")
    document = {
        "artifact_type": RUN_METADATA_ARTIFACT_TYPE,
        "schema_version": RUN_METADATA_SCHEMA_VERSION,
        "run_key": specification.run_key,
        "request": dict(request),
        "specification": specification.model_dump(mode="json"),
    }
    try:
        payload = json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RunMetadataError("run metadata is not canonical JSON") from exc
    if len(payload) > _MAX_BYTES:
        raise RunMetadataError("run metadata exceeds the byte budget")

    root = Path(workspace)
    if root.is_symlink() or not root.is_dir():
        raise RunMetadataError("workspace must be an existing non-symlink directory")
    try:
        catalog = read_catalog(root)
    except Exception as exc:
        raise RunMetadataError("workspace catalog is unavailable") from exc
    if specification.run_name is not None:
        for entry in catalog.runs:
            if entry.run_key == specification.run_key and entry.run_name not in (
                None,
                specification.run_name,
            ):
                raise RunMetadataError("published run metadata conflicts with the run name")
            if (
                entry.run_key != specification.run_key
                and entry.run_name is not None
                and entry.run_name.casefold() == specification.run_name.casefold()
            ):
                raise RunMetadataError("run name is already in use")
        parent = root / _NAMED_DIRECTORY
        if parent.exists() and (parent.is_symlink() or not parent.is_dir()):
            raise RunMetadataError("named runs directory is unsafe")
        parent.mkdir(exist_ok=True)
        directory = parent / specification.run_name
        if directory.exists() and (directory.is_symlink() or not directory.is_dir()):
            raise RunMetadataError("named run directory is unsafe")
        directory.mkdir(exist_ok=True)
        target = directory / _NAMED_FILE
        staging_prefix = ".run."
    else:
        directory = root / _DIRECTORY
        target = directory / f"{specification.run_key}.json"
        staging_prefix = f".{specification.run_key}."
    if directory.exists() and directory.is_symlink():
        raise RunMetadataError("run metadata directory is unsafe")
    directory.mkdir(exist_ok=True)
    if target.exists():
        if target.is_symlink() or not target.is_file() or target.read_bytes() != payload:
            raise RunMetadataError("published run metadata conflicts with the run")
        return target
    fd, staged_name = tempfile.mkstemp(
        dir=str(directory), prefix=staging_prefix, suffix=".staging"
    )
    staged = Path(staged_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
        os.replace(staged, target)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return target


def read_run_metadata(
    workspace: str | Path,
    run_key: str,
    *,
    max_bytes: int = _MAX_BYTES,
) -> dict[str, Any]:
    """Read and fully validate one bounded reconstruction record."""

    validate_stable_identifier(run_key, field_name="run_key")
    root = Path(workspace)
    target = root / _DIRECTORY / f"{run_key}.json"
    if not target.exists():
        try:
            catalog = read_catalog(root)
            entry = next(item for item in catalog.runs if item.run_key == run_key)
            if entry.run_name is not None:
                target = root / _NAMED_DIRECTORY / entry.run_name / _NAMED_FILE
        except Exception:
            named_root = root / _NAMED_DIRECTORY
            candidates: list[Path] = []
            if named_root.is_dir() and not named_root.is_symlink():
                children = tuple(named_root.iterdir())
                if len(children) > _MAX_NAMED_RUNS:
                    raise RunMetadataError("named run registry exceeds its read budget")
                for child in children:
                    candidate = child / _NAMED_FILE
                    if (
                        child.is_dir()
                        and not child.is_symlink()
                        and candidate.is_file()
                        and not candidate.is_symlink()
                        and candidate.stat().st_size <= max_bytes
                    ):
                        try:
                            candidate_document = json.loads(
                                candidate.read_text(encoding="utf-8")
                            )
                        except Exception:
                            continue
                        if candidate_document.get("run_key") == run_key:
                            candidates.append(candidate)
            if len(candidates) != 1:
                raise RunMetadataError("run metadata is unavailable")
            target = candidates[0]
    try:
        if target.is_symlink() or not target.is_file() or target.stat().st_size > max_bytes:
            raise RunMetadataError("run metadata is unavailable")
        payload = target.read_bytes()
        document = json.loads(payload.decode("utf-8"))
    except RunMetadataError:
        raise
    except Exception as exc:
        raise RunMetadataError("run metadata is unavailable") from exc
    if (
        not isinstance(document, dict)
        or document.get("artifact_type") != RUN_METADATA_ARTIFACT_TYPE
        or document.get("schema_version") != RUN_METADATA_SCHEMA_VERSION
        or document.get("run_key") != run_key
        or not isinstance(document.get("request"), dict)
        or not isinstance(document.get("specification"), dict)
    ):
        raise RunMetadataError("run metadata is invalid")
    try:
        specification = RunSpecification.model_validate(document["specification"])
    except Exception as exc:
        raise RunMetadataError("run specification metadata is invalid") from exc
    if specification.run_key != run_key:
        raise RunMetadataError("run specification metadata has a mismatched key")
    return document


__all__ = [
    "RUN_METADATA_ARTIFACT_TYPE",
    "RUN_METADATA_SCHEMA_VERSION",
    "RunMetadataError",
    "publish_run_metadata",
    "read_run_metadata",
]
