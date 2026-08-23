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

RUN_METADATA_SCHEMA_VERSION = "1.0"
RUN_METADATA_ARTIFACT_TYPE = "moeatlas.run_metadata"

_DIRECTORY = "run-metadata"
_MAX_BYTES = 1_000_000


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
    directory = root / _DIRECTORY
    if directory.exists() and directory.is_symlink():
        raise RunMetadataError("run metadata directory is unsafe")
    directory.mkdir(exist_ok=True)
    target = directory / f"{specification.run_key}.json"
    if target.exists():
        if target.is_symlink() or not target.is_file() or target.read_bytes() != payload:
            raise RunMetadataError("published run metadata conflicts with the run")
        return target
    fd, staged_name = tempfile.mkstemp(
        dir=str(directory), prefix=f".{specification.run_key}.", suffix=".staging"
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
    target = Path(workspace) / _DIRECTORY / f"{run_key}.json"
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
