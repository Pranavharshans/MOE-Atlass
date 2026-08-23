"""Runtime-neutral support functions for the Transformers executor."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

from ..discovery import DiscoveryReport
from ..services.run_engine import sanitize_failure_message

_TOKENIZER_SHAPE_FAILURE = "tokenizer encoding must be shaped exactly (1, N)"


def _safe_validation_error(exc: Exception) -> str:
    """Summarize structured validation errors without retaining inputs.

    Pydantic validation entries can contain complete input values.  Read only
    the location, stable error type, and bounded message, ignoring every input
    and context field.  Other exception families remain class-name-only.
    """

    errors = getattr(exc, "errors", None)
    if type(exc).__name__ != "ValidationError" or not callable(errors):
        return f"structured forward failed ({type(exc).__name__})"
    try:
        try:
            entries = errors(include_input=False, include_url=False)
        except TypeError:
            entries = errors()
    except Exception:
        return "structured forward failed (ValidationError)"
    if not isinstance(entries, list):
        return "structured forward failed (ValidationError)"
    summaries: list[str] = []
    for entry in entries[:4]:
        if not isinstance(entry, Mapping):
            continue
        location = entry.get("loc", ())
        if isinstance(location, list | tuple):
            safe_parts = [str(part)[:80] for part in location if type(part) in {str, int}]
            location_text = ".".join(safe_parts) or "value"
        else:
            location_text = "value"
        error_type = entry.get("type")
        stable_type = error_type if isinstance(error_type, str) else "validation"
        message = entry.get("msg")
        stable_message = message if isinstance(message, str) else "validation failed"
        summaries.append(
            sanitize_failure_message(
                f"{location_text} [{stable_type[:80]}]: {stable_message[:240]}"
            )
        )
    if not summaries:
        return "structured forward failed (ValidationError)"
    return "structured forward validation failed: " + "; ".join(summaries)


def _materialize_ids(value: object) -> list[int]:
    current = value
    for method_name in ("detach", "cpu"):
        method = getattr(current, method_name, None)
        current = method() if callable(method) else current
    tolist = getattr(current, "tolist", None)
    rows = tolist() if callable(tolist) else current
    if type(rows) is list and len(rows) == 1 and type(rows[0]) is list:
        ids = rows[0]
    else:
        raise ValueError(_TOKENIZER_SHAPE_FAILURE)
    if any(type(item) is not int or isinstance(item, bool) or item < 0 for item in ids):
        raise ValueError("tokenizer input_ids must be non-negative integers")
    return ids


def _model_input_device(model: object) -> object | None:
    """Resolve the device expected by a model's first input tensor."""

    try:
        device = getattr(model, "device", None)
    except Exception:
        device = None
    if device is not None:
        return device
    try:
        parameters = getattr(model, "parameters", None)
        if callable(parameters):
            first = next(iter(parameters()))
            return getattr(first, "device", None)
    except (StopIteration, Exception):
        return None
    return None


def _move_model_inputs(model: object, encoding: dict[str, object]) -> dict[str, object]:
    """Move tensor-like tokenizer fields without importing a model stack."""

    device = _model_input_device(model)
    if device is None:
        return dict(encoding)
    moved: dict[str, object] = {}
    for key, value in encoding.items():
        to = getattr(value, "to", None)
        if callable(to):
            placed = to(device)
            moved[key] = value if placed is None else placed
        else:
            moved[key] = value
    return moved


def _publish_universal_inspection(workspace: object, run_key: str, report: object) -> None:
    """Persist one immutable universal topology beside the routing shard."""

    if not isinstance(workspace, str | Path):
        raise TypeError("workspace must be a string or Path")
    from ..adapters import build_universal_inspection

    inspection = build_universal_inspection(report)
    root = Path(workspace)
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("workspace must be an existing non-symlink directory")
    directory = root / "inspections"
    if directory.exists() and directory.is_symlink():
        raise RuntimeError("inspection directory must not be a symlink")
    directory.mkdir(exist_ok=True)
    target = directory / f"{run_key}.json"
    payload = inspection.to_json().encode("utf-8")
    if target.exists():
        if target.is_symlink() or not target.is_file() or target.read_bytes() != payload:
            raise RuntimeError("published universal inspection conflicts with the run")
        return
    fd, staged_name = tempfile.mkstemp(dir=str(directory), prefix=f".{run_key}.", suffix=".staging")
    staged = Path(staged_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
        os.replace(staged, target)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise


def _publish_discovery_report(workspace: object, run_key: str, report: DiscoveryReport) -> None:
    """Persist the exact static report used to derive the routing universe."""

    root = Path(workspace)
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("workspace must be an existing non-symlink directory")
    directory = root / "discoveries"
    if directory.exists() and directory.is_symlink():
        raise RuntimeError("discovery directory must not be a symlink")
    directory.mkdir(exist_ok=True)
    target = directory / f"{run_key}.json"
    payload = report.to_json().encode("utf-8")
    if target.exists():
        if target.is_symlink() or not target.is_file() or target.read_bytes() != payload:
            raise RuntimeError("published discovery report conflicts with the run")
        return
    fd, staged_name = tempfile.mkstemp(dir=str(directory), prefix=f".{run_key}.", suffix=".staging")
    staged = Path(staged_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
        os.replace(staged, target)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise


__all__ = [
    "_materialize_ids",
    "_move_model_inputs",
    "_publish_discovery_report",
    "_publish_universal_inspection",
    "_safe_validation_error",
]
