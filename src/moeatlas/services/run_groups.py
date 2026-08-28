"""Durable parent manifests for sequential multi-dataset run groups."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

RUN_GROUP_SCHEMA_VERSION = "1.0"
RUN_GROUP_ARTIFACT_TYPE = "moeatlas.run_group"

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_SLUG_PART = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_BYTES = 1_000_000
_MAX_GROUPS = 1_000


class RunGroupError(RuntimeError):
    """Safe failure for malformed or conflicting run-group state."""


def dataset_slug(dataset_id: str, config_name: str | None, index: int) -> str:
    """Create one stable, collision-resistant child label."""

    if type(dataset_id) is not str or not dataset_id.strip() or type(index) is not int:
        raise TypeError("dataset identity and index are required")
    base = dataset_id.rsplit("/", 1)[-1]
    if config_name:
        base = f"{base}-{config_name}"
    normalized = _SLUG_PART.sub("-", base).strip(".-_").lower() or "dataset"
    return f"{index + 1:02d}-{normalized[:48]}"


def child_run_name(group_name: str, slug: str) -> str:
    """Bind a child to its parent while retaining the existing flat run registry."""

    if not _SAFE_NAME.fullmatch(group_name):
        raise ValueError("run group name is invalid")
    suffix_budget = 79 - len(group_name) - 2
    if suffix_budget < 3:
        raise ValueError("run group name is too long for child run names")
    return f"{group_name}--{slug[:suffix_budget]}"


def _group_root(workspace: str | Path, group_name: str) -> Path:
    if not _SAFE_NAME.fullmatch(group_name):
        raise ValueError("run group name is invalid")
    root = Path(workspace)
    if root.is_symlink() or not root.is_dir():
        raise RunGroupError("workspace must be an existing non-symlink directory")
    runs = root / "runs"
    if runs.exists() and (runs.is_symlink() or not runs.is_dir()):
        raise RunGroupError("named runs directory is unsafe")
    runs.mkdir(exist_ok=True)
    group = runs / group_name
    if group.exists() and (group.is_symlink() or not group.is_dir()):
        raise RunGroupError("run group directory is unsafe")
    return group


def _atomic_json(target: Path, document: Mapping[str, Any]) -> None:
    try:
        payload = json.dumps(
            dict(document),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RunGroupError("run group metadata is not canonical JSON") from exc
    if len(payload) > _MAX_BYTES:
        raise RunGroupError("run group metadata exceeds the byte budget")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink():
        raise RunGroupError("run group directory is unsafe")
    fd, staged_name = tempfile.mkstemp(
        dir=str(target.parent), prefix=f".{target.name}.", suffix=".staging"
    )
    staged = Path(staged_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
        os.replace(staged, target)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise


def publish_run_group(
    workspace: str | Path,
    group_name: str,
    children: Sequence[Mapping[str, Any]],
    *,
    state: str,
) -> Path:
    """Publish the parent index and one dataset-addressable child pointer each."""

    if state not in {"queued", "running", "completed", "partial", "failed", "cancelled"}:
        raise ValueError("run group state is invalid")
    if not children or len(children) > 64:
        raise ValueError("run group children must contain between 1 and 64 entries")
    normalized = [dict(child) for child in children]
    if any(not isinstance(child.get("slug"), str) for child in normalized):
        raise TypeError("each run group child requires a slug")
    group = _group_root(workspace, group_name)
    group.mkdir(exist_ok=True)
    document = {
        "artifact_type": RUN_GROUP_ARTIFACT_TYPE,
        "schema_version": RUN_GROUP_SCHEMA_VERSION,
        "group_name": group_name,
        "state": state,
        "children": normalized,
    }
    _atomic_json(group / "group.json", document)
    for child in normalized:
        _atomic_json(group / "datasets" / child["slug"] / "run.json", child)
    return group / "group.json"


def list_run_groups(workspace: str | Path) -> tuple[dict[str, Any], ...]:
    """Read bounded, non-symlink parent manifests from the named-runs directory."""

    runs = Path(workspace) / "runs"
    if not runs.exists():
        return ()
    if runs.is_symlink() or not runs.is_dir():
        raise RunGroupError("named runs directory is unsafe")
    groups: list[dict[str, Any]] = []
    children = tuple(runs.iterdir())
    if len(children) > _MAX_GROUPS * 2:
        raise RunGroupError("run group registry exceeds its read budget")
    for directory in children:
        target = directory / "group.json"
        if (
            directory.is_symlink()
            or not directory.is_dir()
            or target.is_symlink()
            or not target.is_file()
            or target.stat().st_size > _MAX_BYTES
        ):
            continue
        try:
            document = json.loads(target.read_text(encoding="utf-8"))
        except Exception:
            continue
        if (
            isinstance(document, dict)
            and document.get("artifact_type") == RUN_GROUP_ARTIFACT_TYPE
            and document.get("schema_version") == RUN_GROUP_SCHEMA_VERSION
            and document.get("group_name") == directory.name
            and isinstance(document.get("children"), list)
        ):
            groups.append(document)
    return tuple(sorted(groups, key=lambda group: group["group_name"].casefold()))


__all__ = [
    "RunGroupError",
    "child_run_name",
    "dataset_slug",
    "list_run_groups",
    "publish_run_group",
]
