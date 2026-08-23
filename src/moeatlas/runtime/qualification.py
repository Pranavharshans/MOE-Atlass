"""Pre-weight qualification for a resolved Hugging Face model runtime.

Qualification downloads only small metadata/source files from the already
resolved revision.  It never imports remote model code, installs packages, or
allocates model weights.  The resulting environment identity and install plan
let the server fail early and explain what a model needs before an expensive
load is attempted.
"""

from __future__ import annotations

import ast
import importlib
import importlib.metadata
import json
import re
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core import stable_digest

RUNTIME_QUALIFICATION_SCHEMA_VERSION = "1.0"
_REQUIREMENT = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9_.-]*)(?P<constraint>(?:==|!=|>=|<=|>|<|~=).+)?$"
)
_REMOTE_MODULE = re.compile(r"^(?:[^-]+--)?(?P<module>[A-Za-z_][A-Za-z0-9_.]*)\.[A-Za-z_]\w*$")
_MAX_SOURCE_BYTES = 1_000_000
_MAX_REMOTE_FILES = 32


class RuntimeQualificationError(RuntimeError):
    """Raised when the model cannot be qualified safely."""


@dataclass(frozen=True, slots=True)
class RuntimeQualification:
    """Bounded evidence about whether the current worker can load a model."""

    status: str
    environment_id: str
    model_id: str
    resolved_revision: str
    remote_code_required: bool
    trust_remote_code: bool
    transformers_version: str | None
    declared_transformers_version: str | None
    requirements: tuple[str, ...]
    inspected_remote_files: tuple[str, ...]
    uninspectable_remote_files: tuple[str, ...]
    required_imports: tuple[str, ...]
    missing_packages: tuple[str, ...]
    missing_imports: tuple[str, ...]
    warnings: tuple[str, ...]
    install_plan: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.status not in {"ready", "incompatible"}:
            raise ValueError("qualification status must be ready or incompatible")

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    def to_dict(self) -> dict[str, object]:
        return {
            "declared_transformers_version": self.declared_transformers_version,
            "environment_id": self.environment_id,
            "inspected_remote_files": list(self.inspected_remote_files),
            "install_plan": list(self.install_plan),
            "missing_imports": list(self.missing_imports),
            "missing_packages": list(self.missing_packages),
            "model_id": self.model_id,
            "remote_code_required": self.remote_code_required,
            "required_imports": list(self.required_imports),
            "resolved_revision": self.resolved_revision,
            "schema_version": RUNTIME_QUALIFICATION_SCHEMA_VERSION,
            "status": self.status,
            "transformers_version": self.transformers_version,
            "trust_remote_code": self.trust_remote_code,
            "uninspectable_remote_files": list(self.uninspectable_remote_files),
            "warnings": list(self.warnings),
            "requirements": list(self.requirements),
        }


FetchFile = Callable[[str, str, str, bool], bytes | None]


def _hub_fetch(model_id: str, revision: str, filename: str, allow_network: bool) -> bytes | None:
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=model_id,
            revision=revision,
            filename=filename,
            local_files_only=not allow_network,
        )
    except Exception:
        return None
    try:
        payload = Path(path).read_bytes()
    except OSError:
        return None
    return payload if len(payload) <= _MAX_SOURCE_BYTES else None


def _auto_map_values(config: Mapping[str, Any]) -> tuple[str, ...]:
    auto_map = config.get("auto_map")
    if not isinstance(auto_map, Mapping):
        return ()
    values: list[str] = []
    for value in auto_map.values():
        candidates: Iterable[object] = value if isinstance(value, list) else (value,)
        for candidate in candidates:
            if isinstance(candidate, str) and candidate not in values:
                values.append(candidate)
    return tuple(sorted(values))


def _remote_files(auto_map: Iterable[str]) -> tuple[str, ...]:
    files: set[str] = set()
    for reference in auto_map:
        match = _REMOTE_MODULE.fullmatch(reference)
        if match is None:
            continue
        files.add(match.group("module").replace(".", "/") + ".py")
    return tuple(sorted(files))[:_MAX_REMOTE_FILES]


def _source_imports(payload: bytes) -> tuple[str, ...]:
    try:
        tree = ast.parse(payload.decode("utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return ()
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            for alias in node.names:
                if alias.name == "*":
                    imports.add(node.module)
                else:
                    imports.add(f"{node.module}:{alias.name}")
    return tuple(sorted(imports))


def _requirements(payload: bytes | None) -> tuple[str, ...]:
    if payload is None:
        return ()
    result: set[str] = set()
    for raw_line in payload.decode("utf-8", errors="ignore").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith(("-", "http:", "https:", "git+")):
            continue
        line = line.split(";", 1)[0].strip()
        if _REQUIREMENT.fullmatch(line):
            result.add(line)
    return tuple(sorted(result))


def _distribution_missing(requirement: str) -> bool:
    match = _REQUIREMENT.fullmatch(requirement)
    if match is None:
        return False
    name = match.group("name")
    try:
        importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return True
    return False


def _import_missing(requirement: str) -> bool:
    module_name, separator, attribute = requirement.partition(":")
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return True
    return bool(separator and not hasattr(module, attribute))


def qualify_huggingface_runtime(
    plan: Any,
    *,
    fetch_file: FetchFile = _hub_fetch,
) -> RuntimeQualification:
    """Qualify a resolved Hugging Face plan before loading model weights."""

    source = getattr(plan, "source", None)
    config = getattr(plan, "config", None)
    resolution = getattr(plan, "resolution", None)
    if source is None or config is None or resolution is None:
        raise RuntimeQualificationError("qualification requires a resolved loading plan")
    model_id = source.model_id
    revision = resolution.resolved_model_revision
    allow_network = bool(source.allow_downloads)
    config_payload = fetch_file(model_id, revision, "config.json", allow_network)
    if config_payload is None:
        raise RuntimeQualificationError("model config is unavailable for runtime qualification")
    try:
        model_config = json.loads(config_payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeQualificationError("model config is not valid JSON") from exc
    if not isinstance(model_config, Mapping):
        raise RuntimeQualificationError("model config must be a JSON object")

    auto_map = _auto_map_values(model_config)
    remote_files = _remote_files(auto_map)
    required_imports: set[str] = set()
    inspected_files: list[str] = []
    uninspectable_files: list[str] = []
    warnings: list[str] = []
    for filename in remote_files:
        payload = fetch_file(model_id, revision, filename, allow_network)
        if payload is None:
            warnings.append(f"remote source unavailable for static inspection: {filename}")
            uninspectable_files.append(filename)
            continue
        inspected_files.append(filename)
        required_imports.update(_source_imports(payload))

    requirements = _requirements(
        fetch_file(model_id, revision, "requirements.txt", allow_network)
    )
    missing_packages = tuple(sorted(item for item in requirements if _distribution_missing(item)))
    missing_imports = tuple(sorted(item for item in required_imports if _import_missing(item)))
    remote_code_required = bool(auto_map)
    trust_remote_code = bool(config.trust_remote_code)
    if remote_code_required and not trust_remote_code:
        warnings.append("model requires trusted remote code but execution permission is disabled")

    try:
        transformers_version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        transformers_version = None
    declared_transformers_version = model_config.get("transformers_version")
    if not isinstance(declared_transformers_version, str):
        declared_transformers_version = None
    elif transformers_version and declared_transformers_version != transformers_version:
        warnings.append(
            "model metadata names a different Transformers version; "
            "an isolated smoke run is required"
        )

    install_plan = tuple(sorted(set(missing_packages)))
    incompatible = bool(
        missing_packages
        or missing_imports
        or uninspectable_files
        or transformers_version is None
        or (remote_code_required and not trust_remote_code)
    )
    identity_payload = {
        "model_id": model_id,
        "revision": revision,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "transformers": transformers_version,
        "requirements": requirements,
        "required_imports": tuple(sorted(required_imports)),
    }
    return RuntimeQualification(
        status="incompatible" if incompatible else "ready",
        environment_id=f"runtime:{stable_digest(identity_payload)}",
        model_id=model_id,
        resolved_revision=revision,
        remote_code_required=remote_code_required,
        trust_remote_code=trust_remote_code,
        transformers_version=transformers_version,
        declared_transformers_version=declared_transformers_version,
        requirements=requirements,
        inspected_remote_files=tuple(inspected_files),
        uninspectable_remote_files=tuple(uninspectable_files),
        required_imports=tuple(sorted(required_imports)),
        missing_packages=missing_packages,
        missing_imports=missing_imports,
        warnings=tuple(sorted(warnings)),
        install_plan=install_plan,
    )
