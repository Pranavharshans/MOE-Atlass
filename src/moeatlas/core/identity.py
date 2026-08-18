"""Deterministic, portable identity helpers for MoEAtlas schemas."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping
from typing import Any

_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:/")
_URI_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_COMPONENT_KEY = re.compile(r"^component:([0-9a-f]{64})$")
_TOKEN_KEY = re.compile(r"^token:([0-9a-f]{64})$")
_TOKEN_PHASES = frozenset({"prefill", "decode"})


def canonical_identifier(value: str, *, field_name: str = "identifier") -> str:
    """Normalize a logical identifier and reject machine-specific paths.

    Identifiers are normalized to NFC Unicode, use forward slashes, and may
    contain nested names such as ``org/model``. Absolute POSIX/Windows paths,
    URI schemes, traversal segments, control characters, and whitespace are
    rejected so the result is portable between machines.
    """

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string, got {type(value).__name__}")

    normalized = unicodedata.normalize("NFC", value.strip()).replace("\\", "/")
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    if normalized.startswith(("/", "~", "//")) or _WINDOWS_ABSOLUTE.match(normalized):
        raise ValueError(
            f"{field_name} must be a portable logical identifier, not an absolute path"
        )
    if _URI_SCHEME.match(normalized):
        raise ValueError(f"{field_name} must not contain a URI scheme")
    if any(character.isspace() or ord(character) < 32 for character in normalized):
        raise ValueError(f"{field_name} must not contain whitespace or control characters")
    if any(part in {".", ".."} for part in normalized.split("/")):
        raise ValueError(f"{field_name} must not contain path traversal segments")
    return normalized


def validate_stable_identifier(value: str, *, field_name: str) -> str:
    """Validate an already-canonical schema identifier without changing it."""

    normalized = canonical_identifier(value, field_name=field_name)
    if normalized != value:
        raise ValueError(f"{field_name} must already be in canonical form")
    return value


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible data canonically for hashing.

    Key sorting, fixed separators, NFC normalization at the identifier
    boundary, and disabled NaN values keep the digest independent of Python's
    process hash seed and dictionary insertion order.
    """

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise TypeError("identity input must contain only JSON-serializable values") from exc


def stable_digest(value: Any) -> str:
    """Return a full SHA-256 digest of canonical JSON data."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def make_config_hash(config: Mapping[str, Any]) -> str:
    """Return the canonical config hash stored in a model manifest."""

    if not isinstance(config, Mapping):
        raise TypeError(f"config must be a mapping, got {type(config).__name__}")
    return f"sha256:{stable_digest(dict(config))}"


def make_model_key(identifier: str, revision: str) -> str:
    """Build a readable, portable model key from a logical ID and revision."""

    model_id = canonical_identifier(identifier, field_name="model identifier")
    model_revision = canonical_identifier(revision, field_name="model revision")
    if "@" in model_id:
        raise ValueError(
            "model identifier must not contain '@'; it is reserved as the key separator"
        )
    if "@" in model_revision:
        raise ValueError("model revision must not contain '@'; it is reserved as the key separator")
    return f"model:{model_id}@{model_revision}"


def parse_model_key(model_key: str) -> tuple[str, str]:
    """Parse and validate ``model:<portable-id>@<revision>``.

    Hugging Face-style slashes and revision refs such as ``refs/main`` are
    preserved. ``@`` is reserved for the single separator so malformed or
    ambiguous keys cannot silently bind a manifest to the wrong revision.
    """

    if not isinstance(model_key, str):
        raise TypeError(f"model_key must be a string, got {type(model_key).__name__}")
    if not model_key.startswith("model:"):
        raise ValueError("model_key must use the format model:<portable-id>@<revision>")
    payload = model_key.removeprefix("model:")
    if payload.count("@") != 1:
        raise ValueError(
            "model_key must use the format model:<portable-id>@<revision> with "
            "exactly one '@' separator between the model ID and revision"
        )
    identifier, revision = payload.split("@", 1)
    expected = make_model_key(identifier, revision)
    if expected != model_key:
        raise ValueError("model_key must be in canonical model:<portable-id>@<revision> form")
    return identifier, revision


def make_component_key(
    model_key: str,
    kind: str,
    module_path: str,
    *,
    layer_index: int | None = None,
    expert_index: int | None = None,
) -> str:
    """Build a deterministic component key from semantic component identity.

    The key is a digest rather than a path so it remains stable without
    exposing or depending on an absolute local filesystem path.
    """

    parse_model_key(model_key)
    stable_model_key = model_key
    stable_kind = canonical_identifier(kind, field_name="component kind")
    stable_module_path = canonical_identifier(module_path, field_name="module path")
    for name, index in (("layer_index", layer_index), ("expert_index", expert_index)):
        if index is not None and (isinstance(index, bool) or not isinstance(index, int)):
            raise TypeError(f"{name} must be an integer or None")
        if index is not None and index < 0:
            raise ValueError(f"{name} must be non-negative")

    payload = {
        "expert_index": expert_index,
        "kind": stable_kind,
        "layer_index": layer_index,
        "model_key": stable_model_key,
        "module_path": stable_module_path,
    }
    return f"component:{stable_digest(payload)}"


def parse_component_key(component_key: str) -> str:
    """Validate and return the digest from a canonical component key.

    Component keys are intentionally opaque, lowercase SHA-256 identities.
    Requiring the complete ``component:<64 lowercase hex>`` shape prevents a
    probe target from carrying an ambiguous human label that cannot be linked
    back to a canonical component manifest.
    """

    if not isinstance(component_key, str):
        raise TypeError(f"component_key must be a string, got {type(component_key).__name__}")
    match = _COMPONENT_KEY.fullmatch(component_key)
    if match is None:
        raise ValueError("component_key must use the canonical component:<64 lowercase hex> form")
    return match.group(1)


def make_token_key(
    run_key: str,
    sequence_id: str,
    token_pos: int,
    token_id: int,
    phase: str,
) -> str:
    """Build a portable token identity from stable token coordinates.

    Token text is deliberately excluded: decoded presentation can vary while
    the run, sequence, position, ID, and phase still identify the same token.
    The helper accepts only strict Python types so numeric strings and booleans
    cannot silently alter an event identity.
    """

    stable_run_key = validate_stable_identifier(run_key, field_name="run_key")
    stable_sequence_id = validate_stable_identifier(sequence_id, field_name="sequence_id")
    for name, value in (("token_pos", token_pos), ("token_id", token_id)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    if not isinstance(phase, str):
        raise TypeError(f"phase must be a string, got {type(phase).__name__}")
    if phase not in _TOKEN_PHASES:
        raise ValueError("phase must be one of: prefill, decode")
    payload = {
        "phase": phase,
        "run_key": stable_run_key,
        "sequence_id": stable_sequence_id,
        "token_id": token_id,
        "token_pos": token_pos,
    }
    return f"token:{stable_digest(payload)}"


def parse_token_key(token_key: str) -> str:
    """Validate and return the digest from a canonical token key."""

    if not isinstance(token_key, str):
        raise TypeError(f"token_key must be a string, got {type(token_key).__name__}")
    match = _TOKEN_KEY.fullmatch(token_key)
    if match is None:
        raise ValueError("token_key must use the canonical token:<64 lowercase hex> form")
    return match.group(1)
