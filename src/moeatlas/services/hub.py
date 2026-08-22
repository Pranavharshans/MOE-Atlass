"""Small, bounded Hugging Face Hub metadata lookups for the local UI.

The model and dataset loaders intentionally keep network access behind an
explicit runtime policy.  This module is narrower: it serves public search
suggestions after a user asks for them in the UI.  It never accepts a caller
supplied URL, never sends an access token, and drops unbounded Hub fields before
they cross the server boundary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

HubKind = Literal["model", "dataset"]

HUB_API_ROOT = "https://huggingface.co/api"
HUB_SEARCH_SCHEMA_VERSION = "1.0"

_MAX_QUERY_LENGTH = 200
_MAX_RESULTS = 10
_MAX_RESPONSE_BYTES = 2_000_000
_MAX_TAGS = 12
_MAX_TAG_LENGTH = 120
_MAX_TEXT_LENGTH = 500


class HubSearchError(RuntimeError):
    """Fixed public failure for a bounded Hub search."""

    def __init__(self, reason: str = "unavailable") -> None:
        if reason not in {"unavailable", "invalid_response"}:
            raise ValueError("unsupported Hub search failure reason")
        self.reason = reason
        super().__init__("Hugging Face search is temporarily unavailable")


@dataclass(frozen=True, slots=True)
class HubSearchEntry:
    """Safe, intentionally small subset of one public Hub search result."""

    identifier: str
    kind: HubKind
    author: str | None = None
    downloads: int | None = None
    likes: int | None = None
    pipeline_tag: str | None = None
    library_name: str | None = None
    tags: tuple[str, ...] = ()
    last_modified: str | None = None


def _bounded_text(value: object, *, maximum: int = _MAX_TEXT_LENGTH) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or len(value) > maximum or any(ord(char) < 32 for char in value):
        return None
    return value


def _bounded_non_negative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _normalize_kind(kind: str) -> HubKind:
    if kind == "model":
        return "model"
    if kind == "dataset":
        return "dataset"
    raise ValueError("Hub search kind must be model or dataset")


def _validate_query(query: str) -> str:
    if not isinstance(query, str):
        raise TypeError("Hub search query must be a string")
    normalized = query.strip()
    if len(normalized) < 2 or len(normalized) > _MAX_QUERY_LENGTH:
        raise ValueError("Hub search query must contain between 2 and 200 characters")
    if any(ord(char) < 32 for char in normalized):
        raise ValueError("Hub search query must not contain control characters")
    return normalized


def _validate_limit(limit: int) -> int:
    if type(limit) is not int or isinstance(limit, bool):
        raise TypeError("Hub search limit must be an integer")
    if limit < 1 or limit > _MAX_RESULTS:
        raise ValueError(f"Hub search limit must be between 1 and {_MAX_RESULTS}")
    return limit


def _search_url(kind: HubKind, query: str, limit: int) -> str:
    resource = "models" if kind == "model" else "datasets"
    query_string = urlencode(
        {"search": query, "limit": limit, "full": "false", "config": "false"}
    )
    return f"{HUB_API_ROOT}/{resource}?{query_string}"


def _fetch_json(url: str) -> object:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "MoEAtlas/0.1 public-search",
        },
        method="GET",
    )
    try:
        with urlopen(request, timeout=5.0) as response:  # noqa: S310 - fixed HTTPS host above
            payload = response.read(_MAX_RESPONSE_BYTES + 1)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise HubSearchError() from exc
    if len(payload) > _MAX_RESPONSE_BYTES:
        raise HubSearchError("invalid_response")
    try:
        return json.loads(payload)
    except (TypeError, ValueError) as exc:
        raise HubSearchError("invalid_response") from exc


def _normalize_entry(raw: object, kind: HubKind) -> HubSearchEntry | None:
    if not isinstance(raw, dict):
        return None
    identifier = raw.get("id") or raw.get("modelId")
    if not isinstance(identifier, str):
        return None
    identifier = identifier.strip()
    if not identifier or len(identifier) > _MAX_TEXT_LENGTH:
        return None
    if any(ord(char) < 32 for char in identifier):
        return None
    tags: list[str] = []
    raw_tags = raw.get("tags")
    if isinstance(raw_tags, list):
        for tag in raw_tags[:_MAX_TAGS]:
            normalized = _bounded_text(tag, maximum=_MAX_TAG_LENGTH)
            if normalized is not None:
                tags.append(normalized)
    return HubSearchEntry(
        identifier=identifier,
        kind=kind,
        author=_bounded_text(raw.get("author"), maximum=200),
        downloads=_bounded_non_negative_int(raw.get("downloads")),
        likes=_bounded_non_negative_int(raw.get("likes")),
        pipeline_tag=_bounded_text(raw.get("pipeline_tag"), maximum=200),
        library_name=_bounded_text(raw.get("library_name"), maximum=200),
        tags=tuple(tags),
        last_modified=_bounded_text(raw.get("lastModified"), maximum=80),
    )


def search_hub(kind: str, query: str, *, limit: int = 6) -> tuple[HubSearchEntry, ...]:
    """Return bounded public suggestions for one user-requested query."""

    normalized_kind = _normalize_kind(kind)
    normalized_query = _validate_query(query)
    normalized_limit = _validate_limit(limit)
    payload = _fetch_json(_search_url(normalized_kind, normalized_query, normalized_limit))
    if not isinstance(payload, list):
        raise HubSearchError("invalid_response")
    entries: list[HubSearchEntry] = []
    seen: set[str] = set()
    for raw in payload[:normalized_limit]:
        entry = _normalize_entry(raw, normalized_kind)
        if entry is None or entry.identifier in seen:
            continue
        seen.add(entry.identifier)
        entries.append(entry)
    return tuple(entries)


__all__ = [
    "HUB_SEARCH_SCHEMA_VERSION",
    "HubKind",
    "HubSearchEntry",
    "HubSearchError",
    "search_hub",
]
