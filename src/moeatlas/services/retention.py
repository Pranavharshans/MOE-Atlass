"""Retention evaluation over the workspace run registry (PRD §17).

MoEAtlas storage is append-only and immutable by design, so retention here
is **evaluation, not deletion**: a policy classifies registered runs into
retained and expired sets, and the report is the artifact callers act on.
Nothing in this module rewrites a catalog or removes shard data; dropping
registry entries or exporting-then-expiring remains an explicit caller
decision recorded elsewhere.

Timestamps are canonical UTC ISO-8601 strings and compare
lexicographically, which is exact for that form. Entries without a
registration timestamp sort as oldest. Ordering everywhere is
deterministic: registration timestamp first, then run key.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from moeatlas.store.catalog import RunRegistryEntry

RETENTION_SCHEMA_VERSION = "1.0"
"""Schema version of the retention contracts."""

_RETENTION_REPORT_ARTIFACT_TYPE = "moeatlas.retention_report"

_ERROR_STAGES = frozenset({"contract"})

_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class RetentionError(RuntimeError):
    """Safe fixed-stage failure for retention handling."""

    def __init__(
        self, stage: str, message: str | None = None, *, cause: BaseException | None = None
    ) -> None:
        if stage not in _ERROR_STAGES:
            raise ValueError("retention error stage is not supported")
        self.stage = stage
        text = f"retention failed at {stage}"
        if message:
            text = f"{text}: {message}"
        super().__init__(text)
        if cause is not None:
            self.__cause__ = cause


@dataclass(frozen=True, slots=True)
class RetentionPolicy:
    """Immutable bounds on how many runs stay listed, and how old they may be.

    ``max_runs`` keeps only the newest registered entries; ``before``
    expires every entry registered strictly earlier than the boundary. At
    least one bound must be set — an empty policy would silently claim to
    retain everything while evaluating nothing.
    """

    max_runs: int | None = None
    before: str | None = None

    def __post_init__(self) -> None:
        if self.max_runs is not None:
            if type(self.max_runs) is not int or isinstance(self.max_runs, bool):
                raise TypeError("max_runs must be an integer or None")
            if self.max_runs <= 0:
                raise RetentionError("contract", "max_runs must be strictly positive")
        if self.before is not None:
            if type(self.before) is not str:
                raise TypeError("before must be a canonical UTC timestamp or None")
            if _TIMESTAMP.fullmatch(self.before) is None:
                raise RetentionError(
                    "contract",
                    "before must use canonical YYYY-MM-DDTHH:MM:SSZ form",
                )
        if self.max_runs is None and self.before is None:
            raise RetentionError("contract", "retention requires at least one bound")


def _entry_order_key(entry: RunRegistryEntry) -> tuple[str, str]:
    stamp = entry.registered_at if entry.registered_at is not None else ""
    return (stamp, entry.run_key)


@dataclass(frozen=True, slots=True)
class RetentionReport:
    """Deterministic classification of registered runs under one policy.

    ``retained_keys`` follows registration-timestamp-then-run-key order
    and ``expired_keys`` is sorted; the two are disjoint and
    ``evaluated_count`` equals their total length. The report never
    mutates anything — enforcement stays an explicit caller decision.
    """

    schema_version: str
    evaluated_count: int
    retained_keys: tuple[str, ...]
    expired_keys: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": _RETENTION_REPORT_ARTIFACT_TYPE,
            "schema_version": self.schema_version,
            "evaluated_count": self.evaluated_count,
            "retained_keys": list(self.retained_keys),
            "expired_keys": list(self.expired_keys),
        }

    def to_json(self) -> str:
        """Serialize this report with deterministic key order and no whitespace."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> RetentionReport:
        """Validate one canonical JSON document into an exact report value."""

        try:
            document = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("retention report document is not valid JSON") from exc
        if type(document) is not dict:
            raise ValueError("retention report document must be a JSON object")
        if (
            document.get("artifact_type") != _RETENTION_REPORT_ARTIFACT_TYPE
            or document.get("schema_version") != RETENTION_SCHEMA_VERSION
        ):
            raise ValueError("document is not a retention report artifact")
        try:
            return cls(
                schema_version=document["schema_version"],
                evaluated_count=document["evaluated_count"],
                retained_keys=tuple(document["retained_keys"]),
                expired_keys=tuple(document["expired_keys"]),
            )
        except KeyError as exc:
            raise ValueError("retention report document is missing fields") from exc


def evaluate_retention(
    entries: tuple[RunRegistryEntry, ...], policy: RetentionPolicy
) -> RetentionReport:
    """Classify registry entries into retained and expired sets."""

    if type(entries) is not tuple:
        raise TypeError("entries must be a tuple of run registry entries")
    for entry in entries:
        if not isinstance(entry, RunRegistryEntry):
            raise TypeError("entries entries must be RunRegistryEntry values")
    if type(policy) is not RetentionPolicy:
        raise TypeError("policy must be a RetentionPolicy")
    ordered = sorted(entries, key=_entry_order_key)
    expired: set[str] = set()
    if policy.before is not None:
        for entry in ordered:
            stamp = entry.registered_at
            if stamp is not None and stamp < policy.before:
                expired.add(entry.run_key)
    if policy.max_runs is not None:
        survivors = [entry for entry in ordered if entry.run_key not in expired]
        surplus: list[RunRegistryEntry] = []
        if len(survivors) > policy.max_runs:
            surplus = survivors[: len(survivors) - policy.max_runs]
        for entry in surplus:
            expired.add(entry.run_key)
    retained_keys = tuple(
        entry.run_key for entry in ordered if entry.run_key not in expired
    )
    expired_keys = tuple(sorted(expired))
    return RetentionReport(
        schema_version=RETENTION_SCHEMA_VERSION,
        evaluated_count=len(ordered),
        retained_keys=retained_keys,
        expired_keys=expired_keys,
    )
