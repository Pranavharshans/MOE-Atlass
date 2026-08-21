"""Versioned adapter/plugin registry with provenance, policy, and isolation.

The registry is the single model-neutral surface behind ``adapters list``:
built-in adapters and third-party plugins published under the
``moeatlas.adapters`` entry-point group appear through one contract.
Discovery is metadata-only — a plugin's module is imported just enough to
read its :class:`AdapterDescriptor`, never to load a model. Every failure is
isolated into an explicitly reported row (fixed reason strings) instead of
breaking the listing; name collisions are resolved deterministically
(built-ins win, then lexicographic entry-point value) and reported as
suppressed rows; trust and enable/disable policy is applied per record.

The registry stays model-free: no storage reads, clocks, randomness,
network, or family knowledge beyond what adapters publish about themselves.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.metadata import entry_points as system_entry_points
from typing import Any

from .contracts import AdapterDescriptor

ADAPTER_REGISTRY_SCHEMA_VERSION = "1.0"
"""Schema version of the adapter-registry contracts."""

ENTRY_POINT_GROUP = "moeatlas.adapters"
"""Entry-point group third-party adapter plugins publish under."""

_REGISTRY_ARTIFACT_TYPE = "moeatlas.adapter_registry"

_ERROR_STAGES = frozenset({"contract", "serialization"})

_SOURCES = frozenset({"builtin", "entry_point"})
_STATUSES = frozenset({"enabled", "disabled"})

_FAILURE_REASONS = (
    "descriptor contract violated",
    "entry name mismatch",
    "load failed",
    "missing descriptor",
)


class AdapterRegistryError(RuntimeError):
    """Safe fixed-stage failure for registry collection or serialization."""

    def __init__(
        self, stage: str, message: str | None = None, *, cause: BaseException | None = None
    ) -> None:
        if stage not in _ERROR_STAGES:
            raise ValueError("adapter registry error stage is not supported")
        self.stage = stage
        text = f"adapter registry failed at {stage}"
        if message:
            text = f"{text}: {message}"
        super().__init__(text)
        if cause is not None:
            self.__cause__ = cause


def _trimmed(value: object, field_name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be non-empty and trimmed")
    return value


def _sorted_unique(values: object, field_name: str) -> tuple[str, ...]:
    if type(values) is not tuple:
        raise TypeError(f"{field_name} must be a tuple of strings")
    for entry in values:
        _trimmed(entry, f"{field_name} entries")
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} entries must be unique")
    ordered = tuple(sorted(values))
    if values != ordered:
        raise ValueError(f"{field_name} entries must be sorted lexicographically")
    return values  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class AdapterPluginRecord:
    """Portable identity and provenance for one registered adapter plugin.

    ``source`` distinguishes shipped built-ins from entry-point plugins;
    ``distribution`` records the publishing package when one exists.
    """

    name: str
    version: str
    source: str
    distribution: str | None
    location: str
    architecture_families: tuple[str, ...]
    compatibility_notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _trimmed(self.name, "name")
        _trimmed(self.version, "version")
        if type(self.source) is not str:
            raise TypeError("source must be a string")
        if self.source not in _SOURCES:
            raise ValueError("source must be builtin or entry_point")
        if self.distribution is not None:
            _trimmed(self.distribution, "distribution")
        _trimmed(self.location, "location")
        families = _sorted_unique(
            self.architecture_families, "architecture_families"
        )
        if not families:
            raise ValueError("architecture_families must be non-empty")
        object.__setattr__(self, "architecture_families", families)
        notes = _sorted_unique(self.compatibility_notes, "compatibility_notes")
        object.__setattr__(self, "compatibility_notes", notes)

    @classmethod
    def from_descriptor(
        cls, descriptor: AdapterDescriptor, *, source: str, distribution: str | None,
        location: str,
    ) -> AdapterPluginRecord:
        """Project one validated descriptor into a registry record."""

        if type(descriptor) is not AdapterDescriptor:
            raise TypeError("descriptor must be an AdapterDescriptor")
        return cls(
            name=descriptor.name,
            version=descriptor.version,
            source=source,
            distribution=distribution,
            location=location,
            architecture_families=tuple(descriptor.architecture_families),
            compatibility_notes=tuple(descriptor.compatibility_notes),
        )


@dataclass(frozen=True, slots=True)
class AdapterRegistryEntry:
    """One registry record plus its policy-resolved status."""

    record: AdapterPluginRecord
    status: str

    def __post_init__(self) -> None:
        if type(self.record) is not AdapterPluginRecord:
            raise TypeError("record must be an AdapterPluginRecord")
        if type(self.status) is not str:
            raise TypeError("status must be a string")
        if self.status not in _STATUSES:
            raise ValueError("status must be enabled or disabled")


@dataclass(frozen=True, slots=True)
class AdapterRegistryPolicy:
    """Trust and enable/disable policy applied at listing time.

    ``trusted_sources`` restricts which plugin origins may be enabled;
    ``enabled_names`` (``None`` meaning everyone) allow-lists specific
    plugins; ``disabled_names`` force-disables regardless of everything
    else. A name in both lists is a construction error.
    """

    trusted_sources: tuple[str, ...] = ("builtin", "entry_point")
    enabled_names: tuple[str, ...] | None = None
    disabled_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        sources = _sorted_unique(self.trusted_sources, "trusted_sources")
        if any(source not in _SOURCES for source in sources):
            raise ValueError("trusted_sources holds unknown source names")
        object.__setattr__(self, "trusted_sources", sources)
        if self.enabled_names is not None:
            object.__setattr__(
                self,
                "enabled_names",
                _sorted_unique(self.enabled_names, "enabled_names"),
            )
        disabled = _sorted_unique(self.disabled_names, "disabled_names")
        object.__setattr__(self, "disabled_names", disabled)
        if self.enabled_names is not None:
            overlap = set(self.enabled_names) & set(disabled)
            if overlap:
                raise ValueError(
                    "enabled_names and disabled_names must not overlap"
                )

    def status_for(self, record: AdapterPluginRecord) -> str:
        """Resolve one record's status deterministically."""

        if type(record) is not AdapterPluginRecord:
            raise TypeError("record must be an AdapterPluginRecord")
        if record.source not in self.trusted_sources:
            return "disabled"
        if record.name in self.disabled_names:
            return "disabled"
        if self.enabled_names is not None and record.name not in self.enabled_names:
            return "disabled"
        return "enabled"


def builtin_adapter_records() -> tuple[AdapterPluginRecord, ...]:
    """Records for the adapters shipped inside this package."""

    from .mixtral import MixtralStaticAdapter
    from .qwen3_5_moe import Qwen3_5MoeStaticAdapter
    from .qwen3_moe import Qwen3MoeStaticAdapter

    records = [
        AdapterPluginRecord.from_descriptor(
            adapter().descriptor,
            source="builtin",
            distribution="moeatlas",
            location=f"moeatlas.adapters.{module}",
        )
        for module, adapter in (
            ("mixtral", MixtralStaticAdapter),
            ("qwen3_5_moe", Qwen3_5MoeStaticAdapter),
            ("qwen3_moe", Qwen3MoeStaticAdapter),
        )
    ]
    return tuple(sorted(records, key=lambda record: (record.name, record.location)))


def discover_entry_point_records(
    entry_points: Any,
) -> tuple[tuple[AdapterPluginRecord, ...], tuple[tuple[str, str], ...]]:
    """Load candidate entry points into records with isolated failures.

    Each failure becomes a ``(entry_point_value, fixed_reason)`` row; no
    exception from one plugin ever reaches the caller.
    """

    records: list[AdapterPluginRecord] = []
    failures: list[tuple[str, str]] = []
    for entry_point in entry_points:
        try:
            value = _trimmed(entry_point.value, "entry point value")
        except Exception:
            continue  # unidentifiable candidates cannot even be reported safely
        try:
            loaded = entry_point.load()
        except Exception:
            failures.append((value, "load failed"))
            continue
        descriptor = getattr(loaded, "descriptor", None)
        if descriptor is None:
            failures.append((value, "missing descriptor"))
            continue
        if type(descriptor) is not AdapterDescriptor:
            failures.append((value, "descriptor contract violated"))
            continue
        if descriptor.name != entry_point.name:
            failures.append((value, "entry name mismatch"))
            continue
        distribution = getattr(entry_point, "dist", None)
        dist_name = getattr(distribution, "name", None)
        if dist_name is not None:
            dist_name = str(dist_name)
        records.append(
            AdapterPluginRecord.from_descriptor(
                descriptor,
                source="entry_point",
                distribution=dist_name,
                location=value,
            )
        )
    return (
        tuple(sorted(records, key=lambda record: (record.name, record.location))),
        tuple(sorted(failures)),
    )


def apply_registry_policy(
    records: Any, policy: AdapterRegistryPolicy
) -> tuple[AdapterRegistryEntry, ...]:
    """Attach policy statuses to records and order them deterministically."""

    if type(policy) is not AdapterRegistryPolicy:
        raise TypeError("policy must be an AdapterRegistryPolicy")
    if type(records) is not tuple:
        raise TypeError("records must be a tuple of AdapterPluginRecord values")
    for record in records:
        if type(record) is not AdapterPluginRecord:
            raise TypeError("records must hold only AdapterPluginRecord values")
    entries = [
        AdapterRegistryEntry(record=record, status=policy.status_for(record))
        for record in records
    ]
    entries.sort(
        key=lambda entry: (
            entry.record.name,
            entry.record.version,
            entry.record.source,
            entry.record.location,
        )
    )
    return tuple(entries)


def collect_adapter_registry(
    *,
    entry_points: Any = None,
    policy: AdapterRegistryPolicy | None = None,
    builtin_records: tuple[AdapterPluginRecord, ...] | None = None,
) -> AdapterRegistryReport:
    """Collect built-ins and plugins into one deterministic registry report.

    ``entry_points`` replaces environment discovery for tests; ``None``
    enumerates the real ``moeatlas.adapters`` group. Name collisions keep
    exactly one record (built-ins first, then lexicographic entry-point
    value) and report every suppressed loser.
    """

    if policy is None:
        policy = AdapterRegistryPolicy()
    if builtin_records is None:
        builtin_records = builtin_adapter_records()
    if entry_points is None:
        try:
            discovered = system_entry_points(group=ENTRY_POINT_GROUP)
        except Exception as exc:
            raise AdapterRegistryError("contract", cause=exc) from exc
        plugin_records, failures = discover_entry_point_records(discovered)
    else:
        if type(entry_points) is not tuple:
            raise TypeError("entry_points must be a tuple or None")
        plugin_records, failures = discover_entry_point_records(entry_points)

    candidates = sorted(
        (*builtin_records, *plugin_records),
        key=lambda record: (
            0 if record.source == "builtin" else 1,
            record.location,
        ),
    )
    kept_by_name: dict[str, AdapterPluginRecord] = {}
    collisions: list[tuple[str, str, str]] = []
    for record in candidates:
        winner = kept_by_name.get(record.name)
        if winner is None:
            kept_by_name[record.name] = record
        else:
            collisions.append((record.name, winner.location, record.location))
    collisions.sort(key=lambda row: (row[0], row[2]))
    entries = apply_registry_policy(tuple(kept_by_name.values()), policy)
    return AdapterRegistryReport(
        schema_version=ADAPTER_REGISTRY_SCHEMA_VERSION,
        entries=entries,
        collisions=tuple(collisions),
        failures=failures,
    )


def match_adapters_for_family(
    entries: Any, *, family: str
) -> tuple[AdapterRegistryEntry, ...]:
    """Capability negotiation: enabled records serving one architecture family."""

    _trimmed(family, "family")
    if type(entries) is not tuple:
        raise TypeError("entries must be a tuple of AdapterRegistryEntry values")
    matches = []
    for entry in entries:
        if type(entry) is not AdapterRegistryEntry:
            raise TypeError("entries must hold only AdapterRegistryEntry values")
        if entry.status != "enabled":
            continue
        if family in entry.record.architecture_families:
            matches.append(entry)
    return tuple(sorted(matches, key=lambda entry: (entry.record.name, entry.record.version)))


@dataclass(frozen=True, slots=True)
class AdapterRegistryReport:
    """The canonical registry publication behind ``adapters list``.

    ``entries`` lists every surviving record with its policy status;
    ``collisions`` reports suppressed duplicate names as
    ``(name, kept_location, suppressed_location)`` rows and ``failures``
    reports isolated plugin-load problems as
    ``(entry_point_value, fixed_reason)`` rows.
    """

    schema_version: str
    entries: tuple[AdapterRegistryEntry, ...]
    collisions: tuple[tuple[str, str, str], ...]
    failures: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if self.schema_version != ADAPTER_REGISTRY_SCHEMA_VERSION:
            raise ValueError("schema_version is not supported by this registry")
        if type(self.entries) is not tuple:
            raise TypeError("entries must be a tuple of AdapterRegistryEntry values")
        for entry in self.entries:
            if type(entry) is not AdapterRegistryEntry:
                raise TypeError("entries must hold only AdapterRegistryEntry values")
        if type(self.collisions) is not tuple:
            raise TypeError("collisions must be a tuple of rows")
        for row in self.collisions:
            if type(row) is not tuple or len(row) != 3:
                raise TypeError("collisions rows must be (name, kept, suppressed)")
        if type(self.failures) is not tuple:
            raise TypeError("failures must be a tuple of rows")
        for row in self.failures:
            if type(row) is not tuple or len(row) != 2:
                raise TypeError("failures rows must be (value, reason)")
            if row[1] not in _FAILURE_REASONS:
                raise ValueError("failures rows must use fixed reason vocabulary")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": _REGISTRY_ARTIFACT_TYPE,
            "schema_version": self.schema_version,
            "entries": [
                {
                    "name": entry.record.name,
                    "version": entry.record.version,
                    "source": entry.record.source,
                    "distribution": entry.record.distribution,
                    "location": entry.record.location,
                    "architecture_families": list(
                        entry.record.architecture_families
                    ),
                    "compatibility_notes": list(
                        entry.record.compatibility_notes
                    ),
                    "status": entry.status,
                }
                for entry in self.entries
            ],
            "collisions": [list(row) for row in self.collisions],
            "failures": [list(row) for row in self.failures],
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
    def from_json(cls, payload: str | bytes | bytearray) -> AdapterRegistryReport:
        """Validate one canonical JSON document into an exact report value."""

        try:
            document = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AdapterRegistryError(
                "serialization", cause=exc
            ) from exc
        if type(document) is not dict:
            raise AdapterRegistryError("serialization")
        if (
            document.get("artifact_type") != _REGISTRY_ARTIFACT_TYPE
            or document.get("schema_version") != ADAPTER_REGISTRY_SCHEMA_VERSION
        ):
            raise AdapterRegistryError("serialization")
        try:
            return cls(
                schema_version=document["schema_version"],
                entries=tuple(
                    AdapterRegistryEntry(
                        record=AdapterPluginRecord(
                            name=item["name"],
                            version=item["version"],
                            source=item["source"],
                            distribution=item["distribution"],
                            location=item["location"],
                            architecture_families=tuple(item["architecture_families"]),
                            compatibility_notes=tuple(item["compatibility_notes"]),
                        ),
                        status=item["status"],
                    )
                    for item in document["entries"]
                ),
                collisions=tuple(tuple(row) for row in document["collisions"]),
                failures=tuple(tuple(row) for row in document["failures"]),
            )
        except KeyError as exc:
            raise AdapterRegistryError("serialization") from exc
        except TypeError as exc:
            raise AdapterRegistryError("serialization") from exc
        except ValueError as exc:
            raise AdapterRegistryError("serialization") from exc


__all__ = [
    "ADAPTER_REGISTRY_SCHEMA_VERSION",
    "ENTRY_POINT_GROUP",
    "AdapterPluginRecord",
    "AdapterRegistryEntry",
    "AdapterRegistryError",
    "AdapterRegistryPolicy",
    "AdapterRegistryReport",
    "apply_registry_policy",
    "builtin_adapter_records",
    "collect_adapter_registry",
    "discover_entry_point_records",
    "match_adapters_for_family",
]
