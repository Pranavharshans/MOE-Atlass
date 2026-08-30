"""Contract tests for the versioned adapter plugin registry."""

from __future__ import annotations

import pytest

from moeatlas.adapters import (
    ADAPTER_REGISTRY_SCHEMA_VERSION,
    ENTRY_POINT_GROUP,
    AdapterDescriptor,
    AdapterPluginRecord,
    AdapterRegistryError,
    AdapterRegistryPolicy,
    AdapterRegistryReport,
    apply_registry_policy,
    builtin_adapter_records,
    collect_adapter_registry,
    discover_entry_point_records,
    match_adapters_for_family,
)

# ---------------------------------------------------------------------------
# Fixtures


class _FakeDistribution:
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name


class _FakeEntryPoint:
    def __init__(self, name: str, value: str, dist: object, payload: object) -> None:
        self._name = name
        self.value = value
        self.dist = dist
        self._payload = payload

    @property
    def name(self) -> str:
        return self._name

    def load(self) -> object:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _Plugin:
    def __init__(self, descriptor: AdapterDescriptor) -> None:
        self.descriptor = descriptor


def _descriptor(name: str = "acme-moe", version: str = "2.0") -> AdapterDescriptor:
    return AdapterDescriptor(
        name=name,
        version=version,
        architecture_families=("acme_moe",),
        compatibility_notes=("synthetic test adapter",),
    )


def _entry_point(
    name: str = "acme",
    *,
    value: str = "acme_pkg.plugin:ADAPTER",
    dist_name: str | None = "acme-pkg",
    payload: object | None = None,
) -> _FakeEntryPoint:
    return _FakeEntryPoint(
        name,
        value,
        None if dist_name is None else _FakeDistribution(dist_name),
        _Plugin(_descriptor(name)) if payload is None else payload,
    )


# ---------------------------------------------------------------------------
# Public surface


def test_surface_is_pinned() -> None:
    assert ADAPTER_REGISTRY_SCHEMA_VERSION == "1.0"
    assert ENTRY_POINT_GROUP == "moeatlas.adapters"
    assert str(AdapterRegistryError("contract")) == "adapter registry failed at contract"
    with pytest.raises(ValueError):
        AdapterRegistryError("cancelled")


def test_builtin_records_cover_the_four_shipped_adapters() -> None:
    records = builtin_adapter_records()
    names = [record.name for record in records]
    assert names == sorted(names)
    assert set(names) == {
        "huggingface-mixtral-static",
        "huggingface-qwen3-moe-static",
        "huggingface-qwen3.5-moe-static",
        "huggingface-qwen4-exp-static",
    }
    for record in records:
        assert record.source == "builtin"
        assert record.distribution == "moeatlas"
        assert record.location.startswith("moeatlas.adapters.")
        assert record.architecture_families


def test_plugin_records_are_strict() -> None:
    record = AdapterPluginRecord(
        name="acme",
        version="1.0",
        source="entry_point",
        distribution="acme-pkg",
        location="acme_pkg.plugin:ADAPTER",
        architecture_families=("acme_moe",),
    )
    assert record.compatibility_notes == ()
    # Container-type violations are TypeError.
    with pytest.raises(TypeError):
        AdapterPluginRecord(
            name=["acme"],
            version="1.0",
            source="entry_point",
            distribution=None,
            location="loc",
            architecture_families=("acme_moe",),
        )
    with pytest.raises(TypeError):
        AdapterPluginRecord(
            name="acme",
            version="1.0",
            source="entry_point",
            distribution=None,
            location="loc",
            architecture_families=["acme_moe"],
        )
    # Value violations are ValueError.
    with pytest.raises(ValueError):
        AdapterPluginRecord(
            name="",
            version="1.0",
            source="entry_point",
            distribution=None,
            location="loc",
            architecture_families=("acme_moe",),
        )
    with pytest.raises(ValueError):
        AdapterPluginRecord(
            name="acme",
            version="1.0",
            source="pypi",
            distribution=None,
            location="loc",
            architecture_families=("acme_moe",),
        )
    with pytest.raises(ValueError):
        AdapterPluginRecord(
            name="acme",
            version="1.0",
            source="entry_point",
            distribution=None,
            location="loc",
            architecture_families=("b_moe", "a_moe"),
        )
    with pytest.raises(ValueError):
        AdapterPluginRecord(
            name="acme",
            version="1.0",
            source="entry_point",
            distribution=None,
            location="loc",
            architecture_families=("acme_moe", "acme_moe"),
        )


def test_policy_strictness_and_status_resolution() -> None:
    default = AdapterRegistryPolicy()
    record = AdapterPluginRecord(
        name="acme",
        version="1.0",
        source="entry_point",
        distribution="acme-pkg",
        location="loc",
        architecture_families=("acme_moe",),
    )
    assert apply_registry_policy((record,), default)[0].status == "enabled"
    disabled = AdapterRegistryPolicy(disabled_names=("acme",))
    assert apply_registry_policy((record,), disabled)[0].status == "disabled"
    allowlisted = AdapterRegistryPolicy(enabled_names=("other",))
    assert apply_registry_policy((record,), allowlisted)[0].status == "disabled"
    builtin_only = AdapterRegistryPolicy(trusted_sources=("builtin",))
    assert apply_registry_policy((record,), builtin_only)[0].status == "disabled"
    # Policy conflicts and bad vocabulary are ValueErrors; wrong types are TypeErrors.
    with pytest.raises(ValueError):
        AdapterRegistryPolicy(enabled_names=("acme",), disabled_names=("acme",))
    with pytest.raises(ValueError):
        AdapterRegistryPolicy(trusted_sources=("pypi",))
    with pytest.raises(TypeError):
        AdapterRegistryPolicy(trusted_sources=["builtin"])
    with pytest.raises(TypeError):
        apply_registry_policy((record,), "not a policy")
    with pytest.raises(TypeError):
        apply_registry_policy(("not a record",), default)


# ---------------------------------------------------------------------------
# Discovery


def test_entry_point_discovery_builds_provenanced_records() -> None:
    records, failures = discover_entry_point_records((_entry_point("acme"),))
    assert failures == ()
    assert len(records) == 1
    record = records[0]
    assert record.name == "acme"
    assert record.version == "2.0"
    assert record.source == "entry_point"
    assert record.distribution == "acme-pkg"
    assert record.location == "acme_pkg.plugin:ADAPTER"
    assert record.architecture_families == ("acme_moe",)


def test_entry_point_failure_reasons_are_exact() -> None:
    class _NoDescriptor:
        pass

    class _WrongDescriptorType:
        descriptor = "not a descriptor"

    records, failures = discover_entry_point_records(
        (
            _entry_point("boom", value="boom:OBJ", payload=RuntimeError("x")),
            _entry_point("junk", value="junk:OBJ", payload=_NoDescriptor()),
            _entry_point(
                "mismatch",
                value="mismatch:OBJ",
                payload=_Plugin(_descriptor("other")),
            ),
            _entry_point(
                "wrongtype",
                value="wrongtype:OBJ",
                payload=_WrongDescriptorType(),
            ),
        )
    )
    assert records == ()
    assert failures == (
        ("boom:OBJ", "load failed"),
        ("junk:OBJ", "missing descriptor"),
        ("mismatch:OBJ", "entry name mismatch"),
        ("wrongtype:OBJ", "descriptor contract violated"),
    )


def test_collisions_prefer_builtins_then_lexical_locations() -> None:
    builtin = AdapterPluginRecord(
        name="shared",
        version="1.0",
        source="builtin",
        distribution="moeatlas",
        location="moeatlas.adapters.builtin",
        architecture_families=("f_moe",),
    )
    ep_a = AdapterPluginRecord(
        name="shared",
        version="9.0",
        source="entry_point",
        distribution="pkg-a",
        location="pkg_a.plugin:B",
        architecture_families=("f_moe",),
    )
    ep_b = AdapterPluginRecord(
        name="shared",
        version="9.0",
        source="entry_point",
        distribution="pkg-b",
        location="pkg_b.plugin:A",
        architecture_families=("f_moe",),
    )
    report = collect_adapter_registry(
        entry_points=(),
        builtin_records=(builtin, ep_a, ep_b),
    )
    assert [entry.record.location for entry in report.entries] == [
        "moeatlas.adapters.builtin"
    ]
    assert report.collisions == (
        ("shared", "moeatlas.adapters.builtin", "pkg_a.plugin:B"),
        ("shared", "moeatlas.adapters.builtin", "pkg_b.plugin:A"),
    )


# ---------------------------------------------------------------------------
# Collection and matching


def test_collect_end_to_end_over_fake_entry_points() -> None:
    report = collect_adapter_registry(entry_points=(_entry_point("acme"),))
    names = [entry.record.name for entry in report.entries]
    assert names == sorted(names)
    assert "huggingface-mixtral-static" in names
    assert "acme" in names
    assert all(entry.status == "enabled" for entry in report.entries)
    assert report.failures == ()
    assert report.collisions == ()
    builtin_only = collect_adapter_registry(
        entry_points=(_entry_point("acme"),),
        policy=AdapterRegistryPolicy(trusted_sources=("builtin",)),
    )
    statuses = {
        entry.record.name: entry.status
        for entry in builtin_only.entries
    }
    assert statuses["acme"] == "disabled"
    assert statuses["huggingface-mixtral-static"] == "enabled"


def test_match_adapters_for_family_is_enabled_only_and_sorted() -> None:
    report = collect_adapter_registry(
        entry_points=(_entry_point("acme"),),
        policy=AdapterRegistryPolicy(disabled_names=("acme",)),
    )
    assert match_adapters_for_family(report.entries, family="acme_moe") == ()
    enabled = collect_adapter_registry(entry_points=(_entry_point("acme"),))
    qwen = match_adapters_for_family(enabled.entries, family="qwen3_moe")
    assert [entry.record.name for entry in qwen] == [
        "huggingface-qwen3-moe-static"
    ]
    with pytest.raises(TypeError):
        match_adapters_for_family(enabled.entries, family=7)
    with pytest.raises(ValueError):
        match_adapters_for_family(enabled.entries, family="  ")


# ---------------------------------------------------------------------------
# Serialization


def test_report_round_trips_through_canonical_json() -> None:
    report = collect_adapter_registry(
        entry_points=(
            _entry_point("boom", value="boom:OBJ", payload=RuntimeError("x")),
            _entry_point("acme"),
        )
    )
    restored = AdapterRegistryReport.from_json(report.to_json())
    assert restored == report
    document = report.to_dict()
    assert document["artifact_type"] == "moeatlas.adapter_registry"
    assert document["schema_version"] == ADAPTER_REGISTRY_SCHEMA_VERSION
    with pytest.raises(AdapterRegistryError) as excinfo:
        AdapterRegistryReport.from_json('{"artifact_type": "other"}')
    assert excinfo.value.stage == "serialization"
    with pytest.raises(AdapterRegistryError):
        AdapterRegistryReport.from_json("[]")
    with pytest.raises(AdapterRegistryError):
        AdapterRegistryReport.from_json("{not json")


def test_serialization_is_deterministic() -> None:
    first = collect_adapter_registry(entry_points=(_entry_point("acme"),))
    second = collect_adapter_registry(entry_points=(_entry_point("acme"),))
    assert first.to_json() == second.to_json()
