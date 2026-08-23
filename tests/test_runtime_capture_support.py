"""Model-free contracts for honest static capture-support grading."""

from __future__ import annotations

from moeatlas.discovery import DiscoveryReport
from moeatlas.runtime import classify_capture_support

from .test_runtime_generic_capture import _flat_logits, _HookedModel, _report


def _moe_report() -> DiscoveryReport:
    return _report(_HookedModel(_flat_logits()))


def test_structure_with_router_and_experts_is_only_a_runtime_candidate() -> None:
    report = classify_capture_support(_moe_report())
    assert report.grade == "routing_and_activity_candidate"
    assert report.routing_capture == "candidate"
    assert report.expert_activity_capture == "candidate"
    assert report.router_target_count > 0
    assert report.expert_target_count > 0
    assert any("unproven" in item for item in report.limitations)


def test_missing_moe_facts_is_not_moe_and_never_claims_capture() -> None:
    original = _moe_report()
    document = original.model_dump(mode="json")
    document["facts"]["expert_count"] = None
    document["facts"]["routed_top_k"] = None
    document["facts"]["expert_count_source"] = None
    document["facts"]["routed_top_k_source"] = None
    report = classify_capture_support(DiscoveryReport.model_validate(document))
    assert report.grade == "not_moe"
    assert report.topology_discovered is False
    assert report.routing_capture == "unavailable"
    assert report.expert_activity_capture == "unavailable"


def test_capture_support_is_a_bounded_json_document() -> None:
    document = classify_capture_support(_moe_report()).to_dict()
    assert document["schema_version"] == "1.0"
    assert document["grade"] == "routing_and_activity_candidate"
    assert isinstance(document["limitations"], list)
