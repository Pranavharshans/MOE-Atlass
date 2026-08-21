"""Contract tests for versioned Evidence Cards over tiered expert evidence."""

from __future__ import annotations

import pytest

from moeatlas.analysis import (
    EVIDENCE_CARD_SCHEMA_VERSION,
    EVIDENCE_TIERS,
    EvidenceCard,
    EvidenceCardError,
    TaskAssociationSection,
)

# ---------------------------------------------------------------------------
# Fixtures


def _card(**overrides: object) -> EvidenceCard:
    values: dict[str, object] = {
        "model_fingerprint": "sha256:" + "a" * 64,
        "layer_key": "l0",
        "expert_key": "e0",
        "expert_kind": "routed",
    }
    values.update(overrides)
    return EvidenceCard(**values)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Public surface


def test_surface_is_pinned() -> None:
    assert EVIDENCE_CARD_SCHEMA_VERSION == "1.0"
    assert EVIDENCE_TIERS == ("routing", "behavior", "causal", "replication")
    assert str(EvidenceCardError("contract")) == "evidence card failed at contract"
    with pytest.raises(ValueError):
        EvidenceCardError("cancelled")


def test_identity_is_required_and_strict() -> None:
    card = _card()
    assert card.schema_version == EVIDENCE_CARD_SCHEMA_VERSION
    assert card.expert_kind == "routed"
    with pytest.raises(TypeError):
        EvidenceCard(model_fingerprint=7, layer_key="l0", expert_key="e0",
                     expert_kind="routed")
    with pytest.raises(ValueError):
        _card(model_fingerprint="not-a-digest")
    with pytest.raises(ValueError):
        _card(expert_kind="mystery")
    with pytest.raises(ValueError):
        _card(layer_key="")
    with pytest.raises(ValueError):
        _card(shared_expert_keys=("e0",))  # routed cards carry no shared keys
    shared = _card(expert_kind="shared", shared_expert_keys=("s0",))
    assert shared.shared_expert_keys == ("s0",)


def test_sections_stay_separate_and_optional() -> None:
    card = _card(
        routing=None,
        task_association=None,
        behavior=None,
        causality=None,
        stability=None,
    )
    assert card.routing is None
    assert card.task_association is None
    # A measured section must be the exact section type.
    with pytest.raises(TypeError):
        _card(routing="usage: lots")


def test_task_association_section_is_bounded_and_strict() -> None:
    section = TaskAssociationSection(
        task_keys=("math",),
        enrichment=(2.0,),
        pmi=(1.0,),
        exclusivity=(0.75,),
        example_count=100,
    )
    assert section.example_count == 100
    with pytest.raises(ValueError):
        TaskAssociationSection(task_keys=("math",), enrichment=(2.0, 1.0), pmi=(0.0,),
                               exclusivity=(0.5,), example_count=10)
    with pytest.raises(ValueError):
        TaskAssociationSection(task_keys=("math",), enrichment=(2.0,), pmi=(1.0,),
                               exclusivity=(0.5,), example_count=0)
    with pytest.raises(TypeError):
        TaskAssociationSection(task_keys=("math",), enrichment=("high",), pmi=(1.0,),
                               exclusivity=(0.5,), example_count=10)


def test_limitations_warnings_and_capability_labels_are_canonical() -> None:
    card = _card(
        limitations=("no causal claims",),
        warnings=("fused decoder",),
        capability_labels=(("routing", "partial"),),
    )
    assert card.limitations == ("no causal claims",)
    assert card.capability_labels == (("routing", "partial"),)
    with pytest.raises(ValueError):
        _card(capability_labels=(("mystery", "full"),))
    with pytest.raises(ValueError):
        _card(capability_labels=(("routing", "bogus"),))
    with pytest.raises(TypeError):
        _card(warnings=("w", 7))
    # An explicit empty limitations list is honest evidence, not an error.
    assert _card().limitations == ()


# ---------------------------------------------------------------------------
# Serialization


def test_card_round_trips_through_canonical_json() -> None:
    card = _card(
        task_association=TaskAssociationSection(
            task_keys=("math",), enrichment=(2.0,), pmi=(1.0,),
            exclusivity=(0.5,), example_count=42,
        ),
        warnings=("synthetic",),
    )
    restored = EvidenceCard.from_json(card.to_json())
    assert restored == card
    document = card.to_dict()
    assert document["artifact_type"] == "moeatlas.evidence_card"
    assert document["task_association"]["example_count"] == 42
    with pytest.raises(ValueError):
        EvidenceCard.from_json('{"artifact_type": "other"}')
    with pytest.raises(ValueError):
        EvidenceCard.from_json("[]")
    with pytest.raises(ValueError):
        EvidenceCard.from_json("{not json")


def test_serialization_is_deterministic_and_nan_free() -> None:
    first = _card(warnings=("w",))
    second = _card(warnings=("w",))
    assert first.to_json() == second.to_json()
    assert "NaN" not in first.to_json()
