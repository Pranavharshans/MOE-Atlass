"""Versioned Evidence Cards over tiered expert evidence (PRD §11.5).

An Evidence Card is the structured alternative to a single "specialization
score": identity, routing usage, task association, internal behavior,
causal contribution, and replication/stability stay in separate optional
sections, each explicitly present or absent. Absence means "not measured" —
never inferred, never defaulted — and every card carries its own limitations,
warnings, provenance, and capability labels so downstream UI/API surfaces can
show components rather than collapse them.

The contract is model-neutral: it validates shapes, vocabularies, bounds, and
canonical serialization only. Where the numbers come from (routing shards,
association matrices, intervention runs) is the callers' concern; package
tests exercise cards over synthetic values, and real-model evidence remains
deferred MV work.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

EVIDENCE_CARD_SCHEMA_VERSION = "1.0"
"""Schema version of the Evidence Card contracts."""

_EVIDENCE_CARD_ARTIFACT_TYPE = "moeatlas.evidence_card"

EVIDENCE_TIERS = ("routing", "behavior", "causal", "replication")
"""Fixed evidence tiers; a capability label names exactly one tier."""

_CAPABILITY_LABELS = frozenset({"full", "partial", "unsupported"})
_EXPERT_KINDS = frozenset({"routed", "shared"})
_MAX_LABEL = 200
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")

_ERROR_STAGES = frozenset({"contract", "serialization"})


class EvidenceCardError(RuntimeError):
    """Safe fixed-stage failure for Evidence Card handling."""

    def __init__(
        self, stage: str, message: str | None = None, *, cause: BaseException | None = None
    ) -> None:
        if stage not in _ERROR_STAGES:
            raise ValueError("evidence card error stage is not supported")
        self.stage = stage
        text = f"evidence card failed at {stage}"
        if message:
            text = f"{text}: {message}"
        super().__init__(text)
        if cause is not None:
            self.__cause__ = cause


def _strict_str(value: object, field_name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a string")
    if not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    if len(value) > _MAX_LABEL:
        raise ValueError(f"{field_name} exceeds {_MAX_LABEL} characters")
    return value


def _strict_label_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{field_name} must be a tuple of strings")
    for entry in value:
        _strict_str(entry, field_name)
    return value  # type: ignore[return-value]


def _strict_optional_float(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    if type(value) is not float and type(value) is not int:
        raise TypeError(f"{field_name} must be a number or null")
    result = float(value)
    if result != result or result in (float("inf"), float("-inf")):
        raise ValueError(f"{field_name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class TaskAssociationSection:
    """Per-task association evidence for one expert (tier A inputs).

    Rows are index-aligned with ``task_keys``; ``example_count`` records how
    much evidence stands behind the numbers so uncertainty is visible.
    """

    task_keys: tuple[str, ...]
    enrichment: tuple[float | None, ...]
    pmi: tuple[float | None, ...]
    exclusivity: tuple[float | None, ...]
    example_count: int

    def __post_init__(self) -> None:
        width = len(self.task_keys)
        _strict_label_tuple(self.task_keys, "task_keys")
        if type(self.example_count) is not int or isinstance(self.example_count, bool):
            raise TypeError("example_count must be an integer")
        if self.example_count <= 0:
            raise ValueError("example_count must be a strict positive integer")
        rows = (
            ("enrichment", self.enrichment),
            ("pmi", self.pmi),
            ("exclusivity", self.exclusivity),
        )
        for name, row in rows:
            if type(row) is not tuple or len(row) != width:
                raise ValueError(f"{name} must align with task_keys exactly")
            for entry in row:
                _strict_optional_float(entry, name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_keys": list(self.task_keys),
            "enrichment": list(self.enrichment),
            "pmi": list(self.pmi),
            "exclusivity": list(self.exclusivity),
            "example_count": self.example_count,
        }

    @classmethod
    def from_dict(cls, document: object) -> TaskAssociationSection:
        if not isinstance(document, dict):
            raise TypeError("task association sections must be JSON objects")
        try:
            return cls(
                task_keys=tuple(document["task_keys"]),
                enrichment=tuple(document["enrichment"]),
                pmi=tuple(document["pmi"]),
                exclusivity=tuple(document["exclusivity"]),
                example_count=document["example_count"],
            )
        except KeyError as exc:
            raise ValueError("task association section is missing fields") from exc


@dataclass(frozen=True, slots=True)
class RoutingSection:
    """Routing-usage evidence for one expert (tier A)."""

    usage_share: float | None = None
    normalized_load: float | None = None
    mean_rank: float | None = None
    mean_margin: float | None = None
    mean_entropy: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "usage_share",
            "normalized_load",
            "mean_rank",
            "mean_margin",
            "mean_entropy",
        ):
            value = getattr(self, name)
            checked = _strict_optional_float(value, name)
            if checked is not None and checked < 0.0:
                raise ValueError(f"{name} must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "usage_share": self.usage_share,
            "normalized_load": self.normalized_load,
            "mean_rank": self.mean_rank,
            "mean_margin": self.mean_margin,
            "mean_entropy": self.mean_entropy,
        }

    @classmethod
    def from_dict(cls, document: object) -> RoutingSection:
        if not isinstance(document, dict):
            raise TypeError("routing sections must be JSON objects")
        return cls(
            usage_share=document.get("usage_share"),
            normalized_load=document.get("normalized_load"),
            mean_rank=document.get("mean_rank"),
            mean_margin=document.get("mean_margin"),
            mean_entropy=document.get("mean_entropy"),
        )


@dataclass(frozen=True, slots=True)
class BehaviorSection:
    """Internal-behavior summaries for one expert (tier B)."""

    input_norm_mean: float | None = None
    output_norm_mean: float | None = None
    contribution_mean: float | None = None

    def __post_init__(self) -> None:
        for name in ("input_norm_mean", "output_norm_mean", "contribution_mean"):
            _strict_optional_float(getattr(self, name), name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_norm_mean": self.input_norm_mean,
            "output_norm_mean": self.output_norm_mean,
            "contribution_mean": self.contribution_mean,
        }

    @classmethod
    def from_dict(cls, document: object) -> BehaviorSection:
        if not isinstance(document, dict):
            raise TypeError("behavior sections must be JSON objects")
        return cls(
            input_norm_mean=document.get("input_norm_mean"),
            output_norm_mean=document.get("output_norm_mean"),
            contribution_mean=document.get("contribution_mean"),
        )


@dataclass(frozen=True, slots=True)
class CausalitySection:
    """Intervention deltas for one expert (tier C); recipes live elsewhere."""

    delta_loss: float | None = None
    delta_target_probability: float | None = None
    delta_benchmark: float | None = None
    recipe_fingerprint: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "delta_loss",
            "delta_target_probability",
            "delta_benchmark",
        ):
            _strict_optional_float(getattr(self, name), name)
        if self.recipe_fingerprint is not None:
            _strict_str(self.recipe_fingerprint, "recipe_fingerprint")

    def to_dict(self) -> dict[str, Any]:
        return {
            "delta_loss": self.delta_loss,
            "delta_target_probability": self.delta_target_probability,
            "delta_benchmark": self.delta_benchmark,
            "recipe_fingerprint": self.recipe_fingerprint,
        }

    @classmethod
    def from_dict(cls, document: object) -> CausalitySection:
        if not isinstance(document, dict):
            raise TypeError("causality sections must be JSON objects")
        return cls(
            delta_loss=document.get("delta_loss"),
            delta_target_probability=document.get("delta_target_probability"),
            delta_benchmark=document.get("delta_benchmark"),
            recipe_fingerprint=document.get("recipe_fingerprint"),
        )


@dataclass(frozen=True, slots=True)
class StabilitySection:
    """Replication evidence across perturbations/datasets/seeds/revisions (D)."""

    replicated_seeds: int | None = None
    total_seeds: int | None = None
    replicated_datasets: int | None = None
    total_datasets: int | None = None

    def __post_init__(self) -> None:
        pairs = (
            ("replicated_seeds", "total_seeds"),
            ("replicated_datasets", "total_datasets"),
        )
        for counted, total in pairs:
            checked: dict[str, int | None] = {}
            for name in (counted, total):
                value = getattr(self, name)
                if value is not None and (
                    type(value) is not int or isinstance(value, bool)
                ):
                    raise TypeError(f"{name} must be an integer or null")
                checked[name] = value
            counted_value = checked[counted]
            total_value = checked[total]
            if counted_value is not None and total_value is not None:
                if counted_value < 0 or total_value <= 0 or counted_value > total_value:
                    raise ValueError(f"{counted}/{total} must satisfy 0 <= n <= N")


@dataclass(frozen=True, slots=True)
class EvidenceCard:
    """One expert's structured evidence with separate, optional tiers.

    Every section is present-or-absent by explicit construction; nothing is
    inferred from the other tiers. ``limitations`` and ``warnings`` carry the
    honest boundaries of the evidence, and ``capability_labels`` name each
    tier's coverage as ``full``, ``partial``, or ``unsupported``.
    """

    model_fingerprint: str
    layer_key: str
    expert_key: str
    expert_kind: str
    shared_expert_keys: tuple[str, ...] = ()
    schema_version: str = EVIDENCE_CARD_SCHEMA_VERSION
    routing: RoutingSection | None = None
    task_association: TaskAssociationSection | None = None
    behavior: BehaviorSection | None = None
    causality: CausalitySection | None = None
    stability: StabilitySection | None = None
    limitations: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    capability_labels: tuple[tuple[str, str], ...] = ()
    probe_version: str | None = None
    adapter_name: str | None = None
    adapter_version: str | None = None
    capture_source: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != EVIDENCE_CARD_SCHEMA_VERSION:
            raise ValueError("schema_version is not the exact evidence-card version")
        if type(self.model_fingerprint) is not str:
            raise TypeError("model_fingerprint must be a string")
        if _DIGEST.fullmatch(self.model_fingerprint) is None:
            raise ValueError("model_fingerprint must use the sha256:<64 hex> form")
        for name, section_type in (
            ("routing", RoutingSection),
            ("task_association", TaskAssociationSection),
            ("behavior", BehaviorSection),
            ("causality", CausalitySection),
            ("stability", StabilitySection),
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, section_type):
                raise TypeError(f"{name} must be a {section_type.__name__} or null")
        _strict_str(self.layer_key, "layer_key")
        _strict_str(self.expert_key, "expert_key")
        if self.expert_kind not in _EXPERT_KINDS:
            raise ValueError("expert_kind must be routed or shared")
        _strict_label_tuple(self.shared_expert_keys, "shared_expert_keys")
        if self.expert_kind == "routed" and self.shared_expert_keys:
            raise ValueError("routed experts carry no shared-expert keys")
        _strict_label_tuple(self.limitations, "limitations")
        _strict_label_tuple(self.warnings, "warnings")
        if type(self.capability_labels) is not tuple:
            raise TypeError("capability_labels must be a tuple of (tier, label) pairs")
        seen: set[str] = set()
        for pair in self.capability_labels:
            if type(pair) is not tuple or len(pair) != 2:
                raise TypeError("capability labels must be (tier, label) pairs")
            tier, label = pair
            if tier not in EVIDENCE_TIERS:
                raise ValueError(f"capability tier {tier!r} is not supported")
            if label not in _CAPABILITY_LABELS:
                raise ValueError(f"capability label {label!r} is not supported")
            if tier in seen:
                raise ValueError(f"capability tier {tier!r} appears more than once")
            seen.add(tier)
        for name in ("probe_version", "adapter_name", "adapter_version", "capture_source"):
            value = getattr(self, name)
            if value is not None:
                _strict_str(value, name)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible data without runtime objects."""

        def section(section_object: Any) -> Any:
            return None if section_object is None else section_object.to_dict()

        return {
            "artifact_type": _EVIDENCE_CARD_ARTIFACT_TYPE,
            "schema_version": self.schema_version,
            "model_fingerprint": self.model_fingerprint,
            "layer_key": self.layer_key,
            "expert_key": self.expert_key,
            "expert_kind": self.expert_kind,
            "shared_expert_keys": list(self.shared_expert_keys),
            "routing": section(self.routing),
            "task_association": section(self.task_association),
            "behavior": section(self.behavior),
            "causality": section(self.causality),
            "stability": section(self.stability),
            "limitations": list(self.limitations),
            "warnings": list(self.warnings),
            "capability_labels": [list(pair) for pair in self.capability_labels],
            "probe_version": self.probe_version,
            "adapter_name": self.adapter_name,
            "adapter_version": self.adapter_version,
            "capture_source": self.capture_source,
        }

    def to_json(self) -> str:
        """Serialize this card with deterministic key order and no whitespace."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> EvidenceCard:
        """Validate one canonical JSON document into an exact card value."""

        try:
            document = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("evidence card document is not valid JSON") from exc
        if type(document) is not dict:
            raise ValueError("evidence card document must be a JSON object")
        if (
            document.get("artifact_type") != _EVIDENCE_CARD_ARTIFACT_TYPE
            or document.get("schema_version") != EVIDENCE_CARD_SCHEMA_VERSION
        ):
            raise ValueError("document is not an evidence card artifact")
        try:
            sections: dict[str, Any] = {}
            for name, section_cls in (
                ("routing", RoutingSection),
                ("task_association", TaskAssociationSection),
                ("behavior", BehaviorSection),
                ("causality", CausalitySection),
            ):
                raw = document.get(name)
                sections[name] = None if raw is None else section_cls.from_dict(raw)
            stability_raw = document.get("stability")
            sections["stability"] = (
                None if stability_raw is None else StabilitySection(**stability_raw)
            )
            return cls(
                model_fingerprint=document["model_fingerprint"],
                layer_key=document["layer_key"],
                expert_key=document["expert_key"],
                expert_kind=document["expert_kind"],
                shared_expert_keys=tuple(document["shared_expert_keys"]),
                routing=sections["routing"],
                task_association=sections["task_association"],
                behavior=sections["behavior"],
                causality=sections["causality"],
                stability=sections["stability"],
                limitations=tuple(document["limitations"]),
                warnings=tuple(document["warnings"]),
                capability_labels=tuple(
                    (pair[0], pair[1]) for pair in document["capability_labels"]
                ),
                probe_version=document.get("probe_version"),
                adapter_name=document.get("adapter_name"),
                adapter_version=document.get("adapter_version"),
                capture_source=document.get("capture_source"),
            )
        except KeyError as exc:
            raise ValueError("evidence card document is missing fields") from exc
        except (TypeError, ValueError):
            raise
        except Exception as exc:
            raise ValueError("evidence card document is not usable") from exc
